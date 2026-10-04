from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import make_processed
from credit_risk.data.features import feature_names
from credit_risk.data.split import TimeWindow, hold_out_latest
from credit_risk.data.target import ISSUE_DATE, TARGET
from credit_risk.models.artifact import Provenance, load_artifact, save_artifact
from credit_risk.models.boosting import (
    MAX_TREES,
    FittedBooster,
    fit_booster,
    model_frame,
)
from credit_risk.models.degenerate import degenerate_features
from credit_risk.models.explain import explain
from credit_risk.models.metrics import score

APPLICANT = feature_names()
WITH_PRICING = feature_names(include_lender_pricing=True)
WINDOW = TimeWindow(pd.Timestamp("2012-01-01"), pd.Timestamp("2014-12-01"))


@pytest.fixture(scope="module")
def slices() -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = make_processed(8000)
    return hold_out_latest(frame.loc[WINDOW.contains(frame[ISSUE_DATE])], WINDOW, months=6)


@pytest.fixture(scope="module")
def booster(slices: tuple[pd.DataFrame, pd.DataFrame]) -> FittedBooster:
    return fit_booster(*slices, WITH_PRICING)


def test_early_stopping_on_the_held_out_slice_bounds_the_ensemble(
    booster: FittedBooster,
) -> None:
    assert 0 < booster.trees < MAX_TREES


def test_constrained_features_move_pd_in_one_direction_only(
    slices: tuple[pd.DataFrame, pd.DataFrame], booster: FittedBooster
) -> None:
    applicants = slices[1].head(50)
    for feature, grid, sign in (
        ("fico", np.arange(600, 850, 5), -1),
        ("dti", np.arange(0, 45, 1), 1),
        ("int_rate", np.arange(5, 25, 0.5), 1),
    ):
        # Each applicant, with only `feature` swept over the grid.
        swept = applicants.loc[applicants.index.repeat(grid.size)].copy()
        swept[feature] = np.tile(grid, len(applicants))
        pd_hat = booster.predict_pd(swept).reshape(len(applicants), grid.size)
        steps = sign * np.diff(pd_hat, axis=1)
        assert (steps >= -1e-12).all(), feature


def test_missing_values_and_unseen_categories_are_scored_not_rejected(
    slices: tuple[pd.DataFrame, pd.DataFrame], booster: FittedBooster
) -> None:
    odd = slices[1].head(3).copy()
    odd["purpose"] = pd.Categorical(["wedding", None, "credit_card"])
    odd["grade"] = ["G", "A", None]  # plain strings: the booster restores the dtype
    odd.loc[:, ["dti", "fico", "annual_inc"]] = np.nan
    pd_hat = booster.predict_pd(odd)
    assert pd_hat.shape == (3,)
    assert np.isfinite(pd_hat).all()


def test_pricing_features_are_only_used_when_asked_for(
    slices: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    model = fit_booster(*slices, APPLICANT)
    assert "grade" not in model.model.feature_names_in_
    assert set(model.model.feature_names_in_) == set(APPLICANT)


def test_a_categorical_missing_in_every_row_is_scored_as_missing(
    slices: tuple[pd.DataFrame, pd.DataFrame], booster: FittedBooster
) -> None:
    # A batch with no `purpose` at all arrives as float NaN, not as a categorical.
    loans = slices[1].head(3).copy()
    typed = loans.assign(purpose=pd.Categorical([np.nan] * 3, loans["purpose"].cat.categories))
    untyped = loans.assign(purpose=np.nan)
    np.testing.assert_array_equal(booster.predict_pd(untyped), booster.predict_pd(typed))


def test_categorical_levels_seen_in_fitting_are_recorded(
    slices: tuple[pd.DataFrame, pd.DataFrame], booster: FittedBooster
) -> None:
    assert booster.categories["grade"] == tuple(sorted(slices[0]["grade"].dropna().unique()))
    assert set(booster.categories) == {
        "purpose",
        "application_type",
        "home_ownership",
        "verification_status",
        "grade",
        "sub_grade",
    }


def test_ordinary_training_fits_every_requested_feature(booster: FittedBooster) -> None:
    assert booster.features == WITH_PRICING
    assert booster.dropped == {}
    assert tuple(booster.model.feature_names_in_) == WITH_PRICING


# The real 2010-01 to 2011-06 fitting window: no `mort_acc` reported at all, and
# only individual applications. Both appear later, in calibration and test.
type Degenerate = tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]


@pytest.fixture(scope="module")
def degenerate(slices: tuple[pd.DataFrame, pd.DataFrame]) -> Degenerate:
    fitting, calibration = (frame.copy() for frame in slices)
    fitting["mort_acc"] = np.nan
    fitting["application_type"] = pd.Categorical(["Individual"] * len(fitting))
    # One observed value plus missing values: still two values, so still fitted.
    fitting["emp_length_years"] = np.where(fitting.index % 3 == 0, np.nan, 10.0)
    later = make_processed(2000, seed=1)
    later["application_type"] = pd.Categorical(
        np.where(later.index % 2 == 0, "Joint App", "Individual")
    )
    return fitting, calibration, later


@pytest.fixture(scope="module")
def reduced(degenerate: Degenerate) -> FittedBooster:
    fitting, calibration, _ = degenerate
    return fit_booster(fitting, calibration, WITH_PRICING)


def test_degenerate_means_at_most_one_value_counting_missing(degenerate: Degenerate) -> None:
    assert degenerate_features(degenerate[0], WITH_PRICING) == {
        "mort_acc": "all missing",
        "application_type": "constant 'Individual'",
    }


def test_degenerate_fitting_features_are_left_out_and_recorded(reduced: FittedBooster) -> None:
    assert set(reduced.dropped) == {"mort_acc", "application_type"}
    assert reduced.features == tuple(
        name for name in WITH_PRICING if name not in {"mort_acc", "application_type"}
    )
    assert reduced.features == tuple(reduced.model.feature_names_in_)
    assert "application_type" not in reduced.categories


def test_one_observed_value_plus_missing_is_retained(reduced: FittedBooster) -> None:
    assert "emp_length_years" in reduced.features


def test_features_present_only_after_the_fitting_window_cannot_enter_the_model(
    degenerate: Degenerate, reduced: FittedBooster
) -> None:
    later = degenerate[2]
    assert later["mort_acc"].notna().all()
    pd_hat = reduced.predict_pd(later)
    assert np.isfinite(pd_hat).all()
    # Rewriting the dropped columns changes nothing: they are not read at all.
    rewritten = later.assign(mort_acc=99.0, application_type="Joint App")
    np.testing.assert_array_equal(reduced.predict_pd(rewritten), pd_hat)
    # And a scoring frame without them is complete.
    np.testing.assert_array_equal(
        reduced.predict_pd(later.drop(columns=["mort_acc", "application_type"])), pd_hat
    )


def test_a_level_first_seen_after_fitting_is_scored_as_missing(
    slices: tuple[pd.DataFrame, pd.DataFrame], booster: FittedBooster
) -> None:
    loans = slices[1].head(20).copy()
    loans["purpose"] = loans["purpose"].astype(object)
    unseen = loans.assign(purpose="renewable_energy")
    missing = loans.assign(purpose=None)
    np.testing.assert_array_equal(booster.predict_pd(unseen), booster.predict_pd(missing))


def test_explanations_cover_exactly_the_fitted_features(
    degenerate: Degenerate, reduced: FittedBooster
) -> None:
    later = degenerate[2].head(300)
    result = explain(reduced, later)
    assert tuple(result.contributions.columns) == reduced.features
    expected = reduced.model.decision_function(model_frame(later, reduced.features))
    np.testing.assert_allclose(result.log_odds, expected, atol=1e-9)


def test_a_reduced_booster_round_trips_as_an_artifact(
    tmp_path: Path, degenerate: Degenerate, reduced: FittedBooster
) -> None:
    later = degenerate[2]
    pd_hat = reduced.predict_pd(later)
    directory = save_artifact(
        reduced,
        tmp_path,
        provenance=Provenance(
            data_sha256="ab" * 32,
            train_window=("2010-01", "2011-12"),
            calibration_months=6,
            test_window=("2012-01", "2012-12"),
        ),
        test_scores=score(later[TARGET], pd_hat, reference_rate=0.2),
        reference=later,
    )
    loaded = load_artifact(directory)
    assert loaded.manifest.features == reduced.features
    assert loaded.model.dropped == reduced.dropped
    np.testing.assert_array_equal(loaded.predict_pd(later), pd_hat)


def test_a_window_where_nothing_varies_is_refused(
    slices: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    fitting = slices[0].assign(mort_acc=np.nan)
    with pytest.raises(ValueError, match="no requested feature varies"):
        fit_booster(fitting, slices[1], ("mort_acc",))
