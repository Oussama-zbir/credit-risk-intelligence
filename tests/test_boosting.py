import numpy as np
import pandas as pd
import pytest

from conftest import make_processed
from credit_risk.data.features import feature_names
from credit_risk.data.split import TimeWindow, hold_out_latest
from credit_risk.data.target import ISSUE_DATE
from credit_risk.models.boosting import MAX_TREES, FittedBooster, fit_booster

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
