import numpy as np
import pandas as pd
import pytest

from conftest import make_processed
from credit_risk.data.features import feature_names
from credit_risk.data.split import OutOfTimeSplit, TimeWindow, out_of_time_split
from credit_risk.data.target import TARGET
from credit_risk.models.baseline import FittedBaseline, fit_baseline
from credit_risk.models.metrics import score

APPLICANT = feature_names()
WITH_PRICING = feature_names(include_lender_pricing=True)


@pytest.fixture(scope="module")
def split() -> OutOfTimeSplit:
    return out_of_time_split(
        make_processed(),
        train=TimeWindow(pd.Timestamp("2012-01-01"), pd.Timestamp("2014-12-01")),
        test=TimeWindow(pd.Timestamp("2015-07-01"), pd.Timestamp("2016-12-01")),
    )


def test_baseline_ranks_out_of_time_loans_and_learns_the_true_drivers(
    split: OutOfTimeSplit,
) -> None:
    model = fit_baseline(split.train, APPLICANT)
    test = score(
        split.test[TARGET], model.predict_pd(split.test), reference_rate=model.train_default_rate
    )
    assert test.auc > 0.7
    assert 0.9 * test.default_rate < test.mean_pd < 1.1 * test.default_rate
    coef = model.coefficients()
    # The generator lowers risk with FICO and raises it with DTI; nothing else matters.
    assert coef["fico"] < 0 < coef["dti"]
    assert set(coef.index[:2]) == {"fico", "dti"}


def test_preprocessing_is_learned_from_the_training_window_only(split: OutOfTimeSplit) -> None:
    model = fit_baseline(split.train, APPLICANT)
    imputer = model.pipeline.named_steps["columns"].named_transformers_["numeric"][0]
    revol_util = list(imputer.feature_names_in_).index("revol_util")
    assert imputer.statistics_[revol_util] == pytest.approx(split.train["revol_util"].median())
    assert imputer.statistics_[revol_util] != split.test["revol_util"].median()


def test_missing_values_and_unseen_categories_are_scored_not_rejected(
    split: OutOfTimeSplit,
) -> None:
    model = fit_baseline(split.train, WITH_PRICING)
    odd = split.test.head(3).copy()
    odd["purpose"] = pd.Categorical(["wedding", None, "credit_card"])
    odd["grade"] = pd.Categorical(["G", "A", None])
    odd.loc[:, ["dti", "fico", "annual_inc"]] = np.nan
    pd_hat = model.predict_pd(odd)
    assert pd_hat.shape == (3,)
    assert np.isfinite(pd_hat).all()


def test_missingness_is_an_input_of_its_own(split: OutOfTimeSplit) -> None:
    names = set(fit_baseline(split.train, APPLICANT).coefficients().index)
    assert "missingindicator_revol_util" in names


def test_ordinary_training_fits_every_requested_feature(split: OutOfTimeSplit) -> None:
    model = fit_baseline(split.train, APPLICANT)
    assert model.features == APPLICANT
    assert model.dropped == {}
    assert any(name.startswith("application_type_") for name in model.coefficients().index)


# The real 2010-2011 training window: no `mort_acc` reported at all, and only
# individual applications. Both vary in the later test window.
@pytest.fixture(scope="module")
def degenerate(split: OutOfTimeSplit) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = split.train.copy()
    train["mort_acc"] = np.nan
    train["application_type"] = pd.Categorical(["Individual"] * len(train))
    # One observed value plus missing values: still two values, so still fitted.
    train["emp_length_years"] = np.where(np.arange(len(train)) % 3 == 0, np.nan, 10.0)
    test = split.test.copy()
    test["application_type"] = pd.Categorical(
        np.where(np.arange(len(test)) % 2 == 0, "Joint App", "Individual")
    )
    assert test["mort_acc"].notna().all()
    return train, test


@pytest.fixture(scope="module")
def reduced(degenerate: tuple[pd.DataFrame, pd.DataFrame]) -> FittedBaseline:
    return fit_baseline(degenerate[0], APPLICANT)


def test_degenerate_training_features_are_left_out_and_recorded(reduced: FittedBaseline) -> None:
    assert reduced.dropped == {
        "mort_acc": "all missing",
        "application_type": "constant 'Individual'",
    }
    assert reduced.features == tuple(
        name for name in APPLICANT if name not in {"mort_acc", "application_type"}
    )
    assert tuple(reduced.pipeline.feature_names_in_) == reduced.features


def test_one_observed_value_plus_missing_is_retained(reduced: FittedBaseline) -> None:
    assert "emp_length_years" in reduced.features
    assert "missingindicator_emp_length_years" in reduced.coefficients().index


def test_coefficients_never_name_a_dropped_feature(reduced: FittedBaseline) -> None:
    names = reduced.coefficients().index
    assert not [n for n in names if "mort_acc" in n or n.startswith("application_type")]


def test_dropped_features_are_not_read_when_scoring(
    degenerate: tuple[pd.DataFrame, pd.DataFrame], reduced: FittedBaseline
) -> None:
    test = degenerate[1]
    pd_hat = reduced.predict_pd(test)
    assert np.isfinite(pd_hat).all()
    rewritten = test.assign(mort_acc=99.0, application_type="Joint App")
    np.testing.assert_array_equal(reduced.predict_pd(rewritten), pd_hat)
    removed = test.drop(columns=["mort_acc", "application_type"])
    np.testing.assert_array_equal(reduced.predict_pd(removed), pd_hat)


def test_a_window_where_nothing_varies_is_refused(split: OutOfTimeSplit) -> None:
    with pytest.raises(ValueError, match="no requested feature varies"):
        fit_baseline(split.train.assign(mort_acc=np.nan), ("mort_acc",))
