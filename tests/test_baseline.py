from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import make_processed
from credit_risk.data.features import feature_names
from credit_risk.data.split import OutOfTimeSplit, TimeWindow, out_of_time_split
from credit_risk.models.baseline import fit_baseline
from credit_risk.train import evaluate, main, parse_window

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
    applicant, _ = evaluate(split)
    assert applicant.test.auc > 0.7
    assert (
        0.9 * applicant.test.default_rate
        < applicant.test.mean_pd
        < 1.1 * applicant.test.default_rate
    )
    coef = applicant.model.coefficients()
    # The generator lowers risk with FICO and raises it with DTI; nothing else matters.
    assert coef["fico"] < 0 < coef["dti"]
    assert set(coef.index[:2]) == {"fico", "dti"}


def test_pricing_is_opt_in_and_scored_on_the_same_split(split: OutOfTimeSplit) -> None:
    applicant, pricing = evaluate(split)
    assert applicant.model.features == APPLICANT
    assert pricing.model.features == WITH_PRICING
    assert applicant.test.loans == pricing.test.loans


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


@pytest.mark.parametrize("text", ["2014-01", "2014-13:2015-01", "2015-01:2014-01"])
def test_window_argument_must_be_an_ordered_month_range(text: str) -> None:
    with pytest.raises(Exception, match=r"YYYY-MM|month|after"):
        parse_window(text)


def test_cli_writes_a_report_comparing_both_feature_sets(tmp_path: Path) -> None:
    frame = tmp_path / "loans.parquet"
    make_processed().to_parquet(frame, index=False)
    out = tmp_path / "baseline_report.md"
    main([str(frame), "--train", "2012-01:2014-12", "--test", "2015-07:2016-12", "--out", str(out)])

    report = out.read_text()
    assert "Test (out of time): issued 2015-07 to 2016-12" in report
    assert "| applicant | test |" in report
    assert "| applicant + lender pricing | test |" in report
    assert "## Largest coefficients — applicant" in report
