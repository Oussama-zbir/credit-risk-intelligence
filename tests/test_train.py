from pathlib import Path

import pandas as pd
import pytest

from conftest import make_processed
from credit_risk.data.features import feature_names
from credit_risk.data.split import OutOfTimeSplit, TimeWindow, out_of_time_split
from credit_risk.models.artifact import load_artifact
from credit_risk.models.calibration import Method
from credit_risk.prepare import sha256
from credit_risk.train import RISKIEST_LOANS, Comparison, evaluate, main, parse_window


@pytest.fixture(scope="module")
def split() -> OutOfTimeSplit:
    return out_of_time_split(
        make_processed(20_000),
        train=TimeWindow(pd.Timestamp("2012-01-01"), pd.Timestamp("2014-12-01")),
        test=TimeWindow(pd.Timestamp("2015-07-01"), pd.Timestamp("2016-12-01")),
    )


@pytest.fixture(scope="module")
def results(split: OutOfTimeSplit) -> list[Comparison]:
    return evaluate(split)


def test_every_model_is_scored_on_the_same_out_of_time_loans(
    split: OutOfTimeSplit, results: list[Comparison]
) -> None:
    applicant, pricing = results
    assert applicant.baseline.features == applicant.booster.features == feature_names()
    assert pricing.booster.features == feature_names(include_lender_pricing=True)
    for result in results:
        assert [e.model for e in result.evaluations] == [
            "logistic regression",
            "gradient boosting, raw",
            "gradient boosting + platt",
        ]
        assert {e.test.loans for e in result.evaluations} == {len(split.test)}
        assert all(
            sum(b.loans for b in e.test_reliability) == len(split.test) for e in result.evaluations
        )
        assert result.explanation.contributions.index.equals(result.explained.index)
        assert result.explained.index.isin(split.test.index).all()


def test_booster_ranks_close_to_a_baseline_that_matches_the_true_model(
    results: list[Comparison],
) -> None:
    # The generator's log-odds are linear, so logistic regression is the right
    # model here; the booster only has to come close, on less training data.
    baseline, raw, calibrated = results[0].evaluations
    assert raw.test.auc > 0.7
    assert raw.test.auc > baseline.test.auc - 0.03
    assert calibrated.test.auc > baseline.test.auc - 0.03


def test_platt_recalibrates_without_reordering_the_booster(results: list[Comparison]) -> None:
    raw, platt = results[0].evaluations[1:]
    assert platt.test.auc == pytest.approx(raw.test.auc)


def test_calibration_months_and_method_are_configurable(split: OutOfTimeSplit) -> None:
    applicant, _ = evaluate(split, calibration_months=12, method=Method.ISOTONIC)
    assert applicant.evaluations[2].model == "gradient boosting + isotonic"


@pytest.mark.parametrize("text", ["2014-01", "2014-13:2015-01", "2015-01:2014-01"])
def test_window_argument_must_be_an_ordered_month_range(text: str) -> None:
    with pytest.raises(Exception, match=r"YYYY-MM|month|after"):
        parse_window(text)


def test_cli_writes_a_report_comparing_models_and_feature_sets(tmp_path: Path) -> None:
    frame = tmp_path / "loans.parquet"
    make_processed().to_parquet(frame, index=False)
    out = tmp_path / "model_report.md"
    main(
        [
            str(frame),
            *("--train", "2012-01:2014-12", "--test", "2015-07:2016-12"),
            *("--calibration-months", "6", "--calibration", "isotonic", "--out", str(out)),
        ]
    )

    report = out.read_text()
    assert "Test (out of time): issued 2015-07 to 2016-12" in report
    assert "all but the last 6 training months" in report
    assert "| applicant | logistic regression | test |" in report
    assert "| applicant + lender pricing | gradient boosting + isotonic | test |" in report
    assert "## Reliability on the test window — applicant" in report
    assert report.count("\n| 10 | ") == 1
    assert "## Largest baseline coefficients — applicant" in report
    assert "### applicant + lender pricing\n" in report
    reasons = report.split("## Reason codes for the riskiest loans — applicant\n")[1]
    assert reasons.count("\n| ") == 2 + RISKIEST_LOANS  # header, rule, one row per loan


def test_cli_saves_the_applicant_booster_as_a_loadable_artifact(tmp_path: Path) -> None:
    frame = tmp_path / "loans.parquet"
    make_processed().to_parquet(frame, index=False)
    artifacts = tmp_path / "artifacts"
    main(
        [
            str(frame),
            *("--train", "2012-01:2014-12", "--test", "2015-07:2016-12"),
            *("--out", str(tmp_path / "model_report.md"), "--artifact-dir", str(artifacts)),
        ]
    )

    (saved,) = artifacts.iterdir()
    manifest = load_artifact(saved).manifest
    assert manifest.features == feature_names()
    assert manifest.provenance.data_sha256 == sha256(frame)
    assert manifest.provenance.train_window == ("2012-01", "2014-12")
    assert manifest.provenance.test_window == ("2015-07", "2016-12")
    assert manifest.provenance.calibration_months == 6
