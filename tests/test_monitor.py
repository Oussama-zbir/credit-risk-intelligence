from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import make_processed
from credit_risk import monitor
from credit_risk.data.features import feature_names
from credit_risk.data.split import TimeWindow
from credit_risk.models.artifact import (
    FittedModel,
    ModelFamily,
    Provenance,
    load_artifact,
    save_artifact,
)
from credit_risk.models.baseline import fit_baseline
from credit_risk.models.boosting import fit_booster
from credit_risk.models.drift import Band, DriftError
from credit_risk.models.metrics import score
from credit_risk.prepare import sha256

TRAIN_END = "2014-12-01"
MONITORED = TimeWindow(pd.Timestamp("2015-01-01"), pd.Timestamp("2016-12-01"))


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return make_processed(6_000)


@pytest.fixture(scope="module")
def train(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame["issue_date"] <= TRAIN_END]


@pytest.fixture(scope="module", params=list(ModelFamily))
def fitted(request: pytest.FixtureRequest, train: pd.DataFrame) -> FittedModel:
    if request.param is ModelFamily.LOGISTIC_REGRESSION:
        return fit_baseline(train, feature_names())
    fitting = train[train["issue_date"] < "2014-07-01"]
    calibration = train[train["issue_date"] >= "2014-07-01"]
    return fit_booster(fitting, calibration, feature_names())


def _shifted(frame: pd.DataFrame) -> pd.DataFrame:
    """Later loans with 40 points less FICO; everything else drawn as before."""
    out = frame.copy()
    later = out["issue_date"] > TRAIN_END
    out.loc[later, "fico"] -= 40
    return out


@pytest.fixture(scope="module", params=[False, True], ids=["unchanged", "fico-shifted"])
def saved(
    request: pytest.FixtureRequest,
    tmp_path_factory: pytest.TempPathFactory,
    frame: pd.DataFrame,
    fitted: FittedModel,
) -> tuple[Path, Path, bool]:
    shifted: bool = request.param
    root = tmp_path_factory.mktemp("monitor")
    data = root / "loans.parquet"
    (_shifted(frame) if shifted else frame).to_parquet(data, index=False)
    test = frame[frame["issue_date"] > TRAIN_END]
    artifact = save_artifact(
        fitted,
        root / "artifacts",
        provenance=Provenance(
            data_sha256=sha256(data),
            train_window=("2012-01", "2014-12"),
            calibration_months=None,
            test_window=("2015-01", "2016-12"),
        ),
        test_scores=score(test["is_default"], fitted.predict_pd(test), reference_rate=0.2),
        reference=test,
    )
    return artifact, data, shifted


def test_drift_is_found_where_it_was_put_and_nowhere_else(saved: tuple[Path, Path, bool]) -> None:
    artifact, data, shifted = saved
    report = monitor.run(artifact, data, MONITORED)
    by_name = {s.name: s for s in report.features}
    if shifted:
        assert report.features[0].name == "fico"
        assert by_name["fico"].band is Band.SHIFTED
        assert report.contributions.index[0] == "fico"
        assert report.contributions["shift"]["fico"] > 0  # lower FICO, higher risk
    else:
        assert all(s.band is Band.STABLE for s in report.features)
        assert report.score.band is Band.STABLE


def test_contribution_shifts_sum_to_the_change_in_mean_log_odds(
    saved: tuple[Path, Path, bool], frame: pd.DataFrame
) -> None:
    artifact, data, _ = saved
    model = load_artifact(artifact)
    loans = pd.read_parquet(data)
    reference = loans[loans["issue_date"] <= TRAIN_END]
    monitored = loans[loans["issue_date"] > TRAIN_END]
    report = monitor.measure(model, reference, monitored)

    def mean_log_odds(rows: pd.DataFrame) -> float:
        return float(model.explain(rows).log_odds.mean())

    total = mean_log_odds(monitored) - mean_log_odds(reference)
    assert report.contributions["shift"].sum() == pytest.approx(total, abs=1e-9)


def test_periods_are_quarters_with_outcomes(saved: tuple[Path, Path, bool]) -> None:
    artifact, data, _ = saved
    report = monitor.run(artifact, data, MONITORED)
    assert [p.label for p in report.periods][:2] == ["2015Q1", "2015Q2"]
    assert sum(p.loans for p in report.periods) == report.monitored_loans
    assert all(p.default_rate is not None for p in report.periods)


def test_outcomes_are_optional(saved: tuple[Path, Path, bool]) -> None:
    artifact, data, _ = saved
    loans = pd.read_parquet(data)
    model = load_artifact(artifact)
    report = monitor.measure(
        model,
        loans[loans["issue_date"] <= TRAIN_END],
        loans[loans["issue_date"] > TRAIN_END].drop(columns="is_default"),
    )
    assert all(p.default_rate is None for p in report.periods)
    assert "not yet known" in monitor.render(report)


def test_unseen_levels_are_counted(saved: tuple[Path, Path, bool]) -> None:
    artifact, data, _ = saved
    loans = pd.read_parquet(data)
    later = loans["issue_date"] > TRAIN_END
    purpose = loans["purpose"].astype(object)
    purpose[later & (np.arange(len(loans)) % 50 == 0)] = "crypto"
    loans["purpose"] = pd.Categorical(purpose)
    report = monitor.measure(load_artifact(artifact), loans[~later], loans[later])
    assert report.unseen["purpose"]["crypto"] > 0
    assert "| purpose | crypto |" in monitor.render(report)


def test_a_different_data_file_is_refused(saved: tuple[Path, Path, bool], tmp_path: Path) -> None:
    artifact, data, _ = saved
    other = tmp_path / "other.parquet"
    pd.read_parquet(data).iloc[1:].to_parquet(other, index=False)
    with pytest.raises(DriftError, match="not the file"):
        monitor.run(artifact, other, MONITORED)


def test_a_window_overlapping_training_is_refused(saved: tuple[Path, Path, bool]) -> None:
    artifact, data, _ = saved
    overlapping = TimeWindow(pd.Timestamp("2014-06-01"), pd.Timestamp("2015-06-01"))
    with pytest.raises(DriftError, match="after the training window"):
        monitor.run(artifact, data, overlapping)


def test_an_empty_window_is_refused(saved: tuple[Path, Path, bool]) -> None:
    artifact, data, _ = saved
    empty = TimeWindow(pd.Timestamp("2019-01-01"), pd.Timestamp("2019-12-01"))
    with pytest.raises(DriftError, match="need loans in both windows"):
        monitor.run(artifact, data, empty)


def test_cli_writes_the_report(saved: tuple[Path, Path, bool], tmp_path: Path) -> None:
    artifact, data, _ = saved
    out = tmp_path / "drift.md"
    monitor.main([str(artifact), str(data), "--window", "2015-01:2016-12", "--out", str(out)])
    text = out.read_text()
    for section in ("## Score", "## Feature stability", "## What moved the score"):
        assert section in text


def test_cli_exits_cleanly_on_a_refused_window(saved: tuple[Path, Path, bool]) -> None:
    artifact, data, _ = saved
    with pytest.raises(SystemExit) as exit_info:
        monitor.main([str(artifact), str(data), "--window", "2014-01:2015-01"])
    assert exit_info.value.code == 2
