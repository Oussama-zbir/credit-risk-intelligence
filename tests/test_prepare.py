from pathlib import Path

import pandas as pd
import pytest

from conftest import MakeLoans
from credit_risk.data.features import feature_names
from credit_risk.data.loader import infer_snapshot, read_raw
from credit_risk.data.target import TARGET
from credit_risk.data.validation import DataValidationError
from credit_risk.prepare import main, prepare


@pytest.fixture
def export(tmp_path: Path, make_loans: MakeLoans) -> Path:
    """A small export with the shapes of the real one: a leakage column, footer lines."""
    raw = make_loans(
        {"issue_d": "Jan-2012", "loan_status": "Charged Off", "last_pymnt_d": "Jun-2013"},
        {"issue_d": "Jan-2014", "loan_status": "Fully Paid", "last_pymnt_d": "Jan-2017"},
        {"issue_d": "Jan-2014", "term": " 60 months", "loan_status": "Current"},
        {"issue_d": "Jun-2016", "loan_status": "Late (31-120 days)", "last_pymnt_d": "Mar-2019"},
        {"issue_d": "Jun-2013", "loan_status": "Fully Paid", "last_pymnt_d": "Jun-2016"},
    )
    raw["recoveries"] = 0.0
    footer = pd.DataFrame([{"loan_amnt": None}, {"loan_amnt": None}]).reindex(columns=raw.columns)
    path = tmp_path / "accepted.csv.gz"
    pd.concat([raw, footer]).to_csv(path, index=False)
    return path


def test_loader_reads_only_contract_columns_and_drops_footer_lines(export: Path) -> None:
    raw, report = read_raw(export)
    assert "recoveries" not in raw.columns
    assert "last_pymnt_d" not in raw.columns
    assert (report.rows_read, report.summary_rows_dropped, len(raw)) == (7, 2, 5)


def test_snapshot_is_the_latest_payment_month(export: Path) -> None:
    assert infer_snapshot(export) == pd.Timestamp("2019-03-01")


def test_prepare_produces_the_labelled_population_and_its_report(export: Path) -> None:
    prepared = prepare(export)
    # Snapshot Mar-2019: the Jun-2016 36-month loan is not yet mature, the
    # 60-month Jan-2014 loan is matured but unresolved, three loans are labelled.
    labels = prepared.report.labels
    assert (labels.immature, labels.unresolved, labels.labelled, labels.defaults) == (1, 1, 3, 1)
    assert prepared.frame[TARGET].tolist() == [1, 0, 0]
    assert len(prepared.report.sha256) == 64


def test_contract_errors_stop_the_pipeline(tmp_path: Path, make_loans: MakeLoans) -> None:
    path = tmp_path / "bad.csv"
    raw = make_loans({"fico_range_low": 2000.0})
    raw["last_pymnt_d"] = "Jan-2017"
    raw.to_csv(path, index=False)
    with pytest.raises(DataValidationError, match="fico_range_low"):
        prepare(path)


def test_cli_writes_a_typed_parquet_and_a_report(export: Path, tmp_path: Path) -> None:
    out = tmp_path / "processed"
    main([str(export), "--out-dir", str(out)])

    frame = pd.read_parquet(out / "loans.parquet")
    assert set(feature_names(include_lender_pricing=True)) <= set(frame.columns)
    assert isinstance(frame["grade"].dtype, pd.CategoricalDtype)

    report = (out / "data_report.md").read_text()
    assert "Snapshot (latest `last_pymnt_d`): 2019-03" in report
    assert "| 2012 | 1 | 100.00% | 0.0% |" in report
    assert "| emp_length" not in report
