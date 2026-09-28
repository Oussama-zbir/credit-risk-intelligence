from typing import Any

import pandas as pd
import pytest

from conftest import MakeLoans
from credit_risk.data.contract import feature_columns
from credit_risk.data.features import (
    CATEGORICAL_FEATURES,
    build_features,
    feature_names,
    unconsumed_contract_columns,
)
from credit_risk.data.target import ISSUE_DATE, TARGET, label_loans

SNAPSHOT = pd.Timestamp("2018-12-01")


def features(make_loans: MakeLoans, *rows: dict[str, Any]) -> pd.DataFrame:
    labelled, _ = label_loans(make_loans(*rows), snapshot=SNAPSHOT)
    frame, _ = build_features(labelled)
    return frame


def test_every_contract_feature_column_reaches_a_processed_feature() -> None:
    assert unconsumed_contract_columns() == frozenset()


def test_lender_pricing_stays_opt_in() -> None:
    assert not {"grade", "sub_grade", "int_rate", "installment"} & set(feature_names())
    assert set(feature_names()) < set(feature_names(include_lender_pricing=True))
    # Raw inputs that are replaced by a derived feature do not leak through as-is.
    assert "fico_range_low" not in feature_names()
    assert set(feature_columns()) - set(feature_names()) == {
        "term",
        "emp_length",
        "earliest_cr_line",
        "fico_range_low",
        "fico_range_high",
    }


def test_output_is_typed_and_ordered(make_loans: MakeLoans) -> None:
    frame = features(make_loans, {})
    assert list(frame.columns) == [ISSUE_DATE, *feature_names(include_lender_pricing=True), TARGET]
    for name in CATEGORICAL_FEATURES:
        assert isinstance(frame[name].dtype, pd.CategoricalDtype)
    row = frame.iloc[0]
    assert row["int_rate"] == pytest.approx(11.99)
    assert row["revol_util"] == pytest.approx(45.3)
    assert row["fico"] == 692.0
    assert row["term_months"] == 36
    assert row["emp_length_years"] == 10
    # Aug-2003 to Jan-2014.
    assert row["credit_history_months"] == 125
    assert row[TARGET] == 0


@pytest.mark.parametrize(
    ("raw", "years"),
    [("< 1 year", 0.0), ("1 year", 1.0), ("7 years", 7.0), ("10+ years", 10.0)],
)
def test_emp_length_maps_to_years(make_loans: MakeLoans, raw: str, years: float) -> None:
    assert features(make_loans, {"emp_length": raw})["emp_length_years"].iloc[0] == years


def test_missing_emp_length_stays_missing(make_loans: MakeLoans) -> None:
    assert features(make_loans, {"emp_length": None})["emp_length_years"].isna().all()


def test_unknown_emp_length_is_refused(make_loans: MakeLoans) -> None:
    with pytest.raises(ValueError, match="emp_length"):
        features(make_loans, {"emp_length": "eleven years"})


def test_unparseable_credit_line_date_is_refused(make_loans: MakeLoans) -> None:
    with pytest.raises(ValueError, match="earliest_cr_line"):
        features(make_loans, {"earliest_cr_line": "2003-08"})


def test_impossible_values_become_missing_and_are_counted(make_loans: MakeLoans) -> None:
    labelled, _ = label_loans(
        make_loans({"dti": -1.0}, {"earliest_cr_line": "Mar-2014"}, {}),
        snapshot=SNAPSHOT,
    )
    frame, report = build_features(labelled)
    assert frame["dti"].isna().tolist() == [True, False, False]
    assert frame["credit_history_months"].isna().tolist() == [False, True, False]
    assert (report.negative_dti, report.negative_credit_history) == (1, 1)


def test_rare_home_ownership_values_share_one_bucket(make_loans: MakeLoans) -> None:
    frame = features(
        make_loans,
        {"home_ownership": "ANY"},
        {"home_ownership": "NONE"},
        {"home_ownership": "RENT"},
    )
    assert frame["home_ownership"].tolist() == ["OTHER", "OTHER", "RENT"]


def test_features_are_row_wise(make_loans: MakeLoans) -> None:
    """A loan's features do not depend on which other loans are in the frame."""
    rows: list[dict[str, Any]] = [
        {"annual_inc": 30_000.0, "dti": None},
        {"annual_inc": 250_000.0},
        {},
    ]
    alone = features(make_loans, rows[0])
    together = features(make_loans, *rows)
    numeric = [n for n in feature_names() if n not in CATEGORICAL_FEATURES]
    pd.testing.assert_frame_equal(alone[numeric], together[numeric].iloc[[0]])
