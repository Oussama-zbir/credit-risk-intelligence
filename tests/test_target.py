import pandas as pd
import pytest

from conftest import MakeLoans
from credit_risk.data.target import (
    TARGET,
    UnknownStatusError,
    label_loans,
    maturity_date,
    parse_issue_date,
    parse_term_months,
)

SNAPSHOT = pd.Timestamp("2018-12-01")


def test_charged_off_and_default_are_bad_fully_paid_is_good(make_loans: MakeLoans) -> None:
    raw = make_loans(
        {"loan_status": "Fully Paid"},
        {"loan_status": "Charged Off"},
        {"loan_status": "Default"},
        {"loan_status": "Does not meet the credit policy. Status:Charged Off"},
        {"loan_status": "Does not meet the credit policy. Status:Fully Paid"},
    )
    labelled, report = label_loans(raw, snapshot=SNAPSHOT)
    assert labelled[TARGET].tolist() == [0, 1, 1, 1, 0]
    assert report.defaults == 3
    assert report.default_rate == pytest.approx(0.6)


def test_loans_whose_term_has_not_elapsed_are_not_labelled(make_loans: MakeLoans) -> None:
    raw = make_loans(
        {"issue_d": "Dec-2015", "term": " 36 months"},  # matures Dec-2018: in
        {"issue_d": "Jan-2016", "term": " 36 months"},  # matures Jan-2019: out
        {"issue_d": "Jan-2014", "term": " 60 months"},  # matures Jan-2019: out
        # An early default in a recent vintage is exactly the loan that would
        # bias the rate if immature vintages were labelled.
        {"issue_d": "Jun-2017", "loan_status": "Charged Off"},
    )
    labelled, report = label_loans(raw, snapshot=SNAPSHOT)
    assert labelled["issue_d"].tolist() == ["Dec-2015"]
    assert report.immature == 3
    assert report.defaults == 0


def test_mature_unresolved_loans_are_dropped_and_counted(make_loans: MakeLoans) -> None:
    raw = make_loans({"loan_status": "Late (31-120 days)"}, {"loan_status": "Current"}, {})
    labelled, report = label_loans(raw, snapshot=SNAPSHOT)
    assert len(labelled) == 1
    assert report.unresolved == 2
    assert report.input_rows == 3


@pytest.mark.parametrize("status", ["Written Off", None])
def test_status_outside_the_definition_is_refused(make_loans: MakeLoans, status: object) -> None:
    with pytest.raises(UnknownStatusError):
        label_loans(make_loans({"loan_status": status}), snapshot=SNAPSHOT)


def test_maturity_date_rolls_over_years() -> None:
    issued = parse_issue_date(pd.Series(["Nov-2013", "Mar-2012"]))
    terms = parse_term_months(pd.Series([" 60 months", " 36 months"]))
    assert maturity_date(issued, terms).tolist() == [
        pd.Timestamp("2018-11-01"),
        pd.Timestamp("2015-03-01"),
    ]


def test_unparseable_dates_and_terms_fail_loudly() -> None:
    with pytest.raises(ValueError, match="issue_d"):
        parse_issue_date(pd.Series(["2015-12-01"]))
    with pytest.raises(ValueError, match="term"):
        parse_term_months(pd.Series(["three years"]))
