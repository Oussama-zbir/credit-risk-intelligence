import pandas as pd
import pytest

from conftest import MakeLoans
from credit_risk.data.cohort import CREDIT_POLICY_STATUSES, select_cohort
from credit_risk.data.target import KNOWN_STATUSES, UnknownStatusError, label_loans
from credit_risk.data.validation import DataValidationError, require_valid

POLICY_PAID = "Does not meet the credit policy. Status:Fully Paid"
POLICY_CHARGED_OFF = "Does not meet the credit policy. Status:Charged Off"


def test_loans_outside_the_credit_policy_are_excluded_and_counted(make_loans: MakeLoans) -> None:
    raw = make_loans(
        {"loan_status": "Fully Paid"},
        {"loan_status": POLICY_PAID},
        {"loan_status": "Charged Off"},
        {"loan_status": POLICY_CHARGED_OFF},
        {"loan_status": "Current"},
    )
    eligible, report = select_cohort(raw)
    assert eligible["loan_status"].tolist() == ["Fully Paid", "Charged Off", "Current"]
    assert (report.input_rows, report.outside_credit_policy, report.eligible) == (5, 2, 3)


def test_excluded_loans_may_miss_required_fields_eligible_ones_may_not(
    make_loans: MakeLoans,
) -> None:
    # The real export's policy loans lack bureau fields; once they are removed,
    # the full contract applies to what is left — including the same gaps.
    gaps = {"earliest_cr_line": None, "open_acc": None, "pub_rec": None, "annual_inc": None}
    eligible, _ = select_cohort(make_loans({}, {"loan_status": POLICY_CHARGED_OFF, **gaps}))
    assert require_valid(eligible).ok

    eligible, _ = select_cohort(make_loans({}, {"loan_status": "Charged Off", **gaps}))
    with pytest.raises(DataValidationError, match="earliest_cr_line:null"):
        require_valid(eligible)


def test_unknown_statuses_pass_cohort_selection_and_are_refused_at_labelling(
    make_loans: MakeLoans,
) -> None:
    # Only the exact known policy statuses are excluded; a new variant of the
    # prefix is not quietly dropped but reaches the strict status check.
    raw = make_loans({"loan_status": "Does not meet the credit policy. Status:Current"})
    eligible, report = select_cohort(raw)
    assert report.outside_credit_policy == 0
    with pytest.raises(UnknownStatusError):
        label_loans(eligible, snapshot=pd.Timestamp("2018-12-01"))


@pytest.mark.parametrize("status", sorted(CREDIT_POLICY_STATUSES))
def test_policy_loans_cannot_be_labelled_if_selection_is_skipped(
    make_loans: MakeLoans, status: str
) -> None:
    assert status not in KNOWN_STATUSES
    with pytest.raises(UnknownStatusError, match="select_cohort"):
        label_loans(make_loans({"loan_status": status}), snapshot=pd.Timestamp("2018-12-01"))
