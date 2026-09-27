from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd
import pytest

# One row shaped like the Lending Club export, including its string formats.
BASE_ROW: dict[str, Any] = {
    "issue_d": "Jan-2014",
    "loan_status": "Fully Paid",
    "term": " 36 months",
    "loan_amnt": 10_000.0,
    "purpose": "debt_consolidation",
    "application_type": "Individual",
    "emp_length": "10+ years",
    "home_ownership": "RENT",
    "annual_inc": 60_000.0,
    "verification_status": "Verified",
    "dti": 18.2,
    "fico_range_low": 690.0,
    "fico_range_high": 694.0,
    "earliest_cr_line": "Aug-2003",
    "delinq_2yrs": 0.0,
    "inq_last_6mths": 1.0,
    "open_acc": 9.0,
    "total_acc": 22.0,
    "pub_rec": 0.0,
    "pub_rec_bankruptcies": 0.0,
    "revol_bal": 12_500.0,
    "revol_util": "45.3%",
    "mort_acc": 1.0,
    "grade": "B",
    "sub_grade": "B3",
    "int_rate": "11.99%",
    "installment": 332.1,
}

MakeLoans = Callable[..., pd.DataFrame]


@pytest.fixture
def make_loans() -> MakeLoans:
    """Build a raw frame from per-row overrides of a valid base row."""

    def build(*overrides: dict[str, Any]) -> pd.DataFrame:
        return pd.DataFrame([{**BASE_ROW, **row} for row in overrides or ({},)])

    return build
