from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
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


def make_processed(loans: int = 4000, *, seed: int = 0) -> pd.DataFrame:
    """A processed frame (the output of `build_features`) with known risk drivers.

    Default odds fall with FICO and rise with DTI; the grade is a noisy copy of
    the same risk, so the pricing feature group carries real but overlapping
    signal. Issue dates span 2012-2016 so out-of-time windows can be cut.
    """
    rng = np.random.default_rng(seed)
    fico = rng.normal(700, 30, loans).round()
    dti = rng.uniform(0, 40, loans)
    risk = -0.03 * (fico - 700) + 0.05 * (dti - 20) + rng.normal(0, 0.5, loans)
    is_default = rng.random(loans) < 1 / (1 + np.exp(-(risk - 1.6)))
    grade = pd.cut(
        risk + rng.normal(0, 0.7, loans), [-np.inf, -1, 0, 1, np.inf], labels=list("ABCD")
    )
    int_rate = 7 + 4 * grade.codes + rng.normal(0, 1, loans)
    months = rng.integers(0, 60, loans)
    annual_inc = rng.lognormal(11, 0.5, loans)
    revol_util = rng.uniform(0, 100, loans)
    revol_util[rng.random(loans) < 0.05] = np.nan
    return pd.DataFrame(
        {
            "issue_date": pd.to_datetime(
                pd.DataFrame({"year": 2012 + months // 12, "month": months % 12 + 1, "day": 1})
            ),
            "loan_amnt": rng.uniform(1_000, 35_000, loans).round(-2),
            "annual_inc": annual_inc,
            "dti": dti,
            "delinq_2yrs": rng.poisson(0.3, loans).astype(float),
            "inq_last_6mths": rng.poisson(0.8, loans).astype(float),
            "open_acc": rng.poisson(11, loans).astype(float),
            "total_acc": rng.poisson(25, loans).astype(float),
            "pub_rec": rng.poisson(0.2, loans).astype(float),
            "pub_rec_bankruptcies": rng.poisson(0.1, loans).astype(float),
            "revol_bal": rng.lognormal(9, 1, loans),
            "revol_util": revol_util,
            "mort_acc": rng.poisson(1.5, loans).astype(float),
            "purpose": pd.Categorical(
                rng.choice(["debt_consolidation", "credit_card", "other"], loans)
            ),
            "application_type": pd.Categorical(["Individual"] * loans),
            "home_ownership": pd.Categorical(rng.choice(["RENT", "MORTGAGE", "OWN"], loans)),
            "verification_status": pd.Categorical(
                rng.choice(["Verified", "Source Verified", "Not Verified"], loans)
            ),
            "term_months": rng.choice([36, 60], loans),
            "fico": fico,
            "emp_length_years": rng.integers(0, 11, loans).astype(float),
            "credit_history_months": rng.integers(24, 400, loans).astype(float),
            "int_rate": int_rate,
            "installment": rng.uniform(50, 1_200, loans),
            "grade": grade,
            "sub_grade": pd.Categorical(
                [
                    f"{g}{n}"
                    for g, n in zip(grade.astype(str), rng.integers(1, 4, loans), strict=True)
                ]
            ),
            "is_default": is_default.astype("int8"),
        }
    )
