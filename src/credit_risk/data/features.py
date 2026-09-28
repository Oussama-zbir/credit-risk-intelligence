"""Typed features from labelled raw loans: row-wise parsing only, nothing fitted.

Every transformation here looks at one loan at a time — parse a percent string,
turn `'10+ years'` into 10, measure credit history in months. None of them
learns a statistic from the data (no imputation means, no scaling, no category
frequencies), so running this over the whole file before the out-of-time split
cannot leak the test window into training. Anything that *is* fitted belongs to
a model pipeline and is fitted on the training window only.

Missing values stay missing. Whether a missing `emp_length` is imputed, binned
as its own WoE class or passed to a tree as NaN is a modelling decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import pandas as pd

from credit_risk.data.contract import Role, columns_with
from credit_risk.data.target import ISSUE_DATE, TARGET, TERM_MONTHS
from credit_risk.data.validation import as_number

EMP_LENGTH_YEARS: Final[dict[str, int]] = {
    "< 1 year": 0,
    "1 year": 1,
    **{f"{n} years": n for n in range(2, 10)},
    "10+ years": 10,
}

# Values with a handful of loans each; one bucket instead of three near-empty ones.
HOME_OWNERSHIP_OTHER: Final = frozenset({"ANY", "NONE", "OTHER"})

APPLICANT_NUMERIC: Final = (
    "loan_amnt",
    "annual_inc",
    "dti",
    "delinq_2yrs",
    "inq_last_6mths",
    "open_acc",
    "total_acc",
    "pub_rec",
    "pub_rec_bankruptcies",
    "revol_bal",
    "revol_util",
    "mort_acc",
)
APPLICANT_CATEGORICAL: Final = (
    "purpose",
    "application_type",
    "home_ownership",
    "verification_status",
)
DERIVED: Final[dict[str, tuple[str, ...]]] = {
    TERM_MONTHS: ("term",),
    "fico": ("fico_range_low", "fico_range_high"),
    "emp_length_years": ("emp_length",),
    "credit_history_months": ("earliest_cr_line", "issue_d"),
}
PRICING_NUMERIC: Final = ("int_rate", "installment")
PRICING_CATEGORICAL: Final = ("grade", "sub_grade")

APPLICANT_FEATURES: Final = APPLICANT_NUMERIC + APPLICANT_CATEGORICAL + tuple(DERIVED)
PRICING_FEATURES: Final = PRICING_NUMERIC + PRICING_CATEGORICAL
CATEGORICAL_FEATURES: Final = APPLICANT_CATEGORICAL + PRICING_CATEGORICAL


def feature_names(*, include_lender_pricing: bool = False) -> tuple[str, ...]:
    """Processed columns a model may be trained on; pricing is opt-in, as in the contract."""
    return APPLICANT_FEATURES + (PRICING_FEATURES if include_lender_pricing else ())


def consumed_columns() -> frozenset[str]:
    """Raw contract columns that some processed feature is built from."""
    direct = APPLICANT_NUMERIC + APPLICANT_CATEGORICAL + PRICING_FEATURES
    return frozenset(direct).union(*DERIVED.values())


@dataclass(frozen=True, slots=True)
class FeatureReport:
    """Values present in the raw file but set to missing, and why."""

    negative_dti: int
    negative_credit_history: int


def _month_index(dates: pd.Series) -> pd.Series:
    index: pd.Series = dates.dt.year * 12 + dates.dt.month
    return index


def _parse_months(values: pd.Series, column: str) -> pd.Series:
    parsed = pd.to_datetime(values, format="%b-%Y", errors="coerce")
    bad = values[parsed.isna() & values.notna()]
    if not bad.empty:
        raise ValueError(f"unparseable {column} values: {sorted(bad.astype(str).unique())[:5]}")
    return parsed


def _parse_emp_length(values: pd.Series) -> pd.Series:
    years = values.map(EMP_LENGTH_YEARS)
    bad = values[years.isna() & values.notna()]
    if not bad.empty:
        raise ValueError(f"unknown emp_length values: {sorted(bad.astype(str).unique())[:5]}")
    return years.astype("float64")


def build_features(labelled: pd.DataFrame) -> tuple[pd.DataFrame, FeatureReport]:
    """Turn the output of `label_loans` into a typed modelling frame.

    Returns every applicant and pricing feature plus `issue_date` and the
    target; `feature_names` selects the set a model trains on. Raises on
    values the parsers do not recognise rather than coercing them to missing.
    """
    out = pd.DataFrame(index=labelled.index)
    out[ISSUE_DATE] = labelled[ISSUE_DATE]

    for name in APPLICANT_NUMERIC + PRICING_NUMERIC:
        out[name] = as_number(labelled[name]).astype("float64")

    dti = out["dti"]
    negative_dti = dti < 0
    out["dti"] = dti.mask(negative_dti)

    out[TERM_MONTHS] = labelled[TERM_MONTHS].astype("int64")
    # The bureau reports a 4-point band; the midpoint carries all of its signal.
    out["fico"] = (
        as_number(labelled["fico_range_low"]) + as_number(labelled["fico_range_high"])
    ) / 2
    out["emp_length_years"] = _parse_emp_length(labelled["emp_length"])

    opened = _parse_months(labelled["earliest_cr_line"], "earliest_cr_line")
    history = (_month_index(labelled[ISSUE_DATE]) - _month_index(opened)).astype("float64")
    negative_history = history < 0
    out["credit_history_months"] = history.mask(negative_history)

    for name in CATEGORICAL_FEATURES:
        out[name] = labelled[name].astype("category")
    home = labelled["home_ownership"]
    out["home_ownership"] = home.mask(home.isin(HOME_OWNERSHIP_OTHER), "OTHER").astype("category")

    out[TARGET] = labelled[TARGET].astype("int8")
    report = FeatureReport(
        negative_dti=int(negative_dti.sum()),
        negative_credit_history=int(negative_history.sum()),
    )
    return out[[ISSUE_DATE, *feature_names(include_lender_pricing=True), TARGET]], report


def unconsumed_contract_columns() -> frozenset[str]:
    """Applicant or pricing columns in the contract that no feature uses.

    Empty by construction; a test holds it there, so a column added to the
    contract cannot be silently dropped on the way to the model.
    """
    return frozenset(columns_with(Role.APPLICANT, Role.LENDER_PRICING)) - consumed_columns()
