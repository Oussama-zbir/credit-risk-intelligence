"""The column contract for the Lending Club accepted-loans file.

Every column the project reads is declared here with the *role* it plays, and
the role — not a column's apparent usefulness — decides whether a model may see
it. The file ships ~150 columns; most of them describe what happened to a loan
after it was issued, and a model trained on them scores well offline for the
same reason it is useless at application time: it is reading the outcome.

A column that is not declared is never loaded. Adding a feature is therefore a
reviewed change to this file, with a written reason, rather than a side effect
of `pd.read_csv` returning whatever the vendor added to the export.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Role(StrEnum):
    """Why a column is (or is not) available to a model at decision time."""

    APPLICANT = "applicant"
    """Known from the application and the credit bureau pull at origination."""

    LENDER_PRICING = "lender_pricing"
    """Lending Club's own decision about the applicant: grade and price.

    Known at origination, so not leakage in the temporal sense, but it is the
    output of another credit model. A model that leans on it is partly
    re-learning Lending Club's scorecard. Kept as a separate group so its
    contribution can be measured by training with and without it.
    """

    POST_ORIGINATION = "post_origination"
    """Written after the loan was issued: payments, recoveries, later bureau
    pulls. Direct target leakage — never a feature."""

    EXCLUDED = "excluded"
    """Available at origination but deliberately not used (free text,
    geographic proxies). The reason is recorded on the column."""

    TIME = "time"
    """The issue date: defines vintages, maturity and the out-of-time split."""

    TARGET = "target"
    """The loan outcome the default definition is derived from."""


class Kind(StrEnum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATE = "date"
    TEXT = "text"


@dataclass(frozen=True, slots=True)
class Column:
    name: str
    role: Role
    kind: Kind
    reason: str = ""


_A, _P, _L, _X = Role.APPLICANT, Role.LENDER_PRICING, Role.POST_ORIGINATION, Role.EXCLUDED
_N, _C, _D, _T = Kind.NUMERIC, Kind.CATEGORICAL, Kind.DATE, Kind.TEXT

COLUMNS: tuple[Column, ...] = (
    # Time and target.
    Column("issue_d", Role.TIME, _D, "Month the loan was funded (e.g. 'Dec-2015')."),
    Column("loan_status", Role.TARGET, _C, "Outcome at the data snapshot."),
    Column("term", _A, _C, "' 36 months' or ' 60 months'; also fixes the maturity date."),
    # The request.
    Column("loan_amnt", _A, _N, "Amount the applicant asked for."),
    Column("purpose", _A, _C),
    Column("application_type", _A, _C, "Individual or joint."),
    # Applicant and income.
    Column("emp_length", _A, _C, "Self-reported, '< 1 year' .. '10+ years'."),
    Column("home_ownership", _A, _C),
    Column("annual_inc", _A, _N, "Self-reported."),
    Column("verification_status", _A, _C, "Whether income was verified."),
    Column("dti", _A, _N, "Debt-to-income, in percent, excluding the new loan."),
    # Bureau at origination.
    Column("fico_range_low", _A, _N),
    Column("fico_range_high", _A, _N),
    Column("earliest_cr_line", _A, _D, "Converted to credit-history length at issue."),
    Column("delinq_2yrs", _A, _N),
    Column("inq_last_6mths", _A, _N),
    Column("open_acc", _A, _N),
    Column("total_acc", _A, _N),
    Column("pub_rec", _A, _N),
    Column("pub_rec_bankruptcies", _A, _N),
    Column("revol_bal", _A, _N),
    Column("revol_util", _A, _N, "Percent; arrives as '45.3%' in some exports."),
    Column("mort_acc", _A, _N),
    # Lending Club's decision.
    Column("grade", _P, _C),
    Column("sub_grade", _P, _C),
    Column("int_rate", _P, _N, "Priced from sub_grade, so it carries the same signal."),
    Column("installment", _P, _N, "A function of amount, term and int_rate."),
    # Leakage: written after issue. Declared so the audit is explicit and tested.
    Column("funded_amnt_inv", _L, _N, "Investor funding completes after listing."),
    Column("out_prncp", _L, _N, "Outstanding principal at the snapshot."),
    Column("total_pymnt", _L, _N, "Payments received to date."),
    Column("total_rec_prncp", _L, _N),
    Column("total_rec_int", _L, _N),
    Column("total_rec_late_fee", _L, _N, "Non-zero only for loans that went late."),
    Column("recoveries", _L, _N, "Non-zero only for charged-off loans: the label itself."),
    Column("collection_recovery_fee", _L, _N, "Non-zero only for charged-off loans."),
    Column("last_pymnt_d", _L, _D),
    Column("last_pymnt_amnt", _L, _N),
    Column("last_credit_pull_d", _L, _D),
    Column("last_fico_range_high", _L, _N, "Bureau score pulled after the outcome."),
    Column("last_fico_range_low", _L, _N, "Bureau score pulled after the outcome."),
    Column("hardship_flag", _L, _C),
    Column("debt_settlement_flag", _L, _C, "Settlements follow a charge-off."),
    # Available, but not used.
    Column("emp_title", _X, _T, "Free text, ~500k distinct values; revisit with NLP."),
    Column("title", _X, _T, "Free-text restatement of purpose."),
    Column("desc", _X, _T, "Mostly empty after 2014."),
    Column("zip_code", _X, _C, "3-digit ZIP is a proxy for protected attributes."),
    Column("addr_state", _X, _C, "Geographic proxy; excluded with zip_code."),
)

BY_NAME: dict[str, Column] = {column.name: column for column in COLUMNS}


def columns_with(*roles: Role) -> tuple[str, ...]:
    """Names of the declared columns playing any of `roles`, in contract order."""
    return tuple(column.name for column in COLUMNS if column.role in roles)


def feature_columns(*, include_lender_pricing: bool = False) -> tuple[str, ...]:
    """Columns a model may be trained on.

    Lender pricing is opt-in so that every result states which feature set it
    was measured on.
    """
    roles = (Role.APPLICANT, Role.LENDER_PRICING) if include_lender_pricing else (Role.APPLICANT,)
    return columns_with(*roles)


def loaded_columns() -> tuple[str, ...]:
    """Columns read from the raw file: features, time and target.

    Post-origination and excluded columns are declared but never loaded, so
    they cannot reach a feature matrix by accident further down the pipeline.
    """
    return columns_with(Role.TIME, Role.TARGET, Role.APPLICANT, Role.LENDER_PRICING)
