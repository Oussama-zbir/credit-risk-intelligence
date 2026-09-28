"""Ingestion checks on the raw file, before anything is derived from it.

Two severities. An **error** means the file does not match the contract — a
column is missing, or a value is impossible (a negative loan, a FICO score of
2000) — and the pipeline stops. A **warning** records a known quirk of the
source that is handled downstream, so its size is visible in every run instead
of being cleaned away silently.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import pandas as pd

from credit_risk.data.contract import BY_NAME, Kind, loaded_columns


class Severity(StrEnum):
    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True, slots=True)
class Issue:
    column: str
    check: str
    rows: int
    severity: Severity


@dataclass(frozen=True, slots=True)
class ValidationReport:
    rows: int
    issues: tuple[Issue, ...]

    @property
    def errors(self) -> tuple[Issue, ...]:
        return tuple(issue for issue in self.issues if issue.severity is Severity.ERROR)

    @property
    def ok(self) -> bool:
        return not self.errors


class DataValidationError(ValueError):
    def __init__(self, report: ValidationReport) -> None:
        self.report = report
        summary = ", ".join(f"{i.column}:{i.check} ({i.rows} rows)" for i in report.errors)
        super().__init__(f"raw data violates the contract: {summary}")


@dataclass(frozen=True, slots=True)
class RangeRule:
    column: str
    low: float | None
    high: float | None
    severity: Severity


RANGE_RULES: tuple[RangeRule, ...] = (
    RangeRule("loan_amnt", 1, 40_000, Severity.ERROR),
    RangeRule("annual_inc", 0, None, Severity.ERROR),
    RangeRule("fico_range_low", 300, 850, Severity.ERROR),
    RangeRule("fico_range_high", 300, 850, Severity.ERROR),
    RangeRule("int_rate", 0, 100, Severity.ERROR),
    # Lending Club reports dti of -1 and values far above 100 for a small number
    # of loans (mostly joint applications and zero stated income). Real, and
    # handled at feature time, not a reason to reject the file.
    RangeRule("dti", 0, 100, Severity.WARNING),
    RangeRule("revol_util", 0, 150, Severity.WARNING),
)

# Missing values the source is known to have; reported, never an error.
NULLABLE: frozenset[str] = frozenset(
    {
        "emp_length",
        "dti",
        "revol_util",
        "mort_acc",
        "pub_rec_bankruptcies",
        "inq_last_6mths",
    }
)


def as_number(values: pd.Series) -> pd.Series:
    """Numeric view of a column that may arrive as `'13.56%'`."""
    if pd.api.types.is_numeric_dtype(values):
        return values
    return pd.to_numeric(values.astype("string").str.strip().str.rstrip("%"), errors="coerce")


def validate_raw(raw: pd.DataFrame) -> ValidationReport:
    """Check a raw frame against the column contract without modifying it."""
    issues: list[Issue] = []
    present = set(raw.columns)

    for name in loaded_columns():
        if name not in present:
            issues.append(Issue(name, "missing_column", len(raw), Severity.ERROR))
            continue
        nulls = int(raw[name].isna().sum())
        if nulls:
            severity = Severity.WARNING if name in NULLABLE else Severity.ERROR
            issues.append(Issue(name, "null", nulls, severity))
        if BY_NAME[name].kind is Kind.NUMERIC:
            values = as_number(raw[name])
            unparsed = int((values.isna() & raw[name].notna()).sum())
            if unparsed:
                issues.append(Issue(name, "not_numeric", unparsed, Severity.ERROR))

    for rule in RANGE_RULES:
        if rule.column not in present:
            continue
        values = as_number(raw[rule.column])
        outside = pd.Series(False, index=raw.index)
        if rule.low is not None:
            outside |= values < rule.low
        if rule.high is not None:
            outside |= values > rule.high
        if count := int(outside.sum()):
            issues.append(Issue(rule.column, "out_of_range", count, rule.severity))

    return ValidationReport(rows=len(raw), issues=tuple(issues))


def require_valid(raw: pd.DataFrame) -> ValidationReport:
    """`validate_raw`, raising `DataValidationError` if any check is an error."""
    report = validate_raw(raw)
    if not report.ok:
        raise DataValidationError(report)
    return report
