"""The data report: what one run of the pipeline read, dropped, and produced.

Written as Markdown next to the processed file so that the population a model
was trained on can be checked in review — rows lost at each stage, the default
rate by vintage, and how much of each feature is missing — instead of being
reconstructed from memory when a metric looks wrong.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from credit_risk.data.features import FeatureReport, feature_names
from credit_risk.data.loader import LoadReport
from credit_risk.data.target import ISSUE_DATE, TARGET, TERM_MONTHS, LabelReport
from credit_risk.data.validation import Severity, ValidationReport


@dataclass(frozen=True, slots=True)
class DataReport:
    source: str
    sha256: str
    snapshot: pd.Timestamp
    load: LoadReport
    validation: ValidationReport
    labels: LabelReport
    features: FeatureReport


def vintage_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Loans, default rate and 60-month share per issue year."""
    grouped = frame.groupby(frame[ISSUE_DATE].dt.year)
    table = pd.DataFrame(
        {
            "loans": grouped.size(),
            "default_rate": grouped[TARGET].mean(),
            "share_60m": grouped[TERM_MONTHS].apply(lambda months: (months == 60).mean()),
        }
    )
    table.index.name = "issue_year"
    return table


def missing_table(frame: pd.DataFrame) -> pd.Series:
    """Share of missing values per feature, features without gaps omitted."""
    shares = frame[list(feature_names(include_lender_pricing=True))].isna().mean()
    return shares[shares > 0].sort_values(ascending=False)


def _rows(header: tuple[str, ...], rows: list[tuple[object, ...]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + " --- |" * len(header)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return lines


def render(report: DataReport, frame: pd.DataFrame) -> str:
    load, labels = report.load, report.labels
    lines = [
        "# Data report",
        "",
        f"- Source: `{report.source}`",
        f"- SHA-256: `{report.sha256}`",
        f"- Snapshot (latest `last_pymnt_d`): {report.snapshot:%Y-%m}",
        "",
        "## Population",
        "",
        *_rows(
            ("stage", "rows"),
            [
                ("read", f"{load.rows_read:,}"),
                ("summary lines dropped", f"{load.summary_rows_dropped:,}"),
                ("immature at snapshot", f"{labels.immature:,}"),
                ("matured but unresolved", f"{labels.unresolved:,}"),
                ("labelled", f"{labels.labelled:,}"),
                ("defaults", f"{labels.defaults:,} ({labels.default_rate:.2%})"),
            ],
        ),
        "",
        "## Vintages",
        "",
        *_rows(
            ("issue year", "loans", "default rate", "60-month share"),
            [
                (
                    row.issue_year,
                    f"{row.loans:,}",
                    f"{row.default_rate:.2%}",
                    f"{row.share_60m:.1%}",
                )
                for row in vintage_table(frame).reset_index().itertuples()
            ],
        ),
        "",
        "## Validation warnings",
        "",
    ]
    warnings = [i for i in report.validation.issues if i.severity is Severity.WARNING]
    lines += (
        _rows(("column", "check", "rows"), [(i.column, i.check, f"{i.rows:,}") for i in warnings])
        if warnings
        else ["None."]
    )
    lines += [
        "",
        "## Set to missing at feature time",
        "",
        f"- `dti` below zero: {report.features.negative_dti:,}",
        f"- credit line opened after issue: {report.features.negative_credit_history:,}",
        "",
        "## Missing values by feature",
        "",
    ]
    missing = missing_table(frame)
    lines += (
        _rows(("feature", "missing"), [(name, f"{share:.2%}") for name, share in missing.items()])
        if not missing.empty
        else ["None."]
    )
    return "\n".join(lines) + "\n"
