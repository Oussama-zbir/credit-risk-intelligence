"""The default definition, and the maturity rule that makes it honest.

**Default** is a loan that was charged off (or reported in default) at any
point before the data snapshot. **Good** is a loan that was repaid in full.
Everything else — current, in grace, late but not yet charged off — has not
resolved, and is not a label of either kind.

Dropping unresolved loans is where most public work on this dataset goes wrong.
A loan resolves early only by prepaying or by defaulting; a loan that is simply
paying on schedule stays `Current` until its term ends. So among *recent*
vintages the resolved loans are dominated by early defaulters and prepayers,
and the observed default rate is biased in a direction that depends on the
vintage — exactly the axis the out-of-time split cuts along.

The fix is an observation window: only loans whose full contractual term has
elapsed by the snapshot are labelled. Among those, unresolved loans are rare
(genuine late-stage delinquencies and restructurings), and `label_loans`
reports how many it had to drop so that the rate can be checked, not assumed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import pandas as pd

DEFAULT_STATUSES: Final = frozenset(
    {
        "Charged Off",
        "Default",
        "Does not meet the credit policy. Status:Charged Off",
    }
)
GOOD_STATUSES: Final = frozenset(
    {
        "Fully Paid",
        "Does not meet the credit policy. Status:Fully Paid",
    }
)
UNRESOLVED_STATUSES: Final = frozenset(
    {
        "Current",
        "In Grace Period",
        "Late (16-30 days)",
        "Late (31-120 days)",
    }
)
KNOWN_STATUSES: Final = DEFAULT_STATUSES | GOOD_STATUSES | UNRESOLVED_STATUSES

TARGET: Final = "is_default"
ISSUE_DATE: Final = "issue_date"
TERM_MONTHS: Final = "term_months"


class UnknownStatusError(ValueError):
    """A `loan_status` value the default definition does not classify.

    Raised rather than mapped to either class: a new status silently counted as
    good or bad would move the label without anyone deciding that it should.
    """


@dataclass(frozen=True, slots=True)
class LabelReport:
    """What `label_loans` kept and dropped, so the population is auditable."""

    input_rows: int
    immature: int
    unresolved: int
    labelled: int
    defaults: int

    @property
    def default_rate(self) -> float:
        return self.defaults / self.labelled if self.labelled else float("nan")


def parse_issue_date(values: pd.Series) -> pd.Series:
    """Parse Lending Club's `'Dec-2015'` month format to month-start timestamps."""
    parsed = pd.to_datetime(values, format="%b-%Y", errors="coerce")
    bad = values[parsed.isna() & values.notna()]
    if not bad.empty:
        raise ValueError(f"unparseable issue_d values: {sorted(bad.astype(str).unique())[:5]}")
    return parsed


def parse_term_months(values: pd.Series) -> pd.Series:
    """Parse `' 36 months'` / `' 60 months'` to an integer month count."""
    months = pd.to_numeric(
        values.astype("string").str.extract(r"^\s*(\d+)\s*months\s*$", expand=False),
        errors="coerce",
    )
    bad = values[months.isna() & values.notna()]
    if not bad.empty:
        raise ValueError(f"unparseable term values: {sorted(bad.astype(str).unique())[:5]}")
    return months.astype("Int64")


def maturity_date(issue_date: pd.Series, term_months: pd.Series) -> pd.Series:
    """Month in which each loan's final scheduled payment falls due.

    Month arithmetic on integer month indices, so it stays vectorised over the
    full ~2M-row file instead of building a `DateOffset` per loan.
    """
    index = issue_date.dt.year * 12 + (issue_date.dt.month - 1) + term_months.astype("int64")
    return pd.to_datetime(
        pd.DataFrame({"year": index // 12, "month": index % 12 + 1, "day": 1}),
    )


def label_loans(raw: pd.DataFrame, *, snapshot: pd.Timestamp) -> tuple[pd.DataFrame, LabelReport]:
    """Apply the observation window and the default definition.

    Returns the matured, resolved loans with parsed `issue_date`, `term_months`
    and a binary `is_default`, plus a report of what was dropped and why.
    `snapshot` is the date the export was taken: a loan is mature when its
    final scheduled payment falls due on or before it.
    """
    unknown = set(raw["loan_status"].dropna().unique()) - KNOWN_STATUSES
    if unknown or raw["loan_status"].isna().any():
        found = sorted(unknown) + (["<missing>"] if raw["loan_status"].isna().any() else [])
        raise UnknownStatusError(f"loan_status values outside the default definition: {found}")

    frame = raw.copy()
    frame[ISSUE_DATE] = parse_issue_date(frame["issue_d"])
    frame[TERM_MONTHS] = parse_term_months(frame["term"])

    mature = maturity_date(frame[ISSUE_DATE], frame[TERM_MONTHS]) <= snapshot
    resolved = frame["loan_status"].isin(DEFAULT_STATUSES | GOOD_STATUSES)
    kept = frame.loc[mature & resolved].copy()
    kept[TARGET] = kept["loan_status"].isin(DEFAULT_STATUSES).astype("int8")

    report = LabelReport(
        input_rows=len(frame),
        immature=int((~mature).sum()),
        unresolved=int((mature & ~resolved).sum()),
        labelled=len(kept),
        defaults=int(kept[TARGET].sum()),
    )
    return kept, report
