"""Reading the raw export: only contract columns, and the snapshot it was taken at.

The Lending Club file is ~1.6 GB with ~150 columns. `read_raw` asks pandas for
the contract's loaded columns only, so a post-origination column cannot reach a
frame even by accident, and memory stays proportional to what is used.

Everything is read as text. Parsing is the job of validation and feature
building, which report what they could not parse; letting `read_csv` guess
dtypes would instead turn a stray `'n/a'` into a silently mixed column.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pandas as pd

from credit_risk.data.contract import loaded_columns

SNAPSHOT_COLUMN: Final = "last_pymnt_d"


@dataclass(frozen=True, slots=True)
class LoadReport:
    rows_read: int
    summary_rows_dropped: int


def read_raw(path: Path) -> tuple[pd.DataFrame, LoadReport]:
    """Read the contract's loaded columns from a raw export (`.csv` or `.csv.gz`).

    The export carries summary lines ("Total amount funded in policy code ...")
    in its first column with every other field empty. They are recognised by
    having neither an issue date nor a status, counted, and dropped; any other
    incomplete row is left for validation to reject.
    """
    frame = pd.read_csv(path, usecols=list(loaded_columns()), dtype=str, low_memory=False)
    summary = frame["issue_d"].isna() & frame["loan_status"].isna()
    report = LoadReport(rows_read=len(frame), summary_rows_dropped=int(summary.sum()))
    return frame.loc[~summary].reset_index(drop=True), report


def infer_snapshot(path: Path) -> pd.Timestamp:
    """The month the export was taken: its latest `last_pymnt_d`.

    `last_pymnt_d` is post-origination and never a feature. It is read here on
    its own, as metadata about the file, because the maturity window needs to
    know when "now" was for this export — and a hard-coded date would silently
    go stale the day someone drops in a newer file.
    """
    dates = pd.read_csv(path, usecols=[SNAPSHOT_COLUMN], dtype=str)[SNAPSHOT_COLUMN]
    parsed = pd.to_datetime(dates, format="%b-%Y", errors="coerce")
    if parsed.isna().all():
        raise ValueError(f"no parseable {SNAPSHOT_COLUMN} values in {path}")
    return pd.Timestamp(parsed.max())
