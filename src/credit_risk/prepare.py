"""Raw export -> validated, labelled, typed modelling frame plus a data report.

    python -m credit_risk.prepare data/accepted_2007_to_2018Q4.csv.gz

The same input file always produces the same output: every stage is a pure
function of the file, and the snapshot date comes from the file itself unless
it is pinned with `--snapshot`. The report records the file's SHA-256, so a
processed frame can be traced to the exact export it came from.
"""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from credit_risk.data.features import build_features
from credit_risk.data.loader import infer_snapshot, read_raw
from credit_risk.data.report import DataReport, render
from credit_risk.data.target import label_loans
from credit_risk.data.validation import require_valid


@dataclass(frozen=True, slots=True)
class Prepared:
    frame: pd.DataFrame
    report: DataReport


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(raw_path: Path, *, snapshot: pd.Timestamp | None = None) -> Prepared:
    """Run load -> validate -> label -> features; raises on any contract error."""
    snapshot = infer_snapshot(raw_path) if snapshot is None else snapshot
    raw, load = read_raw(raw_path)
    validation = require_valid(raw)
    labelled, labels = label_loans(raw, snapshot=snapshot)
    frame, features = build_features(labelled)
    report = DataReport(
        source=raw_path.name,
        sha256=sha256(raw_path),
        snapshot=snapshot,
        load=load,
        validation=validation,
        labels=labels,
        features=features,
    )
    return Prepared(frame=frame.reset_index(drop=True), report=report)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("raw", type=Path, help="Lending Club export (.csv or .csv.gz)")
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--snapshot", type=pd.Timestamp, help="override, e.g. 2019-03-01")
    args = parser.parse_args(argv)

    prepared = prepare(args.raw, snapshot=args.snapshot)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    prepared.frame.to_parquet(args.out_dir / "loans.parquet", index=False)
    (args.out_dir / "data_report.md").write_text(render(prepared.report, prepared.frame))
    labels = prepared.report.labels
    print(f"{labels.labelled:,} labelled loans, default rate {labels.default_rate:.2%}")
    print(f"wrote {args.out_dir}/loans.parquet and {args.out_dir}/data_report.md")


if __name__ == "__main__":
    main()
