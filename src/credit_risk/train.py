"""Fit the baseline on one issue-date window and score it on a later one.

    python -m credit_risk.train data/processed/loans.parquet \\
        --train 2012-01:2014-12 --test 2015-07:2016-03

Two models are fitted on the same split: applicant features only, and
applicant features plus Lending Club's own grade and interest rate. The gap
between them is how much of the result is re-learning the lender's scorecard
rather than assessing the borrower, which is why pricing is never silently in
the default feature set.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from credit_risk.data.features import feature_names
from credit_risk.data.split import OutOfTimeSplit, TimeWindow, out_of_time_split
from credit_risk.data.target import TARGET
from credit_risk.models.baseline import FittedBaseline, fit_baseline
from credit_risk.models.metrics import Scores, score

TOP_COEFFICIENTS = 10


@dataclass(frozen=True, slots=True)
class Evaluation:
    name: str
    model: FittedBaseline
    train: Scores
    test: Scores


def parse_window(text: str) -> TimeWindow:
    """`YYYY-MM:YYYY-MM`, both issue months inclusive."""
    start, sep, end = text.partition(":")
    if not sep:
        raise argparse.ArgumentTypeError(f"expected YYYY-MM:YYYY-MM, got {text!r}")
    try:
        return TimeWindow(pd.Timestamp(f"{start}-01"), pd.Timestamp(f"{end}-01"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def evaluate(split: OutOfTimeSplit) -> list[Evaluation]:
    results = []
    for name, pricing in (("applicant", False), ("applicant + lender pricing", True)):
        model = fit_baseline(split.train, feature_names(include_lender_pricing=pricing))
        rate = model.train_default_rate
        results.append(
            Evaluation(
                name=name,
                model=model,
                train=score(
                    split.train[TARGET], model.predict_pd(split.train), reference_rate=rate
                ),
                test=score(split.test[TARGET], model.predict_pd(split.test), reference_rate=rate),
            )
        )
    return results


def _window(window: TimeWindow) -> str:
    return f"{window.start:%Y-%m} to {window.end:%Y-%m}"


def render(split: OutOfTimeSplit, results: list[Evaluation]) -> str:
    lines = [
        "# Baseline: logistic regression",
        "",
        f"- Train: issued {_window(split.train_window)}, {len(split.train):,} loans",
        f"- Test (out of time): issued {_window(split.test_window)}, {len(split.test):,} loans",
        "",
        "Brier skill is measured against predicting the training default rate for every loan.",
        "",
        "| features | window | loans | default rate | mean PD | AUC | Gini | KS | Brier "
        "| Brier skill | log loss |",
        "|" + " --- |" * 11,
    ]
    for result in results:
        for window, s in (("train", result.train), ("test", result.test)):
            lines.append(
                f"| {result.name} | {window} | {s.loans:,} | {s.default_rate:.2%} "
                f"| {s.mean_pd:.2%} | {s.auc:.4f} | {s.gini:.4f} | {s.ks:.4f} "
                f"| {s.brier:.4f} | {s.brier_skill:.2%} | {s.log_loss:.4f} |"
            )
    for result in results:
        top = result.model.coefficients().head(TOP_COEFFICIENTS)
        lines += [
            "",
            f"## Largest coefficients — {result.name}",
            "",
            "| input | coefficient (log-odds) |",
            "| --- | --- |",
            *(f"| {name} | {weight:+.3f} |" for name, weight in top.items()),
        ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("frame", type=Path, help="loans.parquet written by credit_risk.prepare")
    parser.add_argument("--train", type=parse_window, required=True, metavar="YYYY-MM:YYYY-MM")
    parser.add_argument("--test", type=parse_window, required=True, metavar="YYYY-MM:YYYY-MM")
    parser.add_argument("--out", type=Path, default=Path("data/processed/baseline_report.md"))
    args = parser.parse_args(argv)

    split = out_of_time_split(pd.read_parquet(args.frame), train=args.train, test=args.test)
    results = evaluate(split)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(split, results))
    for result in results:
        print(f"{result.name}: test AUC {result.test.auc:.4f}, KS {result.test.ks:.4f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
