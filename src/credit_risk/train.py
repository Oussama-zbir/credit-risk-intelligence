"""Fit models on one issue-date window and compare them on a later one.

    python -m credit_risk.train data/processed/loans.parquet \\
        --train 2012-01:2014-12 --test 2015-07:2016-03

Every model is fitted twice on the same split: applicant features only, and
applicant features plus Lending Club's own grade and interest rate. The gap
between them is how much of the result is re-learning the lender's scorecard
rather than assessing the borrower, which is why pricing is never silently in
the default feature set.

Per feature set, three rows: the logistic-regression baseline (fitted on the
whole training window), gradient boosting as fitted, and the same booster
recalibrated. The booster is fitted on the training window minus its latest
`--calibration-months`, which the calibrator is fitted on instead — so the
booster sees fewer loans than the baseline, and has to win anyway.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

from credit_risk.data.features import feature_names
from credit_risk.data.split import (
    OutOfTimeSplit,
    TimeWindow,
    hold_out_latest,
    out_of_time_split,
)
from credit_risk.data.target import TARGET
from credit_risk.models.baseline import FittedBaseline, fit_baseline
from credit_risk.models.boosting import FittedBooster, fit_booster
from credit_risk.models.calibration import Method
from credit_risk.models.metrics import ReliabilityBin, Scores, reliability, score

type FloatArray = npt.NDArray[np.float64]

TOP_COEFFICIENTS = 10
CALIBRATION_MONTHS = 6
FEATURE_SETS = (("applicant", False), ("applicant + lender pricing", True))


@dataclass(frozen=True, slots=True)
class Evaluation:
    """One model's scores on both windows, and its reliability on the test window."""

    model: str
    train: Scores
    test: Scores
    test_reliability: tuple[ReliabilityBin, ...]


@dataclass(frozen=True, slots=True)
class Comparison:
    """Baseline, raw booster and calibrated booster over one feature set."""

    features: str
    baseline: FittedBaseline
    booster: FittedBooster
    evaluations: tuple[Evaluation, ...]


def parse_window(text: str) -> TimeWindow:
    """`YYYY-MM:YYYY-MM`, both issue months inclusive."""
    start, sep, end = text.partition(":")
    if not sep:
        raise argparse.ArgumentTypeError(f"expected YYYY-MM:YYYY-MM, got {text!r}")
    try:
        return TimeWindow(pd.Timestamp(f"{start}-01"), pd.Timestamp(f"{end}-01"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def evaluate(
    split: OutOfTimeSplit,
    *,
    calibration_months: int = CALIBRATION_MONTHS,
    method: Method = Method.PLATT,
) -> list[Comparison]:
    fitting, calibration = hold_out_latest(
        split.train, split.train_window, months=calibration_months
    )
    # Brier skill reference for every model: the whole training window's rate.
    rate = float(split.train[TARGET].mean())

    def evaluation(model: str, predict: Callable[[pd.DataFrame], FloatArray]) -> Evaluation:
        test_pd = predict(split.test)
        return Evaluation(
            model=model,
            train=score(split.train[TARGET], predict(split.train), reference_rate=rate),
            test=score(split.test[TARGET], test_pd, reference_rate=rate),
            test_reliability=reliability(split.test[TARGET], test_pd),
        )

    results = []
    for name, pricing in FEATURE_SETS:
        features = feature_names(include_lender_pricing=pricing)
        baseline = fit_baseline(split.train, features)
        booster = fit_booster(fitting, calibration, features, method=method)
        results.append(
            Comparison(
                features=name,
                baseline=baseline,
                booster=booster,
                evaluations=(
                    evaluation("logistic regression", baseline.predict_pd),
                    evaluation("gradient boosting, raw", booster.raw_pd),
                    evaluation(f"gradient boosting + {method}", booster.predict_pd),
                ),
            )
        )
    return results


def _window(window: TimeWindow) -> str:
    return f"{window.start:%Y-%m} to {window.end:%Y-%m}"


def render(split: OutOfTimeSplit, results: list[Comparison], *, calibration_months: int) -> str:
    lines = [
        "# Out-of-time model comparison",
        "",
        f"- Train: issued {_window(split.train_window)}, {len(split.train):,} loans",
        f"- Test (out of time): issued {_window(split.test_window)}, {len(split.test):,} loans",
        f"- Gradient boosting: trees fitted on all but the last {calibration_months} "
        "training months, calibrator fitted on those months",
        "",
        "Brier skill is measured against predicting the training default rate for every loan.",
        "Calibration error is the loan-weighted mean |mean PD - default rate| over ten "
        "equal-count bins.",
        "",
        "| features | model | window | loans | default rate | mean PD | AUC | Gini | KS "
        "| Brier | Brier skill | log loss | calibration error |",
        "|" + " --- |" * 13,
    ]
    for result in results:
        for e in result.evaluations:
            for window, s in (("train", e.train), ("test", e.test)):
                lines.append(
                    f"| {result.features} | {e.model} | {window} | {s.loans:,} "
                    f"| {s.default_rate:.2%} | {s.mean_pd:.2%} | {s.auc:.4f} | {s.gini:.4f} "
                    f"| {s.ks:.4f} | {s.brier:.4f} | {s.brier_skill:.2%} | {s.log_loss:.4f} "
                    f"| {s.calibration_error:.2%} |"
                )
    applicant = results[0]
    lines += [
        "",
        f"## Reliability on the test window — {applicant.features}",
        "",
        "Test loans in ten equal-count bins of each model's own PD, lowest first: "
        "mean PD / observed default rate.",
        "",
        "| bin | " + " | ".join(e.model for e in applicant.evaluations) + " |",
        "|" + " --- |" * (1 + len(applicant.evaluations)),
    ]
    for i, bins in enumerate(
        zip(*(e.test_reliability for e in applicant.evaluations), strict=True), 1
    ):
        cells = " | ".join(f"{b.mean_pd:.2%} / {b.default_rate:.2%}" for b in bins)
        lines.append(f"| {i} | {cells} |")
    for result in results:
        top = result.baseline.coefficients().head(TOP_COEFFICIENTS)
        lines += [
            "",
            f"## Largest baseline coefficients — {result.features}",
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
    parser.add_argument(
        "--calibration-months",
        type=int,
        default=CALIBRATION_MONTHS,
        help="latest training months held out to calibrate the booster",
    )
    parser.add_argument("--calibration", type=Method, choices=list(Method), default=Method.PLATT)
    parser.add_argument("--out", type=Path, default=Path("data/processed/model_report.md"))
    args = parser.parse_args(argv)

    split = out_of_time_split(pd.read_parquet(args.frame), train=args.train, test=args.test)
    results = evaluate(split, calibration_months=args.calibration_months, method=args.calibration)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(split, results, calibration_months=args.calibration_months))
    for result in results:
        for e in result.evaluations:
            print(
                f"{result.features} | {e.model}: test AUC {e.test.auc:.4f}, "
                f"calibration error {e.test.calibration_error:.2%}"
            )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
