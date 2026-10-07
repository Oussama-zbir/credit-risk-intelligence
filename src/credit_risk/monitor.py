"""Measure how far a later window has drifted from a saved model's training window.

    python -m credit_risk.monitor artifacts/699a62529632 data/processed/loans.parquet \\
        --window 2013-01:2013-12

The reference is the artifact's own training window, read from the same
processed file. The manifest pins that file's SHA-256, and a different file is
refused: a stability index against data the model was not trained on measures
nothing about the model.

The report answers three questions, in order:

- **Did the score move?** PSI of the PD on the reference PD deciles, for the
  whole window and per quarter, beside mean PD and, where outcomes are known,
  the observed default rate.
- **Which inputs moved?** PSI of every fitted feature (`models/drift.py`).
- **Which moves pushed the score?** A feature can shift a lot and barely move
  the score, or shift a little on a heavily weighted input. The model's own
  explanation is additive in log-odds, so the change in a feature's mean
  contribution between the two windows is exactly its share of the change in
  mean log-odds; the shares sum to the total. This is a scorecard
  "characteristic analysis", done with the model's exact contributions.

Outcomes are reported because the processed file holds matured loans only. A
live monitor would not have them: PSI is the early signal, the default rate the
late confirmation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pandas as pd

from credit_risk.data.split import TimeWindow
from credit_risk.data.target import ISSUE_DATE, TARGET
from credit_risk.models.artifact import ArtifactError, ScoringModel, load_artifact
from credit_risk.models.drift import (
    SHIFT_FROM,
    WATCH_FROM,
    Binning,
    DriftError,
    Stability,
    stability,
)
from credit_risk.prepare import sha256
from credit_risk.train import parse_window

# Mean contributions are estimated on a sample of each window; on the real
# file the standard error at this size is under 0.001 log-odds per feature.
EXPLAINED_LOANS: Final = 50_000
SCORE: Final = "PD"


@dataclass(frozen=True, slots=True)
class Period:
    """One quarter of the monitored window."""

    label: str
    loans: int
    score_psi: float
    mean_pd: float
    default_rate: float | None


@dataclass(frozen=True, slots=True)
class DriftReport:
    model: ScoringModel
    reference_loans: int
    monitored_window: TimeWindow
    monitored_loans: int
    score: Stability
    periods: tuple[Period, ...]
    features: tuple[Stability, ...]  # largest PSI first
    contributions: pd.DataFrame  # index: feature; columns: reference, monitored, shift
    unseen: dict[str, dict[str, int]]  # feature -> unseen level -> loans


def _window(manifest_window: tuple[str, str]) -> TimeWindow:
    start, end = manifest_window
    return TimeWindow(pd.Timestamp(f"{start}-01"), pd.Timestamp(f"{end}-01"))


def reference_window(model: ScoringModel) -> TimeWindow:
    return _window(model.manifest.provenance.train_window)


def _sample(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sample(min(EXPLAINED_LOANS, len(frame)), random_state=0)


def _mean_contributions(model: ScoringModel, frame: pd.DataFrame) -> pd.Series:
    return model.explain(_sample(frame)).contributions.mean()


def measure(model: ScoringModel, reference: pd.DataFrame, monitored: pd.DataFrame) -> DriftReport:
    """Compare `monitored` loans with the model's `reference` (training) loans."""
    if reference.empty or monitored.empty:
        raise DriftError(
            f"need loans in both windows: {len(reference)} reference, {len(monitored)} monitored"
        )
    if monitored[ISSUE_DATE].min() <= reference[ISSUE_DATE].max():
        raise DriftError("monitored loans must all be issued after the reference window")
    window = TimeWindow(monitored[ISSUE_DATE].min(), monitored[ISSUE_DATE].max())

    reference_pd = pd.Series(model.predict_pd(reference), index=reference.index)
    monitored_pd = pd.Series(model.predict_pd(monitored), index=monitored.index)
    score_bins = Binning.fit(reference_pd)
    periods = []
    for quarter, rows in monitored.groupby(monitored[ISSUE_DATE].dt.to_period("Q")):
        pd_hat = monitored_pd.loc[rows.index]
        outcomes = rows[TARGET].dropna() if TARGET in rows else pd.Series(dtype=float)
        periods.append(
            Period(
                label=str(quarter),
                loans=len(rows),
                score_psi=stability(SCORE, reference_pd, pd_hat, binning=score_bins).psi,
                mean_pd=float(pd_hat.mean()),
                default_rate=float(outcomes.mean()) if len(outcomes) else None,
            )
        )

    features = sorted(
        (stability(name, reference[name], monitored[name]) for name in model.model.features),
        key=lambda s: s.psi,
        reverse=True,
    )
    contributions = pd.DataFrame(
        {
            "reference": _mean_contributions(model, reference),
            "monitored": _mean_contributions(model, monitored),
        }
    )
    contributions["shift"] = contributions["monitored"] - contributions["reference"]
    contributions = contributions.reindex(
        contributions["shift"].abs().sort_values(ascending=False).index
    )

    unseen: dict[str, dict[str, int]] = {}
    for name, levels in model.unseen_categories(monitored).items():
        counts = monitored[name].astype(str).value_counts()
        unseen[name] = {level: int(counts[level]) for level in levels}

    return DriftReport(
        model=model,
        reference_loans=len(reference),
        monitored_window=window,
        monitored_loans=len(monitored),
        score=stability(SCORE, reference_pd, monitored_pd, binning=score_bins),
        periods=tuple(periods),
        features=tuple(features),
        contributions=contributions,
        unseen=unseen,
    )


def _months(window: TimeWindow) -> str:
    return f"{window.start:%Y-%m} to {window.end:%Y-%m}"


def render(report: DriftReport) -> str:
    manifest = report.model.manifest
    score = report.score
    lines = [
        f"# Drift report — model {manifest.model_id} ({manifest.family})",
        "",
        f"- Reference (training window): issued "
        f"{_months(_window(manifest.provenance.train_window))}, {report.reference_loans:,} loans",
        f"- Monitored: issued {_months(report.monitored_window)}, {report.monitored_loans:,} loans",
        "",
        f"PSI bands (rule of thumb): under {WATCH_FROM:.2f} stable, {WATCH_FROM:.2f} to "
        f"{SHIFT_FROM:.2f} watch, {SHIFT_FROM:.2f} and above shifted. Bins are fitted on the "
        "reference window: deciles for numbers, one per level for categories, missing and "
        "unseen kept apart.",
        "",
        "## Score",
        "",
        f"PSI of the PD on the reference PD deciles: **{score.psi:.4f} ({score.band})**.",
        "",
        "| quarter | loans | score PSI | mean PD | default rate |",
        "| --- | --- | --- | --- | --- |",
    ]
    for p in report.periods:
        observed = "not yet known" if p.default_rate is None else f"{p.default_rate:.2%}"
        lines.append(
            f"| {p.label} | {p.loans:,} | {p.score_psi:.4f} | {p.mean_pd:.2%} | {observed} |"
        )
    lines += [
        "",
        "| PD bin | reference | monitored | PSI term |",
        "| --- | --- | --- | --- |",
        *(
            f"| {label} | {row.reference:.2%} | {row.actual:.2%} | {row.psi:.4f} |"
            for label, row in score.bins.iterrows()
        ),
        "",
        "## Feature stability",
        "",
        "| feature | PSI | band | largest move |",
        "| --- | --- | --- | --- |",
        *(f"| {s.name} | {s.psi:.4f} | {s.band} | {s.largest_move()} |" for s in report.features),
    ]
    c = report.contributions
    shift = float(c["shift"].sum())
    lines += [
        "",
        "## What moved the score",
        "",
        "Mean contribution of each feature to the model's log-odds in each window "
        f"(on up to {EXPLAINED_LOANS:,} loans per window). The shifts sum to the change in "
        f"mean log-odds: **{shift:+.4f}**.",
        "",
        "| feature | reference | monitored | shift | share of shift |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, row in c.iterrows():
        share = f"{row['shift'] / shift:.0%}" if shift else "-"
        lines.append(
            f"| {name} | {row['reference']:+.4f} | {row['monitored']:+.4f} "
            f"| {row['shift']:+.4f} | {share} |"
        )
    lines += ["", "## Unseen categories", ""]
    if report.unseen:
        lines += ["| feature | level | loans |", "| --- | --- | --- |"]
        for name, levels in report.unseen.items():
            lines += [f"| {name} | {level} | {n:,} |" for level, n in levels.items()]
    else:
        lines.append("None: every categorical value was seen in training.")
    return "\n".join(lines) + "\n"


def run(artifact: Path, frame_path: Path, window: TimeWindow) -> DriftReport:
    model = load_artifact(artifact)
    pinned = model.manifest.provenance.data_sha256
    if sha256(frame_path) != pinned:
        raise DriftError(
            f"{frame_path} is not the file model {model.manifest.model_id} was trained on "
            f"(SHA-256 {pinned[:12]}...)"
        )
    frame = pd.read_parquet(frame_path)
    reference = reference_window(model)
    if window.start <= reference.end:
        raise DriftError(
            f"monitored window must start after the training window ends ({reference.end:%Y-%m})"
        )
    monitored = frame.loc[window.contains(frame[ISSUE_DATE])]
    return measure(model, frame.loc[reference.contains(frame[ISSUE_DATE])], monitored)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("artifact", type=Path, help="artifact directory written by train")
    parser.add_argument("frame", type=Path, help="the loans.parquet the artifact was trained on")
    parser.add_argument("--window", type=parse_window, required=True, metavar="YYYY-MM:YYYY-MM")
    parser.add_argument("--out", type=Path, default=Path("data/processed/drift_report.md"))
    args = parser.parse_args(argv)
    try:
        report = run(args.artifact, args.frame, args.window)
    except (ArtifactError, DriftError) as exc:
        parser.exit(2, f"error: {exc}\n")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(report))
    print(f"score PSI {report.score.psi:.4f} ({report.score.band})")
    for s in report.features[:3]:
        print(f"{s.name}: PSI {s.psi:.4f} ({s.band})")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
