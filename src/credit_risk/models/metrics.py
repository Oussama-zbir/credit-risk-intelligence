"""Metrics for a probability-of-default model on a held-out window.

Two questions, answered separately because a model can pass one and fail the
other:

- **Ranking** — does the model order borrowers by risk? AUC, Gini (2*AUC - 1,
  the form credit teams quote) and KS (largest gap between the cumulative score
  distributions of defaulters and non-defaulters). All three ignore the scale
  of the scores, so they say nothing about whether a 10% PD means 10%.
- **Probabilities** — are the numbers usable for pricing, limits and expected
  loss? Brier score and log loss, plus the Brier skill score against a model
  that predicts the *training* default rate for everyone. The reference uses
  the training rate, not the test rate, because a model deployed on the test
  window could not have known the test rate either; a drift in the base rate
  therefore shows up as lost skill instead of being quietly absorbed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, roc_curve


class MetricError(ValueError):
    """The labels or predictions cannot produce a meaningful metric."""


@dataclass(frozen=True, slots=True)
class Scores:
    loans: int
    default_rate: float
    mean_pd: float
    auc: float
    gini: float
    ks: float
    brier: float
    brier_skill: float
    log_loss: float


def _checked(
    y_true: npt.ArrayLike, pd_hat: npt.ArrayLike
) -> tuple[npt.NDArray[np.int8], npt.NDArray[np.float64]]:
    y = np.asarray(y_true)
    p = np.asarray(pd_hat, dtype=np.float64)
    if y.shape != p.shape or y.ndim != 1:
        raise MetricError(f"labels {y.shape} and predictions {p.shape} must be equal-length 1-D")
    if not np.isin(y, (0, 1)).all():
        raise MetricError("labels must be 0 or 1")
    if np.unique(y).size < 2:
        raise MetricError(f"{y.size} loans and a single class: ranking metrics are undefined")
    if not (np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()):
        raise MetricError("predictions must be probabilities in [0, 1]")
    return y.astype(np.int8), p


def ks_statistic(y_true: npt.ArrayLike, pd_hat: npt.ArrayLike) -> float:
    """Kolmogorov-Smirnov: max over cut-offs of (defaulters caught - goods rejected)."""
    y, p = _checked(y_true, pd_hat)
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def score(y_true: npt.ArrayLike, pd_hat: npt.ArrayLike, *, reference_rate: float) -> Scores:
    """Ranking and probability metrics; `reference_rate` is the training default rate."""
    if not 0 < reference_rate < 1:
        raise MetricError(f"reference default rate must be in (0, 1), got {reference_rate}")
    y, p = _checked(y_true, pd_hat)
    auc = float(roc_auc_score(y, p))
    brier = float(brier_score_loss(y, p))
    reference_brier = float(np.mean((reference_rate - y) ** 2))
    return Scores(
        loans=int(y.size),
        default_rate=float(y.mean()),
        mean_pd=float(p.mean()),
        auc=auc,
        gini=2 * auc - 1,
        ks=ks_statistic(y, p),
        brier=brier,
        brier_skill=1 - brier / reference_brier,
        log_loss=float(log_loss(y, p, labels=[0, 1])),
    )
