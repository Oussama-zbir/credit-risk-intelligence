"""Recalibrating predicted PDs on a slice the model was not fitted on.

A gradient-boosted model is fitted to rank, and its raw scores are often
over-confident at the tails: shrinkage and early trees pull predictions
towards the extremes of the training data, and the model never sees the drift
between its training vintages and the next ones. A calibrator is a
one-dimensional map from raw score to PD, fitted on the held-out latest slice of
the training window (`split.hold_out_latest`).

Two methods, with different failure modes:

- **Platt** fits a logistic curve to the logit of the raw score: two
  parameters, so it needs few defaults to be stable, and it is strictly
  monotone, so the ranking (AUC, KS) is untouched. It can only correct a
  miscalibration shaped like a sigmoid.
- **Isotonic** fits any non-decreasing step function: it corrects any shape of
  miscalibration, but it needs many defaults per step, and flat steps merge
  loans the model had ranked apart, which can cost a little AUC.

Both are fitted with labels from the calibration slice only; neither ever sees
the test window.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

import numpy as np
import numpy.typing as npt
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

# A PD of exactly 0 or 1 is not a credible price; isotonic steps can produce both.
PD_FLOOR: Final = 1e-4
# Only keeps the logit finite; clipping raw scores at PD_FLOOR would tie them.
_LOGIT_EPS: Final = 1e-12


class CalibrationError(ValueError):
    """The calibration slice cannot produce a usable calibrator."""


class Method(StrEnum):
    PLATT = "platt"
    ISOTONIC = "isotonic"


def _logit(p: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    p = np.clip(p, _LOGIT_EPS, 1 - _LOGIT_EPS)
    return np.log(p / (1 - p)).reshape(-1, 1)


@dataclass(frozen=True, slots=True)
class Calibrator:
    method: Method
    mapping: LogisticRegression | IsotonicRegression

    def apply(self, raw: npt.ArrayLike) -> npt.NDArray[np.float64]:
        p = np.asarray(raw, dtype=np.float64)
        if isinstance(self.mapping, LogisticRegression):
            calibrated = self.mapping.predict_proba(_logit(p))[:, 1]
        else:
            calibrated = self.mapping.predict(p)
        return np.clip(np.asarray(calibrated, dtype=np.float64), PD_FLOOR, 1 - PD_FLOOR)


def fit_calibrator(raw: npt.ArrayLike, y_true: npt.ArrayLike, method: Method) -> Calibrator:
    """Map raw scores to PDs using labelled loans the model was not fitted on."""
    p = np.asarray(raw, dtype=np.float64)
    y = np.asarray(y_true)
    if p.shape != y.shape or p.ndim != 1:
        raise CalibrationError(f"scores {p.shape} and labels {y.shape} must be equal-length 1-D")
    if np.unique(y).size < 2:
        raise CalibrationError(f"{y.size} loans and a single class: nothing to calibrate against")

    mapping: LogisticRegression | IsotonicRegression
    if method is Method.PLATT:
        mapping = LogisticRegression(C=np.inf).fit(_logit(p), y)
        if mapping.coef_[0, 0] <= 0:
            # Higher raw risk would map to a lower PD: the model ranks backwards
            # on this slice, and calibrating would hide it rather than fix it.
            raise CalibrationError("raw scores are not positively related to default")
    else:
        mapping = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(p, y)
    return Calibrator(method=method, mapping=mapping)
