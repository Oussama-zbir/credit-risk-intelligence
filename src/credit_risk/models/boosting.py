"""Gradient boosting against the baseline, with its probabilities recalibrated.

scikit-learn's histogram gradient boosting rather than LightGBM or XGBoost: it
is the same algorithm family, already a dependency, and it covers everything
this model needs natively — missing values routed to whichever side of a split
fits them best (no imputation, so "not reported" stays informative), pandas
categoricals split on directly (no one-hot, and a category unseen at fit time is
scored as missing rather than failing), and monotone constraints.

The constraints are for defensibility, not accuracy. A tree ensemble left free
will find a pocket of the training window where a higher FICO score goes with
slightly more defaults, and a lender cannot explain a decline — or price a
loan — on a model that says so. Only relationships a credit officer would
sign off on without seeing the data are constrained; everything else is free.

The number of trees is chosen by early stopping on the held-out latest slice of
the training window — the same slice the calibrator is fitted on — rather than
on scikit-learn's default random validation split, which would let later
vintages steer the fit of earlier ones. Reusing the slice costs a little: the
calibrator sees scores from a model whose length was picked on it, so the
slice's own calibration error is slightly flattering. The test window is never
touched, so every reported number stays honest. The other hyperparameters are
fixed and conservative (shallow trees, large leaves, L2 on leaf values);
tuning them on a time-ordered validation slice is future work, and the
baseline comparison is what says whether it is needed.

No class weights: reweighting defaults would inflate every PD, and the point
of this model is PDs that can be priced on.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from credit_risk.data.features import CATEGORICAL_FEATURES
from credit_risk.data.target import TARGET
from credit_risk.models.calibration import Calibrator, Method, fit_calibrator

# +1: PD may only rise with the feature; -1: only fall.
MONOTONE: Final[dict[str, int]] = {
    "fico": -1,
    "dti": 1,
    "revol_util": 1,
    "inq_last_6mths": 1,
    "int_rate": 1,
}
MAX_TREES: Final = 1000


def model_frame(frame: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    # The booster finds categorical columns by dtype, so they must arrive as one.
    out = frame[list(features)].copy()
    for name in out.columns.intersection(list(CATEGORICAL_FEATURES)):
        out[name] = out[name].astype("category")
    return out


def _raw_pd(
    model: HistGradientBoostingClassifier, frame: pd.DataFrame, features: Sequence[str]
) -> npt.NDArray[np.float64]:
    proba: npt.NDArray[np.float64] = model.predict_proba(model_frame(frame, features))
    return proba[:, 1]


def build_booster(features: Sequence[str], *, seed: int = 0) -> HistGradientBoostingClassifier:
    """An unfitted booster over `features`, a subset of `feature_names(...)`."""
    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=MAX_TREES,
        max_leaf_nodes=15,
        min_samples_leaf=100,
        l2_regularization=1.0,
        early_stopping=True,
        scoring="loss",
        n_iter_no_change=20,
        categorical_features="from_dtype",
        monotonic_cst={name: sign for name, sign in MONOTONE.items() if name in features},
        random_state=seed,
    )


@dataclass(frozen=True, slots=True)
class FittedBooster:
    model: HistGradientBoostingClassifier
    features: tuple[str, ...]
    calibrator: Calibrator

    @property
    def trees(self) -> int:
        """Boosting rounds kept by early stopping."""
        return int(self.model.n_iter_)

    def raw_pd(self, frame: pd.DataFrame) -> npt.NDArray[np.float64]:
        """The booster's own probability, before calibration."""
        return _raw_pd(self.model, frame, self.features)

    def predict_pd(self, frame: pd.DataFrame) -> npt.NDArray[np.float64]:
        return self.calibrator.apply(self.raw_pd(frame))


def fit_booster(
    fitting: pd.DataFrame,
    calibration: pd.DataFrame,
    features: Sequence[str],
    *,
    method: Method = Method.PLATT,
    seed: int = 0,
) -> FittedBooster:
    """Fit the trees on `fitting`, then the calibrator on the later `calibration` slice.

    Tree growth stops once log loss on `calibration` has not improved for 20
    rounds.
    """
    model = build_booster(features, seed=seed)
    model.fit(
        model_frame(fitting, features),
        fitting[TARGET],
        X_val=model_frame(calibration, features),
        y_val=calibration[TARGET],
    )
    calibrator = fit_calibrator(_raw_pd(model, calibration, features), calibration[TARGET], method)
    return FittedBooster(model=model, features=tuple(features), calibrator=calibrator)
