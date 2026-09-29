"""Logistic-regression baseline: the model every later one has to beat.

A regularised logistic regression over standardised features is what a
credit-risk team would ship first and what a regulator can read: one
coefficient per input, monotone in each feature, cheap to score. Gradient
boosting is only worth its opacity if it beats this on the out-of-time window.

Everything the pipeline learns — medians for imputation, scaling moments,
which categories are frequent enough to get their own column — is fitted on the
training window inside one scikit-learn `Pipeline`, so the test window is only
ever transformed, never learned from.

Preprocessing choices, each deliberately simple:

- Monetary amounts are heavy-tailed (a few incomes in the millions); they enter
  as `log1p` so a single applicant cannot dominate the scaler.
- Missing numerics are imputed with the training median *and* flagged, so the
  model can learn that "no revolving utilisation reported" carries risk of its
  own instead of pretending the value was typical.
- Categories rarer than 0.5% of training loans share one "infrequent" column,
  and a category first seen in the test window lands there too rather than
  failing the scoring run.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from credit_risk.data.features import CATEGORICAL_FEATURES
from credit_risk.data.target import TARGET

MONETARY: Final = frozenset({"loan_amnt", "annual_inc", "revol_bal", "installment"})
MIN_CATEGORY_SHARE: Final = 0.005
MISSING_CATEGORY: Final = "MISSING"


def _log1p(values: pd.DataFrame) -> pd.DataFrame:
    return values.clip(lower=0).transform(np.log1p)


def _as_object(values: pd.DataFrame) -> pd.DataFrame:
    # Parquet round-trips categoricals; the imputer wants plain objects with NaN.
    return values.astype(object).where(values.notna(), np.nan)


def build_baseline(features: Sequence[str], *, c: float = 1.0) -> Pipeline:
    """An unfitted pipeline over `features`, a subset of `feature_names(...)`."""
    categorical = [f for f in features if f in CATEGORICAL_FEATURES]
    monetary = [f for f in features if f in MONETARY]
    numeric = [f for f in features if f not in CATEGORICAL_FEATURES and f not in MONETARY]

    def numeric_steps(*first: FunctionTransformer) -> Pipeline:
        return make_pipeline(
            *first,
            SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
            StandardScaler(),
        )

    columns = ColumnTransformer(
        [
            ("numeric", numeric_steps(), numeric),
            (
                "monetary",
                numeric_steps(FunctionTransformer(_log1p, feature_names_out="one-to-one")),
                monetary,
            ),
            (
                "categorical",
                make_pipeline(
                    FunctionTransformer(_as_object, feature_names_out="one-to-one"),
                    SimpleImputer(strategy="constant", fill_value=MISSING_CATEGORY),
                    OneHotEncoder(
                        handle_unknown="infrequent_if_exist",
                        min_frequency=MIN_CATEGORY_SHARE,
                        sparse_output=False,
                    ),
                ),
                categorical,
            ),
        ],
        verbose_feature_names_out=False,
    )
    model = LogisticRegression(C=c, max_iter=2000)
    return Pipeline([("columns", columns), ("model", model)])


@dataclass(frozen=True, slots=True)
class FittedBaseline:
    pipeline: Pipeline
    features: tuple[str, ...]
    train_default_rate: float

    def predict_pd(self, frame: pd.DataFrame) -> npt.NDArray[np.float64]:
        proba: npt.NDArray[np.float64] = self.pipeline.predict_proba(frame[list(self.features)])
        return proba[:, 1]

    def coefficients(self) -> pd.Series:
        """Coefficient per transformed input, largest absolute effect first.

        Inputs are standardised, so magnitudes are comparable across numeric
        features: log-odds change per standard deviation.
        """
        names = self.pipeline.named_steps["columns"].get_feature_names_out()
        weights = self.pipeline.named_steps["model"].coef_[0]
        coef = pd.Series(weights, index=names, name="coefficient")
        return coef.reindex(coef.abs().sort_values(ascending=False).index)


def fit_baseline(train: pd.DataFrame, features: Sequence[str], *, c: float = 1.0) -> FittedBaseline:
    """Fit on the training window only."""
    pipeline = build_baseline(features, c=c)
    pipeline.fit(train[list(features)], train[TARGET])
    return FittedBaseline(
        pipeline=pipeline,
        features=tuple(features),
        train_default_rate=float(train[TARGET].mean()),
    )
