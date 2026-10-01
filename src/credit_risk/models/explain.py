"""Why the booster scored a loan the way it did: TreeSHAP and reason codes.

Contributions are exact path-dependent TreeSHAP values (Lundberg et al., 2020)
in the booster's log-odds: for every loan, the base value plus its feature
contributions equals the model's raw log-odds, and `explain` checks that
identity against `decision_function` before returning anything. Log-odds
rather than PD because that is the scale on which contributions add up.

Calibration does not change the story. Platt is affine in log-odds with a
positive slope (the calibrator refuses a negative one), so it rescales every
contribution by the same factor: the ranking and sign of reason codes are the
same before and after it. Isotonic is monotone but not affine, so under it the
contributions still rank reasons correctly but no longer sum to the calibrated
PD.

Implemented here over the fitted trees rather than through the `shap` package:
it would bring a compiled numba/llvmlite stack into a project that otherwise
needs only scikit-learn, for one algorithm that fits on a page. The price is
reading scikit-learn's private tree arrays (`_predictors`, `_preprocessor`),
which is why the additivity check is not optional: if those internals change,
`explain` raises instead of returning wrong reasons.

The algorithm works leaf by leaf. Under path-dependent TreeSHAP a leaf's share
of E[f(x) | x_S] is its value times, for each feature on its path, either 1/0
(does the loan satisfy every split on that feature?) when the feature is in S,
or the fraction of training loans that took those branches when it is not. A
product game like that has a closed-form Shapley value, computed here for every
distinct satisfy/not-satisfy pattern of a leaf at once, never per loan.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import factorial
from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd

from credit_risk.models.boosting import FittedBooster, model_frame

type FloatArray = npt.NDArray[np.float64]
type BoolArray = npt.NDArray[np.bool_]

# The identity holds to float rounding; anything larger means the trees were misread.
ADDITIVITY_TOLERANCE: Final = 1e-6
REASONS: Final = 4

# Wording for a reason a credit officer would read; unlisted features show their name.
LABELS: Final[dict[str, str]] = {
    "fico": "FICO score",
    "dti": "debt-to-income ratio",
    "revol_util": "revolving credit utilisation",
    "revol_bal": "revolving balance",
    "inq_last_6mths": "credit inquiries in the last 6 months",
    "delinq_2yrs": "delinquencies in the last 2 years",
    "pub_rec": "public records",
    "pub_rec_bankruptcies": "bankruptcies on record",
    "annual_inc": "annual income",
    "loan_amnt": "loan amount",
    "term_months": "loan term",
    "emp_length_years": "length of employment",
    "credit_history_months": "length of credit history",
    "open_acc": "open credit lines",
    "total_acc": "total credit lines",
    "mort_acc": "mortgage accounts",
    "home_ownership": "home ownership",
    "purpose": "loan purpose",
    "verification_status": "income verification",
    "application_type": "application type",
    "int_rate": "interest rate",
    "installment": "monthly instalment",
    "grade": "lender grade",
    "sub_grade": "lender sub-grade",
}


class ExplanationError(RuntimeError):
    """Contributions do not reproduce the model's own scores."""


@dataclass(frozen=True, slots=True)
class Explanation:
    """Per-loan feature contributions, in log-odds of default."""

    base_value: float
    contributions: pd.DataFrame  # loans x features, same index as the explained frame

    @property
    def log_odds(self) -> pd.Series:
        return self.base_value + self.contributions.sum(axis=1)

    def importance(self) -> pd.Series:
        """Mean absolute contribution per feature, largest first."""
        return self.contributions.abs().mean().sort_values(ascending=False)


@dataclass(frozen=True, slots=True)
class Reason:
    feature: str
    value: object
    contribution: float

    def __str__(self) -> str:
        if pd.isna(self.value):  # type: ignore[call-overload]
            shown = "not reported"
        elif isinstance(self.value, float | int):
            shown = f"{self.value:,.2f}".rstrip("0").rstrip(".")
        else:
            shown = str(self.value)
        return f"{LABELS.get(self.feature, self.feature)} ({shown})"


def _weights(d: int) -> FloatArray:
    # Shapley weight of a coalition of size s out of the other d - 1 players.
    return np.array([factorial(s) * factorial(d - 1 - s) / factorial(d) for s in range(d)])


def _encoded(booster: FittedBooster, frame: pd.DataFrame) -> tuple[FloatArray, list[int]]:
    """The matrix the trees split on, and the input column behind each of its columns."""
    model = booster.model
    data = model_frame(frame, booster.features)
    if model._preprocessor is None:
        return data.to_numpy(dtype=np.float64), list(range(len(booster.features)))
    # Categoricals are ordinal-encoded (unknown -> NaN) and moved to the front.
    categorical = np.asarray(model.is_categorical_, dtype=bool)
    order = [*np.flatnonzero(categorical), *np.flatnonzero(~categorical)]
    encoded = np.asarray(model._preprocessor.transform(data), dtype=np.float64)
    return encoded, [int(i) for i in order]


def _goes_left(column: FloatArray, node: np.void, bitsets: npt.NDArray[np.uint32]) -> BoolArray:
    """Route every loan through one split exactly as the fitted predictor does."""
    missing = np.isnan(column)
    if node["is_categorical"]:
        codes = np.where(missing, 0, column).astype(np.int64)
        words = bitsets[node["bitset_idx"]]
        decided = ((words[codes // 32] >> (codes % 32).astype(np.uint32)) & 1).astype(bool)
    else:
        decided = column <= node["num_threshold"]
    return np.where(missing, bool(node["missing_go_to_left"]), decided)


def _explain_tree(
    nodes: npt.NDArray[np.void],
    bitsets: npt.NDArray[np.uint32],
    x: FloatArray,
    phi: FloatArray,
) -> float:
    """Add one tree's contributions to `phi`; return its expected value over training loans."""
    left: dict[int, BoolArray] = {}
    expected = 0.0
    # (node, conditions so far: feature -> (cover fraction, loans satisfying every split))
    stack: list[tuple[int, dict[int, tuple[float, BoolArray]]]] = [(0, {})]
    while stack:
        index, path = stack.pop()
        node = nodes[index]
        if node["is_leaf"]:
            value = float(node["value"])
            expected += value * float(np.prod([z for z, _ in path.values()]))
            _add_leaf(value, path, phi)
            continue
        feature = int(node["feature_idx"])
        if index not in left:
            left[index] = _goes_left(x[:, feature], node, bitsets)
        for child, satisfied in (
            (int(node["left"]), left[index]),
            (int(node["right"]), ~left[index]),
        ):
            cover = nodes[child]["count"] / node["count"]
            z, o = path.get(feature, (1.0, np.ones(len(x), dtype=bool)))
            stack.append((child, {**path, feature: (z * cover, o & satisfied)}))
    return expected


def _add_leaf(value: float, path: dict[int, tuple[float, BoolArray]], phi: FloatArray) -> None:
    if not path:
        return
    features = list(path)
    z = np.array([path[f][0] for f in features])
    bits = 1 << np.arange(len(features), dtype=np.int64)
    pattern = sum(np.where(path[f][1], bit, 0) for f, bit in zip(features, bits, strict=True))
    # Loans sharing a satisfy/not-satisfy pattern share contributions: solve per pattern.
    codes, inverse = np.unique(pattern, return_inverse=True)
    o = ((codes[:, None] & bits) > 0).astype(np.float64)
    weights = _weights(len(features))
    for i, feature in enumerate(features):
        # Coefficient s: sum over coalitions of s other features of prod(o in S) prod(z out).
        poly = np.ones((len(o), 1))
        for j in range(len(features)):
            if j != i:
                grown = np.zeros((len(o), poly.shape[1] + 1))
                grown[:, :-1] = poly * z[j]  # feature j left out of the coalition
                grown[:, 1:] += poly * o[:, [j]]  # feature j in it
                poly = grown
        share = value * (o[:, i] - z[i]) * (poly @ weights)
        phi[:, feature] += share[inverse.reshape(-1)]


def explain(booster: FittedBooster, frame: pd.DataFrame) -> Explanation:
    """Exact TreeSHAP contributions of every feature to every loan in `frame`."""
    model = booster.model
    x, inputs = _encoded(booster, frame)
    phi = np.zeros_like(x)
    base = float(np.asarray(model._baseline_prediction).ravel()[0])
    for (predictor,) in model._predictors:
        base += _explain_tree(predictor.nodes, predictor.raw_left_cat_bitsets, x, phi)

    contributions = np.zeros((len(frame), len(booster.features)))
    contributions[:, inputs] = phi
    explanation = Explanation(
        base_value=base,
        contributions=pd.DataFrame(contributions, index=frame.index, columns=booster.features),
    )
    expected = model.decision_function(model_frame(frame, booster.features))
    gap = np.abs(explanation.log_odds.to_numpy() - expected)
    if gap.size and gap.max() > ADDITIVITY_TOLERANCE:
        raise ExplanationError(f"contributions miss the model's log-odds by up to {gap.max():.2e}")
    return explanation


def reason_codes(
    explanation: Explanation, frame: pd.DataFrame, *, top: int = REASONS
) -> list[tuple[Reason, ...]]:
    """Per loan, the features that raised its PD most, largest first.

    Only positive contributions qualify: a reason for a decline or a higher
    price must be something that made the applicant look riskier than the
    average training loan, never merely "less good than it could be".
    """
    reasons = []
    for index, row in explanation.contributions.iterrows():
        raised = row[row > 0].sort_values(ascending=False).head(top)
        reasons.append(
            tuple(
                Reason(feature=str(f), value=frame.at[index, f], contribution=float(c))
                for f, c in raised.items()
            )
        )
    return reasons
