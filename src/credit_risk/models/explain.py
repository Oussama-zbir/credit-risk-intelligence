"""Why a model scored a loan the way it did: exact contributions and reason codes.

For either model, `explain` returns one contribution per business feature in
the model's log-odds: for every loan, the base value plus its contributions
equals the model's own log-odds, and `explain` checks that identity against
`decision_function` before returning anything. Log-odds rather than PD because
that is the scale on which contributions add up. Reason codes are built from
the contributions the same way for both models.

**Logistic regression** is explained with exact training-centred additive
log-odds contributions; its log-odds are already a sum, so no attribution
method is needed. Each transformed input j contributes
`beta_j * (x_j - training_mean_j)`, and those are summed back to the business
feature they came from — a
numeric value and its missing-value indicator, a monetary amount after `log1p`
and scaling, every one-hot column of a categorical. The base value is
`intercept + beta . training_mean`, the log-odds of the average transformed
training input, so base plus contributions reconstructs `decision_function`
exactly. The means follow the pipeline's real transformed output, so an
unseen level that lands in the infrequent column is measured from that
column's training share, not assumed to be all zeros. Centring matters for
categoricals. Measured from
an all-zero encoding — no level at all, which no applicant has — a level would
be cited whenever its coefficient is positive, however common it is; measured
from the training means (the level frequencies), it is cited only when it
pushes this applicant above the training population.

**Gradient boosting** is explained with exact path-dependent TreeSHAP
(Lundberg et al., 2020), relative to the mean training log-odds.

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

from collections.abc import Sequence
from dataclasses import dataclass
from math import factorial
from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd

from credit_risk.models.baseline import FittedBaseline
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


def _sources(outputs: Sequence[str], baseline: FittedBaseline) -> list[str]:
    """The business feature behind each transformed input of the baseline."""
    categorical = set(baseline.categories)
    sources = []
    for output in outputs:
        owners = [
            feature
            for feature in baseline.features
            if output in (feature, f"missingindicator_{feature}")
            or (feature in categorical and output.startswith(f"{feature}_"))
        ]
        if not owners:
            raise ExplanationError(f"transformed input {output!r} has no source feature")
        sources.append(max(owners, key=len))  # `grade_A` is grade's, not a shorter name's
    return sources


def _linear(baseline: FittedBaseline, frame: pd.DataFrame) -> tuple[Explanation, FloatArray]:
    columns = baseline.pipeline.named_steps["columns"]
    regression = baseline.pipeline.named_steps["model"]
    names = tuple(columns.get_feature_names_out())
    if names != baseline.input_names:
        raise ExplanationError("training input means do not match the pipeline's inputs")
    raw = frame[list(baseline.features)]
    coef = regression.coef_[0]
    means = np.asarray(baseline.input_means, dtype=np.float64)
    terms = (np.asarray(columns.transform(raw), dtype=np.float64) - means) * coef
    sources = _sources(names, baseline)
    owner = np.zeros((len(sources), len(baseline.features)))
    owner[np.arange(len(sources)), [baseline.features.index(f) for f in sources]] = 1.0
    explanation = Explanation(
        base_value=float(regression.intercept_[0] + coef @ means),
        contributions=pd.DataFrame(terms @ owner, index=frame.index, columns=baseline.features),
    )
    return explanation, np.asarray(baseline.pipeline.decision_function(raw), dtype=np.float64)


def _tree_shap(booster: FittedBooster, frame: pd.DataFrame) -> tuple[Explanation, FloatArray]:
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
    return explanation, np.asarray(expected, dtype=np.float64)


def explain(model: FittedBaseline | FittedBooster, frame: pd.DataFrame) -> Explanation:
    """Exact contributions of every fitted feature to every loan in `frame`.

    Training-centred additive contributions for the logistic regression,
    TreeSHAP for the booster; either way the result is checked against the
    model's log-odds.
    """
    if isinstance(model, FittedBaseline):
        explanation, expected = _linear(model, frame)
    else:
        explanation, expected = _tree_shap(model, frame)
    gap = np.abs(explanation.log_odds.to_numpy() - expected)
    if gap.size and gap.max() > ADDITIVITY_TOLERANCE:
        raise ExplanationError(f"contributions miss the model's log-odds by up to {gap.max():.2e}")
    return explanation


def reason_codes(
    explanation: Explanation, frame: pd.DataFrame, *, top: int = REASONS
) -> list[tuple[Reason, ...]]:
    """Per loan, the features that raised its PD most, largest first.

    Only positive contributions qualify: a reason for a decline or a higher
    price must be something that pushed the applicant's risk up, never merely
    "less good than it could be". Values are shown as the applicant gave them
    (`frame`), not as the model encoded them. These are model reason codes —
    the adverse contributors in this model — not a compliant adverse-action
    notice, which needs legal review of wording and of which reasons may be
    cited.
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
