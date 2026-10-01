from itertools import combinations
from math import factorial

import numpy as np
import pandas as pd
import pytest

from conftest import make_processed
from credit_risk.data.features import feature_names
from credit_risk.data.split import TimeWindow, hold_out_latest
from credit_risk.data.target import ISSUE_DATE
from credit_risk.models import explain as explain_module
from credit_risk.models.boosting import FittedBooster, fit_booster, model_frame
from credit_risk.models.explain import ExplanationError, Reason, explain, reason_codes

WITH_PRICING = feature_names(include_lender_pricing=True)
WINDOW = TimeWindow(pd.Timestamp("2012-01-01"), pd.Timestamp("2014-12-01"))

type Slices = tuple[pd.DataFrame, pd.DataFrame]


@pytest.fixture(scope="module")
def slices() -> Slices:
    frame = make_processed(8000)
    return hold_out_latest(frame.loc[WINDOW.contains(frame[ISSUE_DATE])], WINDOW, months=6)


@pytest.fixture(scope="module")
def booster(slices: Slices) -> FittedBooster:
    return fit_booster(*slices, WITH_PRICING)


def test_contributions_add_up_to_the_model_log_odds_including_odd_inputs(
    slices: Slices, booster: FittedBooster
) -> None:
    loans = slices[1].head(200).copy()
    loans["purpose"] = loans["purpose"].astype(object)
    loans.loc[loans.index[:3], "purpose"] = ["wedding", None, "credit_card"]  # unseen, missing
    loans.loc[loans.index[3:6], ["dti", "fico", "revol_util"]] = np.nan
    result = explain(booster, loans)
    expected = booster.model.decision_function(model_frame(loans, booster.features))
    np.testing.assert_allclose(result.log_odds, expected, atol=1e-9)
    assert list(result.contributions.columns) == list(WITH_PRICING)
    assert result.contributions.index.equals(loans.index)


def test_base_value_is_the_mean_training_log_odds(slices: Slices, booster: FittedBooster) -> None:
    # Additivity holds for any cover fractions; this pins them to training counts.
    fitting = slices[0]
    mean = booster.model.decision_function(model_frame(fitting, booster.features)).mean()
    assert explain(booster, fitting.head(5)).base_value == pytest.approx(mean, abs=1e-9)


def _brute_force(booster: FittedBooster, row: pd.Series, features: list[str]) -> np.ndarray:
    """Shapley values by enumerating every coalition of a path-dependent value function."""

    def expected(nodes: np.ndarray, index: int, known: frozenset[int]) -> float:
        node = nodes[index]
        if node["is_leaf"]:
            return float(node["value"])
        f = int(node["feature_idx"])
        if f in known:
            v = float(row[features[f]])
            left = bool(node["missing_go_to_left"]) if np.isnan(v) else v <= node["num_threshold"]
            return expected(nodes, int(node["left"] if left else node["right"]), known)
        return sum(
            float(nodes[c]["count"] / node["count"]) * expected(nodes, c, known)
            for c in (int(node["left"]), int(node["right"]))
        )

    def value(known: frozenset[int]) -> float:
        return sum(expected(p.nodes, 0, known) for (p,) in booster.model._predictors)

    d = len(features)
    phi = np.zeros(d)
    for i in range(d):
        others = [j for j in range(d) if j != i]
        for size in range(d):
            weight = factorial(size) * factorial(d - size - 1) / factorial(d)
            for coalition in combinations(others, size):
                known = frozenset(coalition)
                phi[i] += weight * (value(known | {i}) - value(known))
    return phi


def test_matches_brute_force_shapley_values(slices: Slices) -> None:
    features = ["fico", "dti", "revol_util", "annual_inc"]  # numeric: plain threshold splits
    small = fit_booster(*slices, features)
    loans = slices[1].head(4).copy()
    loans.loc[loans.index[0], "revol_util"] = np.nan
    fast = explain(small, loans).contributions.to_numpy()
    for k, (_, row) in enumerate(loans.iterrows()):
        np.testing.assert_allclose(fast[k], _brute_force(small, row, features), atol=1e-9)


def test_constrained_contributions_move_in_the_constrained_direction(
    slices: Slices, booster: FittedBooster
) -> None:
    applicants = slices[1].head(20)
    for feature, grid, sign in (
        ("fico", np.arange(600, 850, 10), -1),
        ("dti", np.arange(0, 45, 2), 1),
    ):
        swept = applicants.loc[applicants.index.repeat(grid.size)].copy()
        swept[feature] = np.tile(grid, len(applicants))
        swept.index = pd.RangeIndex(len(swept))
        phi = explain(booster, swept).contributions[feature].to_numpy()
        steps = sign * np.diff(phi.reshape(len(applicants), grid.size), axis=1)
        assert (steps >= -1e-12).all(), feature


def test_global_importance_finds_the_simulated_drivers(slices: Slices) -> None:
    applicant = fit_booster(*slices, feature_names())
    top = explain(applicant, slices[1]).importance().head(2)
    assert set(top.index) == {"fico", "dti"}


def test_reason_codes_cite_only_what_raised_the_pd(slices: Slices, booster: FittedBooster) -> None:
    loans = slices[1].head(50).copy()
    risky = loans.index[0]
    loans.loc[risky, ["fico", "dti"]] = [620.0, 39.0]
    result = explain(booster, loans)
    codes = reason_codes(result, loans, top=3)
    assert len(codes) == len(loans)
    for reasons in codes:
        assert len(reasons) <= 3
        contributions = [r.contribution for r in reasons]
        assert all(c > 0 for c in contributions)
        assert contributions == sorted(contributions, reverse=True)
    cited = {r.feature for r in codes[0]}
    assert {"fico", "dti"} <= cited
    assert "FICO score (620)" in [str(r) for r in codes[0]]


def test_misread_trees_raise_instead_of_returning_wrong_reasons(
    slices: Slices, booster: FittedBooster, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(explain_module, "_weights", lambda d: np.full(d, 1.0 / d))
    with pytest.raises(ExplanationError, match="log-odds"):
        explain(booster, slices[1].head(50))


def test_reasons_read_like_a_notice_not_a_float_dump() -> None:
    assert str(Reason("dti", 36.83168340699365, 0.9)) == "debt-to-income ratio (36.83)"
    assert str(Reason("revol_bal", 29946.0, 0.1)) == "revolving balance (29,946)"
    assert str(Reason("revol_util", np.nan, 0.1)) == "revolving credit utilisation (not reported)"
    assert str(Reason("purpose", "wedding", 0.1)) == "loan purpose (wedding)"
