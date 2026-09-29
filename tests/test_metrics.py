import numpy as np
import pytest

from credit_risk.models.metrics import MetricError, ks_statistic, score


def test_perfect_ranking_scores_one_and_constant_scores_are_uninformative() -> None:
    y = [0, 0, 1, 1]
    perfect = score(y, [0.1, 0.2, 0.8, 0.9], reference_rate=0.5)
    assert (perfect.auc, perfect.gini, perfect.ks) == (1.0, 1.0, 1.0)

    flat = score(y, [0.3] * 4, reference_rate=0.5)
    assert (flat.auc, flat.gini, flat.ks) == (0.5, 0.0, 0.0)


def test_ks_is_the_largest_gap_between_the_class_distributions() -> None:
    # Cut between 0.4 and 0.6 catches 2 of 3 defaulters and 1 of 3 goods: 2/3 - 1/3.
    y = [0, 1, 0, 1, 0, 1]
    assert ks_statistic(y, [0.1, 0.2, 0.3, 0.6, 0.7, 0.8]) == pytest.approx(1 / 3)


def test_brier_skill_is_relative_to_the_training_rate_not_the_test_rate() -> None:
    y = np.array([0] * 8 + [1] * 2)
    # Predicting the test rate itself has zero skill against the test rate...
    assert score(y, [0.2] * 10, reference_rate=0.2).brier_skill == pytest.approx(0.0)
    # ...but a model that tracked a base-rate shift gets credit against a stale
    # training rate of 5%, which is what a deployed model would have known.
    shifted = score(y, [0.2] * 10, reference_rate=0.05)
    assert shifted.brier == pytest.approx(0.16)
    assert shifted.brier_skill > 0


def test_ranking_and_calibration_are_reported_separately() -> None:
    y = [0, 0, 1, 1]
    # Same ordering, probabilities ten times too high: identical ranking, worse Brier.
    good = score(y, [0.01, 0.02, 0.08, 0.09], reference_rate=0.5)
    inflated = score(y, [0.1, 0.2, 0.8, 0.9], reference_rate=0.5)
    assert good.auc == inflated.auc
    assert good.brier != inflated.brier


@pytest.mark.parametrize(
    ("y", "p", "message"),
    [
        ([0, 1], [0.5], "equal-length"),
        ([0, 2], [0.1, 0.9], "0 or 1"),
        ([1, 1], [0.1, 0.9], "single class"),
        ([0, 1], [0.1, 1.2], r"\[0, 1\]"),
        ([0, 1], [0.1, np.nan], r"\[0, 1\]"),
    ],
)
def test_inputs_that_cannot_produce_a_metric_are_refused(
    y: list[int], p: list[float], message: str
) -> None:
    with pytest.raises(MetricError, match=message):
        score(y, p, reference_rate=0.2)


def test_reference_rate_must_be_a_probability_strictly_inside_zero_one() -> None:
    with pytest.raises(MetricError, match="reference"):
        score([0, 1], [0.2, 0.8], reference_rate=0.0)
