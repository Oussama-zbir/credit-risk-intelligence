import numpy as np
import numpy.typing as npt
import pytest

from credit_risk.models.calibration import (
    PD_FLOOR,
    CalibrationError,
    Method,
    fit_calibrator,
)
from credit_risk.models.metrics import calibration_error, reliability


def overconfident(loans: int, seed: int) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int_]]:
    """Scores that rank correctly but push every PD too far towards 0 or 1."""
    rng = np.random.default_rng(seed)
    true_logit = rng.normal(-1.5, 1, loans)
    y = (rng.random(loans) < 1 / (1 + np.exp(-true_logit))).astype(int)
    raw = 1 / (1 + np.exp(-(2.5 * true_logit + 1)))
    return raw, y


@pytest.mark.parametrize("method", list(Method))
def test_calibrator_fitted_on_one_slice_repairs_scores_on_another(method: Method) -> None:
    raw_fit, y_fit = overconfident(20_000, seed=0)
    raw_new, y_new = overconfident(20_000, seed=1)
    calibrator = fit_calibrator(raw_fit, y_fit, method)

    before = calibration_error(reliability(y_new, raw_new))
    after = calibration_error(reliability(y_new, calibrator.apply(raw_new)))
    assert before > 0.05
    assert after < 0.015


def test_platt_keeps_the_ranking_isotonic_may_only_merge_it() -> None:
    raw, y = overconfident(5000, seed=0)
    order = np.argsort(raw)
    for method in Method:
        calibrated = fit_calibrator(raw, y, method).apply(raw)[order]
        assert (np.diff(calibrated) >= 0).all()
        if method is Method.PLATT:
            assert (np.diff(calibrated) > 0).all()


def test_calibrated_pds_are_never_exactly_zero_or_one() -> None:
    raw = np.array([0.01, 0.02, 0.9, 0.95])
    y = np.array([0, 0, 1, 1])
    for method in Method:
        pd_hat = fit_calibrator(raw, y, method).apply([0.0, 0.01, 0.99, 1.0])
        assert pd_hat.min() >= PD_FLOOR
        assert pd_hat.max() <= 1 - PD_FLOOR


def test_scores_that_rank_backwards_are_refused_not_flipped() -> None:
    raw, y = overconfident(2000, seed=0)
    with pytest.raises(CalibrationError, match="not positively related"):
        fit_calibrator(1 - raw, y, Method.PLATT)


@pytest.mark.parametrize(
    ("raw", "y", "match"),
    [
        ([0.1, 0.2], [0, 0], "single class"),
        ([0.1, 0.2, 0.3], [0, 1], "equal-length"),
    ],
)
def test_unusable_calibration_slices_are_refused(
    raw: list[float], y: list[int], match: str
) -> None:
    with pytest.raises(CalibrationError, match=match):
        fit_calibrator(raw, y, Method.PLATT)
