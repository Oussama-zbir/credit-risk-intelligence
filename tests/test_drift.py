import numpy as np
import pandas as pd
import pytest

from credit_risk.models.drift import (
    MISSING,
    SHARE_FLOOR,
    UNSEEN,
    Band,
    Binning,
    DriftError,
    psi,
    shares,
    stability,
)


def test_numeric_bins_are_reference_deciles() -> None:
    reference = pd.Series(np.arange(1_000, dtype=float))
    binning = Binning.fit(reference)
    assert len(binning.edges) == 9
    reference_shares = shares(binning.assign(reference), binning.order)
    assert reference_shares.drop(MISSING).to_numpy() == pytest.approx(np.full(10, 0.1))


def test_tied_values_get_one_bin_per_distinct_cut() -> None:
    term = pd.Series([36.0] * 76 + [60.0] * 24)
    binning = Binning.fit(term)
    assert binning.edges == (36.0,)
    assert list(binning.assign(pd.Series([36.0, 60.0]))) == ["<= 36", "> 36"]


def test_a_constant_reference_still_bins_new_values() -> None:
    binning = Binning.fit(pd.Series([5.0] * 10))
    labels = binning.assign(pd.Series([5.0, 9.0, np.nan]))
    assert list(labels) == ["any value", "any value", MISSING]


def test_values_beyond_the_reference_range_land_in_the_outer_bins() -> None:
    binning = Binning.fit(pd.Series(np.arange(100, dtype=float)))
    labels = binning.assign(pd.Series([-1e9, 1e9]))
    assert list(labels) == [binning.order[0], binning.order[-2]]


def test_missing_values_are_their_own_bin() -> None:
    binning = Binning.fit(pd.Series([1.0, 2.0, 3.0, np.nan]))
    assert binning.assign(pd.Series([np.nan, 2.0])).iloc[0] == MISSING


def test_categorical_bins_are_reference_levels_plus_unseen() -> None:
    reference = pd.Series(pd.Categorical(["RENT", "OWN", "RENT"]))
    binning = Binning.fit(reference)
    assert binning.order == ("OWN", "RENT", UNSEEN, MISSING)
    labels = binning.assign(pd.Series(pd.Categorical(["OWN", "crypto", None])))
    assert list(labels) == ["OWN", UNSEEN, MISSING]


def test_a_reference_without_observed_values_is_refused() -> None:
    with pytest.raises(DriftError, match="no observed values"):
        Binning.fit(pd.Series([np.nan, np.nan], name="mort_acc"))


def test_psi_matches_a_hand_computed_value() -> None:
    expected = pd.Series([0.5, 0.5])
    actual = pd.Series([0.7, 0.3])
    by_hand = 0.2 * np.log(0.7 / 0.5) + (-0.2) * np.log(0.3 / 0.5)
    assert psi(expected, actual).sum() == pytest.approx(by_hand)


def test_psi_is_symmetric_and_zero_for_identical_shares() -> None:
    a, b = pd.Series([0.2, 0.3, 0.5]), pd.Series([0.4, 0.4, 0.2])
    assert psi(a, b).sum() == pytest.approx(psi(b, a).sum())
    assert psi(a, a).sum() == 0.0


def test_an_empty_bin_on_one_side_gives_a_finite_psi() -> None:
    terms = psi(pd.Series([0.5, 0.5, 0.0]), pd.Series([0.4, 0.4, 0.2]))
    assert np.isfinite(terms).all()
    assert terms.iloc[2] == pytest.approx((0.2 - SHARE_FLOOR) * np.log(0.2 / SHARE_FLOOR))


def test_same_distribution_is_stable_and_a_shifted_one_is_not() -> None:
    rng = np.random.default_rng(0)
    reference = pd.Series(rng.normal(700, 30, 20_000))
    same = stability("fico", reference, pd.Series(rng.normal(700, 30, 20_000)))
    moved = stability("fico", reference, pd.Series(rng.normal(670, 30, 20_000)))
    assert same.band is Band.STABLE
    assert moved.band is Band.SHIFTED
    assert moved.largest_move().startswith(moved.bins.index[0])


def test_bins_empty_on_both_sides_are_left_out() -> None:
    result = stability("x", pd.Series([1.0, 2.0, 3.0]), pd.Series([1.0, 3.0]))
    assert MISSING not in result.bins.index


@pytest.mark.parametrize(
    ("value", "band"),
    [(0.0, Band.STABLE), (0.0999, Band.STABLE), (0.10, Band.WATCH), (0.25, Band.SHIFTED)],
)
def test_bands_follow_the_rule_of_thumb(value: float, band: Band) -> None:
    assert Band.of(value) is band
