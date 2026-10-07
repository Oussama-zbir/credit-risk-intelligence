"""Population stability: has what the model sees moved since it was trained?

The Population Stability Index compares the share of loans in each bin of a
variable between a reference window (the one the model was trained on) and a
monitored one:

    PSI = sum over bins of (actual - expected) * ln(actual / expected)

It is computed on the score and on every input. It needs no outcomes, which
is the point: labels arrive months or years after a loan is issued (only at
maturity, under this project's default definition), while the inputs and the
score are known on the day of the application.

Bins are fitted on the reference window only and then frozen:

- numeric variables are cut at the reference deciles, so each bin holds about
  a tenth of the reference loans. Heavily tied variables (term is 36 or 60)
  get fewer bins, one per distinct cut. The outer bins are open-ended, so a
  value beyond anything seen in training still lands somewhere.
- categorical variables get one bin per level seen in the reference window,
  plus `unseen` for anything new.
- missing values are always their own bin. A rise in missing income is drift
  in the data feed, and folding it into a value bin would hide it.

An empty bin on one side would make the logarithm infinite, so shares are
floored at `SHARE_FLOOR` first. The usual reading — under 0.10 stable, 0.10 to
0.25 worth watching, above 0.25 a real shift — is an industry rule of thumb,
not a statistical test: on 100,000 loans a PSI of 0.02 is far outside sampling
noise and may still not matter.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from itertools import pairwise
from typing import Final

import numpy as np
import pandas as pd

BINS: Final = 10
SHARE_FLOOR: Final = 1e-4
MISSING: Final = "missing"
UNSEEN: Final = "unseen"
WATCH_FROM: Final = 0.10
SHIFT_FROM: Final = 0.25


class DriftError(ValueError):
    """The windows cannot be compared."""


class Band(StrEnum):
    STABLE = "stable"
    WATCH = "watch"
    SHIFTED = "shifted"

    @classmethod
    def of(cls, psi: float) -> Band:
        if psi >= SHIFT_FROM:
            return cls.SHIFTED
        if psi >= WATCH_FROM:
            return cls.WATCH
        return cls.STABLE


def _number(value: float) -> str:
    return f"{value:,.4g}"


@dataclass(frozen=True, slots=True)
class Binning:
    """Bins fitted on a reference window: interior numeric edges, or categorical levels.

    Numeric bins are right-closed: `<= e0`, `(e0, e1]`, ..., `> e_last`.
    """

    edges: tuple[float, ...] = ()
    levels: tuple[str, ...] | None = None

    @classmethod
    def fit(cls, reference: pd.Series, *, bins: int = BINS) -> Binning:
        if reference.dropna().empty:
            raise DriftError(f"{reference.name} has no observed values in the reference window")
        if not pd.api.types.is_numeric_dtype(reference.dtype):
            return cls(levels=tuple(sorted(set(reference.dropna().astype(str)))))
        values = reference.dropna().to_numpy(dtype=np.float64)
        cuts = np.quantile(values, np.linspace(0, 1, bins + 1)[1:-1])
        # Drop a cut at the maximum: its upper bin would be empty by construction.
        return cls(edges=tuple(float(c) for c in np.unique(cuts) if c < values.max()))

    @property
    def order(self) -> tuple[str, ...]:
        """Every bin label, in display order; missing last."""
        if self.levels is not None:
            return (*self.levels, UNSEEN, MISSING)
        if not self.edges:  # one distinct value in the reference window
            return ("any value", MISSING)
        e = [_number(x) for x in self.edges]
        inner = [f"({lo}, {hi}]" for lo, hi in pairwise(e)]
        return (f"<= {e[0]}", *inner, f"> {e[-1]}", MISSING)

    def assign(self, values: pd.Series) -> pd.Series:
        """The bin label of every value."""
        order = self.order
        if self.levels is not None:
            text = values.astype(object).where(values.notna(), None)
            known = set(self.levels)
            named = [
                MISSING if v is None else (str(v) if str(v) in known else UNSEEN) for v in text
            ]
            return pd.Series(named, index=values.index, dtype=object)
        numbers = values.to_numpy(dtype=np.float64, na_value=np.nan)
        positions = np.searchsorted(np.asarray(self.edges), numbers, side="left")
        labels = np.asarray(order, dtype=object)[positions]
        labels[np.isnan(numbers)] = MISSING
        return pd.Series(labels, index=values.index, dtype=object)


def shares(labels: pd.Series, order: tuple[str, ...]) -> pd.Series:
    """Share of loans per bin, zero for empty ones."""
    if labels.empty:
        raise DriftError("cannot compute shares of an empty window")
    counts = labels.value_counts(normalize=True)
    return counts.reindex(list(order), fill_value=0.0).astype("float64")


def psi(expected: pd.Series, actual: pd.Series) -> pd.Series:
    """Per-bin PSI terms; their sum is the index. Bins empty on both sides contribute 0."""
    e = expected.clip(lower=SHARE_FLOOR)
    a = actual.clip(lower=SHARE_FLOOR)
    terms = (a - e) * np.log(a / e)
    return terms.where((expected > 0) | (actual > 0), 0.0)


@dataclass(frozen=True, slots=True)
class Stability:
    """One variable's PSI between the reference and monitored windows."""

    name: str
    bins: pd.DataFrame  # index: bin label; columns: reference, actual, psi

    @property
    def psi(self) -> float:
        return float(self.bins["psi"].sum())

    @property
    def band(self) -> Band:
        return Band.of(self.psi)

    def largest_move(self) -> str:
        """The bin whose share moved most, as `label: reference -> actual`."""
        moved = str((self.bins["actual"] - self.bins["reference"]).abs().idxmax())
        row = self.bins.loc[moved]
        return f"{moved}: {row['reference']:.1%} -> {row['actual']:.1%}"


def stability(
    name: str, reference: pd.Series, actual: pd.Series, *, binning: Binning | None = None
) -> Stability:
    """PSI of `actual` against `reference`, with bins fitted on `reference` unless given."""
    binning = binning or Binning.fit(reference)
    expected = shares(binning.assign(reference), binning.order)
    observed = shares(binning.assign(actual), binning.order)
    bins = pd.DataFrame({"reference": expected, "actual": observed})
    bins["psi"] = psi(expected, observed)
    keep = (bins["reference"] > 0) | (bins["actual"] > 0)
    return Stability(name=name, bins=bins.loc[keep])
