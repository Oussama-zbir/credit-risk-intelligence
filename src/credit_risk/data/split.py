"""Out-of-time splits.

A credit model is trained on loans issued in the past and used on applicants who
arrive in the future, so it is validated the same way: train on earlier
vintages, test on a later window that the model never saw. A random split
leaks the future into training — macro conditions, underwriting changes and
Lending Club's own repricing all move with calendar time — and flatters every
model by the same amount, which hides exactly the degradation a lender cares
about.

An optional gap between the two windows keeps loans issued in the months just
before the test window out of training, mirroring the delay between a model
being fitted and being deployed.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from credit_risk.data.target import ISSUE_DATE, TARGET


class SplitError(ValueError):
    """The requested windows cannot produce a valid out-of-time split."""


@dataclass(frozen=True, slots=True)
class TimeWindow:
    """Issue months `start` to `end`, both inclusive."""

    start: pd.Timestamp
    end: pd.Timestamp

    def __post_init__(self) -> None:
        if self.start > self.end:
            raise SplitError(f"window starts after it ends: {self.start} > {self.end}")

    def contains(self, dates: pd.Series) -> pd.Series:
        return (dates >= self.start) & (dates <= self.end)


@dataclass(frozen=True, slots=True)
class OutOfTimeSplit:
    train: pd.DataFrame
    test: pd.DataFrame
    train_window: TimeWindow
    test_window: TimeWindow


def out_of_time_split(
    labelled: pd.DataFrame, *, train: TimeWindow, test: TimeWindow
) -> OutOfTimeSplit:
    """Split labelled loans by issue date into a training and a later test window.

    Refuses overlapping windows, a test window that is not strictly later, and
    any window that ends up without both classes — a metric computed on one
    class is not a metric.
    """
    if test.start <= train.end:
        raise SplitError(
            f"test window must start after the training window ends: "
            f"{test.start.date()} <= {train.end.date()}"
        )

    dates = labelled[ISSUE_DATE]
    result = OutOfTimeSplit(
        train=labelled.loc[train.contains(dates)],
        test=labelled.loc[test.contains(dates)],
        train_window=train,
        test_window=test,
    )
    for name, part in (("train", result.train), ("test", result.test)):
        classes = part[TARGET].nunique()
        if classes < 2:
            raise SplitError(f"{name} window has {len(part)} loans and {classes} class(es)")
    return result


def hold_out_latest(
    train: pd.DataFrame, window: TimeWindow, *, months: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a training window into an earlier fitting slice and its last `months`.

    A model whose probabilities are recalibrated needs data it was not fitted on,
    and the test window is not available for that. The held-out slice is the
    *latest* part of the training window rather than a random sample, so the
    calibrator is fitted on the vintages closest to deployment and the fitted
    model is still only ever scored on loans issued after the ones it learned
    from.
    """
    span = (window.end.year - window.start.year) * 12 + window.end.month - window.start.month + 1
    if not 0 < months < span:
        raise SplitError(f"cannot hold out {months} of the {span} months in the training window")
    cut = window.end - pd.DateOffset(months=months - 1)
    latest = train[ISSUE_DATE] >= cut
    parts = train.loc[~latest], train.loc[latest]
    for name, part in zip(("fitting", "held-out"), parts, strict=True):
        classes = part[TARGET].nunique()
        if classes < 2:
            raise SplitError(f"{name} slice has {len(part)} loans and {classes} class(es)")
    return parts
