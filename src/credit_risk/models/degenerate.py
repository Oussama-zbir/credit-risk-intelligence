"""Requested features that cannot be fitted on, decided from the fitting rows alone.

A feature is degenerate when it takes at most one value in the rows a model is
fitted on, counting missing as a value: missing everywhere (the real 2010-2011
window has no `mort_acc` at all) or one level everywhere (only individual
applications). Such a feature cannot inform a fit. The booster crashes binning
an all-missing column; the logistic regression fits it silently and reports a
coefficient that is only an intercept shift. One observed value plus missing
values is two values and stays, because whether a field was reported can
itself predict default.

Only the fitting rows decide. A feature that varies only in a later window was
not learned from and must not be scored on, so each model records the features
it actually fitted and never reads the rest.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def degenerate_features(fitting: pd.DataFrame, features: Sequence[str]) -> dict[str, str]:
    """Requested features with at most one value in `fitting`, each mapped to why."""
    dropped = {}
    for name in features:
        values = fitting[name]
        if values.nunique(dropna=False) > 1:
            continue
        if values.isna().all():
            dropped[name] = "all missing"
        else:
            dropped[name] = f"constant {values.dropna().iloc[0]!r}"
    return dropped


def active_features(
    fitting: pd.DataFrame, features: Sequence[str]
) -> tuple[tuple[str, ...], dict[str, str]]:
    """The requested features to fit on, in request order, and those left out.

    Raises `ValueError` if none of them varies.
    """
    dropped = degenerate_features(fitting, features)
    active = tuple(name for name in features if name not in dropped)
    if not active:
        raise ValueError(f"no requested feature varies in the fitting rows: {dropped}")
    return active, dropped
