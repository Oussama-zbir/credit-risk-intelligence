"""A fitted booster on disk, pinned to the data and code that produced it.

    artifacts/<model_id>/
      manifest.json      what the model is, what it was trained on, how it scored
      model.pkl          the FittedBooster (trees + calibrator)
      reference.parquet  sample loans and the PDs the model gave them at save time

A scoring service must be able to answer "which model produced this PD, and
from which data?" for any decision it made, so an artifact is never a bare
pickle. The manifest records the processed file's SHA-256 (which the data
report ties back to the raw export), the train, calibration and test windows,
the out-of-time test scores, the feature list with every categorical level seen
in training, and the library versions. The model id is the model file's own
SHA-256 prefix, so two artifacts with the same id are the same model, and a
directory is never overwritten.

`load_artifact` refuses an artifact rather than serving it wrong:

- the model file's hash must match the manifest (a truncated copy or a model
  swapped under an old manifest fails here);
- scikit-learn must be the version it was saved with — its pickles are not
  supported across versions, and an old tree object that unpickles cleanly
  under a new release can still misbehave;
- the reference loans must score to the saved PDs. This is the check that
  matters: it catches anything else in the environment (numpy, pandas, a
  changed feature encoding) that would silently move the scores.

The hash detects corruption, not tampering: whoever can rewrite the model file
can rewrite the manifest next to it, and unpickling runs code. Artifacts must
come from a store only the training job can write to.
"""

from __future__ import annotations

import hashlib
import pickle
import platform
import shutil
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Final, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import BaseModel, ConfigDict

from credit_risk.models.boosting import FittedBooster
from credit_risk.models.calibration import Method
from credit_risk.models.explain import Explanation, explain
from credit_risk.models.metrics import Scores

FORMAT_VERSION: Final = 1
MANIFEST: Final = "manifest.json"
MODEL: Final = "model.pkl"
REFERENCE: Final = "reference.parquet"
REFERENCE_LOANS: Final = 200
# Same model, same inputs, same libraries: PDs agree to float rounding, not approximately.
REFERENCE_TOLERANCE: Final = 1e-9
_LIBRARIES: Final = ("numpy", "pandas", "scikit-learn")


class ArtifactError(RuntimeError):
    """The artifact cannot be trusted to reproduce the model that was saved."""


class _Record(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Provenance(_Record):
    """Where the training data came from and how it was cut, as `YYYY-MM` months."""

    data_sha256: str
    train_window: tuple[str, str]
    calibration_months: int
    test_window: tuple[str, str]


class Manifest(_Record):
    format_version: Literal[1]
    model_id: str
    model_sha256: str
    created_at: datetime
    features: tuple[str, ...]
    categories: dict[str, tuple[str, ...]]
    calibration: Method
    trees: int
    base_log_odds: float
    provenance: Provenance
    test_scores: dict[str, float]
    libraries: dict[str, str]


def _libraries() -> dict[str, str]:
    return {"python": platform.python_version(), **{name: version(name) for name in _LIBRARIES}}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True, slots=True)
class ScoringModel:
    """A loaded artifact: the booster plus the manifest that vouches for it."""

    manifest: Manifest
    booster: FittedBooster

    def unseen_categories(self, frame: pd.DataFrame) -> dict[str, list[str]]:
        """Per categorical feature, values in `frame` the model never saw in training.

        They are scored as missing rather than rejected; a service should log
        them, because a new level appearing in volume is drift.
        """
        missing = set(self.booster.features) - set(frame.columns)
        if missing:
            raise ValueError(f"frame lacks model features: {sorted(missing)}")
        unseen = {}
        for name, known in self.manifest.categories.items():
            values = set(frame[name].dropna().astype(str)) - set(known)
            if values:
                unseen[name] = sorted(values)
        return unseen

    def predict_pd(self, frame: pd.DataFrame) -> npt.NDArray[np.float64]:
        self.unseen_categories(frame)  # fails fast on missing columns
        return self.booster.predict_pd(frame)

    def explain(self, frame: pd.DataFrame) -> Explanation:
        return explain(self.booster, frame)


def save_artifact(
    booster: FittedBooster,
    root: Path,
    *,
    provenance: Provenance,
    test_scores: Scores,
    reference: pd.DataFrame,
) -> Path:
    """Write the booster under `root/<model_id>/` and return that directory.

    `reference` is a sample of loans to pin the model's behaviour; at most
    `REFERENCE_LOANS` of them are kept, with only the model's features.
    """
    if reference.empty:
        raise ArtifactError("an artifact needs reference loans to verify itself on load")
    model_bytes = pickle.dumps(booster, protocol=pickle.HIGHEST_PROTOCOL)
    digest = _sha256(model_bytes)
    model_id = digest[:12]
    target = root / model_id
    if target.exists():
        raise ArtifactError(f"{target} already exists; artifacts are never overwritten")

    sample = reference.head(REFERENCE_LOANS)[list(booster.features)].reset_index(drop=True)
    sample["expected_pd"] = booster.predict_pd(sample)
    manifest = Manifest(
        format_version=FORMAT_VERSION,
        model_id=model_id,
        model_sha256=digest,
        created_at=datetime.now(UTC),
        features=booster.features,
        categories=booster.categories,
        calibration=booster.calibrator.method,
        trees=booster.trees,
        base_log_odds=explain(booster, sample.head(1)).base_value,
        provenance=provenance,
        test_scores={name: float(value) for name, value in asdict(test_scores).items()},
        libraries=_libraries(),
    )

    # Written beside the target and renamed into place, so a crash never
    # leaves a half-written artifact under a valid id.
    root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{model_id}-", dir=root))
    try:
        (staging / MODEL).write_bytes(model_bytes)
        sample.to_parquet(staging / REFERENCE, index=False)
        (staging / MANIFEST).write_text(manifest.model_dump_json(indent=2) + "\n")
        staging.rename(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target


def load_artifact(directory: Path) -> ScoringModel:
    """Load and verify an artifact; raises `ArtifactError` instead of serving a doubtful one."""
    try:
        manifest = Manifest.model_validate_json((directory / MANIFEST).read_text())
    except (OSError, ValueError) as exc:
        raise ArtifactError(f"unreadable manifest in {directory}: {exc}") from exc

    model_bytes = (directory / MODEL).read_bytes()
    if _sha256(model_bytes) != manifest.model_sha256:
        raise ArtifactError(f"{MODEL} does not match the manifest of model {manifest.model_id}")

    saved, installed = manifest.libraries["scikit-learn"], version("scikit-learn")
    if saved != installed:
        raise ArtifactError(f"model saved with scikit-learn {saved}, {installed} is installed")

    booster = pickle.loads(model_bytes)  # trusted store only; see module docstring
    if not isinstance(booster, FittedBooster) or booster.features != manifest.features:
        raise ArtifactError(f"{MODEL} is not the booster the manifest describes")

    reference = pd.read_parquet(directory / REFERENCE)
    gap = np.abs(booster.predict_pd(reference) - reference["expected_pd"].to_numpy())
    if gap.max() > REFERENCE_TOLERANCE:
        raise ArtifactError(
            f"reference loans score up to {gap.max():.2e} away from their saved PDs"
        )
    return ScoringModel(manifest=manifest, booster=booster)
