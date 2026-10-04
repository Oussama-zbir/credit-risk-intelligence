"""A fitted model on disk, pinned to the data and code that produced it.

    artifacts/<model_id>/
      manifest.json      what the model is, what it was trained on, how it scored
      model.pkl          the fitted model (FittedBaseline or FittedBooster)
      reference.parquet  sample loans and the PDs the model gave them at save time

A scoring service must be able to answer "which model produced this PD, and
from which data?" for any decision it made, so an artifact is never a bare
pickle. The manifest records the processed file's SHA-256 (which the data
report ties back to the raw export), the train, calibration and test windows,
the out-of-time test scores, the fitted features and the requested ones left
out (with why), every categorical level seen in training, and the library
versions. The model id is the model file's own SHA-256 prefix, so two
artifacts with the same id are the same model, and a directory is never
overwritten.

Two model families are stored, and the manifest says which in `model.family`.
What is common to both — provenance, features, categories, scores — sits at
the top level; what only one family has sits in its `model` block: the
logistic regression's regularisation, intercept and input count, the booster's
trees, calibration method and TreeSHAP base value. Neither carries the other's
fields with placeholder values.

`load_artifact` refuses an artifact rather than serving it wrong:

- the manifest must be this format version, and the pickled model must be the
  family and feature set the manifest describes;
- the model file's hash must match the manifest (a truncated copy or a model
  swapped under an old manifest fails here);
- scikit-learn must be the version it was saved with — its pickles are not
  supported across versions, and an old estimator that unpickles cleanly
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
import json
import pickle
import platform
import shutil
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from importlib.metadata import version
from pathlib import Path
from typing import Annotated, Final, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from credit_risk.models.baseline import FittedBaseline
from credit_risk.models.boosting import FittedBooster
from credit_risk.models.calibration import Method
from credit_risk.models.explain import Explanation, explain
from credit_risk.models.metrics import Scores

# 1: booster only. 2: model family recorded, logistic regression supported.
FORMAT_VERSION: Final = 2
MANIFEST: Final = "manifest.json"
MODEL: Final = "model.pkl"
REFERENCE: Final = "reference.parquet"
REFERENCE_LOANS: Final = 200
# Same model, same inputs, same libraries: PDs agree to float rounding, not approximately.
REFERENCE_TOLERANCE: Final = 1e-9
_LIBRARIES: Final = ("numpy", "pandas", "scikit-learn")

type FittedModel = FittedBaseline | FittedBooster


class ArtifactError(RuntimeError):
    """The artifact cannot be trusted to reproduce the model that was saved."""


class ModelFamily(StrEnum):
    LOGISTIC_REGRESSION = "logistic-regression"
    GRADIENT_BOOSTING = "gradient-boosting"


class _Record(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Provenance(_Record):
    """Where the training data came from and how it was cut, as `YYYY-MM` months.

    `calibration_months` is the latest part of the training window held out to
    calibrate a booster; `None` for a model fitted on the whole window.
    """

    data_sha256: str
    train_window: tuple[str, str]
    calibration_months: int | None
    test_window: tuple[str, str]


class LogisticRegressionDetails(_Record):
    family: Literal[ModelFamily.LOGISTIC_REGRESSION]
    regularization_c: float
    intercept: float
    inputs: int  # columns after imputation indicators and one-hot encoding
    train_default_rate: float


class GradientBoostingDetails(_Record):
    family: Literal[ModelFamily.GRADIENT_BOOSTING]
    trees: int
    calibration: Method
    base_log_odds: float  # TreeSHAP base value


type ModelDetails = Annotated[
    LogisticRegressionDetails | GradientBoostingDetails, Field(discriminator="family")
]


class Manifest(_Record):
    format_version: Literal[2]
    model_id: str
    model_sha256: str
    created_at: datetime
    model: ModelDetails
    features: tuple[str, ...]
    dropped_features: dict[str, str]
    categories: dict[str, tuple[str, ...]]
    provenance: Provenance
    test_scores: dict[str, float]
    libraries: dict[str, str]

    @property
    def family(self) -> ModelFamily:
        return self.model.family


def family_of(model: FittedModel) -> ModelFamily:
    if isinstance(model, FittedBaseline):
        return ModelFamily.LOGISTIC_REGRESSION
    return ModelFamily.GRADIENT_BOOSTING


def _details(model: FittedModel, sample: pd.DataFrame) -> ModelDetails:
    if isinstance(model, FittedBaseline):
        regression = model.pipeline.named_steps["model"]
        return LogisticRegressionDetails(
            family=ModelFamily.LOGISTIC_REGRESSION,
            regularization_c=float(regression.C),
            intercept=float(regression.intercept_[0]),
            inputs=int(regression.coef_.shape[1]),
            train_default_rate=model.train_default_rate,
        )
    return GradientBoostingDetails(
        family=ModelFamily.GRADIENT_BOOSTING,
        trees=model.trees,
        calibration=model.calibrator.method,
        base_log_odds=explain(model, sample.head(1)).base_value,
    )


def _libraries() -> dict[str, str]:
    return {"python": platform.python_version(), **{name: version(name) for name in _LIBRARIES}}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True, slots=True)
class ScoringModel:
    """A loaded artifact: the fitted model plus the manifest that vouches for it."""

    manifest: Manifest
    model: FittedModel

    def unseen_categories(self, frame: pd.DataFrame) -> dict[str, list[str]]:
        """Per categorical feature, values in `frame` the model never saw in training.

        They are scored, not rejected. The booster scores them as missing; the
        logistic regression pools them with the infrequent levels, or encodes
        them as no level at all if training had none. A service should log
        them, because a new level appearing in volume is drift.
        """
        missing = set(self.model.features) - set(frame.columns)
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
        return self.model.predict_pd(frame)

    def explain(self, frame: pd.DataFrame) -> Explanation:
        """TreeSHAP contributions; only gradient-boosting artifacts have them."""
        if not isinstance(self.model, FittedBooster):
            raise TypeError(f"{self.manifest.family} artifacts carry no TreeSHAP explanation")
        return explain(self.model, frame)


def save_artifact(
    model: FittedModel,
    root: Path,
    *,
    provenance: Provenance,
    test_scores: Scores,
    reference: pd.DataFrame,
) -> Path:
    """Write `model` under `root/<model_id>/` and return that directory.

    `reference` is a sample of loans to pin the model's behaviour; at most
    `REFERENCE_LOANS` of them are kept, with only the model's fitted features.
    """
    if reference.empty:
        raise ArtifactError("an artifact needs reference loans to verify itself on load")
    model_bytes = pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
    digest = _sha256(model_bytes)
    model_id = digest[:12]
    target = root / model_id
    if target.exists():
        raise ArtifactError(f"{target} already exists; artifacts are never overwritten")

    sample = reference.head(REFERENCE_LOANS)[list(model.features)].reset_index(drop=True)
    sample["expected_pd"] = model.predict_pd(sample)
    manifest = Manifest(
        format_version=FORMAT_VERSION,
        model_id=model_id,
        model_sha256=digest,
        created_at=datetime.now(UTC),
        model=_details(model, sample),
        features=model.features,
        dropped_features=model.dropped,
        categories=model.categories,
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


def _read_manifest(directory: Path) -> Manifest:
    try:
        raw = json.loads((directory / MANIFEST).read_text())
    except (OSError, ValueError) as exc:
        raise ArtifactError(f"unreadable manifest in {directory}: {exc}") from exc
    found = raw.get("format_version") if isinstance(raw, dict) else None
    if found != FORMAT_VERSION:
        raise ArtifactError(f"artifact format {found!r} is not supported, only {FORMAT_VERSION}")
    try:
        return Manifest.model_validate(raw)
    except ValueError as exc:
        raise ArtifactError(f"unreadable manifest in {directory}: {exc}") from exc


def load_artifact(directory: Path) -> ScoringModel:
    """Load and verify an artifact; raises `ArtifactError` instead of serving a doubtful one."""
    manifest = _read_manifest(directory)

    model_bytes = (directory / MODEL).read_bytes()
    if _sha256(model_bytes) != manifest.model_sha256:
        raise ArtifactError(f"{MODEL} does not match the manifest of model {manifest.model_id}")

    saved, installed = manifest.libraries["scikit-learn"], version("scikit-learn")
    if saved != installed:
        raise ArtifactError(f"model saved with scikit-learn {saved}, {installed} is installed")

    model = pickle.loads(model_bytes)  # trusted store only; see module docstring
    if (
        not isinstance(model, FittedBaseline | FittedBooster)
        or family_of(model) is not manifest.family
        or model.features != manifest.features
    ):
        raise ArtifactError(f"{MODEL} is not the {manifest.family} model the manifest describes")

    reference = pd.read_parquet(directory / REFERENCE)
    gap = np.abs(model.predict_pd(reference) - reference["expected_pd"].to_numpy())
    if gap.max() > REFERENCE_TOLERANCE:
        raise ArtifactError(
            f"reference loans score up to {gap.max():.2e} away from their saved PDs"
        )
    return ScoringModel(manifest=manifest, model=model)
