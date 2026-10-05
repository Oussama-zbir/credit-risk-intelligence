"""HTTP scoring service over one verified model artifact.

    CREDIT_RISK_ARTIFACT_DIR=artifacts/<model_id> \
        uvicorn --factory credit_risk.service:app_from_env

The artifact is loaded and verified once, at startup, by `load_artifact`: a
hash mismatch, a different scikit-learn, or reference loans that no longer
reproduce their saved PDs stop the process before it accepts a request. A
service that cannot trust its model should fail to start, not fail (or worse,
succeed) on every request.

The request schema is the processed applicant feature set — what
`build_features` produces, not the raw Lending Club strings: `fico` is the band
midpoint, `term_months` an integer, `credit_history_months` measured at the
application date. It covers every applicant feature, so it does not change when
a model drops one; at startup the model's features must be a subset of it,
which refuses a model trained with lender pricing (grade and interest rate are
set after scoring, so a client could not supply them). Unknown fields are
rejected rather than ignored, so a misspelt field never silently scores as
"not reported". Only the fields the source file is known to leave empty may be
null; a model was fitted to handle exactly those.

Categorical levels the model never saw in training are scored, not rejected —
the model has defined behaviour for them — but are returned with the result and
logged, because a new level appearing in volume is drift.

Endpoints are synchronous: scoring is CPU-bound numpy, so FastAPI runs it in its
worker threadpool rather than on the event loop.
"""

from __future__ import annotations

import logging
import math
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Annotated, Final, Literal

import pandas as pd
from fastapi import FastAPI, Request
from pydantic import BaseModel, ConfigDict, Field

from credit_risk.models.artifact import (
    ArtifactError,
    ModelFamily,
    Provenance,
    ScoringModel,
    load_artifact,
)
from credit_risk.models.explain import LABELS, reason_codes

ARTIFACT_ENV: Final = "CREDIT_RISK_ARTIFACT_DIR"
MAX_APPLICATIONS: Final = 100

logger = logging.getLogger(__name__)

type Count = Annotated[float, Field(ge=0)]
type Category = Annotated[str, Field(min_length=1, max_length=64)]


class Application(BaseModel):
    """One applicant, in the processed feature space the model was trained on."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    loan_amnt: Annotated[float, Field(ge=1, le=40_000)]
    term_months: Literal[36, 60]
    purpose: Category
    application_type: Category
    annual_inc: Count
    verification_status: Category
    home_ownership: Category
    fico: Annotated[float, Field(ge=300, le=850)]
    credit_history_months: Count
    delinq_2yrs: Count
    open_acc: Count
    total_acc: Count
    pub_rec: Count
    revol_bal: Count
    # The fields the source leaves empty; null means "not reported".
    emp_length_years: Annotated[float, Field(ge=0, le=10)] | None = None
    dti: Count | None = None
    revol_util: Count | None = None
    mort_acc: Count | None = None
    pub_rec_bankruptcies: Count | None = None
    inq_last_6mths: Count | None = None


class ScoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    applications: list[Application] = Field(min_length=1, max_length=MAX_APPLICATIONS)


class ReasonCode(BaseModel):
    feature: str
    label: str
    value: float | str | None  # as the applicant gave it; null when not reported
    contribution: float  # log-odds of default added relative to the training population
    text: str


class Result(BaseModel):
    probability_of_default: float
    reasons: list[ReasonCode]
    unseen_categories: dict[str, list[str]]


class ScoreResponse(BaseModel):
    model_id: str
    model_family: ModelFamily
    results: list[Result]  # in request order


class ModelInfo(BaseModel):
    model_id: str
    model_family: ModelFamily
    created_at: datetime
    features: tuple[str, ...]
    dropped_features: dict[str, str]
    provenance: Provenance
    test_scores: dict[str, float]


class Health(BaseModel):
    status: Literal["ok"]
    model_id: str


def _check_servable(scoring: ScoringModel) -> None:
    unservable = set(scoring.model.features) - set(Application.model_fields)
    if unservable:
        raise ArtifactError(
            f"model {scoring.manifest.model_id} needs features the service does not accept: "
            f"{sorted(unservable)}"
        )


# Every nullable field is numeric. One that is null in every row of a batch
# would otherwise arrive as an object column of None.
_NULLABLE: Final = tuple(
    name for name, field in Application.model_fields.items() if not field.is_required()
)


def _frame(applications: list[Application]) -> pd.DataFrame:
    frame = pd.DataFrame([application.model_dump() for application in applications])
    return frame.astype(dict.fromkeys(_NULLABLE, "float64"))


def _shown(value: object) -> float | str | None:
    if isinstance(value, str):
        return value
    number = float(value)  # type: ignore[arg-type]
    return None if math.isnan(number) else number


def score_applications(scoring: ScoringModel, applications: list[Application]) -> ScoreResponse:
    frame = _frame(applications)
    pds = scoring.predict_pd(frame)
    reasons = reason_codes(scoring.explain(frame), frame)
    results = []
    for row, (pd_hat, codes) in enumerate(zip(pds, reasons, strict=True)):
        unseen = scoring.unseen_categories(frame.iloc[[row]])
        results.append(
            Result(
                probability_of_default=float(pd_hat),
                reasons=[
                    ReasonCode(
                        feature=code.feature,
                        label=LABELS.get(code.feature, code.feature),
                        value=_shown(code.value),
                        contribution=code.contribution,
                        text=str(code),
                    )
                    for code in codes
                ],
                unseen_categories=unseen,
            )
        )
    return ScoreResponse(
        model_id=scoring.manifest.model_id,
        model_family=scoring.manifest.family,
        results=results,
    )


def _scoring(request: Request) -> ScoringModel:
    scoring: ScoringModel = request.app.state.scoring
    return scoring


def create_app(artifact_dir: Path) -> FastAPI:
    """An app that loads and verifies `artifact_dir` on startup, or refuses to start."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        scoring = load_artifact(artifact_dir)
        _check_servable(scoring)
        app.state.scoring = scoring
        logger.info(
            "model loaded: model_id=%s family=%s features=%d",
            scoring.manifest.model_id,
            scoring.manifest.family,
            len(scoring.model.features),
        )
        yield

    app = FastAPI(title="credit-risk-intelligence scoring", lifespan=lifespan)

    @app.get("/health")
    def health(request: Request) -> Health:
        return Health(status="ok", model_id=_scoring(request).manifest.model_id)

    @app.get("/v1/model")
    def model_info(request: Request) -> ModelInfo:
        manifest = _scoring(request).manifest
        return ModelInfo(
            model_id=manifest.model_id,
            model_family=manifest.family,
            created_at=manifest.created_at,
            features=manifest.features,
            dropped_features=manifest.dropped_features,
            provenance=manifest.provenance,
            test_scores=manifest.test_scores,
        )

    @app.post("/v1/score")
    def score(body: ScoreRequest, request: Request) -> ScoreResponse:
        started = time.perf_counter()
        response = score_applications(_scoring(request), body.applications)
        for position, result in enumerate(response.results):
            if result.unseen_categories:
                logger.warning(
                    "unseen categories: model_id=%s application=%d values=%s",
                    response.model_id,
                    position,
                    result.unseen_categories,
                )
        logger.info(
            "scored: model_id=%s applications=%d latency_ms=%.1f",
            response.model_id,
            len(response.results),
            (time.perf_counter() - started) * 1000,
        )
        return response

    return app


def app_from_env() -> FastAPI:
    """Factory for `uvicorn --factory`; the artifact directory comes from the environment."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:     %(name)s %(message)s")
    directory = os.environ.get(ARTIFACT_ENV)
    if not directory:
        raise ArtifactError(f"set {ARTIFACT_ENV} to the artifact directory to serve")
    return create_app(Path(directory))
