import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from conftest import make_processed
from credit_risk.data.features import APPLICANT_FEATURES, feature_names
from credit_risk.data.target import TARGET
from credit_risk.data.validation import NULLABLE
from credit_risk.models.artifact import (
    MODEL,
    ArtifactError,
    FittedModel,
    ModelFamily,
    Provenance,
    save_artifact,
)
from credit_risk.models.baseline import fit_baseline
from credit_risk.models.boosting import FittedBooster, fit_booster
from credit_risk.models.explain import REASONS
from credit_risk.models.metrics import score
from credit_risk.service import (
    ARTIFACT_ENV,
    MAX_APPLICATIONS,
    Application,
    app_from_env,
    create_app,
)

PROVENANCE = Provenance(
    data_sha256="ab" * 32,
    train_window=("2012-01", "2014-12"),
    calibration_months=None,
    test_window=("2015-07", "2016-12"),
)


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return make_processed(6_000)


@pytest.fixture(scope="module")
def train(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame["issue_date"] <= "2014-12-01"]


@pytest.fixture(scope="module")
def test_window(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame["issue_date"] >= "2015-07-01"]


def _save(model: FittedModel, root: Path, test_window: pd.DataFrame) -> Path:
    scores = score(test_window[TARGET], model.predict_pd(test_window), reference_rate=0.2)
    provenance = PROVENANCE
    if isinstance(model, FittedBooster):
        provenance = PROVENANCE.model_copy(update={"calibration_months": 6})
    return save_artifact(
        model, root, provenance=provenance, test_scores=scores, reference=test_window
    )


@pytest.fixture(scope="module")
def models(train: pd.DataFrame) -> dict[ModelFamily, FittedModel]:
    fitting = train[train["issue_date"] < "2014-07-01"]
    calibration = train[train["issue_date"] >= "2014-07-01"]
    return {
        ModelFamily.LOGISTIC_REGRESSION: fit_baseline(train, feature_names()),
        ModelFamily.GRADIENT_BOOSTING: fit_booster(fitting, calibration, feature_names()),
    }


@pytest.fixture(scope="module")
def artifacts(
    tmp_path_factory: pytest.TempPathFactory,
    models: dict[ModelFamily, FittedModel],
    test_window: pd.DataFrame,
) -> dict[ModelFamily, Path]:
    root = tmp_path_factory.mktemp("artifacts")
    return {family: _save(model, root, test_window) for family, model in models.items()}


@pytest.fixture(params=list(ModelFamily))
def family(request: pytest.FixtureRequest) -> ModelFamily:
    family: ModelFamily = request.param
    return family


@pytest.fixture
def client(artifacts: dict[ModelFamily, Path], family: ModelFamily) -> Iterator[TestClient]:
    with TestClient(create_app(artifacts[family])) as client:
        yield client


def _payload(row: pd.Series) -> dict[str, Any]:
    """A processed loan as a client would send it: plain JSON values, null for missing."""
    payload: dict[str, Any] = {}
    for name in APPLICANT_FEATURES:
        value = row[name]
        if isinstance(value, str):
            payload[name] = value
        elif pd.isna(value):
            payload[name] = None
        else:
            payload[name] = int(value) if name == "term_months" else float(value)
    return payload


def _application(**overrides: Any) -> dict[str, Any]:
    return {**_payload(make_processed(1).iloc[0]), **overrides}


def _score(client: TestClient, *applications: dict[str, Any]) -> dict[str, Any]:
    response = client.post("/v1/score", json={"applications": list(applications)})
    assert response.status_code == 200, response.text
    body: dict[str, Any] = response.json()
    return body


# Schema.


def test_request_schema_is_the_applicant_feature_set() -> None:
    assert set(Application.model_fields) == set(APPLICANT_FEATURES)


def test_only_fields_the_source_leaves_empty_are_nullable() -> None:
    processed = {"emp_length": "emp_length_years"}
    expected = {processed.get(name, name) for name in NULLABLE}
    optional = {name for name, field in Application.model_fields.items() if not field.is_required()}
    assert optional == expected


# Scoring, for both families.


def test_service_scores_exactly_like_the_model(
    client: TestClient,
    family: ModelFamily,
    models: dict[ModelFamily, FittedModel],
    test_window: pd.DataFrame,
) -> None:
    loans = test_window.head(MAX_APPLICATIONS)
    body = _score(client, *(_payload(row) for _, row in loans.iterrows()))

    assert body["model_family"] == family
    served = [result["probability_of_default"] for result in body["results"]]
    np.testing.assert_allclose(served, models[family].predict_pd(loans), rtol=0, atol=1e-12)


def test_reasons_are_adverse_contributions_largest_first(client: TestClient) -> None:
    risky = _application(fico=640.0, dti=39.0, inq_last_6mths=4.0)
    (result,) = _score(client, risky)["results"]

    contributions = [reason["contribution"] for reason in result["reasons"]]
    assert 0 < len(contributions) <= REASONS
    assert all(c > 0 for c in contributions)
    assert contributions == sorted(contributions, reverse=True)
    for reason in result["reasons"]:
        assert reason["text"].startswith(reason["label"])
        assert reason["value"] == risky[reason["feature"]]


def test_unreported_values_are_scored_and_shown_as_null(client: TestClient) -> None:
    nullable = [name for name, f in Application.model_fields.items() if not f.is_required()]
    sparse = {k: v for k, v in _application().items() if k not in nullable}
    (result,) = _score(client, sparse)["results"]

    assert 0 < result["probability_of_default"] < 1
    for reason in result["reasons"]:
        if reason["feature"] in nullable:
            assert reason["value"] is None
            assert "not reported" in reason["text"]


def test_unseen_categories_are_scored_returned_and_logged(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="credit_risk.service"):
        known, new = _score(client, _application(), _application(purpose="crypto"))["results"]

    assert known["unseen_categories"] == {}
    assert new["unseen_categories"] == {"purpose": ["crypto"]}
    assert 0 < new["probability_of_default"] < 1
    assert any("crypto" in record.getMessage() for record in caplog.records)


def test_model_and_health_report_the_loaded_artifact(
    client: TestClient, artifacts: dict[ModelFamily, Path], family: ModelFamily
) -> None:
    model_id = artifacts[family].name
    assert client.get("/health").json() == {"status": "ok", "model_id": model_id}

    info = client.get("/v1/model").json()
    assert info["model_id"] == model_id
    assert info["model_family"] == family
    assert info["features"] == list(feature_names())
    assert set(info["test_scores"]) >= {"auc", "brier"}


# Rejected requests.


@pytest.mark.parametrize(
    "overrides",
    [
        {"fico": 900.0},
        {"term_months": 48},
        {"loan_amnt": 0.0},
        {"annual_inc": -1.0},
        {"fico": None},  # required, not one of the fields the source leaves empty
        {"purpose": ""},
        {"ficoo": 700.0},  # a misspelt field must not score as "not reported"
    ],
)
def test_invalid_applications_are_rejected(
    artifacts: dict[ModelFamily, Path], overrides: dict[str, Any]
) -> None:
    with TestClient(create_app(artifacts[ModelFamily.LOGISTIC_REGRESSION])) as client:
        response = client.post("/v1/score", json={"applications": [_application(**overrides)]})
    assert response.status_code == 422


@pytest.mark.parametrize("count", [0, MAX_APPLICATIONS + 1])
def test_batch_size_is_bounded(artifacts: dict[ModelFamily, Path], count: int) -> None:
    with TestClient(create_app(artifacts[ModelFamily.LOGISTIC_REGRESSION])) as client:
        response = client.post("/v1/score", json={"applications": [_application()] * count})
    assert response.status_code == 422


# Startup.


def test_a_corrupted_artifact_stops_startup(
    tmp_path: Path, models: dict[ModelFamily, FittedModel], test_window: pd.DataFrame
) -> None:
    saved = _save(models[ModelFamily.LOGISTIC_REGRESSION], tmp_path, test_window)
    model_file = saved / MODEL
    model_file.write_bytes(model_file.read_bytes()[:-1])

    with pytest.raises(ArtifactError, match="does not match"), TestClient(create_app(saved)):
        pass


def test_a_model_using_lender_pricing_is_refused_at_startup(
    tmp_path: Path, train: pd.DataFrame, test_window: pd.DataFrame
) -> None:
    priced = fit_baseline(train, feature_names(include_lender_pricing=True))
    saved = _save(priced, tmp_path, test_window)

    with pytest.raises(ArtifactError, match="grade"), TestClient(create_app(saved)):
        pass


def test_factory_requires_the_artifact_directory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ARTIFACT_ENV, raising=False)
    with pytest.raises(ArtifactError, match=ARTIFACT_ENV):
        app_from_env()
