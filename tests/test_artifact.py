import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import make_processed
from credit_risk.data.features import feature_names
from credit_risk.data.target import TARGET
from credit_risk.models.artifact import (
    MANIFEST,
    MODEL,
    REFERENCE,
    REFERENCE_LOANS,
    ArtifactError,
    Provenance,
    load_artifact,
    save_artifact,
)
from credit_risk.models.boosting import FittedBooster, fit_booster
from credit_risk.models.metrics import Scores, score

PROVENANCE = Provenance(
    data_sha256="ab" * 32,
    train_window=("2012-01", "2014-06"),
    calibration_months=6,
    test_window=("2015-07", "2016-12"),
)


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return make_processed(6_000)


@pytest.fixture(scope="module")
def booster(frame: pd.DataFrame) -> FittedBooster:
    fitting = frame[frame["issue_date"] < "2014-07-01"]
    calibration = frame[frame["issue_date"].between("2014-07-01", "2014-12-01")]
    return fit_booster(fitting, calibration, feature_names())


@pytest.fixture(scope="module")
def test_window(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame["issue_date"] >= "2015-07-01"]


@pytest.fixture(scope="module")
def test_scores(booster: FittedBooster, test_window: pd.DataFrame) -> Scores:
    return score(test_window[TARGET], booster.predict_pd(test_window), reference_rate=0.2)


@pytest.fixture
def saved(
    tmp_path: Path, booster: FittedBooster, test_scores: Scores, test_window: pd.DataFrame
) -> Path:
    return save_artifact(
        booster,
        tmp_path / "artifacts",
        provenance=PROVENANCE,
        test_scores=test_scores,
        reference=test_window,
    )


def test_loaded_artifact_scores_exactly_like_the_saved_booster(
    saved: Path, booster: FittedBooster, test_window: pd.DataFrame
) -> None:
    model = load_artifact(saved)
    np.testing.assert_array_equal(model.predict_pd(test_window), booster.predict_pd(test_window))
    assert model.explain(test_window.head(3)).base_value == model.manifest.base_log_odds


def test_manifest_pins_the_model_to_its_data_windows_and_libraries(
    saved: Path, booster: FittedBooster, test_scores: Scores
) -> None:
    manifest = load_artifact(saved).manifest
    assert saved.name == manifest.model_id == manifest.model_sha256[:12]
    assert manifest.provenance == PROVENANCE
    assert manifest.features == feature_names()
    assert manifest.trees == booster.trees
    assert manifest.test_scores["auc"] == pytest.approx(test_scores.auc)
    assert manifest.categories["home_ownership"] == ("MORTGAGE", "OWN", "RENT")
    assert set(manifest.categories) == {"purpose", "application_type", "home_ownership",
                                        "verification_status"}  # fmt: skip
    assert {"python", "numpy", "pandas", "scikit-learn"} <= set(manifest.libraries)
    reference = pd.read_parquet(saved / REFERENCE)
    assert len(reference) == REFERENCE_LOANS
    assert list(reference.columns) == [*feature_names(), "expected_pd"]


def test_an_artifact_is_never_overwritten(
    saved: Path, booster: FittedBooster, test_scores: Scores, test_window: pd.DataFrame
) -> None:
    with pytest.raises(ArtifactError, match="never overwritten"):
        save_artifact(
            booster,
            saved.parent,
            provenance=PROVENANCE,
            test_scores=test_scores,
            reference=test_window,
        )
    assert [p.name for p in saved.parent.iterdir()] == [saved.name]  # no staging left behind


def test_saving_without_reference_loans_is_refused(
    tmp_path: Path, booster: FittedBooster, test_scores: Scores, test_window: pd.DataFrame
) -> None:
    with pytest.raises(ArtifactError, match="reference loans"):
        save_artifact(
            booster,
            tmp_path,
            provenance=PROVENANCE,
            test_scores=test_scores,
            reference=test_window.head(0),
        )


def _edit_manifest(directory: Path, **changes: object) -> None:
    path = directory / MANIFEST
    path.write_text(json.dumps({**json.loads(path.read_text()), **changes}))


def test_a_corrupted_model_file_is_refused(saved: Path) -> None:
    (saved / MODEL).write_bytes((saved / MODEL).read_bytes()[:-1])
    with pytest.raises(ArtifactError, match="does not match the manifest"):
        load_artifact(saved)


def test_a_different_scikit_learn_version_is_refused(saved: Path) -> None:
    libraries = json.loads((saved / MANIFEST).read_text())["libraries"]
    _edit_manifest(saved, libraries={**libraries, "scikit-learn": "0.24.2"})
    with pytest.raises(ArtifactError, match=r"saved with scikit-learn 0\.24\.2"):
        load_artifact(saved)


@pytest.mark.parametrize("changes", [{"format_version": 2}, {"surprise": True}])
def test_an_unknown_manifest_format_is_refused(saved: Path, changes: dict[str, object]) -> None:
    _edit_manifest(saved, **changes)
    with pytest.raises(ArtifactError, match="unreadable manifest"):
        load_artifact(saved)


def test_reference_loans_that_no_longer_score_the_same_are_refused(saved: Path) -> None:
    # Stands in for any environment change that moves scores without an error.
    reference = pd.read_parquet(saved / REFERENCE)
    reference["expected_pd"] += 1e-6
    reference.to_parquet(saved / REFERENCE, index=False)
    with pytest.raises(ArtifactError, match="away from their saved PDs"):
        load_artifact(saved)


def test_unseen_categories_are_reported_and_scored_as_missing(
    saved: Path, test_window: pd.DataFrame
) -> None:
    model = load_artifact(saved)
    loans = test_window.head(2).copy()
    loans["purpose"] = loans["purpose"].cat.add_categories("wedding")
    loans.loc[loans.index[0], "purpose"] = "wedding"
    assert model.unseen_categories(loans) == {"purpose": ["wedding"]}
    as_missing = loans.head(1).copy()
    as_missing["purpose"] = np.nan
    assert model.predict_pd(loans)[0] == model.predict_pd(as_missing)[0]


def test_scoring_a_frame_without_every_feature_fails(
    saved: Path, test_window: pd.DataFrame
) -> None:
    with pytest.raises(ValueError, match=r"lacks model features: \['fico'\]"):
        load_artifact(saved).predict_pd(test_window.drop(columns="fico"))
