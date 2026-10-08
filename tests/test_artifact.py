import json
import stat
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
    FittedModel,
    GradientBoostingDetails,
    LogisticRegressionDetails,
    ModelFamily,
    Provenance,
    load_artifact,
    save_artifact,
)
from credit_risk.models.baseline import FittedBaseline, fit_baseline
from credit_risk.models.boosting import FittedBooster, fit_booster
from credit_risk.models.calibration import Method
from credit_risk.models.explain import Reason, explain, reason_codes
from credit_risk.models.metrics import Scores, score

BOOSTER_PROVENANCE = Provenance(
    data_sha256="ab" * 32,
    train_window=("2012-01", "2014-12"),
    calibration_months=6,
    test_window=("2015-07", "2016-12"),
)
BASELINE_PROVENANCE = BOOSTER_PROVENANCE.model_copy(update={"calibration_months": None})
PROVENANCE = {
    ModelFamily.LOGISTIC_REGRESSION: BASELINE_PROVENANCE,
    ModelFamily.GRADIENT_BOOSTING: BOOSTER_PROVENANCE,
}


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return make_processed(6_000)


@pytest.fixture(scope="module")
def train(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame["issue_date"] <= "2014-12-01"]


@pytest.fixture(scope="module")
def booster(train: pd.DataFrame) -> FittedBooster:
    fitting = train[train["issue_date"] < "2014-07-01"]
    calibration = train[train["issue_date"] >= "2014-07-01"]
    return fit_booster(fitting, calibration, feature_names())


@pytest.fixture(scope="module")
def baseline(train: pd.DataFrame) -> FittedBaseline:
    return fit_baseline(train, feature_names())


@pytest.fixture(scope="module")
def test_window(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame["issue_date"] >= "2015-07-01"]


@pytest.fixture(params=list(ModelFamily))
def family(request: pytest.FixtureRequest) -> ModelFamily:
    family: ModelFamily = request.param
    return family


@pytest.fixture
def fitted(family: ModelFamily, baseline: FittedBaseline, booster: FittedBooster) -> FittedModel:
    return baseline if family is ModelFamily.LOGISTIC_REGRESSION else booster


def _scores(model: FittedModel, test_window: pd.DataFrame) -> Scores:
    return score(test_window[TARGET], model.predict_pd(test_window), reference_rate=0.2)


def _save(
    model: FittedModel,
    root: Path,
    test_window: pd.DataFrame,
    provenance: Provenance,
    *,
    reference: pd.DataFrame | None = None,
) -> Path:
    return save_artifact(
        model,
        root,
        provenance=provenance,
        test_scores=_scores(model, test_window),
        reference=test_window if reference is None else reference,
    )


@pytest.fixture
def saved(
    tmp_path: Path, family: ModelFamily, fitted: FittedModel, test_window: pd.DataFrame
) -> Path:
    return _save(fitted, tmp_path / "artifacts", test_window, PROVENANCE[family])


# Shared by both families.


def test_loaded_artifact_scores_exactly_like_the_saved_model(
    saved: Path, fitted: FittedModel, test_window: pd.DataFrame
) -> None:
    model = load_artifact(saved)
    assert type(model.model) is type(fitted)
    np.testing.assert_array_equal(model.predict_pd(test_window), fitted.predict_pd(test_window))


def test_loaded_artifact_explains_exactly_like_the_saved_model(
    saved: Path, fitted: FittedModel, test_window: pd.DataFrame
) -> None:
    loans = test_window.head(100)
    loaded = load_artifact(saved).explain(loans)
    original = explain(fitted, loans)
    assert loaded.base_value == original.base_value
    pd.testing.assert_frame_equal(loaded.contributions, original.contributions)

    def rendered(codes: list[tuple[Reason, ...]]) -> list[list[tuple[str, float]]]:
        return [[(str(r), r.contribution) for r in reasons] for reasons in codes]

    assert rendered(reason_codes(loaded, loans)) == rendered(reason_codes(original, loans))


def test_manifest_pins_the_model_to_its_data_windows_and_libraries(
    saved: Path, family: ModelFamily, fitted: FittedModel, test_window: pd.DataFrame
) -> None:
    manifest = load_artifact(saved).manifest
    assert manifest.format_version == 3
    assert manifest.family is family
    assert saved.name == manifest.model_id == manifest.model_sha256[:12]
    assert manifest.provenance == PROVENANCE[family]
    assert manifest.features == fitted.features == feature_names()
    assert manifest.dropped_features == {}
    assert manifest.test_scores["auc"] == pytest.approx(_scores(fitted, test_window).auc)
    assert manifest.categories["home_ownership"] == ("MORTGAGE", "OWN", "RENT")
    assert set(manifest.categories) == {"purpose", "application_type", "home_ownership",
                                        "verification_status"}  # fmt: skip
    assert {"python", "numpy", "pandas", "scikit-learn"} <= set(manifest.libraries)
    reference = pd.read_parquet(saved / REFERENCE)
    assert len(reference) == REFERENCE_LOANS
    assert list(reference.columns) == [*feature_names(), "expected_pd"]


def test_an_artifact_is_never_overwritten(
    saved: Path, fitted: FittedModel, test_window: pd.DataFrame, family: ModelFamily
) -> None:
    with pytest.raises(ArtifactError, match="never overwritten"):
        _save(fitted, saved.parent, test_window, PROVENANCE[family])
    assert [p.name for p in saved.parent.iterdir()] == [saved.name]  # no staging left behind


def test_an_artifact_is_readable_by_other_users(saved: Path) -> None:
    # The service runs as its own user (the image's non-root one), not the trainer.
    assert stat.S_IMODE(saved.stat().st_mode) == 0o755


def test_saving_without_reference_loans_is_refused(
    tmp_path: Path, fitted: FittedModel, test_window: pd.DataFrame, family: ModelFamily
) -> None:
    with pytest.raises(ArtifactError, match="reference loans"):
        _save(fitted, tmp_path, test_window, PROVENANCE[family], reference=test_window.head(0))


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


@pytest.mark.parametrize("version", [1, 2, 4, None])
def test_other_manifest_format_versions_are_refused(saved: Path, version: object) -> None:
    # Version 1 was booster-only, with no model family; version 2 logistic
    # regressions lack the training input means explanations need.
    _edit_manifest(saved, format_version=version)
    with pytest.raises(ArtifactError, match=rf"format {version!r} is not supported"):
        load_artifact(saved)


def test_unknown_manifest_fields_are_refused(saved: Path) -> None:
    _edit_manifest(saved, surprise=True)
    with pytest.raises(ArtifactError, match="unreadable manifest"):
        load_artifact(saved)


def test_reference_loans_that_no_longer_score_the_same_are_refused(saved: Path) -> None:
    # Stands in for any environment change that moves scores without an error.
    reference = pd.read_parquet(saved / REFERENCE)
    reference["expected_pd"] += 1e-6
    reference.to_parquet(saved / REFERENCE, index=False)
    with pytest.raises(ArtifactError, match="away from their saved PDs"):
        load_artifact(saved)


def test_scoring_a_frame_without_every_feature_fails(
    saved: Path, test_window: pd.DataFrame
) -> None:
    with pytest.raises(ValueError, match=r"lacks model features: \['fico'\]"):
        load_artifact(saved).predict_pd(test_window.drop(columns="fico"))


def test_unseen_categories_are_reported_and_scored(
    saved: Path, family: ModelFamily, test_window: pd.DataFrame
) -> None:
    model = load_artifact(saved)
    loans = test_window.head(2).copy()
    loans["purpose"] = loans["purpose"].cat.add_categories(["wedding", "yacht"])
    loans.loc[loans.index[0], "purpose"] = "wedding"
    assert model.unseen_categories(loans) == {"purpose": ["wedding"]}
    # Every unseen level is treated alike: as missing by the booster, as the
    # infrequent bucket (or no level) by the logistic regression.
    other = loans.head(1).copy()
    other["purpose"] = "yacht"
    assert model.predict_pd(loans)[0] == model.predict_pd(other)[0]
    if family is ModelFamily.GRADIENT_BOOSTING:
        as_missing = loans.head(1).assign(purpose=np.nan)
        assert model.predict_pd(loans)[0] == model.predict_pd(as_missing)[0]


def test_categories_come_from_training_not_the_scored_window(
    tmp_path: Path, train: pd.DataFrame, test_window: pd.DataFrame
) -> None:
    # A level that only exists after training is unseen, whatever window the
    # artifact's reference loans and test scores were drawn from.
    later = test_window.copy()
    later["purpose"] = later["purpose"].cat.add_categories("wedding")
    later.loc[later.index[:50], "purpose"] = "wedding"
    model = fit_baseline(train, feature_names())
    manifest = load_artifact(_save(model, tmp_path, later, BASELINE_PROVENANCE)).manifest
    assert "wedding" not in manifest.categories["purpose"]
    assert manifest.categories["purpose"] == model.categories["purpose"]


def test_a_model_of_another_family_than_the_manifest_says_is_refused(
    saved: Path, family: ModelFamily
) -> None:
    other = (
        GradientBoostingDetails(
            family=ModelFamily.GRADIENT_BOOSTING, trees=1, calibration=Method.PLATT, base_log_odds=0
        )
        if family is ModelFamily.LOGISTIC_REGRESSION
        else LogisticRegressionDetails(
            family=ModelFamily.LOGISTIC_REGRESSION,
            regularization_c=1,
            intercept=0,
            inputs=1,
            train_default_rate=0.2,
        )
    )
    _edit_manifest(saved, model=other.model_dump(mode="json"))
    with pytest.raises(ArtifactError, match=f"is not the {other.family} model"):
        load_artifact(saved)


def test_family_details_are_validated_against_the_named_family(saved: Path) -> None:
    # A logistic-regression block cannot carry trees, nor a booster block an intercept.
    model = json.loads((saved / MANIFEST).read_text())["model"]
    _edit_manifest(saved, model={**model, "trees": 10, "intercept": 0.0})
    with pytest.raises(ArtifactError, match="unreadable manifest"):
        load_artifact(saved)


def test_degenerate_training_features_are_recorded_and_never_needed_to_score(
    tmp_path: Path, family: ModelFamily, train: pd.DataFrame, test_window: pd.DataFrame
) -> None:
    train = train.assign(mort_acc=np.nan)
    if family is ModelFamily.LOGISTIC_REGRESSION:
        model: FittedModel = fit_baseline(train, feature_names())
    else:
        model = fit_booster(
            train[train["issue_date"] < "2014-07-01"],
            train[train["issue_date"] >= "2014-07-01"],
            feature_names(),
        )
    directory = _save(model, tmp_path, test_window, PROVENANCE[family])
    loaded = load_artifact(directory)
    assert loaded.manifest.dropped_features == {"mort_acc": "all missing"}
    assert "mort_acc" not in loaded.manifest.features
    assert "mort_acc" not in pd.read_parquet(directory / REFERENCE).columns
    scored = loaded.predict_pd(test_window.drop(columns="mort_acc"))
    np.testing.assert_array_equal(scored, model.predict_pd(test_window))


# Family-specific metadata.


@pytest.mark.parametrize("family", [ModelFamily.LOGISTIC_REGRESSION])
def test_logistic_regression_manifest_records_only_its_own_parameters(
    saved: Path, baseline: FittedBaseline
) -> None:
    loaded = load_artifact(saved)
    details = loaded.manifest.model
    assert isinstance(details, LogisticRegressionDetails)
    regression = baseline.pipeline.named_steps["model"]
    assert details.regularization_c == regression.C
    assert details.intercept == pytest.approx(regression.intercept_[0])
    assert details.inputs == len(baseline.coefficients())
    assert details.train_default_rate == baseline.train_default_rate
    assert loaded.manifest.provenance.calibration_months is None
    written = json.loads((saved / MANIFEST).read_text())["model"]
    assert set(written) == {"family", "regularization_c", "intercept", "inputs",
                            "train_default_rate"}  # fmt: skip


@pytest.mark.parametrize("family", [ModelFamily.GRADIENT_BOOSTING])
def test_gradient_boosting_manifest_records_trees_calibration_and_shap_base(
    saved: Path, booster: FittedBooster, test_window: pd.DataFrame
) -> None:
    loaded = load_artifact(saved)
    details = loaded.manifest.model
    assert isinstance(details, GradientBoostingDetails)
    assert details.trees == booster.trees
    assert details.calibration == booster.calibrator.method
    assert loaded.explain(test_window.head(3)).base_value == details.base_log_odds
    written = json.loads((saved / MANIFEST).read_text())["model"]
    assert set(written) == {"family", "trees", "calibration", "base_log_odds"}
