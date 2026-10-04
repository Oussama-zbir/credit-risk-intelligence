from dataclasses import replace

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from conftest import make_processed
from credit_risk.data.features import feature_names
from credit_risk.data.split import OutOfTimeSplit, TimeWindow, out_of_time_split
from credit_risk.models import explain as explain_module
from credit_risk.models.baseline import FittedBaseline, fit_baseline
from credit_risk.models.explain import REASONS, ExplanationError, explain, reason_codes

APPLICANT = feature_names()


@pytest.fixture(scope="module")
def split() -> OutOfTimeSplit:
    return out_of_time_split(
        make_processed(),
        train=TimeWindow(pd.Timestamp("2012-01-01"), pd.Timestamp("2014-12-01")),
        test=TimeWindow(pd.Timestamp("2015-07-01"), pd.Timestamp("2016-12-01")),
    )


@pytest.fixture(scope="module")
def baseline(split: OutOfTimeSplit) -> FittedBaseline:
    return fit_baseline(split.train, APPLICANT)


@pytest.fixture(scope="module")
def loans(split: OutOfTimeSplit) -> pd.DataFrame:
    loans = split.test.head(300).copy()
    loans["purpose"] = loans["purpose"].astype(object)
    loans.loc[loans.index[:2], "purpose"] = ["wedding", None]  # unseen, missing
    loans.loc[loans.index[2:5], ["dti", "fico", "revol_util"]] = np.nan
    loans.loc[loans.index[5], ["fico", "dti"]] = [620.0, 39.0]
    return loans


def _coef(baseline: FittedBaseline) -> pd.Series:
    names = baseline.pipeline.named_steps["columns"].get_feature_names_out()
    return pd.Series(baseline.pipeline.named_steps["model"].coef_[0], index=names)


def _transformed(baseline: FittedBaseline, loans: pd.DataFrame) -> pd.DataFrame:
    columns = baseline.pipeline.named_steps["columns"]
    return pd.DataFrame(
        columns.transform(loans[list(baseline.features)]),
        index=loans.index,
        columns=columns.get_feature_names_out(),
    )


@pytest.fixture(scope="module")
def train_means(split: OutOfTimeSplit, baseline: FittedBaseline) -> pd.Series:
    """Transformed training-input means, recomputed here from the training rows."""
    return _transformed(baseline, split.train).mean()


def test_training_input_means_are_stored_in_coefficient_order(
    baseline: FittedBaseline, train_means: pd.Series
) -> None:
    assert baseline.input_names == tuple(_coef(baseline).index)
    np.testing.assert_allclose(baseline.input_means, train_means, rtol=0, atol=1e-12)


def test_input_means_that_do_not_match_the_pipeline_are_refused(
    baseline: FittedBaseline,
) -> None:
    with pytest.raises(ValueError, match="input means"):
        replace(baseline, input_means=baseline.input_means[:-1])
    with pytest.raises(ValueError, match="input means"):
        replace(baseline, input_names=tuple(reversed(baseline.input_names)))


def test_contributions_add_up_exactly_to_the_model_log_odds(
    baseline: FittedBaseline, loans: pd.DataFrame, train_means: pd.Series
) -> None:
    result = explain(baseline, loans)
    expected = baseline.pipeline.decision_function(loans[list(baseline.features)])
    np.testing.assert_allclose(result.log_odds, expected, rtol=0, atol=1e-12)
    intercept = baseline.pipeline.named_steps["model"].intercept_[0]
    assert result.base_value == pytest.approx(intercept + _coef(baseline) @ train_means, abs=1e-12)
    assert tuple(result.contributions.columns) == baseline.features
    assert result.contributions.index.equals(loans.index)


def test_contributions_average_zero_over_the_training_population(
    split: OutOfTimeSplit, baseline: FittedBaseline
) -> None:
    result = explain(baseline, split.train)
    np.testing.assert_allclose(result.contributions.mean(), 0.0, rtol=0, atol=1e-10)
    # So the base value is the mean training log-odds.
    assert result.base_value == pytest.approx(result.log_odds.mean(), abs=1e-10)


def test_every_transformed_input_is_credited_to_exactly_one_business_feature(
    baseline: FittedBaseline, loans: pd.DataFrame, train_means: pd.Series
) -> None:
    # Summing over features loses nothing and double-counts nothing.
    terms = (_transformed(baseline, loans) - train_means) * _coef(baseline)
    result = explain(baseline, loans)
    np.testing.assert_allclose(
        result.contributions.sum(axis=1), terms.sum(axis=1), rtol=0, atol=1e-12
    )


def test_a_numeric_value_and_its_missing_indicator_are_one_feature(
    baseline: FittedBaseline, loans: pd.DataFrame, train_means: pd.Series
) -> None:
    coef, x = _coef(baseline), _transformed(baseline, loans) - train_means
    assert "missingindicator_revol_util" in coef.index
    expected = (
        x["revol_util"] * coef["revol_util"]
        + x["missingindicator_revol_util"] * coef["missingindicator_revol_util"]
    )
    contribution = explain(baseline, loans).contributions["revol_util"]
    np.testing.assert_allclose(contribution, expected, rtol=0, atol=1e-12)
    # Reporting the field missing moves the contribution through the indicator.
    missing = loans["revol_util"].isna()
    assert missing.any()
    assert contribution[missing].nunique() == 1
    assert "missingindicator_revol_util" not in explain(baseline, loans).contributions


def test_a_monetary_amount_is_explained_on_its_original_feature(
    split: OutOfTimeSplit, baseline: FittedBaseline, loans: pd.DataFrame
) -> None:
    # Recomputed from the fitted scaler, independently of the column transformer.
    columns = baseline.pipeline.named_steps["columns"]
    names = next(cols for name, _, cols in columns.transformers_ if name == "monetary")
    scaler = columns.named_transformers_["monetary"][-1]
    i = list(names).index("annual_inc")
    mean, scale = float(scaler.mean_[i]), float(scaler.scale_[i])

    def standardised(amounts: pd.Series) -> npt.NDArray[np.float64]:
        return (np.log1p(amounts.to_numpy(dtype=np.float64)) - mean) / scale

    centre = standardised(split.train["annual_inc"]).mean()
    expected = _coef(baseline)["annual_inc"] * (standardised(loans["annual_inc"]) - centre)
    contribution = explain(baseline, loans).contributions["annual_inc"]
    np.testing.assert_allclose(contribution, expected, rtol=0, atol=1e-12)


def test_a_categorical_is_measured_from_the_training_level_frequencies(
    split: OutOfTimeSplit, baseline: FittedBaseline, loans: pd.DataFrame
) -> None:
    coef = _coef(baseline)
    frequency = split.train["home_ownership"].astype(str).value_counts(normalize=True)
    typical = sum(coef[f"home_ownership_{level}"] * share for level, share in frequency.items())
    contributions = explain(baseline, loans).contributions
    for index in loans.index[5:20]:
        level = loans.at[index, "home_ownership"]
        assert contributions.at[index, "home_ownership"] == pytest.approx(
            coef[f"home_ownership_{level}"] - typical, abs=1e-12
        )
    # A level never seen in training sets no one-hot column: it sits at
    # minus the typical level's effect, and the decomposition stays exact.
    purpose = split.train["purpose"].astype(str).value_counts(normalize=True)
    typical_purpose = sum(coef[f"purpose_{level}"] * share for level, share in purpose.items())
    assert "purpose_wedding" not in coef.index
    assert contributions.at[loans.index[0], "purpose"] == pytest.approx(-typical_purpose)
    unseen = loans.head(1)
    np.testing.assert_allclose(
        explain(baseline, unseen).log_odds,
        baseline.pipeline.decision_function(unseen[list(baseline.features)]),
        rtol=0,
        atol=1e-12,
    )
    assert not any(name.startswith("purpose_") for name in contributions.columns)


def test_an_unseen_level_follows_the_encoders_infrequent_bucket(
    split: OutOfTimeSplit, loans: pd.DataFrame
) -> None:
    # A level rarer than MIN_CATEGORY_SHARE gets an infrequent column, and the
    # encoder sends unseen levels there too: not to an all-zero encoding.
    train = split.train.copy()
    train["purpose"] = train["purpose"].astype(object)
    train.loc[train.index[:5], "purpose"] = "moving"  # well under 0.5% of training loans
    model = fit_baseline(train, APPLICANT)
    coef = _coef(model)
    assert "purpose_infrequent_sklearn" in coef.index

    applicant = loans.head(1).copy()
    applicant["purpose"] = applicant["purpose"].astype(object)
    unseen = applicant.assign(purpose="wedding")
    rare = applicant.assign(purpose="moving")
    x = _transformed(model, unseen)
    assert x.at[unseen.index[0], "purpose_infrequent_sklearn"] == 1.0
    pd.testing.assert_frame_equal(x, _transformed(model, rare))

    means = pd.Series(model.input_means, index=model.input_names)
    levels = [name for name in coef.index if name.startswith("purpose_")]
    expected = sum(coef[n] * (x.at[unseen.index[0], n] - means[n]) for n in levels)
    result = explain(model, unseen)
    contribution = result.contributions.at[unseen.index[0], "purpose"]
    assert contribution == pytest.approx(expected, abs=1e-12)
    assert contribution == explain(model, rare).contributions.at[rare.index[0], "purpose"]
    all_zero = -sum(coef[n] * means[n] for n in levels)
    assert contribution != pytest.approx(all_zero)  # the bucket's coefficient counts
    np.testing.assert_allclose(
        result.log_odds,
        model.pipeline.decision_function(unseen[list(model.features)]),
        rtol=0,
        atol=1e-12,
    )


def test_dropped_features_never_appear_in_explanations_or_reasons(
    split: OutOfTimeSplit, loans: pd.DataFrame
) -> None:
    train = split.train.assign(mort_acc=np.nan)
    train["application_type"] = pd.Categorical(["Individual"] * len(train))
    reduced = fit_baseline(train, APPLICANT)
    assert set(reduced.dropped) == {"mort_acc", "application_type"}

    scored = loans.drop(columns=["mort_acc", "application_type"])
    result = explain(reduced, scored)
    assert tuple(result.contributions.columns) == reduced.features
    assert {"mort_acc", "application_type"}.isdisjoint(result.contributions.columns)
    cited = {r.feature for reasons in reason_codes(result, scored) for r in reasons}
    assert {"mort_acc", "application_type"}.isdisjoint(cited)


def test_reason_codes_cite_only_adverse_contributions_largest_first(
    baseline: FittedBaseline, loans: pd.DataFrame
) -> None:
    result = explain(baseline, loans)
    codes = reason_codes(result, loans)
    assert len(codes) == len(loans)
    for index, reasons in zip(loans.index, codes, strict=True):
        assert len(reasons) <= REASONS
        contributions = [r.contribution for r in reasons]
        assert all(c > 0 for c in contributions)
        assert contributions == sorted(contributions, reverse=True)
        adverse = result.contributions.loc[index]
        assert len(reasons) == min(REASONS, int((adverse > 0).sum()))


def test_reasons_show_the_applicants_own_values_not_encoded_ones(
    baseline: FittedBaseline, loans: pd.DataFrame
) -> None:
    codes = reason_codes(explain(baseline, loans), loans)
    for index, reasons in zip(loans.index, codes, strict=True):
        for reason in reasons:
            raw = loans.at[index, reason.feature]
            assert (pd.isna(raw) and pd.isna(reason.value)) or reason.value == raw
    # A low FICO score and a high DTI are this model's adverse drivers.
    risky = [str(r) for r in codes[5]]
    assert "FICO score (620)" in risky
    assert "debt-to-income ratio (39)" in risky


def test_an_input_that_belongs_to_no_feature_raises(baseline: FittedBaseline) -> None:
    with pytest.raises(ExplanationError, match="no source feature"):
        explain_module._sources(["mystery_input"], baseline)
