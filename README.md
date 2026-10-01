# Credit Risk Intelligence

[![CI](https://github.com/Oussama-zbir/credit-risk-intelligence/actions/workflows/ci.yml/badge.svg)](https://github.com/Oussama-zbir/credit-risk-intelligence/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](pyproject.toml)
[![mypy strict](https://img.shields.io/badge/mypy-strict-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Probability-of-default modelling on public Lending Club data, built the way a
lender would need it: a leakage-audited feature contract, an honest default
definition, out-of-time validation, a logistic-regression baseline and
calibrated gradient boosting measured against it — and, in later milestones,
explainability, serving and drift monitoring.

> **Status: model comparison.** The data contract, default definition,
> out-of-time split, preprocessing pipeline, a logistic-regression baseline and
> a calibrated gradient-boosting challenger exist and are tested on synthetic
> rows. No results over
> the real file are published yet, so this README quotes no metrics.

## Why the data layer comes first

It is easy to get a near-perfect AUC on this dataset, and the number means
nothing: it comes from two mistakes this repository is built to prevent.

1. **Target leakage.** The file includes columns written *after* a loan was
   issued — payments received, recoveries, settlement flags. `recoveries` is
   non-zero only for charged-off loans; a model that sees it is reading the
   label. Here every column is declared with a role in
   [`contract.py`](src/credit_risk/data/contract.py), and post-origination
   columns are never loaded.
2. **Right-censored labels.** Dropping loans still `Current` leaves recent
   vintages dominated by early defaulters and prepayers. Here a loan is
   labelled only once its full term has elapsed by the snapshot
   ([`target.py`](src/credit_risk/data/target.py)).

Validation is **out-of-time** — train on earlier vintages, test on a strictly
later window, with an optional deployment gap
([`split.py`](src/credit_risk/data/split.py)). Lending Club's own grade and
interest rate are kept as an opt-in feature group, so the uplift from
re-learning the lender's scorecard is measured rather than hidden.

The reasoning behind each choice — dataset selection, default definition,
observation window, leakage audit — is in [`docs/DATA.md`](docs/DATA.md).

## Layout

```
src/credit_risk/data/
  contract.py     every column, its role, and why
  validation.py   ingestion checks: errors stop the pipeline, warnings are counted
  target.py       default definition + maturity window, with a LabelReport
  split.py        out-of-time split between issue-date windows
  loader.py       reads contract columns only; snapshot from the file itself
  features.py     row-wise typed features: nothing fitted, so no split leakage
  report.py       population, vintages, warnings and missingness as Markdown
src/credit_risk/models/
  baseline.py     logistic regression; every preprocessing statistic fitted on train only
  boosting.py     monotone-constrained gradient boosting, early-stopped on a later slice
  calibration.py  Platt / isotonic recalibration fitted on that slice, never on test
  metrics.py      AUC, Gini, KS (ranking); Brier, log loss, reliability bins (probabilities)
  explain.py      exact TreeSHAP over the booster's trees; adverse-action reason codes
src/credit_risk/prepare.py   raw export -> data/processed/loans.parquet + data_report.md
src/credit_risk/train.py     out-of-time fit + comparison -> model_report.md
```

## Preparing the data

Download `accepted_2007_to_2018Q4.csv.gz` from the
[Lending Club dataset on Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
into `data/`, then:

```bash
python -m credit_risk.prepare data/accepted_2007_to_2018Q4.csv.gz
```

The pipeline reads only contract columns, stops on any contract error, labels
matured loans, builds typed features and writes a Parquet frame plus a data
report recording the file's SHA-256, the snapshot date and the rows lost at
each stage.

## Training and comparing models

```bash
python -m credit_risk.train data/processed/loans.parquet \
    --train 2012-01:2014-12 --test 2015-07:2016-03 \
    --calibration-months 6 --calibration platt
```

Windows are required arguments: they depend on which vintages are mature at
the snapshot, which the data report shows. Every model is fitted twice on the
same split — applicant features only, and applicant features plus Lending
Club's grade and interest rate — and `model_report.md` puts the logistic
baseline, the raw booster and the recalibrated booster side by side, with a
reliability table (mean PD against observed default rate, per decile) for each.

The model is deliberately plain: `log1p` on monetary amounts, median
imputation plus a missingness indicator, standardisation, one-hot categories
with rare and unseen levels pooled, L2-regularised logistic regression. Ranking
(AUC, Gini, KS) and probability quality (Brier, log loss) are reported
separately, and Brier skill is measured against predicting the *training*
default rate, so a shift in the base rate between windows costs the model
skill rather than being absorbed into the reference.

The challenger is scikit-learn's histogram gradient boosting: missing values
and categories handled natively, and PD constrained to move in one direction
with FICO, DTI, revolving utilisation, recent inquiries and interest rate, so
no decline can be explained by "a higher score raised your risk". The last
`--calibration-months` of the training window are held out, in time order: the
number of trees is early-stopped on them and the calibrator is fitted on them,
so the test window is never used for either. Platt is the default because it
is stable with few defaults and cannot reorder loans; isotonic corrects any
shape of miscalibration but overfits a small slice. Whether recalibration
helps at all is an empirical question the report answers per run — a booster
trained on log loss is often close to calibrated already.

The report then explains the booster on a sample of the test window: mean
absolute TreeSHAP contribution per feature, and reason codes for the riskiest
loans — only features that *raised* the PD, largest first, as a decline notice
would cite them. TreeSHAP is implemented in `explain.py` directly over the
fitted trees rather than through the `shap` package, which would add a compiled
numba stack for one algorithm. That means reading scikit-learn's private tree
arrays, so every call checks that base value plus contributions reproduces the
model's log-odds and raises if not; the tests also compare against brute-force
Shapley values over every coalition, and check that constrained features'
contributions move only in their constrained direction. Contributions are in
the booster's log-odds: Platt calibration rescales them all by one positive
factor, so the reasons and their order do not change.

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
ruff check . && ruff format --check . && mypy && pytest
```

Tests use synthetic rows shaped like the Lending Club export (including its
`'13.56%'` and `' 36 months'` string formats); no data download is needed. The
real file goes in `data/`, which is git-ignored.

## Roadmap

1. ~~Data contract, default definition, out-of-time split~~
2. ~~Reproducible preprocessing pipeline, with a data report~~ (committed report
   over the real file pending)
3. ~~Logistic-regression baseline: AUC/Gini, KS, Brier on the out-of-time
   window~~ (published results over the real file pending)
4. ~~Gradient boosting against that baseline, probability calibration~~
   (published results over the real file pending)
5. ~~SHAP explanations: global importance and per-applicant reason codes~~
6. FastAPI scoring behind a versioned model artifact
7. PSI / drift monitoring across vintages

## Limitations

Lending Club is one lender's *accepted* population: there are no rejected
applicants, so no reject inference, and income is largely self-reported.
