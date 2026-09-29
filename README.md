# Credit Risk Intelligence

[![CI](https://github.com/Oussama-zbir/credit-risk-intelligence/actions/workflows/ci.yml/badge.svg)](https://github.com/Oussama-zbir/credit-risk-intelligence/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](pyproject.toml)
[![mypy strict](https://img.shields.io/badge/mypy-strict-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Probability-of-default modelling on public Lending Club data, built the way a
lender would need it: a leakage-audited feature contract, an honest default
definition, out-of-time validation, and — in later milestones — a baseline
scorecard, calibrated gradient boosting, explainability, serving and drift
monitoring.

> **Status: baseline model.** The data contract, default definition,
> out-of-time split, preprocessing pipeline and a logistic-regression baseline
> with its evaluation exist and are tested on synthetic rows. No results over
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
  metrics.py      AUC, Gini, KS (ranking) and Brier, Brier skill, log loss (probabilities)
src/credit_risk/prepare.py   raw export -> data/processed/loans.parquet + data_report.md
src/credit_risk/train.py     out-of-time fit + evaluation -> baseline_report.md
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

## Training the baseline

```bash
python -m credit_risk.train data/processed/loans.parquet \
    --train 2012-01:2014-12 --test 2015-07:2016-03
```

Windows are required arguments: they depend on which vintages are mature at
the snapshot, which the data report shows. Two models are fitted on the same
split — applicant features only, and applicant features plus Lending Club's
grade and interest rate — and the report puts them side by side.

The model is deliberately plain: `log1p` on monetary amounts, median
imputation plus a missingness indicator, standardisation, one-hot categories
with rare and unseen levels pooled, L2-regularised logistic regression. Ranking
(AUC, Gini, KS) and probability quality (Brier, log loss) are reported
separately, and Brier skill is measured against predicting the *training*
default rate, so a shift in the base rate between windows costs the model
skill rather than being absorbed into the reference.

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
4. Gradient boosting against that baseline, probability calibration, SHAP
5. FastAPI scoring behind a versioned model artifact
6. PSI / drift monitoring across vintages

## Limitations

Lending Club is one lender's *accepted* population: there are no rejected
applicants, so no reject inference, and income is largely self-reported.
