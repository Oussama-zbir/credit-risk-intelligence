# Credit Risk Intelligence

[![CI](https://github.com/Oussama-zbir/credit-risk-intelligence/actions/workflows/ci.yml/badge.svg)](https://github.com/Oussama-zbir/credit-risk-intelligence/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](pyproject.toml)
[![mypy strict](https://img.shields.io/badge/mypy-strict-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Probability-of-default modelling on public Lending Club data, built the way a
lender would need it: a leakage-audited feature contract, an honest default
definition, out-of-time validation, a logistic-regression baseline and
calibrated gradient boosting measured against it, exact model reason codes and a
versioned model artifact served over HTTP — and, in a later milestone, drift
monitoring.

> **Status: scoring service.** The data contract, default definition,
> out-of-time split, preprocessing pipeline, a logistic-regression baseline, a
> calibrated gradient-boosting challenger, its explanations, a self-verifying
> model artifact and a FastAPI service over it exist and are tested on
> synthetic rows. No
> results over the real file are published yet, so this README quotes no
> metrics.

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
  cohort.py       modelling population: drops pre-2011 loans outside the credit policy, counted
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
  degenerate.py   features with no variation in a model's fitting rows, left out of that model
  metrics.py      AUC, Gini, KS (ranking); Brier, log loss, reliability bins (probabilities)
  explain.py      exact log-odds contributions (training-centred for LR, TreeSHAP for GBM)
  artifact.py     versioned model (either family) on disk: manifest, hash, reference-PD checks
src/credit_risk/prepare.py   raw export -> data/processed/loans.parquet + data_report.md
src/credit_risk/train.py     out-of-time fit + comparison -> model_report.md
src/credit_risk/service.py   FastAPI: PD + reason codes from one verified artifact
```

## Preparing the data

Download `accepted_2007_to_2018Q4.csv.gz` from the
[Lending Club dataset on Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
into `data/`, then:

```bash
python -m credit_risk.prepare data/accepted_2007_to_2018Q4.csv.gz
```

The pipeline reads only contract columns, removes the 2007–2010 loans issued
outside Lending Club's credit policy (counted in the report), stops on any
contract error in what remains, labels matured loans, builds typed features
and writes a Parquet frame plus a data report recording the file's SHA-256,
the snapshot date and the rows lost at each stage.

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

A requested feature with no variation in the rows a model is fitted on —
missing everywhere, or a single level — is left out of that model and listed
under "Features left out" in the report. The rule is
`nunique(dropna=False) <= 1` on the fitting rows only, so one observed value
plus missing values is kept (missingness can carry risk) and the test window
never decides; details in [`docs/DATA.md`](docs/DATA.md#features-with-no-variation-in-a-fitting-window).

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

The logistic-regression champion is explained with exact training-centred
additive log-odds contributions — no attribution method is needed, because its
log-odds are already a sum. Each transformed input contributes
`beta_j * (x_j - training_mean_j)`, the base value is
`intercept + beta . training_mean`, and `explain` credits each term back to its
business feature — a numeric value with its
missing-value indicator, a monetary amount after `log1p` and scaling, every
one-hot column of a categorical — and checks the same identity against
`decision_function`. Features dropped at fit time have no terms, so they never
appear. Every term is measured from its mean over the training rows, stored
with the model at fit time, so a contribution is relative to the training
population: a categorical level is cited only if it puts this applicant above
that population, not merely because its coefficient is positive. Reason codes
are the positive contributions, largest first, shown with the applicant's own
values: model reason codes for review, not a compliant adverse-action notice.

## Model artifact

```bash
python -m credit_risk.train data/processed/loans.parquet \
    --train 2012-01:2014-12 --test 2015-07:2016-03 \
    --artifact-dir artifacts --artifact-model logistic-regression
# or, for the calibrated booster:
#   --artifact-dir artifacts --artifact-model gradient-boosting
```

saves one applicant-only model — the deployable kind, since grade and interest
rate are set by the lender after scoring — as `artifacts/<model_id>/`: the
pickled model, a `manifest.json` and a `reference.parquet` of 200 test loans
(fitted features only) with the PDs the model gave them.
`logistic-regression` saves the baseline fitted on the whole training window;
`gradient-boosting` saves the booster with its calibrator.

**Which model is saved is always an explicit choice; it is never picked
automatically by test AUC.** `--artifact-dir` without `--artifact-model` (or the
reverse) is refused before training starts. A champion is chosen once, on a
development window, and frozen before the confirmation window is scored;
letting the CLI keep whichever model won on the run's own test window would
turn that window into a selection set and its scores into optimistic ones.

The manifest (format 3) names the model family and records what both families
share: the processed file's SHA-256, the train, calibration (booster only) and
test windows, the out-of-time test scores, the fitted features and any
requested ones left out with why, every categorical level seen in training,
and the library versions. Family-specific fields live in a `model` block —
regularisation, intercept and input count for the logistic regression; trees,
calibration method and TreeSHAP base value for the booster — so neither
carries placeholder values for the other. The model id is the model file's
hash prefix, and an existing id is never overwritten.

`load_artifact` refuses to serve rather than serve wrong: the manifest must be
format 3 (formats 1 and 2 are refused, not migrated) and the pickle the
family and features it describes, the model file
must match the manifest's hash, scikit-learn must be the version it was saved
with (its pickles are not portable across releases), and the reference loans
must reproduce their saved PDs to 1e-9 — the check that catches a numpy,
pandas or encoding change that would move scores without raising. Categorical
levels unseen in training are scored (as missing by the booster; pooled with
infrequent levels, or as no level, by the logistic regression) and reported by
`unseen_categories`, for a service to log as drift. The hash detects
corruption, not tampering: unpickling runs code, so artifacts must come from a
store only training writes.

## Scoring service

```bash
pip install -e ".[service]"
CREDIT_RISK_ARTIFACT_DIR=artifacts/<model_id> \
    uvicorn --factory credit_risk.service:app_from_env
```

| endpoint | returns |
| --- | --- |
| `GET /health` | `ok` and the loaded `model_id` |
| `GET /v1/model` | family, features, dropped features, provenance, out-of-time test scores |
| `POST /v1/score` | per application: PD, up to 4 reason codes, unseen categorical levels |

```bash
curl -s localhost:8000/v1/score -H 'content-type: application/json' -d '{"applications": [{
  "loan_amnt": 15000, "term_months": 60, "purpose": "small_business",
  "application_type": "Individual", "annual_inc": 42000,
  "verification_status": "Not Verified", "home_ownership": "RENT", "fico": 662,
  "credit_history_months": 50, "delinq_2yrs": 1, "open_acc": 6, "total_acc": 12,
  "pub_rec": 0, "revol_bal": 9000, "dti": 31.5, "revol_util": 88,
  "inq_last_6mths": 3}]}'
```

Every response carries `model_id` and `model_family`, so any decision can be
traced back to its manifest. Design choices:

- **Verified at startup, not per request.** The artifact goes through
  `load_artifact` (hash, scikit-learn version, reference-PD replay) in the app's
  lifespan. If any check fails, uvicorn exits before it binds; it never serves
  a model it cannot vouch for. A format-2 artifact, for example, stops startup
  with `artifact format 2 is not supported`.
- **The request is the processed applicant feature set**, the output of
  `build_features` (`fico` as the band midpoint, `term_months` as 36 or 60,
  `credit_history_months` at application). It covers all 20 applicant
  features, so a model that drops one still accepts the same request. At
  startup the model's features must be a subset of it, which refuses a model
  trained on lender pricing: grade and rate are set after scoring, so a client
  could not supply them. Unknown fields are a 422, so a misspelt field cannot
  score silently as "not reported". Only the six fields the source leaves
  empty may be null.
- **Unseen categorical levels are scored, returned and logged, not rejected.**
  The model has defined behaviour for them, and a new level appearing in
  volume is drift. Scoring the real champion shows why it matters: an unseen
  `purpose` falls into the logistic regression's infrequent bucket and becomes
  that applicant's top adverse reason.
- **Batches of 1–100 applications**, scored in one vectorised call. Endpoints
  are synchronous: scoring is CPU-bound numpy, so FastAPI runs it in its
  threadpool rather than blocking the event loop.

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
5. ~~Explanations and per-applicant reason codes: TreeSHAP for the booster,
   training-centred additive log-odds contributions for the logistic regression~~
6. ~~Versioned, self-verifying model artifact~~ (format 3: logistic regression or booster)
7. ~~FastAPI scoring service over that artifact: PD + reason codes per applicant~~
8. PSI / drift monitoring across vintages

## Limitations

Lending Club is one lender's *accepted* population: there are no rejected
applicants, so no reject inference, and income is largely self-reported.
