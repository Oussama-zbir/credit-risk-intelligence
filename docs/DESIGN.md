# Design: models, artifact, service, container, drift

The README is the overview. This document holds the reasoning behind each stage
after the data layer, which is covered in [`DATA.md`](DATA.md). Measured
results are in [`RESULTS.md`](RESULTS.md).

## Models

### Feature windows and features left out

Training windows are required arguments: they depend on which vintages are
mature at the snapshot, which the data report shows. Every model is fitted
twice on the same split — applicant features only, and applicant features plus
Lending Club's grade and interest rate — and `model_report.md` puts the
logistic baseline, the raw booster and the recalibrated booster side by side,
with a reliability table (mean PD against observed default rate, per decile)
for each.

A requested feature with no variation in the rows a model is fitted on —
missing everywhere, or a single level — is left out of that model and listed
under "Features left out" in the report. The rule is
`nunique(dropna=False) <= 1` on the fitting rows only, so one observed value
plus missing values is kept (missingness can carry risk) and the test window
never decides; details in
[`DATA.md`](DATA.md#features-with-no-variation-in-a-fitting-window).

### Logistic-regression baseline

The model is deliberately plain: `log1p` on monetary amounts, median
imputation plus a missingness indicator, standardisation, one-hot categories
with rare and unseen levels pooled, L2-regularised logistic regression. Every
preprocessing statistic is fitted on the training window only. Ranking (AUC,
Gini, KS) and probability quality (Brier, log loss) are reported separately,
and Brier skill is measured against predicting the *training* default rate, so
a shift in the base rate between windows costs the model skill rather than
being absorbed into the reference.

### Gradient-boosting challenger and calibration

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

### Explanations and reason codes

The report explains the booster on a sample of the test window: mean absolute
TreeSHAP contribution per feature, and reason codes for the riskiest loans —
only features that *raised* the PD, largest first, as a decline notice would
cite them. TreeSHAP is implemented in `explain.py` directly over the fitted
trees rather than through the `shap` package, which would add a compiled numba
stack for one algorithm. That means reading scikit-learn's private tree arrays,
so every call checks that base value plus contributions reproduces the model's
log-odds and raises if not; the tests also compare against brute-force Shapley
values over every coalition, and check that constrained features'
contributions move only in their constrained direction. Contributions are in
the booster's log-odds: Platt calibration rescales them all by one positive
factor, so the reasons and their order do not change.

The logistic-regression champion is explained with exact training-centred
additive log-odds contributions — no attribution method is needed, because its
log-odds are already a sum. Each transformed input contributes
`beta_j * (x_j - training_mean_j)`, the base value is
`intercept + beta . training_mean`, and `explain` credits each term back to its
business feature — a numeric value with its missing-value indicator, a
monetary amount after `log1p` and scaling, every one-hot column of a
categorical — and checks the same identity against `decision_function`.
Features dropped at fit time have no terms, so they never appear. Every term is
measured from its mean over the training rows, stored with the model at fit
time, so a contribution is relative to the training population: a categorical
level is cited only if it puts this applicant above that population, not
merely because its coefficient is positive. Reason codes are the positive
contributions, largest first, shown with the applicant's own values: model
reason codes for review, not a compliant adverse-action notice.

## Model artifact

`train --artifact-dir DIR --artifact-model {logistic-regression,gradient-boosting}`
saves one applicant-only model — the deployable kind, since grade and interest
rate are set by the lender after scoring — as `DIR/<model_id>/`: the pickled
model, a `manifest.json` and a `reference.parquet` of 200 test loans (fitted
features only) with the PDs the model gave them. `logistic-regression` saves
the baseline fitted on the whole training window; `gradient-boosting` saves the
booster with its calibrator.

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
hash prefix, and an existing id is never overwritten. The directory is staged
and renamed into place, so a reader never sees a half-written artifact.

`load_artifact` refuses to serve rather than serve wrong: the manifest must be
format 3 (formats 1 and 2 are refused, not migrated) and the pickle the family
and features it describes, the model file must match the manifest's hash,
scikit-learn must be the version it was saved with (its pickles are not
portable across releases), and the reference loans must reproduce their saved
PDs to 1e-9 — the check that catches a numpy, pandas or encoding change that
would move scores without raising. Categorical levels unseen in training are
scored (as missing by the booster; pooled with infrequent levels, or as no
level, by the logistic regression) and reported by `unseen_categories`, for a
service to log as drift. The hash detects corruption, not tampering:
unpickling runs code, so artifacts must come from a store only training
writes.

## Scoring service

Every response carries `model_id` and `model_family`, so any decision can be
traced back to its manifest.

- **Verified at startup, not per request.** The artifact goes through
  `load_artifact` (hash, scikit-learn version, reference-PD replay) in the
  app's lifespan. If any check fails, uvicorn exits before it binds; it never
  serves a model it cannot vouch for. A format-2 artifact, for example, stops
  startup with `artifact format 2 is not supported`.
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

## Container

- **The model is mounted, not baked in.** One image serves any artifact, a new
  model is a new mount rather than a new build, and the image holds no data.
  It runs as a non-root user with a read-only root filesystem. Artifacts
  saved before `save_artifact` set the directory to 0755 are 0700, which that
  user cannot read on Linux; `chmod 755` them once.
- **Training and serving share pinned versions.** An artifact refuses to load
  under a scikit-learn other than the one that saved it, so the image installs
  from `constraints.txt`. Train with `pip install -e . -c constraints.txt` to
  get an artifact the image will serve; anything else stops startup with
  `model saved with scikit-learn X, Y is installed`. The pins are the
  champion's training versions. CI's quality job stays unpinned so upstream
  releases are still tested.
- **A failed check is a failed container.** uvicorn runs with `--lifespan on`,
  so an artifact that does not verify exits the process (code 3) before it
  binds, and an orchestrator sees a crash, not a healthy service without a
  model.

`scripts/container_smoke.sh` is the CI check. It trains a synthetic artifact
with the pinned versions, serves it from the image (read-only filesystem and
mount), checks `/health` and a `/v1/score` call, then serves a copy whose
`model.pkl` no longer matches its manifest and requires the container to
refuse to start. The image is about 900 MB, almost all numpy, scipy, pandas
and pyarrow wheels.

## Drift monitoring

The reference is the artifact's own training window, read from the processed
file whose SHA-256 the manifest pins; another file is refused, and so is a
monitored window that overlaps training. `drift_report.md` answers three
questions:

- **Did the score move?** PSI of the PD on the training PD deciles, overall and
  per quarter, beside mean PD and, when outcomes exist, the default rate.
- **Which inputs moved?** PSI of every fitted feature. Bins are fitted on the
  training window and frozen: deciles for numbers (fewer when values tie, like
  term), one bin per training level for categories, and separate `missing`
  and `unseen` bins, so a feed that stops sending a field shows up as drift
  instead of being folded into a value bin.
- **Which moves pushed the score?** The model's explanation is additive in
  log-odds, so the change in a feature's mean contribution between the windows
  is exactly its share of the change in mean log-odds. A feature can move a
  lot and barely matter, or move a little on a large coefficient; this table
  tells them apart.

PSI needs no labels, so it is the signal available on the day of application.
The 0.10 / 0.25 bands are a rule of thumb, not a test, and on the real champion
the score PSI stayed "stable" while calibration slipped by 2 points
([findings](RESULTS.md#drift-2013-against-the-training-window)). The service
already returns unseen categorical levels per request; this report counts them
over a window.
