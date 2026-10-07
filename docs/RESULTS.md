# Results on the real Lending Club file

What the pipeline produces on the public export, how the champion was chosen,
and what the numbers do and do not show. The full generated reports are in
[`docs/results/`](results/); this page quotes them and adds the reading.

## Protocol

The champion is chosen once, on a development window, then frozen; only then
is the confirmation window scored.

| step | train | test | purpose |
| --- | --- | --- | --- |
| development | 2010-01 – 2011-12 (33,257 loans) | 2012-01 – 2012-12 (53,367) | compare models, pick the champion |
| confirmation | 2010-01 – 2012-12 (86,624 loans) | 2013-01 – 2013-12 (134,804) | score the frozen choice once |

2007–2009 vintages are left out: under 6,600 labelled loans, all 36-month, and
issued under early underwriting. Windows end at 2013 because a 60-month loan is
labelled only once its term has elapsed by the snapshot (2019-03), so later
vintages hold 36-month loans only (60-month share 7.4% in 2014, 0% from 2015).
In both steps the booster's trees are fitted on all but the last 6 training
months, which are used to early-stop them and fit a Platt calibrator; the
logistic regression is fitted on the whole training window.

## Population

From [`data_report.md`](results/data_report.md): 2,260,668 loans read, 2,749
excluded as issued outside the credit policy, 1,474,753 not yet matured at the
snapshot, 6,889 matured but unresolved, leaving **776,277 labelled loans with a
15.15% default rate**. No contract errors in the modelling cohort.

## Development: 2012 out of time (applicant features)

From [`model_report_dev_2012.md`](results/model_report_dev_2012.md):

| model | train AUC | AUC | Gini | KS | Brier | log loss | mean PD vs 16.20% observed | calib. error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **logistic regression** | 0.7104 | **0.6781** | **0.3561** | **0.2573** | **0.1284** | **0.4167** | 14.32% | 1.88% |
| gradient boosting, raw | 0.7508 | 0.6706 | 0.3413 | 0.2466 | 0.1295 | 0.4209 | 13.95% | 2.37% |
| gradient boosting + Platt | 0.7508 | 0.6706 | 0.3413 | 0.2466 | 0.1291 | 0.4187 | 15.18% | **1.40%** |

**Champion: the logistic regression.** It wins on ranking (AUC, Gini, KS) and on
both proper scoring rules; the calibrated booster is better only on binned
calibration error. The booster's train-to-test AUC drop (0.080) is two and a
half times the baseline's (0.032): with 2 years of data and monotone
constraints, the extra capacity fitted the training vintages, not the future.
The baseline is also simpler to explain exactly and to audit, so a tie would
have gone to it as well.

## Confirmation: 2013 out of time (frozen choice)

From [`model_report_confirm_2013.md`](results/model_report_confirm_2013.md):

| model | AUC | Gini | KS | Brier | log loss | mean PD vs 15.60% observed | calib. error |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **logistic regression (champion)** | **0.6744** | **0.3489** | **0.2533** | **0.1254** | **0.4089** | 17.51% | 1.91% |
| gradient boosting + Platt | 0.6717 | 0.3434 | 0.2474 | 0.1254 | 0.4090 | 17.01% | 1.41% |

The ranking holds on a window 2.5 times larger and one year later: AUC moves
from 0.6781 to 0.6744. This is the run that produced the deployed artifact
`699a62529632`; its manifest records the same test scores and the processed
file's SHA-256.

## What the numbers say

- **An AUC near 0.67 is the honest number.** Only application-time columns are
  loaded and loans are labelled only after maturity. A model on this file that
  reports an AUC in the 0.9s is usually reading post-origination columns
  such as `recoveries` or `total_rec_prncp`.
- **Lender pricing adds little that can be deployed.** Adding Lending Club's own
  grade and interest rate moves the logistic regression's AUC by +0.001 in
  development and +0.011 in confirmation, and makes the booster worse in
  development (0.6668). Grade and rate are set by the lender after scoring, so
  the service refuses a model that needs them; the comparison only measures how
  much of the lender's scorecard the applicant features already recover.
- **Calibration drifts between windows in both directions.** In 2012 the
  observed default rate rose to 16.20% from 14.38% in training and the
  baseline under-predicted (mean PD 14.32%). In 2013 the default rate barely
  moved (15.50% to 15.60%) yet the champion over-predicted (17.51%), in every
  decile and most in the top one (37.89% predicted, 32.29% observed). 2013
  loans looked riskier on their inputs than they turned out to be, and a
  stable default rate would not have flagged it. Which inputs moved is
  measured in [Drift: 2013 against the training window](#drift-2013-against-the-training-window).
- **Recalibration fixes level, not ranking.** Platt scaling cuts the booster's
  calibration error from 2.76% to 1.41% in 2013 and leaves AUC unchanged, as a
  monotone transform must.
- **Degenerate features are window-dependent.** `mort_acc` is missing for every
  2010–2011 loan, so the development models leave it out; the confirmation
  models, whose training window includes 2012, use it. `application_type` is
  constant `Individual` in both windows and is left out of every model.

## Drift: 2013 against the training window

From [`drift_report_2013.md`](results/drift_report_2013.md): the champion
artifact `699a62529632` against its own training window (2010-01 – 2012-12,
86,624 loans), monitored on the 134,804 loans issued in 2013. No outcomes are
needed for any PSI; the default rate is shown only because these loans have
matured.

| signal | value | reading |
| --- | --- | --- |
| score PSI (PD on training deciles) | 0.0668 | stable by the 0.10 rule of thumb, in every quarter (0.061–0.077) |
| change in mean log-odds | +0.174 | mean PD 17.5% against 15.5% in training |
| largest feature PSI | `mort_acc` 4.30 | missing for 47.0% of training loans, 0.0% in 2013 |
| next | `fico` 0.152, `purpose` 0.149, `pub_rec` 0.101 | FICO at or below 672: 15.8% -> 23.4% of loans |
| `term_months` PSI | 0.0016 | 60-month share 23.8% -> 25.5% |

Each feature's share of the +0.174 log-odds shift comes from the change in its
mean contribution (the model's exact additive explanation, so the shares sum to
the total): **FICO +0.118 (68%)**, loan amount +0.058, `mort_acc` +0.028; purpose
(-0.042, more `credit_card` loans) and income (-0.042) pulled the other way.
Term added +0.016.

- **The score PSI did not raise an alarm, and the model was 2 points off.** A
  0.067 PSI is "stable", yet every decile was over-predicted. The rule of thumb
  is about how far the population mix moved, not about whether the PDs are
  still right; it is an early signal to pair with outcome-based calibration
  checks, not a substitute for them.
- **The over-prediction is mostly FICO.** 2013 brought more applicants at the
  low end of the FICO range, and the model scored them as riskier — but the
  default rate did not follow. If the model were correctly specified and only
  the applicant mix had moved, its calibration would have held, so either the
  FICO–default relationship weakened in 2013 or the model's single linear FICO
  term overstates risk at the low end, where 2013 put more of its loans. These
  numbers do not separate the two.
- **The earlier term hypothesis was wrong.** `RESULTS.md` used to point to the
  60-month share (18.5% -> 25.5%), but 18.5% was 2012 alone. Against the
  champion's whole 2010–2012 training window (23.8%) the share barely moved,
  and term explains 9% of the shift.
- **The largest PSI is a data-feed change, not a population change.** Lending
  Club did not report `mort_acc` for 2010–2011 loans or for 14% of 2012's; the
  champion learned an imputed value plus a missing indicator for nearly half
  its training rows and never sees a missing value in 2013. It contributes 16% of the shift. A feed change
  like this is the kind of thing PSI catches and an outcome chart cannot.
- **Why only 2013 is monitored.** The processed file holds matured loans only,
  so from 2014 the 60-month loans are missing by construction (see
  Protocol); PSI there would measure the labelling rule, not the applicants.
  A live monitor would compute it on every application instead.

## Explanations in the reports

The reports' "What drives the booster" and "Reason codes for the riskiest
loans" sections explain the **booster** (TreeSHAP), the challenger. The
champion's per-applicant reason codes (training-centred log-odds
contributions) are what `POST /v1/score` returns; its largest coefficients are
in each report's "Largest baseline coefficients" table.

## Reproducing

```bash
python -m credit_risk.prepare data/accepted_2007_to_2018Q4.csv.gz
python -m credit_risk.train data/processed/loans.parquet \
    --train 2010-01:2011-12 --test 2012-01:2012-12 \
    --out data/processed/model_report_dev_2012.md
python -m credit_risk.train data/processed/loans.parquet \
    --train 2010-01:2012-12 --test 2013-01:2013-12 \
    --out data/processed/model_report_confirm_2013.md \
    --artifact-dir artifacts --artifact-model logistic-regression
python -m credit_risk.monitor artifacts/699a62529632 data/processed/loans.parquet \
    --window 2013-01:2013-12 --out data/processed/drift_report_2013.md
```

Calibration defaults (6 months, Platt) apply. Regenerated at commit `e2a1f43`
with Python 3.12.13, numpy 2.5.3, pandas 3.0.6 and scikit-learn 1.9.1, all
three reports were byte-identical to the original runs, and `loans.parquet` had
SHA-256 `1f9059fcc210b51a56257fef1e42d49491adebba832963abde10da3ed0d07662`.
The drift report was generated with the same versions; `monitor` refuses any
other `loans.parquet` than the one in the artifact's manifest. Source file
SHA-256 is in the data report. The artifact id will match only
with the same library versions, since it is the hash of the pickled model.

## Limitations of these results

- One development window and one confirmation window: no confidence intervals
  or repeated rolling-origin evaluation, so a 0.003 AUC difference is not by
  itself evidence that one model ranks better.
- The booster was not tuned beyond early stopping; a time-ordered
  hyperparameter search might narrow the gap, though its train-to-test drop
  suggests capacity is not what it lacks.
- Accepted loans only: performance on the full applicant population, including
  those Lending Club rejected, is unknown.
