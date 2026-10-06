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
  stable default rate would not have flagged it. These reports do not say
  which inputs moved; the 60-month share, for one, rose from 18.5% to 25.5%,
  and term is among the largest drivers. Measuring that per feature and on the
  score is what the PSI milestone is for.
- **Recalibration fixes level, not ranking.** Platt scaling cuts the booster's
  calibration error from 2.76% to 1.41% in 2013 and leaves AUC unchanged, as a
  monotone transform must.
- **Degenerate features are window-dependent.** `mort_acc` is missing for every
  2010–2011 loan, so the development models leave it out; the confirmation
  models, whose training window includes 2012, use it. `application_type` is
  constant `Individual` in both windows and is left out of every model.

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
```

Calibration defaults (6 months, Platt) apply. Regenerated at commit `e2a1f43`
with Python 3.12.13, numpy 2.5.3, pandas 3.0.6 and scikit-learn 1.9.1, all
three reports were byte-identical to the original runs, and `loans.parquet` had
SHA-256 `1f9059fcc210b51a56257fef1e42d49491adebba832963abde10da3ed0d07662`.
Source file SHA-256 is in the data report. The artifact id will match only
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
