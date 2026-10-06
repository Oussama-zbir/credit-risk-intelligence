# Out-of-time model comparison

- Train: issued 2010-01 to 2011-12, 33,257 loans
- Test (out of time): issued 2012-01 to 2012-12, 53,367 loans
- Gradient boosting: trees fitted on all but the last 6 training months, calibrator fitted on those months

Brier skill is measured against predicting the training default rate for every loan.
Calibration error is the loan-weighted mean |mean PD - default rate| over ten equal-count bins.

| features | model | window | loans | default rate | mean PD | AUC | Gini | KS | Brier | Brier skill | log loss | calibration error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| applicant | logistic regression | train | 33,257 | 14.38% | 14.38% | 0.7104 | 0.4208 | 0.3098 | 0.1145 | 7.00% | 0.3773 | 0.89% |
| applicant | logistic regression | test | 53,367 | 16.20% | 14.32% | 0.6781 | 0.3561 | 0.2573 | 0.1284 | 5.63% | 0.4167 | 1.88% |
| applicant | gradient boosting, raw | train | 33,257 | 14.38% | 13.92% | 0.7508 | 0.5016 | 0.3737 | 0.1099 | 10.76% | 0.3613 | 1.54% |
| applicant | gradient boosting, raw | test | 53,367 | 16.20% | 13.95% | 0.6706 | 0.3413 | 0.2466 | 0.1295 | 4.80% | 0.4209 | 2.37% |
| applicant | gradient boosting + platt | train | 33,257 | 14.38% | 15.12% | 0.7508 | 0.5016 | 0.3737 | 0.1099 | 10.76% | 0.3622 | 1.91% |
| applicant | gradient boosting + platt | test | 53,367 | 16.20% | 15.18% | 0.6706 | 0.3413 | 0.2466 | 0.1291 | 5.11% | 0.4187 | 1.40% |
| applicant + lender pricing | logistic regression | train | 33,257 | 14.38% | 14.38% | 0.7141 | 0.4282 | 0.3154 | 0.1140 | 7.42% | 0.3755 | 0.64% |
| applicant + lender pricing | logistic regression | test | 53,367 | 16.20% | 16.40% | 0.6792 | 0.3583 | 0.2616 | 0.1283 | 5.73% | 0.4157 | 1.04% |
| applicant + lender pricing | gradient boosting, raw | train | 33,257 | 14.38% | 13.96% | 0.7513 | 0.5026 | 0.3724 | 0.1096 | 11.02% | 0.3608 | 1.56% |
| applicant + lender pricing | gradient boosting, raw | test | 53,367 | 16.20% | 14.09% | 0.6668 | 0.3336 | 0.2439 | 0.1301 | 4.41% | 0.4221 | 2.42% |
| applicant + lender pricing | gradient boosting + platt | train | 33,257 | 14.38% | 15.04% | 0.7513 | 0.5026 | 0.3724 | 0.1097 | 10.89% | 0.3621 | 2.34% |
| applicant + lender pricing | gradient boosting + platt | test | 53,367 | 16.20% | 15.21% | 0.6668 | 0.3336 | 0.2439 | 0.1296 | 4.74% | 0.4199 | 1.39% |

## Features left out

Requested features with no variation in the rows a model was fitted on (the booster's rows exclude the calibration months). That model never reads them.

| features | feature | logistic regression | gradient boosting |
| --- | --- | --- | --- |
| applicant | mort_acc | all missing | all missing |
| applicant | application_type | constant 'Individual' | constant 'Individual' |
| applicant + lender pricing | mort_acc | all missing | all missing |
| applicant + lender pricing | application_type | constant 'Individual' | constant 'Individual' |

## Reliability on the test window — applicant

Test loans in ten equal-count bins of each model's own PD, lowest first: mean PD / observed default rate.

| bin | logistic regression | gradient boosting, raw | gradient boosting + platt |
| --- | --- | --- | --- |
| 1 | 4.22% / 4.53% | 3.56% / 4.72% | 4.24% / 4.72% |
| 2 | 6.81% / 7.27% | 5.83% / 7.85% | 6.77% / 7.85% |
| 3 | 8.54% / 10.16% | 7.46% / 9.59% | 8.55% / 9.59% |
| 4 | 10.08% / 11.58% | 9.05% / 11.88% | 10.25% / 11.88% |
| 5 | 11.61% / 13.42% | 10.75% / 13.81% | 12.04% / 13.81% |
| 6 | 13.28% / 15.53% | 12.72% / 16.11% | 14.10% / 16.11% |
| 7 | 15.29% / 18.19% | 15.12% / 18.16% | 16.56% / 18.16% |
| 8 | 17.97% / 21.10% | 18.17% / 21.78% | 19.67% / 21.78% |
| 9 | 22.53% / 25.17% | 22.96% / 24.81% | 24.46% / 24.81% |
| 10 | 32.87% / 35.03% | 33.90% / 33.26% | 35.15% / 33.26% |

## Largest baseline coefficients — applicant

| input | coefficient (log-odds) |
| --- | --- |
| purpose_small_business | +0.688 |
| purpose_car | -0.486 |
| purpose_wedding | -0.467 |
| home_ownership_MORTGAGE | -0.411 |
| purpose_credit_card | -0.411 |
| term_months | +0.411 |
| verification_status_Not Verified | -0.391 |
| home_ownership_OWN | -0.372 |
| home_ownership_RENT | -0.350 |
| verification_status_Source Verified | -0.347 |

## Largest baseline coefficients — applicant + lender pricing

| input | coefficient (log-odds) |
| --- | --- |
| purpose_small_business | +0.665 |
| grade_G | -0.661 |
| int_rate | +0.504 |
| home_ownership_MORTGAGE | -0.455 |
| purpose_car | -0.449 |
| sub_grade_F3 | -0.433 |
| home_ownership_OWN | -0.420 |
| purpose_wedding | -0.416 |
| purpose_credit_card | -0.408 |
| home_ownership_RENT | -0.406 |

## What drives the booster

Mean absolute TreeSHAP contribution to the booster's log-odds over 5,000 test loans. Platt calibration rescales every contribution by one positive factor; isotonic keeps their order, not their sum.

### applicant

| feature | mean abs contribution |
| --- | --- |
| term_months | 0.348 |
| fico | 0.299 |
| annual_inc | 0.279 |
| inq_last_6mths | 0.137 |
| revol_util | 0.129 |
| purpose | 0.110 |
| loan_amnt | 0.099 |
| emp_length_years | 0.080 |
| credit_history_months | 0.058 |
| revol_bal | 0.042 |

### applicant + lender pricing

| feature | mean abs contribution |
| --- | --- |
| int_rate | 0.345 |
| annual_inc | 0.272 |
| term_months | 0.237 |
| inq_last_6mths | 0.106 |
| purpose | 0.096 |
| sub_grade | 0.094 |
| revol_util | 0.080 |
| emp_length_years | 0.070 |
| credit_history_months | 0.047 |
| fico | 0.035 |

## Reason codes for the riskiest loans — applicant

Features that raised each loan's PD most, with their log-odds contribution.

| PD | reasons |
| --- | --- |
| 63.84% | loan purpose (small_business) +0.80; loan amount (25,000) +0.44; loan term (60) +0.44; credit inquiries in the last 6 months (3) +0.26 |
| 56.16% | loan term (60) +0.51; FICO score (667) +0.48; loan amount (24,500) +0.32; public records (1) +0.30 |
| 56.10% | loan term (60) +0.55; FICO score (662) +0.45; revolving credit utilisation (95) +0.44; loan amount (25,875) +0.43 |
| 52.27% | loan purpose (small_business) +0.74; revolving credit utilisation (98.2) +0.47; loan term (60) +0.40; FICO score (697) +0.21 |
| 52.06% | loan term (60) +0.75; loan amount (25,000) +0.49; FICO score (687) +0.26; length of credit history (300) +0.18 |
