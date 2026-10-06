# Out-of-time model comparison

- Train: issued 2010-01 to 2012-12, 86,624 loans
- Test (out of time): issued 2013-01 to 2013-12, 134,804 loans
- Gradient boosting: trees fitted on all but the last 6 training months, calibrator fitted on those months

Brier skill is measured against predicting the training default rate for every loan.
Calibration error is the loan-weighted mean |mean PD - default rate| over ten equal-count bins.

| features | model | window | loans | default rate | mean PD | AUC | Gini | KS | Brier | Brier skill | log loss | calibration error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| applicant | logistic regression | train | 86,624 | 15.50% | 15.50% | 0.6929 | 0.3858 | 0.2799 | 0.1226 | 6.42% | 0.3999 | 0.52% |
| applicant | logistic regression | test | 134,804 | 15.60% | 17.51% | 0.6744 | 0.3489 | 0.2533 | 0.1254 | 4.70% | 0.4089 | 1.91% |
| applicant | gradient boosting, raw | train | 86,624 | 15.50% | 16.01% | 0.7108 | 0.4217 | 0.3059 | 0.1207 | 7.87% | 0.3936 | 0.91% |
| applicant | gradient boosting, raw | test | 134,804 | 15.60% | 18.35% | 0.6717 | 0.3434 | 0.2474 | 0.1261 | 4.22% | 0.4111 | 2.76% |
| applicant | gradient boosting + platt | train | 86,624 | 15.50% | 14.81% | 0.7108 | 0.4217 | 0.3059 | 0.1208 | 7.76% | 0.3937 | 1.30% |
| applicant | gradient boosting + platt | test | 134,804 | 15.60% | 17.01% | 0.6717 | 0.3434 | 0.2474 | 0.1254 | 4.73% | 0.4090 | 1.41% |
| applicant + lender pricing | logistic regression | train | 86,624 | 15.50% | 15.51% | 0.6956 | 0.3912 | 0.2831 | 0.1223 | 6.65% | 0.3989 | 0.39% |
| applicant + lender pricing | logistic regression | test | 134,804 | 15.60% | 17.60% | 0.6854 | 0.3708 | 0.2701 | 0.1248 | 5.22% | 0.4059 | 2.01% |
| applicant + lender pricing | gradient boosting, raw | train | 86,624 | 15.50% | 15.99% | 0.7073 | 0.4145 | 0.3046 | 0.1210 | 7.63% | 0.3947 | 0.90% |
| applicant + lender pricing | gradient boosting, raw | test | 134,804 | 15.60% | 17.93% | 0.6812 | 0.3624 | 0.2617 | 0.1251 | 4.94% | 0.4074 | 2.34% |
| applicant + lender pricing | gradient boosting + platt | train | 86,624 | 15.50% | 14.88% | 0.7073 | 0.4145 | 0.3046 | 0.1212 | 7.47% | 0.3951 | 1.26% |
| applicant + lender pricing | gradient boosting + platt | test | 134,804 | 15.60% | 16.65% | 0.6812 | 0.3624 | 0.2617 | 0.1245 | 5.39% | 0.4057 | 1.05% |

## Features left out

Requested features with no variation in the rows a model was fitted on (the booster's rows exclude the calibration months). That model never reads them.

| features | feature | logistic regression | gradient boosting |
| --- | --- | --- | --- |
| applicant | application_type | constant 'Individual' | constant 'Individual' |
| applicant + lender pricing | application_type | constant 'Individual' | constant 'Individual' |

## Reliability on the test window — applicant

Test loans in ten equal-count bins of each model's own PD, lowest first: mean PD / observed default rate.

| bin | logistic regression | gradient boosting, raw | gradient boosting + platt |
| --- | --- | --- | --- |
| 1 | 5.19% / 4.47% | 5.83% / 4.42% | 5.32% / 4.42% |
| 2 | 8.42% / 7.12% | 9.31% / 7.54% | 8.52% / 7.54% |
| 3 | 10.56% / 9.56% | 11.40% / 9.24% | 10.45% / 9.24% |
| 4 | 12.49% / 10.99% | 13.29% / 11.25% | 12.20% / 11.25% |
| 5 | 14.44% / 13.12% | 15.23% / 13.46% | 14.01% / 13.46% |
| 6 | 16.58% / 15.40% | 17.44% / 15.45% | 16.07% / 15.45% |
| 7 | 19.14% / 17.30% | 20.06% / 17.37% | 18.53% / 17.37% |
| 8 | 22.59% / 20.72% | 23.49% / 20.50% | 21.77% / 20.50% |
| 9 | 27.76% / 24.99% | 28.66% / 24.61% | 26.70% / 24.61% |
| 10 | 37.89% / 32.29% | 38.79% / 32.11% | 36.50% / 32.11% |

## Largest baseline coefficients — applicant

| input | coefficient (log-odds) |
| --- | --- |
| purpose_small_business | +0.594 |
| purpose_credit_card | -0.484 |
| purpose_wedding | -0.457 |
| purpose_car | -0.447 |
| home_ownership_MORTGAGE | -0.367 |
| fico | -0.357 |
| term_months | +0.352 |
| verification_status_Verified | -0.348 |
| verification_status_Not Verified | -0.339 |
| annual_inc | -0.319 |

## Largest baseline coefficients — applicant + lender pricing

| input | coefficient (log-odds) |
| --- | --- |
| purpose_small_business | +0.573 |
| purpose_wedding | -0.460 |
| purpose_credit_card | -0.453 |
| purpose_car | -0.435 |
| verification_status_Verified | -0.343 |
| home_ownership_MORTGAGE | -0.336 |
| annual_inc | -0.319 |
| verification_status_Not Verified | -0.301 |
| purpose_major_purchase | -0.299 |
| home_ownership_OWN | -0.290 |

## What drives the booster

Mean absolute TreeSHAP contribution to the booster's log-odds over 5,000 test loans. Platt calibration rescales every contribution by one positive factor; isotonic keeps their order, not their sum.

### applicant

| feature | mean abs contribution |
| --- | --- |
| term_months | 0.300 |
| fico | 0.297 |
| annual_inc | 0.247 |
| inq_last_6mths | 0.148 |
| loan_amnt | 0.128 |
| revol_util | 0.106 |
| mort_acc | 0.091 |
| purpose | 0.080 |
| emp_length_years | 0.055 |
| pub_rec | 0.025 |

### applicant + lender pricing

| feature | mean abs contribution |
| --- | --- |
| int_rate | 0.389 |
| annual_inc | 0.246 |
| term_months | 0.176 |
| purpose | 0.070 |
| inq_last_6mths | 0.069 |
| sub_grade | 0.056 |
| revol_util | 0.053 |
| emp_length_years | 0.049 |
| fico | 0.033 |
| mort_acc | 0.024 |

## Reason codes for the riskiest loans — applicant

Features that raised each loan's PD most, with their log-odds contribution.

| PD | reasons |
| --- | --- |
| 54.60% | loan term (60) +0.60; loan amount (26,375) +0.40; FICO score (677) +0.38; revolving credit utilisation (93.9) +0.29 |
| 53.77% | loan term (60) +0.50; FICO score (672) +0.48; loan amount (35,000) +0.47; credit inquiries in the last 6 months (5) +0.47 |
| 53.49% | loan term (60) +0.45; FICO score (662) +0.43; annual income (39,352) +0.30; revolving credit utilisation (90.8) +0.29 |
| 51.62% | loan term (60) +0.63; FICO score (682) +0.33; loan amount (25,000) +0.31; revolving credit utilisation (87.6) +0.21 |
| 51.33% | loan term (60) +0.57; FICO score (662) +0.48; loan amount (30,000) +0.36; revolving credit utilisation (96.3) +0.30 |
