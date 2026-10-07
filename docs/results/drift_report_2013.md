# Drift report — model 699a62529632 (logistic-regression)

- Reference (training window): issued 2010-01 to 2012-12, 86,624 loans
- Monitored: issued 2013-01 to 2013-12, 134,804 loans

PSI bands (rule of thumb): under 0.10 stable, 0.10 to 0.25 watch, 0.25 and above shifted. Bins are fitted on the reference window: deciles for numbers, one per level for categories, missing and unseen kept apart.

## Score

PSI of the PD on the reference PD deciles: **0.0668 (stable)**.

| quarter | loans | score PSI | mean PD | default rate |
| --- | --- | --- | --- | --- |
| 2013Q1 | 22,706 | 0.0606 | 16.98% | 15.02% |
| 2013Q2 | 30,668 | 0.0659 | 17.48% | 16.25% |
| 2013Q3 | 37,571 | 0.0769 | 17.70% | 15.56% |
| 2013Q4 | 43,859 | 0.0664 | 17.63% | 15.46% |

| PD bin | reference | monitored | PSI term |
| --- | --- | --- | --- |
| <= 0.05587 | 10.00% | 5.31% | 0.0297 |
| (0.05587, 0.07754] | 10.00% | 6.83% | 0.0121 |
| (0.07754, 0.0966] | 10.00% | 8.35% | 0.0030 |
| (0.0966, 0.1153] | 10.00% | 9.49% | 0.0003 |
| (0.1153, 0.1347] | 10.00% | 10.20% | 0.0000 |
| (0.1347, 0.1563] | 10.00% | 10.60% | 0.0004 |
| (0.1563, 0.1831] | 10.00% | 11.29% | 0.0016 |
| (0.1831, 0.22] | 10.00% | 11.69% | 0.0026 |
| (0.22, 0.2858] | 10.00% | 12.65% | 0.0062 |
| > 0.2858 | 10.00% | 13.58% | 0.0109 |

## Feature stability

| feature | PSI | band | largest move |
| --- | --- | --- | --- |
| mort_acc | 4.3010 | shifted | missing: 47.0% -> 0.0% |
| fico | 0.1518 | watch | <= 672: 15.8% -> 23.4% |
| purpose | 0.1488 | watch | credit_card: 16.9% -> 24.3% |
| pub_rec | 0.1005 | watch | > 0: 3.8% -> 11.9% |
| pub_rec_bankruptcies | 0.0997 | stable | > 0: 3.1% -> 10.6% |
| revol_bal | 0.0916 | stable | <= 2,216: 10.0% -> 3.5% |
| loan_amnt | 0.0840 | stable | <= 4,000: 11.3% -> 6.0% |
| revol_util | 0.0736 | stable | <= 15.8: 10.0% -> 4.0% |
| dti | 0.0718 | stable | > 24.89: 10.0% -> 17.8% |
| credit_history_months | 0.0614 | stable | <= 79: 10.1% -> 5.6% |
| open_acc | 0.0599 | stable | <= 5: 13.3% -> 7.6% |
| verification_status | 0.0566 | stable | Verified: 37.5% -> 49.1% |
| emp_length_years | 0.0469 | stable | > 9: 26.8% -> 34.0% |
| total_acc | 0.0366 | stable | <= 10: 11.3% -> 6.8% |
| home_ownership | 0.0264 | stable | RENT: 45.8% -> 38.2% |
| annual_inc | 0.0231 | stable | <= 3.1e+04: 10.2% -> 6.9% |
| delinq_2yrs | 0.0145 | stable | <= 0: 87.4% -> 83.4% |
| inq_last_6mths | 0.0043 | stable | <= 0: 49.4% -> 51.6% |
| term_months | 0.0016 | stable | > 36: 23.8% -> 25.5% |

## What moved the score

Mean contribution of each feature to the model's log-odds in each window (on up to 50,000 loans per window). The shifts sum to the change in mean log-odds: **+0.1742**.

| feature | reference | monitored | shift | share of shift |
| --- | --- | --- | --- | --- |
| fico | +0.0006 | +0.1183 | +0.1177 | 68% |
| loan_amnt | +0.0002 | +0.0578 | +0.0576 | 33% |
| purpose | -0.0004 | -0.0425 | -0.0421 | -24% |
| annual_inc | -0.0003 | -0.0422 | -0.0419 | -24% |
| mort_acc | +0.0002 | +0.0283 | +0.0282 | 16% |
| revol_bal | -0.0004 | -0.0282 | -0.0278 | -16% |
| revol_util | +0.0003 | +0.0236 | +0.0233 | 13% |
| open_acc | +0.0003 | +0.0224 | +0.0221 | 13% |
| term_months | -0.0002 | +0.0157 | +0.0159 | 9% |
| emp_length_years | -0.0006 | +0.0125 | +0.0131 | 8% |
| pub_rec_bankruptcies | +0.0002 | +0.0129 | +0.0127 | 7% |
| inq_last_6mths | +0.0004 | -0.0076 | -0.0080 | -5% |
| total_acc | -0.0002 | -0.0072 | -0.0070 | -4% |
| credit_history_months | +0.0001 | +0.0070 | +0.0069 | 4% |
| dti | +0.0000 | +0.0069 | +0.0069 | 4% |
| pub_rec | +0.0000 | +0.0065 | +0.0065 | 4% |
| home_ownership | +0.0001 | -0.0059 | -0.0060 | -3% |
| verification_status | -0.0000 | -0.0025 | -0.0025 | -1% |
| delinq_2yrs | +0.0000 | -0.0012 | -0.0012 | -1% |

## Unseen categories

None: every categorical value was seen in training.
