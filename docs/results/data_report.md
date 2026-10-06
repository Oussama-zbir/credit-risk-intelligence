# Data report

- Source: `accepted_2007_to_2018Q4.csv.gz`
- SHA-256: `55c16f75120f897683f02e7aabcf080d0e4a20c4832feb1d592cfa941bd62a2d`
- Snapshot (latest `last_pymnt_d`): 2019-03

## Population

| stage | rows |
| --- | --- |
| read | 2,260,701 |
| summary lines dropped | 33 |
| cohort input | 2,260,668 |
| excluded: outside the historical credit policy | 2,749 |
| modelling cohort (validated) | 2,257,919 |
| immature at snapshot | 1,474,753 |
| matured but unresolved | 6,889 |
| labelled | 776,277 |
| defaults | 117,611 (15.15%) |

## Vintages

| issue year | loans | default rate | 60-month share |
| --- | --- | --- | --- |
| 2007 | 251 | 17.93% | 0.0% |
| 2008 | 1,562 | 15.81% | 0.0% |
| 2009 | 4,716 | 12.60% | 0.0% |
| 2010 | 11,536 | 12.89% | 26.6% |
| 2011 | 21,721 | 15.18% | 35.1% |
| 2012 | 53,367 | 16.20% | 18.5% |
| 2013 | 134,804 | 15.60% | 25.5% |
| 2014 | 175,509 | 14.63% | 7.4% |
| 2015 | 283,026 | 14.89% | 0.0% |
| 2016 | 89,785 | 16.10% | 0.0% |

## Validation warnings

| column | check | rows |
| --- | --- | --- |
| emp_length | null | 146,873 |
| dti | null | 1,711 |
| inq_last_6mths | null | 1 |
| pub_rec_bankruptcies | null | 697 |
| revol_util | null | 1,762 |
| mort_acc | null | 47,281 |
| dti | out_of_range | 2,563 |
| revol_util | out_of_range | 28 |

## Set to missing at feature time

- `dti` below zero: 1
- credit line opened after issue: 0

## Missing values by feature

| feature | missing |
| --- | --- |
| mort_acc | 6.09% |
| emp_length_years | 5.77% |
| pub_rec_bankruptcies | 0.09% |
| revol_util | 0.06% |
| dti | 0.00% |
