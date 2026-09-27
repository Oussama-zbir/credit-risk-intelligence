# Data: source, default definition, leakage audit

This document records the decisions that fix *what is being predicted, on which
loans, from which information*. Every model result in this repository is only
as meaningful as these choices, so they are written down before any model is.

## Source

**Lending Club accepted loans, 2007–2018Q4** (the public export redistributed on
Kaggle as `accepted_2007_to_2018Q4.csv`, ~2.26M loans, ~150 columns). The raw
file is not committed; it goes in `data/`, which is git-ignored.

### Why this dataset

The project's central methodological claim is **out-of-time validation**, so the
dataset has to have a usable time axis. That rules out the usual alternatives:

| Dataset | Rows | Time axis | Verdict |
| --- | --- | --- | --- |
| Lending Club 2007–2018 | ~2.3M | `issue_d`, monthly, 12 years | **Chosen** |
| Home Credit Default Risk | ~307k | none (relative days only) | No out-of-time split possible |
| German Credit (UCI) | 1,000 | none | Too small, no time axis |
| Give Me Some Credit | 150k | none | No time axis |

Lending Club also spans a full cycle of underwriting changes and repricing,
which is what makes an out-of-time test *fail differently* from a random one —
the point of running it.

Known weaknesses, accepted: it is a single lender's *accepted* population
(no rejects, so no reject inference), income is mostly self-reported, and the
lender's own grade is in the file (see Lender pricing below).

## Default definition

| `loan_status` | Label |
| --- | --- |
| `Charged Off`, `Default`, `Does not meet the credit policy. Status:Charged Off` | default (1) |
| `Fully Paid`, `Does not meet the credit policy. Status:Fully Paid` | good (0) |
| `Current`, `In Grace Period`, `Late (16-30 days)`, `Late (31-120 days)` | unresolved — not labelled |
| anything else, or missing | **error** (`UnknownStatusError`) |

An unknown status is refused rather than mapped: a new vendor value silently
counted as good or bad would move the label without anyone deciding it should.

## Observation window (maturity)

A loan is labelled only if its **final scheduled payment fell due on or before
the snapshot date** (`issue month + term months <= snapshot`).

Without this, dropping `Current` loans biases recent vintages: a loan resolves
early only by prepaying or defaulting, so the resolved part of a young vintage
over-represents both. The bias grows toward the end of the data — precisely the
window an out-of-time test is drawn from. With the maturity rule, 60-month loans
contribute only through vintages issued five years before the snapshot, which
shrinks the usable window; that is the price of an unbiased label.

`label_loans` returns a `LabelReport` (input rows, immature, unresolved,
labelled, defaults) so the dropped population is inspected on every run. The
snapshot is a parameter, set from the export's latest `last_pymnt_d`.

## Leakage audit

Every column the project reads is declared in
[`src/credit_risk/data/contract.py`](../src/credit_risk/data/contract.py) with a
role. Undeclared columns are never loaded.

- **Applicant** — known from the application and the bureau pull at origination.
  The default feature set.
- **Lender pricing** — `grade`, `sub_grade`, `int_rate`, `installment`. Known at
  origination, but they are the output of Lending Club's own credit model.
  Opt-in (`feature_columns(include_lender_pricing=True)`), so every result
  states which feature set it used, and the uplift from re-learning the
  lender's scorecard is measured rather than hidden.
- **Post-origination** — payments, recoveries, settlement flags, last payment
  and later bureau pulls. `recoveries` and `debt_settlement_flag` are non-zero
  *only* for charged-off loans — they are the label. Declared so the audit is
  explicit and tested; never loaded.
- **Excluded** — free text (`emp_title`, `title`, `desc`) and geography
  (`zip_code`, `addr_state`). Geography is excluded as a proxy for protected
  attributes, a constraint any regulated lender faces.

## Ingestion checks

`validate_raw` runs before anything is derived. **Errors** stop the pipeline
(missing contract column, unexpected nulls, non-numeric values, impossible
ranges such as FICO outside 300–850). **Warnings** record known source quirks
handled later — `dti` of -1 or above 100, `revol_util` above 100%, and nulls in
columns the source is known to leave empty — so their size is visible in every
run instead of being cleaned away silently.
