"""Which loans belong to the modelling population, decided before validation.

The export mixes two populations. Alongside the loans Lending Club issued under
its credit policy, it carries 2007-2010 loans whose `loan_status` reads
`Does not meet the credit policy. Status:...`: loans issued under criteria the
lender later stopped applying. They are not the population a model scored at
application time would see, and they are also where the bureau fields the
contract requires (`earliest_cr_line`, `open_acc`, `pub_rec`, ...) are missing.

Removing them is a modelling decision, so it is made here, once, by name and
with a count — not by weakening validation (`NULLABLE` would then let the same
gap through for every other loan) and not by a `dropna()` that would remove
rows without saying which or why. Validation then holds the remaining
population to the full contract.

Statuses are matched exactly. A new variant of the prefix is not excluded; it
reaches `label_loans`, which refuses it as an unknown status.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import pandas as pd

CREDIT_POLICY_STATUSES: Final = frozenset(
    {
        "Does not meet the credit policy. Status:Fully Paid",
        "Does not meet the credit policy. Status:Charged Off",
    }
)


@dataclass(frozen=True, slots=True)
class CohortReport:
    """Rows entering cohort selection, and how many each exclusion removed."""

    input_rows: int
    outside_credit_policy: int

    @property
    def eligible(self) -> int:
        return self.input_rows - self.outside_credit_policy


def select_cohort(raw: pd.DataFrame) -> tuple[pd.DataFrame, CohortReport]:
    """Drop loans issued outside Lending Club's credit policy; count them."""
    outside_policy = raw["loan_status"].isin(CREDIT_POLICY_STATUSES)
    report = CohortReport(
        input_rows=len(raw),
        outside_credit_policy=int(outside_policy.sum()),
    )
    return raw.loc[~outside_policy].reset_index(drop=True), report
