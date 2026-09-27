import pandas as pd
import pytest

from conftest import MakeLoans
from credit_risk.data.split import SplitError, TimeWindow, out_of_time_split
from credit_risk.data.target import label_loans

SNAPSHOT = pd.Timestamp("2018-12-01")


def window(start: str, end: str) -> TimeWindow:
    return TimeWindow(pd.Timestamp(start), pd.Timestamp(end))


@pytest.fixture
def labelled(make_loans: MakeLoans) -> pd.DataFrame:
    rows = [
        {"issue_d": month, "loan_status": status}
        for month in ("Jan-2013", "Jun-2013", "Dec-2013", "Mar-2014", "Sep-2014")
        for status in ("Fully Paid", "Charged Off")
    ]
    frame, _ = label_loans(make_loans(*rows), snapshot=SNAPSHOT)
    return frame


def test_split_assigns_loans_by_issue_month(labelled: pd.DataFrame) -> None:
    split = out_of_time_split(
        labelled, train=window("2013-01-01", "2013-12-01"), test=window("2014-01-01", "2014-12-01")
    )
    assert split.train["issue_d"].unique().tolist() == ["Jan-2013", "Jun-2013", "Dec-2013"]
    assert split.test["issue_d"].unique().tolist() == ["Mar-2014", "Sep-2014"]
    assert split.train["issue_date"].max() < split.test["issue_date"].min()


def test_gap_months_belong_to_neither_window(labelled: pd.DataFrame) -> None:
    split = out_of_time_split(
        labelled, train=window("2013-01-01", "2013-06-01"), test=window("2014-01-01", "2014-12-01")
    )
    used = set(split.train["issue_d"]) | set(split.test["issue_d"])
    assert "Dec-2013" not in used


@pytest.mark.parametrize(
    ("train", "test"),
    [
        (("2013-01-01", "2013-12-01"), ("2013-12-01", "2014-12-01")),  # shared month
        (("2014-01-01", "2014-12-01"), ("2013-01-01", "2013-12-01")),  # test in the past
    ],
)
def test_test_window_must_be_strictly_later(
    labelled: pd.DataFrame, train: tuple[str, str], test: tuple[str, str]
) -> None:
    with pytest.raises(SplitError, match="must start after"):
        out_of_time_split(labelled, train=window(*train), test=window(*test))


def test_single_class_window_is_refused(labelled: pd.DataFrame) -> None:
    only_good = labelled[labelled["loan_status"] == "Fully Paid"]
    with pytest.raises(SplitError, match="class"):
        out_of_time_split(
            only_good,
            train=window("2013-01-01", "2013-12-01"),
            test=window("2014-01-01", "2014-12-01"),
        )


def test_inverted_window_is_refused() -> None:
    with pytest.raises(SplitError, match="starts after"):
        window("2014-01-01", "2013-01-01")
