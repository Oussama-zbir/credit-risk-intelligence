import pandas as pd
import pytest

from conftest import MakeLoans, make_processed
from credit_risk.data.split import SplitError, TimeWindow, hold_out_latest, out_of_time_split
from credit_risk.data.target import ISSUE_DATE, TARGET, label_loans

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


def test_hold_out_latest_takes_the_last_months_of_the_training_window() -> None:
    frame = make_processed()
    window = TimeWindow(pd.Timestamp("2012-01-01"), pd.Timestamp("2014-12-01"))
    train = frame.loc[window.contains(frame[ISSUE_DATE])]
    fitting, held_out = hold_out_latest(train, window, months=6)

    assert len(fitting) + len(held_out) == len(train)
    assert fitting[ISSUE_DATE].max() == pd.Timestamp("2014-06-01")
    assert held_out[ISSUE_DATE].min() == pd.Timestamp("2014-07-01")


@pytest.mark.parametrize("months", [0, 36, 40])
def test_hold_out_must_leave_something_to_fit_on(months: int) -> None:
    frame = make_processed()
    window = TimeWindow(pd.Timestamp("2012-01-01"), pd.Timestamp("2014-12-01"))
    with pytest.raises(SplitError, match="cannot hold out"):
        hold_out_latest(frame.loc[window.contains(frame[ISSUE_DATE])], window, months=months)


def test_single_class_held_out_slice_is_refused() -> None:
    frame = make_processed()
    window = TimeWindow(pd.Timestamp("2012-01-01"), pd.Timestamp("2014-12-01"))
    train = frame.loc[window.contains(frame[ISSUE_DATE])].copy()
    train.loc[train[ISSUE_DATE] >= pd.Timestamp("2014-10-01"), TARGET] = 0
    with pytest.raises(SplitError, match="held-out slice"):
        hold_out_latest(train, window, months=3)
