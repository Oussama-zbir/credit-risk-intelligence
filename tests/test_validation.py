import pytest

from conftest import MakeLoans
from credit_risk.data.validation import (
    DataValidationError,
    Severity,
    require_valid,
    validate_raw,
)


def checks(report_issues: object) -> set[tuple[str, str, Severity]]:
    assert isinstance(report_issues, tuple)
    return {(i.column, i.check, i.severity) for i in report_issues}


def test_a_contract_shaped_file_passes(make_loans: MakeLoans) -> None:
    report = require_valid(make_loans({}, {"int_rate": 7.5, "revol_util": 12.0}))
    assert report.ok
    assert report.issues == ()


def test_missing_contract_column_is_an_error(make_loans: MakeLoans) -> None:
    report = validate_raw(make_loans().drop(columns=["fico_range_low"]))
    assert ("fico_range_low", "missing_column", Severity.ERROR) in checks(report.issues)
    assert not report.ok


def test_impossible_values_are_errors(make_loans: MakeLoans) -> None:
    report = validate_raw(make_loans({"loan_amnt": -5.0}, {"fico_range_high": 2000.0}))
    assert {
        ("loan_amnt", "out_of_range", Severity.ERROR),
        ("fico_range_high", "out_of_range", Severity.ERROR),
    } <= checks(report.issues)


def test_known_source_quirks_are_warnings_not_errors(make_loans: MakeLoans) -> None:
    report = validate_raw(make_loans({"dti": -1.0}, {"dti": 999.0}, {"emp_length": None}))
    assert report.ok
    assert checks(report.issues) == {
        ("dti", "out_of_range", Severity.WARNING),
        ("emp_length", "null", Severity.WARNING),
    }
    assert next(i.rows for i in report.issues if i.column == "dti") == 2


def test_percent_strings_are_checked_as_numbers(make_loans: MakeLoans) -> None:
    report = validate_raw(make_loans({"int_rate": "250%"}, {"revol_util": "n/a"}))
    assert {
        ("int_rate", "out_of_range", Severity.ERROR),
        ("revol_util", "not_numeric", Severity.ERROR),
    } <= checks(report.issues)


def test_unexpected_nulls_are_errors_and_require_valid_raises(make_loans: MakeLoans) -> None:
    with pytest.raises(DataValidationError, match="annual_inc:null") as caught:
        require_valid(make_loans({"annual_inc": None}))
    assert not caught.value.report.ok
