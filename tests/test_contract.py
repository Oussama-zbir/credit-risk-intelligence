from credit_risk.data.contract import (
    COLUMNS,
    Role,
    columns_with,
    feature_columns,
    loaded_columns,
)


def test_every_column_is_declared_once() -> None:
    names = [column.name for column in COLUMNS]
    assert len(names) == len(set(names))


def test_leakage_and_excluded_columns_are_never_loaded() -> None:
    forbidden = set(columns_with(Role.POST_ORIGINATION, Role.EXCLUDED))
    assert forbidden
    assert forbidden.isdisjoint(loaded_columns())
    assert forbidden.isdisjoint(feature_columns(include_lender_pricing=True))


def test_features_never_include_the_target_or_the_time_axis() -> None:
    features = set(feature_columns(include_lender_pricing=True))
    assert "loan_status" not in features
    assert "issue_d" not in features


def test_lender_pricing_is_opt_in() -> None:
    pricing = set(columns_with(Role.LENDER_PRICING))
    assert {"grade", "sub_grade", "int_rate"} <= pricing
    assert pricing.isdisjoint(feature_columns())
    assert pricing <= set(feature_columns(include_lender_pricing=True))


def test_outcome_columns_that_define_the_label_are_classified_as_leakage() -> None:
    # Recoveries and settlements are non-zero only for charged-off loans.
    leakage = set(columns_with(Role.POST_ORIGINATION))
    assert {"recoveries", "debt_settlement_flag", "total_pymnt", "last_fico_range_high"} <= leakage


def test_every_excluded_column_records_why() -> None:
    for column in COLUMNS:
        if column.role is Role.EXCLUDED:
            assert column.reason, column.name
