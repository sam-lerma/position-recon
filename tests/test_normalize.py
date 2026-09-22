"""The feeds disagree about format. These tests pin down how that is resolved."""

from __future__ import annotations

import pandas as pd
import pytest

from recon.normalize import (
    InconsistentLotPriceError,
    MissingColumnsError,
    NormalizationError,
    UnmappedAccountError,
    normalize_internal,
    normalize_pb,
)


def internal_raw(**overrides) -> pd.DataFrame:
    row = {
        "as_of_date": "2026-09-18",
        "account_code": "EQ-LC-01",
        "cusip": "abc123xyz",
        "quantity": -1500.0,
        "price": 42.5,
        "market_value": -63750.0,
        "currency": "usd",
    }
    row.update(overrides)
    return pd.DataFrame([row])


def pb_raw(**overrides) -> pd.DataFrame:
    row = {
        "business_date": "09/18/2026",
        "account": "EQLC01",
        "cusip": "ABC123XYZ",
        "long_short": "L",
        "quantity": "1,500.00",
        "price": 42.5,
        "market_value": "63,750.00",
        "ccy": "USD",
    }
    row.update(overrides)
    return pd.DataFrame([row])


class TestInternal:
    def test_cleans_identifiers_and_parses_the_date(self, account_map):
        result = normalize_internal(internal_raw(cusip="  abc123xyz "), account_map)
        assert result.loc[0, "cusip"] == "ABC123XYZ"
        assert result.loc[0, "as_of_date"] == pd.Timestamp("2026-09-18")
        assert result.loc[0, "currency"] == "USD"

    def test_keeps_the_sign_on_a_short(self, account_map):
        result = normalize_internal(internal_raw(), account_map)
        assert result.loc[0, "quantity"] == -1500.0
        assert result.loc[0, "market_value"] == -63750.0

    def test_rejects_an_account_that_is_not_in_the_map(self, account_map):
        with pytest.raises(UnmappedAccountError, match="EQ-LC-99"):
            normalize_internal(internal_raw(account_code="EQ-LC-99"), account_map)

    def test_rejects_a_feed_that_is_missing_a_column(self, account_map):
        raw = internal_raw().drop(columns=["market_value"])
        with pytest.raises(MissingColumnsError, match="market_value"):
            normalize_internal(raw, account_map)


class TestPrimeBroker:
    def test_maps_the_broker_account_code_to_ours(self, account_map):
        result = normalize_pb(pb_raw(), account_map)
        assert result.loc[0, "account_id"] == "EQ-LC-01"

    def test_applies_the_side_flag_to_quantity_and_market_value(self, account_map):
        result = normalize_pb(pb_raw(long_short="S"), account_map)
        assert result.loc[0, "quantity"] == -1500.0
        assert result.loc[0, "market_value"] == -63750.0

    def test_strips_thousands_separators(self, account_map):
        result = normalize_pb(pb_raw(quantity="1,234,567.00"), account_map)
        assert result.loc[0, "quantity"] == 1_234_567.0

    def test_rejects_an_unexpected_side_flag(self, account_map):
        with pytest.raises(NormalizationError, match="long_short"):
            normalize_pb(pb_raw(long_short="X"), account_map)

    def test_rejects_an_account_the_broker_invented(self, account_map):
        with pytest.raises(UnmappedAccountError, match="EQLC99"):
            normalize_pb(pb_raw(account="EQLC99"), account_map)

    def test_aggregates_lots_into_one_position(self, account_map):
        lots = pd.concat(
            [
                pb_raw(quantity="1,000.00", market_value="42,500.00"),
                pb_raw(quantity="500.00", market_value="21,250.00"),
            ],
            ignore_index=True,
        )
        result = normalize_pb(lots, account_map)

        assert len(result) == 1
        assert result.loc[0, "quantity"] == 1500.0
        assert result.loc[0, "market_value"] == 63_750.0
        assert result.loc[0, "price"] == 42.5

    def test_rejects_lots_of_one_security_marked_at_different_prices(self, account_map):
        lots = pd.concat(
            [pb_raw(price=42.5), pb_raw(price=44.0)],
            ignore_index=True,
        )
        with pytest.raises(InconsistentLotPriceError, match="ABC123XYZ"):
            normalize_pb(lots, account_map)

    def test_keeps_separate_accounts_separate_when_aggregating(self, account_map):
        lots = pd.concat(
            [pb_raw(), pb_raw(account="FICORE01")],
            ignore_index=True,
        )
        result = normalize_pb(lots, account_map)
        assert set(result["account_id"]) == {"EQ-LC-01", "FI-CORE-01"}
