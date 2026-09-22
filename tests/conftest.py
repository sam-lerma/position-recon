"""Builders for the small hand made frames the unit tests reconcile."""

from __future__ import annotations

import pandas as pd
import pytest

from recon.normalize import CANONICAL_COLUMNS

DATE = pd.Timestamp("2026-09-18")


def position(
    cusip: str,
    quantity: float,
    price: float,
    market_value: float | None = None,
    account_id: str = "EQ-LC-01",
    as_of_date: pd.Timestamp = DATE,
    currency: str = "USD",
) -> dict:
    """One normalized position. Market value defaults to quantity times price."""
    return {
        "as_of_date": as_of_date,
        "account_id": account_id,
        "cusip": cusip,
        "quantity": float(quantity),
        "price": float(price),
        "market_value": float(quantity * price if market_value is None else market_value),
        "currency": currency,
    }


def frame(*positions: dict) -> pd.DataFrame:
    """A normalized side of the reconciliation."""
    if not positions:
        return pd.DataFrame(columns=CANONICAL_COLUMNS).astype(
            {"quantity": float, "price": float, "market_value": float}
        )
    return pd.DataFrame(list(positions))[CANONICAL_COLUMNS]


@pytest.fixture
def account_map() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "account_id": ["EQ-LC-01", "FI-CORE-01"],
            "pb_account_code": ["EQLC01", "FICORE01"],
            "account_name": ["EQ LC Fund 1", "FI Core Fund 1"],
        }
    )
