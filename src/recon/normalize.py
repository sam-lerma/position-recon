"""Bring both position feeds onto one schema so they can be joined.

The two sides disagree about almost everything except the economics: account
codes, date formats, how a short is represented, whether numbers arrive as
strings, and whether a position is one row or several lots. All of that is
resolved here so that `reconcile` only ever sees clean, comparable frames.

Canonical schema
    as_of_date      datetime64, midnight
    account_id      internal account identifier
    cusip           upper case, stripped
    quantity        signed float, long positive
    price           float, equity in currency, bond in percent of par
    market_value    signed float, reporting currency (USD)
    currency        ISO code of the position currency
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from recon.config import ACCOUNT_MAP_PATH

CANONICAL_COLUMNS = [
    "as_of_date",
    "account_id",
    "cusip",
    "quantity",
    "price",
    "market_value",
    "currency",
]

INTERNAL_REQUIRED = [
    "as_of_date",
    "account_code",
    "cusip",
    "quantity",
    "price",
    "market_value",
    "currency",
]

PB_REQUIRED = [
    "business_date",
    "account",
    "cusip",
    "long_short",
    "quantity",
    "price",
    "market_value",
    "ccy",
]

# Lots of the same security on the same day carry the same mark, so anything
# above float noise means the file is not what we think it is.
LOT_PRICE_TOLERANCE = 1e-9


class NormalizationError(ValueError):
    """Raised when a feed cannot be trusted enough to reconcile."""


class MissingColumnsError(NormalizationError):
    pass


class UnmappedAccountError(NormalizationError):
    pass


class InconsistentLotPriceError(NormalizationError):
    pass


class DuplicatePositionError(NormalizationError):
    pass


def load_account_map(path: Path | str = ACCOUNT_MAP_PATH) -> pd.DataFrame:
    """Reference table mapping the prime broker's account codes to ours."""
    account_map = pd.read_csv(path, dtype=str)
    _require_columns(account_map, ["account_id", "pb_account_code"], "account map")
    return account_map.assign(
        account_id=lambda df: df["account_id"].str.strip().str.upper(),
        pb_account_code=lambda df: df["pb_account_code"].str.strip().str.upper(),
    )


def normalize_internal(raw: pd.DataFrame, account_map: pd.DataFrame) -> pd.DataFrame:
    """Normalize the internal book of record extract.

    Quantities are already signed and one row is one position, so the work is
    cleaning identifiers and checking every account is one we know about.
    """
    _require_columns(raw, INTERNAL_REQUIRED, "internal feed")
    df = raw.copy()

    df["as_of_date"] = pd.to_datetime(df["as_of_date"], format="%Y-%m-%d")
    df["account_id"] = df["account_code"].astype(str).str.strip().str.upper()
    df["cusip"] = df["cusip"].astype(str).str.strip().str.upper()
    df["quantity"] = _to_float(df["quantity"])
    df["price"] = _to_float(df["price"])
    df["market_value"] = _to_float(df["market_value"])
    df["currency"] = df["currency"].astype(str).str.strip().str.upper()

    known = set(account_map["account_id"])
    _reject_unknown_accounts(df["account_id"], known, side="internal")

    # The docstring above claims one row is one position. Check it, because a
    # re-delivered file or a book split by strategy fans out across the join
    # into breaks that net to nothing.
    _reject_duplicate_positions(df)

    return df[CANONICAL_COLUMNS].reset_index(drop=True)


def normalize_pb(raw: pd.DataFrame, account_map: pd.DataFrame) -> pd.DataFrame:
    """Normalize the prime broker position file.

    Three things differ from the internal extract: the file is lot level, so it
    is aggregated to position level; shorts arrive as a positive quantity with a
    side flag, so the sign is applied here; and the numeric columns arrive as
    formatted strings.
    """
    _require_columns(raw, PB_REQUIRED, "prime broker feed")
    df = raw.copy()

    df["as_of_date"] = pd.to_datetime(df["business_date"], format="%m/%d/%Y")
    df["cusip"] = df["cusip"].astype(str).str.strip().str.upper()
    df["currency"] = df["ccy"].astype(str).str.strip().str.upper()

    pb_code = df["account"].astype(str).str.strip().str.upper()
    mapping = account_map.set_index("pb_account_code")["account_id"]
    _reject_unknown_accounts(pb_code, set(mapping.index), side="prime broker")
    df["account_id"] = pb_code.map(mapping)

    # Shorts are reported as a positive quantity plus a side flag.
    side = df["long_short"].astype(str).str.strip().str.upper()
    unknown_side = set(side.unique()) - {"L", "S"}
    if unknown_side:
        raise NormalizationError(
            f"prime broker feed has unexpected long_short values: {sorted(unknown_side)}"
        )
    sign = np.where(side == "S", -1.0, 1.0)
    df["quantity"] = _to_float(df["quantity"]).abs() * sign
    df["market_value"] = _to_float(df["market_value"]).abs() * sign
    df["price"] = _to_float(df["price"])

    return _aggregate_lots(df)


def _aggregate_lots(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse lot level rows to one row per position."""
    grouped = df.groupby(["as_of_date", "account_id", "cusip"], as_index=False, sort=False)
    agg = grouped.agg(
        quantity=("quantity", "sum"),
        market_value=("market_value", "sum"),
        price=("price", "first"),
        currency=("currency", "first"),
        _price_min=("price", "min"),
        _price_max=("price", "max"),
    )

    disagrees = (agg["_price_max"] - agg["_price_min"]) > LOT_PRICE_TOLERANCE
    if disagrees.any():
        offenders = agg.loc[disagrees, ["account_id", "cusip"]].head(5).to_dict("records")
        raise InconsistentLotPriceError(
            f"lots of the same security are marked at different prices: {offenders}"
        )

    return agg[CANONICAL_COLUMNS].reset_index(drop=True)


def _reject_duplicate_positions(df: pd.DataFrame) -> None:
    grain = ["as_of_date", "account_id", "cusip"]
    duplicated = df.duplicated(subset=grain, keep=False)
    if duplicated.any():
        offenders = (
            df.loc[duplicated, ["account_id", "cusip"]].drop_duplicates().head(5).to_dict("records")
        )
        raise DuplicatePositionError(
            f"internal feed has more than one row for the same position: {offenders}"
        )


def _to_float(series: pd.Series) -> pd.Series:
    """Parse numbers that may arrive as strings with thousands separators."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    cleaned = series.astype(str).str.replace(",", "", regex=False).str.strip()
    return pd.to_numeric(cleaned, errors="coerce").astype(float)


def _require_columns(df: pd.DataFrame, required: list[str], label: str) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise MissingColumnsError(f"{label} is missing columns: {missing}")


def _reject_unknown_accounts(codes: pd.Series, known: set[str], side: str) -> None:
    unknown = sorted(set(codes.dropna().unique()) - known)
    if unknown:
        raise UnmappedAccountError(
            f"{side} feed has accounts that are not in the account map: {unknown}"
        )
