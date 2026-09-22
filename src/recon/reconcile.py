"""Compare the two normalized feeds and classify every difference.

The comparison is a full outer join on position grain followed by a single
vectorized classification pass. Rules are ordered and the first match wins.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from recon.config import (
    DEFAULT_TOLERANCES,
    MARKET_VALUE_BREAK,
    MATCHED,
    MISSING_IN_INTERNAL,
    MISSING_IN_PB,
    POSITION_GRAIN,
    PRICE_BREAK,
    QUANTITY_BREAK,
    Tolerances,
)

COMPARED = ["quantity", "price", "market_value"]

OUTPUT_COLUMNS = [
    "as_of_date",
    "account_id",
    "cusip",
    "currency",
    "break_type",
    "quantity_internal",
    "quantity_pb",
    "quantity_diff",
    "price_internal",
    "price_pb",
    "price_diff",
    "market_value_internal",
    "market_value_pb",
    "market_value_diff",
    "abs_market_value_diff",
]


def reconcile(
    internal: pd.DataFrame,
    pb: pd.DataFrame,
    tolerances: Tolerances = DEFAULT_TOLERANCES,
) -> pd.DataFrame:
    """Join both sides on position grain and label every row.

    Returns every position, matched or not. Filter with `breaks_only` when you
    want the exception queue; keeping the matched rows here makes the match rate
    computable from the same frame.
    """
    merged = internal.merge(
        pb,
        on=POSITION_GRAIN,
        how="outer",
        suffixes=("_internal", "_pb"),
        indicator=False,
    )

    # Capture which side is actually present before filling, because after the
    # fill a missing position is indistinguishable from a flat one.
    internal_present = merged["quantity_internal"].notna().to_numpy()
    pb_present = merged["quantity_pb"].notna().to_numpy()

    for column in COMPARED:
        for side in ("internal", "pb"):
            merged[f"{column}_{side}"] = merged[f"{column}_{side}"].astype(float).fillna(0.0)

    merged["currency"] = merged["currency_internal"].fillna(merged["currency_pb"])

    for column in COMPARED:
        merged[f"{column}_diff"] = merged[f"{column}_internal"] - merged[f"{column}_pb"]
    merged["abs_market_value_diff"] = merged["market_value_diff"].abs()

    merged["break_type"] = _classify(merged, internal_present, pb_present, tolerances)

    return (
        merged[OUTPUT_COLUMNS]
        .sort_values(["as_of_date", "account_id", "cusip"])
        .reset_index(drop=True)
    )


def _classify(
    merged: pd.DataFrame,
    internal_present: np.ndarray,
    pb_present: np.ndarray,
    tolerances: Tolerances,
) -> np.ndarray:
    """Label each row with the first rule it trips.

    The tolerance arguments are ordered (prime broker, internal) on purpose:
    np.isclose scales its relative tolerance off the second argument, and the
    internal book is the reference we are measuring the broker against.
    """
    quantity_matches = np.isclose(
        merged["quantity_pb"].to_numpy(),
        merged["quantity_internal"].to_numpy(),
        rtol=0.0,
        atol=tolerances.quantity_abs,
    )
    price_matches = np.isclose(
        merged["price_pb"].to_numpy(),
        merged["price_internal"].to_numpy(),
        rtol=tolerances.price_rel,
        atol=0.0,
    )
    market_value_matches = np.isclose(
        merged["market_value_pb"].to_numpy(),
        merged["market_value_internal"].to_numpy(),
        rtol=tolerances.market_value_rel,
        atol=tolerances.market_value_abs,
    )

    conditions = [
        internal_present & ~pb_present,
        ~internal_present & pb_present,
        ~quantity_matches,
        ~price_matches,
        ~market_value_matches,
    ]
    choices = [
        MISSING_IN_PB,
        MISSING_IN_INTERNAL,
        QUANTITY_BREAK,
        PRICE_BREAK,
        MARKET_VALUE_BREAK,
    ]
    return np.select(conditions, choices, default=MATCHED)


def breaks_only(recon: pd.DataFrame) -> pd.DataFrame:
    """The exception queue: everything that is not a clean match."""
    return recon.loc[recon["break_type"] != MATCHED].reset_index(drop=True)


def match_rate(recon: pd.DataFrame) -> float:
    """Share of positions that matched, on the union of both sides."""
    if len(recon) == 0:
        return 1.0
    return float((recon["break_type"] == MATCHED).mean())
