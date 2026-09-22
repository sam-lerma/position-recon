"""Compare the two normalized feeds and classify every difference.

The comparison is a full outer join on position grain followed by a single
vectorized classification pass. Rules are ordered and the first match wins.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from recon.config import (
    CURRENCY_BREAK,
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
    # validate rejects a duplicated position on either side rather than fanning
    # it out into breaks that net to nothing. indicator is what tells us which
    # side a row came from: deriving that from a null quantity would also flag a
    # row we did receive but failed to parse.
    merged = internal.merge(
        pb,
        on=POSITION_GRAIN,
        how="outer",
        suffixes=("_internal", "_pb"),
        indicator="_source",
        validate="one_to_one",
    )

    source = merged["_source"].astype(str)
    internal_present = source.isin(["left_only", "both"]).to_numpy()
    pb_present = source.isin(["right_only", "both"]).to_numpy()

    for column in COMPARED:
        for side in ("internal", "pb"):
            merged[f"{column}_{side}"] = merged[f"{column}_{side}"].astype(float).fillna(0.0)

    currency_internal = merged["currency_internal"]
    currency_pb = merged["currency_pb"]
    merged["currency"] = currency_internal.fillna(currency_pb)
    currency_disagrees = (
        internal_present & pb_present & (currency_internal != currency_pb)
    ).to_numpy()

    for column in COMPARED:
        merged[f"{column}_diff"] = merged[f"{column}_internal"] - merged[f"{column}_pb"]
    merged["abs_market_value_diff"] = merged["market_value_diff"].abs()

    merged["break_type"] = _classify(
        merged, internal_present, pb_present, currency_disagrees, tolerances
    )

    return (
        merged[OUTPUT_COLUMNS]
        .sort_values(["as_of_date", "account_id", "cusip"])
        .reset_index(drop=True)
    )


def _classify(
    merged: pd.DataFrame,
    internal_present: np.ndarray,
    pb_present: np.ndarray,
    currency_disagrees: np.ndarray,
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

    # A book of record carries zero rows for positions that have been closed,
    # and a broker correctly drops them. One side flat and the other absent is
    # agreement, so it is matched ahead of the missing-side rules rather than
    # sitting in the queue forever at nil value.
    flat = np.isclose(
        merged["quantity_internal"].to_numpy() + merged["quantity_pb"].to_numpy(),
        0.0,
        rtol=0.0,
        atol=tolerances.quantity_abs,
    ) & np.isclose(
        merged["market_value_internal"].to_numpy() + merged["market_value_pb"].to_numpy(),
        0.0,
        rtol=0.0,
        atol=tolerances.market_value_abs,
    )
    one_side_only = internal_present ^ pb_present

    conditions = [
        one_side_only & flat,
        internal_present & ~pb_present,
        ~internal_present & pb_present,
        currency_disagrees,
        ~quantity_matches,
        ~price_matches,
        ~market_value_matches,
    ]
    choices = [
        MATCHED,
        MISSING_IN_PB,
        MISSING_IN_INTERNAL,
        CURRENCY_BREAK,
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
