"""Everything the dashboard shows, as plain functions over the break history.

None of this imports Dash. The callbacks pick filters and hand them to these
functions, which keeps the part that decides what a number means testable
without starting a browser.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from recon.config import AGE_BUCKETS, BREAK_TYPES, DEFAULT_DATA_DIR

HISTORY_PATH = DEFAULT_DATA_DIR / "output" / "breaks_history.csv"
SUMMARY_PATH = DEFAULT_DATA_DIR / "output" / "daily_summary.csv"

AGED_THRESHOLD_DAYS = 5

QUEUE_COLUMNS = {
    "account_id": "Account",
    "cusip": "CUSIP",
    "description": "Security",
    "break_type": "Break",
    "age_business_days": "Age",
    "quantity_internal": "Qty internal",
    "quantity_pb": "Qty broker",
    "quantity_diff": "Qty diff",
    "market_value_diff": "MV diff USD",
}

QUEUE_NUMERIC = ["Age", "Qty internal", "Qty broker", "Qty diff", "MV diff USD"]


def load_history(path: Path | str = HISTORY_PATH) -> pd.DataFrame:
    """Read the break history and restore the types CSV loses."""
    history = pd.read_csv(path, parse_dates=["as_of_date", "first_seen_date"])
    history["age_bucket"] = pd.Categorical(
        history["age_bucket"], categories=AGE_BUCKETS, ordered=True
    )
    return history


def load_summary(path: Path | str = SUMMARY_PATH) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["as_of_date"])


def as_of_dates(history: pd.DataFrame) -> list[pd.Timestamp]:
    return sorted(history["as_of_date"].unique())


def accounts(history: pd.DataFrame) -> list[str]:
    return sorted(history["account_id"].unique())


def apply_filters(
    history: pd.DataFrame,
    selected_accounts: list[str] | None = None,
    selected_types: list[str] | None = None,
) -> pd.DataFrame:
    """Filter by account and break type. Empty or None means no filter."""
    filtered = history
    if selected_accounts:
        filtered = filtered[filtered["account_id"].isin(selected_accounts)]
    if selected_types:
        filtered = filtered[filtered["break_type"].isin(selected_types)]
    return filtered


def on_date(history: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    return history[history["as_of_date"] == pd.Timestamp(as_of)]


def kpis(day: pd.DataFrame, summary: pd.DataFrame, as_of: pd.Timestamp) -> dict:
    """The four numbers an ops lead wants before opening the queue."""
    as_of = pd.Timestamp(as_of)
    row = summary[summary["as_of_date"] == as_of]
    return {
        "open_breaks": int(len(day)),
        "new_today": int(day["is_new"].sum()) if len(day) else 0,
        "aged": int((day["age_business_days"] > AGED_THRESHOLD_DAYS).sum()) if len(day) else 0,
        "abs_market_value_diff": float(day["abs_market_value_diff"].sum()) if len(day) else 0.0,
        "match_rate": float(row["match_rate"].iloc[0]) if len(row) else float("nan"),
        "positions": int(row["positions"].iloc[0]) if len(row) else 0,
    }


def breaks_by_type(day: pd.DataFrame) -> pd.DataFrame:
    """Counts and money by break type, in severity order with zeros kept.

    The axis stays put when a filter changes, which is worth more on a
    dashboard than ordering the bars by size.
    """
    counts = day.groupby("break_type", observed=False).agg(
        breaks=("cusip", "size"),
        abs_market_value_diff=("abs_market_value_diff", "sum"),
    )
    return (
        counts.reindex(BREAK_TYPES)
        .fillna(0.0)
        .astype({"breaks": int})
        .reset_index()
        .rename(columns={"index": "break_type"})
    )


def breaks_by_age(day: pd.DataFrame) -> pd.DataFrame:
    """Counts by age bucket, every bucket present even when empty."""
    counts = (
        day.groupby("age_bucket", observed=False)["cusip"]
        .size()
        .reindex(AGE_BUCKETS)
        .fillna(0)
        .astype(int)
    )
    return counts.rename("breaks").rename_axis("age_bucket").reset_index()


def break_trend(history: pd.DataFrame) -> pd.DataFrame:
    """Open breaks per day, for the whole filtered history."""
    trend = (
        history.groupby("as_of_date")["cusip"]
        .size()
        .rename("breaks")
        .reset_index()
        .sort_values("as_of_date")
    )
    return trend.reset_index(drop=True)


def exception_queue(
    day: pd.DataFrame,
    limit: int | None = None,
    labels: dict[str, str] | None = None,
) -> pd.DataFrame:
    """The detail rows, biggest money first.

    `labels` renames the break types for display. The dashboard passes the same
    labels the charts use so the two never disagree.
    """
    if day.empty:
        return pd.DataFrame(columns=list(QUEUE_COLUMNS.values()))

    ordered = day.sort_values("abs_market_value_diff", ascending=False)
    if limit:
        ordered = ordered.head(limit)

    queue = ordered[list(QUEUE_COLUMNS)].rename(columns=QUEUE_COLUMNS)
    if labels:
        queue["Break"] = queue["Break"].map(labels).fillna(queue["Break"])
    for column in QUEUE_NUMERIC:
        queue[column] = queue[column].round(0)
    return queue.reset_index(drop=True)
