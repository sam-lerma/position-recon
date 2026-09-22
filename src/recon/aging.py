"""Age each break in business days.

A break is the same item for as long as it stays open on consecutive business
days. Close it and reopen it later and the clock restarts, which is what ops
expect when they ask how long something has been outstanding.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from recon.config import AGE_BUCKETS, BREAK_KEY

AGED_COLUMNS = ["first_seen_date", "age_business_days", "age_bucket", "is_new"]


def add_break_age(history: pd.DataFrame) -> pd.DataFrame:
    """Add first seen date, business day age and an age bucket.

    `history` is every daily break frame stacked together. Aging has to be
    computed over history rather than one day at a time, which is why the daily
    run reloads what it has already written.
    """
    if history.empty:
        empty = history.copy()
        empty["first_seen_date"] = pd.Series(dtype="datetime64[ns]")
        empty["age_business_days"] = pd.Series(dtype="int64")
        empty["age_bucket"] = pd.Categorical([], categories=AGE_BUCKETS, ordered=True)
        empty["is_new"] = pd.Series(dtype="bool")
        return empty

    df = history.sort_values([*BREAK_KEY, "as_of_date"]).reset_index(drop=True)

    # A gap of more than one business day since this key was last seen means the
    # break closed and a new one opened.
    previous = df.groupby(BREAK_KEY, observed=True)["as_of_date"].shift(1)
    gap = _business_days_between(previous, df["as_of_date"])
    starts_run = previous.isna() | (gap > 1)

    run_id = starts_run.groupby([df[column] for column in BREAK_KEY], observed=True).cumsum()
    df["first_seen_date"] = df.groupby([*BREAK_KEY, run_id], observed=True)["as_of_date"].transform(
        "min"
    )

    age = _business_days_between(df["first_seen_date"], df["as_of_date"])
    df["age_business_days"] = age.astype("int64")
    df["age_bucket"] = _bucket(df["age_business_days"])
    df["is_new"] = df["age_business_days"] == 0

    return df.sort_values(["as_of_date", "account_id", "cusip"]).reset_index(drop=True)


def _business_days_between(start: pd.Series, end: pd.Series) -> pd.Series:
    """Business days from start to end, NaN where start is missing."""
    out = np.full(len(start), np.nan)
    present = start.notna().to_numpy()
    if present.any():
        start_days = start[present].to_numpy().astype("datetime64[D]")
        end_days = end[present].to_numpy().astype("datetime64[D]")
        out[present] = np.busday_count(start_days, end_days)
    return pd.Series(out, index=start.index)


def _bucket(age: pd.Series) -> pd.Categorical:
    labels = np.select(
        [age <= 1, age <= 5, age <= 10],
        AGE_BUCKETS[:3],
        default=AGE_BUCKETS[3],
    )
    return pd.Categorical(labels, categories=AGE_BUCKETS, ordered=True)
