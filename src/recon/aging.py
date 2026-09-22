"""Age each break in business days.

Two things decide a break's age, and they are deliberately separate.

Identity: a break is the same item for as long as it stays open on consecutive
days that we actually reconciled. Run detection walks the sequence of
reconciled dates rather than the calendar, so neither a market holiday nor a
broker file that never arrived splits one break into two. Inferring identity
from calendar gaps was the earlier approach and it reset every open break's
clock on Thanksgiving.

Duration: once a run's first date is known, age is trading days elapsed on the
NYSE calendar in `recon.calendar`. Elapsed time is the right unit to report to
ops whether or not we managed to reconcile every day in between.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from recon.calendar import business_days_between
from recon.config import AGE_BUCKETS, BREAK_KEY

AGED_COLUMNS = ["first_seen_date", "age_business_days", "age_bucket", "is_new"]


class UnknownReconciliationDateError(ValueError):
    """History holds a date that is not in the list of dates reconciled."""


def add_break_age(
    history: pd.DataFrame,
    reconciled_dates: Iterable[pd.Timestamp] | None = None,
) -> pd.DataFrame:
    """Add first seen date, business day age and an age bucket.

    `history` is every daily break frame stacked together. `reconciled_dates` is
    every date the pipeline actually ran, which is not the same as the dates
    present in `history`: a day on which everything matched contributes no rows.
    Passing it is what keeps a clean day from looking like a gap. It defaults to
    the dates found in `history`, which is only correct when every day had at
    least one break.
    """
    if history.empty:
        empty = history.copy()
        empty["first_seen_date"] = pd.Series(dtype="datetime64[ns]")
        empty["age_business_days"] = pd.Series(dtype="int64")
        empty["age_bucket"] = pd.Categorical([], categories=AGE_BUCKETS, ordered=True)
        empty["is_new"] = pd.Series(dtype="bool")
        return empty

    df = history.sort_values([*BREAK_KEY, "as_of_date"]).reset_index(drop=True)
    sequence = _date_sequence(df, reconciled_dates)

    # Position in the reconciled sequence, so adjacency means "the next run we
    # did", not "the next weekday".
    position = df["as_of_date"].map(sequence)
    previous = position.groupby([df[column] for column in BREAK_KEY], observed=True).shift(1)
    starts_run = previous.isna() | ((position - previous) > 1)

    run_id = starts_run.groupby([df[column] for column in BREAK_KEY], observed=True).cumsum()
    df["first_seen_date"] = df.groupby([*BREAK_KEY, run_id], observed=True)["as_of_date"].transform(
        "min"
    )

    age = business_days_between(df["first_seen_date"], df["as_of_date"])
    df["age_business_days"] = age.astype("int64")
    df["age_bucket"] = _bucket(df["age_business_days"])
    df["is_new"] = df["age_business_days"] == 0

    return df.sort_values(["as_of_date", "account_id", "cusip"]).reset_index(drop=True)


def _date_sequence(
    history: pd.DataFrame,
    reconciled_dates: Iterable[pd.Timestamp] | None,
) -> dict[pd.Timestamp, int]:
    """Map each reconciled date to its position in the sequence."""
    if reconciled_dates is None:
        dates = sorted(pd.Series(history["as_of_date"].unique()))
    else:
        dates = sorted({pd.Timestamp(date) for date in reconciled_dates})

    sequence = {pd.Timestamp(date): index for index, date in enumerate(dates)}
    unknown = sorted(set(history["as_of_date"].unique()) - set(sequence))
    if unknown:
        raise UnknownReconciliationDateError(
            "break history holds dates that are not in the reconciled set: "
            f"{[str(pd.Timestamp(date).date()) for date in unknown]}"
        )
    return sequence


def _bucket(age: pd.Series) -> pd.Categorical:
    labels = np.select(
        [age <= 1, age <= 5, age <= 10],
        AGE_BUCKETS[:3],
        default=AGE_BUCKETS[3],
    )
    return pd.Categorical(labels, categories=AGE_BUCKETS, ordered=True)
