"""The business day calendar the aging runs on.

`np.busday_count` with no calendar counts Monday to Friday, which is not a
trading calendar: it ages a break across Thanksgiving and Christmas as though
the desk were open. This module supplies the NYSE holiday schedule so that
"business day" means what ops mean by it.

The observance rules are the exchange's, not the federal ones. The NYSE is open
on Columbus Day and Veterans Day and closed on Good Friday, so the federal
calendar in pandas is the wrong list. New Year's Day moves to the Monday when it
falls on a Sunday but does not move back to the Friday when it falls on a
Saturday, which is why it uses a different observance from Independence Day and
Christmas.

Production would take this from the exchange calendar service or a maintained
package rather than a rule set pinned here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    GoodFriday,
    Holiday,
    USLaborDay,
    USMartinLutherKingJr,
    USMemorialDay,
    USPresidentsDay,
    USThanksgivingDay,
    nearest_workday,
    sunday_to_monday,
)

CALENDAR_START = "2015-01-01"
CALENDAR_END = "2035-12-31"


class NYSEHolidayCalendar(AbstractHolidayCalendar):
    """The regular NYSE holiday schedule.

    Ad hoc closures, such as a state funeral or a hurricane, are not rules and
    would have to be added as explicit dates.
    """

    rules = [
        Holiday("New Year's Day", month=1, day=1, observance=sunday_to_monday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday(
            "Juneteenth",
            month=6,
            day=19,
            start_date=pd.Timestamp("2022-06-20"),
            observance=nearest_workday,
        ),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("Christmas Day", month=12, day=25, observance=nearest_workday),
    ]


def market_holidays(
    start: str | pd.Timestamp = CALENDAR_START,
    end: str | pd.Timestamp = CALENDAR_END,
) -> pd.DatetimeIndex:
    return NYSEHolidayCalendar().holidays(pd.Timestamp(start), pd.Timestamp(end))


def business_day_calendar(
    start: str | pd.Timestamp = CALENDAR_START,
    end: str | pd.Timestamp = CALENDAR_END,
) -> np.busdaycalendar:
    holidays = market_holidays(start, end).to_numpy().astype("datetime64[D]")
    return np.busdaycalendar(holidays=holidays)


TRADING_CALENDAR = business_day_calendar()


def business_days(end: pd.Timestamp, periods: int) -> list[pd.Timestamp]:
    """The `periods` trading days ending on or before `end`."""
    end_day = np.datetime64(pd.Timestamp(end).date(), "D")
    last = np.busday_offset(end_day, 0, roll="backward", busdaycal=TRADING_CALENDAR)
    offsets = np.arange(-(periods - 1), 1)
    days = np.busday_offset(last, offsets, roll="backward", busdaycal=TRADING_CALENDAR)
    return [pd.Timestamp(day) for day in days]


def trading_days_between(
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> list[pd.Timestamp]:
    """Every trading day from start to end inclusive."""
    days = np.arange(
        np.datetime64(pd.Timestamp(start).date(), "D"),
        np.datetime64(pd.Timestamp(end).date(), "D") + 1,
    )
    return [pd.Timestamp(day) for day in days[np.is_busday(days, busdaycal=TRADING_CALENDAR)]]


def business_days_between(start: pd.Series, end: pd.Series) -> pd.Series:
    """Trading days from start to end, NaN where start is missing."""
    out = np.full(len(start), np.nan)
    present = start.notna().to_numpy()
    if present.any():
        start_days = start[present].to_numpy().astype("datetime64[D]")
        end_days = end[present].to_numpy().astype("datetime64[D]")
        out[present] = np.busday_count(start_days, end_days, busdaycal=TRADING_CALENDAR)
    return pd.Series(out, index=start.index)
