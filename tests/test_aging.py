"""Aging in business days, including what happens over a weekend."""

from __future__ import annotations

import pandas as pd
import pytest

from recon.aging import add_break_age
from recon.config import QUANTITY_BREAK


def history(*rows: tuple[str, str, str]) -> pd.DataFrame:
    """Break history from (date, account, cusip) triples."""
    return pd.DataFrame(
        [
            {
                "as_of_date": pd.Timestamp(date),
                "account_id": account,
                "cusip": cusip,
                "break_type": QUANTITY_BREAK,
                "abs_market_value_diff": 1000.0,
            }
            for date, account, cusip in rows
        ]
    )


def ages(result: pd.DataFrame) -> list[int]:
    return result.sort_values("as_of_date")["age_business_days"].tolist()


class TestAge:
    def test_a_break_seen_for_the_first_time_is_zero_days_old(self):
        result = add_break_age(history(("2026-09-14", "A", "X")))
        assert result.loc[0, "age_business_days"] == 0
        assert bool(result.loc[0, "is_new"]) is True

    def test_age_increments_on_consecutive_business_days(self):
        result = add_break_age(
            history(
                ("2026-09-14", "A", "X"),
                ("2026-09-15", "A", "X"),
                ("2026-09-16", "A", "X"),
            )
        )
        assert ages(result) == [0, 1, 2]
        assert result["first_seen_date"].nunique() == 1

    def test_a_weekend_does_not_age_a_break(self):
        # Friday to Monday is one business day, not three.
        result = add_break_age(history(("2026-09-18", "A", "X"), ("2026-09-21", "A", "X")))
        assert ages(result) == [0, 1]

    def test_closing_and_reopening_restarts_the_clock(self):
        # Present Monday and Tuesday, absent Wednesday, back on Thursday.
        result = add_break_age(
            history(
                ("2026-09-14", "A", "X"),
                ("2026-09-15", "A", "X"),
                ("2026-09-17", "A", "X"),
            )
        )
        assert ages(result) == [0, 1, 0]
        assert result["first_seen_date"].nunique() == 2

    def test_each_break_is_aged_on_its_own(self):
        result = add_break_age(
            history(
                ("2026-09-14", "A", "X"),
                ("2026-09-15", "A", "X"),
                ("2026-09-15", "A", "Y"),
            )
        )
        by_cusip = result.set_index(["cusip", "as_of_date"])["age_business_days"]
        assert by_cusip[("X", pd.Timestamp("2026-09-15"))] == 1
        assert by_cusip[("Y", pd.Timestamp("2026-09-15"))] == 0

    def test_the_same_security_in_two_accounts_is_two_breaks(self):
        result = add_break_age(
            history(
                ("2026-09-14", "A", "X"),
                ("2026-09-15", "A", "X"),
                ("2026-09-15", "B", "X"),
            )
        )
        by_account = result.set_index(["account_id", "as_of_date"])["age_business_days"]
        assert by_account[("A", pd.Timestamp("2026-09-15"))] == 1
        assert by_account[("B", pd.Timestamp("2026-09-15"))] == 0


class TestBuckets:
    @pytest.mark.parametrize(
        ("days_open", "expected"),
        [
            (1, "0-1d"),
            (2, "0-1d"),
            (3, "2-5d"),
            (6, "2-5d"),
            (7, "6-10d"),
            (11, "6-10d"),
            (12, "11d+"),
            (20, "11d+"),
        ],
    )
    def test_bucket_boundaries(self, days_open: int, expected: str):
        dates = pd.bdate_range("2026-08-03", periods=days_open)
        result = add_break_age(history(*[(str(d.date()), "A", "X") for d in dates]))
        assert result.iloc[-1]["age_bucket"] == expected

    def test_buckets_sort_oldest_last(self):
        dates = pd.bdate_range("2026-08-03", periods=15)
        result = add_break_age(history(*[(str(d.date()), "A", "X") for d in dates]))
        assert list(result["age_bucket"].cat.categories) == ["0-1d", "2-5d", "6-10d", "11d+"]
        assert result["age_bucket"].cat.ordered


class TestEdges:
    def test_an_empty_history_keeps_its_shape(self):
        result = add_break_age(history())
        assert len(result) == 0
        assert {"first_seen_date", "age_business_days", "age_bucket", "is_new"} <= set(
            result.columns
        )
