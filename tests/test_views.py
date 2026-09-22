"""What the dashboard puts on screen, tested without starting one."""

from __future__ import annotations

import pandas as pd
import pytest

from app import views
from recon.config import AGE_BUCKETS, BREAK_TYPES, MISSING_IN_PB, PRICE_BREAK, QUANTITY_BREAK


@pytest.fixture
def day() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _break("EQ-LC-01", "AAA", QUANTITY_BREAK, age=0, mv_diff=50_000.0),
            _break("EQ-LC-01", "BBB", QUANTITY_BREAK, age=3, mv_diff=-120_000.0),
            _break("EQ-LC-02", "CCC", PRICE_BREAK, age=8, mv_diff=9_000.0),
            _break("FI-CORE-01", "DDD", MISSING_IN_PB, age=14, mv_diff=250_000.0),
        ]
    )


def _break(account, cusip, break_type, age, mv_diff) -> dict:
    bucket = (
        AGE_BUCKETS[0]
        if age <= 1
        else AGE_BUCKETS[1]
        if age <= 5
        else (AGE_BUCKETS[2] if age <= 10 else AGE_BUCKETS[3])
    )
    return {
        "as_of_date": pd.Timestamp("2026-09-18"),
        "account_id": account,
        "cusip": cusip,
        "description": f"{cusip} Inc",
        "break_type": break_type,
        "age_business_days": age,
        "age_bucket": bucket,
        "is_new": age == 0,
        "quantity_internal": 1000.0,
        "quantity_pb": 900.0,
        "quantity_diff": 100.0,
        "market_value_diff": mv_diff,
        "abs_market_value_diff": abs(mv_diff),
    }


@pytest.fixture
def summary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "as_of_date": pd.Timestamp("2026-09-18"),
                "positions": 100,
                "matched": 96,
                "breaks": 4,
                "match_rate": 0.96,
                "abs_market_value_diff": 429_000.0,
            }
        ]
    )


class TestKpis:
    def test_counts_and_money(self, day, summary):
        result = views.kpis(day, summary, pd.Timestamp("2026-09-18"))
        assert result["open_breaks"] == 4
        assert result["new_today"] == 1
        assert result["abs_market_value_diff"] == pytest.approx(429_000.0)
        assert result["match_rate"] == pytest.approx(0.96)

    def test_aged_counts_only_breaks_past_the_threshold(self, day, summary):
        # Ages are 0, 3, 8 and 14 business days; the threshold is 5.
        assert views.kpis(day, summary, pd.Timestamp("2026-09-18"))["aged"] == 2

    def test_an_empty_day_is_all_zeros(self, summary):
        result = views.kpis(day=pd.DataFrame(), summary=summary, as_of="2026-09-18")
        assert result["open_breaks"] == 0
        assert result["abs_market_value_diff"] == 0.0

    def test_a_date_with_no_summary_row_gives_no_match_rate(self, day, summary):
        result = views.kpis(day, summary, pd.Timestamp("2026-09-17"))
        assert pd.isna(result["match_rate"])


class TestBreaksByType:
    def test_every_type_is_present_in_severity_order(self, day):
        result = views.breaks_by_type(day)
        assert result["break_type"].tolist() == BREAK_TYPES

    def test_counts_and_money_roll_up(self, day):
        result = views.breaks_by_type(day).set_index("break_type")
        assert result.loc[QUANTITY_BREAK, "breaks"] == 2
        assert result.loc[QUANTITY_BREAK, "abs_market_value_diff"] == pytest.approx(170_000.0)
        assert result.loc["MARKET_VALUE_BREAK", "breaks"] == 0

    def test_money_uses_absolute_differences_so_they_do_not_net_off(self, day):
        # The two quantity breaks are +50k and -120k. A queue that reported
        # -70k of exposure would be understating the work.
        result = views.breaks_by_type(day).set_index("break_type")
        assert result.loc[QUANTITY_BREAK, "abs_market_value_diff"] == pytest.approx(170_000.0)


class TestBreaksByAge:
    def test_every_bucket_is_present_even_when_empty(self, day):
        result = views.breaks_by_age(day)
        assert result["age_bucket"].tolist() == AGE_BUCKETS

    def test_breaks_land_in_the_right_buckets(self, day):
        result = views.breaks_by_age(day).set_index("age_bucket")["breaks"]
        assert result.tolist() == [1, 1, 1, 1]


class TestFilters:
    def test_filtering_by_account(self, day):
        assert len(views.apply_filters(day, selected_accounts=["EQ-LC-01"])) == 2

    def test_filtering_by_break_type(self, day):
        assert len(views.apply_filters(day, selected_types=[PRICE_BREAK])) == 1

    def test_filters_combine(self, day):
        result = views.apply_filters(
            day, selected_accounts=["EQ-LC-01"], selected_types=[QUANTITY_BREAK]
        )
        assert len(result) == 2

    def test_no_filter_returns_everything(self, day):
        assert len(views.apply_filters(day, [], [])) == 4


class TestTrend:
    def test_one_row_per_date_in_order(self, day):
        two_days = pd.concat(
            [day, day.assign(as_of_date=pd.Timestamp("2026-09-17")).head(2)],
            ignore_index=True,
        )
        result = views.break_trend(two_days)
        assert result["as_of_date"].tolist() == [
            pd.Timestamp("2026-09-17"),
            pd.Timestamp("2026-09-18"),
        ]
        assert result["breaks"].tolist() == [2, 4]


class TestExceptionQueue:
    def test_biggest_money_first(self, day):
        result = views.exception_queue(day)
        assert result["CUSIP"].tolist() == ["DDD", "BBB", "AAA", "CCC"]

    def test_limit_takes_the_largest(self, day):
        assert views.exception_queue(day, limit=2)["CUSIP"].tolist() == ["DDD", "BBB"]

    def test_an_empty_day_still_has_the_columns(self):
        result = views.exception_queue(pd.DataFrame())
        assert result.empty
        assert "MV diff USD" in result.columns

    def test_break_types_can_be_relabelled_for_display(self, day):
        result = views.exception_queue(day, labels={QUANTITY_BREAK: "Quantity"})
        assert "Quantity" in set(result["Break"])
        # An unmapped type falls back to its raw name rather than going blank.
        assert MISSING_IN_PB in set(result["Break"])
