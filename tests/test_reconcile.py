"""Classification rules, including the order they are applied in."""

from __future__ import annotations

import pytest

from recon.config import (
    MARKET_VALUE_BREAK,
    MATCHED,
    MISSING_IN_INTERNAL,
    MISSING_IN_PB,
    PRICE_BREAK,
    QUANTITY_BREAK,
    Tolerances,
)
from recon.reconcile import breaks_only, match_rate, reconcile
from tests.conftest import frame, position


def classify(internal_side, pb_side, tolerances=None) -> str:
    """Reconcile one position and return the label it was given."""
    kwargs = {"tolerances": tolerances} if tolerances else {}
    result = reconcile(frame(internal_side), frame(pb_side), **kwargs)
    assert len(result) == 1
    return result.loc[0, "break_type"]


class TestMatching:
    def test_identical_positions_match(self):
        assert classify(position("AAA", 1000, 25.0), position("AAA", 1000, 25.0)) == MATCHED

    def test_a_penny_of_market_value_rounding_is_not_a_break(self):
        internal = position("AAA", 1000, 25.0, market_value=25_000.00)
        pb = position("AAA", 1000, 25.0, market_value=25_000.02)
        assert classify(internal, pb) == MATCHED

    def test_a_price_difference_inside_tolerance_is_not_a_break(self):
        # One basis point is the tolerance, so half of one is a match.
        internal = position("AAA", 1000, 100.0)
        pb = position("AAA", 1000, 100.005, market_value=100_000.0)
        assert classify(internal, pb) == MATCHED


class TestBreakTypes:
    def test_a_quantity_difference_is_a_quantity_break(self):
        internal = position("AAA", 1000, 25.0)
        pb = position("AAA", 900, 25.0)
        assert classify(internal, pb) == QUANTITY_BREAK

    def test_a_price_difference_beyond_tolerance_is_a_price_break(self):
        internal = position("AAA", 1000, 100.0)
        pb = position("AAA", 1000, 100.5, market_value=100_500.0)
        assert classify(internal, pb) == PRICE_BREAK

    def test_matching_quantity_and_price_with_a_different_value_is_a_value_break(self):
        # What an FX rate difference looks like: the position agrees, the
        # translated value does not.
        internal = position("AAA", 1000, 100.0, market_value=108_350.0, currency="EUR")
        pb = position("AAA", 1000, 100.0, market_value=108_025.0, currency="EUR")
        assert classify(internal, pb) == MARKET_VALUE_BREAK

    def test_a_position_only_we_have_is_missing_at_the_broker(self):
        result = reconcile(frame(position("AAA", 1000, 25.0)), frame())
        assert result.loc[0, "break_type"] == MISSING_IN_PB

    def test_a_position_only_the_broker_has_is_missing_internally(self):
        result = reconcile(frame(), frame(position("AAA", 1000, 25.0)))
        assert result.loc[0, "break_type"] == MISSING_IN_INTERNAL


class TestClassificationOrder:
    """A missing side must not be reported as a difference against zero."""

    def test_a_missing_broker_position_is_not_a_quantity_break(self):
        result = reconcile(frame(position("AAA", 1000, 25.0)), frame())
        assert result.loc[0, "break_type"] == MISSING_IN_PB
        assert result.loc[0, "quantity_pb"] == 0.0

    def test_a_quantity_break_wins_over_the_price_break_it_causes(self):
        internal = position("AAA", 1000, 25.0)
        pb = position("AAA", 900, 26.0)
        assert classify(internal, pb) == QUANTITY_BREAK


class TestDifferences:
    def test_the_value_difference_on_a_missing_position_is_the_whole_position(self):
        result = reconcile(frame(position("AAA", 1000, 25.0)), frame())
        assert result.loc[0, "market_value_diff"] == 25_000.0
        assert result.loc[0, "abs_market_value_diff"] == 25_000.0

    def test_differences_are_internal_minus_broker(self):
        internal = position("AAA", 1000, 25.0)
        pb = position("AAA", 900, 25.0, market_value=22_500.0)
        result = reconcile(frame(internal), frame(pb))
        assert result.loc[0, "quantity_diff"] == 100.0
        assert result.loc[0, "market_value_diff"] == 2_500.0


class TestTolerances:
    def test_tolerances_are_configurable(self):
        internal = position("AAA", 1000, 100.0)
        pb = position("AAA", 1000, 100.5, market_value=100_500.0)
        loose = Tolerances(price_rel=1e-2, market_value_rel=1e-2)
        assert classify(internal, pb, tolerances=loose) == MATCHED

    def test_loosening_one_tolerance_falls_through_to_the_next_rule(self):
        # Widening the price tolerance stops this being a price break, but the
        # value that price produced is still half a percent out, so it is still
        # a break. Loosening a tolerance must not make the money disappear.
        internal = position("AAA", 1000, 100.0)
        pb = position("AAA", 1000, 100.5, market_value=100_500.0)
        loose_price_only = Tolerances(price_rel=1e-2)
        assert classify(internal, pb, tolerances=loose_price_only) == MARKET_VALUE_BREAK

    def test_the_price_tolerance_scales_with_the_internal_price(self):
        # Fifty cents is well beyond one basis point on a $25 stock and well
        # inside it on a $25,000 bond future.
        cheap = classify(position("AAA", 1, 25.0), position("AAA", 1, 25.5, market_value=25.5))
        rich = classify(
            position("BBB", 1, 25_000.0),
            position("BBB", 1, 25_000.5, market_value=25_000.5),
        )
        assert cheap == PRICE_BREAK
        assert rich == MATCHED


class TestSummaries:
    def test_breaks_only_drops_the_matches(self):
        internal = frame(position("AAA", 1000, 25.0), position("BBB", 500, 10.0))
        pb = frame(position("AAA", 1000, 25.0), position("BBB", 400, 10.0))
        result = reconcile(internal, pb)

        assert len(result) == 2
        assert len(breaks_only(result)) == 1
        assert breaks_only(result).loc[0, "cusip"] == "BBB"

    def test_match_rate_counts_the_union_of_both_sides(self):
        internal = frame(position("AAA", 1000, 25.0), position("BBB", 500, 10.0))
        pb = frame(position("AAA", 1000, 25.0))
        assert match_rate(reconcile(internal, pb)) == pytest.approx(0.5)

    def test_match_rate_of_an_empty_reconciliation_is_one(self):
        assert match_rate(reconcile(frame(), frame())) == 1.0
