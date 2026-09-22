"""End to end: generate feeds, run the pipeline, check what comes out."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from recon.config import BREAK_TYPES, MATCHED, QUANTITY_BREAK
from recon.generate import generate
from recon.normalize import load_account_map
from recon.pipeline import discover_dates, reconcile_date, run

END_DATE = pd.Timestamp("2026-09-18")
DAYS = 20


@pytest.fixture(scope="session")
def generated(tmp_path_factory) -> tuple[Path, Path]:
    """One generated dataset shared by the tests in this module."""
    root = tmp_path_factory.mktemp("recon")
    data_dir = root / "data"
    reference_dir = root / "reference"
    generate(data_dir=data_dir, reference_dir=reference_dir, end_date=END_DATE, days=DAYS)
    return data_dir, reference_dir


@pytest.fixture(scope="session")
def outputs(generated) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dir, reference_dir = generated
    return run(
        data_dir=data_dir,
        account_map_path=reference_dir / "account_map.csv",
        security_master_path=reference_dir / "security_master.csv",
    )


class TestDiscovery:
    def test_finds_every_day_that_has_both_feeds(self, generated):
        data_dir, _ = generated
        dates = discover_dates(data_dir)
        assert len(dates) == DAYS
        assert dates[-1] == END_DATE

    def test_ignores_a_day_the_broker_has_not_sent(self, generated, tmp_path):
        data_dir, _ = generated
        partial = tmp_path / "partial"
        (partial / "raw" / "internal").mkdir(parents=True)
        (partial / "raw" / "prime_broker").mkdir(parents=True)
        source = data_dir / "raw" / "internal" / f"internal_positions_{END_DATE:%Y%m%d}.csv"
        (partial / "raw" / "internal" / source.name).write_text(source.read_text())

        assert discover_dates(partial) == []


class TestHistory:
    def test_the_history_holds_only_breaks(self, outputs):
        history, _ = outputs
        assert len(history) > 0
        assert MATCHED not in set(history["break_type"])

    def test_the_generator_produces_every_break_type(self, outputs):
        history, _ = outputs
        assert set(history["break_type"]) == set(BREAK_TYPES)

    def test_every_break_is_aged(self, outputs):
        history, _ = outputs
        assert history["age_business_days"].notna().all()
        assert history["age_bucket"].notna().all()
        assert (history["age_business_days"] >= 0).all()

    def test_a_break_is_never_older_than_the_history(self, outputs):
        history, _ = outputs
        assert history["age_business_days"].max() < DAYS

    def test_every_break_carries_a_security_description(self, outputs):
        history, _ = outputs
        assert history["description"].notna().all()

    def test_counts_reconcile_with_the_daily_summary(self, outputs):
        history, summary = outputs
        per_day = history.groupby("as_of_date").size()
        expected = summary.set_index("as_of_date")["breaks"]
        pd.testing.assert_series_equal(per_day, expected, check_names=False, check_dtype=False)


class TestSummary:
    def test_one_row_per_business_day(self, outputs):
        _, summary = outputs
        assert len(summary) == DAYS

    def test_match_rate_is_high_but_not_perfect(self, outputs):
        _, summary = outputs
        assert summary["match_rate"].between(0.80, 0.999).all()

    def test_matched_and_breaks_account_for_every_position(self, outputs):
        _, summary = outputs
        assert (summary["matched"] + summary["breaks"] == summary["positions"]).all()


class TestRepeatability:
    def test_running_twice_gives_the_same_answer(self, generated, outputs):
        data_dir, reference_dir = generated
        history, _ = outputs
        again, _ = run(
            data_dir=data_dir,
            account_map_path=reference_dir / "account_map.csv",
            security_master_path=reference_dir / "security_master.csv",
            write=False,
        )
        pd.testing.assert_frame_equal(history, again)

    def test_as_of_truncates_the_history(self, generated):
        data_dir, reference_dir = generated
        cutoff = pd.Timestamp("2026-09-04")
        history, summary = run(
            data_dir=data_dir,
            as_of=cutoff,
            account_map_path=reference_dir / "account_map.csv",
            security_master_path=reference_dir / "security_master.csv",
            write=False,
        )
        assert history["as_of_date"].max() <= cutoff
        assert summary["as_of_date"].max() <= cutoff

    def test_a_data_directory_with_no_feeds_fails_loudly(self, tmp_path, generated):
        _, reference_dir = generated
        with pytest.raises(FileNotFoundError):
            run(
                data_dir=tmp_path,
                account_map_path=reference_dir / "account_map.csv",
                write=False,
            )


class TestKnownDifference:
    """A break planted by hand, to prove the pipeline finds what it should."""

    def test_a_single_planted_quantity_difference_is_the_only_break(self, tmp_path):
        data_dir = tmp_path / "data"
        internal_dir = data_dir / "raw" / "internal"
        pb_dir = data_dir / "raw" / "prime_broker"
        internal_dir.mkdir(parents=True)
        pb_dir.mkdir(parents=True)

        account_map_path = tmp_path / "account_map.csv"
        pd.DataFrame({"account_id": ["EQ-LC-01"], "pb_account_code": ["EQLC01"]}).to_csv(
            account_map_path, index=False
        )

        (internal_dir / "internal_positions_20260918.csv").write_text(
            "as_of_date,account_code,cusip,quantity,price,market_value,currency\n"
            "2026-09-18,EQ-LC-01,aaa111aaa,1000.00,25.0000,25000.00,USD\n"
            "2026-09-18,EQ-LC-01,BBB222BBB,2000.00,10.0000,20000.00,USD\n"
        )
        # The broker is 100 shares light on the first position.
        (pb_dir / "pb_positions_20260918.csv").write_text(
            "business_date,account,cusip,long_short,quantity,price,market_value,ccy\n"
            '09/18/2026,EQLC01,AAA111AAA,L,"900.00",25.0000,"22,500.00",USD\n'
            '09/18/2026,EQLC01,BBB222BBB,L,"2,000.00",10.0000,"20,000.00",USD\n'
        )

        recon = reconcile_date(
            data_dir, pd.Timestamp("2026-09-18"), load_account_map(account_map_path)
        )
        breaks = recon[recon["break_type"] != MATCHED]

        assert len(breaks) == 1
        assert breaks.iloc[0]["cusip"] == "AAA111AAA"
        assert breaks.iloc[0]["break_type"] == QUANTITY_BREAK
        assert breaks.iloc[0]["quantity_diff"] == 100.0
        assert breaks.iloc[0]["market_value_diff"] == 2_500.0
