"""The daily run: load both feeds, reconcile, age the breaks, write the outputs.

Aging is computed over the full history rather than one day at a time, so the
run reloads every date it can see and rewrites the history file. At this volume
that is simpler and cheaper than maintaining incremental state, and it makes a
rerun idempotent. At production volume this becomes an upsert into a table.

    python -m recon.pipeline
    python -m recon.pipeline --as-of 2026-09-18
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import pandas as pd

from recon.aging import add_break_age
from recon.calendar import trading_days_between
from recon.config import (
    ACCOUNT_MAP_PATH,
    DEFAULT_DATA_DIR,
    DEFAULT_TOLERANCES,
    MATCHED,
    SECURITY_MASTER_PATH,
    Tolerances,
)
from recon.normalize import load_account_map, normalize_internal, normalize_pb
from recon.reconcile import breaks_only, reconcile

logger = logging.getLogger(__name__)

DATE_IN_FILENAME = re.compile(r"(\d{8})\.csv$")

# CUSIPs are frequently all numeric, and most of the Treasury range is. Without
# this, 037833100 is inferred as an integer and reaches the dashboard as
# 37833100, matching nothing in any security master. Worse, inference is per
# file, so one side can keep the zero while the other loses it and a clean
# position becomes two breaks.
IDENTIFIER_DTYPES = {
    "cusip": str,
    "account_code": str,
    "account": str,
}

# Summarized per account, not per day. A dashboard filtered to one account
# needs that account's match rate, and a firm-wide figure beside a filtered
# break count is worse than no figure at all.
SUMMARY_COLUMNS = [
    "as_of_date",
    "account_id",
    "positions",
    "matched",
    "breaks",
    "match_rate",
    "abs_market_value_diff",
]


def discover_dates(data_dir: Path | str) -> list[pd.Timestamp]:
    """Dates that have both an internal and a broker file."""
    data_dir = Path(data_dir)
    internal = _dates_in(data_dir / "raw" / "internal")
    pb = _dates_in(data_dir / "raw" / "prime_broker")
    return sorted(internal & pb)


def _dates_in(directory: Path) -> set[pd.Timestamp]:
    if not directory.exists():
        return set()
    found = set()
    for path in directory.glob("*.csv"):
        match = DATE_IN_FILENAME.search(path.name)
        if match:
            found.add(pd.Timestamp(match.group(1)))
    return found


def reconcile_date(
    data_dir: Path | str,
    date: pd.Timestamp,
    account_map: pd.DataFrame,
    tolerances: Tolerances = DEFAULT_TOLERANCES,
) -> pd.DataFrame:
    """Normalize and reconcile a single business day."""
    data_dir = Path(data_dir)
    internal_path = data_dir / "raw" / "internal" / f"internal_positions_{date:%Y%m%d}.csv"
    pb_path = data_dir / "raw" / "prime_broker" / f"pb_positions_{date:%Y%m%d}.csv"

    internal = normalize_internal(pd.read_csv(internal_path, dtype=IDENTIFIER_DTYPES), account_map)
    pb = normalize_pb(pd.read_csv(pb_path, dtype=IDENTIFIER_DTYPES), account_map)
    return reconcile(internal, pb, tolerances)


def run(
    data_dir: Path | str = DEFAULT_DATA_DIR,
    as_of: pd.Timestamp | None = None,
    account_map_path: Path | str = ACCOUNT_MAP_PATH,
    security_master_path: Path | str = SECURITY_MASTER_PATH,
    tolerances: Tolerances = DEFAULT_TOLERANCES,
    write: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run every available date up to `as_of` and return (break history, summary)."""
    data_dir = Path(data_dir)
    account_map = load_account_map(account_map_path)

    dates = discover_dates(data_dir)
    if as_of is not None:
        dates = [date for date in dates if date <= pd.Timestamp(as_of)]
    if not dates:
        raise FileNotFoundError(f"no paired feed files found under {data_dir / 'raw'}")

    _warn_about_missing_days(dates)

    daily_breaks: list[pd.DataFrame] = []
    summary_rows: list[dict] = []

    for date in dates:
        recon = reconcile_date(data_dir, date, account_map, tolerances)
        breaks = breaks_only(recon)
        daily_breaks.append(breaks)
        summary_rows.extend(_summarize(recon, breaks, date))
        logger.info("%s: %s breaks of %s positions", date.date(), len(breaks), len(recon))

    # Aging needs every date we ran, not just the dates that produced breaks: a
    # day on which everything matched contributes no rows, and treating that as
    # a gap would close and reopen every break that spans it.
    history = add_break_age(pd.concat(daily_breaks, ignore_index=True), reconciled_dates=dates)
    history = _add_security_description(history, security_master_path)
    summary = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)

    if write:
        output_dir = data_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        history.to_csv(output_dir / "breaks_history.csv", index=False)
        summary.to_csv(output_dir / "daily_summary.csv", index=False)

    return history, summary


def _summarize(recon: pd.DataFrame, breaks: pd.DataFrame, date: pd.Timestamp) -> list[dict]:
    matched = recon["break_type"] == MATCHED
    per_account = recon.assign(_matched=matched).groupby("account_id", observed=True)
    money = breaks.groupby("account_id", observed=True)["abs_market_value_diff"].sum()

    rows = []
    for account_id, group in per_account:
        rows.append(
            {
                "as_of_date": date,
                "account_id": account_id,
                "positions": int(len(group)),
                "matched": int(group["_matched"].sum()),
                "breaks": int((~group["_matched"]).sum()),
                "match_rate": float(group["_matched"].mean()),
                "abs_market_value_diff": float(money.get(account_id, 0.0)),
            }
        )
    return rows


def _warn_about_missing_days(dates: list[pd.Timestamp]) -> None:
    """Say so when a trading day in the range has no paired feed.

    discover_dates only returns days where both files arrived. Dropping the rest
    silently would leave a hole in the history that nothing downstream reports.
    """
    expected = trading_days_between(dates[0], dates[-1])
    missing = sorted(set(expected) - set(dates))
    if missing:
        logger.warning(
            "no paired feed for %s trading day(s) in range: %s",
            len(missing),
            ", ".join(f"{date:%Y-%m-%d}" for date in missing),
        )


def _add_security_description(
    history: pd.DataFrame,
    security_master_path: Path | str,
) -> pd.DataFrame:
    """Left join the security master so the exception queue is readable."""
    path = Path(security_master_path)
    if not path.exists():
        return history.assign(description=history["cusip"], instrument_type="UNKNOWN")

    master = pd.read_csv(path, dtype={"cusip": str})[["cusip", "description", "instrument_type"]]
    merged = history.merge(master, on="cusip", how="left", validate="many_to_one")
    merged["description"] = merged["description"].fillna(merged["cusip"])
    merged["instrument_type"] = merged["instrument_type"].fillna("UNKNOWN")
    return merged


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--as-of", default=None, help="run dates up to this one, YYYY-MM-DD")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    history, summary = run(
        data_dir=args.data_dir,
        as_of=pd.Timestamp(args.as_of) if args.as_of else None,
    )
    latest_date = summary["as_of_date"].max()
    latest = summary[summary["as_of_date"] == latest_date]
    positions = int(latest["positions"].sum())
    matched = int(latest["matched"].sum())
    print(
        f"{summary['as_of_date'].nunique()} days reconciled through "
        f"{latest_date:%Y-%m-%d}. "
        f"{int(latest['breaks'].sum())} open breaks across "
        f"{len(latest)} accounts, "
        f"match rate {matched / positions:.2%}, "
        f"absolute difference ${latest['abs_market_value_diff'].sum():,.0f}."
    )
    print(f"wrote {Path(args.data_dir) / 'output' / 'breaks_history.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
