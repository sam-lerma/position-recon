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

SUMMARY_COLUMNS = [
    "as_of_date",
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

    internal = normalize_internal(pd.read_csv(internal_path), account_map)
    pb = normalize_pb(pd.read_csv(pb_path), account_map)
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

    daily_breaks: list[pd.DataFrame] = []
    summary_rows: list[dict] = []

    for date in dates:
        recon = reconcile_date(data_dir, date, account_map, tolerances)
        breaks = breaks_only(recon)
        daily_breaks.append(breaks)
        summary_rows.append(
            {
                "as_of_date": date,
                "positions": int(len(recon)),
                "matched": int((recon["break_type"] == MATCHED).sum()),
                "breaks": int(len(breaks)),
                "match_rate": float((recon["break_type"] == MATCHED).mean()),
                "abs_market_value_diff": float(breaks["abs_market_value_diff"].sum()),
            }
        )
        logger.info("%s: %s breaks of %s positions", date.date(), len(breaks), len(recon))

    history = add_break_age(pd.concat(daily_breaks, ignore_index=True))
    history = _add_security_description(history, security_master_path)
    summary = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)

    if write:
        output_dir = data_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        history.to_csv(output_dir / "breaks_history.csv", index=False)
        summary.to_csv(output_dir / "daily_summary.csv", index=False)

    return history, summary


def _add_security_description(
    history: pd.DataFrame,
    security_master_path: Path | str,
) -> pd.DataFrame:
    """Left join the security master so the exception queue is readable."""
    path = Path(security_master_path)
    if not path.exists():
        return history.assign(description=history["cusip"], instrument_type="UNKNOWN")

    master = pd.read_csv(path)[["cusip", "description", "instrument_type"]]
    merged = history.merge(master, on="cusip", how="left")
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
    latest = summary.iloc[-1]
    print(
        f"{len(summary)} days reconciled through {latest['as_of_date']:%Y-%m-%d}. "
        f"{latest['breaks']} open breaks, "
        f"match rate {latest['match_rate']:.2%}, "
        f"absolute difference ${latest['abs_market_value_diff']:,.0f}."
    )
    print(f"wrote {Path(args.data_dir) / 'output' / 'breaks_history.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
