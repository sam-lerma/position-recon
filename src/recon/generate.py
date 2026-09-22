"""Generate the synthetic feeds the reconciliation runs against.

No client data is involved. The generator builds a small book, walks prices
forward, then injects break episodes with causes a middle office would
recognise. Episodes have a start date and a duration, so a break stays open
across consecutive days and the aging in `recon.aging` has something real to
measure. Everything is driven by a seeded generator, so a given seed and end
date always produce the same files.

    python -m recon.generate --days 20
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from recon.calendar import business_days
from recon.config import DEFAULT_DATA_DIR, REFERENCE_DIR

DEFAULT_SEED = 20260921
DEFAULT_DAYS = 20

STRATEGIES = ["EQ-LC", "EQ-SC", "EQ-INTL", "FI-CORE", "FI-CRED", "MULTI"]
ISSUERS = [
    "Alderton",
    "Brightmoor",
    "Calderwood",
    "Drayfield",
    "Eastvale",
    "Fernhill",
    "Granleigh",
    "Harrowgate",
    "Inverness",
    "Jubilee",
    "Kestrelton",
    "Lanchester",
    "Marbridge",
    "Northwick",
    "Oakhaven",
    "Pemberly",
    "Quarrydon",
    "Ravensmoor",
    "Stonebridge",
    "Thornbury",
    "Underhill",
    "Vantagepoint",
    "Westmere",
    "Yarrowfield",
]
SUFFIXES = ["Industries", "Capital", "Holdings", "Technologies", "Resources"]

CURRENCIES = {"USD": 1.0, "EUR": 1.0835, "GBP": 1.2714}

BREAK_KINDS = {
    "late_booking": 0.22,
    "corporate_action": 0.12,
    "missing_in_pb": 0.14,
    "missing_in_internal": 0.12,
    "stale_price": 0.20,
    "dirty_price": 0.10,
    "fx_rate": 0.10,
    "currency_mismatch": 0.06,
}


@dataclass
class Security:
    cusip: str
    description: str
    instrument_type: str
    currency: str
    base_price: float


@dataclass
class Episode:
    """One break, open for `length` business days from `start_index`."""

    account_id: str
    cusip: str
    kind: str
    start_index: int
    length: int
    params: dict = field(default_factory=dict)

    def active_on(self, index: int) -> bool:
        return self.start_index <= index < self.start_index + self.length


def generate(
    data_dir: Path | str = DEFAULT_DATA_DIR,
    reference_dir: Path | str = REFERENCE_DIR,
    end_date: pd.Timestamp | None = None,
    days: int = DEFAULT_DAYS,
    seed: int = DEFAULT_SEED,
) -> dict[str, Path]:
    """Write reference data and one internal and one broker file per day."""
    data_dir = Path(data_dir)
    reference_dir = Path(reference_dir)
    internal_dir = data_dir / "raw" / "internal"
    pb_dir = data_dir / "raw" / "prime_broker"
    for directory in (internal_dir, pb_dir, reference_dir):
        directory.mkdir(parents=True, exist_ok=True)

    # Clear previous feeds. Without this, a run with different dates leaves the
    # old files behind and the pipeline reconciles a mix of the two, including
    # days the current calendar says the market was shut.
    for directory in (internal_dir, pb_dir):
        for stale in directory.glob("*.csv"):
            stale.unlink()

    rng = np.random.default_rng(seed)
    dates = _business_days(end_date, days)

    accounts = _build_accounts()
    securities = _build_securities(rng)
    book = _build_book(rng, accounts, securities)
    prices = _price_paths(rng, securities, dates)
    fx = _fx_paths(rng, dates)
    episodes = _build_episodes(rng, accounts, securities, book, dates)

    _write_reference(accounts, securities, reference_dir)

    written: dict[str, Path] = {}
    for index, date in enumerate(dates):
        internal = _internal_rows(date, book, securities, prices, fx)
        pb = _pb_rows(rng, index, dates, internal, securities, prices, fx, episodes)

        internal_path = internal_dir / f"internal_positions_{date:%Y%m%d}.csv"
        pb_path = pb_dir / f"pb_positions_{date:%Y%m%d}.csv"
        _format_internal(rng, internal).to_csv(internal_path, index=False)
        _format_pb(pb, accounts).to_csv(pb_path, index=False)
        written[f"{date:%Y-%m-%d}"] = internal_path

    return written


def _business_days(end_date: pd.Timestamp | None, days: int) -> list[pd.Timestamp]:
    """Trading days, so no feed is written for a day the market was shut."""
    if end_date is None:
        end_date = pd.Timestamp.today().normalize()
    return business_days(pd.Timestamp(end_date), days)


def _build_accounts() -> pd.DataFrame:
    rows = []
    for strategy in STRATEGIES:
        for number in (1, 2):
            account_id = f"{strategy}-{number:02d}"
            rows.append(
                {
                    "account_id": account_id,
                    "pb_account_code": account_id.replace("-", ""),
                    "account_name": f"{strategy.replace('-', ' ')} Fund {number}",
                }
            )
    return pd.DataFrame(rows)


def _build_securities(rng: np.random.Generator) -> dict[str, Security]:
    securities: dict[str, Security] = {}
    alphabet = np.array(list("0123456789ABCDEFGHJKLMNPQRSTUVWXYZ"))

    def new_cusip() -> str:
        return "".join(rng.choice(alphabet, size=9))

    for index in range(110):
        issuer = f"{ISSUERS[index % len(ISSUERS)]} {SUFFIXES[index % len(SUFFIXES)]}"
        currency = str(rng.choice(["USD", "USD", "USD", "USD", "EUR", "GBP"]))
        cusip = new_cusip()
        securities[cusip] = Security(
            cusip=cusip,
            description=f"{issuer} Inc",
            instrument_type="EQUITY",
            currency=currency,
            base_price=float(np.round(rng.uniform(18.0, 420.0), 2)),
        )

    for index in range(45):
        issuer = ISSUERS[(index + 7) % len(ISSUERS)]
        coupon = float(np.round(rng.uniform(2.5, 7.0), 3))
        maturity = int(rng.integers(2028, 2046))
        currency = str(rng.choice(["USD", "USD", "USD", "EUR"]))
        cusip = new_cusip()
        securities[cusip] = Security(
            cusip=cusip,
            description=f"{issuer} Capital {coupon:.3f}% {maturity}",
            instrument_type="BOND",
            currency=currency,
            base_price=float(np.round(rng.uniform(88.0, 109.0), 4)),
        )

    return securities


def _build_book(
    rng: np.random.Generator,
    accounts: pd.DataFrame,
    securities: dict[str, Security],
) -> dict[tuple[str, str], int]:
    """Persistent holdings. Quantities are whole shares or whole par."""
    equities = [s for s in securities.values() if s.instrument_type == "EQUITY"]
    bonds = [s for s in securities.values() if s.instrument_type == "BOND"]
    book: dict[tuple[str, str], int] = {}

    for account_id in accounts["account_id"]:
        if account_id.startswith("FI"):
            pool, count = bonds, int(rng.integers(20, 30))
        elif account_id.startswith("MULTI"):
            pool, count = equities + bonds, int(rng.integers(28, 38))
        else:
            pool, count = equities, int(rng.integers(24, 34))

        chosen = rng.choice(len(pool), size=count, replace=False)
        for position in chosen:
            security = pool[int(position)]
            if security.instrument_type == "BOND":
                quantity = int(rng.integers(100, 3000)) * 1000
            else:
                quantity = int(rng.integers(1, 400)) * 100
                if rng.random() < 0.12:
                    quantity = -quantity
            book[(account_id, security.cusip)] = quantity

    return book


def _price_paths(
    rng: np.random.Generator,
    securities: dict[str, Security],
    dates: list[pd.Timestamp],
) -> dict[tuple[pd.Timestamp, str], float]:
    prices: dict[tuple[pd.Timestamp, str], float] = {}
    for security in securities.values():
        is_bond = security.instrument_type == "BOND"
        sigma = 0.0022 if is_bond else 0.013
        decimals = 4 if is_bond else 2
        level = security.base_price
        for date in dates:
            level = level * float(np.exp(rng.normal(0.0, sigma)))
            prices[(date, security.cusip)] = float(np.round(level, decimals))
    return prices


def _fx_paths(
    rng: np.random.Generator,
    dates: list[pd.Timestamp],
) -> dict[tuple[pd.Timestamp, str], float]:
    fx: dict[tuple[pd.Timestamp, str], float] = {}
    for currency, start in CURRENCIES.items():
        level = start
        for date in dates:
            if currency != "USD":
                level = level * float(np.exp(rng.normal(0.0, 0.0035)))
            fx[(date, currency)] = float(np.round(level, 6))
    return fx


def _market_value(quantity: float, price: float, fx: float, instrument_type: str) -> float:
    """Bonds are quoted in percent of par, equities in currency per share."""
    gross = quantity * price / 100.0 if instrument_type == "BOND" else quantity * price
    return float(np.round(gross * fx, 2))


def _build_episodes(
    rng: np.random.Generator,
    accounts: pd.DataFrame,
    securities: dict[str, Security],
    book: dict[tuple[str, str], int],
    dates: list[pd.Timestamp],
) -> list[Episode]:
    """Seed break episodes with a cause, a start date and a duration."""
    held = list(book.keys())
    kinds = list(BREAK_KINDS)
    weights = np.array([BREAK_KINDS[kind] for kind in kinds])
    weights = weights / weights.sum()

    episodes: list[Episode] = []
    used: set[tuple[str, str]] = set()
    target = 75

    while len(episodes) < target:
        kind = str(rng.choice(kinds, p=weights))
        candidate = _pick_key(rng, kind, held, securities, book, accounts)
        if candidate is None or candidate in used:
            continue
        used.add(candidate)
        account_id, cusip = candidate

        # A stale price needs a previous day to be stale from. Episodes may run
        # past the last date; they are simply still open when the history ends.
        first_possible = 1 if kind == "stale_price" else 0
        start_index = int(rng.integers(first_possible, len(dates)))
        length = int(rng.integers(1, 16))
        episodes.append(
            Episode(
                account_id=account_id,
                cusip=cusip,
                kind=kind,
                start_index=start_index,
                length=length,
                params=_episode_params(rng, kind, book.get(candidate), securities[cusip]),
            )
        )

    return episodes


def _pick_key(
    rng: np.random.Generator,
    kind: str,
    held: list[tuple[str, str]],
    securities: dict[str, Security],
    book: dict[tuple[str, str], int],
    accounts: pd.DataFrame,
) -> tuple[str, str] | None:
    """Pick a position the break kind can actually happen to."""
    if kind == "missing_in_internal":
        # A position only the broker has, so it must not be in the book.
        account_id = str(rng.choice(accounts["account_id"].to_numpy()))
        cusip = str(rng.choice(list(securities)))
        return None if (account_id, cusip) in book else (account_id, cusip)

    if kind == "dirty_price":
        candidates = [key for key in held if securities[key[1]].instrument_type == "BOND"]
    elif kind == "corporate_action":
        # Bonds do not split. Restricting this was a correctness fix, not a
        # cosmetic one: it was putting 3 for 1 splits on 2036 corporates.
        candidates = [key for key in held if securities[key[1]].instrument_type == "EQUITY"]
    elif kind == "fx_rate":
        candidates = [key for key in held if securities[key[1]].currency != "USD"]
    else:
        candidates = held

    if not candidates:
        return None
    return candidates[int(rng.integers(len(candidates)))]


def _episode_params(
    rng: np.random.Generator,
    kind: str,
    quantity: int | None,
    security: Security,
) -> dict:
    if kind == "late_booking":
        base = abs(quantity or 10_000)
        size = max(100, int(np.round(base * float(rng.uniform(0.05, 0.4)) / 100) * 100))
        return {"trade_quantity": size * int(rng.choice([-1, 1]))}
    if kind == "corporate_action":
        return {"split_factor": float(rng.choice([2.0, 3.0]))}
    if kind == "missing_in_internal":
        if security.instrument_type == "BOND":
            return {"quantity": int(rng.integers(100, 1500)) * 1000}
        return {"quantity": int(rng.integers(1, 250)) * 100}
    if kind == "dirty_price":
        return {"accrued": float(np.round(rng.uniform(0.35, 2.40), 4))}
    if kind == "currency_mismatch":
        wrong = [code for code in CURRENCIES if code != security.currency]
        return {"reported_currency": str(rng.choice(wrong))}
    if kind == "fx_rate":
        return {"fx_error": float(rng.choice([-1, 1])) * float(rng.uniform(0.002, 0.006))}
    return {}


def _internal_rows(
    date: pd.Timestamp,
    book: dict[tuple[str, str], int],
    securities: dict[str, Security],
    prices: dict[tuple[pd.Timestamp, str], float],
    fx: dict[tuple[pd.Timestamp, str], float],
) -> list[dict]:
    rows = []
    for (account_id, cusip), quantity in book.items():
        security = securities[cusip]
        price = prices[(date, cusip)]
        rate = fx[(date, security.currency)]
        rows.append(
            {
                "as_of_date": date,
                "account_id": account_id,
                "cusip": cusip,
                "quantity": float(quantity),
                "price": price,
                "market_value": _market_value(quantity, price, rate, security.instrument_type),
                "currency": security.currency,
                # The rate this row was valued at. An fx_rate episode overrides
                # it, and lot splitting reuses it rather than re-deriving the
                # correct rate and quietly undoing the break.
                "fx_rate": rate,
            }
        )
    return rows


def _pb_rows(
    rng: np.random.Generator,
    index: int,
    dates: list[pd.Timestamp],
    internal: list[dict],
    securities: dict[str, Security],
    prices: dict[tuple[pd.Timestamp, str], float],
    fx: dict[tuple[pd.Timestamp, str], float],
    episodes: list[Episode],
) -> list[dict]:
    """Start from the internal picture, then apply whatever went wrong."""
    date = dates[index]
    rows = {(row["account_id"], row["cusip"]): dict(row) for row in internal}
    active = [episode for episode in episodes if episode.active_on(index)]

    for episode in active:
        key = (episode.account_id, episode.cusip)
        security = securities[episode.cusip]
        rate = fx[(date, security.currency)]

        if episode.kind == "missing_in_pb":
            rows.pop(key, None)
            continue

        if episode.kind == "missing_in_internal":
            quantity = episode.params["quantity"]
            price = prices[(date, episode.cusip)]
            rows[key] = {
                "as_of_date": date,
                "account_id": episode.account_id,
                "cusip": episode.cusip,
                "quantity": float(quantity),
                "price": price,
                "market_value": _market_value(quantity, price, rate, security.instrument_type),
                "currency": security.currency,
                "fx_rate": rate,
            }
            continue

        row = rows.get(key)
        if row is None:
            continue

        if episode.kind == "late_booking":
            row["quantity"] = row["quantity"] - episode.params["trade_quantity"]
        elif episode.kind == "corporate_action":
            # The broker has not applied the split, so it still holds the
            # pre-split quantity at the pre-split price. Quantity is out by the
            # factor, price by its inverse, and the two market values agree.
            # Moving quantity alone would invent a difference worth half the
            # position.
            factor = episode.params["split_factor"]
            row["quantity"] = float(np.round(row["quantity"] / factor))
            row["price"] = float(np.round(row["price"] * factor, 4))
        elif episode.kind == "stale_price":
            row["price"] = prices[(dates[max(0, index - 1)], episode.cusip)]
        elif episode.kind == "dirty_price":
            row["price"] = float(np.round(row["price"] + episode.params["accrued"], 4))
        elif episode.kind == "currency_mismatch":
            # The broker's security master has the wrong trading currency, so
            # the numbers agree and the label on them does not.
            row["currency"] = episode.params["reported_currency"]
        elif episode.kind == "fx_rate":
            rate = float(np.round(rate * (1.0 + episode.params["fx_error"]), 6))

        row["fx_rate"] = rate
        row["market_value"] = _market_value(
            row["quantity"], row["price"], rate, security.instrument_type
        )

    return _split_into_lots(rng, list(rows.values()), securities)


def _split_into_lots(
    rng: np.random.Generator,
    rows: list[dict],
    securities: dict[str, Security],
) -> list[dict]:
    """Report some positions as several tax lots, as a broker file would.

    Each lot is valued on its own, so the lots sum back to within a cent or two
    of the position, not exactly. That is what the absolute market value
    tolerance is there to absorb.
    """
    out: list[dict] = []
    for row in rows:
        quantity = int(row["quantity"])
        if abs(quantity) < 300 or rng.random() > 0.2:
            out.append(row)
            continue

        security = securities[row["cusip"]]
        rate = row["fx_rate"]
        count = int(rng.choice([2, 3]))
        cuts = sorted(rng.choice(np.arange(1, abs(quantity)), size=count - 1, replace=False))
        edges = [0, *cuts, abs(quantity)]
        sizes = [edges[i + 1] - edges[i] for i in range(count)]
        sign = 1 if quantity > 0 else -1

        for size in sizes:
            if size == 0:
                continue
            lot = dict(row)
            lot["quantity"] = float(sign * size)
            lot["market_value"] = _market_value(
                sign * size, row["price"], rate, security.instrument_type
            )
            out.append(lot)

    return out


def _format_internal(rng: np.random.Generator, rows: list[dict]) -> pd.DataFrame:
    """Write the internal extract, with the identifier untidiness of a real one."""
    df = pd.DataFrame(rows)
    cusips = df["cusip"].to_numpy(copy=True)
    lower = rng.random(len(cusips)) < 0.15
    trailing = rng.random(len(cusips)) < 0.10
    cusips = np.where(lower, np.char.lower(cusips.astype(str)), cusips)
    cusips = np.where(trailing, np.char.add(cusips.astype(str), " "), cusips)

    return pd.DataFrame(
        {
            "as_of_date": df["as_of_date"].dt.strftime("%Y-%m-%d"),
            "account_code": df["account_id"],
            "cusip": cusips,
            "quantity": df["quantity"].map(lambda value: f"{value:.2f}"),
            "price": df["price"].map(lambda value: f"{value:.4f}"),
            "market_value": df["market_value"].map(lambda value: f"{value:.2f}"),
            "currency": df["currency"],
        }
    )


def _format_pb(rows: list[dict], accounts: pd.DataFrame) -> pd.DataFrame:
    """Write the broker file: broker account codes, side flag, formatted numbers."""
    df = pd.DataFrame(rows)
    pb_codes = accounts.set_index("account_id")["pb_account_code"]

    return pd.DataFrame(
        {
            "business_date": df["as_of_date"].dt.strftime("%m/%d/%Y"),
            "account": df["account_id"].map(pb_codes),
            "cusip": df["cusip"].str.upper(),
            "long_short": np.where(df["quantity"] < 0, "S", "L"),
            "quantity": df["quantity"].abs().map(lambda value: f"{value:,.2f}"),
            "price": df["price"].map(lambda value: f"{value:.4f}"),
            "market_value": df["market_value"].abs().map(lambda value: f"{value:,.2f}"),
            "ccy": df["currency"],
        }
    )


def _write_reference(
    accounts: pd.DataFrame,
    securities: dict[str, Security],
    reference_dir: Path,
) -> None:
    accounts.to_csv(reference_dir / "account_map.csv", index=False)
    pd.DataFrame(
        [
            {
                "cusip": security.cusip,
                "description": security.description,
                "instrument_type": security.instrument_type,
                "currency": security.currency,
            }
            for security in securities.values()
        ]
    ).to_csv(reference_dir / "security_master.csv", index=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--reference-dir", default=str(REFERENCE_DIR))
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS)
    parser.add_argument("--end-date", default=None, help="last business day, YYYY-MM-DD")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)

    written = generate(
        data_dir=args.data_dir,
        reference_dir=args.reference_dir,
        end_date=pd.Timestamp(args.end_date) if args.end_date else None,
        days=args.days,
        seed=args.seed,
    )
    dates = sorted(written)
    print(f"wrote {len(dates)} days of feeds, {dates[0]} to {dates[-1]}, into {args.data_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
