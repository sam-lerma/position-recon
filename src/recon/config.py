"""Tolerances, break labels and the break key.

Everything that a business user would argue about lives here rather than being
scattered through the comparison code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Paths default to the checkout, which is how the repo is meant to be run, and
# can be pointed elsewhere by environment or by the --data-dir flag.
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = Path(os.environ.get("RECON_DATA_DIR", REPO_ROOT / "data"))
REFERENCE_DIR = Path(os.environ.get("RECON_REFERENCE_DIR", REPO_ROOT / "reference"))
ACCOUNT_MAP_PATH = REFERENCE_DIR / "account_map.csv"
SECURITY_MASTER_PATH = REFERENCE_DIR / "security_master.csv"


@dataclass(frozen=True)
class Tolerances:
    """Match tolerances.

    quantity_abs       absolute, in shares or par. Share counts should agree
                       exactly, so this only absorbs float noise.
    price_rel          relative, applied to the internal price.
    market_value_rel   relative, applied to the internal market value.
    market_value_abs   absolute floor in USD, so that penny rounding on small
                       positions does not raise a break on its own.
    """

    quantity_abs: float = 0.01
    price_rel: float = 1e-4
    market_value_rel: float = 5e-4
    market_value_abs: float = 1.00


DEFAULT_TOLERANCES = Tolerances()

MISSING_IN_PB = "MISSING_IN_PB"
MISSING_IN_INTERNAL = "MISSING_IN_INTERNAL"
QUANTITY_BREAK = "QUANTITY_BREAK"
PRICE_BREAK = "PRICE_BREAK"
MARKET_VALUE_BREAK = "MARKET_VALUE_BREAK"
MATCHED = "MATCHED"

# Classification order. The first matching rule wins, so the two missing-side
# rules must come first: an outer join leaves NaN on the absent side and every
# numeric comparison below would otherwise fire on it.
BREAK_TYPES = [
    MISSING_IN_PB,
    MISSING_IN_INTERNAL,
    QUANTITY_BREAK,
    PRICE_BREAK,
    MARKET_VALUE_BREAK,
]

# The unit ops investigates: one security in one account. A break that changes
# type while it is open stays the same item in the queue and keeps its age.
BREAK_KEY = ["account_id", "cusip"]

POSITION_GRAIN = ["as_of_date", "account_id", "cusip"]

AGE_BUCKETS = ["0-1d", "2-5d", "6-10d", "11d+"]
