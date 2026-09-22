"""One place for the colours and labels the dashboard uses.

The same values are declared as custom properties in assets/style.css; change
them in both. Charts are drawn for the light surface only.
"""

from __future__ import annotations

from recon.config import (
    MARKET_VALUE_BREAK,
    MISSING_IN_INTERNAL,
    MISSING_IN_PB,
    PRICE_BREAK,
    QUANTITY_BREAK,
)

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#6f6e69"
GRID = "#e6e5e1"

SERIES_BLUE = "#2a78d6"

# Age is ordered, so it gets an ordinal ramp of one hue rather than four
# unrelated colours: darker reads as older, which is also worse.
AGE_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#184f95"]

STATUS_CRITICAL = "#d03b3b"
STATUS_GOOD = "#0ca30c"

FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"

BREAK_LABELS = {
    MISSING_IN_PB: "Missing at broker",
    MISSING_IN_INTERNAL: "Missing internally",
    QUANTITY_BREAK: "Quantity",
    PRICE_BREAK: "Price",
    MARKET_VALUE_BREAK: "Market value",
}
