"""Break dashboard.

Run the pipeline first, then:

    python -m app.dashboard

The callback does no analysis of its own. It reads the filter values, hands
them to `app.views`, and passes the result to `app.figures`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from dash import Dash, Input, Output, dash_table, dcc, html
from dash.dash_table.Format import Format, Group, Scheme

from app import views
from app.figures import break_trend_figure, breaks_by_age_figure, breaks_by_type_figure
from app.theme import BREAK_LABELS, GRID, SURFACE, TEXT_SECONDARY
from recon.config import BREAK_TYPES

TABLE_PAGE_SIZE = 12
TABLE_FONT = (
    "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"
)

# Widths keep the money column on screen at laptop width instead of pushing it
# into a horizontal scroll.
COLUMN_WIDTHS = {
    "Account": "106px",
    "CUSIP": "108px",
    "Security": "auto",
    "Break": "140px",
    "Age": "52px",
    "Qty internal": "100px",
    "Qty broker": "100px",
    "Qty diff": "100px",
    "MV diff USD": "116px",
}


def build_app(
    history_path: Path | str = views.HISTORY_PATH,
    summary_path: Path | str = views.SUMMARY_PATH,
) -> Dash:
    _require_outputs(history_path, summary_path)
    history = views.load_history(history_path)
    summary = views.load_summary(summary_path)

    dates = views.as_of_dates(history)
    latest = dates[-1]

    app = Dash(__name__, title="Position Break Dashboard")
    app.layout = _layout(history, dates, latest)

    @app.callback(
        Output("kpi-row", "children"),
        Output("by-type", "figure"),
        Output("by-age", "figure"),
        Output("trend", "figure"),
        Output("queue", "data"),
        Output("queue", "columns"),
        Input("as-of", "value"),
        Input("accounts", "value"),
        Input("break-types", "value"),
    )
    def update(as_of: str, selected_accounts: list[str], selected_types: list[str]):
        filtered = views.apply_filters(history, selected_accounts, selected_types)
        day = views.on_date(filtered, pd.Timestamp(as_of))

        queue = views.exception_queue(day, labels=BREAK_LABELS)
        columns = [_column(name) for name in queue.columns]

        return (
            _kpi_tiles(views.kpis(day, summary, pd.Timestamp(as_of))),
            breaks_by_type_figure(views.breaks_by_type(day)),
            breaks_by_age_figure(views.breaks_by_age(day)),
            break_trend_figure(views.break_trend(filtered)),
            queue.to_dict("records"),
            columns,
        )

    return app


def _require_outputs(*paths: Path | str) -> None:
    """Fail with something useful rather than a traceback on a missing file."""
    missing = [str(path) for path in paths if not Path(path).exists()]
    if missing:
        raise SystemExit(
            "The dashboard reads what the pipeline writes, and these are not there yet:\n"
            + "\n".join(f"  {path}" for path in missing)
            + "\n\nRun them first:\n"
            "  python -m recon.generate\n"
            "  python -m recon.pipeline"
        )


def _layout(history: pd.DataFrame, dates: list[pd.Timestamp], latest: pd.Timestamp) -> html.Div:
    return html.Div(
        className="page",
        children=[
            html.Header(
                className="header",
                children=[
                    html.H1("Position break dashboard"),
                    html.P(
                        "Internal book of record against the prime broker. Synthetic data.",
                        className="subtitle",
                    ),
                ],
            ),
            html.Div(
                className="filters",
                children=[
                    _filter(
                        "As of",
                        dcc.Dropdown(
                            id="as-of",
                            options=[
                                {"label": f"{date:%d %b %Y}", "value": f"{date:%Y-%m-%d}"}
                                for date in reversed(dates)
                            ],
                            value=f"{latest:%Y-%m-%d}",
                            clearable=False,
                        ),
                    ),
                    _filter(
                        "Accounts",
                        dcc.Dropdown(
                            id="accounts",
                            options=views.accounts(history),
                            multi=True,
                            placeholder="All accounts",
                        ),
                    ),
                    _filter(
                        "Break type",
                        dcc.Dropdown(
                            id="break-types",
                            options=[
                                {"label": BREAK_LABELS[break_type], "value": break_type}
                                for break_type in BREAK_TYPES
                            ],
                            multi=True,
                            placeholder="All types",
                        ),
                    ),
                ],
            ),
            html.Div(id="kpi-row", className="kpi-row"),
            html.Div(
                className="chart-row",
                children=[
                    html.Div(
                        dcc.Graph(id="by-type", config={"displayModeBar": False}),
                        className="card",
                    ),
                    html.Div(
                        dcc.Graph(id="by-age", config={"displayModeBar": False}),
                        className="card",
                    ),
                ],
            ),
            html.Div(
                dcc.Graph(id="trend", config={"displayModeBar": False}),
                className="card",
            ),
            html.Div(
                className="card",
                children=[
                    html.H2("Exception queue", className="card-title"),
                    html.P(
                        "Largest absolute market value difference first. "
                        "Sort or filter any column.",
                        className="card-note",
                    ),
                    _table(),
                ],
            ),
        ],
    )


def _filter(label: str, control) -> html.Div:
    return html.Div(className="filter", children=[html.Label(label), control])


def _kpi_tiles(kpis: dict) -> list[html.Div]:
    match_rate = kpis["match_rate"]
    tiles = [
        ("Open breaks", f"{kpis['open_breaks']:,}", f"of {kpis['positions']:,} positions", ""),
        ("New today", f"{kpis['new_today']:,}", "first seen on this date", ""),
        (
            f"Aged over {views.AGED_THRESHOLD_DAYS} days",
            f"{kpis['aged']:,}",
            "business days open",
            "critical" if kpis["aged"] else "good",
        ),
        (
            "Absolute difference",
            _money(kpis["abs_market_value_diff"]),
            "match rate " + ("n/a" if pd.isna(match_rate) else f"{match_rate:.2%}"),
            "",
        ),
    ]
    return [
        html.Div(
            className="kpi",
            children=[
                html.Span(label, className="kpi-label"),
                html.Span(value, className=f"kpi-value {tone}".strip()),
                html.Span(note, className="kpi-note"),
            ],
        )
        for label, value, note, tone in tiles
    ]


def _money(value: float) -> str:
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:,.1f}m"
    if abs(value) >= 1_000:
        return f"${value / 1_000:,.0f}k"
    return f"${value:,.0f}"


def _column(name: str) -> dict:
    """Column spec: numbers right aligned, grouped, no decimals."""
    column = {"name": name, "id": name, "type": "text"}
    if name in views.QUEUE_NUMERIC:
        column["type"] = "numeric"
        column["format"] = Format(group=Group.yes, precision=0, scheme=Scheme.fixed)
    return column


def _table() -> dash_table.DataTable:
    return dash_table.DataTable(
        id="queue",
        page_size=TABLE_PAGE_SIZE,
        sort_action="native",
        filter_action="native",
        style_as_list_view=True,
        style_table={"overflowX": "auto"},
        style_header={
            "backgroundColor": SURFACE,
            "fontFamily": TABLE_FONT,
            "borderBottom": f"1px solid {GRID}",
            "color": TEXT_SECONDARY,
            "fontWeight": "600",
            "fontSize": "12px",
            "textAlign": "left",
        },
        style_cell={
            "backgroundColor": SURFACE,
            "border": "none",
            "borderBottom": f"1px solid {GRID}",
            "color": "#0b0b0b",
            "fontFamily": TABLE_FONT,
            "fontSize": "13px",
            "fontVariantNumeric": "tabular-nums",
            "padding": "10px 10px",
            "textAlign": "left",
            "overflow": "hidden",
            "textOverflow": "ellipsis",
        },
        style_cell_conditional=[
            {
                "if": {"column_id": column},
                "textAlign": "right" if column in views.QUEUE_NUMERIC else "left",
                "width": width,
                "minWidth": width,
                "maxWidth": "260px" if width == "auto" else width,
            }
            for column, width in COLUMN_WIDTHS.items()
        ],
        style_data_conditional=[
            {
                "if": {"filter_query": "{Age} > 5", "column_id": "Age"},
                "color": "#d03b3b",
                "fontWeight": "600",
            }
        ],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", default=str(views.HISTORY_PATH))
    parser.add_argument("--summary", default=str(views.SUMMARY_PATH))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(argv)

    build_app(args.history, args.summary).run(host=args.host, port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
