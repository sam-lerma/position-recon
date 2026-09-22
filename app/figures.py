"""Plotly figures for the dashboard.

Each figure takes a frame from `app.views` and returns a figure. Counts are
labelled on the marks themselves, so the bar charts carry no value axis and no
gridlines; the reader is comparing a handful of numbers, not estimating them
off a scale.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from app.theme import (
    AGE_RAMP,
    BREAK_LABELS,
    FONT,
    GRID,
    SERIES_BLUE,
    SURFACE,
    TEXT_MUTED,
    TEXT_SECONDARY,
)

CORNER_RADIUS = 4


def _base_layout(title: str, height: int) -> dict:
    return {
        "title": {
            "text": title,
            "font": {"size": 15, "color": "#0b0b0b", "family": FONT},
            "x": 0,
            "xanchor": "left",
            "y": 0.97,
            "yanchor": "top",
        },
        "height": height,
        "paper_bgcolor": SURFACE,
        "plot_bgcolor": SURFACE,
        "font": {"family": FONT, "size": 12, "color": TEXT_SECONDARY},
        "margin": {"l": 8, "r": 16, "t": 48, "b": 8},
        "showlegend": False,
        "hoverlabel": {
            "bgcolor": "#ffffff",
            "bordercolor": GRID,
            "font": {"family": FONT, "size": 12, "color": "#0b0b0b"},
        },
    }


def breaks_by_type_figure(by_type: pd.DataFrame) -> go.Figure:
    """Horizontal bars, most severe at the top, count labelled on each bar."""
    if by_type.empty:
        return empty_figure("Breaks by type")

    labels = by_type["break_type"].map(BREAK_LABELS).fillna(by_type["break_type"])

    figure = go.Figure(
        go.Bar(
            x=by_type["breaks"],
            y=labels,
            orientation="h",
            marker={"color": SERIES_BLUE, "cornerradius": CORNER_RADIUS},
            customdata=by_type["abs_market_value_diff"],
            text=by_type["breaks"],
            textposition="outside",
            textfont={"color": TEXT_SECONDARY, "size": 12},
            cliponaxis=False,
            hovertemplate=(
                "<b>%{y}</b><br>%{x} open<br>$%{customdata:,.0f} absolute difference<extra></extra>"
            ),
        )
    )
    layout = _base_layout("Breaks by type", height=260)
    layout["margin"]["l"] = 8
    figure.update_layout(**layout)
    figure.update_xaxes(visible=False, range=[0, max(1, by_type["breaks"].max()) * 1.18])
    figure.update_yaxes(
        autorange="reversed",
        showgrid=False,
        zeroline=False,
        ticks="",
        tickfont={"size": 12, "color": TEXT_SECONDARY},
    )
    return figure


def breaks_by_age_figure(by_age: pd.DataFrame) -> go.Figure:
    """Vertical bars on an ordinal ramp: darker is older."""
    if by_age.empty:
        return empty_figure("Breaks by age")

    figure = go.Figure(
        go.Bar(
            x=by_age["age_bucket"].astype(str),
            y=by_age["breaks"],
            marker={"color": AGE_RAMP[: len(by_age)], "cornerradius": CORNER_RADIUS},
            text=by_age["breaks"],
            textposition="outside",
            textfont={"color": TEXT_SECONDARY, "size": 12},
            cliponaxis=False,
            hovertemplate="<b>%{x} old</b><br>%{y} open<extra></extra>",
        )
    )
    layout = _base_layout("Breaks by age", height=260)
    figure.update_layout(**layout, bargap=0.45)
    figure.update_xaxes(
        showgrid=False, zeroline=False, ticks="", tickfont={"size": 12, "color": TEXT_SECONDARY}
    )
    figure.update_yaxes(visible=False, range=[0, max(1, by_age["breaks"].max()) * 1.2])
    return figure


def break_trend_figure(trend: pd.DataFrame) -> go.Figure:
    """Open breaks per business day."""
    if trend.empty:
        return empty_figure("Open breaks by day")

    figure = go.Figure(
        go.Scatter(
            x=trend["as_of_date"],
            y=trend["breaks"],
            mode="lines+markers",
            line={"color": SERIES_BLUE, "width": 2},
            marker={"size": 8, "color": SERIES_BLUE, "line": {"color": SURFACE, "width": 2}},
            hovertemplate="%{y} open<extra></extra>",
        )
    )
    figure.update_layout(**_base_layout("Open breaks by day", height=280), hovermode="x unified")
    figure.update_xaxes(
        showgrid=False,
        zeroline=False,
        showspikes=True,
        spikemode="across",
        spikethickness=1,
        spikecolor=GRID,
        spikedash="solid",
        tickformat="%-d %b",
        tickfont={"size": 12, "color": TEXT_SECONDARY},
    )
    figure.update_yaxes(
        showgrid=True,
        gridcolor=GRID,
        zeroline=False,
        rangemode="tozero",
        ticks="",
        tickfont={"size": 12, "color": TEXT_SECONDARY},
    )
    return figure


def empty_figure(title: str) -> go.Figure:
    """Shown when the filters leave nothing to draw."""
    figure = go.Figure()
    figure.update_layout(**_base_layout(title, height=260))
    figure.update_xaxes(visible=False)
    figure.update_yaxes(visible=False)
    figure.add_annotation(
        text="No breaks for this selection",
        showarrow=False,
        font={"family": FONT, "size": 13, "color": TEXT_MUTED},
        x=0.5,
        y=0.45,
        xref="paper",
        yref="paper",
    )
    return figure
