"""Plotly figures with a restrained, consistent style (plotly is an optional dependency)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .metrics import UNDRAFTED_PICK
from .value import PickCurve

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'

CONFIG = {"displayModeBar": False}


def _style(fig: go.Figure, title: str, height: int = 420) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left", y=0.98, yanchor="top", font=dict(size=15, color=INK)),
        font=dict(family=FONT, size=12, color=INK_2),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        height=height,
        margin=dict(l=56, r=16, t=60, b=48),
        showlegend=False,
        hoverlabel=dict(bgcolor="#ffffff", bordercolor=AXIS, font=dict(family=FONT, size=12, color=INK)),
    )
    fig.update_xaxes(
        showgrid=False, linecolor=AXIS, ticks="", tickfont=dict(color=MUTED), title_font=dict(color=MUTED), fixedrange=True
    )
    fig.update_yaxes(
        gridcolor=GRID,
        zeroline=False,
        showline=False,
        ticks="",
        tickfont=dict(color=MUTED),
        title_font=dict(color=MUTED),
        fixedrange=True,
    )
    return fig


def _hover(df: pd.DataFrame) -> list[str]:
    return [
        f"<b>{r.player_name}</b><br>{r.team_abbreviation} · {r.season}"
        f"<br>{r.mpg:.1f} MPG · pick {'UDFA' if r.undrafted else int(r.overall_pick)}"
        for r in df.itertuples()
    ]


def minutes_impact_scatter(df: pd.DataFrame, y: str, y_label: str, zero_line: bool = False) -> go.Figure:
    """Minutes per game against an impact metric; one dot per rookie season."""
    data = df.dropna(subset=[y])
    fig = go.Figure(
        go.Scatter(
            x=data["mpg"],
            y=data[y],
            mode="markers",
            marker=dict(color=BLUE, size=9, opacity=0.75, line=dict(color=SURFACE, width=1.5)),
            text=_hover(data),
            hovertemplate="%{text}<br>" + y_label + ": %{y:.2f}<extra></extra>",
        )
    )
    if zero_line:
        fig.add_hline(y=0, line=dict(color=AXIS, width=1))
    fig.update_xaxes(title="Minutes per game")
    fig.update_yaxes(title=y_label)
    return _style(fig, f"Playing time vs. {y_label}")


def pick_curve_chart(df: pd.DataFrame, curve: PickCurve, label: str, highlight: list[str] | None = None) -> go.Figure:
    """Production against draft slot with the fitted expectation curve."""
    data = df.dropna(subset=[curve.metric])
    selected = data["season"].isin(highlight) if highlight else pd.Series(True, index=data.index)
    fig = go.Figure()
    for mask, color, opacity in ((~selected, AXIS, 0.5), (selected, BLUE, 0.8)):
        part = data[mask]
        if part.empty:
            continue
        fig.add_scatter(
            x=part["pick_model"],
            y=part[curve.metric],
            mode="markers",
            marker=dict(color=color, size=8, opacity=opacity, line=dict(color=SURFACE, width=1)),
            text=_hover(part),
            hovertemplate="%{text}<br>" + label + ": %{y:,.1f}<extra></extra>",
        )
    xs = np.linspace(1, UNDRAFTED_PICK, 120)
    fig.add_scatter(x=xs, y=curve.predict(xs), mode="lines", line=dict(color=INK, width=2), hoverinfo="skip")
    fig.update_xaxes(
        type="log",
        tickvals=[1, 2, 3, 5, 10, 20, 30, 45, UNDRAFTED_PICK],
        ticktext=["1", "2", "3", "5", "10", "20", "30", "45", "UDFA"],
        title="Draft pick (log scale)",
    )
    fig.update_yaxes(title=label)
    return _style(fig, f"{label} by draft slot (R² = {curve.r_squared:.2f})")


def surplus_bars(df: pd.DataFrame, column: str, label: str, n: int = 10) -> go.Figure:
    """Best and worst rookies relative to what their draft slot predicts."""
    data = df.dropna(subset=[column])
    top = data.nlargest(n, column)
    bottom = data.nsmallest(n, column)
    chosen = pd.concat([top, bottom]).drop_duplicates().sort_values(column)
    names = [
        f"{r.player_name} ({r.season[2:4]}-{r.season[-2:]}, {'UDFA' if r.undrafted else '#' + str(int(r.overall_pick))})"
        for r in chosen.itertuples()
    ]
    fig = go.Figure(
        go.Bar(
            x=chosen[column],
            y=names,
            orientation="h",
            marker_color=[BLUE if v >= 0 else ORANGE for v in chosen[column]],
            text=[f"{v:+,.0f}" if abs(v) >= 100 else f"{v:+.1f}" for v in chosen[column]],
            textposition="outside",
            textfont=dict(color=INK_2),
            cliponaxis=False,
            hovertemplate="<b>%{y}</b><br>" + label + ": %{x:+,.1f}<extra></extra>",
        )
    )
    fig.update_xaxes(title=label, zeroline=True, zerolinecolor=AXIS, showgrid=True, gridcolor=GRID)
    fig.update_yaxes(showgrid=False, tickfont=dict(color=INK_2))
    _style(fig, f"Largest gaps vs. draft-slot expectation: {label}", height=max(380, 26 * len(chosen) + 120))
    fig.update_layout(margin=dict(l=210, r=48, t=60, b=48))
    try:
        fig.update_traces(marker_cornerradius=3)
    except (ValueError, TypeError):
        pass
    return fig
