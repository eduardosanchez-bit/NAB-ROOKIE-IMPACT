"""Expected rookie production by draft slot, and value over that expectation.

Idea: the earlier a player is picked, the more the league expects from him.
We fit a smooth curve ``metric = a + b * ln(pick)`` on all rookies and treat the
residual as production above (+) or below (-) what the draft slot predicts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DEFAULT_METRICS = ("min", "gmsc_total", "gmsc_per36")


@dataclass(frozen=True)
class PickCurve:
    """Fitted relationship between draft pick and a production metric."""

    metric: str
    intercept: float
    slope: float
    r_squared: float

    def predict(self, pick) -> np.ndarray:
        return self.intercept + self.slope * np.log(np.asarray(pick, dtype=float))


def fit_pick_curve(df: pd.DataFrame, metric: str, pick_col: str = "pick_model") -> PickCurve:
    """Ordinary least squares of ``metric`` on ``ln(pick)``."""
    data = df[[pick_col, metric]].dropna()
    if len(data) < 3:
        raise ValueError("Need at least 3 rookies to fit a pick curve.")
    x = np.log(data[pick_col].to_numpy(dtype=float))
    y = data[metric].to_numpy(dtype=float)
    design = np.column_stack([np.ones_like(x), x])
    (intercept, slope), *_ = np.linalg.lstsq(design, y, rcond=None)
    fitted = intercept + slope * x
    ss_res = float(((y - fitted) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")
    return PickCurve(metric, float(intercept), float(slope), r2)


def add_value_over_pick(
    df: pd.DataFrame, metrics: tuple[str, ...] = DEFAULT_METRICS, pick_col: str = "pick_model"
) -> tuple[pd.DataFrame, dict[str, PickCurve]]:
    """Add ``<metric>_expected`` and ``<metric>_vs_expected`` columns.

    Returns the enriched frame and the fitted curves, keyed by metric.
    """
    out = df.copy()
    curves: dict[str, PickCurve] = {}
    for m in metrics:
        curve = fit_pick_curve(out, m, pick_col)
        curves[m] = curve
        out[f"{m}_expected"] = curve.predict(out[pick_col])
        out[f"{m}_vs_expected"] = out[m] - out[f"{m}_expected"]
    return out, curves


def class_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per season: how many rookies played and how much they contributed."""
    g = df.groupby("season")
    summary = pd.DataFrame(
        {
            "Rookies": g.size(),
            "Median MPG": g["mpg"].median(),
            "Avg court %": g["court_pct"].mean() * 100,
            "Avg GmSc/36": g["gmsc_per36"].mean(),
        }
    )
    if "pie" in df:
        summary["Avg PIE %"] = g["pie"].mean() * 100
    return summary.round(1)
