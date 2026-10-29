"""Streamlit app: how much do NBA rookies contribute, and how does it compare with their draft slot?

Run with:  python -m streamlit run app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from rookies import add_value_over_pick, class_summary  # noqa: E402
from rookies import charts  # noqa: E402

DATA_PATH = Path(os.environ.get("ROOKIES_DATA", Path(__file__).parent / "data" / "processed" / "rookies.csv"))

st.set_page_config(page_title="NBA Rookie Impact", page_icon="🏀", layout="wide")


@st.cache_data
def load(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def plot(fig) -> None:
    st.plotly_chart(fig, width="stretch", theme=None, config=charts.CONFIG)


st.title("NBA Rookie Impact")
st.caption("How much do rookies play, what do they contribute, and is it more or less than their draft slot predicts?")

if not DATA_PATH.exists():
    st.info(
        "No data found yet. Download it with `python scripts/update_data.py` "
        "(see the README), then restart the app."
    )
    st.stop()

raw = load(str(DATA_PATH))

# ---------------------------------------------------------------- Filters
with st.sidebar:
    st.header("Filters")
    seasons = sorted(raw["season"].unique())
    chosen = st.multiselect("Seasons", seasons, default=seasons)
    min_minutes = st.slider("Minimum total minutes", 0, 2000, 300, 50, help="Removes rookies with very small samples.")
    teams = sorted(raw["team_abbreviation"].dropna().unique())
    team_filter = st.multiselect("Teams (optional)", teams)

pool = raw[raw["min"] >= min_minutes]
if len(pool) < 10:
    st.warning("Too few rookies with this minutes threshold. Lower it in the sidebar.")
    st.stop()

# The draft curve is fitted on every rookie above the threshold, so seasons are comparable.
valued, curves = add_value_over_pick(pool)

view = valued[valued["season"].isin(chosen)]
if team_filter:
    view = view[view["team_abbreviation"].isin(team_filter)]
if view.empty:
    st.warning("No rookies match these filters.")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Rookie seasons", f"{len(view):,}")
c2.metric("Median minutes per game", f"{view['mpg'].median():.1f}")
c3.metric("Rookies at 20+ MPG", f"{(view['mpg'] >= 20).mean():.0%}")
c4.metric("Avg time on court", f"{view['court_pct'].mean():.0%}", help="Share of the team's game time the rookie is on court.")

tab_over, tab_impact, tab_value, tab_data = st.tabs(["Overview", "Playing time & impact", "Draft value", "Data"])

with tab_over:
    st.subheader("By rookie class")
    st.dataframe(class_summary(view), width="stretch")
    st.caption(
        "Court % is the share of team game time a rookie is on the floor. "
        "GmSc/36 is Hollinger's Game Score per 36 minutes."
    )

with tab_impact:
    options = {
        "Game Score per 36 min": ("gmsc_per36", False),
        "PIE (NBA player impact estimate)": ("pie", False),
        "Net rating with rookie on court vs. team": ("net_vs_team", True),
        "On/off net rating difference": ("onoff", True),
    }
    available = {k: v for k, v in options.items() if v[0] in view.columns and view[v[0]].notna().any()}
    label = st.selectbox("Impact metric", list(available))
    column, zero = available[label]
    plot(charts.minutes_impact_scatter(view, column, label, zero_line=zero))
    if column in ("onoff", "net_vs_team"):
        st.caption(
            "Net rating is points per 100 possessions. Rookie on/off numbers are noisy and depend on "
            "who else is on court: read them as a rough signal, not as a verdict."
        )

with tab_value:
    metric_options = {
        "Total Game Score (volume)": "gmsc_total",
        "Game Score per 36 (rate)": "gmsc_per36",
        "Minutes played": "min",
    }
    label = st.selectbox("Production metric", list(metric_options))
    metric = metric_options[label]
    plot(charts.pick_curve_chart(valued, curves[metric], label, highlight=chosen))
    shift = curves[metric].slope * np.log(10)
    st.caption(
        "The black line is the expected value at each draft slot, fitted on every rookie in the selected seasons "
        f"with at least {min_minutes} minutes. Going from pick 1 to pick 10 changes the expected value by {shift:+,.1f}."
    )
    plot(charts.surplus_bars(view, f"{metric}_vs_expected", f"{label} vs. expected"))
    st.caption("Undrafted players are modelled as pick 61. Players who never played in the NBA as rookies are not in the data.")

with tab_data:
    query = st.text_input("Search player")
    table = view if not query else view[view["player_name"].str.contains(query, case=False, na=False)]
    cols = [
        "player_name", "season", "team_abbreviation", "age", "overall_pick", "gp", "min", "mpg", "court_pct",
        "gmsc_per36", "pie", "net_vs_team", "onoff", "gmsc_total_vs_expected",
    ]
    cols = [c for c in cols if c in table.columns]
    st.dataframe(table[cols].sort_values("min", ascending=False), width="stretch", hide_index=True)
    st.download_button("Download CSV", table.to_csv(index=False), "nba_rookies.csv", "text/csv")

with st.expander("Definitions and limitations"):
    st.markdown(
        """
        - **Rookie**: player flagged as a rookie by the NBA for that regular season.
        - **Court %**: player minutes divided by team game minutes.
        - **Game Score**: PTS + 0.4 FGM − 0.7 FGA − 0.4 (FTA − FTM) + 0.7 OREB + 0.3 DREB + STL + 0.7 AST + 0.7 BLK − 0.4 PF − TOV.
        - **Value vs. draft slot**: residual from a fit of production on ln(pick). Positive means more than the slot predicts.
        - **Limitations**: rookies traded mid-season are attributed to their last team; players who did not play are
          excluded (survivorship); bad teams give rookies more minutes on purpose, so playing time is not independent of
          team quality; rookie contract salaries are not included yet.
        """
    )
