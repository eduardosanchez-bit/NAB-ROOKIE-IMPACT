"""Turn raw NBA downloads into one tidy table with one row per rookie season."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

UNDRAFTED_PICK = 61  # undrafted players are modelled as the pick after the 60th


def game_score(df: pd.DataFrame) -> pd.Series:
    """Hollinger's Game Score from box-score totals (raw upper-case NBA column names).

    GmSc = PTS + 0.4 FGM - 0.7 FGA - 0.4 (FTA - FTM) + 0.7 OREB + 0.3 DREB
           + STL + 0.7 AST + 0.7 BLK - 0.4 PF - TOV
    """
    return (
        df["PTS"]
        + 0.4 * df["FGM"]
        - 0.7 * df["FGA"]
        - 0.4 * (df["FTA"] - df["FTM"])
        + 0.7 * df["OREB"]
        + 0.3 * df["DREB"]
        + df["STL"]
        + 0.7 * df["AST"]
        + 0.7 * df["BLK"]
        - 0.4 * df["PF"]
        - df["TOV"]
    )


def read_raw(raw_dir: Path, kind: str) -> pd.DataFrame:
    files = sorted(Path(raw_dir).glob(f"{kind}_*.csv"))
    if not files:
        raise FileNotFoundError(f"No '{kind}_*.csv' files in {raw_dir}. Run `python scripts/update_data.py` first.")
    return pd.concat((pd.read_csv(f) for f in files), ignore_index=True)


def _subset(df: pd.DataFrame, keys: list[str], wanted: list[str]) -> pd.DataFrame:
    return df[keys + [c for c in wanted if c in df.columns]]


def add_team_context(out: pd.DataFrame, teams_base: pd.DataFrame, teams_adv: pd.DataFrame) -> pd.DataFrame:
    """Attach team record / rating and compute the share of team game time each rookie plays."""
    team_keys = ["TEAM_ID", "SEASON"]
    t = _subset(teams_base, team_keys, ["GP", "MIN", "W_PCT"]).rename(
        columns={"GP": "TEAM_GP", "MIN": "TEAM_MIN", "W_PCT": "TEAM_W_PCT"}
    )
    ta = _subset(teams_adv, team_keys, ["NET_RATING", "PACE"]).rename(
        columns={"NET_RATING": "TEAM_NET_RATING", "PACE": "TEAM_PACE"}
    )
    out = out.merge(t, on=team_keys, how="left").merge(ta, on=team_keys, how="left")

    # Team "MIN" is either 5 x game minutes (240 per game) or game minutes (48 per game).
    per_game = out["TEAM_MIN"] / out["TEAM_GP"]
    team_game_minutes = np.where(per_game > 100, out["TEAM_MIN"] / 5, out["TEAM_MIN"])
    out["COURT_PCT"] = out["MIN"] / team_game_minutes  # share of team game time on court
    out["MIN_SHARE"] = out["COURT_PCT"] / 5  # share of all team player-minutes
    return out


def add_draft(out: pd.DataFrame, draft: pd.DataFrame) -> pd.DataFrame:
    d = (
        draft[["PERSON_ID", "SEASON", "ROUND_NUMBER", "OVERALL_PICK"]]
        .rename(columns={"PERSON_ID": "PLAYER_ID", "SEASON": "DRAFT_YEAR"})
        .drop_duplicates("PLAYER_ID")
        .assign(DRAFT_YEAR=lambda x: pd.to_numeric(x["DRAFT_YEAR"], errors="coerce"))
    )
    out = out.merge(d, on="PLAYER_ID", how="left")
    out["UNDRAFTED"] = out["OVERALL_PICK"].isna()
    out["PICK_MODEL"] = out["OVERALL_PICK"].fillna(UNDRAFTED_PICK)
    out["SEASON_START"] = out["SEASON"].str[:4].astype(int)
    out["DRAFTED_IN_CLASS"] = out["DRAFT_YEAR"] == out["SEASON_START"]
    return out


def add_onoff(out: pd.DataFrame, onoff: pd.DataFrame) -> pd.DataFrame:
    """Team net rating with the player on court vs. off court."""
    keys = ["TEAM_ID", "PLAYER_ID", "SEASON"]
    base = onoff.rename(columns={"VS_PLAYER_ID": "PLAYER_ID"})
    on = base[base["COURT"] == "on"][keys + ["MIN", "NET_RATING"]].rename(
        columns={"MIN": "ON_MIN", "NET_RATING": "ON_NET"}
    )
    off = base[base["COURT"] == "off"][keys + ["MIN", "NET_RATING"]].rename(
        columns={"MIN": "OFF_MIN", "NET_RATING": "OFF_NET"}
    )
    out = out.merge(on, on=keys, how="left").merge(off, on=keys, how="left")
    out["ONOFF"] = out["ON_NET"] - out["OFF_NET"]
    return out


def build_rookie_table(raw_dir: Path | str, min_minutes: float = 0.0) -> pd.DataFrame:
    """Merge all raw files into one table (lower-case columns, one row per rookie season)."""
    raw = Path(raw_dir)
    keys = ["PLAYER_ID", "SEASON"]

    out = read_raw(raw, "players_base")

    per100 = read_raw(raw, "players_per100")
    p100_cols = ["PTS", "AST", "REB", "STL", "BLK", "TOV"]
    p100 = _subset(per100, keys, p100_cols).rename(columns={c: f"{c}_PER100" for c in p100_cols})
    adv = read_raw(raw, "players_adv")
    adv_cols = ["POSS", "OFF_RATING", "DEF_RATING", "NET_RATING", "TS_PCT", "USG_PCT", "AST_PCT", "REB_PCT", "PIE"]
    out = out.merge(p100, on=keys, how="left").merge(_subset(adv, keys, adv_cols), on=keys, how="left")

    out = add_team_context(out, read_raw(raw, "teams_base"), read_raw(raw, "teams_adv"))
    out = add_draft(out, pd.read_csv(raw / "draft.csv"))
    if list(raw.glob("onoff_*.csv")):
        out = add_onoff(out, read_raw(raw, "onoff"))

    out["MPG"] = out["MIN"] / out["GP"]
    out["GMSC_TOTAL"] = game_score(out)
    out["GMSC_PER36"] = out["GMSC_TOTAL"] / out["MIN"] * 36
    if "NET_RATING" in out and "TEAM_NET_RATING" in out:
        out["NET_VS_TEAM"] = out["NET_RATING"] - out["TEAM_NET_RATING"]

    out = out[out["MIN"] >= min_minutes].sort_values(["SEASON", "MIN"], ascending=[True, False])
    out.columns = [c.lower() for c in out.columns]
    return out.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the processed rookie dataset.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/rookies.csv"))
    parser.add_argument("--min-minutes", type=float, default=50.0, help="drop rookies with fewer total minutes")
    args = parser.parse_args()
    table = build_rookie_table(args.raw_dir, args.min_minutes)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print(f"Wrote {len(table)} rookie seasons to {args.out}")


if __name__ == "__main__":
    main()
