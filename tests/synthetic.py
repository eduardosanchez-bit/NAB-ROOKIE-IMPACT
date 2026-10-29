"""Synthetic raw data with the same schema as the NBA downloads (for tests and demos only)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TEAM_GAMES = 82


def make_synthetic_raw(
    raw_dir: Path | str,
    seasons: tuple[str, ...] = ("2022-23", "2023-24"),
    n_teams: int = 6,
    rookies_per_team: int = 3,
    seed: int = 0,
) -> Path:
    rng = np.random.default_rng(seed)
    raw = Path(raw_dir)
    raw.mkdir(parents=True, exist_ok=True)

    draft_rows = []
    pid = 1000
    for season in seasons:
        start = int(season[:4])
        teams_base, teams_adv = [], []
        base, per100, adv, onoff = [], [], [], []

        for t in range(1, n_teams + 1):
            team_id = 100 + t
            team_net = float(rng.normal(0, 4))
            teams_base.append(
                dict(TEAM_ID=team_id, TEAM_NAME=f"Team {t}", GP=TEAM_GAMES, MIN=TEAM_GAMES * 240,
                     W_PCT=float(np.clip(0.5 + team_net / 30, 0.1, 0.9)), SEASON=season)
            )
            teams_adv.append(dict(TEAM_ID=team_id, NET_RATING=team_net, PACE=float(rng.normal(99, 2)), SEASON=season))

            for _ in range(rookies_per_team):
                pid += 1
                pick = None if rng.random() < 0.2 else int(rng.integers(1, 61))
                pick_eff = pick or 61
                minutes = float(np.clip(1900 - 330 * np.log(pick_eff) + rng.normal(0, 300), 60, 3000))
                gp = int(np.clip(minutes / rng.uniform(10, 30), 5, TEAM_GAMES))
                skill = rng.normal(0, 1)
                pts = minutes * (0.38 + 0.05 * skill + 0.01 * rng.normal())
                fga = pts * 0.85
                fta = pts * 0.2
                row = dict(
                    PLAYER_ID=pid, PLAYER_NAME=f"Player {pid}", TEAM_ID=team_id, TEAM_ABBREVIATION=f"T{t:02d}",
                    AGE=int(rng.integers(19, 25)), GP=gp, MIN=minutes, FGM=fga * 0.45, FGA=fga, FTM=fta * 0.75,
                    FTA=fta, OREB=minutes * 0.03, DREB=minutes * 0.09, REB=minutes * 0.12,
                    AST=minutes * (0.07 + 0.01 * skill), TOV=minutes * 0.05, STL=minutes * 0.02,
                    BLK=minutes * 0.01, PF=minutes * 0.07, PTS=pts, PLUS_MINUS=float(rng.normal(-20, 80)),
                    SEASON=season,
                )
                base.append(row)
                per100.append(
                    dict(PLAYER_ID=pid, SEASON=season, PTS=pts / minutes * 48, AST=row["AST"] / minutes * 48,
                         REB=row["REB"] / minutes * 48, STL=1.5, BLK=0.8, TOV=3.0)
                )
                adv.append(
                    dict(PLAYER_ID=pid, SEASON=season, POSS=minutes * 2.05, OFF_RATING=float(rng.normal(105, 6)),
                         DEF_RATING=float(rng.normal(110, 5)), NET_RATING=float(rng.normal(-4, 7)),
                         TS_PCT=float(np.clip(rng.normal(0.54, 0.05), 0.4, 0.7)),
                         USG_PCT=float(np.clip(rng.normal(0.19, 0.04), 0.08, 0.35)),
                         AST_PCT=float(np.clip(rng.normal(0.12, 0.06), 0.02, 0.4)),
                         REB_PCT=float(np.clip(rng.normal(0.09, 0.04), 0.02, 0.25)),
                         PIE=float(np.clip(0.07 + 0.015 * skill + 0.00002 * minutes / 10, 0.01, 0.2)))
                )
                team_minutes = TEAM_GAMES * 48
                onoff.append(dict(TEAM_ID=team_id, VS_PLAYER_ID=pid, COURT="on", MIN=minutes,
                                  NET_RATING=team_net + float(rng.normal(1, 4)), SEASON=season))
                onoff.append(dict(TEAM_ID=team_id, VS_PLAYER_ID=pid, COURT="off", MIN=team_minutes - minutes,
                                  NET_RATING=team_net - float(rng.normal(1, 2)), SEASON=season))
                if pick is not None:
                    draft_rows.append(
                        dict(PERSON_ID=pid, PLAYER_NAME=row["PLAYER_NAME"], SEASON=str(start),
                             ROUND_NUMBER=1 if pick <= 30 else 2, OVERALL_PICK=pick)
                    )

        for name, rows in (("players_base", base), ("players_per100", per100), ("players_adv", adv),
                           ("teams_base", teams_base), ("teams_adv", teams_adv), ("onoff", onoff)):
            pd.DataFrame(rows).to_csv(raw / f"{name}_{season}.csv", index=False)

    pd.DataFrame(draft_rows).to_csv(raw / "draft.csv", index=False)
    return raw
