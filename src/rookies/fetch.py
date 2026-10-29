"""Download raw NBA data (stats.nba.com through ``nba_api``) into CSV files.

Every request is cached on disk: files that already exist are skipped, so the
script can be interrupted and re-run safely.

Usage:
    python -m rookies.fetch --start 2015 --end 2025
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Callable

import pandas as pd

DEFAULT_RAW_DIR = Path("data/raw")


def season_label(start_year: int) -> str:
    """2023 -> '2023-24'."""
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def _retry(fn: Callable, retries: int = 3, wait: float = 3.0):
    last: Exception | None = None
    for attempt in range(retries):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - network errors vary widely
            last = exc
            time.sleep(wait * (attempt + 1))
    raise RuntimeError(f"NBA API request failed after {retries} attempts: {last}") from last


def _save(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _player_frame(season: str, measure: str, mode: str) -> pd.DataFrame:
    from nba_api.stats.endpoints import leaguedashplayerstats

    ep = _retry(
        lambda: leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            season_type_all_star="Regular Season",
            measure_type_detailed_defense=measure,
            per_mode_detailed=mode,
            player_experience_nullable="Rookie",
            timeout=60,
        )
    )
    return ep.get_data_frames()[0].assign(SEASON=season)


def _team_frame(season: str, measure: str) -> pd.DataFrame:
    from nba_api.stats.endpoints import leaguedashteamstats

    ep = _retry(
        lambda: leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            season_type_all_star="Regular Season",
            measure_type_detailed_defense=measure,
            per_mode_detailed="Totals",
            timeout=60,
        )
    )
    return ep.get_data_frames()[0].assign(SEASON=season)


def _onoff_frame(season: str, team_ids: list[int], pause: float) -> pd.DataFrame:
    """On/off court net rating for every player of every team in a season."""
    from nba_api.stats.endpoints import teamplayeronoffsummary

    frames = []
    for team_id in team_ids:
        ep = _retry(
            lambda: teamplayeronoffsummary.TeamPlayerOnOffSummary(
                team_id=team_id,
                season=season,
                season_type_all_star="Regular Season",
                measure_type_detailed_defense="Advanced",
                per_mode_detailed="Totals",
                timeout=60,
            )
        )
        data = ep.get_normalized_dict()
        for key, court in (
            ("PlayersOnCourtTeamPlayerOnOffSummary", "on"),
            ("PlayersOffCourtTeamPlayerOnOffSummary", "off"),
        ):
            frames.append(pd.DataFrame(data[key]).assign(COURT=court, SEASON=season))
        time.sleep(pause)
    return pd.concat(frames, ignore_index=True)


def fetch_draft(raw_dir: Path, first_year: int = 2005) -> None:
    path = raw_dir / "draft.csv"
    if path.exists():
        return
    from nba_api.stats.endpoints import drafthistory

    ep = _retry(lambda: drafthistory.DraftHistory(league_id="00", timeout=60))
    df = ep.get_data_frames()[0]
    df = df[pd.to_numeric(df["SEASON"], errors="coerce") >= first_year]
    _save(df, path)
    print(f"  draft history: {len(df)} picks")


def fetch_season(start_year: int, raw_dir: Path, pause: float = 1.0, with_onoff: bool = True) -> None:
    season = season_label(start_year)
    print(f"Season {season}")

    jobs = {
        "players_base": lambda: _player_frame(season, "Base", "Totals"),
        "players_per100": lambda: _player_frame(season, "Base", "Per100Possessions"),
        "players_adv": lambda: _player_frame(season, "Advanced", "Totals"),
        "teams_base": lambda: _team_frame(season, "Base"),
        "teams_adv": lambda: _team_frame(season, "Advanced"),
    }
    for kind, job in jobs.items():
        path = raw_dir / f"{kind}_{season}.csv"
        if path.exists():
            continue
        _save(job(), path)
        print(f"  {kind}: saved")
        time.sleep(pause)

    onoff_path = raw_dir / f"onoff_{season}.csv"
    if with_onoff and not onoff_path.exists():
        teams = pd.read_csv(raw_dir / f"teams_base_{season}.csv")
        _save(_onoff_frame(season, teams["TEAM_ID"].tolist(), pause), onoff_path)
        print("  onoff: saved")


def fetch_all(start: int, end: int, raw_dir: Path = DEFAULT_RAW_DIR, pause: float = 1.0, with_onoff: bool = True) -> None:
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    fetch_draft(raw_dir)
    for year in range(start, end + 1):
        fetch_season(year, raw_dir, pause, with_onoff)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download raw NBA rookie data.")
    parser.add_argument("--start", type=int, default=2015, help="first season start year (2015 -> 2015-16)")
    parser.add_argument("--end", type=int, default=2025, help="last season start year (2025 -> 2025-26)")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--pause", type=float, default=1.0, help="seconds between requests")
    parser.add_argument("--skip-onoff", action="store_true", help="skip the slow on/off court requests")
    args = parser.parse_args()
    fetch_all(args.start, args.end, args.raw_dir, args.pause, not args.skip_onoff)
    print("Done.")


if __name__ == "__main__":
    main()
