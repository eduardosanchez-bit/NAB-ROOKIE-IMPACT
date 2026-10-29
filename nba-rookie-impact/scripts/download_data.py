"""Download raw NBA data (stats.nba.com through ``nba_api``) into CSV files.

This script deliberately does NOT use pandas: it only needs ``nba_api`` and the
standard library, so it can run on machines where pandas is blocked. It writes
to ``data/raw/``. Requests already on disk are skipped, so it is safe to stop
and re-run.

stats.nba.com ignores most cloud servers (Codespaces, Streamlit Cloud...), so run
this from a normal home connection.

Usage (from the project root):
    python scripts/download_data.py --start 2022 --end 2024 --skip-onoff   # quick test
    python scripts/download_data.py                                         # 2015-16 .. 2025-26
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = ROOT / "data" / "raw"
TIMEOUT = 30  # seconds per request


def season_label(start_year: int) -> str:
    """2023 -> '2023-24'."""
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def _retry(fn: Callable, retries: int = 2, wait: float = 3.0):
    last: Exception | None = None
    for attempt in range(retries):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - network errors vary widely
            last = exc
            time.sleep(wait * (attempt + 1))
    raise RuntimeError(f"NBA API request failed after {retries} attempts: {type(last).__name__}: {last}") from last


def _save(rows: list[dict], path: Path) -> None:
    if not rows:
        raise RuntimeError(f"The API returned no rows for {path.name}.")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_column(path: Path, column: str) -> list[str]:
    with path.open(newline="", encoding="utf-8") as f:
        return [row[column] for row in csv.DictReader(f)]


def _first_dataset(endpoint) -> list[dict]:
    """Rows of the first (only) dataset of an endpoint, without pandas."""
    return next(iter(endpoint.get_normalized_dict().values()))


# ------------------------------------------------------------------ API calls
def check_connection(timeout: int = 15) -> bool:
    from nba_api.stats.endpoints import leaguedashteamstats

    try:
        leaguedashteamstats.LeagueDashTeamStats(season=season_label(2023), timeout=timeout)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"Could not reach stats.nba.com ({type(exc).__name__}).")
        return False


def _player_rows(season: str, measure: str, mode: str) -> list[dict]:
    from nba_api.stats.endpoints import leaguedashplayerstats

    ep = _retry(
        lambda: leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            season_type_all_star="Regular Season",
            measure_type_detailed_defense=measure,
            per_mode_detailed=mode,
            player_experience_nullable="Rookie",
            timeout=TIMEOUT,
        )
    )
    return [{**row, "SEASON": season} for row in _first_dataset(ep)]


def _team_rows(season: str, measure: str) -> list[dict]:
    from nba_api.stats.endpoints import leaguedashteamstats

    ep = _retry(
        lambda: leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            season_type_all_star="Regular Season",
            measure_type_detailed_defense=measure,
            per_mode_detailed="Totals",
            timeout=TIMEOUT,
        )
    )
    return [{**row, "SEASON": season} for row in _first_dataset(ep)]


def _onoff_rows(season: str, team_ids: list[int], pause: float) -> list[dict]:
    """On/off court net rating for every player of every team in a season."""
    from nba_api.stats.endpoints import teamplayeronoffsummary

    rows: list[dict] = []
    for team_id in team_ids:
        ep = _retry(
            lambda: teamplayeronoffsummary.TeamPlayerOnOffSummary(
                team_id=team_id,
                season=season,
                season_type_all_star="Regular Season",
                measure_type_detailed_defense="Advanced",
                per_mode_detailed="Totals",
                timeout=TIMEOUT,
            )
        )
        data = ep.get_normalized_dict()
        for key, court in (
            ("PlayersOnCourtTeamPlayerOnOffSummary", "on"),
            ("PlayersOffCourtTeamPlayerOnOffSummary", "off"),
        ):
            rows.extend({**row, "COURT": court, "SEASON": season} for row in data[key])
        time.sleep(pause)
    return rows


def _draft_rows(first_year: int = 2005) -> list[dict]:
    from nba_api.stats.endpoints import drafthistory

    ep = _retry(lambda: drafthistory.DraftHistory(league_id="00", timeout=TIMEOUT))
    keep = []
    for row in _first_dataset(ep):
        try:
            if int(row["SEASON"]) >= first_year:
                keep.append(row)
        except (TypeError, ValueError):
            continue
    return keep


# ------------------------------------------------------------------ orchestration
def fetch_season(start_year: int, raw_dir: Path, pause: float = 1.0, with_onoff: bool = True) -> None:
    season = season_label(start_year)
    print(f"Season {season}")

    jobs = {
        "players_base": lambda: _player_rows(season, "Base", "Totals"),
        "players_per100": lambda: _player_rows(season, "Base", "Per100Possessions"),
        "players_adv": lambda: _player_rows(season, "Advanced", "Totals"),
        "teams_base": lambda: _team_rows(season, "Base"),
        "teams_adv": lambda: _team_rows(season, "Advanced"),
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
        team_ids = [int(t) for t in _read_column(raw_dir / f"teams_base_{season}.csv", "TEAM_ID")]
        _save(_onoff_rows(season, team_ids, pause), onoff_path)
        print("  onoff: saved")


def fetch_all(start: int, end: int, raw_dir: Path = DEFAULT_RAW_DIR, pause: float = 1.0, with_onoff: bool = True) -> None:
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    draft_path = raw_dir / "draft.csv"
    if not draft_path.exists():
        _save(_draft_rows(), draft_path)
        print("Draft history saved")
    for year in range(start, end + 1):
        fetch_season(year, raw_dir, pause, with_onoff)


def main() -> None:
    global TIMEOUT
    parser = argparse.ArgumentParser(description="Download raw NBA rookie data (no pandas needed).")
    parser.add_argument("--start", type=int, default=2015, help="first season start year (2015 -> 2015-16)")
    parser.add_argument("--end", type=int, default=2025, help="last season start year (2025 -> 2025-26)")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--pause", type=float, default=1.0, help="seconds between requests")
    parser.add_argument("--timeout", type=int, default=TIMEOUT, help="seconds to wait for each request")
    parser.add_argument("--skip-onoff", action="store_true", help="skip the slow on/off court requests")
    parser.add_argument("--no-check", action="store_true", help="skip the initial connection check")
    args = parser.parse_args()
    TIMEOUT = args.timeout

    if not args.no_check and not check_connection():
        raise SystemExit(
            "stats.nba.com is not answering from this machine. It usually blocks cloud servers "
            "(Codespaces, etc.): run this script from your own computer and home/mobile connection."
        )
    fetch_all(args.start, args.end, args.raw_dir, args.pause, not args.skip_onoff)
    print("Done. Next: commit data/raw and run `python scripts/build_dataset.py` where pandas works.")


if __name__ == "__main__":
    main()
