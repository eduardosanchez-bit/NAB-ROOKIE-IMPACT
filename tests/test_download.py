"""The download script must work without pandas and produce files the build step can read."""

import csv
import importlib.util
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from rookies import build_rookie_table
from synthetic import make_synthetic_raw

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "download_data.py"


def load_script():
    spec = importlib.util.spec_from_file_location("download_data", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_script_imports_without_pandas_or_numpy():
    code = (
        "import sys; sys.modules['pandas'] = None; sys.modules['numpy'] = None; "
        f"import runpy; runpy.run_path(r'{SCRIPT}', run_name='not_main'); print('ok')"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_season_label():
    mod = load_script()
    assert mod.season_label(2023) == "2023-24"
    assert mod.season_label(1999) == "1999-00"


def test_save_writes_union_of_columns_and_rejects_empty(tmp_path):
    mod = load_script()
    mod._save([{"a": 1}, {"a": 2, "b": 3}], tmp_path / "x.csv")
    rows = read_rows(tmp_path / "x.csv")
    assert rows == [{"a": "1", "b": ""}, {"a": "2", "b": "3"}]
    with pytest.raises(RuntimeError):
        mod._save([], tmp_path / "empty.csv")


def test_download_then_build_roundtrip(tmp_path, monkeypatch):
    """Serve synthetic rows through the download script, then build from what it wrote."""
    api = make_synthetic_raw(tmp_path / "fake_api")
    mod = load_script()

    def rows(kind, season):
        return read_rows(api / f"{kind}_{season}.csv")

    player_kinds = {
        ("Base", "Totals"): "players_base",
        ("Base", "Per100Possessions"): "players_per100",
        ("Advanced", "Totals"): "players_adv",
    }
    monkeypatch.setattr(mod, "_player_rows", lambda season, measure, mode: rows(player_kinds[(measure, mode)], season))
    monkeypatch.setattr(mod, "_team_rows", lambda season, measure: rows("teams_base" if measure == "Base" else "teams_adv", season))
    monkeypatch.setattr(mod, "_onoff_rows", lambda season, team_ids, pause: rows("onoff", season))
    monkeypatch.setattr(mod, "_draft_rows", lambda first_year=2005: read_rows(api / "draft.csv"))

    out = tmp_path / "raw"
    mod.fetch_all(2022, 2023, out, pause=0)
    assert (out / "draft.csv").exists()
    assert (out / "onoff_2023-24.csv").exists()

    built = build_rookie_table(out)
    pd.testing.assert_frame_equal(built, build_rookie_table(api))

    # re-running skips what is already on disk (no API calls)
    monkeypatch.setattr(mod, "_player_rows", lambda *a, **k: pytest.fail("should not re-download"))
    mod.fetch_all(2022, 2023, out, pause=0)
