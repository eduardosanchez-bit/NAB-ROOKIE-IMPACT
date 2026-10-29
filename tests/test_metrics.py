import numpy as np
import pandas as pd
import pytest

from rookies import UNDRAFTED_PICK, build_rookie_table, game_score
from rookies.metrics import add_team_context
from synthetic import make_synthetic_raw


@pytest.fixture(scope="module")
def table(tmp_path_factory):
    raw = make_synthetic_raw(tmp_path_factory.mktemp("raw"))
    return build_rookie_table(raw)


def test_game_score_hand_calculation():
    row = pd.DataFrame(
        [dict(PTS=10, FGM=4, FGA=10, FTM=1, FTA=2, OREB=1, DREB=2, STL=1, AST=2, BLK=1, PF=3, TOV=2)]
    )
    # 10 + 1.6 - 7 - 0.4 + 0.7 + 0.6 + 1 + 1.4 + 0.7 - 1.2 - 2
    assert game_score(row).iloc[0] == pytest.approx(5.4)


def test_one_row_per_rookie_season(table):
    assert len(table) == 2 * 6 * 3
    assert not table.duplicated(["player_id", "season"]).any()


def test_court_pct_is_share_of_team_game_time(table):
    expected = table["min"] / (82 * 48)
    assert np.allclose(table["court_pct"], expected)
    assert np.allclose(table["min_share"], expected / 5)
    assert table["court_pct"].between(0, 1).all()


def test_team_minutes_scale_does_not_matter():
    players = pd.DataFrame([dict(PLAYER_ID=1, SEASON="2023-24", TEAM_ID=1, MIN=2400.0)])
    adv = pd.DataFrame([dict(TEAM_ID=1, SEASON="2023-24", NET_RATING=1.0, PACE=99.0)])
    five_man = pd.DataFrame([dict(TEAM_ID=1, SEASON="2023-24", GP=82, MIN=82 * 240, W_PCT=0.5)])
    game_min = pd.DataFrame([dict(TEAM_ID=1, SEASON="2023-24", GP=82, MIN=82 * 48, W_PCT=0.5)])
    a = add_team_context(players, five_man, adv)["COURT_PCT"].iloc[0]
    b = add_team_context(players, game_min, adv)["COURT_PCT"].iloc[0]
    assert a == pytest.approx(b) == pytest.approx(2400 / (82 * 48))


def test_gmsc_per36_is_rate_of_total(table):
    assert np.allclose(table["gmsc_per36"], table["gmsc_total"] / table["min"] * 36)


def test_draft_merge_and_undrafted_flag(table):
    und = table[table["undrafted"]]
    assert len(und) > 0
    assert und["overall_pick"].isna().all()
    assert (und["pick_model"] == UNDRAFTED_PICK).all()
    drafted = table[~table["undrafted"]]
    assert (drafted["pick_model"] == drafted["overall_pick"]).all()
    assert drafted["drafted_in_class"].all()


def test_onoff_difference(table):
    assert np.allclose(table["onoff"], table["on_net"] - table["off_net"])
    assert table["onoff"].notna().all()


def test_net_vs_team(table):
    assert np.allclose(table["net_vs_team"], table["net_rating"] - table["team_net_rating"])


def test_min_minutes_filter(tmp_path):
    raw = make_synthetic_raw(tmp_path)
    full = build_rookie_table(raw)
    filtered = build_rookie_table(raw, min_minutes=1000)
    assert 0 < len(filtered) < len(full)
    assert (filtered["min"] >= 1000).all()


def test_missing_raw_files_give_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="update_data"):
        build_rookie_table(tmp_path)
