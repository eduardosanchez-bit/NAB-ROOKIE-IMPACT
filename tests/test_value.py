import numpy as np
import pandas as pd
import pytest

from rookies import add_value_over_pick, build_rookie_table, class_summary, fit_pick_curve
from synthetic import make_synthetic_raw


@pytest.fixture(scope="module")
def table(tmp_path_factory):
    raw = make_synthetic_raw(tmp_path_factory.mktemp("raw"), n_teams=10)
    return build_rookie_table(raw)


def test_fit_recovers_known_curve():
    picks = np.arange(1, 61)
    df = pd.DataFrame({"pick_model": picks, "y": 500 - 100 * np.log(picks)})
    curve = fit_pick_curve(df, "y")
    assert curve.slope == pytest.approx(-100)
    assert curve.intercept == pytest.approx(500)
    assert curve.r_squared == pytest.approx(1.0)
    assert curve.predict(1) == pytest.approx(500)


def test_fit_needs_enough_rows():
    with pytest.raises(ValueError):
        fit_pick_curve(pd.DataFrame({"pick_model": [1, 2], "y": [1, 2]}), "y")


def test_earlier_picks_are_expected_to_play_more(table):
    _, curves = add_value_over_pick(table)
    assert curves["min"].slope < 0
    assert curves["min"].predict(1) > curves["min"].predict(60)


def test_residuals_average_to_zero_and_columns_exist(table):
    valued, _ = add_value_over_pick(table)
    for m in ("min", "gmsc_total", "gmsc_per36"):
        assert f"{m}_expected" in valued and f"{m}_vs_expected" in valued
        assert valued[f"{m}_vs_expected"].mean() == pytest.approx(0, abs=1e-6)
    assert np.allclose(valued["min_vs_expected"], valued["min"] - valued["min_expected"])


def test_class_summary_has_one_row_per_season(table):
    s = class_summary(table)
    assert list(s.index) == sorted(table["season"].unique())
    assert s["Rookies"].sum() == len(table)
