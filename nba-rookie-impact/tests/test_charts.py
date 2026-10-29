import pytest

pytest.importorskip("plotly")

from rookies import add_value_over_pick, build_rookie_table  # noqa: E402
from rookies import charts  # noqa: E402
from synthetic import make_synthetic_raw  # noqa: E402


@pytest.fixture(scope="module")
def valued(tmp_path_factory):
    raw = make_synthetic_raw(tmp_path_factory.mktemp("raw"), n_teams=10)
    return add_value_over_pick(build_rookie_table(raw))


def test_minutes_impact_scatter(valued):
    df, _ = valued
    fig = charts.minutes_impact_scatter(df, "gmsc_per36", "Game Score per 36", zero_line=True)
    assert len(fig.data[0].x) == len(df)


def test_pick_curve_chart_has_points_and_curve(valued):
    df, curves = valued
    fig = charts.pick_curve_chart(df, curves["gmsc_total"], "Total Game Score", highlight=[df["season"].iloc[0]])
    assert len(fig.data) == 3  # other seasons, highlighted season, fitted curve
    assert fig.layout.xaxis.type == "log"


def test_surplus_bars_sorted_and_coloured_by_sign(valued):
    df, _ = valued
    fig = charts.surplus_bars(df, "gmsc_total_vs_expected", "Game Score vs. expected", n=5)
    x = list(fig.data[0].x)
    assert x == sorted(x)
    colours = fig.data[0].marker.color
    assert all((c == charts.BLUE) == (v >= 0) for c, v in zip(colours, x))
