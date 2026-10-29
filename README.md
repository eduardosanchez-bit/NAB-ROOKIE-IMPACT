# NBA Rookie Impact

How much do NBA rookies play, what do they contribute to their teams, and is that more or less than their draft slot predicts?

The project downloads official NBA statistics, builds one row per rookie season, and serves an interactive dashboard.

## Questions it answers

1. **Playing time and impact**: minutes, share of team game time on court, possessions, Game Score, PIE, team net rating with and without the rookie.
2. **Draft value**: expected production at each draft slot, and which rookies over- or under-performed it.

Planned next: predicting second-year outcomes from rookie-year stats, and the effect of rookie minutes on team results (see Roadmap).

## Data

All data comes from stats.nba.com through the [`nba_api`](https://github.com/swar/nba_api) package, for regular seasons 2015-16 to 2025-26:

| File | Content |
|---|---|
| `players_base_*` | Totals per rookie (minutes, points, assists, rebounds…) |
| `players_per100_*` | Per-100-possession box score |
| `players_adv_*` | Possessions, usage, TS%, PIE, net rating |
| `teams_base_*`, `teams_adv_*` | Team record, minutes, net rating, pace |
| `onoff_*` | Team net rating with each player on and off court |
| `draft.csv` | Draft year, round and overall pick |

Raw downloads go to `data/raw/` (git-ignored). The processed table `data/processed/rookies.csv` **is committed** so the app runs anywhere without downloading anything.

## Setup

```bash
git clone https://github.com/<your-user>/nba-rookie-impact.git
cd nba-rookie-impact
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 1. Download the data

```bash
python scripts/update_data.py --start 2022 --end 2024 --skip-onoff   # quick first test (~1 minute)
python scripts/update_data.py                                         # everything (~15-20 minutes)
```

Requests are cached on disk, so you can stop and re-run without starting over. stats.nba.com sometimes blocks or throttles cloud servers; if it times out, try again later, increase `--pause`, or run the script from a home connection.

### 2. Run the app

```bash
python -m streamlit run app.py
```

### 3. Tests

```bash
python -m pytest
```

The tests use synthetic data with the same schema as the real downloads, so they run offline.

## Method

```
Court %         = rookie minutes / team game minutes
Game Score      = PTS + 0.4 FGM − 0.7 FGA − 0.4 (FTA − FTM) + 0.7 OREB + 0.3 DREB
                  + STL + 0.7 AST + 0.7 BLK − 0.4 PF − TOV
Net vs. team    = net rating with rookie on court − team net rating
Expected(pick)  = a + b · ln(pick)          (least squares over all rookies)
Value vs. draft = actual − Expected(pick)   (undrafted modelled as pick 61)
```

## Project structure

```
nba-rookie-impact/
├── app.py                    # Streamlit dashboard
├── scripts/update_data.py    # download + build in one command
├── src/rookies/
│   ├── fetch.py              # nba_api → raw CSV (cached)
│   ├── metrics.py            # merge, court %, Game Score, on/off, draft
│   ├── value.py              # expected production by draft slot
│   └── charts.py             # Plotly figures
├── tests/                    # pytest suite on synthetic data
└── data/processed/rookies.csv
```

## Limitations

- Rookies traded mid-season are attributed to their last team, so their team context and on/off numbers can be missing or off.
- Players drafted who never played as rookies are not in the data (survivorship), which makes the expected-production curve slightly optimistic for late picks.
- Bad teams give rookies more minutes on purpose, so playing time is not independent of team quality. Do not read "more rookie minutes → worse team" as causal.
- On/off ratings for rookies are noisy and depend on teammates.
- Rookie contract salaries are not included yet, so "value" is measured against draft slot, not dollars.

## Roadmap

- [ ] Review of the first full run and fixes
- [ ] Rookie scale salaries → surplus value per dollar
- [ ] Second-year prediction from rookie-year stats
- [ ] Team-level effect of rookie minutes with controls for tanking
- [ ] Notebook with a worked example

## License

MIT
