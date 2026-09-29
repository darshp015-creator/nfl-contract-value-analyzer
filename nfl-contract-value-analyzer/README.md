# NFL Contract Value Analyzer

A Streamlit dashboard for position-relative NFL production and contract comparisons. The default dataset contains **2,166 real player-season records from 2022–2025**, including 555 players in the 2025 regular season (the season ending in early 2026). Fictional demo data remains a separate option.

## Run locally
Use Python 3.12:
```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```
Windows activation: `.venv\Scripts\activate`.
Open the Local URL printed by Streamlit in your browser.

## Features
- Position-specific production scores, with a box-score fallback for older CSV uploads
- Contract APY, season cap hit, and season cash paid selectors
- Rookie/veteran contract filters and optional separate peer benchmarks
- Automatic real-data loading with source dates, coverage, and downloadable exclusions
- Rankings, Plotly scatterplot, player details, same-position comparisons, and historical trends
- Chronological model checks against median salary, with empirical error bands

Display filters do not change the fitted peer population. Salary choices and scoring options do. Minimum workload filters help identify small samples.

## Scoring and valuation
Scores combine weighted midrank percentiles within position and season:

| Position | Components and weights |
| --- | --- |
| QB | Passing EPA per attempt plus sack 45%; CPOE 25%; lower turnover rate 15%; passing plus rushing EPA 15% |
| RB | Rushing EPA per carry 35%; scrimmage yards 25%; yards per carry 20%; receiving EPA 20% |
| WR / TE | Receiving EPA per target 35%; yards per target 25%; team target share 25%; catch rate 15% |

EPA means expected points added; CPOE means completion percentage over expectation. Missing components are omitted and the remaining weights are renormalized. These weights are transparent design choices, not learned causal contributions. Routes, blocking, defensive play, age, guarantees, and future performance are absent. Yards per route run is unavailable in the public source used here.

Each player's benchmark excludes their own cost. With at least eight peers and variable scores, a StandardScaler + Ridge(alpha=5) model predicts log cost from production score and games. Predictions are bounded by the observed peer cost range. Smaller cohorts use median cost. Peer groups share season, position, and (by default) rookie/veteran/unknown deal category. No peers means no benchmark. Value ratio = benchmark / cost; surplus = benchmark − cost. These are descriptive comparisons, not estimates of true player worth.

The detail view's 10th–90th percentile peer cost range describes market spread, not prediction uncertainty. The historical test trains only on earlier seasons and compares against their median cost. Error bands use absolute errors from strictly earlier test seasons, with at least 20 calibration records in the cohort. Their observed coverage is reported; no coverage guarantee is claimed. Test-season production is already known: this is retrospective salary fit, not a preseason forecast. Players may recur across seasons; dollars are nominal and contracts are reconstructed from a later snapshot.

Under default APY/position scoring/separate deal settings, the 2025 test covers 520 players: mean absolute error is about $3.61M versus $4.60M for the median baseline (21.5% lower). Coverage and results change with the cost/scoring choice. Cohorts without enough training records are omitted.

## Data and refresh
Sources: [nflverse player statistics](https://github.com/nflverse/nflverse-data/releases/tag/stats_player) and [nflverse historical Over The Cap contracts](https://github.com/nflverse/nflverse-data/releases/tag/contracts). Snapshot retrieved September 28, 2026. See `data/README.md` and the dashboard's Data & methods tab.

The app automatically opens the bundled snapshot; it does not schedule remote refreshes. Rebuild the snapshot explicitly:
```sh
python scripts/prepare_data.py
```
This downloads public source files to `.data-cache/`; use `python scripts/prepare_data.py --refresh` to download fresh source files. Source file hashes and retrieval dates are recorded in `sources.json`. Review resulting coverage and exclusions before publishing. APY is the latest identifiable deal signed by season end, which can include extensions starting later. Single-team players with verified contract matches are included; multi-team and ambiguous records are excluded.

## Project layout
- `app.py`: dashboard
- `src/metrics.py`: core CSV validation and box-score proxy
- `src/analytics.py`: position scoring, peer valuation, chronological tests
- `src/valuation.py`: retained original benchmark API
- `data/nfl_2022_2025.csv`, `sources.json`, `exclusions.csv`: real snapshot and provenance
- `data/players.csv`: fictional demo data
- `scripts/prepare_data.py`: reproducible source join
- `tests/`: model and dashboard checks

## Tests
```sh
python -m pip install pytest
python -m pytest -q
```
Tests check salary exclusion, chronological leakage, cost switching, missing metrics, contract groups, and dashboard interactions.

## GitHub and Streamlit deployment
The existing repository stores the project inside `nfl-contract-value-analyzer/`:
- Repository: `darshp015-creator/nfl-contract-value-analyzer`
- Branch: `main`
- Main file path: `nfl-contract-value-analyzer/app.py`
- Python: 3.12; no secrets needed

Keep `requirements.txt`, `src/`, and `data/` beside `app.py`. Upload extracted files to the matching repository folder, then commit. Community Cloud updates from GitHub. If starting a new repository with the project contents at its root, use `app.py` instead. Do not upload the ZIP as the application.
