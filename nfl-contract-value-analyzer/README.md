# NFL Contract Value Analyzer

Streamlit dashboard with Pandas, Plotly, and scikit-learn. **Bundled data is entirely fictional.** Team abbreviations and season labels are sample categories, not historical observations.

## Run
Use Python 3.12:
```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```
Windows activation: `.venv\Scripts\activate`.

## Features
- Season, position, team, games, and name filters
- League rankings, salary-vs-production scatterplot, player detail view
- CSV uploads and filtered CSV exports
- 96 demo records: 48 fictional players across 2024 and 2025

## Files
- `app.py`: dashboard entry point
- `requirements.txt`: pinned dependencies
- `src/metrics.py`: validation and production score
- `src/valuation.py`: position-relative salary model
- `data/players.csv`: demo data
- `tests/test_analyzer.py`: model and dashboard checks
- `.streamlit/config.toml`: theme

## Model
Production is a PPR-style counting proxy: passing yards × .04 + passing TDs × 4 − interceptions × 2 + rushing yards × .1 + rushing TDs × 6 + receptions + receiving yards × .1 + receiving TDs × 6. Season totals retain missed-game impact. Midrank percentiles compare only the same position and season.

Each player's benchmark excludes their own salary. With at least five other players and variable production, StandardScaler + Ridge(alpha=5) predicts log annual salary from production. Exponentiated predictions are clamped to the observed peer salary range. Smaller or constant cohorts use peer median; no peers means no valuation. All calculations precede display filters.

Value ratio = benchmark / salary; surplus = benchmark − salary. Ratios ≥1.2 indicate below-benchmark salary; ≤.8 indicate above-benchmark salary. Position rank compares ratios within each position and season. This is a descriptive geometric salary benchmark, not expected future value.

Only QB, RB, WR, and TE are supported. Counting statistics omit blocking, defense, scheme, age, guarantees, contract timing, and future performance. Rookie contracts can look unusually efficient. APY differs from cap hit and cash paid. Demo outcomes have no real-world valuation meaning. The model has not been validated on real NFL data.

## Real data
Upload a conforming CSV or replace `data/players.csv`. See `data/README.md`. Use one aggregate regular-season row per stable player ID and season. Document sources, retrieval dates, APY timing, traded-player conventions, and unmatched rows. Do not mix demo and real data. No external ingestion is wired in.

## Tests
```sh
python -m pip install pytest
python -m pytest -q
```

## GitHub upload
Open your repository, choose **Add file → Upload files** (or **uploading an existing file** for an empty repository). Drag the contents of this folder into the upload area, including `src/` and `data/`, and click **Commit changes**. Upload the extracted contents, not the ZIP or outer folder. Verify `app.py` is visible at repository root.

Alternatively, from this folder:
```sh
git init -b main
git add .
git commit -m "Build NFL analyzer MVP"
git remote add origin https://github.com/darshp015-creator/nfl-contract-value-analyzer.git
git push -u origin main
```
Authenticate normally; never put credentials in files.

## Streamlit Community Cloud
1. Sign in at https://share.streamlit.io/ and choose Create app.
2. Select `darshp015-creator/nfl-contract-value-analyzer`.
3. Branch: `main`. Main file path: `app.py`.
4. Advanced settings: Python 3.12. No secrets required.
5. Deploy, then confirm the demo warning and dashboard render.

The branch and file must already be committed on GitHub. See the official deployment guide: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
