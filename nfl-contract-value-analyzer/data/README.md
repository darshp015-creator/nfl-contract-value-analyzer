# Data and upload schema

`nfl_2022_2025.csv` contains real regular-season data. `players.csv` contains fictional demo data. They are never combined. Snapshot sources and coverage are in `sources.json`; rejected records and reasons are in `exclusions.csv`.

## Required upload columns
- `player_id`, `player_name`, `team`: nonblank strings
- `season`: integer season year; 2025 denotes the 2025–2026 NFL season
- `position`: QB, RB, WR, or TE
- `games`: integer 0–17
- `salary`: positive APY in dollars (15000000 means $15M)
- `is_demo`: true or false
- `passing_yards`, `rushing_yards`, `receiving_yards`: integer net yards; negatives are valid
- `passing_tds`, `interceptions`, `rushing_tds`, `receptions`, `receiving_tds`: nonnegative integer totals

One row per player ID and season. Zero-game players must have zero statistics. Missing or infinite required values are rejected. Use zero only for truly inapplicable statistics, not unknown values.

## Optional improvements
- `cap_hit`, `cash_paid`: season costs in dollars; missing/nonpositive costs are not valued
- `contract_type`: Drafted or UDFA => Rookie deal; UFA, SFA, Extension, RFA, ERFA, Franchise, Transition => Veteran deal; other values => Unknown. This describes the contract, not first-year playing status.
- Advanced scoring requires these columns (individual cells may be missing): `passing_epa`, `passing_cpoe`, `rushing_epa`, `receiving_epa`, `attempts`, `sacks_suffered`, `targets`, `carries`, `team_targets`, `rushing_fumbles_lost`, `sack_fumbles_lost`.
- `passing_cpoe` uses percentage points; EPA fields are season totals; opportunities are counts; `team_targets` is the team's full regular-season target total.

Older basic CSVs work with the box-score proxy. Download the bundled dataset from Data & methods as a complete template.

## Provenance and limits
Production comes from nflverse regular-season statistics. Contracts come from nflverse's historical Over The Cap snapshot. Source contract amounts in millions are converted to dollars. Stable GSIS IDs join the sources. Annual cash and cap values are deduplicated rather than summed across repeated contract histories.

Included players must have single-team weekly statistics, a matching team contract, identifiable APY, and evidence of earned cash in that season. Weekly and season core totals are checked. Multi-team, ambiguous, practice/reserve-only and unmatched records are excluded. APY uses the latest identifiable non-practice deal signed by season end; extensions may start later. Historical values retrieved in 2026 are not frozen season-opening snapshots. Coverage therefore does not represent every player or a full team payroll.

Supported source players / included: 2022 608/539; 2023 577/530; 2024 589/542; 2025 610/555. The 2025 season includes regular-season games played in January 2026, but excludes playoffs. Trends show only available records; absent years do not mean zero production or cost.
