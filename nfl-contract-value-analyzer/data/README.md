# Dataset schema

All bundled names, IDs, salaries, and statistics are fictional. Salary means annual average contract value (APY) in nominal USD.

Required columns:
- player_id, player_name, team: nonblank strings
- season: whole-number season year
- position: QB, RB, WR, TE
- games: integer 0–17
- salary: positive annual dollars (15000000 means $15M)
- is_demo: true or false; never mix demo and real rows
- passing_yards, passing_tds, interceptions, rushing_yards, rushing_tds, receptions, receiving_yards, receiving_tds: nonnegative integer season totals

One player_id per season. Zero-game players must have zero production. Missing or infinite numeric values are rejected. Set genuinely inapplicable stats to zero, but do not replace unknown stats with zero. Rare negative net-yardage records require a deliberate schema adjustment before import.

For real data, join verified production and contract sources by stable ID and season. Aggregate traded players before joining and choose a documented team convention. Record provenance and the point in time at which APY was measured. Coverage affects the benchmark: a partial uploaded league produces partial peer cohorts. The app does not scrape or verify source data.
