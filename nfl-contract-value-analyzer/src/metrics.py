"""Validate player-season totals and score production within each position."""
import numpy as np
import pandas as pd

STATS = ['passing_yards', 'passing_tds', 'interceptions', 'rushing_yards',
         'rushing_tds', 'receptions', 'receiving_yards', 'receiving_tds']
REQUIRED = ['player_id', 'player_name', 'season', 'team', 'position', 'games',
            'salary', 'is_demo', *STATS]

def validate_data(frame):
    missing = sorted(set(REQUIRED) - set(frame.columns))
    if missing:
        raise ValueError('Missing columns: ' + ', '.join(missing))
    df = frame.copy()
    if df.empty:
        raise ValueError('The dataset has no player rows.')
    for col in ['player_id', 'player_name', 'team', 'position']:
        if df[col].isna().any() or df[col].astype(str).str.strip().eq('').any():
            raise ValueError(f'{col} must not be blank.')
        df[col] = df[col].astype(str).str.strip()
    if not df.position.isin(['QB', 'RB', 'WR', 'TE']).all():
        raise ValueError('Supported positions: QB, RB, WR, TE.')
    for col in ['season', 'games', 'salary', *STATS]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        if not np.isfinite(df[col]).all() or (df[col] < 0).any():
            raise ValueError(f'{col} must contain finite, nonnegative numbers.')
    if (df.salary <= 0).any():
        raise ValueError('salary must be positive annual dollars, not millions.')
    for col in ['season', 'games', *STATS]:
        if (df[col] % 1 != 0).any():
            raise ValueError(f'{col} must contain whole numbers.')
    if not df.games.between(0, 17).all():
        raise ValueError('games must be between 0 and 17.')
    if (df.loc[df.games.eq(0), STATS].sum(axis=1) > 0).any():
        raise ValueError('Players with zero games cannot have production.')
    if df.duplicated(['player_id', 'season']).any():
        raise ValueError('Use one aggregate row per player_id and season.')
    flags = df.is_demo.astype(str).str.lower().map({'true': True, 'false': False})
    if flags.isna().any():
        raise ValueError('is_demo must be true or false.')
    if flags.nunique() > 1:
        raise ValueError('Do not mix demo and real players in the same dataset.')
    df['is_demo'] = flags
    df['season'] = df.season.astype(int)
    return df

def add_production(frame):
    df = validate_data(frame)
    df['production_points'] = (
        .04 * df.passing_yards + 4 * df.passing_tds - 2 * df.interceptions
        + .1 * df.rushing_yards + 6 * df.rushing_tds + df.receptions
        + .1 * df.receiving_yards + 6 * df.receiving_tds)
    groups = df.groupby(['season', 'position']).production_points
    df['production_percentile'] = 100 * (groups.rank(method='average') - .5) / groups.transform('size')
    df['points_per_million'] = df.production_points / (df.salary / 1_000_000)
    return df
