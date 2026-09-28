"""Descriptive peer salary benchmark; not a forecast of contract worth."""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from src.metrics import add_production

def evaluate_players(frame):
    df = add_production(frame).reset_index(drop=True)
    df['benchmark_salary'] = np.nan
    df['model_method'] = ''
    df['peer_count'] = 0
    for _, group in df.groupby(['season', 'position']):
        for idx in group.index:
            peers = group.drop(index=idx)
            df.loc[idx, 'peer_count'] = len(peers)
            if peers.empty:
                method = 'Unavailable: no position peers'
            elif len(peers) < 5 or peers.production_points.nunique() < 2:
                df.loc[idx, 'benchmark_salary'] = peers.salary.median()
                method = 'Peer median (small or constant cohort)'
            else:
                model = make_pipeline(StandardScaler(), Ridge(alpha=5.0))
                model.fit(peers[['production_points']], np.log(peers.salary))
                predicted = np.exp(model.predict(df.loc[[idx], ['production_points']])[0])
                df.loc[idx, 'benchmark_salary'] = np.clip(predicted, peers.salary.min(), peers.salary.max())
                method = 'Ridge on log salary; held-out player'
            df.loc[idx, 'model_method'] = method
    df['surplus'] = df.benchmark_salary - df.salary
    df['value_ratio'] = df.benchmark_salary / df.salary
    df['value_label'] = np.select(
        [df.value_ratio.isna(), df.value_ratio >= 1.2, df.value_ratio <= .8],
        ['Insufficient peers', 'Below benchmark salary', 'Above benchmark salary'],
        default='Near benchmark salary')
    df['position_value_rank'] = df.groupby(['season', 'position']).value_ratio.rank(
        ascending=False, method='min').astype('Int64')
    return df
