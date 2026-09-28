from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.analytics import score_players,evaluate,chronological_validation,deal_group

ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture(scope='module')
def real():
    return pd.read_csv(ROOT/'data/nfl_2022_2025.csv')

def test_snapshot_and_groups(real):
    assert len(real)==2166
    assert real[real.season.eq(2025)].shape[0]==555
    assert not real.duplicated(['player_id','season']).any()
    assert deal_group('Drafted')=='Rookie deal'
    assert deal_group('UDFA')=='Rookie deal'
    assert deal_group('Extension')=='Veteran deal'
    assert deal_group('SFA / UFA')=='Veteran deal'
    assert deal_group('Other')=='Unknown'

def test_score_direction_missingness_and_zero_workload(real):
    d=real[real.position.eq('WR')&real.season.eq(2025)].head(3).copy()
    d['targets']=[100,100,0];d['receiving_epa']=[50,-50,0]
    d['receiving_yards']=[1200,400,0];d['receptions']=[80,40,0]
    d['team_targets']=500
    scored=score_players(d)
    assert scored.production_score.iloc[0]>scored.production_score.iloc[1]
    assert pd.isna(scored.production_score.iloc[2])
    assert scored.components_used.iloc[2]==0
    assert scored.production_score.dropna().between(0,100).all()

def test_own_salary_exclusion_and_contract_isolation(real):
    d=real[real.position.eq('QB')&real.season.eq(2025)].copy().reset_index(drop=True)
    before=evaluate(d)
    idx=before[before.eligible&before.contract_group.eq('Veteran deal')].index[0]
    d.loc[idx,'salary']*=5
    after=evaluate(d)
    assert before.loc[idx,'benchmark_salary']==pytest.approx(after.loc[idx,'benchmark_salary'])
    # A rookie cost cannot influence veteran benchmarks when cohorts are separated.
    d=real[real.position.eq('QB')&real.season.eq(2025)].copy().reset_index(drop=True)
    d.loc[before.contract_group.eq('Rookie deal'),'salary']*=10
    after=evaluate(d)
    veterans=before.contract_group.eq('Veteran deal')
    np.testing.assert_allclose(before.loc[veterans,'benchmark_salary'],after.loc[veterans,'benchmark_salary'],equal_nan=True)

def test_cost_switch_and_missing_cost(real):
    d=real[real.season.eq(2025)&real.position.eq('RB')].head(12).copy().reset_index(drop=True)
    d.loc[0,'cap_hit']=np.nan
    result=evaluate(d,'Season cap hit')
    assert pd.isna(result.loc[0,'benchmark_salary'])
    assert not result.loc[0,'eligible']
    np.testing.assert_allclose(result.cost,d.cap_hit,equal_nan=True)
    result=evaluate(d,'Season cash paid')
    np.testing.assert_allclose(result.cost,d.cash_paid)

def test_legacy_upload_remains_supported():
    d=pd.read_csv(ROOT/'data/players.csv')
    result=evaluate(d,scoring='Box-score proxy')
    assert result.benchmark_salary.notna().all()
    assert result.contract_group.eq('Unknown').all()
    assert result.cap_hit.isna().all()

def test_time_split_has_no_test_salary_leakage(real):
    d=score_players(real[real.position.eq('QB')].copy())
    d['cost']=d.salary; d['eligible']=d.cost.gt(0)&d.production_score.notna()
    summary,pred=chronological_validation(d)
    assert pred.training_through.lt(pred.season).all()
    assert set(summary.season)=={2023,2024,2025}
    assert pred[pred.season.eq(2023)].lower.isna().all()
    d.loc[d.season.eq(2025),'cost']*=2
    _,changed=chronological_validation(d)
    latest=pred.season.eq(2025)
    for col in ['prediction','baseline','lower','upper']:
        np.testing.assert_allclose(pred.loc[latest,col],changed.loc[latest,col],equal_nan=True)
    assert pred.loc[latest,'absolute_error'].mean()!=changed.loc[latest,'absolute_error'].mean()
