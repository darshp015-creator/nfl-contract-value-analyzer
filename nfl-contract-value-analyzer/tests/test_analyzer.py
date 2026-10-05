from pathlib import Path
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest
from src.metrics import validate_data
from src.valuation import evaluate_players
ROOT = Path(__file__).resolve().parents[1]

def test_negative_yards_are_valid_but_negative_counts_are_not():
    df = pd.read_csv(ROOT/'data/players.csv').iloc[:1].copy()
    df.loc[0, 'rushing_yards'] = -10
    assert validate_data(df).rushing_yards.iloc[0] == -10
    df.loc[0, 'games'] = 0
    with pytest.raises(ValueError, match='zero games'):
        validate_data(df)
    df.loc[0, 'games'] = 1
    df.loc[0, 'receptions'] = -1
    with pytest.raises(ValueError, match='nonnegative'):
        validate_data(df)

def test_model():
    df = pd.read_csv(ROOT/'data/players.csv')
    result = evaluate_players(df)
    assert len(result) == 96
    assert result.peer_count.eq(11).all()
    assert result.benchmark_salary.notna().all()
    df.loc[0, 'salary'] *= 10
    assert evaluate_players(df).loc[0, 'benchmark_salary'] == pytest.approx(result.loc[0, 'benchmark_salary'])
    assert evaluate_players(df.iloc[:1]).benchmark_salary.isna().all()
    df.loc[0, 'salary'] = 0
    with pytest.raises(ValueError):
        validate_data(df)

def test_dashboard():
    app = AppTest.from_file(str(ROOT/'app.py'),default_timeout=60).run()
    assert not app.exception
    assert app.metric[0].value == '555'
    assert len(app.tabs) == 6
    assert not any('DEMO' in w.value for w in app.warning)
    app.multiselect(key='positions').set_value(['QB']).run()
    assert not app.exception
    assert app.metric[0].value == '78'
    app.selectbox(key='contract_filter').set_value('Rookie deal').run()
    assert app.dataframe[0].value.contract_group.eq('Rookie deal').all()
    app.selectbox(key='cost').set_value('Season cap hit').run()
    assert not app.exception
    app.selectbox(key='compare_position').set_value('WR').run()
    assert not app.exception
    assert app.selectbox(key='compare_a').value != app.selectbox(key='compare_b').value
    app.text_input(key='search').set_value('no such player').run()
    assert not app.exception
    assert any('No players match' in i.value for i in app.info)
    app.radio(key='source').set_value('Demo').run()
    assert not app.exception
    assert app.selectbox(key='scoring').value == 'Box-score proxy'
    assert app.selectbox(key='cost').value == 'Contract APY'

def test_validation_view():
    app = AppTest.from_file(str(ROOT/'app.py'),default_timeout=60).run()
    app.button(key='run_validation').click().run()
    assert not app.exception
    assert any('Latest test season (2025)' in i.value for i in app.info)


def test_adjustment_and_cap_controls():
    app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=60).run()
    assert not app.exception
    assert app.checkbox(key='adjust_samples').value
    assert app.checkbox(key='cap_view').value
    assert any('Why this player ranks here' in h.value for h in app.subheader)
    initial=app.dataframe[0].value.set_index('player_name').production_score.sort_index()
    app.checkbox(key='adjust_samples').uncheck().run()
    raw=app.dataframe[0].value.set_index('player_name').production_score.sort_index()
    assert not initial.equals(raw)
    ratios=app.dataframe[0].value.set_index('player_name').value_ratio.sort_index()
    app.checkbox(key='cap_view').uncheck().run()
    pd.testing.assert_series_equal(ratios,app.dataframe[0].value.set_index('player_name').value_ratio.sort_index())
    assert not app.exception
