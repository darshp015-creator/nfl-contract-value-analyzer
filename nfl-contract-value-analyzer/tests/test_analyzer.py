from pathlib import Path
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest
from src.metrics import validate_data
from src.valuation import evaluate_players
ROOT = Path(__file__).resolve().parents[1]

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
    app = AppTest.from_file(str(ROOT/'app.py')).run(timeout=30)
    assert not app.exception
    assert app.metric[0].value == '48'
    assert len(app.warning) == 1
    assert len(app.get('plotly_chart')) == 1
    app.sidebar.multiselect[0].set_value(['QB']).run()
    assert not app.exception
    assert app.metric[0].value == '12'
    next(w for w in app.selectbox if w.label == 'Choose a player').set_value('demo-QB-05').run()
    assert any('Demo QB 05' in s.value for s in app.subheader)
    app.sidebar.text_input[0].set_value('no such player').run()
    assert not app.exception
    assert 'No players match' in app.info[0].value
