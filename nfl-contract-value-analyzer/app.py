from pathlib import Path
from io import BytesIO
import pandas as pd
import plotly.express as px
import streamlit as st
from src.metrics import STATS
from src.valuation import evaluate_players

st.set_page_config(page_title='NFL Contract Value Analyzer', page_icon='🏈', layout='wide')
ROOT = Path(__file__).resolve().parent

@st.cache_data
def analyze(contents):
    return evaluate_players(pd.read_csv(BytesIO(contents)))

st.title('NFL Contract Value Analyzer')
st.caption('Explore production, annual salary, and value within each position.')
with st.sidebar:
    st.header('Build your view')
    upload = st.file_uploader('Optional: load your own CSV', type='csv')
    st.caption('Schema and salary definitions are in the README. Empty selections show all.')
try:
    data = analyze(upload.getvalue() if upload else (ROOT / 'data/players.csv').read_bytes())
except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
    st.error(f'Could not load this dataset: {exc}')
    st.stop()
if data.is_demo.any():
    st.warning('DEMO DATA — fictional players, salaries, and statistics. These are not real NFL contract assessments.')
else:
    st.info('User-provided data. Accuracy and provenance have not been verified.')
with st.sidebar:
    season = st.selectbox('Season', sorted(data.season.unique(), reverse=True))
    season_data = data[data.season.eq(season)]
    positions = st.multiselect('Positions', sorted(season_data.position.unique()))
    teams = st.multiselect('Teams', sorted(season_data.team.unique()))
    min_games = st.slider('Minimum games', 0, 17, 0)
    search = st.text_input('Search player', placeholder='Enter a name')
    sort_by = st.selectbox('Rank by', ['Value ratio', 'Salary surplus', 'Production percentile'])
view = season_data.copy()
if positions:
    view = view[view.position.isin(positions)]
if teams:
    view = view[view.team.isin(teams)]
view = view[view.games.ge(min_games) & view.player_name.str.contains(search.strip(), case=False, regex=False)]
view = view.sort_values({'Value ratio': 'value_ratio', 'Salary surplus': 'surplus',
                         'Production percentile': 'production_percentile'}[sort_by], ascending=False, na_position='last')
st.caption('Benchmarks use the full uploaded position/season cohort before display filters. Salary = annual average contract value (APY), in USD.')
if view.empty:
    st.info('No players match these filters. Clear the search or broaden your selections.')
    st.stop()
a, b, c, d = st.columns(4)
a.metric('Players in view', len(view))
b.metric('Total annual salary', f'${view.salary.sum()/1e6:,.1f}M')
c.metric('Median value ratio', f'{view.value_ratio.median():.2f}×' if view.value_ratio.notna().any() else 'N/A')
d.metric('Median production percentile', f'{view.production_percentile.median():.0f}/100')
rankings, player, method = st.tabs(['League value rankings', 'Player detail', 'How it works'])
with rankings:
    st.subheader('Salary vs. production')
    chart = px.scatter(view, x='salary', y='production_percentile', color='position',
                       hover_name='player_name', size='games', size_max=24,
                       hover_data={'team': True, 'salary': ':$,.0f', 'production_percentile': ':.1f',
                                   'value_ratio': ':.2f', 'games': True},
                       labels={'salary': 'Annual salary (USD)', 'production_percentile': 'Production percentile within position', 'position': 'Position'},
                       color_discrete_sequence=['#22c55e', '#38bdf8', '#fbbf24', '#c084fc'])
    chart.update_traces(marker={'sizemin': 5})
    chart.update_layout(height=420, legend_title_text='Position', margin=dict(l=10,r=10,t=15,b=10))
    chart.update_yaxes(range=[0, 100])
    chart.update_xaxes(tickprefix='$', tickformat='~s')
    st.plotly_chart(chart, use_container_width=True)
    st.subheader('League value rankings')
    st.caption('Value ratio = peer salary benchmark ÷ annual salary. Higher means cheaper relative to the model. Position rank compares only the same position.')
    cols = ['player_name','team','position','games','salary','production_percentile',
            'benchmark_salary','surplus','value_ratio','position_value_rank','value_label']
    st.dataframe(view[cols], hide_index=True, width='stretch', column_config={
        'salary': st.column_config.NumberColumn('Annual salary', format='$%d'),
        'benchmark_salary': st.column_config.NumberColumn('Peer benchmark', format='$%d'),
        'surplus': st.column_config.NumberColumn('Salary surplus', format='$%d'),
        'value_ratio': st.column_config.NumberColumn('Value ratio', format='%.2f×'),
        'production_percentile': st.column_config.ProgressColumn('Production percentile', min_value=0,max_value=100,format='%.1f'),
        'position_value_rank': 'Position value rank', 'player_name': 'Player', 'value_label': 'Model assessment'})
    st.download_button('Download filtered rankings', view.to_csv(index=False).encode(),
                       file_name=f'nfl-value-{season}.csv', mime='text/csv')
with player:
    options = view.sort_values('player_name').player_id.tolist()
    selected = st.selectbox('Choose a player', options, format_func=lambda pid: str(view.loc[view.player_id.eq(pid), 'player_name'].iloc[0]))
    row = view[view.player_id.eq(selected)].iloc[0]
    st.subheader(f'{row.player_name} · {row.position} · {row.team}')
    st.caption(f'{season} regular season • {int(row.games)} games • {int(row.peer_count)} position peers')
    x,y,z = st.columns(3)
    x.metric('Annual salary', f'${row.salary/1e6:.2f}M')
    y.metric('Peer salary benchmark', f'${row.benchmark_salary/1e6:.2f}M' if pd.notna(row.benchmark_salary) else 'N/A')
    z.metric('Salary surplus / deficit', f'${row.surplus/1e6:+.2f}M' if pd.notna(row.surplus) else 'N/A')
    st.write(f'**{row.value_label}**')
    st.write(f'Production: **{row.production_points:.1f} points** · Position percentile: **{row.production_percentile:.1f}** · Points per $1M: **{row.points_per_million:.1f}**')
    st.dataframe(pd.DataFrame({'Statistic': [s.replace('_',' ').title() for s in STATS], 'Total': [int(row[s]) for s in STATS]}), hide_index=True, width='stretch')
    st.caption(f'Benchmark method: {row.model_method}. This is a descriptive comparison, not a forecast or definitive fair contract value.')
with method:
    st.markdown('''### Production score
A PPR-style proxy: passing yards × 0.04 + passing TDs × 4 − interceptions × 2
+ rushing yards × 0.1 + rushing TDs × 6 + receptions + receiving yards × 0.1 + receiving TDs × 6.
Season totals retain missed-game impact. Percentiles compare the same position and season; ties share a midrank percentile.

### Salary benchmark
For each player, scikit-learn Ridge regression predicts log annual salary from production points using
other players at the same position in the same season. The player’s own salary is excluded.
Predictions are limited to the peer salary range. Fewer than five peers (or constant production)
uses the peer median; no peers means no estimate. Value ratio = benchmark / actual salary;
surplus = benchmark − salary. Ratios ≥ 1.2 are below benchmark salary; ≤ 0.8 are above it.

### Limits
This MVP covers QB, RB, WR, and TE only. Counting stats omit blocking, defense, scheme,
contract timing, age, guarantees, and future performance. Rookie deals can look unusually efficient.
APY differs from a season’s cap hit or cash paid. Models describe associations, not causal value.
Demo results have no real-world valuation meaning. For real analysis, join verified season stats and
contract data by stable player ID and season, validate coverage, and evaluate on held-out seasons.''')
