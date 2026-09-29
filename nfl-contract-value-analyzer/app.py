from io import BytesIO
from pathlib import Path
import json
import pandas as pd
import plotly.express as px
import streamlit as st
from src.metrics import STATS
from src.analytics import ADVANCED_REQUIRED, COMPONENTS, COSTS, LABELS, evaluate, chronological_validation

st.set_page_config(page_title='NFL Contract Value Analyzer',page_icon='🏈',layout='wide')
ROOT=Path(__file__).resolve().parent

@st.cache_data
def analyze(contents,cost_label,scoring,same_deal):
    return evaluate(pd.read_csv(BytesIO(contents)),cost_label,scoring,same_deal)

@st.cache_data
def validate_history(data,same_deal):
    return chronological_validation(data,same_deal)

def money(value):
    return f'${value/1e6:,.2f}M' if pd.notna(value) else 'Unavailable'

def md_money(value):
    return money(value).replace('$',r'\$')

def numeric(value,suffix='',decimals=2):
    return f'{value:,.{decimals}f}{suffix}' if pd.notna(value) else 'Unavailable'

def player_label(frame,pid):
    row=frame[frame.player_id.eq(pid)].iloc[0]
    return f'{row.player_name} · {row.team}'

st.title('NFL Contract Value Analyzer')
st.caption('Compare production and contract costs within each position. Explore 2022–2025.')
with st.sidebar:
    st.header('Build your view')
    source=st.radio('Data source',['NFL 2022–2025','Upload CSV','Demo'],key='source')
    upload=st.file_uploader('Load your own CSV',type='csv') if source=='Upload CSV' else None
    if source=='Upload CSV' and upload is None:
        st.info('Choose a CSV to begin. Existing app-format CSVs still work.')
        st.stop()
try:
    contents=upload.getvalue() if upload else (ROOT/'data'/('players.csv' if source=='Demo' else 'nfl_2022_2025.csv')).read_bytes()
    raw=pd.read_csv(BytesIO(contents))
except (OSError,ValueError,UnicodeDecodeError) as exc:
    st.error(f'Could not read the dataset: {exc}')
    st.stop()
with st.sidebar:
    available_costs=[label for label,col in COSTS.items() if col in raw and pd.to_numeric(raw[col],errors='coerce').gt(0).any()]
    cost_label=st.selectbox('Cost measure',available_costs or ['Contract APY'],key='cost')
    advanced=all(col in raw for col in ADVANCED_REQUIRED)
    scoring=st.selectbox('Production scoring',['Position metrics','Box-score proxy'] if advanced else ['Box-score proxy'],key='scoring')
    if not advanced:
        st.caption('Advanced fields are absent. This file supports the box-score proxy only.')
    same_deal=st.checkbox('Benchmark within contract group',value=True,key='same_deal',help='Compare rookie deals with rookie deals and veteran deals with veteran deals. Unknown contracts form a separate group.')
try:
    with st.spinner('Calculating position benchmarks…'):
        data=analyze(contents,cost_label,scoring,same_deal)
except (ValueError,KeyError,TypeError) as exc:
    st.error(f'Could not analyze this dataset: {exc}')
    st.stop()
if data.is_demo.any():
    st.warning('DEMO DATA — fictional players, salaries, and statistics.')
elif source=='NFL 2022–2025':
    snapshot_date=json.loads((ROOT/'data/sources.json').read_text())['retrieved_date']
    st.info(f'Real regular-season data · nflverse + Over The Cap · snapshot retrieved {snapshot_date}. Excludes playoffs and unresolved contract matches.')
else:
    st.info('Uploaded data. Its accuracy and provenance have not been verified by the app.')
with st.sidebar:
    season=st.selectbox('Season',sorted(data.season.unique(),reverse=True),key='season')
    season_data=data[data.season.eq(season)]
    positions=st.multiselect('Positions',sorted(season_data.position.unique()),key='positions')
    teams=st.multiselect('Teams',sorted(season_data.team.unique()),key='teams')
    contract_filter=st.selectbox('Contract group',['All','Rookie deal','Veteran deal','Unknown'],key='contract_filter')
    min_games=st.slider('Minimum games',0,17,0,key='min_games')
    min_opps=st.slider('Minimum opportunities',0,500,0,step=10,key='min_opps',disabled=not advanced)
    search=st.text_input('Search player',key='search')
    sort_by=st.selectbox('Rank by',['Value ratio','Cost surplus','Production score'],key='sort')
    st.caption('Empty position/team selections include all. Opportunities: QB attempts + sacks; RB carries + targets; WR/TE targets.')
view=season_data.copy()
if positions:
    view=view[view.position.isin(positions)]
if teams:
    view=view[view.team.isin(teams)]
if contract_filter!='All':
    view=view[view.contract_group.eq(contract_filter)]
view=view[view.games.ge(min_games)&view.player_name.str.contains(search.strip(),case=False,regex=False)]
if min_opps and advanced:
    view=view[view.opportunities.ge(min_opps)]
view=view.sort_values({'Value ratio':'value_ratio','Cost surplus':'surplus','Production score':'production_score'}[sort_by],ascending=False,na_position='last')
st.caption(f'{cost_label} in nominal USD. Benchmarks use all eligible players in the same season and position'+(' and contract group.' if same_deal else '.')+' Display filters do not refit the model.')
a,b,c,d=st.columns(4)
a.metric('Players in view',len(view))
b.metric('Total selected cost',money(view.loc[view.cost.gt(0),'cost'].sum()) if view.cost.gt(0).any() else 'Unavailable')
c.metric('Median value ratio',numeric(view.value_ratio.median(),'×'))
d.metric('Players with a benchmark',int(view.benchmark_salary.notna().sum()))
rankings,details,compare,trends,validation,sources=st.tabs(['Rankings','Player detail','Compare players','Season trends','Model validation','Data & methods'])
with rankings:
    if view.empty:
        st.info('No players match these filters. Clear the search or broaden your selections.')
    else:
        missing=int((~view.eligible).sum())
        if missing:
            st.caption(f'{missing} displayed players lack a positive selected cost or a usable score. Their valuations remain unavailable.')
        st.subheader('Cost vs. production')
        plotted=view[view.eligible]
        if not plotted.empty:
            fig=px.scatter(plotted,x='cost',y='production_score',color='position',symbol='contract_group',size='games',size_max=22,
                           hover_name='player_name',hover_data={'team':True,'cost':':$,.0f','value_ratio':':.2f','small_sample':True},
                           labels={'cost':cost_label+' (USD)','production_score':'Production score (0–100)','contract_group':'Contract'},
                           color_discrete_sequence=['#22c55e','#38bdf8','#fbbf24','#c084fc'])
            fig.update_traces(marker={'sizemin':5})
            fig.update_layout(height=430,margin=dict(l=10,r=10,t=10,b=10))
            fig.update_yaxes(range=[0,100]);fig.update_xaxes(tickprefix='$',tickformat='~s')
            st.plotly_chart(fig,use_container_width=True)
        st.caption('Higher value ratio means lower cost relative to this model, not proven football value. Small-sample flags use QB <100, RB <75, WR/TE <50 opportunities.')
        columns=['player_name','team','position','contract_group','games','cost','production_score','benchmark_salary','surplus','value_ratio','small_sample','value_label']
        st.dataframe(view[columns],hide_index=True,width='stretch',column_config={
            'player_name':'Player','contract_group':'Contract','cost':st.column_config.NumberColumn(cost_label,format='$%d'),
            'production_score':st.column_config.ProgressColumn('Production score',min_value=0,max_value=100,format='%.1f'),
            'benchmark_salary':st.column_config.NumberColumn('Peer cost benchmark',format='$%d'),
            'surplus':st.column_config.NumberColumn('Cost surplus',format='$%d'),
            'value_ratio':st.column_config.NumberColumn('Value ratio',format='%.2f×')})
        st.download_button('Download filtered rankings',view.to_csv(index=False).encode(),f'nfl-rankings-{season}.csv','text/csv')
with details:
    if view.empty:
        st.info('Broaden the filters to see player details.')
    else:
        selected=st.selectbox('Choose a player',view.sort_values('player_name').player_id.tolist(),format_func=lambda pid:player_label(view,pid),key='detail_player')
        row=view[view.player_id.eq(selected)].iloc[0]
        st.subheader(f'{row.player_name} · {row.position} · {row.team}')
        st.caption(f'{season} · {row.contract_group} · {int(row.games)} games · {int(row.peer_count)} eligible peers')
        x,y,z=st.columns(3)
        x.metric(cost_label,money(row.cost)); y.metric('Peer cost benchmark',money(row.benchmark_salary));z.metric('Value ratio',numeric(row.value_ratio,'×'))
        st.write(f'**{row.value_label}** · Production score: **{numeric(row.production_score,decimals=1)} / 100**')
        if row.small_sample:
            st.warning('Small workload: efficiency rates can be unstable. Check the raw stats and opportunity count.')
        st.caption(f'Peer cost range (10th–90th percentile): {md_money(row.peer_low)} to {md_money(row.peer_high)}. This shows peer spread, not a confidence interval.')
        if scoring=='Position metrics':
            st.dataframe(pd.DataFrame({'Metric':[LABELS[col] for col,_,_ in COMPONENTS[row.position]],'Value':[row[col] for col,_,_ in COMPONENTS[row.position]],'Weight':[weight for _,weight,_ in COMPONENTS[row.position]]}),hide_index=True,width='stretch')
            st.caption(f'{int(row.components_used)} of 4 components available. Missing components are omitted and available weights renormalized; no production is invented.')
        st.dataframe(pd.DataFrame({'Statistic':[s.replace('_',' ').title() for s in STATS],'Total':[int(row[s]) for s in STATS]}),hide_index=True,width='stretch')
        st.caption('Benchmark method: '+row.model_method)
with compare:
    st.subheader('Compare two players at the same position')
    st.caption('Uses the full selected-season dataset, independent of rankings filters. Cost measure and scoring above still apply.')
    cp=st.selectbox('Comparison position',sorted(season_data.position.unique()),key='compare_position')
    pool=season_data[season_data.position.eq(cp)].sort_values('player_name')
    if len(pool)<2:
        st.info('At least two players at this position are needed.')
    else:
        left,right=st.columns(2)
        p1=left.selectbox('Player A',pool.player_id.tolist(),format_func=lambda pid:player_label(pool,pid),key='compare_a')
        p2=right.selectbox('Player B',[pid for pid in pool.player_id if pid!=p1],format_func=lambda pid:player_label(pool,pid),key='compare_b')
        pair=pool.set_index('player_id').loc[[p1,p2]]
        for col,(_,row) in zip([left,right],pair.iterrows()):
            col.metric(cost_label,money(row.cost));col.metric('Production score',numeric(row.production_score,decimals=1));col.metric('Value ratio',numeric(row.value_ratio,'×'))
            col.caption(f'{row.contract_group} · {int(row.games)} games · {numeric(row.opportunities,decimals=0)} opportunities')
        metric_cols=['games','opportunities','production_score','benchmark_salary','surplus',*STATS]
        if scoring=='Position metrics':
            metric_cols += [col for col,_,_ in COMPONENTS[cp]]
        table=pair[metric_cols].T
        table.columns=[f'{r.player_name} ({r.team})' for _,r in pair.iterrows()]
        table.index=[LABELS.get(col,col.replace('_',' ').title()) for col in metric_cols]
        st.dataframe(table,width='stretch')
        st.caption('Differences are descriptive. Score is position-relative; the model does not establish which player causes more wins.')
with trends:
    st.subheader('Player history')
    choices=data.sort_values(['player_name','season']).drop_duplicates('player_id')
    name_map=choices.set_index('player_id').player_name.to_dict()
    pid=st.selectbox('Player history',choices.player_id.tolist(),format_func=lambda p:name_map[p],key='trend_player')
    history=data[data.player_id.eq(pid)].sort_values('season')
    if len(history)<2:
        st.info('Only one matched season is available for this player. Missing seasons are not filled in.')
    else:
        l,r=st.columns(2)
        l.plotly_chart(px.line(history,x='season',y='cost',markers=True,labels={'cost':cost_label+' (USD)','season':'Season'}).update_xaxes(dtick=1),use_container_width=True)
        r.plotly_chart(px.line(history,x='season',y='production_score',markers=True,labels={'production_score':'Position-relative score','season':'Season'}).update_xaxes(dtick=1).update_yaxes(range=[0,100]),use_container_width=True)
    st.dataframe(history[['season','team','position','contract_group','games','salary','cap_hit','cash_paid','production_score','value_ratio']],hide_index=True,width='stretch')
    st.caption('Matched seasons only. Costs are nominal dollars; APY includes extensions signed by that season. Scores are relative to each season’s position cohort.')
with validation:
    st.subheader('Does the model beat a simple baseline?')
    st.write('Train on earlier seasons and test on the next complete season. The baseline is the median cost in the same training cohort. Lower mean absolute error (MAE) is better.')
    st.caption('This evaluates retrospective salary fit using test-season production—not a preseason forecast. Test salaries never enter training. Same players may appear in different seasons. Nominal costs are not inflation-adjusted.')
    if st.button('Run historical test',key='run_validation'):
        with st.spinner('Testing later seasons against earlier data…'):
            summary,predictions=validate_history(data,same_deal)
        if summary.empty:
            st.info('Need at least two seasons and eight eligible earlier-season records per cohort.')
        else:
            st.dataframe(summary,hide_index=True,width='stretch',column_config={
                'model_mae':st.column_config.NumberColumn('Model MAE',format='$%d'),
                'baseline_mae':st.column_config.NumberColumn('Median baseline MAE',format='$%d'),
                'improvement_pct':st.column_config.NumberColumn('Improvement (%)',format='%.1f'),
                'band_coverage':st.column_config.NumberColumn('Observed band coverage (0–1)',format='%.2f')})
            latest=predictions[predictions.season.eq(predictions.season.max())]
            wins=latest.absolute_error.mean()<latest.baseline_error.mean()
            st.info(f'Latest test season ({int(latest.season.max())}): model MAE {md_money(latest.absolute_error.mean())}; baseline MAE {md_money(latest.baseline_error.mean())}. '+('The model improves on this baseline.' if wins else 'The model does not beat this baseline. Treat its valuations cautiously.'))
            st.caption('Error bands use the 90th percentile of absolute prediction errors from strictly earlier validation seasons in the same cohort (minimum 20). Coverage is measured, not guaranteed. Early seasons have no band.')
            st.download_button('Download test predictions and error bands',predictions.to_csv(index=False).encode(),'historical-validation.csv','text/csv')
with sources:
    st.subheader('Data coverage and provenance')
    if source=='NFL 2022–2025':
        meta=json.loads((ROOT/'data/sources.json').read_text())
        st.write('Retrieved **'+meta['retrieved_date']+'** · '+meta['attribution'])
        st.dataframe(pd.DataFrame(meta['coverage']),hide_index=True,width='stretch')
        st.write(meta['contract_definition']);st.write(meta['cohort_definition'])
        st.markdown('[nflverse stats releases](https://github.com/nflverse/nflverse-data/releases/tag/stats_player) · [Contract data](https://github.com/nflverse/nflverse-data/releases/tag/contracts) · [Contract definitions](https://nflreadr.nflverse.com/articles/dictionary_contracts.html)')
        st.download_button('Download bundled source data',contents,'nfl_2022_2025.csv','text/csv')
        st.download_button('Download excluded players',(ROOT/'data/exclusions.csv').read_bytes(),'excluded-players.csv','text/csv')
    else:
        st.write('CSV metadata is supplied by the uploader. Older CSVs may lack contract types, advanced metrics, cap hits, or cash paid. Missing optional fields remain unavailable.')
    st.subheader('Cost definitions')
    st.markdown('**APY:** average annual contract value. **Cap hit:** that season’s reported cap charge. **Cash paid:** that season’s reported cash. Bonus timing and restructures make these different. Missing/nonpositive cost rows are excluded from that measure’s model; they are not replaced with APY.')
    st.subheader('Position-specific scoring')
    for pos,components in COMPONENTS.items():
        st.write(f'**{pos}:** '+', '.join(f'{LABELS[c]} {w:.0%}'+(' (lower is better)' if not high else '') for c,w,high in components))
    st.caption('Each metric becomes a midrank percentile within position and season, then weights are combined. These are transparent starter weights, not learned or validated measures of wins. Missing components renormalize the remaining weights. Zero-opportunity players have no advanced score.')
    st.markdown('EPA means expected points added. Passing EPA rate divides by attempts + sacks and excludes scramble opportunities. Target share divides player targets by all team targets in that regular season. **Routes are not in this dataset, so yards per target is shown; it is not yards per route run.** CPOE measures completion percentage above expectation.')
    st.subheader('Contract groups and model limits')
    st.write('Rookie deal means the matched contract is Drafted or UDFA. Veteran deal means a free-agent deal, extension, tender, or tag. Other/absent types are Unknown. This labels the contract, not whether the player is in their first NFL season.')
    st.write('Peer benchmarks use StandardScaler + Ridge(alpha=5) on log cost, with score and games as inputs. Every player is excluded from their own fit. Fewer than eight peers or constant scores uses the peer median. Predictions are clamped to observed peer costs. Display filters do not change these cohorts.')
    st.write('The box-score option retains the original PPR-style counting proxy. Neither score captures blocking, protection, defense, scheme, age, guarantees, injury context, or future performance. Historical matching uses year-level records retrieved later, so it is not an exact contract-effective-date audit.')
