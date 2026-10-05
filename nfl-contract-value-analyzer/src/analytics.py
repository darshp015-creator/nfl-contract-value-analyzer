"""Position scoring, peer benchmarks, and chronological salary-fit checks."""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from src.metrics import add_production
from src.context import SALARY_CAPS, RATE_WORKLOAD

COSTS = {'Contract APY':'salary','Season cap hit':'cap_hit','Season cash paid':'cash_paid'}
COMPONENTS = {
    'QB': [('passing_epa_rate',.45,True),('passing_cpoe',.25,True),('turnover_rate',.15,False),('total_epa',.15,True)],
    'RB': [('rushing_epa_rate',.35,True),('scrimmage_yards',.25,True),('yards_per_carry',.20,True),('receiving_epa',.20,True)],
    'WR': [('receiving_epa_rate',.35,True),('yards_per_target',.25,True),('target_share',.25,True),('catch_rate',.15,True)],
    'TE': [('receiving_epa_rate',.35,True),('yards_per_target',.25,True),('target_share',.25,True),('catch_rate',.15,True)],
}
ADVANCED_REQUIRED = ['passing_epa','passing_cpoe','rushing_epa','receiving_epa','attempts',
                     'sacks_suffered','targets','carries','team_targets','rushing_fumbles_lost','sack_fumbles_lost']
LABELS = {'passing_epa_rate':'Passing EPA / attempt + sack','passing_cpoe':'Completion % over expectation (points)',
          'turnover_rate':'QB turnover rate','total_epa':'Passing + rushing EPA',
          'rushing_epa_rate':'Rushing EPA / carry','scrimmage_yards':'Scrimmage yards',
          'yards_per_carry':'Yards / carry','receiving_epa':'Receiving EPA',
          'receiving_epa_rate':'Receiving EPA / target','yards_per_target':'Yards / target',
          'target_share':'Share of team targets','catch_rate':'Receptions / targets'}

def deal_group(value):
    types = {s.strip() for s in str(value).split('/')}
    if types and types <= {'Drafted','UDFA'}:
        return 'Rookie deal'
    if types and types <= {'UFA','SFA','Extension','RFA','ERFA','Franchise','Transition'}:
        return 'Veteran deal'
    return 'Unknown'

def score_players(frame, scoring='Position metrics', adjust_samples=True):
    df = add_production(frame).reset_index(drop=True)
    df['contract_group'] = df.get('contract_type',pd.Series('',index=df.index)).map(deal_group)
    for col in set(ADVANCED_REQUIRED + ['cap_hit','cash_paid']):
        if col not in df:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col],errors='coerce').replace([np.inf,-np.inf],np.nan)
    def rate(numerator,denominator):
        return numerator / denominator.where(denominator>0)
    df['dropbacks'] = df.attempts + df.sacks_suffered
    df['passing_epa_rate'] = rate(df.passing_epa,df.dropbacks)
    df['turnover_rate'] = rate(df.interceptions+df.rushing_fumbles_lost+df.sack_fumbles_lost,df.dropbacks+df.carries)
    df['qb_plays'] = df.dropbacks + df.carries
    df['total_epa'] = df.passing_epa+df.rushing_epa
    df['rushing_epa_rate'] = rate(df.rushing_epa,df.carries)
    df['scrimmage_yards'] = df.rushing_yards+df.receiving_yards
    df['yards_per_carry'] = rate(df.rushing_yards,df.carries)
    df['receiving_epa_rate'] = rate(df.receiving_epa,df.targets)
    df['yards_per_target'] = rate(df.receiving_yards,df.targets)
    df['target_share'] = rate(df.targets,df.team_targets)
    df['catch_rate'] = rate(df.receptions,df.targets)
    df['opportunities'] = np.select([df.position.eq('QB'),df.position.eq('RB')],[df.dropbacks,df.carries+df.targets],default=df.targets)
    df['small_sample'] = df.opportunities < df.position.map({'QB':100,'RB':75,'WR':50,'TE':50})
    df['sample_adjusted'] = bool(adjust_samples and scoring=='Position metrics')
    df['raw_production_score'] = np.nan
    df['components_used'] = 0
    df['production_score'] = np.nan
    if scoring == 'Box-score proxy':
        df['production_score'] = df.production_percentile
        df['raw_production_score'] = df.production_score
        df['components_used'] = 1
    else:
        for (_,pos),group in df.groupby(['season','position']):
            numerator = pd.Series(0.,index=group.index)
            denominator = numerator.copy()
            raw_numerator = numerator.copy()
            eligible = group.opportunities.gt(0)
            for col,weight,higher in COMPONENTS[pos]:
                values = group[col].where(eligible)
                count = values.notna().sum()
                pct = 100 * (values.rank(ascending=higher,method='average')-.5)/max(count,1)
                raw_numerator += pct.fillna(0)*weight
                df.loc[group.index,col+'_raw_score'] = pct
                reliability = pd.Series(1.,index=group.index)
                if adjust_samples and col in RATE_WORKLOAD:
                    exposure,strength = RATE_WORKLOAD[col]
                    n = group[exposure].clip(lower=0)
                    reliability = n/(n+strength)
                    # Missing exposure cannot support an efficiency estimate.
                    pct = 50 + reliability*(pct-50)
                df.loc[group.index,col+'_reliability'] = reliability
                df.loc[group.index,col+'_score'] = pct
                numerator += pct.fillna(0)*weight
                denominator += pct.notna()*weight
                df.loc[group.index,'components_used'] += values.notna().astype(int)
            df.loc[group.index,'raw_production_score'] = raw_numerator / sum(weight*group[col].where(eligible).notna() for col,weight,_ in COMPONENTS[pos]).replace(0,np.nan)
            df.loc[group.index,'production_score'] = numerator / denominator.where(denominator>0)
    return df

def _fit_predict(train, test):
    if len(train)<8 or train.production_score.nunique()<2:
        return np.repeat(train.cost.median(),len(test)), 'Peer median'
    model = make_pipeline(StandardScaler(),Ridge(alpha=5.0))
    features = ['production_score','games']
    model.fit(train[features],np.log(train.cost))
    estimates = np.exp(model.predict(test[features]))
    return np.clip(estimates,train.cost.min(),train.cost.max()), 'Ridge: production + games'

def evaluate(frame,cost_label='Contract APY',scoring='Position metrics',same_deal=True,adjust_samples=True):
    df = score_players(frame,scoring,adjust_samples)
    df['cost'] = df[COSTS[cost_label]]
    df['league_cap'] = df.season.map(SALARY_CAPS)
    df['cost_cap_pct'] = 100*df.cost/df.league_cap
    # Keep missing/nonpositive costs visible in coverage, but never divide by them.
    df['eligible'] = df.cost.gt(0)&df.production_score.notna()
    df['benchmark_salary'] = np.nan
    df['peer_low'] = np.nan
    df['peer_high'] = np.nan
    df['peer_count'] = 0
    df['model_method'] = 'Unavailable'
    keys = ['season','position']+(['contract_group'] if same_deal else [])
    for _,group in df[df.eligible].groupby(keys):
        for idx in group.index:
            peers=group.drop(index=idx)
            df.loc[idx,'peer_count']=len(peers)
            if peers.empty:
                continue
            prediction,method=_fit_predict(peers,df.loc[[idx]])
            df.loc[idx,'benchmark_salary']=prediction[0]
            df.loc[idx,'model_method']=method
            df.loc[idx,['peer_low','peer_high']]=peers.cost.quantile([.1,.9]).to_numpy()
    df['benchmark_cap_pct']=100*df.benchmark_salary/df.league_cap
    df['surplus']=df.benchmark_salary-df.cost
    df['value_ratio']=df.benchmark_salary/df.cost.where(df.cost>0)
    df['value_label']=np.select([df.value_ratio.isna(),df.value_ratio.ge(1.2),df.value_ratio.le(.8)],
                               ['Unavailable','Below benchmark cost','Above benchmark cost'],default='Near benchmark cost')
    df['position_value_rank']=df.groupby(keys).value_ratio.rank(ascending=False,method='min').astype('Int64')
    return df

def chronological_validation(df,same_deal=True,normalize_cap=False):
    """Test complete later seasons; no test salary enters training or error bands."""
    available=df[df.eligible].copy()
    available['cost_scale'] = available.season.map(SALARY_CAPS)/100 if normalize_cap else 1.
    available = available[available.cost_scale.notna()]
    years=sorted(available.season.unique())
    if len(years)<2:
        return pd.DataFrame(),pd.DataFrame()
    keys=['position']+(['contract_group'] if same_deal else [])
    predictions=[]
    for year in years[1:]:
        for groupkey,test in available[available.season.eq(year)].groupby(keys):
            groupkey=groupkey if isinstance(groupkey,tuple) else (groupkey,)
            train=available[available.season.lt(year)]
            for key,value in zip(keys,groupkey):
                train=train[train[key].eq(value)]
            if len(train)<8:
                continue
            fitting=train.copy()
            fitting['cost']=fitting.cost/fitting.cost_scale
            pred,method=_fit_predict(fitting,test)
            pred=pred*test.cost_scale.to_numpy()
            out=test[['player_id','player_name','season','position','contract_group','cost']].copy()
            out['prediction']=pred
            out['baseline']=fitting.cost.median()*test.cost_scale.to_numpy()
            out['cost_scale']=test.cost_scale.to_numpy()
            out['cap_normalized']=normalize_cap
            out['method']=method
            out['training_rows']=len(train)
            out['training_through']=max(train.season)
            predictions.append(out)
    if not predictions:
        return pd.DataFrame(),pd.DataFrame()
    result=pd.concat(predictions,ignore_index=True)
    result['absolute_error']=(result.prediction-result.cost).abs()
    result['baseline_error']=(result.baseline-result.cost).abs()
    result['lower']=np.nan
    result['upper']=np.nan
    for idx,row in result.iterrows():
        calibration=result[result.season.lt(row.season)&result.position.eq(row.position)]
        if same_deal:
            calibration=calibration[calibration.contract_group.eq(row.contract_group)]
        if len(calibration)>=20:
            radius=(calibration.absolute_error/calibration.cost_scale).quantile(.9)*row.cost_scale
            result.loc[idx,'lower']=max(0,row.prediction-radius)
            result.loc[idx,'upper']=row.prediction+radius
    summaries=[]
    for (year,pos),group in result.groupby(['season','position']):
        with_band=group[group.lower.notna()]
        summaries.append(dict(season=int(year),position=pos,players=len(group),
                              model_mae=group.absolute_error.mean(),baseline_mae=group.baseline_error.mean(),
                              improvement_pct=100*(1-group.absolute_error.mean()/group.baseline_error.mean()) if group.baseline_error.mean()>0 else np.nan,
                              band_players=len(with_band),band_coverage=with_band.cost.between(with_band.lower,with_band.upper).mean() if len(with_band) else np.nan))
    return pd.DataFrame(summaries),result
