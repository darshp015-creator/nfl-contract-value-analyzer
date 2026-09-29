"""Build a sourced 2022–2025 snapshot. Run with --cache pointing to downloaded files."""
import argparse
import json
import hashlib
from datetime import date
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

BASE = 'https://github.com/nflverse/nflverse-data/releases/download'
NAMES = ['Cardinals','Falcons','Ravens','Bills','Panthers','Bears','Bengals','Browns','Cowboys','Broncos','Lions','Packers','Texans','Colts','Jaguars','Chiefs','Raiders','Chargers','Rams','Dolphins','Vikings','Patriots','Saints','Giants','Jets','Eagles','Steelers','49ers','Seahawks','Buccaneers','Titans','Commanders']
ABBR = ['ARI','ATL','BAL','BUF','CAR','CHI','CIN','CLE','DAL','DEN','DET','GB','HOU','IND','JAX','KC','LV','LAC','LA','MIA','MIN','NE','NO','NYG','NYJ','PHI','PIT','SF','SEA','TB','TEN','WAS']
TEAMS = dict(zip(NAMES, ABBR))
CORE = ['passing_yards','passing_tds','passing_interceptions','rushing_yards','rushing_tds','receptions','receiving_yards','receiving_tds']
EXTRA = ['passing_epa','passing_cpoe','rushing_epa','receiving_epa','attempts','sacks_suffered','targets','carries','completions','rushing_fumbles_lost','sack_fumbles_lost','receiving_fumbles_lost']

def contract_group(types):
    if types and types <= {'Drafted','UDFA'}:
        return 'Rookie deal'
    if types and types <= {'UFA','SFA','Extension','RFA','ERFA','Franchise','Transition'}:
        return 'Veteran deal'
    return 'Unknown'

def build(cache, output, refresh=False):
    cache.mkdir(parents=True, exist_ok=True)
    source_files = []
    def fetch(path, filename):
        target = cache / filename
        if refresh or not target.exists():
            temporary = target.with_suffix(target.suffix + '.download')
            urlretrieve(f'{BASE}/{path}', temporary)
            temporary.replace(target)
        source_files.append(dict(url=f'{BASE}/{path}', filename=filename,
                                 retrieved_date=date.fromtimestamp(target.stat().st_mtime).isoformat(),
                                 sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
        return target
    contracts = pd.read_parquet(fetch('contracts/historical_contracts.parquet','contracts.parquet'))
    by_id = {pid:g for pid,g in contracts.groupby('gsis_id')}
    rows, excluded, coverage = [], [], []
    for year in range(2022,2026):
        stats = pd.read_csv(fetch(f'stats_player/stats_player_reg_{year}.csv',f'stats_player_reg_{year}.csv'))
        weekly = pd.read_csv(fetch(f'stats_player/stats_player_week_{year}.csv',f'stats_player_week_{year}.csv'),low_memory=False)
        weekly = weekly[weekly.season_type.eq('REG') & weekly.season.eq(year)]
        tc = 'team' if 'team' in weekly else 'recent_team'
        team_counts = weekly.groupby('player_id')[tc].nunique()
        team_targets = weekly.groupby(tc).targets.sum()
        totals = weekly.groupby('player_id')[CORE].sum()
        stats = stats[stats.position.isin(['QB','RB','WR','TE']) & stats.season_type.eq('REG') & stats.season.eq(year)]
        start = len(rows)
        for _, p in stats.iterrows():
            def omit(reason):
                excluded.append(dict(season=year,player_id=p.player_id,player_name=p.player_display_name,position=p.position,reason=reason))
            if team_counts.get(p.player_id,0) != 1:
                omit('Multiple teams or missing weekly team evidence'); continue
            cc = by_id.get(p.player_id)
            if cc is None:
                omit('No GSIS contract match'); continue
            h = pd.DataFrame([r for a in cc.contract_history for r in a]).drop_duplicates()
            h = h[(h.year_signed<=year)&(h.apy>0)&~h.contract_type.isin(['Practice','Reserve'])]
            if h.empty:
                omit('No eligible contract signed by season'); continue
            h = h[h.year_signed.eq(h.year_signed.max())]
            h = h[h.team.map(TEAMS).eq(p.recent_team)]
            if h.empty:
                omit('Latest contract team mismatch'); continue
            if h.apy.nunique()>1:
                h = h[h.amount_earned>0]
            if h.empty or h.apy.nunique()!=1:
                omit('Ambiguous same-year APY'); continue
            sh = pd.DataFrame([r for a in cc.season_history for r in a
                               if str(r['year'])==str(year) and TEAMS.get(r['team'])==p.recent_team]).drop_duplicates()
            if sh.empty or not sh.cash_paid.gt(0).any():
                omit('No positive season cash evidence'); continue
            contract = h.iloc[0]
            types = set(h.contract_type)
            row = dict(player_id=p.player_id,player_name=p.player_display_name,season=year,
                       team=p.recent_team,position=p.position,games=int(p.games),
                       salary=round(float(contract.apy)*1e6),is_demo=False,
                       contract_type=' / '.join(sorted(types)),contract_group=contract_group(types),
                       contract_year_signed=int(contract.year_signed))
            for col in CORE:
                assert pd.notna(p[col]) and p[col] == totals.loc[p.player_id,col], (year,p.player_id,col)
                row['interceptions' if col=='passing_interceptions' else col] = int(p[col])
            for col in EXTRA:
                row[col] = p[col] if pd.notna(p[col]) else None
            row['team_targets'] = int(team_targets[p.recent_team])
            for source, dest in [('cap_number','cap_hit'),('cash_paid','cash_paid')]:
                values = sh[source].dropna().unique()
                # Distinct source values are ambiguous; never sum repeated player histories.
                row[dest] = round(float(values[0])*1e6) if len(values)==1 else None
            row.update(contract_source=cc.iloc[0].player_page,
                       stats_source=f'{BASE}/stats_player/stats_player_reg_{year}.csv',
                       retrieved_date=max(item['retrieved_date'] for item in source_files))
            rows.append(row)
        coverage.append(dict(season=year,source_players=len(stats),included=len(rows)-start,excluded=len(stats)-(len(rows)-start)))
    df = pd.DataFrame(rows).sort_values(['season','position','player_name'])
    assert not df.duplicated(['player_id','season']).any()
    assert df.games.between(1,17).all()
    output.mkdir(parents=True,exist_ok=True)
    df.to_csv(output/'nfl_2022_2025.csv',index=False)
    pd.DataFrame(excluded).to_csv(output/'exclusions.csv',index=False)
    metadata = dict(retrieved_date=max(item['retrieved_date'] for item in source_files),source_files=source_files,seasons=list(range(2022,2026)),coverage=coverage,
                    stats_source=f'{BASE}/stats_player',contracts_source=f'{BASE}/contracts/historical_contracts.parquet',
                    attribution='nflverse contributors and Over The Cap',
                    contract_definition='APY of the latest identifiable non-practice deal signed by season end; extensions may start later. Historical data retrieved in 2026, not frozen opening-day snapshots.',
                    cohort_definition='Supported positions with single-team weekly stats and a matching identifiable contract. Multi-team, ambiguous, and unmatched records are excluded.')
    (output/'sources.json').write_text(json.dumps(metadata,indent=2))
    print(json.dumps(dict(rows=len(df),coverage=coverage,contracts=df.contract_group.value_counts().to_dict(),missing=df[['cap_hit','cash_paid']].isna().sum().to_dict()),indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache',type=Path,default=Path('.data-cache'))
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'data')
    parser.add_argument('--refresh',action='store_true',help='Download source files again before rebuilding')
    args=parser.parse_args()
    build(args.cache,args.output,args.refresh)
