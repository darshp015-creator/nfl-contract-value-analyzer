"""Published league caps and plain-language benchmark explanations."""
import pandas as pd

# League-wide per-team base caps, not team-specific adjusted caps/rollovers.
# https://www.nfl.com/news/nfl-salary-cap-set-at-255-4m-per-team-for-2024-regular-season
# https://www.nfl.com/news/nfl-sets-salary-cap-at-279-2-million-per-team-for-2025-season
SALARY_CAPS = {2022:208_200_000, 2023:224_800_000, 2024:255_400_000, 2025:279_200_000}

# Prior strength in opportunities. Fixed design choices, not fitted to test years.
# Volume components are deliberately left unchanged.
RATE_WORKLOAD = {
    'passing_epa_rate':('dropbacks',100), 'passing_cpoe':('attempts',100),
    'turnover_rate':('qb_plays',100), 'rushing_epa_rate':('carries',75),
    'yards_per_carry':('carries',75), 'receiving_epa_rate':('targets',50),
    'yards_per_target':('targets',50), 'catch_rate':('targets',50),
}

def explain_player(row, components, labels, same_deal=True):
    """Describe observed inputs; never imply causal or predictive attribution."""
    lines=[]
    if pd.notna(row.production_score):
        lines.append(f"Production score: {row.production_score:.1f}/100; before workload adjustment: {row.raw_production_score:.1f}/100.")
    else:
        lines.append('Production score unavailable: no usable production components.')
    if row.sample_adjusted:
        lines.append('Efficiency component scores move toward the position midpoint when opportunities are limited. Season volume components stay unchanged.')
    if pd.notna(row.opportunities):
        lines.append(f"Workload: {int(row.opportunities)} opportunities over {int(row.games)} games." + (' Small sample: efficiency remains uncertain.' if row.small_sample else ''))
    available=[(labels[c],row.get(c+'_score')) for c,_,_ in components if pd.notna(row.get(c+'_score'))]
    if available:
        high=max(available,key=lambda x:x[1]);low=min(available,key=lambda x:x[1])
        lines.append(f"Highest component: {high[0]} ({high[1]:.1f}/100). Lowest: {low[0]} ({low[1]:.1f}/100). These describe the score, not each feature's effect on salary.")
    cohort=f"{int(row.season)} {row.position}"+ (f" / {row.contract_group}" if same_deal else '')
    lines.append(f"Benchmark cohort: {int(row.peer_count)} other eligible {cohort} players. This player's cost is excluded. Method: {row.model_method}.")
    if pd.notna(row.value_ratio):
        lines.append(f"Peer benchmark is {row.value_ratio:.2f} times the selected cost; rank {int(row.position_value_rank)} within that cohort. Higher ratios indicate lower cost relative to the model.")
    else:
        lines.append('No value ranking: selected cost, score, or eligible peers are missing.')
    if row.peer_count<8:
        lines.append('Thin peer group: treat the median benchmark with caution.')
    if row.contract_group=='Unknown':
        lines.append('Contract classification is unknown; these contracts may not be economically comparable.')
    return lines
