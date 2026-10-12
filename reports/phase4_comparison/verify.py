import pandas as pd
import numpy as np

out_dir = r'C:\BloodDonor\reports\phase4_comparison'
df = pd.read_csv(f'{out_dir}\\episode_results.csv')
agg = pd.read_csv(f'{out_dir}\\ranking_policy_metrics.csv')
stats = pd.read_csv(f'{out_dir}\\pairwise_comparisons.csv')

print('--- Scenarios ---')
print(f'Total rows in episode_results: {len(df)}')
scenarios_per_policy = df.groupby('Policy')['Scenario_ID'].nunique()
print(scenarios_per_policy)
print(f'All policies have 500 scenarios: {(scenarios_per_policy == 500).all()}')

print('\n--- Metric Recalculation ---')
recalc_agg = df.groupby('Policy').apply(lambda x: pd.Series({
    'Deadline_Fulfilment_Rate': x['Quota_Met_Deadline'].mean() * 100,
    'Eventual_Fulfilment_Rate': x['Quota_Met_Eventual'].mean() * 100,
    'Mean_Time_To_Fulfill_Eventual': x.loc[x['Quota_Met_Eventual'], 'Time_To_Fulfill_Eventual'].mean(),
    'Median_Time_To_Fulfill_Eventual': x.loc[x['Quota_Met_Eventual'], 'Time_To_Fulfill_Eventual'].median(),
    'Invitations_Per_Successful_Request': x.loc[x['Quota_Met_Eventual'], 'Alerts_Sent'].sum() / max(1, x['Quota_Met_Eventual'].sum()),
    'Mean_Shortfall_Ratio': (x['Shortfall'].sum() / max(1, x['Units_Req'].sum())),
    'Alert_Fatigue_Index': x['Alerts_Sent'].sum() / max(1, x['Units_Req'].sum()),
    'Late_Arrival_Rate': (x['Late_Arrivals'].sum() / max(1, x['Units_Fulfilled'].sum())) * 100,
    'No_Suitable_Donor_Rate': x['No_Suitable_Donors'].mean() * 100,
    'Failed_Requests_Count': (~x['Quota_Met_Eventual']).sum()
}), include_groups=False).reset_index()

print('Metrics match:', agg.equals(recalc_agg))
if not agg.equals(recalc_agg):
    print(agg.compare(recalc_agg))

print('\n--- Stats Preview ---')
print(stats)
