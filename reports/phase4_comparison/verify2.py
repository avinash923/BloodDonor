import pandas as pd
import numpy as np

out_dir = r'C:\BloodDonor\reports\phase4_comparison'
df = pd.read_csv(f'{out_dir}\\episode_results.csv')
agg = pd.read_csv(f'{out_dir}\\ranking_policy_metrics.csv')
stats = pd.read_csv(f'{out_dir}\\pairwise_comparisons.csv')

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

numeric_cols = agg.select_dtypes(include=[np.number]).columns
all_close = np.allclose(agg[numeric_cols].fillna(0), recalc_agg[numeric_cols].fillna(0))
print(f'Numeric metrics match (isclose): {all_close}')
