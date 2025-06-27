# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 14:29:09 2024

@author: WatabeT
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 13:49:32 2024

@author: WatabeT
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 13:46:14 2023

@author: WatabeT
"""
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('pdf', fonttype=42)
matplotlib.rc('font',size = 8)
plt.rcParams["font.family"] = "Arial"
import math
from read_flimagecsv import arrange_for_multipos3, csv_to_df, detect_uncaging, value_normalize, everymin_normalize
import numpy as np
from scipy import stats
from scipy.stats import levene, bartlett

save_True = True
target_roi_num = 1
time_bin = 10

min_lower = 20
min_upper = 30



# one_of_filepath = r"\\RY-LAB-WS04\ImagingData\Tetsuya\20241213\24well\highmagGFP200ms55p\tpem_1\Analysis"

# one_of_filepath = r"\\RY-LAB-WS04\ImagingData\Tetsuya\20250104\24well\highmag_Trans5ms\tpem\Analysis - Copy\A2_00_1_1__highmag_2__TimeCourse__all_combined.csv"
# well_dict = {
#         "C1":"DMSO",
#         # "A2":"APV 50 uM",
#         # "D1":"APV 5 uM",
#         # "E1":"APV 0.5 uM",
#         # "F1":"APV 0.05 uM"
#         }

one_of_filepath = r"\\RY-LAB-WS04\ImagingData\Tetsuya\20250110\24well\highmag_GFP200ms55p\tpem\Analysis - Copy\A4_00_1_1__highmag_2__TimeCourse.csv"
well_dict = {"":"all"}

# one_of_filepath = r"\\RY-LAB-WS04\ImagingData\Tetsuya\20250114\24well\highmag_GFP200ms47p\tpem\Analysis - Copy\A2_00_8_2__highmag_3__TimeCourse.csv"
# well_dict = {"":"all"}

# one_of_filepath = r"\\RY-LAB-WS04\ImagingData\Tetsuya\20250115\24well7\highmag_GFP200ms47p\tpem\Analysis - Copy\B1_00_3_1__highmag_2__TimeCourse.csv"
# well_dict = {"":"all"}

swarmplot_ylim = [-0.4,3.1]

res_dict = {}




# allcombined_df_savepath= r"\\RY-LAB-WS04\ImagingData\Tetsuya\20240827\24well_0808GFP\highmag_Trans5ms\tpem_tpem2_combined.xxx"


allcombined_df=pd.DataFrame()

csvpath = r"G:\ImagingData\Tetsuya\20250623\Analysis\dend_test_aligned_TimeCourse.csv"
resultdf=csv_to_df(csvpath,
                    ch_list=[2],
                    prefix_list=["sumIntensity_bg-ROI"]
                    )
                    # ch_list=[1,2])#,
                    # prefix_list=["sumIntensity_bg-ROI"])

resultdf.head()

for each_ROInum in resultdf["ROInum"].unique():
    each_df = resultdf[resultdf["ROInum"] == each_ROInum]
    mean_sumIntensity_bg_ROI = each_df["sumIntensity_bg-ROI"].mean()
    each_df["norm_sumIntensity_bg-ROI"] = each_df["sumIntensity_bg-ROI"] / mean_sumIntensity_bg_ROI
    each_df["ROInum"] = each_ROInum
    each_df["ROInum_str"] = str(each_ROInum)
    allcombined_df = pd.concat([allcombined_df, each_df], ignore_index=True)
    
    

plt.figure(figsize = [10,4])
##################################
sns.lineplot(x="Time_min", y="norm_sumIntensity_bg-ROI",
                legend=False, hue = "ROInum_str",
                data = allcombined_df[allcombined_df['ch']==2],
                )
plt.ylabel("Norm. spine volume (a.u.)")
plt.xlabel("Time (min)")
plt.plot([24,58],[4.1,4.1],c='red',ls = '-', zorder = 1)
plt.text(24,4.1,"Mg 0.1 mM",ha="left",va="bottom",zorder=100)
plt.gca().spines[['right', 'top']].set_visible(False)
plt.savefig(csvpath[:-4]+"_norm_sumIntensity_bg-ROI_plot.png", bbox_inches = "tight", dpi = 200)

# Add condition labels based on frame number
# After 20th frame is considered as Mg added condition
allcombined_df['condition'] = 'None'
allcombined_df.loc[allcombined_df['NthFrame'] >= 26, 'condition'] = 'Mg_added'
allcombined_df.loc[allcombined_df['NthFrame'] <= 20, 'condition'] = 'baseline'

# Calculate variation metrics for each ROI
variation_data = []

for roi in allcombined_df['ROInum_str'].unique():
    roi_data = allcombined_df[allcombined_df['ROInum_str'] == roi]
    
    # Baseline data (before Mg addition)
    baseline_data = roi_data[roi_data['condition'] == 'baseline']['norm_sumIntensity_bg-ROI']
    
    # Mg added data (after Mg addition)
    mg_data = roi_data[roi_data['condition'] == 'Mg_added']['norm_sumIntensity_bg-ROI']
    
    # Calculate variation metrics
    baseline_cv = baseline_data.std() / baseline_data.mean()  # Coefficient of variation
    mg_cv = mg_data.std() / mg_data.mean()
    
    baseline_range = baseline_data.max() - baseline_data.min()  # Range
    mg_range = mg_data.max() - mg_data.min()
    
    baseline_iqr = baseline_data.quantile(0.75) - baseline_data.quantile(0.25)  # IQR
    mg_iqr = mg_data.quantile(0.75) - mg_data.quantile(0.25)
    
    # Store data for plotting
    variation_data.append({
        'ROI': roi,
        'baseline_cv': baseline_cv,
        'mg_cv': mg_cv,
        'baseline_range': baseline_range,
        'mg_range': mg_range,
        'baseline_iqr': baseline_iqr,
        'mg_iqr': mg_iqr
    })

# Convert to DataFrame
variation_df = pd.DataFrame(variation_data)

# Create swarm plots for different variation metrics
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 1. Coefficient of Variation comparison
baseline_cv_data = variation_df['baseline_cv'].values
mg_cv_data = variation_df['mg_cv'].values

# Paired t-test for CV
cv_t_stat, cv_p_val = stats.ttest_rel(baseline_cv_data, mg_cv_data)

axes[0].plot([-0.2, 0.2], [baseline_cv_data.mean(), baseline_cv_data.mean()], 'k-', linewidth=2, zorder=1)
axes[0].plot([0.8, 1.2], [mg_cv_data.mean(), mg_cv_data.mean()], 'k-', linewidth=2, zorder=1)
axes[0].plot([0, 1], [baseline_cv_data.mean(), mg_cv_data.mean()], 'k--', alpha=0.5, zorder=1)

sns.swarmplot(data=[baseline_cv_data, mg_cv_data], ax=axes[0], palette=['lightblue', 'lightcoral'])
axes[0].set_xticklabels(['Baseline', 'Mg Added'])
axes[0].set_ylabel('Coefficient of Variation')
axes[0].set_title(f'CV Comparison\np = {cv_p_val:.4f}')
axes[0].spines[['right', 'top']].set_visible(False)

# 2. Range comparison
baseline_range_data = variation_df['baseline_range'].values
mg_range_data = variation_df['mg_range'].values

# Paired t-test for Range
range_t_stat, range_p_val = stats.ttest_rel(baseline_range_data, mg_range_data)

axes[1].plot([-0.2, 0.2], [baseline_range_data.mean(), baseline_range_data.mean()], 'k-', linewidth=2, zorder=1)
axes[1].plot([0.8, 1.2], [mg_range_data.mean(), mg_range_data.mean()], 'k-', linewidth=2, zorder=1)
axes[1].plot([0, 1], [baseline_range_data.mean(), mg_range_data.mean()], 'k--', alpha=0.5, zorder=1)

sns.swarmplot(data=[baseline_range_data, mg_range_data], ax=axes[1], palette=['lightblue', 'lightcoral'])
axes[1].set_xticklabels(['Baseline', 'Mg Added'])
axes[1].set_ylabel('Range')
axes[1].set_title(f'Range Comparison\np = {range_p_val:.4f}')
axes[1].spines[['right', 'top']].set_visible(False)

# 3. IQR comparison
baseline_iqr_data = variation_df['baseline_iqr'].values
mg_iqr_data = variation_df['mg_iqr'].values

# Paired t-test for IQR
iqr_t_stat, iqr_p_val = stats.ttest_rel(baseline_iqr_data, mg_iqr_data)

axes[2].plot([-0.2, 0.2], [baseline_iqr_data.mean(), baseline_iqr_data.mean()], 'k-', linewidth=2, zorder=1)
axes[2].plot([0.8, 1.2], [mg_iqr_data.mean(), mg_iqr_data.mean()], 'k-', linewidth=2, zorder=1)
axes[2].plot([0, 1], [baseline_iqr_data.mean(), mg_iqr_data.mean()], 'k--', alpha=0.5, zorder=1)

sns.swarmplot(data=[baseline_iqr_data, mg_iqr_data], ax=axes[2], palette=['lightblue', 'lightcoral'])
axes[2].set_xticklabels(['Baseline', 'Mg Added'])
axes[2].set_ylabel('IQR')
axes[2].set_title(f'IQR Comparison\np = {iqr_p_val:.4f}')
axes[2].spines[['right', 'top']].set_visible(False)

plt.tight_layout()
plt.savefig(csvpath[:-4]+"_variation_comparison_swarm.png", bbox_inches="tight", dpi=200)
plt.show()

# Print statistical summary
print("=== Variation Comparison Statistics ===")
print(f"Number of ROIs: {len(variation_df)}")
print(f"\nCoefficient of Variation:")
print(f"  Baseline mean ± std: {baseline_cv_data.mean():.4f} ± {baseline_cv_data.std():.4f}")
print(f"  Mg added mean ± std: {mg_cv_data.mean():.4f} ± {mg_cv_data.std():.4f}")
print(f"  Paired t-test: t = {cv_t_stat:.4f}, p = {cv_p_val:.4f}")

print(f"\nRange:")
print(f"  Baseline mean ± std: {baseline_range_data.mean():.4f} ± {baseline_range_data.std():.4f}")
print(f"  Mg added mean ± std: {mg_range_data.mean():.4f} ± {mg_range_data.std():.4f}")
print(f"  Paired t-test: t = {range_t_stat:.4f}, p = {range_p_val:.4f}")

print(f"\nIQR:")
print(f"  Baseline mean ± std: {baseline_iqr_data.mean():.4f} ± {baseline_iqr_data.std():.4f}")
print(f"  Mg added mean ± std: {mg_iqr_data.mean():.4f} ± {mg_iqr_data.std():.4f}")
print(f"  Paired t-test: t = {iqr_t_stat:.4f}, p = {iqr_p_val:.4f}")

# Calculate percentage reduction
cv_reduction = ((baseline_cv_data.mean() - mg_cv_data.mean()) / baseline_cv_data.mean()) * 100
range_reduction = ((baseline_range_data.mean() - mg_range_data.mean()) / baseline_range_data.mean()) * 100
iqr_reduction = ((baseline_iqr_data.mean() - mg_iqr_data.mean()) / baseline_iqr_data.mean()) * 100

print(f"\n=== Percentage Reduction ===")
print(f"CV reduction: {cv_reduction:.1f}%")
print(f"Range reduction: {range_reduction:.1f}%")
print(f"IQR reduction: {iqr_reduction:.1f}%")