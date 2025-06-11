import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

combined_df_reject_bad_data_pkl_path = r"\\ry-lab-yas15\Users\Yasudalab\Documents\Tetsuya_Imaging\20250531\combined\combined_df_reject_bad_data.pkl"
LTP_point_df_pkl_path = r"\\ry-lab-yas15\Users\Yasudalab\Documents\Tetsuya_Imaging\20250531\combined\LTP_point_df.pkl"


combined_df_reject_bad_data_df = pd.read_pickle(combined_df_reject_bad_data_pkl_path)
LTP_point_df = pd.read_pickle(LTP_point_df_pkl_path)

two_conditions = ["4_lines", "2_lines"]

one_of_filepath_dict = {
    r"\\ry-lab-yas15\Users\Yasudalab\Documents\Tetsuya_Imaging\20250531\4lines_2_auto\lowmag1__highmag_1_002.flim":"4_lines",
    r"\\ry-lab-yas15\Users\Yasudalab\Documents\Tetsuya_Imaging\20250531\2lines_1\lowmag1__highmag_1_002.flim":"2_lines",
    r"\\ry-lab-yas15\Users\Yasudalab\Documents\Tetsuya_Imaging\20250601\2lines3_auto\lowmag1__highmag_1_002.flim":"2_lines",
    r"\\ry-lab-yas15\Users\Yasudalab\Documents\Tetsuya_Imaging\20250601\4lines_3_auto\lowmag1__highmag_1_002.flim":"4_lines",
    r"\\ry-lab-yas15\Users\Yasudalab\Documents\Tetsuya_Imaging\20250602\lowmag1__highmag_1_002.flim":"2_lines",
    r"\\ry-lab-yas15\Users\Yasudalab\Documents\Tetsuya_Imaging\20250602\4lines_neuron4\lowmag1__highmag_1_002.flim":"4_lines",        
}

filepath_parent_dict = {}
for key, value in one_of_filepath_dict.items():
    filepath_parent_dict[os.path.dirname(key)] = value

for index, row in combined_df_reject_bad_data_df.iterrows():
    parent_dir = os.path.dirname(combined_df_reject_bad_data_df.at[index, 'filepath_without_number'])
    combined_df_reject_bad_data_df.loc[index, 'condition'] = filepath_parent_dict[parent_dir]

for index, row in LTP_point_df.iterrows():
    parent_dir = os.path.dirname(LTP_point_df.at[index, 'label'])
    LTP_point_df.loc[index, 'condition'] = filepath_parent_dict[parent_dir]


for each_condition in LTP_point_df["condition"].unique():
    print(each_condition)
    LTP_point_df_each_condition = LTP_point_df[LTP_point_df["condition"] == each_condition]
    print("mean: ",LTP_point_df_each_condition["norm_intensity"].mean())
    print("std: ",LTP_point_df_each_condition["norm_intensity"].std())
    print("number of data: ",len(LTP_point_df_each_condition))

#do stats test, welch's t-test between two conditions
from scipy import stats

two_conditions = ["4_lines", "2_lines"]

four_lines_df = LTP_point_df[LTP_point_df["condition"] == "4_lines"]["norm_intensity"].values
two_lines_df = LTP_point_df[LTP_point_df["condition"] == "2_lines"]["norm_intensity"].values

print(stats.ttest_ind(four_lines_df, two_lines_df, equal_var=False))




LTP_point_df.to_csv(os.path.join(os.path.dirname(LTP_point_df_pkl_path), "LTP_point_df_with_condition.csv"), index=False)





plt.figure(figsize=(2, 3))
sns.swarmplot(y="norm_intensity", 
            data=LTP_point_df, legend=False, x="condition")

# #plot mean line and std line
# plt.axhline(LTP_point_df["norm_intensity"].mean(), color="red", linestyle="-",
#             xmin=0.3, xmax=0.7)
# plt.axhline(LTP_point_df["norm_intensity"].mean() + LTP_point_df["norm_intensity"].std(), color="red", linestyle="-",
#             xmin=0.4, xmax=0.6)
# plt.axhline(LTP_point_df["norm_intensity"].mean() - LTP_point_df["norm_intensity"].std(), color="red", linestyle="-",
#             xmin=0.4, xmax=0.6)
#plot text showing mean and std in mean plusminus std way
# plt.text(0.5, LTP_point_df["norm_intensity"].mean(), f"{LTP_point_df['norm_intensity'].mean():.2f} ± {LTP_point_df['norm_intensity'].std():.2f}",
#         ha='center', va='bottom', fontsize=8, color="black")

y_label = plt.ylabel("Normalized $\Delta$volume (a.u.)")
y_lim = plt.ylim()
#delete the right and top axis
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
# plt.savefig(os.path.join(save_plot_folder, "intensity_swarm_binned.png"), dpi=150,bbox_inches="tight")
plt.show()

