import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pathlib
import numpy as np
import os

combined_df_csv_path = r"G:\ImagingData\Tetsuya\20250913\auto1\summary\combined_df_after_analysis.csv"
LTP_point_df_csv_path = r"G:\ImagingData\Tetsuya\20250913\auto1\summary\LTP_point_df_after_analysis.csv"

dict_pos_to_group = {
    'A5_': 'Cont',
    'B5_': 'Doramapimod', 
    'C5_': 'Erlotinib',
}

LTP_point_df = pd.read_csv(LTP_point_df_csv_path)
combined_df = pd.read_csv(combined_df_csv_path)

path_stem_set = set()
for each_label in LTP_point_df["label"].unique():
    path_stem = pathlib.Path(each_label).stem
    stem_first_three = path_stem[:3]
    path_stem_set.add(stem_first_three)

print(path_stem_set)

#check all of the path_stem_set is in dict_pos_to_group
assert len(path_stem_set) == len(dict_pos_to_group), "path_stem_set and dict_pos_to_group have different lengths"
for each_path_stem in path_stem_set:
    if each_path_stem not in dict_pos_to_group:
        print(each_path_stem)
        assert False
else:
    print("all of the path_stem_set is in dict_pos_to_group")



save_plot_folder = os.path.join(os.path.dirname(LTP_point_df_csv_path), "plot")
os.makedirs(save_plot_folder, exist_ok=True)

#put group to LTP_point_df based on the label's path stem
LTP_point_df["condition"] = LTP_point_df["label"].map(lambda x: dict_pos_to_group[pathlib.Path(x).stem[:3]])
combined_df["condition"] = combined_df["filepath_without_number"].map(lambda x: dict_pos_to_group[pathlib.Path(x).stem[:3]])


max_hour = LTP_point_df["time_after_started_experiment_hour"].max()
top_98p_intn = np.percentile(LTP_point_df["norm_intensity"], 98)
bottom_2p_intn = np.percentile(LTP_point_df["norm_intensity"], 2)


p = sns.swarmplot(x="condition", y="norm_intensity", data=LTP_point_df)
plt.ylabel("Normalized $\Delta$volume (a.u.)")
plt.xlabel("")
#put mean and std line for each group 
sns.boxplot(showmeans=True,
            meanline=True,
            meanprops={'color': 'k', 'ls': '-', 'lw': 2},
            medianprops={'visible': False},
            whiskerprops={'visible': False},
            zorder=10,
            x="condition",
            y="norm_intensity",
            data=LTP_point_df,
            showfliers=False,
            showbox=False,
            showcaps=False,
            ax=p)
# show value in the boxplot in the left side of the boxplot . Show n also.
for each_group in LTP_point_df["condition"].unique():
    each_group_df = LTP_point_df[LTP_point_df["condition"] == each_group]
    plt.text(each_group, each_group_df["norm_intensity"].mean(), 
    f"{each_group_df['norm_intensity'].mean():.2f}",
        ha='left', va='bottom', fontsize=12, color="black")
    plt.text(each_group, top_98p_intn*0.9, 
    f"n = {each_group_df['norm_intensity'].count()}",
        ha='center', va='bottom', fontsize=12, color="black")

plt.ylim([bottom_2p_intn, top_98p_intn])
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.savefig(os.path.join(save_plot_folder, "deltavolume_against_condition.png"), dpi=150,bbox_inches="tight")
plt.show()


for each_group in LTP_point_df["condition"].unique():
    each_group_df = LTP_point_df[LTP_point_df["condition"] == each_group]
    plt.figure(figsize=(4, 3))
    plt.title(each_group)
    plt.ylim([bottom_2p_intn, top_98p_intn])
    plt.xlim([0, max_hour*1.05])
    sns.scatterplot(x='time_after_started_experiment_hour', y="norm_intensity", 
            data=each_group_df, hue="label",legend=False, palette=['k'])
    y_label = plt.ylabel("Normalized $\Delta$volume (a.u.)")
    x_label = plt.xlabel("Time (hour) after started experiment")
    plt.xlim([0, max_hour*1.05])
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.savefig(os.path.join(save_plot_folder, f"deltavolume_against_time_after_start_hour_{each_group}.png"), dpi=150,bbox_inches="tight")
    plt.show()


