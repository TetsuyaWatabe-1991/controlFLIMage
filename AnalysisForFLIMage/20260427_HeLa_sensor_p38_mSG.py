# -*- coding: utf-8 -*-
import os
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

one_of_filepath = r"\\RY-LAB-WS04\ImagingData\Tetsuya\20260329\p38\auto1\Analysis\copied\Thr_p38cnt_2_pos1_001_concat_TimeCourse.csv"
save_folder = os.path.join(os.path.dirname(one_of_filepath), "plot")
os.makedirs(save_folder, exist_ok=True)
csvlist = glob.glob(one_of_filepath[:one_of_filepath.rfind("\\")]+"\\*_TimeCourse.csv")


normalize_frame_range_0start = [0, 4]


group_dict = {
    "p38cnt_":"p38 sensor (control)",
    "p38_":"p38 sensor",
    "Thr_p38cnt_":"p38 Ser to Thr (control)",
    "Thr_p38_":"p38 Ser to Thr",
}

combined_df = pd.DataFrame()
for csvpath in csvlist:
    print(csvpath)
    resultdf=csv_to_df(csvpath,
                       ch_list=[1,2])
    basename = os.path.basename(csvpath)

    match_count = 0
    for each_group, each_group_name in group_dict.items():
        if basename.startswith(each_group):
            resultdf["group"] = each_group_name
            match_count += 1

    if match_count != 1:
        raise ValueError(f"{match_count} matches found for {basename}")
        
    combined_df = pd.concat([combined_df, resultdf], ignore_index=True)


#%%
unique_id = 0
for each_FilePath in combined_df["FilePath"].unique():
    each_file_df = combined_df[combined_df["FilePath"] == each_FilePath]
    for each_ROInum in each_file_df["ROInum"].unique():
        each_ROIdf = each_file_df[each_file_df["ROInum"] == each_ROInum]
        combined_df.loc[each_ROIdf.index, "unique_id"] = str(unique_id)
        unique_id += 1
        for each_ch in each_ROIdf["ch"].unique():
            each_ch_df = each_ROIdf[each_ROIdf["ch"] == each_ch].sort_values(by="NthFrame")
            each_ch_baseline_df = each_ch_df[(each_ch_df["NthFrame"] >= normalize_frame_range_0start[0]) & (each_ch_df["NthFrame"] < normalize_frame_range_0start[1])]
            for each_column in each_ch_df.columns:
                if (each_column.endswith("ROI") and not each_column.startswith("nPixels")):
                    each_ROI_column_df = each_ch_df[each_column]
                    baseval = each_ch_baseline_df[each_column].mean()
                    if each_column in ["Lifetime-ROI", "Lifetime_fit-ROI", "Fraction2-ROI", "Fraction2_fit-ROI"]:
                        each_ROI_column_df = each_ROI_column_df - baseval
                    else:
                        each_ROI_column_df = each_ROI_column_df / baseval
                    
                    combined_df.loc[each_ROI_column_df.index, each_column+"_normalized"] = each_ROI_column_df

#%% time normalization
ignore_sc_difference_within = 50
for each_group in combined_df["group"].unique():
    each_group_df = combined_df[combined_df["group"] == each_group]
    unique_time_sec = each_group_df["time_sec"].unique()
    unique_time_sec.sort()

    for nth_time_sec in range(len(unique_time_sec) - 1):
        time_diff = unique_time_sec[nth_time_sec+1] - unique_time_sec[nth_time_sec]
        # print(time_diff)
        if time_diff > ignore_sc_difference_within:
            continue

        new_time_sec = unique_time_sec[nth_time_sec] + time_diff/2

        #ここで、combined_dfのtime_secをnew_time_secに更新する
        combined_df.loc[(combined_df["time_sec"] == unique_time_sec[nth_time_sec]) & (combined_df["group"] == each_group), "time_sec"] = new_time_sec
        combined_df.loc[(combined_df["time_sec"] == unique_time_sec[nth_time_sec+1]) & (combined_df["group"] == each_group), "time_sec"] = new_time_sec



#%% plot
drug_time_minute = {0: "Anisomycin", (104-21): "SB203580"}
ch = 1

combined_df["time_minute"] = combined_df["time_sec"]/60 - 21
ylim = [-0.18, 0.11]
x_lim = [combined_df["time_minute"].min() - (combined_df["time_minute"].max() - combined_df["time_minute"].min())*0.05,
         combined_df["time_minute"].max() + (combined_df["time_minute"].max() - combined_df["time_minute"].min())*0.05]



for each_group in combined_df["group"].unique():
    each_group_df = combined_df[(combined_df["group"] == each_group) & (combined_df["ch"] == ch)]
    num_unique_id = each_group_df["unique_id"].nunique()
    fig, ax = plt.subplots(figsize=(3, 3))

    # Individual traces in light gray
    for uid, uid_df in each_group_df.groupby("unique_id"):
        uid_df_sorted = uid_df.sort_values("time_minute")
        ax.plot(uid_df_sorted["time_minute"], uid_df_sorted["Lifetime-ROI_normalized"],
                color="gray", alpha=0.3, linewidth=0.8)

    # Mean ± SEM in bold red
    sns.lineplot(x="time_minute", y="Lifetime-ROI_normalized",
                 data=each_group_df, legend=False,
                 color="#C0392B", linewidth=2.0, ax=ax)

    # Horizontal reference line at 0
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.8, alpha=0.4)

    # Drug application: vertical dashed line + italic label at top
    for t_min, drug_name in drug_time_minute.items():
        ax.axvline(x=t_min, color="black", linestyle="--", linewidth=1.0, alpha=0.7)

        if "control" in each_group:
            continue
        ax.text(t_min + 1, ylim[1] * 0.98, drug_name,
                fontsize=7, va="top", ha="left")

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)

    ax.set_ylim(ylim)
    ax.set_xlim(x_lim)
    ax.set_xlabel("Time (min)", fontsize=10)
    ax.set_ylabel("\u0394 Lifetime (ns)", fontsize=10)
    ax.tick_params(axis="both", labelsize=9, direction="out", length=4, width=1.2)
    ax.set_title(each_group, fontsize=11)#, fontweight="bold")

    ax.text(0.97, 0.03, f"{num_unique_id} cells",
            transform=ax.transAxes, fontsize=9,
            ha="right", va="bottom", color="k")

    plt.tight_layout()
    savepath = os.path.join(save_folder, f"{each_group}_Lifetime-ROI_normalized_plot.png")
    plt.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.show()


#%%