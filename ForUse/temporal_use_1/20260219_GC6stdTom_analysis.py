# %%
import os
import sys
import datetime
sys.path.append(r"..\..")
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_rel
import matplotlib
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial"]
from custom_plot import plt

def format_p_value(p: float) -> str:
    """Format p-value for figure annotation (e.g. p = 0.02, p < 0.001)."""
    if p != p:  # NaN
        return "n.s."
    if p < 0.001:
        return "p < 0.001"
    return f"p = {p:.3f}"

photon_threshold_for_intensity = 30
photon_threshold_for_lifetime = 1000

unc_total_frame_first_unc_dict = {
    33: 2,
    55: 5,
}

group_header_dict = {
    "0121": "0121",
    "1203": "1203"
}

df_save_path_1 = r"//RY-LAB-WS04/ImagingData/Tetsuya/20260217/auto1\combined_df_1.pkl"
out_csv_path = r"//RY-LAB-WS04/ImagingData/Tetsuya/20260217/auto1\combined_df_1_intensity_lifetime_all_frames.csv"

combined_df = pd.read_pickle(df_save_path_1)
fulltimeseries_df = pd.read_csv(out_csv_path)

# normalize the intensity by dividing the intensity by the number of summed frames
fulltimeseries_df.loc[:,"Spine_Ch1_intensity_div_by_nAve"] = fulltimeseries_df.loc[:,"Spine_Ch1_intensity"] / fulltimeseries_df.loc[:,"nAveFrame"]
fulltimeseries_df.loc[:,"Spine_Ch2_intensity_div_by_nAve"] = fulltimeseries_df.loc[:,"Spine_Ch2_intensity"] / fulltimeseries_df.loc[:,"nAveFrame"]

combined_df["dt"] = pd.to_datetime(combined_df["dt_str"])

save_folder = os.path.join(os.path.dirname(df_save_path_1), "summary")
os.makedirs(save_folder, exist_ok=True)


#%% align the time_sec based on the uncaging timing

fulltimeseries_df.loc[:,"aligned_time_sec"] = -999.99

summary_df = pd.DataFrame()
for each_group in fulltimeseries_df["group"].unique():
    for each_set_label in fulltimeseries_df[fulltimeseries_df["group"] == each_group]["set_label"].unique():
        group_set_id = f"{each_group}_{each_set_label}"

        each_df = fulltimeseries_df[(fulltimeseries_df["group"] == each_group) & (fulltimeseries_df["set_label"] == each_set_label)]

        # name the each_df with group_set_id
        fulltimeseries_df.loc[each_df.index, "group_set_id"] = group_set_id

        # uncaging trigger time is 0 seconds
        length_of_unc_df = len(each_df[each_df["phase"] == "unc"])
        if length_of_unc_df in unc_total_frame_first_unc_dict.keys():
            unc_trigger_time = each_df[each_df["phase"] == "unc"]["elapsed_time_sec"].iloc[unc_total_frame_first_unc_dict[length_of_unc_df] - 1]
            aligned_values = each_df["elapsed_time_sec"].values - unc_trigger_time
        else:
            raise ValueError(f"Length of each df is not in unc_total_frame_first_unc_dict: {length_of_unc_df}")        
        fulltimeseries_df.loc[each_df.index, "aligned_time_sec"] = aligned_values
        
        #normalize the lifetime, by subtracting the mean of the lifetime of frames in pre phase
        pre_phase_lifetime = each_df[each_df["phase"] == "pre"]["Spine_Ch1_lifetime"].mean()
        fulltimeseries_df.loc[each_df.index, "Spine_Ch1_lifetime_normalized"] = fulltimeseries_df.loc[each_df.index, "Spine_Ch1_lifetime"] - pre_phase_lifetime

        #normalize the intensity, by dividing the intensity by the mean of the intensity of frames in pre phase and subtract 1
        pre_phase_intensity = each_df[each_df["phase"] == "pre"]["Spine_Ch1_intensity_div_by_nAve"].mean()
        fulltimeseries_df.loc[each_df.index, "Spine_Ch1_intensity_normalized"] = fulltimeseries_df.loc[each_df.index, "Spine_Ch1_intensity_div_by_nAve"] / pre_phase_intensity - 1
        pre_phase_intensity_ch2 = each_df[each_df["phase"] == "pre"]["Spine_Ch2_intensity_div_by_nAve"].mean()
        fulltimeseries_df.loc[each_df.index, "Spine_Ch2_intensity_normalized"] = fulltimeseries_df.loc[each_df.index, "Spine_Ch2_intensity_div_by_nAve"] / pre_phase_intensity_ch2 - 1

        #summary for each group_set_id
        each_summary_dict = {}
        each_summary_dict["group"] = each_group
        each_summary_dict["set_label"] = each_set_label
        each_summary_dict["group_set_id"] = group_set_id

        for each_ch in ['Ch1', 'Ch2']:
            for each_phase in ['pre', 'post']:
                for each_signal in ['lifetime', 'intensity']:
                    each_summary_dict[f"{each_ch}_{each_phase}_{each_signal}"] = each_df[each_df["phase"] == each_phase][f"Spine_{each_ch}_{each_signal}"].mean()

        each_summary_dict["delta_lifetime"] = each_summary_dict["Ch1_post_lifetime"] - each_summary_dict["Ch1_pre_lifetime"]
        each_summary_dict["delta_FF0_intensity"] = each_summary_dict["Ch1_post_intensity"] / each_summary_dict["Ch1_pre_intensity"] - 1


        summary_df = pd.concat([summary_df, pd.DataFrame([each_summary_dict])], ignore_index=True)


#bin the time
pre_phase_sec = float(fulltimeseries_df[fulltimeseries_df["phase"] == "pre"]["aligned_time_sec"].mean())
post_phase_sec = float(fulltimeseries_df[fulltimeseries_df["phase"] == "post"]["aligned_time_sec"].mean())
pre_index = fulltimeseries_df[fulltimeseries_df["phase"] == "pre"].index
post_index = fulltimeseries_df[fulltimeseries_df["phase"] == "post"].index
fulltimeseries_df.loc[pre_index, "binned_time_sec"] = pre_phase_sec
fulltimeseries_df.loc[post_index, "binned_time_sec"] = post_phase_sec

#calc bin time during uncaging
average_bin = 0
num_bin = 0
for each_group_set_id in fulltimeseries_df["group_set_id"].unique():
    each_df = fulltimeseries_df[fulltimeseries_df["group_set_id"] == each_group_set_id]
    bin_sec_during_uncaging = float((each_df[each_df["phase"] == "unc"]["aligned_time_sec"].max() - each_df[each_df["phase"] == "unc"]["aligned_time_sec"].min()) / len(each_df[each_df["phase"] == "unc"]))
    average_bin += bin_sec_during_uncaging
    num_bin += 1
average_bin = average_bin / num_bin

for each_group_set_id in fulltimeseries_df["group_set_id"].unique():
    each_df = fulltimeseries_df[fulltimeseries_df["group_set_id"] == each_group_set_id]
    unc_df = each_df[each_df["phase"] == "unc"]
    length_of_unc_df = len(unc_df)
    if length_of_unc_df in unc_total_frame_first_unc_dict.keys():
        time_0_nth = unc_total_frame_first_unc_dict[length_of_unc_df] - 1
    else:
        raise ValueError(f"Length of each uncaging df is not in unc_total_frame_first_unc_dict: {length_of_unc_df}")
    time_list = [i*average_bin for i in range(-time_0_nth,len(unc_df)-time_0_nth)]

    fulltimeseries_df.loc[unc_df.index, "binned_time_sec"] = time_list



# %% line plot
#plot each data, with light thin color lines
plot_info_dict = {
    # "lifetime": {"ylabel": r"$\Delta$lifetime (ns)", "y": "Spine_Ch1_lifetime_normalized", "errorbar": "se"},
    "intensity": {"ylabel": r"$\Delta$spine volume (a.u.)", 
    "xlabel": "Time (sec)",
    "x": "aligned_time_sec",
    "y": "Spine_Ch2_intensity_normalized", "errorbar": "se"}
    }

for each_header, each_header_name in group_header_dict.items():
    eachgroup_df = fulltimeseries_df[fulltimeseries_df["group"].str.contains(each_header)]
    if len(eachgroup_df) == 0:
        continue
    for each_plot_type, each_plot_info in plot_info_dict.items():
        plt.figure(figsize=(5, 3))
        g = sns.lineplot(
                    x = each_plot_info["x"],
                    y = each_plot_info["y"], 
                    data = eachgroup_df, 
                    hue = "group_set_id", 
                    linewidth = 0.5, 
                    alpha = 0.5,
                    palette = "tab10",
                    legend = False,
                    )
        #greek delta lifetime
        plt.ylabel(each_plot_info["ylabel"])
        plt.xlabel(each_plot_info["xlabel"])
        plt.title(each_header_name)
        
        # #plot mean with SEM
        # sns.lineplot(x = each_plot_info["x"], 
        #             y = each_plot_info["y"],
        #             data = eachgroup_df,
        #             errorbar = each_plot_info["errorbar"],
        #             linewidth = 2,
        #             color = "r",
        #             )

        ylim = plt.gca().get_ylim()
        ninty_percent_ylim = ylim[0] + (ylim[1] - ylim[0]) * 0.9
        plt.plot([0, 2.048*29], [ninty_percent_ylim, ninty_percent_ylim], "k-")
        plt.text(0, ninty_percent_ylim*1.005, "uncaging", ha="left", va="bottom")
        
        #delete right and top border
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)
        savepath = os.path.join(save_folder, f"{each_header}_{each_plot_type}_lineplot_time_series.png")
        plt.savefig(savepath, dpi=150, bbox_inches = "tight")
        plt.show()

# %% swarm plot

plot_info_dict = {
    # "lifetime": {"ylabel": r"$\Delta$lifetime (ns)",
    #                             "y": "delta_lifetime",
    #                             "ylim" : [-0.19, 0.29]},
                  "intensity": {"ylabel": r"$\Delta$spine volume (a.u.)", 
                                "y": "delta_FF0_intensity", 
                                "ylim" : [-0.4, 5.9]}
                }

for each_header, each_header_name in group_header_dict.items():
    each_header_summary_df = summary_df[summary_df["group"].str.contains(each_header)]
    if len(each_header_summary_df) == 0:
        continue
    for each_plot_type, each_plot_info in plot_info_dict.items():
        plt.figure(figsize=(2, 3))
        p = sns.swarmplot(y=each_plot_info["y"],
                    data=each_header_summary_df,
                    palette = "tab10",
                    )

        sns.boxplot(showmeans=True,
            meanline=True,
            meanprops={'color': 'r', 'ls': '-', 'lw': 1},
            medianprops={'visible': False},
            whiskerprops={'visible': False},
            zorder=10,
            y=each_plot_info["y"],
            data=each_header_summary_df,
            showfliers=False,
            showbox=False,
            showcaps=False,
            ax=p)

        plt.ylabel(each_plot_info["ylabel"])
        plt.ylim(each_plot_info["ylim"])
        plt.title(each_header_name)
        #delete right and top border
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)
        savepath = os.path.join(save_folder, f"{each_header}_{each_plot_type}_plot_swarmplot.png")
        plt.savefig(savepath, dpi=150, bbox_inches = "tight")
        plt.show()

# %%  Get ratio of Ch1 to Ch2


for each_header, each_header_name in group_header_dict.items():
    
    each_header_summary_df = summary_df[summary_df["group"].str.contains(each_header)].copy()
    each_header_summary_df["ratio_Ch2_over_Ch1_post"] = each_header_summary_df["Ch2_post_intensity"] / each_header_summary_df["Ch1_post_intensity"]
    each_header_summary_df["ratio_Ch2_over_Ch1_pre"] = each_header_summary_df["Ch2_pre_intensity"] / each_header_summary_df["Ch1_pre_intensity"]
    if len(each_header_summary_df) == 0:
        continue
    x_pre, x_post = 0, 1

    plt.figure(figsize=(2.5, 3))
    ax = plt.gca()

    # Paired lines: one line per group_set_id (pre -> post)
    for _, row in each_header_summary_df.iterrows():
        ax.plot(
            [x_pre, x_post],
            [row["ratio_Ch2_over_Ch1_pre"], row["ratio_Ch2_over_Ch1_post"]],
            color="gray",
            linewidth=0.8,
            alpha=0.7,
            zorder=1,
        )

    # Individual points at pre and post
    ax.scatter(
        [x_pre] * len(each_header_summary_df),
        each_header_summary_df["ratio_Ch2_over_Ch1_pre"],
        color="gray",
        s=20,
        alpha=0.8,
        zorder=2,
        edgecolors="none",
    )
    ax.scatter(
        [x_post] * len(each_header_summary_df),
        each_header_summary_df["ratio_Ch2_over_Ch1_post"],
        color="gray",
        s=20,
        alpha=0.8,
        zorder=2,
        edgecolors="none",
    )

    # Mean +/- SEM
    pre_mean = float(each_header_summary_df["ratio_Ch2_over_Ch1_pre"].mean())
    pre_sem = float(each_header_summary_df["ratio_Ch2_over_Ch1_pre"].sem())
    post_mean = float(each_header_summary_df["ratio_Ch2_over_Ch1_post"].mean())
    post_sem = float(each_header_summary_df["ratio_Ch2_over_Ch1_post"].sem())
    ax.errorbar(
        [x_pre],
        [pre_mean],
        yerr=[pre_sem],
        color="k",
        capsize=3,
        capthick=1,
        linewidth=1.5,
        fmt="o",
        markersize=6,
        zorder=3,
    )
    ax.errorbar(
        [x_post],
        [post_mean],
        yerr=[post_sem],
        color="k",
        capsize=3,
        capthick=1,
        linewidth=1.5,
        fmt="o",
        markersize=6,
        zorder=3,
    )

    ax.set_xticks([x_pre, x_post])
    ax.set_xticklabels(["Pre", "Post"])
    ax.set_ylabel("Ch2/Ch1 intensity ratio")
    ax.set_title(each_header_name)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Paired t-test (pre vs post)
    _, p_ratio = ttest_rel(
        each_header_summary_df["ratio_Ch2_over_Ch1_pre"],
        each_header_summary_df["ratio_Ch2_over_Ch1_post"],
    )
    ax.text(0.98, 0.98, format_p_value(float(p_ratio)), transform=ax.transAxes, fontsize=9, va="top", ha="right")

    savepath = os.path.join(save_folder, f"{each_header}_ratio_Ch2_over_Ch1_paired.png")
    plt.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.show()

# %% Paired pre-post plot for Ch1 lifetime

for each_header, each_header_name in group_header_dict.items():
    each_header_summary_df = summary_df[summary_df["group"].str.contains(each_header)].copy()
    if len(each_header_summary_df) == 0:
        continue

    x_pre, x_post = 0, 1

    plt.figure(figsize=(2.5, 3))
    ax = plt.gca()

    # Paired lines: one line per group_set_id (pre -> post)
    for _, row in each_header_summary_df.iterrows():
        ax.plot(
            [x_pre, x_post],
            [row["Ch1_pre_lifetime"], row["Ch1_post_lifetime"]],
            color="gray",
            linewidth=0.8,
            alpha=0.7,
            zorder=1,
        )

    # Individual points at pre and post
    ax.scatter(
        [x_pre] * len(each_header_summary_df),
        each_header_summary_df["Ch1_pre_lifetime"],
        color="gray",
        s=20,
        alpha=0.8,
        zorder=2,
        edgecolors="none",
    )
    ax.scatter(
        [x_post] * len(each_header_summary_df),
        each_header_summary_df["Ch1_post_lifetime"],
        color="gray",
        s=20,
        alpha=0.8,
        zorder=2,
        edgecolors="none",
    )

    # Mean +/- SEM
    pre_mean = float(each_header_summary_df["Ch1_pre_lifetime"].mean())
    pre_sem = float(each_header_summary_df["Ch1_pre_lifetime"].sem())
    post_mean = float(each_header_summary_df["Ch1_post_lifetime"].mean())
    post_sem = float(each_header_summary_df["Ch1_post_lifetime"].sem())
    ax.errorbar(
        [x_pre],
        [pre_mean],
        yerr=[pre_sem],
        color="k",
        capsize=3,
        capthick=1,
        linewidth=1.5,
        fmt="o",
        markersize=6,
        zorder=3,
    )
    ax.errorbar(
        [x_post],
        [post_mean],
        yerr=[post_sem],
        color="k",
        capsize=3,
        capthick=1,
        linewidth=1.5,
        fmt="o",
        markersize=6,
        zorder=3,
    )

    ax.set_xticks([x_pre, x_post])
    ax.set_xticklabels(["Pre", "Post"])
    ax.set_ylabel("Ch1 lifetime (ns)")
    ax.set_title(each_header_name)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Fix y-axis range to 0.35 ns (centered on data mean)
    y_center = (pre_mean + post_mean) / 2.0
    ax.set_ylim(y_center - 0.175, y_center + 0.175)

    # Show mean difference (post - pre) and paired t-test in the graph
    delta_mean = post_mean - pre_mean
    _, p_lifetime = ttest_rel(
        each_header_summary_df["Ch1_pre_lifetime"],
        each_header_summary_df["Ch1_post_lifetime"],
    )
    ax.text(0.98, 0.98, r"$\Delta$ = " + f"{delta_mean:.2f} ns\n" + format_p_value(float(p_lifetime)), transform=ax.transAxes, fontsize=9, va="top", ha="right")

    savepath = os.path.join(save_folder, f"{each_header}_Ch1_lifetime_paired.png")
    plt.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.show()
