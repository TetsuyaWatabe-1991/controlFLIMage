# %%
import json
import os
import sys
import glob
import datetime
sys.path.append(r"..\..")
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_rel
import matplotlib
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial"]
from custom_plot import plt
import numpy as np

def format_p_value(p: float) -> str:
    """Format p-value for figure annotation (e.g. p = 0.02, p < 0.001)."""
    if p != p:  # NaN
        return "n.s."
    if p < 0.001:
        return "p < 0.001"
    return f"p = {p:.3f}"


powermeter_folder= r"//RY-LAB-WS04/Users/yasudalab/Documents/Tetsuya_Imaging/powermeter"
assert os.path.exists(powermeter_folder), f"powermeter_folder does not exist: {powermeter_folder}"

photon_threshold_for_intensity = 30
photon_threshold_for_lifetime = 1000
from_Thorlab_to_coherent_factor = 1/3

unc_total_frame_first_unc_dict = {
    33: 2,
    55: 5,
}

group_header_dict = {
    "cnt": "Control",
    "Ab": "Beta Amyloid"
}


LTP_data_point_after_min_between = [25,35]

df_save_path_1 = r"//RY-LAB-WS04/ImagingData/Tetsuya/20260218/auto1\combined_df_1.pkl"
out_csv_path = r"//RY-LAB-WS04/ImagingData/Tetsuya/20260218/auto1\combined_df_1_intensity_lifetime_all_frames.csv"

combined_df = pd.read_pickle(df_save_path_1)
fulltimeseries_df = pd.read_csv(out_csv_path)

# normalize the intensity by dividing the intensity by the number of summed frames
fulltimeseries_df.loc[:,"Spine_Ch1_intensity_div_by_nAve"] = fulltimeseries_df.loc[:,"Spine_Ch1_intensity"] / fulltimeseries_df.loc[:,"nAveFrame"]
fulltimeseries_df.loc[:,"Spine_Ch2_intensity_div_by_nAve"] = fulltimeseries_df.loc[:,"Spine_Ch2_intensity"] / fulltimeseries_df.loc[:,"nAveFrame"]

combined_df["dt"] = pd.to_datetime(combined_df["dt_str"])

save_folder = os.path.join(os.path.dirname(df_save_path_1), "summary")
os.makedirs(save_folder, exist_ok=True)



#%% get uncaging power
combined_df["uncaging_power"] = np.nan
fulltimeseries_df["uncaging_power"] = np.nan

unc_df = combined_df[combined_df['phase'] == 'unc']
for each_file in unc_df["file_path"].unique():
    statedict = unc_df[unc_df["file_path"] == each_file]["statedict"].values[0]
    uncaging_power = statedict["State.Uncaging.Power"]
    combined_df.loc[combined_df["file_path"] == each_file, "uncaging_power"] = uncaging_power
    fulltimeseries_df.loc[fulltimeseries_df["file_path"] == each_file, "uncaging_power"] = uncaging_power

print("uncaging power in combined_df:")
print(combined_df["uncaging_power"].unique())
print("uncaging power in fulltimeseries_df:")
print(fulltimeseries_df["uncaging_power"].unique())

list_of_uncaging_power = list(combined_df["uncaging_power"].unique()[combined_df["uncaging_power"].unique() > 0])

for each_uncaging_power in list_of_uncaging_power:
    print(each_uncaging_power)

earliest_acq_time = datetime.datetime.strptime(fulltimeseries_df["acq_time_str"].min(), "%Y-%m-%dT%H:%M:%S.%f")

powermeter_ini_files = glob.glob(os.path.join(powermeter_folder, "*.json")) +\
                    glob.glob(os.path.join(powermeter_folder, "old/*.json"))

#get the latest powermeter json file before the earliest acq time
latest_powermeter_json_file = None

datetime_only_arr = np.array([int(os.path.basename(each_file).replace(".json", "")) for each_file in powermeter_ini_files])
latest_powermeter_json_basename = f"{datetime_only_arr[datetime_only_arr < int(earliest_acq_time.strftime("%Y%m%d%H%M"))].max()}.json"
latest_powermeter_json_file = [each_file for each_file in powermeter_ini_files if os.path.basename(each_file) == latest_powermeter_json_basename]
assert len(latest_powermeter_json_file) == 1, f"Latest powermeter json file is not found: {latest_powermeter_json_file}"
latest_powermeter_json_file = latest_powermeter_json_file[0]

# Build power % -> power mW dict from powermeter calibration JSON (same approach as GCaMP_unc_combined_titration / pyautogui_LaserPower_PM16_121)
# JSON format: {"Laser1": {percent: mW, ...}, "Laser2": {percent: mW, ...}}; Laser2 = 720 nm uncaging
with open(latest_powermeter_json_file, "r") as f:
    powermeter_calib = json.load(f)
x_percent = np.array(list(powermeter_calib["Laser2"].keys())).astype(float)
y_mW = np.array(list(powermeter_calib["Laser2"].values())).astype(float)
power_slope, power_intercept = np.polyfit(x_percent, y_mW, 1)
# Dict: uncaging_power (%) -> mW for each percent used in this experiment
power_percent_to_mW_dict = {
    int(p): round(float(power_slope * p + power_intercept), 2)
    for p in list_of_uncaging_power
}

power_percent_to_mW_dict = {
    int(p): round(float(power_slope * p + power_intercept), 2)
    for p in list_of_uncaging_power
}

power_percent_to_coherent_mW_dict = {
    int(p): round(float(power_slope * p + power_intercept) * from_Thorlab_to_coherent_factor, 1)
    for p in list_of_uncaging_power
}

for each_file in unc_df["file_path"].unique():
    statedict = unc_df[unc_df["file_path"] == each_file]["statedict"].values[0]
    uncaging_power = statedict["State.Uncaging.Power"]
    fulltimeseries_df.loc[fulltimeseries_df["file_path"] == each_file, "uncaging_power_coherent_mW"] = power_percent_to_coherent_mW_dict[int(uncaging_power)]
    combined_df.loc[combined_df["file_path"] == each_file, "uncaging_power_coherent_mW"] = power_percent_to_coherent_mW_dict[int(uncaging_power)]



#%% align the time_sec based on the uncaging timing

fulltimeseries_df.loc[:,"aligned_time_sec"] = -999.99

summary_df = pd.DataFrame()
for each_group in fulltimeseries_df["group"].unique():
    for each_set_label in fulltimeseries_df[fulltimeseries_df["group"] == each_group]["set_label"].unique():
        group_set_id = f"{each_group}_{each_set_label}"
        # print(f"Processing {group_set_id}")
        each_df = fulltimeseries_df[(fulltimeseries_df["group"] == each_group) & (fulltimeseries_df["set_label"] == each_set_label)]
        # name the each_df with group_set_id
        fulltimeseries_df.loc[each_df.index, "group_set_id"] = group_set_id
        each_df.loc[:, "group_set_id"] = group_set_id

        # uncaging trigger time is 0 seconds
        length_of_unc_df = len(each_df[each_df["phase"] == "unc"])
        if length_of_unc_df in unc_total_frame_first_unc_dict.keys():
            unc_trigger_time = each_df[each_df["phase"] == "unc"]["elapsed_time_sec"].iloc[unc_total_frame_first_unc_dict[length_of_unc_df] - 1]
            aligned_values = each_df["elapsed_time_sec"].values - unc_trigger_time
        else:
            raise ValueError(f"Length of each df is not in unc_total_frame_first_unc_dict: {length_of_unc_df}")        
        each_df.loc[:, "aligned_time_sec"] = aligned_values
        fulltimeseries_df.loc[each_df.index, "aligned_time_sec"] = aligned_values
        
        #normalize the lifetime, by subtracting the mean of the lifetime of frames in pre phase
        pre_phase_lifetime = each_df[each_df["phase"] == "pre"]["Spine_Ch1_lifetime"].mean()
        fulltimeseries_df.loc[each_df.index, "Spine_Ch1_lifetime_normalized"] = fulltimeseries_df.loc[each_df.index, "Spine_Ch1_lifetime"] - pre_phase_lifetime

        #normalize the intensity, by dividing the intensity by the mean of the intensity of frames in pre phase and subtract 1
        pre_phase_intensity = each_df[each_df["phase"] == "pre"]["Spine_Ch1_intensity_div_by_nAve"].mean()
        fulltimeseries_df.loc[each_df.index, "Spine_Ch1_intensity_normalized"] = fulltimeseries_df.loc[each_df.index, "Spine_Ch1_intensity_div_by_nAve"] / pre_phase_intensity - 1
        pre_phase_intensity_ch2 = each_df[each_df["phase"] == "pre"]["Spine_Ch2_intensity_div_by_nAve"].mean()
        fulltimeseries_df.loc[each_df.index, "Spine_Ch2_intensity_normalized"] = fulltimeseries_df.loc[each_df.index, "Spine_Ch2_intensity_div_by_nAve"] / pre_phase_intensity_ch2 - 1




        #transient analysis normalization
        each_unc_df = each_df[each_df["phase"] == "unc"]
        first_uncaging_frame = unc_total_frame_first_unc_dict[len(each_unc_df)]
        for each_ch in ['Ch1', 'Ch2']:
            for each_signal in ['lifetime', 'intensity']:
                first_uncaging_frame = unc_total_frame_first_unc_dict[len(each_unc_df)]
                before_uncaging_during_unc_phase_signal = each_unc_df[each_unc_df["slice"] < first_uncaging_frame][f"Spine_{each_ch}_{each_signal}"].mean()

                if not before_uncaging_during_unc_phase_signal >0 :
                    transient_normalized_signal = np.nan
                else:
                    if each_signal == "lifetime":
                        transient_normalized_signal = each_unc_df[f"Spine_{each_ch}_{each_signal}"] - before_uncaging_during_unc_phase_signal
                        
                    elif each_signal == "intensity":
                        transient_normalized_signal = each_unc_df[f"Spine_{each_ch}_{each_signal}"] / before_uncaging_during_unc_phase_signal

                    
                fulltimeseries_df.loc[each_unc_df.index, f"transient_{each_ch}_{each_signal}"] = transient_normalized_signal


        #summary for each group_set_id
        each_summary_dict = {}
        each_summary_dict["group"] = each_group
        each_summary_dict["set_label"] = each_set_label
        each_summary_dict["group_set_id"] = group_set_id

        # uncaging_power_coherent_mW = each_df[each_df["phase"] == "unc"]["uncaging_power_coherent_mW"]
        # if len(uncaging_power_coherent_mW) != 1:
        #     raise ValueError(f"Length of uncaging power coherent mW is not 1: {len(uncaging_power_coherent_mW)}")
        # each_summary_dict["uncaging_power_coherent_mW"] = uncaging_power_coherent_mW.iloc[0]

        # transient analysis
        for each_ch in ['Ch1', 'Ch2']:
            each_phase = "unc"
            for each_signal in ['lifetime', 'intensity']:
                each_transient_signal_df = fulltimeseries_df.loc[each_unc_df.index, f"transient_{each_ch}_{each_signal}"]
                first_uncaging_frame = unc_total_frame_first_unc_dict[len(each_transient_signal_df)]
                representative_transient_signal = each_transient_signal_df.iloc[first_uncaging_frame]
                each_summary_dict[f"transient_{each_ch}_{each_signal}"] = representative_transient_signal
                # if each_transient_signal_df.max() > 0:
                #     print(each_transient_signal_df)
                #     assert False
                
            each_phase = 'pre'
            for each_signal in ['lifetime', 'intensity']:
                each_summary_dict[f"{each_ch}_{each_phase}_{each_signal}"] = each_df[each_df["phase"] == each_phase][f"Spine_{each_ch}_{each_signal}"].mean()
            each_phase = 'post'
            for each_signal in ['lifetime', 'intensity']:
                post_LTP_data_point_df = each_df[(each_df["phase"] == each_phase) 
                                                & (each_df["aligned_time_sec"] > LTP_data_point_after_min_between[0]*60) 
                                                & (each_df["aligned_time_sec"] < LTP_data_point_after_min_between[1]*60)
                                                & (each_df["group_set_id"] == group_set_id)
                                                ]
                if len(post_LTP_data_point_df) >0:
                    each_summary_dict[f"{each_ch}_{each_phase}_{each_signal}"] = post_LTP_data_point_df[f"Spine_{each_ch}_{each_signal}"].mean()
                    # print("ok1")
                else:
                    post_LTP_data_point_df = each_df[(each_df["phase"] == each_phase) & (each_df["aligned_time_sec"] > LTP_data_point_after_min_between[0]*60)]
                    if len(post_LTP_data_point_df) >0:
                        each_summary_dict[f"{each_ch}_{each_phase}_{each_signal}"] = post_LTP_data_point_df[f"Spine_{each_ch}_{each_signal}"].mean()
                        # print("ok2")
                    else:
                        each_summary_dict[f"{each_ch}_{each_phase}_{each_signal}"] = np.nan
                        print(f"No LTP data point found for {group_set_id} {each_ch} {each_phase} {each_signal}")
        
        each_summary_dict["delta_lifetime_ch1"] = each_summary_dict["Ch1_post_lifetime"] - each_summary_dict["Ch1_pre_lifetime"]
        each_summary_dict["delta_FF0_intensity_ch1"] = each_summary_dict["Ch1_post_intensity"] / each_summary_dict["Ch1_pre_intensity"] - 1
        each_summary_dict["delta_lifetime_ch2"] = each_summary_dict["Ch2_post_lifetime"] - each_summary_dict["Ch2_pre_lifetime"]
        each_summary_dict["delta_FF0_intensity_ch2"] = each_summary_dict["Ch2_post_intensity"] / each_summary_dict["Ch2_pre_intensity"] - 1

        summary_df = pd.concat([summary_df, pd.DataFrame([each_summary_dict])], ignore_index=True)

# %%

#bin the time
pre_phase_sec = float(fulltimeseries_df[fulltimeseries_df["phase"] == "pre"]["aligned_time_sec"].mean())
post_phase_sec = float(fulltimeseries_df[fulltimeseries_df["phase"] == "post"]["aligned_time_sec"].mean())
pre_index = fulltimeseries_df[fulltimeseries_df["phase"] == "pre"].index
post_index = fulltimeseries_df[fulltimeseries_df["phase"] == "post"].index
fulltimeseries_df.loc[pre_index, "binned_time_sec"] = pre_phase_sec
fulltimeseries_df.loc[post_index, "binned_time_sec"] = post_phase_sec

#calc bin time during uncaging, and assign uncaging power
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
    uncaging_power_coherent_mW_list = list(unc_df["uncaging_power_coherent_mW"].unique())
    if len(uncaging_power_coherent_mW_list) != 1:
        raise ValueError(f"Length of uncaging power coherent mW is not 1: {len(uncaging_power_coherent_mW_list)}")
    else:
        uncaging_power_coherent_mW = uncaging_power_coherent_mW_list[0]

    length_of_unc_df = len(unc_df)
    if length_of_unc_df in unc_total_frame_first_unc_dict.keys():
        time_0_nth = unc_total_frame_first_unc_dict[length_of_unc_df] - 1
    else:
        raise ValueError(f"Length of each uncaging df is not in unc_total_frame_first_unc_dict: {length_of_unc_df}")
    time_list = [i*average_bin for i in range(-time_0_nth,len(unc_df)-time_0_nth)]

    fulltimeseries_df.loc[each_df.index, "uncaging_power_coherent_mW"] = uncaging_power_coherent_mW
    summary_df.loc[summary_df["group_set_id"] == each_group_set_id, "uncaging_power_coherent_mW"] = uncaging_power_coherent_mW



# %% line plot
#plot each data, with light thin color lines
swarm_ylim = [summary_df["delta_FF0_intensity_ch2"].min()-0.1, summary_df["delta_FF0_intensity_ch2"].max()+0.1]

plot_info_dict = {
    # "lifetime": {"ylabel": r"$\Delta$lifetime (ns)", "y": "Spine_Ch1_lifetime_normalized", "errorbar": "se"},
    "intensity": {"ylabel": r"$\Delta$spine volume (a.u.)", 
                "xlabel": "Time (sec)",
                "x": "aligned_time_sec",
                "y": "Spine_Ch2_intensity_normalized", 
                "errorbar": "se",
                # "ylim": [-0.4, 5.9]
                "ylim": swarm_ylim
                },
    }

for each_header, each_header_name in group_header_dict.items():
    eachgroup_df = fulltimeseries_df[fulltimeseries_df["group"].str.contains(each_header)]
    if len(eachgroup_df) == 0:
        continue

    for each_uncaging_power_coherent_mW in fulltimeseries_df["uncaging_power_coherent_mW"].unique():
        each_group_same_unc_pow_df = eachgroup_df[eachgroup_df["uncaging_power_coherent_mW"] == each_uncaging_power_coherent_mW]

        plot_df = each_group_same_unc_pow_df

        for each_plot_type, each_plot_info in plot_info_dict.items():
            plt.figure(figsize=(5, 3))
            g = sns.lineplot(
                        x = each_plot_info["x"],
                        y = each_plot_info["y"],
                        data = plot_df,
                        hue = "group_set_id",
                        linewidth = 0.5,
                        alpha = 0.5,
                        palette = "tab10",
                        legend = False,
                        marker = "o",
                        markersize = 4,
                        )
            #greek delta lifetime
            plt.ylabel(each_plot_info["ylabel"])
            plt.xlabel(each_plot_info["xlabel"])
            plt.title(each_header_name+ f", uncaging {each_uncaging_power_coherent_mW} mW")
            plt.ylim(each_plot_info["ylim"])
            # #plot mean with SEM

            ylim = plt.gca().get_ylim()
            ninty_percent_ylim = ylim[0] + (ylim[1] - ylim[0]) * 0.9
            plt.plot([0, 2.048*29], [ninty_percent_ylim, ninty_percent_ylim], "k-")
            plt.text(0, ninty_percent_ylim*1.005, "uncaging", ha="left", va="bottom")

            plt.fill_between(np.array(LTP_data_point_after_min_between)*60,
            each_plot_info["ylim"][0], each_plot_info["ylim"][1], 
            color="pink", 
            alpha=0.3)
            
            #delete right and top border
            plt.gca().spines["top"].set_visible(False)
            plt.gca().spines["right"].set_visible(False)
            savepath = os.path.join(save_folder, f"{each_header}_{each_plot_type}_{each_uncaging_power_coherent_mW}mW_lineplot_time_series.png")
            plt.savefig(savepath, dpi=150, bbox_inches = "tight")
            plt.show()

# %% swarm plot
swarm_ylim = [summary_df["delta_FF0_intensity_ch2"].min()-0.1, summary_df["delta_FF0_intensity_ch2"].max()+0.1]
plot_info_dict = {
    # "lifetime": {"ylabel": r"$\Delta$lifetime (ns)",
    #                             "y": "delta_lifetime",
    #                             "ylim" : [-0.19, 0.29]},
                  "intensity": {"ylabel": r"$\Delta$spine volume (a.u.)", 
                                "y": "delta_FF0_intensity_ch2", 
                                "ylim" : swarm_ylim}
                }

for each_header, each_header_name in group_header_dict.items():
    each_header_summary_df = summary_df[summary_df["group"].str.contains(each_header)]
    if len(each_header_summary_df) == 0:
        continue
    for each_uncaging_power_coherent_mW in fulltimeseries_df["uncaging_power_coherent_mW"].unique():
        each_group_same_unc_pow_df = each_header_summary_df[each_header_summary_df["uncaging_power_coherent_mW"] == each_uncaging_power_coherent_mW]
        plot_df = each_group_same_unc_pow_df
        for each_plot_type, each_plot_info in plot_info_dict.items():
            plt.figure(figsize=(2, 3))
            p = sns.swarmplot(y=each_plot_info["y"],
                        data=plot_df,
                        palette = "tab10",
                        )

            sns.boxplot(showmeans=True,
                meanline=True,
                meanprops={'color': 'r', 'ls': '-', 'lw': 1},
                medianprops={'visible': False},
                whiskerprops={'visible': False},
                zorder=10,
                y=each_plot_info["y"],
                data=plot_df,
                showfliers=False,
                showbox=False,
                showcaps=False,
                ax=p)

            mean = plot_df[each_plot_info["y"]].mean()
            std = plot_df[each_plot_info["y"]].std()
            plt.text(0.2, mean, f"{mean:.2f} ± {std:.2f}", ha="left", va="bottom")
            
            plt.ylabel(each_plot_info["ylabel"])
            plt.ylim(each_plot_info["ylim"])
            plt.title(each_header_name+ f", uncaging {each_uncaging_power_coherent_mW} mW")
            #delete right and top border
            plt.gca().spines["top"].set_visible(False)
            plt.gca().spines["right"].set_visible(False)
            savepath = os.path.join(save_folder, f"{each_header}_{each_plot_type}_{each_uncaging_power_coherent_mW}mW_plot_swarmplot.png")
            plt.savefig(savepath, dpi=150, bbox_inches = "tight")
            plt.show()


# %%
# %% line plot, transient
#plot each data, with light thin color lines
plot_info_dict = {
    "GCaMP_transient_intensity": {"ylabel": r"GCaMP F/F0", 
                            "xlabel": "Time (sec)",
                            "x": "aligned_time_sec",
                            "y": "transient_Ch1_intensity", 
                            "errorbar": "se",
                            "ylim": [-0.4, 25]}
    }

for each_header, each_header_name in group_header_dict.items():
    eachgroup_df = fulltimeseries_df[fulltimeseries_df["group"].str.contains(each_header)]
    if len(eachgroup_df) == 0:
        continue

    for each_uncaging_power_coherent_mW in fulltimeseries_df["uncaging_power_coherent_mW"].unique():
        each_group_same_unc_pow_df = fulltimeseries_df[fulltimeseries_df["uncaging_power_coherent_mW"] == each_uncaging_power_coherent_mW]
        plot_df = each_group_same_unc_pow_df[each_group_same_unc_pow_df["phase"] == "unc"]

        for each_plot_type, each_plot_info in plot_info_dict.items():
            plt.figure(figsize=(5, 3))
            g = sns.lineplot(
                        x = each_plot_info["x"],
                        y = each_plot_info["y"],
                        data = plot_df,
                        hue = "group_set_id",
                        linewidth = 0.5,
                        alpha = 0.5,
                        palette = "tab10",
                        legend = False,
                        marker = "o",
                        markersize = 4,
                        )
            #greek delta lifetime
            plt.ylabel(each_plot_info["ylabel"])
            plt.xlabel(each_plot_info["xlabel"])
            plt.title(each_header_name+ f", uncaging {each_uncaging_power_coherent_mW} mW")
            plt.ylim(each_plot_info["ylim"])
            # #plot mean with SEM

            ylim = plt.gca().get_ylim()
            ninty_percent_ylim = ylim[0] + (ylim[1] - ylim[0]) * 0.9
            plt.plot([0, 2.048*29], [ninty_percent_ylim, ninty_percent_ylim], "k-")
            plt.text(0, ninty_percent_ylim*1.005, "uncaging", ha="left", va="bottom")

            # plt.fill_between(np.array(LTP_data_point_after_min_between)*60,
            # each_plot_info["ylim"][0], each_plot_info["ylim"][1], 
            # color="pink", 
            # alpha=0.3)
            
            #delete right and top border
            plt.gca().spines["top"].set_visible(False)
            plt.gca().spines["right"].set_visible(False)
            savepath = os.path.join(save_folder, f"{each_header}_{each_plot_type}_{each_uncaging_power_coherent_mW}mW_lineplot_transient_time_series.png")
            plt.savefig(savepath, dpi=150, bbox_inches = "tight")
            plt.show()


# %%
# %% swarm plot for transient
swarm_ylim = [summary_df["transient_Ch1_intensity"].min()-0.1, summary_df["transient_Ch1_intensity"].max()+0.1]
plot_info_dict = {
    "GCaMP_transient_intensity": {"ylabel": r"GCaMP F/F0", 
                            "xlabel": "Time (sec)",
                            "x": "aligned_time_sec",
                            "y": "transient_Ch1_intensity", 
                            "errorbar": "se",
                            "ylim": swarm_ylim}
                }

for each_header, each_header_name in group_header_dict.items():
    each_header_summary_df = summary_df[summary_df["group"].str.contains(each_header)]
    if len(each_header_summary_df) == 0:
        continue
    for each_uncaging_power_coherent_mW in fulltimeseries_df["uncaging_power_coherent_mW"].unique():
        each_group_same_unc_pow_df = each_header_summary_df[each_header_summary_df["uncaging_power_coherent_mW"] == each_uncaging_power_coherent_mW]
        plot_df = each_group_same_unc_pow_df
        for each_plot_type, each_plot_info in plot_info_dict.items():
            plt.figure(figsize=(2, 3))
            p = sns.swarmplot(y=each_plot_info["y"],
                        data=plot_df,
                        palette = "tab10",
                        )

            sns.boxplot(showmeans=True,
                meanline=True,
                meanprops={'color': 'r', 'ls': '-', 'lw': 1},
                medianprops={'visible': False},
                whiskerprops={'visible': False},
                zorder=10,
                y=each_plot_info["y"],
                data=plot_df,
                showfliers=False,
                showbox=False,
                showcaps=False,
                ax=p)

            mean = plot_df[each_plot_info["y"]].mean()
            std = plot_df[each_plot_info["y"]].std()
            plt.text(0.2, mean, f"{mean:.2f} ± {std:.2f}", ha="left", va="bottom")
            
            plt.ylabel(each_plot_info["ylabel"])
            plt.ylim(each_plot_info["ylim"])
            plt.title(each_header_name+ f", uncaging {each_uncaging_power_coherent_mW} mW")
            #delete right and top border
            plt.gca().spines["top"].set_visible(False)
            plt.gca().spines["right"].set_visible(False)
            savepath = os.path.join(save_folder, f"{each_header}_{each_plot_type}_{each_uncaging_power_coherent_mW}mW_plot_transient_swarmplot.png")
            plt.savefig(savepath, dpi=150, bbox_inches = "tight")
            plt.show()


# %%
print("plots were saved to:")
print(save_folder)
# %%
