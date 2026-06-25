# %%
import os
import sys
import datetime
sys.path.append(r"..\..")
import pandas as pd
import seaborn as sns
from custom_plot import plt

photon_threshold_for_intensity = 30
photon_threshold_for_lifetime = 1000
Ab_added_datetime_for_GC6s = datetime.datetime(2026, 1, 22, 22, 47, 0, 0)
Ab_time_delta_hour_for_camui = 28/60

pkl_path = r"Z:\User-Personal\Tetsuya_Zdrive\Data\202601\20260122\auto1\transient_combined_df.pkl"
csv_path = r"Z:\User-Personal\Tetsuya_Zdrive\Data\202601\20260122\auto1\transient_combined_df_full_timeseries.csv"

combined_df = pd.read_pickle(pkl_path)
fulltimeseries_df = pd.read_csv(csv_path)

#reanalyze time information
#start of the experiment, the time I soak the slice and get widefiled image
#get time info from that file's saved datetime using os 
first_timepoint_filepath = r"Z:\User-Personal\Tetsuya_Zdrive\Data\202601\20260122\20260122_193853.tif"
first_timepoint_datetime = datetime.datetime.fromtimestamp(os.path.getmtime(first_timepoint_filepath))

combined_df["dt"] = pd.to_datetime(combined_df["dt_str"])
combined_df["relative_time_sec"] = (combined_df["dt"] - first_timepoint_datetime).dt.total_seconds()
combined_df["relative_time_min"] = combined_df["relative_time_sec"] / 60
combined_df["relative_time_hour"] = combined_df["relative_time_min"] / 60


relative_time_sec_Ab_added = (Ab_added_datetime - first_timepoint_datetime).total_seconds()
relative_time_min_Ab_added = relative_time_sec_Ab_added / 60
relative_time_hour_Ab_added = relative_time_min_Ab_added / 60



save_folder = os.path.join(os.path.dirname(pkl_path), "summary")
os.makedirs(save_folder, exist_ok=True)
# %%
#camui_df is the dataframe of the camui data, if group contains "camui", then it is the camui data
camui_df = fulltimeseries_df[fulltimeseries_df["group"].str.contains("Camui")]


camui_rejected_df = camui_df[camui_df["Spine_Ch1_total_photon"] < photon_threshold_for_lifetime]

camui_rejected_file_path_list = camui_rejected_df["file_path"].unique()

camui_df_photon_filtered = camui_df[~camui_df["file_path"].isin(camui_rejected_file_path_list)].copy()

#normalize the camui lifetime, by subtracting the mean of the lifetime of the first 2 frames for each file_path
camui_df_photon_filtered.loc[:,"Spine_Ch1_lifetime_normalized"] = camui_df_photon_filtered.groupby("file_path")["Spine_Ch1_lifetime"].transform(lambda x: x - x.iloc[0:2].mean())



for each_file_path in camui_df_photon_filtered["file_path"].unique():
    each_df = camui_df_photon_filtered[camui_df_photon_filtered["file_path"] == each_file_path]
    normalized_mean = each_df[each_df["time_sec"].between(13, 23)]["Spine_Ch1_lifetime_normalized"].mean()
    #update the combined_df based on camui_df_photon_filtered
    combined_df.loc[combined_df["file_path"] == each_file_path, "Spine_Ch1_lifetime_normalized_mean"] = normalized_mean

#reject<- True in combined_df if photon threshold is not met
for each_file_path in camui_df["file_path"].unique():
    # each_df = GC6s_df[GC6s_df["file_path"] == each_file_path]
    if camui_df.loc[camui_df["file_path"] == each_file_path, "Spine_Ch1_total_photon"].min() < photon_threshold_for_lifetime:
        combined_df.loc[combined_df["file_path"] == each_file_path, "reject"] = True





GC6s_df = fulltimeseries_df[fulltimeseries_df["group"].str.contains("GC6s")]


GC6s_rejected_df = GC6s_df[GC6s_df["Spine_Ch1_total_photon"] < photon_threshold_for_intensity]

GC6s_rejected_file_path_list = GC6s_rejected_df["file_path"].unique()

GC6s_df_photon_filtered = GC6s_df[~GC6s_df["file_path"].isin(GC6s_rejected_file_path_list)].copy()

#normalize the GC6s intensity, by the mean of the intensity of the first 2 frames for each file_path
GC6s_df_photon_filtered.loc[:,"Spine_Ch1_intensity_normalized"] = GC6s_df_photon_filtered.groupby("file_path")["Spine_Ch1_intensity"].transform(lambda x: x / x.iloc[0:2].mean())

#get mean of normalized intensity for each file_path
#calculate that mean from time between 13 to 25 seconds, which means between 5th to 10th uncaging

for each_file_path in GC6s_df_photon_filtered["file_path"].unique():
    each_df = GC6s_df_photon_filtered[GC6s_df_photon_filtered["file_path"] == each_file_path]
    normalized_mean = each_df[each_df["time_sec"].between(13, 23)]["Spine_Ch1_intensity_normalized"].mean()
    #update the combined_df based on GC6s_df_photon_filtered
    combined_df.loc[combined_df["file_path"] == each_file_path, "Spine_Ch1_intensity_normalized_mean"] = normalized_mean


#reject<- True in combined_df if photon threshold is not met
for each_file_path in GC6s_df["file_path"].unique():
    # each_df = GC6s_df[GC6s_df["file_path"] == each_file_path]
    if GC6s_df.loc[GC6s_df["file_path"] == each_file_path, "Spine_Ch1_total_photon"].min() < photon_threshold_for_intensity:
        combined_df.loc[combined_df["file_path"] == each_file_path, "reject"] = True


#save the combined_df
combined_df.to_csv(os.path.join(save_folder, "combined_df_with_normalized_data.csv"))
camui_df_photon_filtered.to_csv(os.path.join(save_folder, "camui_df_photon_filtered_with_normalized_data.csv"))
GC6s_df_photon_filtered.to_csv(os.path.join(save_folder, "GC6s_df_photon_filtered_with_normalized_data.csv"))


# %%
#plot each data, with light thin color lines
plt.figure(figsize=(5, 3))
sns.lineplot(x = "time_sec", y = "Spine_Ch1_lifetime_normalized", 
            data = camui_df_photon_filtered, 
            hue = "file_path", 
            linewidth = 0.5, 
            alpha = 0.5,
            palette = "tab10",
            )
#greek delta lifetime
plt.ylabel(r"$\Delta$ Lifetime (ns)")
plt.xlabel("Time (sec)")
plt.title("mStayGold-Camui")
#delete the legend

#plot mean with SEM
sns.lineplot(x = "time_sec", y = "Spine_Ch1_lifetime_normalized",
            data = camui_df_photon_filtered, errorbar = "se",
            linewidth = 2,
            color = "r",
            )
#get current ylim
ylim = plt.gca().get_ylim()
ninty_percent_ylim = ylim[0] + (ylim[1] - ylim[0]) * 0.9
plt.plot([4.096, 4.096+2.048*29], [ninty_percent_ylim, ninty_percent_ylim], "k-")
plt.text(4.096+2.048*2, ninty_percent_ylim*1.005, "uncaging", ha="left", va="bottom")
#delete the legend
plt.gca().get_legend().remove()
#delete right and top border
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
savepath = os.path.join(save_folder, "Camui_transient_plot_all_data.png")
plt.savefig(savepath, dpi=150, bbox_inches = "tight")
plt.show()

# %%
#swarm plot
plt.figure(figsize=(2, 3))
p = sns.swarmplot(y="Spine_Ch1_lifetime_normalized_mean",
            data=combined_df,
            palette = "tab10",
            )
#show mean using mean line
sns.boxplot(showmeans=True,
            meanline=True,
            meanprops={'color': 'r', 'ls': '-', 'lw': 1},
            medianprops={'visible': False},
            whiskerprops={'visible': False},
            zorder=10,
            y="Spine_Ch1_lifetime_normalized_mean",
            data=combined_df,
            showfliers=False,
            showbox=False,
            showcaps=False,
            ax=p)
plt.ylabel(r"$\Delta$ Lifetime (ns)")

plt.title("mStayGold-Camui")
#delete right and top border
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
savepath = os.path.join(save_folder, "Camui_transient_plot_swarmplot.png")
plt.savefig(savepath, dpi=150, bbox_inches = "tight")
plt.show()


# %% plot GC6s
plt.figure(figsize=(5, 3))
sns.lineplot(x = "time_sec", y = "Spine_Ch1_intensity_normalized", 
            data = GC6s_df_photon_filtered, 
            hue = "file_path", 
            linewidth = 0.5, 
            alpha = 0.5,
            palette = "tab10",
            )
plt.ylabel("F/F0")
plt.xlabel("Time (sec)")
plt.title("GCaMP6s")
#delete the legend

#plot mean with SEM
sns.lineplot(x = "time_sec", y = "Spine_Ch1_intensity_normalized",
            data = GC6s_df_photon_filtered, errorbar = "se",
            linewidth = 2,
            color = "r",
            )
#get current ylim
ylim = plt.gca().get_ylim()
ninty_percent_ylim = ylim[0] + (ylim[1] - ylim[0]) * 0.9
plt.plot([4.096, 4.096+2.048*29], [ninty_percent_ylim, ninty_percent_ylim], "k-")
plt.text(4.096+2.048*2, ninty_percent_ylim*1.005, "uncaging", ha="left", va="bottom")
#delete the legend
plt.gca().get_legend().remove()
#delete right and top border
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
savepath = os.path.join(save_folder, "GC6s_transient_plot_all_data.png")
plt.savefig(savepath, dpi=150, bbox_inches = "tight")
plt.show()

# %% swarm plot
plt.figure(figsize=(2, 3))
p = sns.swarmplot(y="Spine_Ch1_intensity_normalized_mean",
            data=combined_df,
            palette = "tab10",
            )
#show mean using mean line
sns.boxplot(showmeans=True,
            meanline=True,
            meanprops={'color': 'r', 'ls': '-', 'lw': 1},
            medianprops={'visible': False},
            whiskerprops={'visible': False},
            zorder=10,
            y="Spine_Ch1_intensity_normalized_mean",
            data=combined_df,
            showfliers=False,
            showbox=False,
            showcaps=False,
            ax=p)
plt.ylabel("F/F0")
plt.title("GCaMP6s")
#delete right and top border
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
savepath = os.path.join(save_folder, "GC6s_transient_plot_swarmplot.png")
plt.savefig(savepath, dpi=150, bbox_inches = "tight")
plt.show()
# %%　analyze the signal change vs time after experiment start

camui_combined_df = combined_df[combined_df["group"].str.contains("Camui")]
GC6s_combined_df = combined_df[combined_df["group"].str.contains("GC6s")]

#scatter plot the signal change vs time after experiment start
plt.figure(figsize=(5, 3))
sns.scatterplot(x = "relative_time_hour", 
            y = "Spine_Ch1_lifetime_normalized_mean",
            data = camui_combined_df,
            hue = "file_path",
            palette = "tab10",
            )
plt.ylabel(r"$\Delta$ Lifetime (ns)")
plt.xlabel("Time (min) after started experiment")
plt.title("mStayGold-Camui")
ylim = plt.gca().get_ylim()
xlim = plt.gca().get_xlim()
ninty_percent_ylim = ylim[0] + (ylim[1] - ylim[0]) * 1.05
plt.plot([relative_time_hour_Ab_added + Ab_time_delta_hour_for_camui, xlim[1]], 
        [ninty_percent_ylim, ninty_percent_ylim], "k-")
    
#A beta 1 micro molar
plt.text(relative_time_hour_Ab_added*1.005 + Ab_time_delta_hour_for_camui, 
        ninty_percent_ylim*1.005, 
        r"A$\beta$ 1 $\mu$M", 
        ha="left", 
        va="bottom")

#delete the legend
plt.gca().get_legend().remove()
#delete right and top border
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)

savepath = os.path.join(save_folder,
                    "Camui_lifetime_against_time.png")
plt.savefig(savepath, dpi=150, bbox_inches = "tight")
plt.show()

# %%
#scatter plot the signal change vs time after experiment start

plt.figure(figsize=(5, 3))
sns.scatterplot(x = "relative_time_hour", 
            y = "Spine_Ch1_intensity_normalized_mean",
            data = GC6s_combined_df,
            hue = "file_path",
            palette = "tab10",
            )
plt.ylabel("F/F0")
plt.xlabel("Time (hour) after started experiment")
plt.title("GCaMP6s")
# plt.title("Signal change vs time after started experiment")
#delete the legend
ylim = plt.gca().get_ylim()
xlim = plt.gca().get_xlim()
ninty_percent_ylim = ylim[0] + (ylim[1] - ylim[0]) * 1.05
plt.plot([relative_time_hour_Ab_added, xlim[1]], 
        [ninty_percent_ylim, ninty_percent_ylim], "k-")
    
#A beta 1 micro molar
plt.text(relative_time_hour_Ab_added*1.005, 
        ninty_percent_ylim*1.005, 
        r"A$\beta$ 1 $\mu$M", 
        ha="left", 
        va="bottom")


plt.gca().get_legend().remove()
#delete right and top border
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
#short name please
savepath = os.path.join(save_folder,
                    "GC6s_intensity_against_time.png")
plt.savefig(savepath, dpi=150, bbox_inches = "tight")
plt.show()
# %%
print("summary for number of data points, included and excluded")

total_camui = len(combined_df[combined_df["group"].str.contains("Camui")])
total_GC6s = len(combined_df[combined_df["group"].str.contains("GC6s")])
included_camui = len(combined_df[combined_df["group"].str.contains("Camui") & ~combined_df["reject"]])
included_GC6s = len(combined_df[combined_df["group"].str.contains("GC6s") & ~combined_df["reject"]])
excluded_camui = len(combined_df[combined_df["group"].str.contains("Camui") & combined_df["reject"]])
excluded_GC6s = len(combined_df[combined_df["group"].str.contains("GC6s") & combined_df["reject"]])

#print table
print("| Group | Total | Included | Excluded |")
print("|-------|-------|--------------|----------|")
print(f"| Camui | {total_camui} | {included_camui} | {excluded_camui} |")
print(f"| GC6s | {total_GC6s} | {included_GC6s} | {excluded_GC6s} |")

print(f"mStayGold-camui: N = {included_camui} spines")
print(f"GCaMP6s: N = {included_GC6s} spines")
# %%


