# %%
import os
import sys
import json
import glob
import datetime
import pandas as pd
import numpy as np
import matplotlib
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial"]
sys.path.append(r"..\..")
from custom_plot import plt
import seaborn as sns
from scipy import stats

LTP_data_point_after_min_between = [25, 35]
unc_total_frame_first_unc_dict = {33: 2, 55: 5}
ROI_name_list = ["Spine", "DendriticShaft"]
powermeter_folder = r"//RY-LAB-WS04/Users/yasudalab/Documents/Tetsuya_Imaging/powermeter"
from_Thorlab_to_coherent_factor = 1 / 3

dataset_list = [
    (r"//RY-LAB-WS04/ImagingData/Tetsuya/20260318/auto1/combined_df_1.pkl",
     r"//RY-LAB-WS04/ImagingData/Tetsuya/20260318/auto1/combined_df_1_intensity_lifetime_all_frames.csv"),
    (r"//RY-LAB-WS04/ImagingData/Tetsuya/20260324/auto1/combined_df_1.pkl",
     r"//RY-LAB-WS04/ImagingData/Tetsuya/20260324/auto1/combined_df_1_intensity_lifetime_all_frames.csv"),
    (r"//RY-LAB-WS04/ImagingData/Tetsuya/20260326/cytiva_woKA/combined_df_1.pkl",
     r"//RY-LAB-WS04/ImagingData/Tetsuya/20260326/cytiva_woKA/combined_df_1_intensity_lifetime_all_frames.csv"),
    (r"//RY-LAB-WS04/ImagingData/Tetsuya/20260326/Gibco_woKA/combined_df_1.pkl",
     r"//RY-LAB-WS04/ImagingData/Tetsuya/20260326/Gibco_woKA/combined_df_1_intensity_lifetime_all_frames.csv"),
    (r"//RY-LAB-WS04/ImagingData/Tetsuya/20260326/CM_withKA/combined_df_1.pkl",
     r"//RY-LAB-WS04/ImagingData/Tetsuya/20260326/CM_withKA/combined_df_1_intensity_lifetime_all_frames.csv"),
    (r"//RY-LAB-WS04/ImagingData/Tetsuya/20260326/CM_woKA/combined_df_1.pkl",
     r"//RY-LAB-WS04/ImagingData/Tetsuya/20260326/CM_woKA/combined_df_1_intensity_lifetime_all_frames.csv"),
    (r"//RY-LAB-WS04/ImagingData/Tetsuya/20260327/0319young/auto1/combined_df_1.pkl",
     r"//RY-LAB-WS04/ImagingData/Tetsuya/20260327/0319young/auto1/combined_df_1_intensity_lifetime_all_frames.csv"),
    (r"//RY-LAB-WS04/ImagingData/Tetsuya/20260327/0310/auto1/combined_df_1.pkl",
     r"//RY-LAB-WS04/ImagingData/Tetsuya/20260327/0310/auto1/combined_df_1_intensity_lifetime_all_frames.csv"),
]

save_folder = r"//RY-LAB-WS04/ImagingData/Tetsuya/20260328/combine_recent"

# %% load and process each dataset
all_summary_df = pd.DataFrame()

for df_save_path_1, out_csv_path in dataset_list:
    if not os.path.exists(df_save_path_1):
        print(f"Skipping (not found): {df_save_path_1}")
        continue

    dataset_label = os.path.basename(os.path.dirname(df_save_path_1))
    _parts = df_save_path_1.replace("\\", "/").split("/")
    date_label = _parts[_parts.index("Tetsuya") + 1] if "Tetsuya" in _parts else dataset_label

    combined_df = pd.read_pickle(df_save_path_1)
    fulltimeseries_df = pd.read_csv(out_csv_path)

    for each_ROI_name in ROI_name_list:
        for each_ch in ['Ch1', 'Ch2']:
            fulltimeseries_df[f"{each_ROI_name}_{each_ch}_intensity_div_by_nAve"] = (
                fulltimeseries_df[f"{each_ROI_name}_{each_ch}_intensity"] / fulltimeseries_df["nAveFrame"]
            )

    # build power % -> coherent mW dict using the powermeter JSON closest before earliest acq time
    unc_combined = combined_df[combined_df["phase"] == "unc"]
    fulltimeseries_df["uncaging_power"] = np.nan
    for each_file in unc_combined["file_path"].unique():
        statedict = unc_combined[unc_combined["file_path"] == each_file]["statedict"].values[0]
        unc_power = int(statedict["State.Uncaging.Power"])
        fulltimeseries_df.loc[fulltimeseries_df["file_path"] == each_file, "uncaging_power"] = unc_power

    earliest_acq_time = datetime.datetime.strptime(
        fulltimeseries_df["acq_time_str"].min(), "%Y-%m-%dT%H:%M:%S.%f"
    )
    powermeter_ini_files = (
        glob.glob(os.path.join(powermeter_folder, "*.json"))
        + glob.glob(os.path.join(powermeter_folder, "old/*.json"))
    )
    datetime_only_arr = np.array(
        [int(os.path.basename(f).replace(".json", "")) for f in powermeter_ini_files]
    )
    earliest_int = int(earliest_acq_time.strftime("%Y%m%d%H%M"))
    candidates = datetime_only_arr[datetime_only_arr < earliest_int]
    if len(candidates) == 0:
        print(f"  WARNING: No powermeter JSON found before {earliest_acq_time} for {dataset_label}. Skipping mW conversion.")
        fulltimeseries_df["uncaging_power_coherent_mW"] = np.nan
    else:
        latest_json_basename = f"{candidates.max()}.json"
        latest_json_path = [f for f in powermeter_ini_files if os.path.basename(f) == latest_json_basename][0]
        with open(latest_json_path, "r") as _f:
            powermeter_calib = json.load(_f)
        x_percent = np.array(list(powermeter_calib["Laser2"].keys())).astype(float)
        y_mW = np.array(list(powermeter_calib["Laser2"].values())).astype(float)
        power_slope, power_intercept = np.polyfit(x_percent, y_mW, 1)
        list_of_uncaging_power = list(fulltimeseries_df["uncaging_power"].dropna().unique())
        power_percent_to_coherent_mW_dict = {
            int(p): round(float(power_slope * p + power_intercept) * from_Thorlab_to_coherent_factor, 1)
            for p in list_of_uncaging_power
        }
        fulltimeseries_df["uncaging_power_coherent_mW"] = np.nan
        for each_file in unc_combined["file_path"].unique():
            statedict = unc_combined[unc_combined["file_path"] == each_file]["statedict"].values[0]
            unc_power_pct = int(statedict["State.Uncaging.Power"])
            if unc_power_pct in power_percent_to_coherent_mW_dict:
                fulltimeseries_df.loc[
                    fulltimeseries_df["file_path"] == each_file, "uncaging_power_coherent_mW"
                ] = power_percent_to_coherent_mW_dict[unc_power_pct]
        print(f"  Powermeter JSON: {latest_json_basename}, power map: {power_percent_to_coherent_mW_dict}")

    summary_df = pd.DataFrame()
    for each_group in fulltimeseries_df["group"].unique():
        for each_set_label in fulltimeseries_df[fulltimeseries_df["group"] == each_group]["set_label"].unique():
            group_set_id = f"{each_group}_{each_set_label}"
            each_df = fulltimeseries_df[
                (fulltimeseries_df["group"] == each_group) &
                (fulltimeseries_df["set_label"] == each_set_label)
            ].copy()

            length_of_unc_df = len(each_df[each_df["phase"] == "unc"])
            if length_of_unc_df not in unc_total_frame_first_unc_dict:
                print(f"  Skipping {group_set_id}: unexpected unc length {length_of_unc_df}")
                continue

            unc_trigger_time = each_df[each_df["phase"] == "unc"]["elapsed_time_sec"].iloc[
                unc_total_frame_first_unc_dict[length_of_unc_df] - 1
            ]
            each_df["aligned_time_sec"] = each_df["elapsed_time_sec"].values - unc_trigger_time

            ch2_pre_mean = each_df[each_df["phase"] == "pre"]["Spine_Ch2_intensity"].mean()
            if not ch2_pre_mean > 0:
                continue

            post_window_df = each_df[
                (each_df["phase"] == "post") &
                (each_df["aligned_time_sec"] > LTP_data_point_after_min_between[0] * 60) &
                (each_df["aligned_time_sec"] < LTP_data_point_after_min_between[1] * 60)
            ]
            if len(post_window_df) == 0:
                post_window_df = each_df[
                    (each_df["phase"] == "post") &
                    (each_df["aligned_time_sec"] > LTP_data_point_after_min_between[0] * 60)
                ]
            if len(post_window_df) == 0:
                print(f"  No post data for {group_set_id}")
                continue

            delta_FF0 = post_window_df["Spine_Ch2_intensity"].mean() / ch2_pre_mean - 1

            first_post_df = each_df[each_df["phase"] == "post"].sort_values("aligned_time_sec")
            if len(first_post_df) == 0:
                continue
            first_post_FF0 = first_post_df["Spine_Ch2_intensity"].iloc[0] / ch2_pre_mean - 1
            first_post_time_min = first_post_df["aligned_time_sec"].iloc[0] / 60

            unc_power_vals = each_df[each_df["phase"] == "unc"]["uncaging_power"].dropna().unique()
            unc_power = int(unc_power_vals[0]) if len(unc_power_vals) > 0 else np.nan
            unc_mW_vals = each_df[each_df["phase"] == "unc"]["uncaging_power_coherent_mW"].dropna().unique()
            unc_mW = float(unc_mW_vals[0]) if len(unc_mW_vals) > 0 else np.nan

            # GCaMP transient (Ch1): normalize by pre-trigger frames within unc phase,
            # then take the value at the trigger frame
            first_unc_frame_idx = unc_total_frame_first_unc_dict[length_of_unc_df]
            each_unc_df_s = each_df[each_df["phase"] == "unc"]
            transient_spine_ch1 = np.nan
            transient_shaft_ch1 = np.nan
            for _roi, _key in [("Spine", "transient_spine_ch1"),
                                ("DendriticShaft", "transient_shaft_ch1")]:
                _col = f"{_roi}_Ch1_intensity"
                if _col not in each_unc_df_s.columns:
                    continue
                _before = each_unc_df_s[each_unc_df_s["slice"] < first_unc_frame_idx][_col]
                _before_mean = _before.mean() if len(_before) > 0 else np.nan
                if _before_mean and _before_mean > 0:
                    _val = each_unc_df_s[_col].iloc[first_unc_frame_idx] / _before_mean
                    if _key == "transient_spine_ch1":
                        transient_spine_ch1 = _val
                    else:
                        transient_shaft_ch1 = _val

            summary_df = pd.concat([summary_df, pd.DataFrame([{
                "group_set_id": f"{dataset_label}_{group_set_id}",
                "dataset": dataset_label,
                "date": date_label,
                "group": each_group,
                "uncaging_power": unc_power,
                "uncaging_power_coherent_mW": unc_mW,
                "delta_FF0_intensity_ch2": delta_FF0,
                "first_post_FF0_intensity_ch2": first_post_FF0,
                "first_post_time_min": first_post_time_min,
                "transient_Spine_Ch1_intensity": transient_spine_ch1,
                "transient_DendriticShaft_Ch1_intensity": transient_shaft_ch1,
            }])], ignore_index=True)

    all_summary_df = pd.concat([all_summary_df, summary_df], ignore_index=True)
    print(f"Loaded {dataset_label}: {len(summary_df)} sets")

print(f"\nTotal: {len(all_summary_df)} sets")

# build x-axis label for 1st post frame with actual time range from data
_t_min = int(np.floor(all_summary_df["first_post_time_min"].min()))
_t_max = int(np.ceil(all_summary_df["first_post_time_min"].max()))
first_post_xlabel = (
    r"$\Delta$spine volume (a.u.) "
    f"[{_t_min}–{_t_max} min]"
)

# %% scatter plot
fig, ax = plt.subplots(figsize=(5, 3.5))

# Round to nearest 0.2 mW to group close power values together
round_to = 0.2
all_summary_df["unc_mW_grouped"] = (
    (all_summary_df["uncaging_power_coherent_mW"] / round_to).round() * round_to
)

unique_mW = sorted(all_summary_df["unc_mW_grouped"].dropna().unique())
norm = matplotlib.colors.Normalize(vmin=min(unique_mW), vmax=max(unique_mW))
cmap = matplotlib.cm.coolwarm

for mW_val in unique_mW:
    subset = all_summary_df[all_summary_df["unc_mW_grouped"] == mW_val]
    ax.scatter(
        subset["first_post_FF0_intensity_ch2"],
        subset["delta_FF0_intensity_ch2"],
        color=cmap(norm(mW_val)),
        label=f"{mW_val:.1f}",
        s=20,
        zorder=3,
    )

ax.legend(title="Unc. power (mW)", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)

x_range = all_summary_df["first_post_FF0_intensity_ch2"].max() - all_summary_df["first_post_FF0_intensity_ch2"].min()
y_range = all_summary_df["delta_FF0_intensity_ch2"].max() - all_summary_df["delta_FF0_intensity_ch2"].min()
xlim = [all_summary_df["first_post_FF0_intensity_ch2"].min() - x_range * 0.1,
        all_summary_df["first_post_FF0_intensity_ch2"].max() + x_range * 0.1]
ylim = [all_summary_df["delta_FF0_intensity_ch2"].min() - y_range * 0.1,
        all_summary_df["delta_FF0_intensity_ch2"].max() + y_range * 0.1]

ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.plot([xlim[0], xlim[1]], [0, 0], "--", color="gray", linewidth=0.5)
ax.plot([0, 0], [ylim[0], ylim[1]], "--", color="gray", linewidth=0.5)

# trend line with p-value
plot_data = all_summary_df[["first_post_FF0_intensity_ch2", "delta_FF0_intensity_ch2"]].dropna()
slope, intercept, r_value, p_value, _ = stats.linregress(
    plot_data["first_post_FF0_intensity_ch2"], plot_data["delta_FF0_intensity_ch2"]
)
x_fit = np.array([xlim[0], xlim[1]])
ax.plot(x_fit, slope * x_fit + intercept, "-", color="black", linewidth=1, zorder=2)
ax.text(
    0.05, 0.95,
    f"$R^2$ = {r_value**2:.2f}\n$y$ = {slope:.2f}$x$ {'+' if intercept >= 0 else '-'} {abs(intercept):.2f}",
    transform=ax.transAxes,
    va="top", ha="left",
    fontsize=8,
)

ax.set_xlabel(first_post_xlabel)
ax.set_ylabel(r"$\Delta$spine volume (a.u.) [25-35 min]")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
savepath = os.path.join(save_folder, "combined_scatter_1st_post_vs_ltp.png")
plt.savefig(savepath, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved to: {savepath}")

# %% swarm plot: delta_FF0_intensity_ch2 grouped by 1st post frame bins (width=1)
sw_data = all_summary_df[["first_post_FF0_intensity_ch2", "delta_FF0_intensity_ch2"]].dropna().copy()

bin_min = int(np.floor(sw_data["first_post_FF0_intensity_ch2"].min()))
bin_max = int(np.ceil(sw_data["first_post_FF0_intensity_ch2"].max()))
bins = np.arange(bin_min, bin_max + 1, 1)
labels = [f"{int(b)}–{int(b+1)}" for b in bins[:-1]]
sw_data["bin"] = pd.cut(sw_data["first_post_FF0_intensity_ch2"], bins=bins, labels=labels)

fig2, ax2 = plt.subplots(figsize=(5, 3.5))
sns.swarmplot(x="bin", y="delta_FF0_intensity_ch2", data=sw_data, ax=ax2,
              size=3, color="steelblue", order=labels)

bin_means = sw_data.groupby("bin", observed=True)["delta_FF0_intensity_ch2"].mean()
for i, lbl in enumerate(labels):
    if lbl in bin_means.index and not np.isnan(bin_means[lbl]):
        mean_val = bin_means[lbl]
        ax2.plot([i - 0.3, i + 0.3], [mean_val, mean_val],
                 "-", color="black", linewidth=2)
        ax2.text(i + 0.33, mean_val, f"{mean_val:.2f}",
                 va="center", ha="left", fontsize=7, color="black")

ax2.axhline(0, color="gray", linestyle="--", linewidth=0.5)

bin_counts = sw_data.groupby("bin", observed=True)["delta_FF0_intensity_ch2"].count()
tick_labels = [
    f"{lbl}\n$n$={bin_counts[lbl]}" if lbl in bin_counts.index else lbl
    for lbl in labels
]
ax2.set_xticks(range(len(labels)))
ax2.set_xticklabels(tick_labels, fontsize=8)

ax2.set_xlabel(first_post_xlabel)
ax2.set_ylabel(r"$\Delta$spine volume (a.u.) [25–35 min]")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

plt.tight_layout()
savepath2 = os.path.join(save_folder, "combined_swarm_binned_1st_post_vs_ltp.png")
plt.savefig(savepath2, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved to: {savepath2}")

# %% scatter plot colored by date
fig3, ax3 = plt.subplots(figsize=(5, 3.5))

sorted_dates = sorted(all_summary_df["date"].dropna().unique())
date_palette = dict(zip(sorted_dates, sns.color_palette("tab10", n_colors=len(sorted_dates))))

for date_val in sorted_dates:
    subset = all_summary_df[all_summary_df["date"] == date_val]
    ax3.scatter(
        subset["first_post_FF0_intensity_ch2"],
        subset["delta_FF0_intensity_ch2"],
        color=date_palette[date_val],
        label=date_val,
        s=20,
        zorder=3,
    )

x_range3 = all_summary_df["first_post_FF0_intensity_ch2"].max() - all_summary_df["first_post_FF0_intensity_ch2"].min()
y_range3 = all_summary_df["delta_FF0_intensity_ch2"].max() - all_summary_df["delta_FF0_intensity_ch2"].min()
xlim3 = [all_summary_df["first_post_FF0_intensity_ch2"].min() - x_range3 * 0.1,
         all_summary_df["first_post_FF0_intensity_ch2"].max() + x_range3 * 0.1]
ylim3 = [all_summary_df["delta_FF0_intensity_ch2"].min() - y_range3 * 0.1,
         all_summary_df["delta_FF0_intensity_ch2"].max() + y_range3 * 0.1]

ax3.set_xlim(xlim3)
ax3.set_ylim(ylim3)
ax3.plot([xlim3[0], xlim3[1]], [0, 0], "--", color="gray", linewidth=0.5)
ax3.plot([0, 0], [ylim3[0], ylim3[1]], "--", color="gray", linewidth=0.5)

x_fit3 = np.array([xlim3[0], xlim3[1]])
ax3.plot(x_fit3, slope * x_fit3 + intercept, "-", color="black", linewidth=1, zorder=2)
ax3.text(
    0.05, 0.95,
    f"$R^2$ = {r_value**2:.2f}\n$y$ = {slope:.2f}$x$ {'+' if intercept >= 0 else '-'} {abs(intercept):.2f}",
    transform=ax3.transAxes,
    va="top", ha="left",
    fontsize=8,
)

ax3.legend(title="Date", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0, fontsize=7)
ax3.set_xlabel(first_post_xlabel)
ax3.set_ylabel(r"$\Delta$spine volume (a.u.) [25–35 min]")
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)

plt.tight_layout()
savepath3 = os.path.join(save_folder, "combined_scatter_1st_post_vs_ltp_by_date.png")
plt.savefig(savepath3, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved to: {savepath3}")

# %% scatter plot split by cAMP enhancer (with / without)
# with: 20260318 group containing "B3", and all of 20260324
# without: everything else
def _assign_cAMP(row):
    if row["date"] == "20260324":
        return "with cAMP enhancer"
    # if row["date"] == "20260318" and "B3" in str(row["group"]):
    if row["date"] == "20260318":
        return "with cAMP enhancer"
    return "without cAMP enhancer"

all_summary_df["cAMP_enhancer"] = all_summary_df.apply(_assign_cAMP, axis=1)

camp_groups = ["with cAMP enhancer", "without cAMP enhancer"]
camp_colors = {"with cAMP enhancer": "tomato", "without cAMP enhancer": "steelblue"}

fig4, axes4 = plt.subplots(1, 2, figsize=(8, 3.5), sharey=True)

x_all = all_summary_df["first_post_FF0_intensity_ch2"]
y_all = all_summary_df["delta_FF0_intensity_ch2"]
x_range4 = x_all.max() - x_all.min()
y_range4 = y_all.max() - y_all.min()
xlim4 = [x_all.min() - x_range4 * 0.1, x_all.max() + x_range4 * 0.1]
ylim4 = [y_all.min() - y_range4 * 0.1, y_all.max() + y_range4 * 0.1]

for ax4, camp_label in zip(axes4, camp_groups):
    subset = all_summary_df[all_summary_df["cAMP_enhancer"] == camp_label].dropna(
        subset=["first_post_FF0_intensity_ch2", "delta_FF0_intensity_ch2"]
    )
    ax4.scatter(
        subset["first_post_FF0_intensity_ch2"],
        subset["delta_FF0_intensity_ch2"],
        color=camp_colors[camp_label],
        s=20, zorder=3,
    )

    ax4.set_xlim(xlim4)
    ax4.set_ylim(ylim4)
    ax4.plot(xlim4, [0, 0], "--", color="gray", linewidth=0.5)
    ax4.plot([0, 0], ylim4, "--", color="gray", linewidth=0.5)

    if len(subset) >= 2:
        sl, ic, rv, pv, _ = stats.linregress(
            subset["first_post_FF0_intensity_ch2"], subset["delta_FF0_intensity_ch2"]
        )
        x_fit4 = np.array(xlim4)
        ax4.plot(x_fit4, sl * x_fit4 + ic, "-", color="black", linewidth=1, zorder=2)
        ax4.text(
            0.05, 0.95,
            f"$R^2$ = {rv**2:.2f}\n$y$ = {sl:.2f}$x$ {'+' if ic >= 0 else '-'} {abs(ic):.2f}\n$n$ = {len(subset)}",
            transform=ax4.transAxes, va="top", ha="left", fontsize=7,
        )

    ax4.set_title(camp_label, fontsize=9)
    ax4.set_xlabel(first_post_xlabel)
    ax4.spines["top"].set_visible(False)
    ax4.spines["right"].set_visible(False)

axes4[0].set_ylabel(r"$\Delta$spine volume (a.u.) [25–35 min]")

plt.tight_layout()
savepath4 = os.path.join(save_folder, "combined_scatter_1st_post_vs_ltp_by_cAMP.png")
plt.savefig(savepath4, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved to: {savepath4}")

# %% swarm plot by bin, split by cAMP enhancer
fig5, axes5 = plt.subplots(1, 2, figsize=(9, 3.5), sharey=True)

sw_all = all_summary_df[["first_post_FF0_intensity_ch2", "delta_FF0_intensity_ch2", "cAMP_enhancer"]].dropna().copy()
sw_bin_min = int(np.floor(sw_all["first_post_FF0_intensity_ch2"].min()))
sw_bin_max = int(np.ceil(sw_all["first_post_FF0_intensity_ch2"].max()))
sw_bins = np.arange(sw_bin_min, sw_bin_max + 1, 1)
sw_labels = [f"{int(b)}–{int(b+1)}" for b in sw_bins[:-1]]
sw_all["bin"] = pd.cut(sw_all["first_post_FF0_intensity_ch2"], bins=sw_bins, labels=sw_labels)

y_all_sw = sw_all["delta_FF0_intensity_ch2"]
ylim5 = [y_all_sw.min() - (y_all_sw.max() - y_all_sw.min()) * 0.1,
         y_all_sw.max() + (y_all_sw.max() - y_all_sw.min()) * 0.1]

for ax5, camp_label in zip(axes5, camp_groups):
    subset_sw = sw_all[sw_all["cAMP_enhancer"] == camp_label]
    sns.swarmplot(x="bin", y="delta_FF0_intensity_ch2", data=subset_sw, ax=ax5,
                  size=3, color=camp_colors[camp_label], order=sw_labels)

    bin_means5 = subset_sw.groupby("bin", observed=True)["delta_FF0_intensity_ch2"].mean()
    bin_counts5 = subset_sw.groupby("bin", observed=True)["delta_FF0_intensity_ch2"].count()
    for i, lbl in enumerate(sw_labels):
        if lbl in bin_means5.index and not np.isnan(bin_means5[lbl]):
            mean_val5 = bin_means5[lbl]
            ax5.plot([i - 0.3, i + 0.3], [mean_val5, mean_val5],
                     "-", color="black", linewidth=2)
            ax5.text(i + 0.33, mean_val5, f"{mean_val5:.2f}",
                     va="center", ha="left", fontsize=7, color="black")

    tick_labels5 = [
        f"{lbl}\n$n$={bin_counts5[lbl]}" if lbl in bin_counts5.index else lbl
        for lbl in sw_labels
    ]
    ax5.set_xticks(range(len(sw_labels)))
    ax5.set_xticklabels(tick_labels5, fontsize=8)
    ax5.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    ax5.set_ylim(ylim5)
    ax5.set_title(camp_label, fontsize=9)
    ax5.set_xlabel(first_post_xlabel)
    ax5.spines["top"].set_visible(False)
    ax5.spines["right"].set_visible(False)

axes5[0].set_ylabel(r"$\Delta$spine volume (a.u.) [25–35 min]")

plt.tight_layout()
savepath5 = os.path.join(save_folder, "combined_swarm_binned_by_cAMP.png")
plt.savefig(savepath5, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved to: {savepath5}")

# %% helper: scatter plot of GCaMP transient vs delta spine volume (colored by mW)
def _plot_transient_scatter(x_col, xlabel, savename):
    plot_t = all_summary_df[["uncaging_power_coherent_mW", "unc_mW_grouped",
                              x_col, "delta_FF0_intensity_ch2"]].dropna()
    if len(plot_t) < 2:
        print(f"Not enough data for {savename}")
        return

    fig_t, ax_t = plt.subplots(figsize=(5, 3.5))

    for mW_val in sorted(plot_t["unc_mW_grouped"].unique()):
        sub = plot_t[plot_t["unc_mW_grouped"] == mW_val]
        ax_t.scatter(sub[x_col], sub["delta_FF0_intensity_ch2"],
                     color=cmap(norm(mW_val)), label=f"{mW_val:.1f}", s=20, zorder=3)

    x_r = plot_t[x_col].max() - plot_t[x_col].min()
    y_r = plot_t["delta_FF0_intensity_ch2"].max() - plot_t["delta_FF0_intensity_ch2"].min()
    xlim_t = [plot_t[x_col].min() - x_r * 0.1, plot_t[x_col].max() + x_r * 0.1]
    ylim_t = [plot_t["delta_FF0_intensity_ch2"].min() - y_r * 0.1,
              plot_t["delta_FF0_intensity_ch2"].max() + y_r * 0.1]
    ax_t.set_xlim(xlim_t)
    ax_t.set_ylim(ylim_t)
    ax_t.plot(xlim_t, [0, 0], "--", color="gray", linewidth=0.5)
    ax_t.plot([0, 0], ylim_t, "--", color="gray", linewidth=0.5)

    sl_t, ic_t, rv_t, _, _ = stats.linregress(plot_t[x_col], plot_t["delta_FF0_intensity_ch2"])
    x_fit_t = np.array(xlim_t)
    ax_t.plot(x_fit_t, sl_t * x_fit_t + ic_t, "-", color="black", linewidth=1, zorder=2)
    ax_t.text(0.05, 0.95,
              f"$R^2$ = {rv_t**2:.2f}\n$y$ = {sl_t:.2f}$x$ {'+' if ic_t >= 0 else '-'} {abs(ic_t):.2f}",
              transform=ax_t.transAxes, va="top", ha="left", fontsize=8)

    ax_t.legend(title="Unc. power (mW)", bbox_to_anchor=(1.02, 1), loc="upper left",
                borderaxespad=0, fontsize=7)
    ax_t.set_xlabel(xlabel)
    ax_t.set_ylabel(r"$\Delta$spine volume (a.u.) [25–35 min]")
    ax_t.spines["top"].set_visible(False)
    ax_t.spines["right"].set_visible(False)
    plt.tight_layout()
    sp = os.path.join(save_folder, savename)
    plt.savefig(sp, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved to: {sp}")

_plot_transient_scatter(
    x_col="transient_DendriticShaft_Ch1_intensity",
    xlabel="GCaMP Dendritic Shaft F/F0 [at uncaging]",
    savename="combined_scatter_transient_shaft_vs_ltp.png",
)
_plot_transient_scatter(
    x_col="transient_Spine_Ch1_intensity",
    xlabel="GCaMP Spine F/F0 [at uncaging]",
    savename="combined_scatter_transient_spine_vs_ltp.png",
)

# %% helper: scatter split by cAMP enhancer
def _plot_transient_scatter_by_cAMP(x_col, xlabel, savename):
    plot_c = all_summary_df[["cAMP_enhancer", x_col, "delta_FF0_intensity_ch2"]].dropna()
    if len(plot_c) < 2:
        print(f"Not enough data for {savename}")
        return

    x_r = plot_c[x_col].max() - plot_c[x_col].min()
    y_r = plot_c["delta_FF0_intensity_ch2"].max() - plot_c["delta_FF0_intensity_ch2"].min()
    xlim_c = [plot_c[x_col].min() - x_r * 0.1, plot_c[x_col].max() + x_r * 0.1]
    ylim_c = [plot_c["delta_FF0_intensity_ch2"].min() - y_r * 0.1,
              plot_c["delta_FF0_intensity_ch2"].max() + y_r * 0.1]

    fig_c, axes_c = plt.subplots(1, 2, figsize=(8, 3.5), sharey=True)
    for ax_c, camp_label in zip(axes_c, camp_groups):
        sub = plot_c[plot_c["cAMP_enhancer"] == camp_label]
        ax_c.scatter(sub[x_col], sub["delta_FF0_intensity_ch2"],
                     color=camp_colors[camp_label], s=20, zorder=3)
        ax_c.set_xlim(xlim_c)
        ax_c.set_ylim(ylim_c)
        ax_c.plot(xlim_c, [0, 0], "--", color="gray", linewidth=0.5)
        ax_c.plot([0, 0], ylim_c, "--", color="gray", linewidth=0.5)

        if len(sub) >= 2:
            sl_c, ic_c, rv_c, _, _ = stats.linregress(sub[x_col], sub["delta_FF0_intensity_ch2"])
            x_fit_c = np.array(xlim_c)
            ax_c.plot(x_fit_c, sl_c * x_fit_c + ic_c, "-", color="black", linewidth=1, zorder=2)
            ax_c.text(0.05, 0.95,
                      f"$R^2$ = {rv_c**2:.2f}\n$y$ = {sl_c:.2f}$x$ {'+' if ic_c >= 0 else '-'} {abs(ic_c):.2f}\n$n$ = {len(sub)}",
                      transform=ax_c.transAxes, va="top", ha="left", fontsize=7)

        ax_c.set_title(camp_label, fontsize=9)
        ax_c.set_xlabel(xlabel)
        ax_c.spines["top"].set_visible(False)
        ax_c.spines["right"].set_visible(False)

    axes_c[0].set_ylabel(r"$\Delta$spine volume (a.u.) [25–35 min]")
    plt.tight_layout()
    sp = os.path.join(save_folder, savename)
    plt.savefig(sp, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved to: {sp}")

_plot_transient_scatter_by_cAMP(
    x_col="transient_DendriticShaft_Ch1_intensity",
    xlabel="GCaMP Dendritic Shaft F/F0 [at uncaging]",
    savename="combined_scatter_transient_shaft_vs_ltp_by_cAMP.png",
)
_plot_transient_scatter_by_cAMP(
    x_col="transient_Spine_Ch1_intensity",
    xlabel="GCaMP Spine F/F0 [at uncaging]",
    savename="combined_scatter_transient_spine_vs_ltp_by_cAMP.png",
)
