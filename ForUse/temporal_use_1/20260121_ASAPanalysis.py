# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 16:04:09 2026

@author: yasudalab
"""
# %%
import sys
sys.path.append("../../")
import numpy as np
from AnalysisForFLIMage.read_flimagecsv import csv_to_df
from custom_plot import plt
# csvpath = r"G:\ImagingData\Tetsuya\20260121\Analysis\ASAP5_linescan__TimeCourse - Copy.csv"
# csvpath = r"G:\ImagingData\Tetsuya\20260121\Analysis\ASAP5_neuro2_linescan__TimeCourse - Copy.csv"
csvpath = r"G:\ImagingData\Tetsuya\20260121\Analysis\ASAP5_pos3_linescan__TimeCourse - Copy.csv"
resultdf_original = csv_to_df(csvpath)

print(resultdf_original.head())

# %%

time_1frame_ms = 2

ROInum = 1
resultdf = resultdf_original[resultdf_original["ROInum"] == ROInum]

plt.figure(figsize=(15, 3))
plt.plot(resultdf["time_sec"], resultdf["sumIntensity-ROI"],
         linewidth=0.1)
plt.ylabel("intensity (a.u.)")
plt.xlabel("time (sec)")
savepath = csvpath[:-4] + f"_ROInum{ROInum}_sumIntensity-ROI_plot.png"
plt.savefig(savepath, dpi=450, bbox_inches="tight")
plt.show()


# %%



#moving average over plus minus 1 frame
resultdf["sumIntensity-ROI_moving_average_3frame"] = resultdf["sumIntensity-ROI"].rolling(window=3).mean()
resultdf["sumIntensity-ROI_moving_average_5frame"] = resultdf["sumIntensity-ROI"].rolling(window=5).mean()
resultdf["sumIntensity-ROI_moving_average_11frame"] = resultdf["sumIntensity-ROI"].rolling(window=11).mean()
resultdf["Lifetime-ROI_moving_average_5frame"] = resultdf["Lifetime-ROI"].rolling(window=5).mean()
#%%
yaxis = "sumIntensity-ROI"
yaxis2 = "sumIntensity-ROI_moving_average_11frame"
plt.figure(figsize=(15, 3))
plt.plot(resultdf["time_sec"], resultdf[yaxis],
         linewidth=0.1, label="raw", color="gray")
# plt.plot(resultdf["time_sec"], resultdf["sumIntensity-ROI_moving_average_3frame"],
#          linewidth=0.1)
# plt.plot(resultdf["time_sec"], resultdf["sumIntensity-ROI_moving_average_5frame"],
#          linewidth=0.1)
plt.plot(resultdf["time_sec"], resultdf[yaxis2],
         linewidth=0.1, 
         label=f"moving average, {time_1frame_ms*10} ms",
         color = 'k')
plt.ylabel(yaxis)
plt.xlabel("time (sec)")
plt.legend()
savepath = csvpath[:-4] + f"_ROInum{ROInum}_{yaxis}_moving_average_plot.png"
plt.savefig(savepath, dpi=450, bbox_inches="tight")
plt.show()






# %%
plt.figure(figsize=(5, 3))
time_sec_from = 72
time_sec_to = time_sec_from + 5

col_palette = ["gray", "k"]
# col_palette = ["pink", "red"]

# yaxis = "sumIntensity-ROI"
# yaxis2 = "sumIntensity-ROI_moving_average_5frame"

yaxis = "Lifetime-ROI"
yaxis2 = "Lifetime-ROI_moving_average_5frame"
resultdf_subset = resultdf[(resultdf["time_sec"] >= time_sec_from) & (resultdf["time_sec"] <= time_sec_to)]
plt.plot(resultdf_subset["time_sec"], resultdf_subset[yaxis],
         linewidth=0.1, label="raw", color=col_palette[0])
# plt.plot(resultdf["time_sec"], resultdf["sumIntensity-ROI_moving_average_3frame"],
#          linewidth=0.1)
# plt.plot(resultdf["time_sec"], resultdf["sumIntensity-ROI_moving_average_5frame"],
#          linewidth=0.1)
plt.plot(resultdf_subset["time_sec"], resultdf_subset[yaxis2],
         linewidth=0.4,
         label=f"moving average, {time_1frame_ms*10} ms", 
         color = col_palette[1])
plt.ylabel(yaxis)
plt.xlabel("time (sec)")
# plt.legend()
ylim_max = resultdf_subset[yaxis2].max()*1.1
ylim_min = resultdf_subset[yaxis2].min()*0.9
plt.ylim(ylim_min, ylim_max)
tick_interval = 1
plt.xticks(np.arange(time_sec_from, time_sec_to+tick_interval, tick_interval), fontsize=12)
savepath = csvpath[:-4] + f"_ROInum{ROInum}_{yaxis}_moving_average_plot_time_{time_sec_from}_{time_sec_to}.png"
plt.savefig(savepath, dpi=450, bbox_inches="tight")
plt.show()





# %% plot both ROI 1 and 2




time_1frame_ms = 2

ROInum = 2
resultdf = resultdf_original[resultdf_original["ROInum"] == ROInum]

plt.figure(figsize=(15, 3))
plt.plot(resultdf["time_sec"], resultdf["sumIntensity-ROI"],
         linewidth=0.1)
plt.ylabel("intensity (a.u.)")
plt.xlabel("time (sec)")
savepath = csvpath[:-4] + f"_ROInum{ROInum}_sumIntensity-ROI_plot.png"
plt.savefig(savepath, dpi=450, bbox_inches="tight")
plt.show()


# %% plot both ROI 1 and 2 on the same plot

resultdf1 = resultdf_original[resultdf_original["ROInum"] == 1] 
resultdf2 = resultdf_original[resultdf_original["ROInum"] == 2]
#moving average over plus minus 1 frame
resultdf1["sumIntensity-ROI_moving_average_5frame"] = resultdf1["sumIntensity-ROI"].rolling(window=5).mean()
resultdf2["sumIntensity-ROI_moving_average_5frame"] = resultdf2["sumIntensity-ROI"].rolling(window=5).mean()
resultdf1["Lifetime-ROI_moving_average_5frame"] = resultdf1["Lifetime-ROI"].rolling(window=5).mean()
resultdf2["Lifetime-ROI_moving_average_5frame"] = resultdf2["Lifetime-ROI"].rolling(window=5).mean()
#%%
yaxis = "sumIntensity-ROI"
yaxis2 = "sumIntensity-ROI_moving_average_5frame"
fig = plt.figure(figsize=(15, 3))
plt.plot(resultdf1["time_sec"], resultdf1[yaxis],
         linewidth=0.1, label="Neuron 1", color="gray")
plt.plot(resultdf1["time_sec"], resultdf1[yaxis2],
         linewidth=0.4, label=f"moving average{time_1frame_ms*5} ms", color = 'k')

plt.plot(resultdf2["time_sec"], resultdf2[yaxis],
         linewidth=0.1, label="Neuron 2", color="pink")
plt.plot(resultdf2["time_sec"], resultdf2[yaxis2],
         linewidth=0.4, label=f"moving average{time_1frame_ms*5} ms", 
         color = 'red')

plt.ylabel(yaxis)
plt.xlabel("time (sec)")
#outside the plot
# plt.legend(loc="outside right upper")

plt.legend(bbox_to_anchor=(1, 1),
          bbox_transform=fig.transFigure)
savepath = csvpath[:-4] + f"_ROInum{ROInum}_{yaxis}_moving_average_plot.png"
plt.savefig(savepath, dpi=450, bbox_inches="tight")
plt.show()

# %% plot both ROI 1 and 2 on the same plot




# %%
