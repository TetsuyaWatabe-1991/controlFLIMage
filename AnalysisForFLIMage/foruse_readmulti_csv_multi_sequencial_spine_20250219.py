# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 14:29:09 2024

@author: WatabeT
"""
import os
import glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('pdf', fonttype=42)
matplotlib.rc('font',size = 8)
plt.rcParams["font.family"] = "Arial"
import math
from read_flimagecsv import arrange_for_multipos3, csv_to_df, detect_uncaging, value_normalize, everymin_normalize

save_True = True

one_of_filepath = r"G:\ImagingData\Tetsuya\20250218\24well_B6_FxGFP_cut0204_aav0211\highmag_GFP200ms47p\highmag_GFP100ms47p\tpem\Analysis\copy\1_1__highmag_1_align_TimeCourse.csv"

one_of_filepath = r"G:\ImagingData\Tetsuya\20250217\24well_Cas9_FXGFP__0206B6_FlxGC6stdTom\highmag_Trans5ms\tpem\Analysis\copied\B1_00_1_1__highmag_1__ALL_TimeCourse.csv"

csvlist = glob.glob(one_of_filepath[:one_of_filepath.rfind("\\")]+"\\*_TimeCourse.csv")
one_of_filepath = csvlist[0]
allcombined_df_savepath= one_of_filepath[:-4] + "_combined.csv"
summarized_df_savepath= one_of_filepath[:-4] + "_combined_summarized.csv"

allcombined_df=pd.DataFrame()

for csvpath in csvlist:
    print(csvpath)
    resultdf=csv_to_df(csvpath,
                       ch_list=[1])
    
    if len(resultdf)<2:
        print("len = ",len(resultdf),"   less than 2")
        continue
    
    if len(resultdf)<36:
        print("len = ",len(resultdf),"   less than 38")
        continue
    
    df = detect_uncaging(resultdf) 
    
    # Step 1: Identify rows before the 0 -> 1 transition
    df['label_trigger'] = (df['during_uncaging'] == 0) & (df['during_uncaging'].shift(-1) == 1)
    
    # Step 2: Assign group labels
    df['group_label'] = df['label_trigger'].cumsum()
    
    # Drop the helper column (optional)
    df = df.drop(columns=['label_trigger'])
    for NthFrame in df["NthFrame"].unique():
        NthFrame_df = df[df["NthFrame"] == NthFrame]
        df.loc[NthFrame_df.index, "group"] = NthFrame_df['group_label'].max()

    allcombined_df=pd.concat([allcombined_df,df],
                             ignore_index=True)

allcombined_df.to_csv(allcombined_df_savepath)


spine_vol_list = []
time_aligned_df = pd.DataFrame()
summarized_df = pd.DataFrame()
unique_spine_label = 0
for each_filepath in allcombined_df["FilePath"].unique():
    each_file_df = allcombined_df[allcombined_df["FilePath"] == each_filepath]
    
    for each_group in each_file_df["group"].unique()[1:]:
        
        each_group_df = each_file_df[(each_file_df["group"] == each_group)&
                                     (each_file_df["ROInum"] == each_group)]
        each_group_df["NormFrame"] = each_group_df["NthFrame"] - each_group_df["NthFrame"].min()
        each_group_df["Norm_time_min"] = each_group_df["Time_min"] - each_group_df["Time_min"].min()
        
        intensity_numerator = each_group_df[each_group_df["NormFrame"] == 0 ]["sumIntensity-ROI"].values[0]
        each_group_df["Norm_intensity"] = each_group_df["sumIntensity-ROI"] / intensity_numerator
        each_group_df["unique_spine_label"] = str(unique_spine_label)
        time_aligned_df = pd.concat([time_aligned_df,
                                     each_group_df],
                                    ignore_index=True)
        unique_spine_label +=1
        
        try:
        
            norm_time_of_interest = each_group_df[each_group_df["Norm_time_min"] > 25]["Norm_time_min"].min()
            spine_vol = each_group_df[each_group_df["Norm_time_min"] == norm_time_of_interest]["Norm_intensity"].values[0]
            
            filename = os.path.basename(each_group_df["FilePath"].values[0])
            
            each_result = pd.DataFrame({
                            "filename":[filename],
                            "spine_vol":[spine_vol],
                            "spine_label":[each_group],
                            "norm_time_of_interest":[norm_time_of_interest],
                                  })
            
            if spine_vol < 10:
                spine_vol_list.append(spine_vol)
                summarized_df = pd.concat([summarized_df,each_result])
        except:
            continue

summarized_df.to_csv(summarized_df_savepath)
        
# plt.figure(figsize = [3,2])
# ##################################
sns.lineplot(x="Norm_time_min", y="Norm_intensity",
                legend=False, hue = "unique_spine_label", #marker='o',
                data = time_aligned_df
                )

np.array(spine_vol_list).mean()

# plt.plot([allcombined_df["time_min_norm"].min(),
#           allcombined_df["time_min_norm"].max()],
#          [1,1],c='gray',ls = '--', zorder = 1)


# plt.ylabel("Spine volume (a.u.)")
# plt.xlabel("Time (min)")
# ymin, ymax = plt.gca().get_ylim()
# uncaging_lineheight = ymax 
# plt.plot([0,1],[uncaging_lineheight]*2,"k-")
# plt.text(1,uncaging_lineheight*1.02,"Uncaging",
#          ha="center",va="bottom",zorder=100)
# plt.gca().spines[['right', 'top']].set_visible(False)

# savepath = allcombined_df_savepath[:-4]+"_norm_sumIntensity_bg-ROI_plot.png"

# plt.savefig(one_of_filepath[:-4]+"_mean_plot.pdf", format="pdf", bbox_inches="tight")
# if save_True:
#     plt.savefig(savepath, bbox_inches = "tight", dpi = 200)
#     plt.savefig(savepath[:-4]+".pdf", bbox_inches = "tight", dpi = 200)
# plt.show()




# # max_time_minute = math.ceil(allcombined_df.time_min_norm.max()/time_bin)*time_bin
# max_time_minute = math.ceil(28/time_bin)*time_bin

# min_time_minute = math.floor(allcombined_df.time_min_norm.min()/time_bin)*time_bin

# num_bin = int((max_time_minute - min_time_minute)/time_bin )

# time_binned_df = pd.DataFrame()
# for csvpath in allcombined_df.FilePath.unique():

#     each_csv_df = allcombined_df[(allcombined_df['FilePath'] == csvpath)&
#                             (allcombined_df['ROInum'] == target_roi_num)&
#                             (allcombined_df['during_uncaging'] == 0)]
    
#     for nth_time_bin in range(num_bin):
#         min_time = min_time_minute + nth_time_bin * time_bin
#         max_time = min_time + time_bin
#         bin_df = each_csv_df[(min_time <= each_csv_df["time_min_norm"])&
#                         (each_csv_df["time_min_norm"]< max_time)]
    
#         bin_mean = bin_df["norm_sumIntensity_bg-ROI"].mean()
#         bin_time = (max_time + min_time)/2
        
#         each_time_df = pd.DataFrame({
#                             "FilePath":[csvpath],
#                             "bin_time":[bin_time],
#                             "bin_mean":[bin_mean],
#                             })
#         time_binned_df = pd.concat([time_binned_df, each_time_df],
#                                    ignore_index=True)
        
# fig, ax = plt.subplots(figsize = [2,1])

# sns.lineplot(x="bin_time", y="bin_mean",
#             legend=False, 
#             data = time_binned_df,
#             errorbar = "se",
#             err_style = "bars",
#             palette = ["m"]
#             )
# ax.plot([time_binned_df["bin_time"].min(),
#          time_binned_df["bin_time"].max()],
#          [1,1], "--" , color = "gray")

# uncaging_ypos = 3.7
# ax.plot([0,1],
#          [uncaging_ypos,uncaging_ypos], 
#          color = "k")
# ax.text(-2,uncaging_ypos + 0.1, "Uncaging")

# ax.set_ylabel("Norm. spine (a.u.)")
# ax.set_xlabel("Time (min)")
# plt.ylim([0.86, 3.7])
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)

# plt.savefig(one_of_filepath[:-4]+"_mean_plot.pdf", format="pdf", bbox_inches="tight")
# plt.savefig(one_of_filepath[:-4]+"_mean_plot.png", format="png", bbox_inches="tight")

# # プロットを表示
# plt.show()





# sns.lineplot(x="time_min_norm", y="norm_sumIntensity_bg-ROI",
#                 legend=False, hue = "CellName", marker='o',
#                 data = allcombined_df[allcombined_df['ch']==1],
#                 zorder = 10)



# plt.plot([allcombined_df["time_min_norm"].min(),
#           allcombined_df["time_min_norm"].max()],
#          [1,1],c='gray',ls = '--', zorder = 1)

# plt.ylabel("Spine volume (a.u.)")
# plt.xlabel("Time (min)")
# plt.ylim([0.7, 4.1])
# plt.gca().spines[['right', 'top']].set_visible(False)

# savepath = allcombined_df_savepath[:-4]+"_norm_sumIntensity_bg-ROI_plot_ylimited.png"
# if save_True:
#     plt.savefig(savepath, bbox_inches = "tight", dpi = 200)
#     plt.savefig(savepath[:-4]+".pdf", bbox_inches = "tight", dpi = 200)
# plt.show()





# # adf = allcombined_df[(allcombined_df["binned_min"]>25)&
# #                      (allcombined_df["binned_min"]<35)&
# #                      (allcombined_df["ch"]==1)]
# adf = allcombined_df[(allcombined_df["binned_min"]>20)&
#                      (allcombined_df["binned_min"]<30)&
#                      (allcombined_df["ch"]==1)]

# # groupdf = adf.groupby(["FilePath",
# #                        "ROInum"]).mean()

# groupdf = adf.groupby(["FilePath",
#                         "ROInum"])

# Mean = groupdf["norm_sumIntensity_bg-ROI"].mean()
# print("Mean, ",Mean)

# plt.figure(figsize = [2,4])
# plt.plot([-0.2,0.2],[Mean.mean(),Mean.mean()],'k-')
# xmin, xmax = plt.gca().get_xlim()
# sns.swarmplot(Mean,palette = ['gray'])
# plt.title("25 to 35 min after uncaging")
# plt.ylabel("Spine volume")
# plt.text(0.3,Mean.mean(),str(round(Mean.mean(),2)))
# plt.gca().spines[['right', 'top']].set_visible(False)

# if save_True:
#     savepath = allcombined_df_savepath[:-4]+"_norm_sumIntensity_bg-ROI_mean_swarm.png"
#     plt.savefig(savepath, bbox_inches = "tight", dpi = 200)
#     savepath = allcombined_df_savepath[:-4]+"_norm_sumIntensity_bg-ROI_mean_swarm.pdf"
#     plt.savefig(savepath, bbox_inches = "tight", dpi = 200)
# plt.show()
