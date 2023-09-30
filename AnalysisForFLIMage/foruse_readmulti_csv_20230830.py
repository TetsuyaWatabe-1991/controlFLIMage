# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 13:46:14 2023

@author: WatabeT
"""
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import csv
from read_flimagecsv import arrange_for_multipos3, csv_to_df, detect_uncaging, value_normalize, everymin_normalize

save_True = False

csvlist=[
 	r"\\ry-lab-yas15\Users\Yasudalab\Documents\Tetsuya_Imaging\20230622\set1\Analysis\aligned_pos1_high__TimeCourse - Copy.csv",
 	r"\\ry-lab-yas15\Users\Yasudalab\Documents\Tetsuya_Imaging\20230622\set1\Analysis\aligned_pos2_high_TimeCourse - Copy.csv",
 	r"\\ry-lab-yas15\Users\Yasudalab\Documents\Tetsuya_Imaging\20230622\set1\Analysis\aligned_pos3_high__TimeCourse - Copy.csv",
    r"\\ry-lab-yas15\Users\Yasudalab\Documents\Tetsuya_Imaging\20230622\set1\Analysis\aligned_pos4__TimeCourse - Copy.csv",
    r"\\ry-lab-yas15\Users\Yasudalab\Documents\Tetsuya_Imaging\20230731\set2_SliceB\Analysis\pos2_high__ALL_TimeCourse.csv",
    r"\\ry-lab-yas15\Users\Yasudalab\Documents\Tetsuya_Imaging\20230801\set1SliceB\Analysis\pos1_high__ALL_TimeCourse.csv",
    r"\\ry-lab-yas15\Users\Yasudalab\Documents\Tetsuya_Imaging\20230801\set2sliceB\Analysis\pos1_high__ALL_TimeCourse.csv",
    r"\\ry-lab-yas15\Users\Yasudalab\Documents\Tetsuya_Imaging\20230801\set2sliceB\Analysis\pos2_high__ALL_TimeCourse.csv",
    r"\\ry-lab-yas15\Users\Yasudalab\Documents\Tetsuya_Imaging\20230801\set2sliceB\Analysis\pos3_high__ALL_TimeCourse.csv",
    ]

# csvlist=[
#     r"\\ry-lab-yas15\Users\Yasudalab\Documents\Tetsuya_Imaging\20230622\set2\Analysis\aligned_pos3_high__TimeCourse - Copy.csv",
#     r"\\ry-lab-yas15\Users\Yasudalab\Documents\Tetsuya_Imaging\20230622\set2\Analysis\aligned_pos1_high__TimeCourse - Copy.csv",
#     r"\\ry-lab-yas15\Users\Yasudalab\Documents\Tetsuya_Imaging\20230622\set2\Analysis\aligned_pos2_high__TimeCourse - Copy.csv",
#     r"\\ry-lab-yas15\Users\Yasudalab\Documents\Tetsuya_Imaging\20230731\set1_sliceA\Analysis\pos1_high__ALL_TimeCourse.csv",
#     r"\\ry-lab-yas15\Users\Yasudalab\Documents\Tetsuya_Imaging\20230731\set3_sliceA\Analysis\pos4_high__ALL_TimeCourse.csv",
#     r"\\ry-lab-yas15\Users\Yasudalab\Documents\Tetsuya_Imaging\20230731\set3_sliceA\Analysis\pos3_high__ALL_TimeCourse.csv",
#     r"\\ry-lab-yas15\Users\Yasudalab\Documents\Tetsuya_Imaging\20230731\set4_sliceA\Analysis\pos2_high__ALL_TimeCourse.csv",
#     r"\\ry-lab-yas15\Users\Yasudalab\Documents\Tetsuya_Imaging\20230731\set4_sliceA\Analysis\pos1_high__ALL_TimeCourse.csv"
# ]

allcombined_df_savepath=r"\\ry-lab-yas15\Users\Yasudalab\Documents\Tetsuya_Imaging\20230801\combined_B.csv"

allcombined_df=pd.DataFrame()
for csvpath in csvlist:
    print(csvpath)
    resultdf=csv_to_df(csvpath,
                       ch_list=[1,2])#,
                       # prefix_list=["sumIntensity_bg-ROI"])
    resultdf = detect_uncaging(resultdf) 
    resultdf = arrange_for_multipos3(resultdf)
    
    resultdf = value_normalize(resultdf,
                               prefix = "sumIntensity_bg-ROI")
    
    # resultdf = value_normalize(resultdf,
    #                     prefix = "Fraction2_fit-ROI",
    #                     normalize_subtraction = True)
    
    resultdf = resultdf[resultdf["during_uncaging"]==0]
                        
    for ROInum in resultdf["ROInum"].unique(): 
        eachROIdf = resultdf[resultdf["ROInum"] ==  ROInum]

        eachROIdf["CellName"] = resultdf.loc[:,"FilePath"]+"_"+str(ROInum)
        allcombined_df=allcombined_df.append(eachROIdf,ignore_index=True)


allcombined_df = everymin_normalize(allcombined_df)

if save_True:
    allcombined_df.to_csv(allcombined_df_savepath)



##################################
sns.lineplot(x="time_min_norm", y="norm_sumIntensity_bg-ROI",
                legend=False, hue = "CellName", marker='o',
                data = allcombined_df[allcombined_df['ch']==2],
                zorder = 10)

plt.plot([allcombined_df["time_min_norm"].min(),
          allcombined_df["time_min_norm"].max()],
         [1,1],c='gray',ls = '--', zorder = 1)

plt.ylabel("Spine volume (a.u.)")
plt.xlabel("Time (min)")
plt.xlim([-15,30])
plt.ylim([0.57,4.6])

uncaging_lineheight = 2
plt.plot([0,2],[uncaging_lineheight]*2,"k-")
plt.text(1,uncaging_lineheight*1.02,"Uncaging",
         ha="center",va="bottom",zorder=100)

savepath = allcombined_df_savepath[:-4]+"_norm_sumIntensity_bg-ROI_plot.png"
if save_True:
    plt.savefig(savepath, bbox_inches = "tight", dpi = 200)
plt.show()

adf = allcombined_df[(allcombined_df["binned_min"]>25)&
                     (allcombined_df["binned_min"]<35)&
                     (allcombined_df["ch"]==1)]

groupdf = adf.groupby(["FilePath","ROInum"]).mean()


Mean = groupdf["norm_sumIntensity_bg-ROI"].mean()
SEM = groupdf["norm_sumIntensity_bg-ROI"].std()/(len(groupdf)**0.5)
print("Mean, ",Mean)
print("SEM, ",SEM)

if save_True:
    groupdf.to_csv(allcombined_df_savepath[:-4]+"_groupby_mean.csv")


##################################

# sns.lineplot(x="binned_min", y="norm_sumIntensity_bg-ROI",
#                 legend=False, hue = "CellName", marker='o',
#                 data = allcombined_df[allcombined_df['ch']==1],
#                 zorder = 10)

# plt.plot([allcombined_df["time_min_norm"].min(),
#           allcombined_df["time_min_norm"].max()],
#          [1,1],c='gray',ls = '--', zorder = 1)

# plt.ylabel("Spine volume (a.u.)")
# plt.xlabel("Time (min)")

# uncaging_lineheight = 4
# plt.plot([0,2],[uncaging_lineheight]*2,"k-")
# plt.text(1,uncaging_lineheight*1.02,"Uncaging",
#          ha="center",va="bottom")

# plt.show()


################################

# sns.lineplot(x="time_min_norm", y="Fraction2_fit-ROI",
#                 legend=False, hue = "CellName", marker='o',
#                 data = allcombined_df[allcombined_df['ch']==1])
# plt.ylabel("Spine volume (a.u.)")
# plt.xlabel("Time (min)")

# # plt.plot([0,2],[2,2],"k-")
# # plt.text(1,2.05,"Uncaging",ha="center",va="bottom")
# plt.ylim([0.0,0.9])  
# plt.show()


################################

# sns.lineplot(x="time_min_norm", y="norm_Fraction2_fit-ROI",
#                 legend=False, hue = "CellName", marker='o',
#                 data = allcombined_df[allcombined_df['ch']==1],
#                 zorder = 10)

# plt.plot([allcombined_df["time_min_norm"].min(),
#           allcombined_df["time_min_norm"].max()],
#          [0,0],c='gray',ls = '--', zorder = 1)

# plt.ylabel("\u0394 Fraction2")
# plt.xlabel("Time (min)")

# uncaging_lineheight = 0.5
# plt.plot([0,2],[uncaging_lineheight]*2,"k-")
# plt.text(1,uncaging_lineheight*1.02,"Uncaging",
#          ha="center",va="bottom",zorder=100)

# plt.ylim([-0.9,0.9])  

# savepath = allcombined_df_savepath[:-4]+"_norm_Fraction2_plot.png"
# if save_True:
#     plt.savefig(savepath, bbox_inches = "tight", dpi = 200)

# plt.show()
