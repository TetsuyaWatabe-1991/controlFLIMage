# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 14:29:09 2024

@author: WatabeT
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 13:49:32 2024

@author: WatabeT
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 13:46:14 2023

@author: WatabeT
"""
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

save_True = True
target_roi_num = 1
time_bin = 5


# one_of_filepath = r"\\RY-LAB-WS04\ImagingData\Tetsuya\20240827\24well_0808GFP\highmag_Trans5ms\tpem\Analysis\copied\C1_00_5_1__highmag_2__TimeCourse.csv"
# csvlist = glob.glob(one_of_filepath[:one_of_filepath.rfind("\\")]+"\\*_TimeCourse.csv")

csvlist = [
    r"\\RY-LAB-WS04\ImagingData\Tetsuya\20241029\24well_1011_1016_FLEXGFP\highmag_GFP200ms55p\tpem\Analysis - Copy\A1_72h_dend4_28um__TimeCourse.csv",
    r"\\RY-LAB-WS04\ImagingData\Tetsuya\20241029\24well_1011_1016_FLEXGFP\highmag_GFP200ms55p\tpem\Analysis - Copy\A1_72h_dend3_28um__TimeCourse.csv",
    r"\\RY-LAB-WS04\ImagingData\Tetsuya\20241029\24well_1011_1016_FLEXGFP\highmag_GFP200ms55p\tpem\Analysis - Copy\A1_72h_dend2_34um__TimeCourse.csv",
    r"\\RY-LAB-WS04\ImagingData\Tetsuya\20241029\24well_1011_1016_FLEXGFP\highmag_GFP200ms55p\tpem\Analysis - Copy\A1_72h_dend1__TimeCourse.csv",
    r"\\RY-LAB-WS04\ImagingData\Tetsuya\20241029\24well_1011_1016_FLEXGFP\highmag_GFP200ms55p\tpem\Analysis - Copy\A1_72h_dend5_23um__TimeCourse.csv"
    ]

# csvlist = [
#     r"\\RY-LAB-WS04\ImagingData\Tetsuya\20241029\24well_1011_1016_FLEXGFP\highmag_GFP200ms55p\tpem\Analysis - Copy\A1_dendrite_11_15um_41h__TimeCourse.csv",
#     r"\\RY-LAB-WS04\ImagingData\Tetsuya\20241029\24well_1011_1016_FLEXGFP\highmag_GFP200ms55p\tpem\Analysis - Copy\A1_dendrite_10_20um_40h__TimeCourse.csv",
#     r"\\RY-LAB-WS04\ImagingData\Tetsuya\20241029\24well_1011_1016_FLEXGFP\highmag_GFP200ms55p\tpem\Analysis - Copy\A1_dendrite_9_12um_40h__TimeCourse.csv",
#     r"\\RY-LAB-WS04\ImagingData\Tetsuya\20241029\24well_1011_1016_FLEXGFP\highmag_GFP200ms55p\tpem\Analysis - Copy\A1_dendrite_13_10um_42h__TimeCourse.csv",
#     r"\\RY-LAB-WS04\ImagingData\Tetsuya\20241029\24well_1011_1016_FLEXGFP\highmag_GFP200ms55p\tpem\Analysis - Copy\A1_dendrite_12_25um_42h__TimeCourse.csv"
#     ]


csvlist = [
    r"\\RY-LAB-WS04\ImagingData\Tetsuya\20241029\24well_1011_1016_FLEXGFP\highmag_GFP200ms55p\tpem\Analysis - Copy\A1_dendrite_7_25um_28h__TimeCourse.csv",
    r"\\RY-LAB-WS04\ImagingData\Tetsuya\20241029\24well_1011_1016_FLEXGFP\highmag_GFP200ms55p\tpem\Analysis - Copy\A1_dendrite_6_25um_27h__TimeCourse.csv",
    # r"\\RY-LAB-WS04\ImagingData\Tetsuya\20241029\24well_1011_1016_FLEXGFP\highmag_GFP200ms55p\tpem\Analysis - Copy\A1_dendrite_5_25um_27h__TimeCourse.csv",
    # r"\\RY-LAB-WS04\ImagingData\Tetsuya\20241029\24well_1011_1016_FLEXGFP\highmag_GFP200ms55p\tpem\Analysis - Copy\A1_dendrite_4_25um_25h__TimeCourse.csv",
    # r"\\RY-LAB-WS04\ImagingData\Tetsuya\20241029\24well_1011_1016_FLEXGFP\highmag_GFP200ms55p\tpem\Analysis - Copy\A1_dendrite_3_24h__TimeCourse.csv",
    # r"\\RY-LAB-WS04\ImagingData\Tetsuya\20241029\24well_1011_1016_FLEXGFP\highmag_GFP200ms55p\tpem\Analysis - Copy\A1_dendrite_2_20h__TimeCourse.csv",
    # r"\\RY-LAB-WS04\ImagingData\Tetsuya\20241029\24well_1011_1016_FLEXGFP\highmag_GFP200ms55p\tpem\Analysis - Copy\A1_dendrite_20h__TimeCourse.csv"
]

one_of_filepath = csvlist[0]
allcombined_df_savepath= one_of_filepath[:-4] + "_combined.csv"

# allcombined_df_savepath= r"\\RY-LAB-WS04\ImagingData\Tetsuya\20240827\24well_0808GFP\highmag_Trans5ms\tpem_tpem2_combined.xxx"


allcombined_df=pd.DataFrame()
for csvpath in csvlist:
    print(csvpath)
    resultdf=csv_to_df(csvpath,
                       ch_list=[1])
                       # ch_list=[1,2])#,
                       # prefix_list=["sumIntensity_bg-ROI"])
    
    if len(resultdf)<2:
        print("len = ",len(resultdf),"   less than 2")
        continue
    
    if len(resultdf)<38:
        print("len = ",len(resultdf),"   less than 38")
        continue
    
    resultdf2 = detect_uncaging(resultdf) 
    resultdf3 = arrange_for_multipos3(resultdf2)
    resultdf4 = value_normalize(resultdf3, prefix = "sumIntensity_bg-ROI")
    resultdf5 = resultdf4
    
    for ROInum in resultdf5["ROInum"].unique(): 
        eachROIdf = resultdf5[resultdf5["ROInum"] ==  ROInum]
        eachROIdf["CellName"] = resultdf5.loc[:,"FilePath"]+"_"+str(ROInum)
        allcombined_df=pd.concat([allcombined_df,eachROIdf],ignore_index=True)


allcombined_df = everymin_normalize(allcombined_df)


plt.figure(figsize = [3,2])
##################################
sns.lineplot(x="time_min_norm", y="norm_sumIntensity_bg-ROI",
                legend=False, hue = "CellName", #marker='o',
                data = allcombined_df[allcombined_df['ch']==1],
                zorder = 10)

plt.plot([allcombined_df["time_min_norm"].min(),
          allcombined_df["time_min_norm"].max()],
         [1,1],c='gray',ls = '--', zorder = 1)


plt.ylabel("Spine volume (a.u.)")
plt.xlabel("Time (min)")
ymin, ymax = plt.gca().get_ylim()
uncaging_lineheight = ymax 
plt.plot([0,1],[uncaging_lineheight]*2,"k-")
plt.text(1,uncaging_lineheight*1.02,"Uncaging",
         ha="center",va="bottom",zorder=100)
plt.gca().spines[['right', 'top']].set_visible(False)

savepath = allcombined_df_savepath[:-4]+"_norm_sumIntensity_bg-ROI_plot.png"

plt.savefig(one_of_filepath[:-4]+"_mean_plot.pdf", format="pdf", bbox_inches="tight")
if save_True:
    plt.savefig(savepath, bbox_inches = "tight", dpi = 200)
    plt.savefig(savepath[:-4]+".pdf", bbox_inches = "tight", dpi = 200)
plt.show()




# max_time_minute = math.ceil(allcombined_df.time_min_norm.max()/time_bin)*time_bin
max_time_minute = math.ceil(28/time_bin)*time_bin

min_time_minute = math.floor(allcombined_df.time_min_norm.min()/time_bin)*time_bin

num_bin = int((max_time_minute - min_time_minute)/time_bin )

time_binned_df = pd.DataFrame()
for csvpath in allcombined_df.FilePath.unique():

    each_csv_df = allcombined_df[(allcombined_df['FilePath'] == csvpath)&
                            (allcombined_df['ROInum'] == target_roi_num)&
                            (allcombined_df['during_uncaging'] == 0)]
    
    for nth_time_bin in range(num_bin):
        min_time = min_time_minute + nth_time_bin * time_bin
        max_time = min_time + time_bin
        bin_df = each_csv_df[(min_time <= each_csv_df["time_min_norm"])&
                        (each_csv_df["time_min_norm"]< max_time)]
    
        bin_mean = bin_df["norm_sumIntensity_bg-ROI"].mean()
        bin_time = (max_time + min_time)/2
        
        each_time_df = pd.DataFrame({
                            "FilePath":[csvpath],
                            "bin_time":[bin_time],
                            "bin_mean":[bin_mean],
                            })
        time_binned_df = pd.concat([time_binned_df, each_time_df],
                                   ignore_index=True)
        
fig, ax = plt.subplots(figsize = [2,1])

sns.lineplot(x="bin_time", y="bin_mean",
            legend=False, 
            data = time_binned_df,
            errorbar = "se",
            err_style = "bars",
            palette = ["m"]
            )
ax.plot([time_binned_df["bin_time"].min(),
         time_binned_df["bin_time"].max()],
         [1,1], "--" , color = "gray")

uncaging_ypos = 3.7
ax.plot([0,1],
         [uncaging_ypos,uncaging_ypos], 
         color = "k")
ax.text(-2,uncaging_ypos + 0.1, "Uncaging")

ax.set_ylabel("Norm. spine (a.u.)")
ax.set_xlabel("Time (min)")
plt.ylim([0.86, 3.7])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.savefig(one_of_filepath[:-4]+"_mean_plot.pdf", format="pdf", bbox_inches="tight")
plt.savefig(one_of_filepath[:-4]+"_mean_plot.png", format="png", bbox_inches="tight")

# プロットを表示
plt.show()





sns.lineplot(x="time_min_norm", y="norm_sumIntensity_bg-ROI",
                legend=False, hue = "CellName", marker='o',
                data = allcombined_df[allcombined_df['ch']==1],
                zorder = 10)



plt.plot([allcombined_df["time_min_norm"].min(),
          allcombined_df["time_min_norm"].max()],
         [1,1],c='gray',ls = '--', zorder = 1)

plt.ylabel("Spine volume (a.u.)")
plt.xlabel("Time (min)")
plt.ylim([0.7, 4.1])
plt.gca().spines[['right', 'top']].set_visible(False)

savepath = allcombined_df_savepath[:-4]+"_norm_sumIntensity_bg-ROI_plot_ylimited.png"
if save_True:
    plt.savefig(savepath, bbox_inches = "tight", dpi = 200)
    plt.savefig(savepath[:-4]+".pdf", bbox_inches = "tight", dpi = 200)
plt.show()





adf = allcombined_df[(allcombined_df["binned_min"]>25)&
                     (allcombined_df["binned_min"]<35)&
                     (allcombined_df["ch"]==1)]

# groupdf = adf.groupby(["FilePath",
#                        "ROInum"]).mean()

groupdf = adf.groupby(["FilePath",
                        "ROInum"])

Mean = groupdf["norm_sumIntensity_bg-ROI"].mean()
print("Mean, ",Mean)

plt.figure(figsize = [2,4])
plt.plot([-0.2,0.2],[Mean.mean(),Mean.mean()],'k-')
xmin, xmax = plt.gca().get_xlim()
sns.swarmplot(Mean,palette = ['gray'])
plt.title("25 to 35 min after uncaging")
plt.ylabel("Spine volume")
plt.text(0.3,Mean.mean(),str(round(Mean.mean(),2)))
plt.gca().spines[['right', 'top']].set_visible(False)

if save_True:
    savepath = allcombined_df_savepath[:-4]+"_norm_sumIntensity_bg-ROI_mean_swarm.png"
    plt.savefig(savepath, bbox_inches = "tight", dpi = 200)
    savepath = allcombined_df_savepath[:-4]+"_norm_sumIntensity_bg-ROI_mean_swarm.pdf"
    plt.savefig(savepath, bbox_inches = "tight", dpi = 200)
plt.show()
