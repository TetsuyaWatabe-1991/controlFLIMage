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
import csv
import math
from read_flimagecsv import arrange_for_multipos3, csv_to_df, detect_uncaging, value_normalize, everymin_normalize

save_True = True
target_roi_num = 1
time_bin = 10


one_of_filepath = r"\\RY-LAB-WS04\ImagingData\Tetsuya\20240827\24well_0808GFP\highmag_Trans5ms\tpem\Analysis\copied\C1_00_5_1__highmag_2__TimeCourse.csv"
csvlist1 = glob.glob(one_of_filepath[:one_of_filepath.rfind("\\")]+"\\*_TimeCourse.csv")

one_of_filepath = r"\\RY-LAB-WS04\ImagingData\Tetsuya\20240827\24well_0808GFP\highmag_Trans5ms\tpem2\Analysis\copied\C1_00_1_2__highmag_1__TimeCourse.csv"
csvlist2 = glob.glob(one_of_filepath[:one_of_filepath.rfind("\\")]+"\\*_TimeCourse.csv")
csvlist = csvlist1 + csvlist2


# csvlist = glob.glob(one_of_filepath[:one_of_filepath.rfind("\\")]+"\\*_TimeCourse.csv")

allcombined_df_savepath= one_of_filepath[:one_of_filepath.rfind("\\")]+"\\result_.csv"

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
if save_True:
    plt.savefig(savepath, bbox_inches = "tight", dpi = 200)
    plt.savefig(savepath[:-4]+".pdf", bbox_inches = "tight", dpi = 200)
plt.show()



time_binned_df=pd.DataFrame()
max_time_minute = math.ceil(allcombined_df.time_min_norm.max()/time_bin)*time_bin

min_time_minute = math.floor(allcombined_df.time_min_norm.min()/time_bin)*time_bin

num_bin = int((max_time_minute - min_time_minute)/time_bin )

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
        

    resultdf = resultdf[resultdf["during_uncaging"]==0]
    
sns.lineplot(x="bin_time", y="bin_mean",
            legend=False, 
            data = time_binned_df,
            errorbar = "se",
            err_style = "bars",
            palette = ["m"]
            )



sns.lineplot(x="time_min_norm", y="norm_sumIntensity_bg-ROI",
                legend=False, hue = "CellName", marker='o',
                data = allcombined_df[allcombined_df['ch']==1],
                zorder = 10)

plt.plot([allcombined_df["time_min_norm"].min(),
          allcombined_df["time_min_norm"].max()],
         [1,1],c='gray',ls = '--', zorder = 1)

plt.ylabel("Spine volume (a.u.)")
plt.xlabel("Time (min)")
plt.ylim([0.7,2.6])
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
