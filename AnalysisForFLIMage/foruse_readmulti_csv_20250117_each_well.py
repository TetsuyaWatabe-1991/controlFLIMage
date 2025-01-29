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
time_bin = 10

min_lower = 20
min_upper = 30



# one_of_filepath = r"\\RY-LAB-WS04\ImagingData\Tetsuya\20241213\24well\highmagGFP200ms55p\tpem_1\Analysis"

# one_of_filepath = r"\\RY-LAB-WS04\ImagingData\Tetsuya\20250104\24well\highmag_Trans5ms\tpem\Analysis - Copy\A2_00_1_1__highmag_2__TimeCourse__all_combined.csv"
# well_dict = {
#         "C1":"DMSO",
#         # "A2":"APV 50 uM",
#         # "D1":"APV 5 uM",
#         # "E1":"APV 0.5 uM",
#         # "F1":"APV 0.05 uM"
#         }

one_of_filepath = r"\\RY-LAB-WS04\ImagingData\Tetsuya\20250110\24well\highmag_GFP200ms55p\tpem\Analysis - Copy\A4_00_1_1__highmag_2__TimeCourse.csv"
well_dict = {"":"all"}

# one_of_filepath = r"\\RY-LAB-WS04\ImagingData\Tetsuya\20250114\24well\highmag_GFP200ms47p\tpem\Analysis - Copy\A2_00_8_2__highmag_3__TimeCourse.csv"
# well_dict = {"":"all"}

# one_of_filepath = r"\\RY-LAB-WS04\ImagingData\Tetsuya\20250115\24well7\highmag_GFP200ms47p\tpem\Analysis - Copy\B1_00_3_1__highmag_2__TimeCourse.csv"
# well_dict = {"":"all"}

swarmplot_ylim = [-0.4,3.1]

res_dict = {}


for each_wellname in well_dict:
        
    csvlist = glob.glob(one_of_filepath[:one_of_filepath.rfind("\\")]+"\\"+each_wellname+"*_TimeCourse.csv")

    one_of_filepath = csvlist[0]
    allcombined_df_savepath= one_of_filepath[:-4] + "_" + each_wellname+"_" + well_dict[each_wellname] + "_combined.csv"
    
    
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
        
        if len(resultdf)<36:
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
    allcombined_df.to_csv(allcombined_df_savepath)
    
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
    
    
    
    
    
    # adf = allcombined_df[(allcombined_df["binned_min"]>25)&
    #                      (allcombined_df["binned_min"]<35)&
    #                      (allcombined_df["ch"]==1)]
    adf = allcombined_df[(allcombined_df["binned_min"]>min_lower)&
                         (allcombined_df["binned_min"]<min_upper)&
                         (allcombined_df["ch"]==1)]
    
    # groupdf = adf.groupby(["FilePath",
    #                        "ROInum"]).mean()
    
    groupdf = adf.groupby(["FilePath",
                            "ROInum"])
    
    Mean = groupdf["norm_sumIntensity_bg-ROI"].mean()
    print("Mean, ",Mean)

    res_dict[each_wellname] = Mean
    
    
    plt.figure(figsize = [1.5,2])
    plt.plot([-0.4,0.4],[Mean.mean() - 1,Mean.mean() -1],'k-',zorder=101)
    xmin, xmax = plt.gca().get_xlim()
    sns.swarmplot(Mean -1 ,palette = ['gray'])
    plt.title(f"{min_lower} to {min_upper} min after uncaging")
    plt.ylabel("\u0394Volume")
    # plt.ylim([0.4, 2.5])
    plt.xlim([-1,1])
    plt.text(0.5,Mean.mean() - 1,
             "Mean "+str(round(Mean.mean() - 1,2)))
    plt.text(0.5,Mean.mean() - 1 + Mean.std(),
             "Std "+str(round(Mean.std(),2)))
    plt.gca().spines[['right', 'top']].set_visible(False)
    plt.errorbar(0, Mean.mean() - 1, yerr=Mean.std(),
                 lw=2, color='k', capsize=5, capthick=2,zorder=100)
    plt.ylim(swarmplot_ylim)
        
    if save_True:
        savepath = allcombined_df_savepath[:-4]+"_norm_sumIntensity_bg-ROI_mean_swarm.png"
        plt.savefig(savepath, bbox_inches = "tight", dpi = 200)
        savepath = allcombined_df_savepath[:-4]+"_norm_sumIntensity_bg-ROI_mean_swarm.pdf"
        plt.savefig(savepath, bbox_inches = "tight", dpi = 200)
    plt.show()


# import numpy as np
# from scipy.stats import ttest_ind

# # データの例
# group1 = np.array([12.5, 13.1, 14.0, 15.2, 13.3])
# group2 = np.array([15.5, 14.8, 16.1, 14.5, 15.0])

# group1 = res_dict["C1"]
# group2 = res_dict["A2"]

# # Welchのt検定
# t_stat, p_value = ttest_ind(group1, group2, equal_var=False)

# print("t値:", t_stat)
# print("p値:", p_value)