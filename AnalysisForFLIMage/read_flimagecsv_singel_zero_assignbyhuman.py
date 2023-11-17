# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 22:14:35 2023

@author: yasudalab
"""


import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import csv

from read_flimagecsv import value_normalize,csv_to_df,assing_before, everymin_normalize

def arrange_for_singlepos(resultdf, exclude_first = False,
                          time_min_range=[-20, 60]):
    
    newdf = pd.DataFrame()

    for ROInum in resultdf["ROInum"].unique(): 
        eachROIdf = resultdf[resultdf["ROInum"] ==  ROInum]
        zerotime = eachROIdf[eachROIdf["first_uncaging"] == 1]["time_sec"].values[0]
        eachROIdf["time_sec_norm"] = eachROIdf["time_sec"] - zerotime
        eachROIdf["time_min_norm"] = eachROIdf["time_sec_norm"]/60
        
        if exclude_first == True:
            eachROIdf = eachROIdf[~ (eachROIdf["NthFrame"] == 0)]
            
        eachROIdf = eachROIdf[(eachROIdf["time_min_norm"] >= time_min_range[0])&
                              (eachROIdf["time_min_norm"] <= time_min_range[1])]
 
        newdf = pd.concat([newdf,eachROIdf]) 
    
    return newdf



if __name__ == "__main__":
    
    save_True = True

    csvlist=[
    r"G:\ImagingData\Tetsuya\20231116\Analysis\GFP_2_aligned_TimeCourse - Copy.csv",
    r"G:\ImagingData\Tetsuya\20231116\Analysis\GFP_3_aligned_TimeCourse - Copy.csv",
    r"G:\ImagingData\Tetsuya\20231116\Analysis\GFP_4_aligned_TimeCourse - Copy.csv"
    ]
    
    beforelist = [
                [3,6],
                [0,3],
                [0,3]
                ]
    ylimlist = [
                [0.5,4.1],
                [0.5,3.6],
                [0.5,3.1]
                ]
    
    uncaging_lineheight = 2.8
    
    lifetimeshow = False
    nthlist = 0
    intensity_ch_1or2 = 2
    

    csvpath = csvlist[nthlist]
    before = beforelist[nthlist]
    ylim = ylimlist[nthlist]
    
    allcombined_df_savepath = csvpath[:-4]+"_modified.csv"
    
    
    
    allcombined_df = pd.DataFrame()
    
    resultdf=csv_to_df(csvpath,
                       ch_list=[1,2])#,
                       # prefix_list=["sumIntensity_bg-ROI"])
    resultdf = assing_before(resultdf,before)
    resultdf = arrange_for_singlepos(resultdf)
    resultdf = value_normalize(resultdf,
                               prefix = "sumIntensity_bg-ROI")

    resultdf = value_normalize(resultdf,
                        prefix = "Fraction2_fit-ROI",
                        normalize_subtraction = True)

    # resultdf = resultdf[resultdf["during_uncaging"]==0]
                        
    for ROInum in resultdf["ROInum"].unique(): 
        eachROIdf = resultdf[resultdf["ROInum"] ==  ROInum]

        eachROIdf["CellName"] = resultdf.loc[:,"FilePath"]+"_"+str(ROInum)
        allcombined_df=pd.concat([allcombined_df,eachROIdf],ignore_index=True)
    
    # allcombined_df = everymin_normalize(allcombined_df)
    
    if save_True==True:
        allcombined_df.to_csv(allcombined_df_savepath)
    
    # allcombined_df = allcombined_df[(allcombined_df["time_min_norm"]<0)|
    #                                 (allcombined_df["time_min_norm"]>24)
    #                                 ]


    ##################################
    sns.lineplot(x="time_min_norm", y="norm_sumIntensity_bg-ROI",
                    legend=False, hue = "CellName", marker='o',
                    data = allcombined_df[allcombined_df['ch']==intensity_ch_1or2],
                    zorder = 10)
    
    plt.plot([allcombined_df["time_min_norm"].min(),
              allcombined_df["time_min_norm"].max()],
             [1,1],c='gray',ls = '--', zorder = 1)
    
    plt.ylabel("Spine volume (a.u.)")
    plt.xlabel("Time (min)")
    # plt.ylim([0.57,5.7])
    
    
    plt.plot([0,0.1],[uncaging_lineheight]*2,"k-")
    plt.text(1,uncaging_lineheight*1.02,"Uncaging",
             ha="center",va="bottom",zorder=100)
    plt.ylim(ylim)
    
    savepath = allcombined_df_savepath[:-4]+"_norm_sumIntensity_bg-ROI_plot.png"
    
    
    if save_True==True:
        plt.savefig(savepath, bbox_inches = "tight", dpi = 200)
    plt.show()
    
    ##################################
    # sns.lineplot(x="binned_min", y="norm_sumIntensity_bg-ROI",
    #                 legend=False, hue = "CellName", marker='o',
    #                 data = allcombined_df[allcombined_df['ch']==intensity_ch_1or2],
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
    
    
        
    if lifetimeshow == True:

        ################################
        
        sns.lineplot(x="time_min_norm", y="Fraction2_fit-ROI",
                        legend=False, hue = "CellName", marker='o',
                        data = allcombined_df[allcombined_df['ch']==1])
        plt.ylabel("Spine volume (a.u.)")
        plt.xlabel("Time (min)")
        
        # plt.plot([0,2],[2,2],"k-")
        # plt.text(1,2.05,"Uncaging",ha="center",va="bottom")
        plt.ylim([0.0,0.9])  
        plt.show()
        
        
        ################################
        
        sns.lineplot(x="time_min_norm", y="norm_Fraction2_fit-ROI",
                        legend=False, hue = "CellName", marker='o',
                        data = allcombined_df[allcombined_df['ch']==1],
                        zorder = 10)
        
        plt.plot([allcombined_df["time_min_norm"].min(),
                  allcombined_df["time_min_norm"].max()],
                 [0,0],c='gray',ls = '--', zorder = 1)
        
        plt.ylabel("\u0394 Fraction2")
        plt.xlabel("Time (min)")
        
        uncaging_lineheight = 0.5
        plt.plot([0,2],[uncaging_lineheight]*2,"k-")
        plt.text(1,uncaging_lineheight*1.02,"Uncaging",
                 ha="center",va="bottom",zorder=100)
        
        plt.ylim([-0.9,0.9])  
        
        savepath = allcombined_df_savepath[:-4]+"_norm_Fraction2_plot.png"

        if save_True==True:
            plt.savefig(savepath, bbox_inches = "tight", dpi = 200)
            
        plt.show()
        
        ##################################