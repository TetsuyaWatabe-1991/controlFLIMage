# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 09:43:39 2023

@author: WatabeT
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 14:51:24 2023

@author: yasudalab
"""
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 11:12:05 2023

@author: yasudalab
"""

import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import csv

from read_flimagecsv import value_normalize,csv_to_df,detect_uncaging,arrange_for_multipos3,everymin_normalize

def pltsetting(plt):
    # plt.rcParams['font.family'] = 'Arial'
    # plt.rcParams['font.size'] = fontsize

    ax=plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_color('#000000')
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['right'].set_color('#000000')
    ax.spines['left'].set_linewidth(1)
    
    ax.tick_params(axis='both', width=1, length=4,pad=2)
    ax.yaxis.labelpad = 4
    ax.xaxis.labelpad = 4
    
    ax.xaxis.label.set_color('#000000')
    ax.yaxis.label.set_color('#000000')
    ax.xaxis.set_tick_params(width=1)
    ax.yaxis.set_tick_params(width=1)
    
def arrange_for_singlepos(resultdf, exclude_first = False,
                          time_min_range=[-20, 43]):
    
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




save_True = True

csvlist=[
r"\\ry-lab-yas15\Users\Yasudalab\Documents\Tetsuya_Imaging\20230907\Analysis\GFP_slice1_aligned_TimeCourse - Copy.csv",
r"\\ry-lab-yas15\Users\Yasudalab\Documents\Tetsuya_Imaging\20230907\Analysis\GFP_slice3_aligned_TimeCourse - Copy.csv",
r"\\ry-lab-yas15\Users\Yasudalab\Documents\Tetsuya_Imaging\20230907\Analysis\GFP_slice2_aligned_ALL_TimeCourse - Copy.csv"
]

csvpath = csvlist[2]
zeroframe_firstframeAs1 = 8

dend_roinum = 4


########################################################################
########################################################################
    
zeroframe = zeroframe_firstframeAs1 - 1

allcombined_df_savepath = csvpath[:-4]+"_modified.csv"

lifetimeshow = False
intensity_ch_1or2 = 1


allcombined_df = pd.DataFrame()

resultdf=csv_to_df(csvpath,
                   ch_list=[1,2])#,
                   # prefix_list=["sumIntensity_bg-ROI"])
resultdf = detect_uncaging(resultdf) 


resultdf.loc[resultdf[resultdf['NthFrame']==zeroframe].index,'first_uncaging']=1



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
    allcombined_df=allcombined_df.append(eachROIdf,ignore_index=True)

allcombined_df = everymin_normalize(allcombined_df)

if save_True==True:
    allcombined_df.to_csv(allcombined_df_savepath)

# allcombined_df = allcombined_df[(allcombined_df["time_min_norm"]<0)|
#                                 (allcombined_df["time_min_norm"]>24)
#                                 ]


##################################
sns.lineplot(x="time_min_norm", y="norm_sumIntensity_bg-ROI",
                legend=False, hue = "CellName", marker='o',
                data = allcombined_df[
                            (allcombined_df['ch']==intensity_ch_1or2)&
                            (allcombined_df['ROInum']!=dend_roinum)
                            ],
                zorder = 10)

sns.lineplot(x="time_min_norm", y="norm_sumIntensity_bg-ROI",
                legend=False, hue = "CellName", marker='',
                linewidth=4,
                data = allcombined_df[
                            (allcombined_df['ch']==intensity_ch_1or2)&
                            (allcombined_df['ROInum']==dend_roinum)
                            ],
                palette=['k'],
                zorder = 1)

plt.plot([allcombined_df["time_min_norm"].min(),
          allcombined_df["time_min_norm"].max()],
         [1,1],c='gray',ls = '--', zorder = 1)

plt.ylabel("Spine volume (a.u.)")
plt.xlabel("Time (min)")
# plt.ylim([0.57,5.7])

uncaging_lineheight = 3.4
plt.plot([0,2],[uncaging_lineheight]*2,"k-")

plt.ylim([0.1,1.71])
currentylim = plt.gca().get_ylim()
plt.text(1,currentylim[1],"NMDA\n|",
         ha="center",va="top",zorder=100)

savepath = allcombined_df_savepath[:-4]+"_norm_sumIntensity_bg-ROI_plot.png"

pltsetting(plt)
if save_True==True:
    plt.savefig(savepath, bbox_inches = "tight", dpi = 200)
plt.show()



    
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