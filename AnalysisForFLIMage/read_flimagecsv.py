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

def csv_to_df(csvpath,ch_list=[1],
              prefix_list=['Lifetime-ROI',
                           'Lifetime_fit-ROI',
                           'Fraction2-ROI',
                           'Fraction2_fit-ROI',
                           'meanIntensity-ROI',
                           'meanIntensity_bg-ROI',
                           'sumIntensity-ROI',
                           'sumIntensity_bg-ROI',
                           'nPixels-ROI']
                            ):
    resultdf=pd.DataFrame()
    with open(csvpath) as f:
        reader = csv.reader(f)
    
        for row in reader:
            
            try:
                row0=row[0]
            except:
                row0=""
                
            if row0=="nROIs":
                nROIs=int(row[1])
                # print(nROIs)
    
            if row0=="Time (s)":
                timelist=[float(x) for x in row[1:-1]]
                   
        for i in range(len(timelist)):
            for ch in ch_list:
                for ROInum in range(1,nROIs+1):
                    resultdf=resultdf.append({
                        "FilePath":csvpath,
                        "ROInum":ROInum,
                        "time_sec":timelist[i],
                        "NthFrame":i,
                        "ch":ch,
                        },ignore_index=True)
    # print(resultdf)
    for ch in ch_list:
        for prefix in prefix_list:
            for ROInum in range(1,nROIs+1):
                rowname=prefix+str(ROInum)+f"-ch{ch}"
                
                with open(csvpath) as f:
                    reader = csv.reader(f)
                    for row in reader:
                        try:
                            row0=row[0]
                        except:
                            row0=""
                        if row0==rowname:
                            idxlist=resultdf[(resultdf["ROInum"]==ROInum)&
                                              (resultdf["ch"]==ch)].index         
                            resultdf.loc[idxlist,prefix]=[float(x) for x in row[1:-1]]
                            
               
    resultdf["Time_min"]=resultdf["time_sec"]/60
    resultdf["strROInum"]=resultdf["ROInum"].astype(int).astype(str)
        
    return resultdf



def detect_uncaging(resultdf, time_threshold = 5):
    
    resultdf["during_uncaging"]=0
    resultdf["first_uncaging"]=0
    
    for ch in resultdf['ch'].unique():
        
        for ROInum in resultdf["ROInum"].unique(): 
            first_uncaging = True
            previous_one_short = False
        
            for NthFrame in resultdf["NthFrame"].unique()[:-1]:
                Time_N = resultdf[(resultdf["ch"]==ch)&
                                  (resultdf["NthFrame"]==NthFrame)&
                                 (resultdf["ROInum"]==ROInum)]["time_sec"]
                Time_Nplus1 = resultdf[(resultdf["ch"]==ch)&
                                       (resultdf["NthFrame"]==NthFrame+1)&
                                       (resultdf["ROInum"]==ROInum)]["time_sec"]
                
                rownum = Time_N.keys()
                delta_sec = float(Time_Nplus1.values - Time_N.values)
                
                resultdf.loc[rownum,"delta_sec"] = delta_sec
                
                if delta_sec <= time_threshold:
                    resultdf.loc[rownum,"during_uncaging"] = 1
                    previous_one_short = True
                    if first_uncaging == True:
                        first_uncaging = False
                        resultdf.loc[rownum,"first_uncaging"]=1
                    
                elif previous_one_short == True:
                    resultdf.loc[rownum,"during_uncaging"] = 1
                    previous_one_short = False
                else:
                    resultdf.loc[rownum,"during_uncaging"] = 0
    return resultdf
    


def arrange_for_multipos3(resultdf, exclude_first = True,
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


def value_normalize(resultdf,
                    prefix = "sumIntensity_bg-ROI",
                    normalize_subtraction = False):
    newdf = pd.DataFrame()
    
    
    for ch in resultdf['ch'].unique():
        
        for ROInum in resultdf["ROInum"].unique():
            
            eachROIdf = resultdf[
                                (resultdf["ROInum"] ==  ROInum)&
                                (resultdf["ch"] ==  ch)
                                ]
            
            pre_value = eachROIdf[eachROIdf["time_sec_norm"]<0][prefix].mean()
            
            if normalize_subtraction == True:
                eachROIdf["norm_" + prefix] = eachROIdf[prefix] - pre_value
            else:
                eachROIdf["norm_" + prefix] = eachROIdf[prefix]/pre_value
             
            newdf = pd.concat([newdf,eachROIdf]) 
    
    return newdf



def everymin_normalize(allcombined_df,
                       bin_percent_median=0.99,
                       time_threshold = 5):
    
    bin_sec = allcombined_df[allcombined_df['delta_sec']>time_threshold]['delta_sec'].median()*bin_percent_median    
    allcombined_df["binned_sec"] = bin_sec*(allcombined_df["time_sec_norm"]//bin_sec)
    allcombined_df["binned_min"] = allcombined_df["binned_sec"]/60
    return allcombined_df


if __name__ == "__main__":

    # csvlist=[r"\\ry-lab-yas16\Users\yasudalab\Documents\Data\Tetsuya\2022\08262022\Analysis\concatenated_GFP_cell2_spine1_alined_TimeCourse.csv",
    #           r"\\ry-lab-yas16\Users\yasudalab\Documents\Data\Tetsuya\2022\08262022\Analysis\concatenated_GFP_cell3_aligned_TimeCourse.csv",
    #           r"\\ry-lab-yas16\Users\yasudalab\Documents\Data\Tetsuya\2022\08252022\Analysis\concatenated_GFP_cell2_spine1_alined_TimeCourse.csv",
    #           r"\\ry-lab-yas16\Users\yasudalab\Documents\Data\Tetsuya\2022\08252022\Analysis\concatenated_GFP_cell1_lower_spine2_alined_TimeCourse.csv",
    #           r"\\ry-lab-yas16\Users\yasudalab\Documents\Data\Tetsuya\2022\08252022\Analysis\concatenated_GFP_cell1_lower_spine3_alined_TimeCourse.csv"]
    
    
    csvlist = glob.glob(r"C:\Users\Yasudalab\Documents\Tetsuya_Imaging\20230713\set2_Rab10CY\Analysis\*Copy.csv")
    
    allcombined_df_savepath=r"C:\Users\Yasudalab\Documents\Tetsuya_Imaging\20230713\set2_Rab10CY\Analysis\combined.csv"

    
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
        
        resultdf = value_normalize(resultdf,
                            prefix = "Fraction2_fit-ROI",
                            normalize_subtraction = True)
        
        resultdf = resultdf[resultdf["during_uncaging"]==0]
                            
        for ROInum in resultdf["ROInum"].unique(): 
            eachROIdf = resultdf[resultdf["ROInum"] ==  ROInum]
    
            eachROIdf["CellName"] = resultdf.loc[:,"FilePath"]+"_"+str(ROInum)
            allcombined_df=allcombined_df.append(eachROIdf,ignore_index=True)
    
    
    allcombined_df = everymin_normalize(allcombined_df)
    
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
    # plt.ylim([0.57,5.7])
    
    uncaging_lineheight = 4
    plt.plot([0,2],[uncaging_lineheight]*2,"k-")
    plt.text(1,uncaging_lineheight*1.02,"Uncaging",
             ha="center",va="bottom",zorder=100)
    
    savepath = allcombined_df_savepath[:-4]+"_norm_sumIntensity_bg-ROI_plot.png"
    plt.savefig(savepath, bbox_inches = "tight", dpi = 200)
    plt.show()
    
    ##################################
    
    sns.lineplot(x="binned_min", y="norm_sumIntensity_bg-ROI",
                    legend=False, hue = "CellName", marker='o',
                    data = allcombined_df[allcombined_df['ch']==1],
                    zorder = 10)
    
    plt.plot([allcombined_df["time_min_norm"].min(),
              allcombined_df["time_min_norm"].max()],
             [1,1],c='gray',ls = '--', zorder = 1)
    
    plt.ylabel("Spine volume (a.u.)")
    plt.xlabel("Time (min)")
    
    uncaging_lineheight = 4
    plt.plot([0,2],[uncaging_lineheight]*2,"k-")
    plt.text(1,uncaging_lineheight*1.02,"Uncaging",
             ha="center",va="bottom")
    
    plt.show()
    
    
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
    plt.savefig(savepath, bbox_inches = "tight", dpi = 200)
    
    plt.show()
    
    ##################################