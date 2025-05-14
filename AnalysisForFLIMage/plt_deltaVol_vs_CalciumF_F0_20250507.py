# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 12:56:38 2025

@author: yasudalab
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

csv_path = r"G:\ImagingData\Tetsuya\20250506\onHarp\plot\result - CopyLTP_highmag.csv"
csv_path = r"G:\ImagingData\Tetsuya\20250508\plot\result_LTP.csv"

spine_F_F0_max = 15
shaft_F_F0_max = 15

spine_xlim_max = spine_F_F0_max * 1.1
shaft_xlim_max = shaft_F_F0_max * 1.1

xlim = [0,15]



df = pd.read_csv(csv_path)
df['delta_vol'] = df['spine_vol'] - 1

x_axis = ' dend_F_F0'
y_axis = 'delta_vol'
# plt.figure(figsize = [3,2])
# sns.scatterplot(data = df,
#                 x = x_axis,
#                 y = y_axis)
# plt.show()

plt.figure(figsize = [3,2])
sns.scatterplot(data = df,
                x = x_axis,
                y = y_axis)

plt.xlim(xlim)

plt.ylabel("\u0394spine vol")
plt.xlabel("Dendritic shaft F/F0 ")


savepath = csv_path[:-5]+"_shaftF_F0_vs_deltavol.png"
plt.savefig(savepath, dpi = 150, bbox_inches = "tight")
plt.show()




x_axis = ' spine_F_F0'
y_axis = 'delta_vol'
plt.figure(figsize = [3,2])
sns.scatterplot(data = df,
                x = x_axis,
                y = y_axis)

plt.xlim(xlim)
# plt.ylabel("Spine F/F0")
plt.ylabel("\u0394spine vol")
plt.xlabel("Stimulated spine F/F0 ")

savepath = csv_path[:-5]+"_spineF_F0_vs_deltavol.png"
plt.savefig(savepath, dpi = 150, bbox_inches = "tight")
plt.show()


mean_delta_vol = df['delta_vol'].mean()
SD_delta_vol = df['delta_vol'].std()

# print("mean_delta_vol:", round(mean_delta_vol*100),"(%)" )
# print("SD_delta_vol:", round(SD_delta_vol*100),"(%)" )
# print("Mean with SD:", round(mean_delta_vol*100),"\u00B1" , round(SD_delta_vol*100),"(%)")

print("\n","-"*60)
print("Include all")

print("Mean with SD:", round(mean_delta_vol*100),"\u00B1" , round(SD_delta_vol*100),"(%)",
      "  min:",round(df['delta_vol'].min()*100),
      "  max:",round(df['delta_vol'].max()*100)
      )
print("N = ",len(df))



reject_spine_delta_vol_threshold = 3
reject_very_high_df = df[df['delta_vol'] < reject_spine_delta_vol_threshold]
after_reject_mean_delta_vol = reject_very_high_df['delta_vol'].mean()
after_reject_SD_delta_vol = reject_very_high_df['delta_vol'].std()

print("\n","-"*60)
print("after rejecting very large delta vol (>",reject_spine_delta_vol_threshold*100,"(%)")

print("Mean with SD:", 
      round(after_reject_mean_delta_vol*100),"\u00B1" ,
      round(after_reject_SD_delta_vol*100),"(%)",
            "  min:",round(reject_very_high_df['delta_vol'].min()*100),
            "  max:",round(reject_very_high_df['delta_vol'].max()*100)
            )
print("N = ",len(reject_very_high_df))







shaft_F_F0_threshold = 5
shaft_high_df = df[df[' dend_F_F0'] > shaft_F_F0_threshold]
shaft_high_mean_delta_vol = shaft_high_df['delta_vol'].mean()
shaft_high_SD_delta_vol = shaft_high_df['delta_vol'].std()

print("\n","-"*60)


print("Limited to shaft  F/F0 > ",shaft_F_F0_threshold)


print("Mean with SD:", 
      round(shaft_high_mean_delta_vol*100),"\u00B1" ,
      round(shaft_high_SD_delta_vol*100),"(%)",
            "  min:",round(shaft_high_df['delta_vol'].min()*100),
            "  max:",round(shaft_high_df['delta_vol'].max()*100)
            )

print("N = ",len(shaft_high_df))




