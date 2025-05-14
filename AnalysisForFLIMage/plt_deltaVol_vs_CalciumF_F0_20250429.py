# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 12:56:38 2025

@author: yasudalab
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# csv_path = r"G:\ImagingData\Tetsuya\20250428\B6GC6sTom0331\lowmag\Analysis\copy\lowmag_pos1__highmag_1_concat_TimeCourse_combined_summarized_with_F_F0.csv"
# csv_path = r"G:\ImagingData\Tetsuya\20250428\B6GC6sTom0331\Analysis\copy\pos2_d2_19um_align_TimeCourse_combined_summarized_withF_F0.csv"
csv_path = r"G:\ImagingData\Tetsuya\20250428\B6GC6sTom0331\lowmag\Analysis\overnight_copy\overnight_pos2_align_TimeCourse_combined_summarized_withCaF_F0.csv"

df = pd.read_csv(csv_path)

df['delta_vol'] = df['spine_vol'] - 1

x_axis = ' dend_F_F0'
y_axis = 'delta_vol'
sns.scatterplot(data = df,
                x = x_axis,
                y = y_axis)

plt.show()

plt.figure(figsize = [3,2])
sns.scatterplot(data = df,
                x = x_axis,
                y = y_axis)
# sns.regplot(data = df,
#                 x = x_axis,
#                 y = y_axis)

plt.xlim([0,31])
# plt.ylabel("Spine F/F0")
plt.ylabel("\u0394spine vol")
plt.xlabel("Dendritic shaft F/F0 ")

# savepath = csvpath[:-5]+f"linreg_spine_pow_{each_pow}.png"
# savepath = csvpath[:-5]+f"linreg_shaft_pow_{each_pow}.png"
# plt.savefig(savepath, dpi = 150, bbox_inches = "tight")
plt.show()



df.loc[df[' dend_F_F0'] > 10,
       'delta_vol'
       ].mean()


df.loc[df[' dend_F_F0'] < 10,
       'delta_vol'
       ].mean()


x_axis = ' spine_F_F0'
y_axis = 'delta_vol'
plt.figure(figsize = [3,2])
sns.scatterplot(data = df,
                x = x_axis,
                y = y_axis)
# plt.ylim([0,1.4])
plt.xlim([0,31])
# plt.ylabel("Spine F/F0")
plt.ylabel("\u0394spine vol")
plt.xlabel("Stimulated spine F/F0 ")

# savepath = csvpath[:-5]+f"linreg_spine_pow_{each_pow}.png"
# savepath = csvpath[:-5]+f"linreg_shaft_pow_{each_pow}.png"
# plt.savefig(savepath, dpi = 150, bbox_inches = "tight")
plt.show()






