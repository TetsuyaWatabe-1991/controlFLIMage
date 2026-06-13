import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append("../../")
from FLIMageFileReader2 import FileReader

combined_df_pkl_path = r"G:\ImagingData\Tetsuya\20251003\auto1\combined_df_1.pkl"
LTP_point_df_pkl_path = r"G:\ImagingData\Tetsuya\20251003\auto1\LTP_point_df.pkl"

combined_df_df = pd.read_pickle(combined_df_pkl_path)
LTP_point_df = pd.read_pickle(LTP_point_df_pkl_path)


label_dict = {
    "Apical_dist": "Apical_distal",
    "Apical_prox": "Apical_proximal",
    "Basal_": "Basal",
}
LTP_point_df["Region"] = None

for each_label in LTP_point_df["label"].unique():
    
    eachlabel_combined_df = combined_df_df[combined_df_df["label"] == each_label]
    uncaging_df = eachlabel_combined_df[eachlabel_combined_df["uncaging_frame"] == True]
    if (len(uncaging_df)) != 1:
        print(each_label)
        print(len(uncaging_df))
        assert False

    # if label_dict's key is contained in each_label, then add the value to LTP_point_df
    for each_key in label_dict:
        if each_key in each_label:
            LTP_point_df.loc[LTP_point_df["label"] == each_label, "Region"] = label_dict[each_key]
            break
    else:
        print(each_label)
        assert False, f"each_label {each_label} is not in label_dict"

print(LTP_point_df["Region"].unique())

reject_threshold_too_large = 4
reject_threshold_too_small = -2

LTP_point_df["GCaMP_Spine_F_F0"] = LTP_point_df["GCaMP_Spine_F_F0"].astype(float)
LTP_point_df["GCaMP_DendriticShaft_F_F0"] = LTP_point_df["GCaMP_DendriticShaft_F_F0"].astype(float)


#if NaN in LTP_point_df, remove the record
LTP_point_df = LTP_point_df.dropna()


#font arial 12
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'Arial'

plt.figure(figsize=(5, 3))
p = sns.swarmplot(x="Region", y="norm_intensity", data=LTP_point_df)
#show mean using mean line
sns.boxplot(showmeans=True,
            meanline=True,
            meanprops={'color': 'r', 'ls': '-', 'lw': 1},
            medianprops={'visible': False},
            whiskerprops={'visible': False},
            zorder=10,
            x="Region",
            y="norm_intensity",
            data=LTP_point_df,
            showfliers=False,
            showbox=False,
            showcaps=False,
            ax=p)
plt.ylabel("Normalized $\Delta$volume (a.u.)")
plt.xlabel("Region")

for x_pos, Region in zip(p.get_xticks(), label_dict.keys()):
    print(x_pos, Region)
    each_Region_df = LTP_point_df[LTP_point_df["Region"] == Region]
    print(each_Region_df["norm_intensity"].mean())
    p.text(x_pos-0.3, each_Region_df["norm_intensity"].mean(), f"{each_Region_df['norm_intensity'].mean():.2f}", ha='center', va='bottom', fontsize=8, color="black")
    

#delete the right and top axis
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.savefig(os.path.join(os.path.dirname(LTP_point_df_pkl_path), "summary","LTP_vs_laserpow_swarmplot.png"),
             dpi=150,bbox_inches="tight", transparent=True)
plt.show()


# plot same thing but reject extreme delta volume
LTP_point_df_cut_extreme = LTP_point_df[(LTP_point_df["norm_intensity"] < reject_threshold_too_large) 
                                        & (LTP_point_df["norm_intensity"] > reject_threshold_too_small)]

plt.figure(figsize=(5, 3))
p = sns.swarmplot(x="Region", y="norm_intensity", data=LTP_point_df_cut_extreme)
#show mean using mean line
sns.boxplot(showmeans=True,
            meanline=True,
            meanprops={'color': 'r', 'ls': '-', 'lw': 1},
            medianprops={'visible': False},
            whiskerprops={'visible': False},
            zorder=10,
            x="Region",
            y="norm_intensity",
            data=LTP_point_df_cut_extreme,
            showfliers=False,
            showbox=False,
            showcaps=False,
            ax=p)
plt.ylabel("Normalized $\Delta$volume")
plt.xlabel("Region")
for x_pos, Region in zip(p.get_xticks(), label_dict.keys()):
    print(x_pos, Region)
    each_Region_df = LTP_point_df_cut_extreme[LTP_point_df_cut_extreme["Region"] == Region]
    print(each_Region_df["norm_intensity"].mean())
    p.text(x_pos-0.3, each_Region_df["norm_intensity"].mean(), f"{each_Region_df['norm_intensity'].mean():.2f}", ha='center', va='bottom', fontsize=8, color="black")
    
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.savefig(os.path.join(os.path.dirname(LTP_point_df_pkl_path), "summary","LTP_vs_laserpow_swarmplot_cut_extreme.png"),
            dpi=150,bbox_inches="tight", transparent=True) 
plt.show()


# %%
# plot GCaMP Spine F/F0
plt.figure(figsize=(5, 3))
p = sns.swarmplot(x="Region", y="GCaMP_Spine_F_F0", data=LTP_point_df_cut_extreme)
#show mean using mean line
sns.boxplot(showmeans=True,
            meanline=True,
            meanprops={'color': 'r', 'ls': '-', 'lw': 1},
            medianprops={'visible': False},
            whiskerprops={'visible': False},
            zorder=10,
            x="Region",
            y="GCaMP_Spine_F_F0",
            data=LTP_point_df_cut_extreme,
            showfliers=False,
            showbox=False,
            showcaps=False,
            ax=p)
plt.ylabel("GCaMP Spine F/F0")
plt.xlabel("Region")
for x_pos, Region in zip(p.get_xticks(), label_dict.keys()):
    print(x_pos, Region)
    each_Region_df = LTP_point_df_cut_extreme[LTP_point_df_cut_extreme["Region"] == Region]
    print(each_Region_df["GCaMP_Spine_F_F0"].mean())
    p.text(x_pos-0.3, each_Region_df["GCaMP_Spine_F_F0"].mean(), f"{each_Region_df['GCaMP_Spine_F_F0'].mean():.2f}", ha='center', va='bottom', fontsize=8, color="black")
    
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.savefig(os.path.join(os.path.dirname(LTP_point_df_pkl_path), "summary","GCaMP_Spine_F_F0_vs_laserpow_swarmplot_cut_extreme.png"),
            dpi=150,bbox_inches="tight", transparent=True) 
plt.show()


# %%
# plot GCaMP Dendritic Shaft F/F0
plt.figure(figsize=(5, 3))
p = sns.swarmplot(x="Region", y="GCaMP_DendriticShaft_F_F0", data=LTP_point_df_cut_extreme)
#show mean using mean line
sns.boxplot(showmeans=True,
            meanline=True,
            meanprops={'color': 'r', 'ls': '-', 'lw': 1},
            medianprops={'visible': False},
            whiskerprops={'visible': False},
            zorder=10,
            x="Region",
            y="GCaMP_DendriticShaft_F_F0",
            data=LTP_point_df_cut_extreme,
            showfliers=False,
            showbox=False,
            showcaps=False,
            ax=p)
plt.ylabel("GCaMP Dendritic Shaft F/F0")
plt.xlabel("Region")
for x_pos, Region in zip(p.get_xticks(), label_dict.keys()):
    print(x_pos, Region)
    each_Region_df = LTP_point_df_cut_extreme[LTP_point_df_cut_extreme["Region"] == Region]
    print(each_Region_df["GCaMP_DendriticShaft_F_F0"].mean())
    p.text(x_pos-0.3, each_Region_df["GCaMP_DendriticShaft_F_F0"].mean(), f"{each_Region_df['GCaMP_DendriticShaft_F_F0'].mean():.2f}", ha='center', va='bottom', fontsize=8, color="black")
    
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.savefig(os.path.join(os.path.dirname(LTP_point_df_pkl_path), "summary","GCaMP_DendriticShaft_F_F0_vs_laserpow_swarmplot_cut_extreme.png"),
            dpi=150,bbox_inches="tight", transparent=True) 
plt.show()
# %%

for x_pos, uncpow in zip(p.get_xticks(), uncpow_dict.keys()):
    print(x_pos, uncpow)
    each_uncaging_pow_mw_df = LTP_point_df[LTP_point_df["uncaging_pow"] == uncpow]
    #plot volume against GCaMP_Spine_F_F0
    plt.figure(figsize=(3, 3))
    plt.title(f"{uncpow_dict[uncpow]} mW")
    plt.scatter(each_uncaging_pow_mw_df["GCaMP_Spine_F_F0"],each_uncaging_pow_mw_df["norm_intensity"], color="black", s = 10)
    plt.ylabel("$\Delta$volume")
    plt.xlabel("GCaMP Spine F/F0")
    plt.ylim(-0.5,3.7)
    plt.xlim(0,20)
    plt.savefig(os.path.join(os.path.dirname(LTP_point_df_pkl_path), "summary",f"LTP_vs_GCaMP_Spine_F_F0_{uncpow_dict[uncpow]}.png"),
             dpi=150,bbox_inches="tight", transparent=True)
    plt.show()
    

    plt.figure(figsize=(3, 3))
    plt.title(f"{uncpow_dict[uncpow]} mW")
    plt.scatter(each_uncaging_pow_mw_df["GCaMP_DendriticShaft_F_F0"],each_uncaging_pow_mw_df["norm_intensity"], color="black", s = 10)
    plt.ylabel("$\Delta$volume")
    plt.xlabel("GCaMP shaft F/F0")
    plt.ylim(-0.5,3.7)
    plt.xlim(0,20)
    plt.savefig(os.path.join(os.path.dirname(LTP_point_df_pkl_path), "summary",f"LTP_vs_GCaMP_Shaft_F_F0_{uncpow_dict[uncpow]}.png"),
             dpi=150,bbox_inches="tight", transparent=True)
    plt.show()


    

from sklearn.linear_model import LinearRegression
#plot volume against GCaMP_Spine_F_F0
plt.figure(figsize=(3, 3))
plt.title(f"1.6 to 3.3 mW")
plt.scatter(LTP_point_df["GCaMP_Spine_F_F0"],LTP_point_df["norm_intensity"], color="black", s = 10, alpha=0.5)
plt.ylabel("$\Delta$volume")
plt.xlabel("GCaMP Spine F/F0")
plt.ylim(-0.5,3.7)
plt.xlim(0,20)
#trend line with regression r squared
model = LinearRegression()
#ValueError: Expected a 2-dimensional container but got <class 'pandas.core.series.Series'> instead. Pass a DataFrame containing a single row (i.e. single sample) or a single column (i.e. single feature) instead.
gcamp_spine_f_f0 = LTP_point_df["GCaMP_Spine_F_F0"].values.reshape(-1, 1)
norm_intensity = LTP_point_df["norm_intensity"].values.reshape(-1, 1)
model.fit(gcamp_spine_f_f0,norm_intensity)
plt.plot(LTP_point_df["GCaMP_Spine_F_F0"],model.predict(gcamp_spine_f_f0), color="red", label=f"r^2 = {model.score(gcamp_spine_f_f0,norm_intensity):.3f}")
#print the equation and r squared
#unsupported format string passed to numpy.ndarray.__format__
display(f"y = {model.coef_[0][0]:.3f}x + {model.intercept_[0]:.3f}")
display(f"r^2 = {model.score(gcamp_spine_f_f0,norm_intensity):.3f}")
plt.savefig(os.path.join(os.path.dirname(LTP_point_df_pkl_path), "summary",f"LTP_vs_GCaMP_Spine_F_F0_all_power.png"),
            dpi=150,bbox_inches="tight", transparent=True)
plt.show()


plt.figure(figsize=(3, 3))
plt.title(f"1.6 to 3.3 mW")
plt.scatter(LTP_point_df["GCaMP_DendriticShaft_F_F0"],LTP_point_df["norm_intensity"], color="black", s = 10, alpha=0.5)
plt.ylabel("$\Delta$volume")
plt.xlabel("GCaMP shaft F/F0")
plt.ylim(-0.5,3.7)
plt.xlim(0,20)
#trend line with regression r squared
model = LinearRegression()
#ValueError: Expected a 2-dimensional container but got <class 'pandas.core.series.Series'> instead. Pass a DataFrame containing a single row (i.e. single sample) or a single column (i.e. single feature) instead.
gcamp_shaft_f_f0 = LTP_point_df["GCaMP_DendriticShaft_F_F0"].values.reshape(-1, 1)
norm_intensity = LTP_point_df["norm_intensity"].values.reshape(-1, 1)
model.fit(gcamp_shaft_f_f0,norm_intensity)
plt.plot(LTP_point_df["GCaMP_DendriticShaft_F_F0"],model.predict(gcamp_shaft_f_f0), color="red", label=f"r^2 = {model.score(gcamp_shaft_f_f0,norm_intensity):.3f}")
#print the equation and r squared
#unsupported format string passed to numpy.ndarray.__format__
display(f"y = {model.coef_[0][0]:.3f}x + {model.intercept_[0]:.3f}")
display(f"r^2 = {model.score(gcamp_shaft_f_f0,norm_intensity):.3f}") 
plt.savefig(os.path.join(os.path.dirname(LTP_point_df_pkl_path), "summary",f"LTP_vs_GCaMP_Shaft_F_F0_all_power.png"),
            dpi=150,bbox_inches="tight", transparent=True)
plt.show()
