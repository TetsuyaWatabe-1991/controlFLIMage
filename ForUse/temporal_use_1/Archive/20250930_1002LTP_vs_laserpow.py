import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append("../../")
from FLIMageFileReader2 import FileReader

combined_df_pkl_path = r"G:\ImagingData\Tetsuya\20250930\auto1\combined_df_1.pkl"
LTP_point_df_pkl_path = r"G:\ImagingData\Tetsuya\20250930\auto1\LTP_point_df.pkl"

combined_df_df = pd.read_pickle(combined_df_pkl_path)
LTP_point_df = pd.read_pickle(LTP_point_df_pkl_path)


for each_label in LTP_point_df["label"].unique():
    
    eachlabel_combined_df = combined_df_df[combined_df_df["label"] == each_label]
    uncaging_df = eachlabel_combined_df[eachlabel_combined_df["uncaging_frame"] == True]
    if (len(uncaging_df)) != 1:
        print(each_label)
        print(len(uncaging_df))
        assert False

    uncaging_path = uncaging_df["file_path"].values[0]
    uncaging_iminfo = FileReader()
    uncaging_iminfo.read_imageFile(uncaging_path, False)
    uncaging_pow = uncaging_iminfo.statedict["State.Uncaging.Power"]
    print(uncaging_pow)
    LTP_point_df.loc[LTP_point_df["label"] == each_label, "uncaging_pow"] = uncaging_pow


print(LTP_point_df["uncaging_pow"].unique())

uncpow_dict = {
    67: 3.3,
    68: 3.3,
    59: 3.3,
    43: 2.4,
    49: 2.4
}


for each_unc_pow in LTP_point_df["uncaging_pow"].unique():
    #get number of such data
    print(each_unc_pow, len(LTP_point_df[LTP_point_df["uncaging_pow"] == each_unc_pow]))



reject_threshold_too_large = 4
reject_threshold_too_small = -2


LTP_point_df["uncaging_pow_mw"] = LTP_point_df["uncaging_pow"].map(uncpow_dict)

#if NaN in LTP_point_df, remove the record
LTP_point_df = LTP_point_df.dropna()


#font arial 12
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'Arial'

plt.figure(figsize=(5, 3))
p = sns.swarmplot(x="uncaging_pow_mw", y="norm_intensity", data=LTP_point_df)
#show mean using mean line
sns.boxplot(showmeans=True,
            meanline=True,
            meanprops={'color': 'r', 'ls': '-', 'lw': 1},
            medianprops={'visible': False},
            whiskerprops={'visible': False},
            zorder=10,
            x="uncaging_pow_mw",
            y="norm_intensity",
            data=LTP_point_df,
            showfliers=False,
            showbox=False,
            showcaps=False,
            ax=p)
plt.ylabel("Normalized $\Delta$volume (a.u.)")
plt.xlabel("Uncaging Power (mW)")

for x_pos, uncpow in zip(p.get_xticks(), uncpow_dict.keys()):
    print(x_pos, uncpow)
    each_uncaging_pow_mw_df = LTP_point_df[LTP_point_df["uncaging_pow"] == uncpow]
    print(each_uncaging_pow_mw_df["norm_intensity"].mean())
    p.text(x_pos-0.3, each_uncaging_pow_mw_df["norm_intensity"].mean(), f"{each_uncaging_pow_mw_df['norm_intensity'].mean():.2f}", ha='center', va='bottom', fontsize=8, color="black")
    

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
p = sns.swarmplot(x="uncaging_pow_mw", y="norm_intensity", data=LTP_point_df_cut_extreme)
#show mean using mean line
sns.boxplot(showmeans=True,
            meanline=True,
            meanprops={'color': 'r', 'ls': '-', 'lw': 1},
            medianprops={'visible': False},
            whiskerprops={'visible': False},
            zorder=10,
            x="uncaging_pow_mw",
            y="norm_intensity",
            data=LTP_point_df_cut_extreme,
            showfliers=False,
            showbox=False,
            showcaps=False,
            ax=p)
plt.ylabel("Normalized $\Delta$volume (a.u.)")
plt.xlabel("Uncaging Power (mW)")
for x_pos, uncpow in zip(p.get_xticks(), uncpow_dict.keys()):
    print(x_pos, uncpow)
    each_uncaging_pow_mw_df = LTP_point_df_cut_extreme[LTP_point_df_cut_extreme["uncaging_pow"] == uncpow]
    print(each_uncaging_pow_mw_df["norm_intensity"].mean())
    p.text(x_pos-0.3, each_uncaging_pow_mw_df["norm_intensity"].mean(), f"{each_uncaging_pow_mw_df['norm_intensity'].mean():.2f}", ha='center', va='bottom', fontsize=8, color="black")
    
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.savefig(os.path.join(os.path.dirname(LTP_point_df_pkl_path), "summary","LTP_vs_laserpow_swarmplot_cut_extreme.png"),
            dpi=150,bbox_inches="tight", transparent=True) 
plt.show()






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



LTP_point_df["GCaMP_Spine_F_F0"] = LTP_point_df["GCaMP_Spine_F_F0"].astype(float)
LTP_point_df["GCaMP_DendriticShaft_F_F0"] = LTP_point_df["GCaMP_DendriticShaft_F_F0"].astype(float)
LTP_point_df.to_csv(os.path.join(os.path.dirname(LTP_point_df_pkl_path), "summary", "LTP_point_df_with_uncaging_pow.csv"), index=False)

# %%

with_label_LTP_point_df = pd.read_csv(os.path.join(os.path.dirname(LTP_point_df_pkl_path), "summary", "LTP_point_df_with_condition_label.csv"))
from scipy import stats

extreme_threshold_too_large = 4
extreme_threshold_too_small = -0.9


ltp_point_df_with_unc_pow_cut_extreme = with_label_LTP_point_df[(with_label_LTP_point_df["norm_intensity"] < extreme_threshold_too_large) 
                                        & (with_label_LTP_point_df["norm_intensity"] > extreme_threshold_too_small)]

for each_unc_pow_mw in with_label_LTP_point_df["uncaging_pow_mw"].unique():
    each_df = ltp_point_df_with_unc_pow_cut_extreme[ltp_point_df_with_unc_pow_cut_extreme["uncaging_pow_mw"] == each_unc_pow_mw]
    #swarmplot
    plt.figure(figsize=(3, 3))
    plt.title(f"Uncaging {each_unc_pow_mw} mW")
    p = sns.swarmplot(x="condition", y="norm_intensity", data=each_df)
    plt.ylabel("Normalized $\Delta$volume (a.u.)")
    plt.xlabel("")

    # show mean using mean line
    sns.boxplot(showmeans=True,
                meanline=True,
                meanprops={'color': 'r', 'ls': '-', 'lw': 1},
                medianprops={'visible': False},
                whiskerprops={'visible': False},
                zorder=10,
                x="condition",
                y="norm_intensity",
                data=each_df,
                showfliers=False,
                showbox=False,
                showcaps=False,
                ax=p)

    for x_pos, condition in zip(p.get_xticks(), each_df["condition"].unique()):
        print(x_pos, condition)
        each_condition_df = each_df[each_df["condition"] == condition]
        print(each_condition_df["norm_intensity"].mean())
        p.text(x_pos-0.3, each_condition_df["norm_intensity"].mean(), f"{each_condition_df['norm_intensity'].mean():.2f}", ha='center', va='bottom', fontsize=8, color="black")


    #statistics, t-test between two conditions, welch's t-test
    
    four_lines_df = each_df[each_df["condition"] == "cont"]["norm_intensity"].values
    two_lines_df = each_df[each_df["condition"] == "Tln"]["norm_intensity"].values
    print(stats.ttest_ind(four_lines_df, two_lines_df, equal_var=False))

    #plot the t-test result on the top of the plot
    plt.text(0.5, 0.9, f"t-test: {stats.ttest_ind(four_lines_df, two_lines_df, equal_var=False).pvalue:.3f}", ha='center', va='top', fontsize=8, color="black")

    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.savefig(os.path.join(os.path.dirname(LTP_point_df_pkl_path), 
                        "summary",
                        f"delta_volume_swarmplot_{each_unc_pow_mw}mw_cut_extreme.png"),
             dpi=150,bbox_inches="tight", transparent=True)
    plt.show()



for each_unc_pow_mw in with_label_LTP_point_df["uncaging_pow_mw"].unique():
    each_df = ltp_point_df_with_unc_pow_cut_extreme[ltp_point_df_with_unc_pow_cut_extreme["uncaging_pow_mw"] == each_unc_pow_mw]
    #swarmplot 
    plt.figure(figsize=(3, 3))
    plt.title(f"Uncaging {each_unc_pow_mw} mW")
    p = sns.swarmplot(x="condition", y="GCaMP_Spine_F_F0", data=each_df)
    plt.ylabel("GCaMP Spine F/F0")
    plt.xlabel("")

    # show mean using mean line
    sns.boxplot(showmeans=True,
                meanline=True,
                meanprops={'color': 'r', 'ls': '-', 'lw': 1},
                medianprops={'visible': False},
                whiskerprops={'visible': False},
                zorder=10,
                x="condition",
                y="GCaMP_Spine_F_F0",
                data=each_df,
                showfliers=False,
                showbox=False,
                showcaps=False,
                ax=p)

    for x_pos, condition in zip(p.get_xticks(), each_df["condition"].unique()):
        print(x_pos, condition)
        each_condition_df = each_df[each_df["condition"] == condition]
        print(each_condition_df["GCaMP_Spine_F_F0"].mean())
        p.text(x_pos-0.3, each_condition_df["GCaMP_Spine_F_F0"].mean(), f"{each_condition_df['GCaMP_Spine_F_F0'].mean():.2f}", ha='center', va='bottom', fontsize=8, color="black")


    #statistics, t-test between two conditions, welch's t-test
    
    cont_values = each_df[each_df["condition"] == "cont"]["GCaMP_Spine_F_F0"].astype(float).values
    tln_values = each_df[each_df["condition"] == "Tln"]["GCaMP_Spine_F_F0"].astype(float).values
    print(stats.ttest_ind(cont_values, tln_values, equal_var=False))

    #plot the t-test result on the top of the plot
    plt.text(0.5, 0.9, f"t-test: {stats.ttest_ind(cont_values, tln_values, equal_var=False).pvalue:.3f}", ha='center', va='top', fontsize=8, color="black")

    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.savefig(os.path.join(os.path.dirname(LTP_point_df_pkl_path), 
                        "summary",
                        f"LTP_vs_GCaMP_Spine_F_F0_{each_unc_pow_mw}mw_cut_extreme.png"),
             dpi=150,bbox_inches="tight", transparent=True)
    plt.show()




# %%

# shaft GCaMP F/F0
for each_unc_pow_mw in with_label_LTP_point_df["uncaging_pow_mw"].unique():
    each_df = ltp_point_df_with_unc_pow_cut_extreme[ltp_point_df_with_unc_pow_cut_extreme["uncaging_pow_mw"] == each_unc_pow_mw]
    #swarmplot 
    plt.figure(figsize=(3, 3))
    plt.title(f"Uncaging {each_unc_pow_mw} mW")
    p = sns.swarmplot(x="condition", y="GCaMP_DendriticShaft_F_F0", data=each_df)
    plt.ylabel("GCaMP shaft F/F0")
    plt.xlabel("")

    # show mean using mean line
    sns.boxplot(showmeans=True,
                meanline=True,
                meanprops={'color': 'r', 'ls': '-', 'lw': 1},
                medianprops={'visible': False},
                whiskerprops={'visible': False},
                zorder=10,
                x="condition",
                y="GCaMP_DendriticShaft_F_F0",
                data=each_df,
                showfliers=False,
                showbox=False,
                showcaps=False,
                ax=p)

    for x_pos, condition in zip(p.get_xticks(), each_df["condition"].unique()):
        print(x_pos, condition)
        each_condition_df = each_df[each_df["condition"] == condition]
        print(each_condition_df["GCaMP_DendriticShaft_F_F0"].mean())
        p.text(x_pos-0.3, each_condition_df["GCaMP_DendriticShaft_F_F0"].mean(), f"{each_condition_df['GCaMP_DendriticShaft_F_F0'].mean():.2f}", ha='center', va='bottom', fontsize=8, color="black")


    #statistics, t-test between two conditions, welch's t-test
    
    cont_values = each_df[each_df["condition"] == "cont"]["GCaMP_DendriticShaft_F_F0"].astype(float).values
    tln_values = each_df[each_df["condition"] == "Tln"]["GCaMP_DendriticShaft_F_F0"].astype(float).values
    print(stats.ttest_ind(cont_values, tln_values, equal_var=False))

    #plot the t-test result on the top of the plot
    plt.text(0.5, 0.9, f"t-test: {stats.ttest_ind(cont_values, tln_values, equal_var=False).pvalue:.3f}", ha='center', va='top', fontsize=8, color="black")

    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.savefig(os.path.join(os.path.dirname(LTP_point_df_pkl_path), 
                        "summary",
                        f"LTP_vs_GCaMP_DendriticShaft_F_F0_{each_unc_pow_mw}mw_cut_extreme.png"),
             dpi=150,bbox_inches="tight", transparent=True)
    plt.show()


# %%
# spine GCaMP F/F0
for each_unc_pow_mw in with_label_LTP_point_df["uncaging_pow_mw"].unique():
    each_df = ltp_point_df_with_unc_pow_cut_extreme[ltp_point_df_with_unc_pow_cut_extreme["uncaging_pow_mw"] == each_unc_pow_mw]
    #swarmplot 
    plt.figure(figsize=(3, 3))
    plt.title(f"Uncaging {each_unc_pow_mw} mW")
    p = sns.swarmplot(x="condition", y="GCaMP_Spine_F_F0", data=each_df)
    plt.ylabel("GCaMP spine F/F0")
    plt.xlabel("")

    # show mean using mean line
    sns.boxplot(showmeans=True,
                meanline=True,
                meanprops={'color': 'r', 'ls': '-', 'lw': 1},
                medianprops={'visible': False},
                whiskerprops={'visible': False},
                zorder=10,
                x="condition",
                y="GCaMP_Spine_F_F0",
                data=each_df,
                showfliers=False,
                showbox=False,
                showcaps=False,
                ax=p)

    for x_pos, condition in zip(p.get_xticks(), each_df["condition"].unique()):
        print(x_pos, condition)
        each_condition_df = each_df[each_df["condition"] == condition]
        print(each_condition_df["GCaMP_Spine_F_F0"].mean())
        p.text(x_pos-0.3, each_condition_df["GCaMP_Spine_F_F0"].mean(), f"{each_condition_df['GCaMP_Spine_F_F0'].mean():.2f}", ha='center', va='bottom', fontsize=8, color="black")


    #statistics, t-test between two conditions, welch's t-test
    
    cont_values = each_df[each_df["condition"] == "cont"]["GCaMP_Spine_F_F0"].astype(float).values
    tln_values = each_df[each_df["condition"] == "Tln"]["GCaMP_Spine_F_F0"].astype(float).values
    print(stats.ttest_ind(cont_values, tln_values, equal_var=False))

    #plot the t-test result on the top of the plot
    plt.text(0.5, 0.9, f"t-test: {stats.ttest_ind(cont_values, tln_values, equal_var=False).pvalue:.3f}", ha='center', va='top', fontsize=8, color="black")

    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.savefig(os.path.join(os.path.dirname(LTP_point_df_pkl_path), 
                        "summary",
                        f"LTP_vs_GCaMP_Spine_F_F0_{each_unc_pow_mw}mw_cut_extreme.png"),
             dpi=150,bbox_inches="tight", transparent=True)
    plt.show()

# %%

for each_condition in with_label_LTP_point_df["condition"].unique():
    each_df = with_label_LTP_point_df[with_label_LTP_point_df["condition"] == each_condition]
    
    for each_unc_pow_mw in with_label_LTP_point_df["uncaging_pow_mw"].unique():
        print(each_condition, each_unc_pow_mw)
        each_uncaging_pow_mw_df = each_df[each_df["uncaging_pow_mw"] == each_unc_pow_mw]
        #plot volume against GCaMP_Spine_F_F0
        plt.figure(figsize=(3, 3))
        plt.title(f"{each_condition} {each_unc_pow_mw} mW")
        plt.scatter(each_uncaging_pow_mw_df["GCaMP_Spine_F_F0"],each_uncaging_pow_mw_df["norm_intensity"], color="black", s = 10)
        plt.ylabel("$\Delta$volume")
        plt.xlabel("GCaMP Spine F/F0")
        plt.ylim(-0.5,3.7)
        plt.xlim(0,20)
        plt.savefig(os.path.join(os.path.dirname(LTP_point_df_pkl_path), "summary",f"LTP_vs_GCaMP_Spine_F_F0_{each_condition}_{each_unc_pow_mw}mw.png"),
                dpi=150,bbox_inches="tight", transparent=True)
        plt.show()
        

        plt.figure(figsize=(3, 3))
        plt.title(f"{each_condition} {each_unc_pow_mw} mW")
        plt.scatter(each_uncaging_pow_mw_df["GCaMP_DendriticShaft_F_F0"],each_uncaging_pow_mw_df["norm_intensity"], color="black", s = 10)
        plt.ylabel("$\Delta$volume")
        plt.xlabel("GCaMP shaft F/F0")
        plt.ylim(-0.5,3.7)
        plt.xlim(0,20)
        plt.savefig(os.path.join(os.path.dirname(LTP_point_df_pkl_path), "summary",f"LTP_vs_GCaMP_Shaft_F_F0_{each_condition}_{each_unc_pow_mw}mw.png"),
                dpi=150,bbox_inches="tight", transparent=True)
        plt.show()
        

# %%
for each_condition in with_label_LTP_point_df["condition"].unique():
    each_df = with_label_LTP_point_df[with_label_LTP_point_df["condition"] == each_condition]
    
    for each_unc_pow_mw in with_label_LTP_point_df["uncaging_pow_mw"].unique():
        print(each_condition, each_unc_pow_mw)
        print("N = ", len(each_df[each_df["uncaging_pow_mw"] == each_unc_pow_mw]))
        print()
        print("--------------------------------")
        
# %%
