# %%
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append("../../")
from FLIMageFileReader2 import FileReader

combined_df_pkl_path = r"G:\ImagingData\Tetsuya\20251108\combined_df1.pkl"
LTP_point_df_pkl_path = r"G:\ImagingData\Tetsuya\20251108\LTP_point_df.pkl"

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

    position = os.path.basename(each_label)[0 : os.path.basename(each_label).find("_highmag")]
    LTP_point_df.loc[LTP_point_df["label"] == each_label, "position"] = position

print(LTP_point_df["uncaging_pow"].unique())
print(LTP_point_df["position"].unique())

# %%
uncpow_dict = {
    34: 2,
    52: 3,
}


reject_threshold_too_large = 4
reject_threshold_too_small = -2


LTP_point_df["uncaging_pow_mw"] = LTP_point_df["uncaging_pow"].map(uncpow_dict)

# below did not work
#LTP_point_df["pos_et_pow_label"] = LTP_point_df["position"] + "_" + str(LTP_point_df["uncaging_pow_mw"])

LTP_point_df["pos_et_pow_label"] = LTP_point_df["position"]  + LTP_point_df["uncaging_pow_mw"].astype(str) + " mW"


LTP_point_df["GCaMP_Spine_F_F0"] = LTP_point_df["GCaMP_Spine_F_F0"].astype(float)
LTP_point_df["GCaMP_DendriticShaft_F_F0"] = LTP_point_df["GCaMP_DendriticShaft_F_F0"].astype(float)


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
plt.ylabel("Normalized $\Delta$volume")
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


# %%
# plot GCaMP Spine F/F0
plt.figure(figsize=(5, 3))
p = sns.swarmplot(x="uncaging_pow_mw", y="GCaMP_Spine_F_F0", data=LTP_point_df_cut_extreme)
#show mean using mean line
sns.boxplot(showmeans=True,
            meanline=True,
            meanprops={'color': 'r', 'ls': '-', 'lw': 1},
            medianprops={'visible': False},
            whiskerprops={'visible': False},
            zorder=10,
            x="uncaging_pow_mw",
            y="GCaMP_Spine_F_F0",
            data=LTP_point_df_cut_extreme,
            showfliers=False,
            showbox=False,
            showcaps=False,
            ax=p)
plt.ylabel("GCaMP Spine F/F0")
plt.xlabel("Uncaging Power (mW)")
plt.ylim(0,30)
for x_pos, uncpow in zip(p.get_xticks(), uncpow_dict.keys()):
    print(x_pos, uncpow)
    each_uncaging_pow_mw_df = LTP_point_df_cut_extreme[LTP_point_df_cut_extreme["uncaging_pow"] == uncpow]
    print(each_uncaging_pow_mw_df["GCaMP_Spine_F_F0"].mean())
    p.text(x_pos-0.3, each_uncaging_pow_mw_df["GCaMP_Spine_F_F0"].mean(), f"{each_uncaging_pow_mw_df['GCaMP_Spine_F_F0'].mean():.2f}", ha='center', va='bottom', fontsize=8, color="black")
    
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.savefig(os.path.join(os.path.dirname(LTP_point_df_pkl_path), "summary","GCaMP_Spine_F_F0_vs_laserpow_swarmplot_cut_extreme.png"),
            dpi=150,bbox_inches="tight", transparent=True) 
plt.show()


# %%
# plot GCaMP Dendritic Shaft F/F0
plt.figure(figsize=(5, 3))
p = sns.swarmplot(x="uncaging_pow_mw", y="GCaMP_DendriticShaft_F_F0", data=LTP_point_df_cut_extreme)
#show mean using mean line
sns.boxplot(showmeans=True,
            meanline=True,
            meanprops={'color': 'r', 'ls': '-', 'lw': 1},
            medianprops={'visible': False},
            whiskerprops={'visible': False},
            zorder=10,
            x="uncaging_pow_mw",
            y="GCaMP_DendriticShaft_F_F0",
            data=LTP_point_df_cut_extreme,
            showfliers=False,
            showbox=False,
            showcaps=False,
            ax=p)
plt.ylabel("GCaMP Dendritic Shaft F/F0")
plt.xlabel("Uncaging Power (mW)")
plt.ylim(0,30)
for x_pos, uncpow in zip(p.get_xticks(), uncpow_dict.keys()):
    print(x_pos, uncpow)
    each_uncaging_pow_mw_df = LTP_point_df_cut_extreme[LTP_point_df_cut_extreme["uncaging_pow"] == uncpow]
    print(each_uncaging_pow_mw_df["GCaMP_DendriticShaft_F_F0"].mean())
    p.text(x_pos-0.3, each_uncaging_pow_mw_df["GCaMP_DendriticShaft_F_F0"].mean(), f"{each_uncaging_pow_mw_df['GCaMP_DendriticShaft_F_F0'].mean():.2f}", ha='center', va='bottom', fontsize=8, color="black")
    
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


LTP_point_df.to_csv(os.path.join(os.path.dirname(LTP_point_df_pkl_path), "summary",f"LTP_point_df_with_uncaging_powe.csv"), index=False)


# %%

ylabel_dict = {
    "norm_intensity": "Normalized $\Delta$volume",
    "GCaMP_Spine_F_F0": "GCaMP Spine F/F0",
    "GCaMP_DendriticShaft_F_F0": "GCaMP Dendritic Shaft F/F0",
}

x_axis = "pos_et_pow_label"
for plotting_object in ylabel_dict.keys():
    plt.figure(figsize=(5, 3))
    p = sns.swarmplot(x=x_axis, y=plotting_object, data=LTP_point_df_cut_extreme)
    #show mean using mean line
    sns.boxplot(showmeans=True,
                meanline=True,
                meanprops={'color': 'r', 'ls': '-', 'lw': 1},
                medianprops={'visible': False},
                whiskerprops={'visible': False},
                zorder=10,
                x=x_axis,
                y=plotting_object,
                data=LTP_point_df_cut_extreme,
                showfliers=False,
                showbox=False,
                showcaps=False,
                ax=p)

    plt.ylabel(ylabel_dict[plotting_object])
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("")
    for x_pos, combined_label in zip(p.get_xticks(), LTP_point_df_cut_extreme[x_axis].unique()):
        print(x_pos, combined_label)
        each_combined_label_df = LTP_point_df_cut_extreme[LTP_point_df_cut_extreme[x_axis] == combined_label]
        print(each_combined_label_df[plotting_object].mean())
        p.text(x_pos-0.3, each_combined_label_df[plotting_object].mean(), f"{each_combined_label_df[plotting_object].mean():.2f}", ha='center', va='bottom', fontsize=8, color="black")
        
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.savefig(os.path.join(os.path.dirname(LTP_point_df_pkl_path), "summary",f"LTP_vs_laserpow_swarmplot_cut_extreme_{plotting_object}_vs_position.png"),
                dpi=150,bbox_inches="tight", transparent=True) 
    plt.show()


# %%



x_axis = "position"
for uncaging_pow in uncpow_dict.keys():
    df_each_pow = LTP_point_df_cut_extreme[LTP_point_df_cut_extreme["uncaging_pow"] == uncaging_pow]
    for plotting_object in ylabel_dict.keys():
        plt.figure(figsize=(5, 3))
        plt.title(f"{uncpow_dict[uncaging_pow]} mW")
        p = sns.swarmplot(x=x_axis, y=plotting_object, data=df_each_pow)
        #show mean using mean line
        sns.boxplot(showmeans=True,
                    meanline=True,
                    meanprops={'color': 'r', 'ls': '-', 'lw': 1},
                    medianprops={'visible': False},
                    whiskerprops={'visible': False},
                    zorder=10,
                    x=x_axis,
                    y=plotting_object,
                    data=df_each_pow,
                    showfliers=False,
                    showbox=False,
                    showcaps=False,
                    ax=p)

        plt.ylabel(ylabel_dict[plotting_object])
        plt.xticks(rotation=45, ha='right')
        plt.xlabel("")
        for x_pos, combined_label in zip(p.get_xticks(), df_each_pow[x_axis].unique()):
            print(x_pos, combined_label)
            each_combined_label_df = df_each_pow[df_each_pow[x_axis] == combined_label]
            print(each_combined_label_df[plotting_object].mean())
            p.text(x_pos-0.3, each_combined_label_df[plotting_object].mean(), f"{each_combined_label_df[plotting_object].mean():.2f}", ha='center', va='bottom', fontsize=8, color="black")
            
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.savefig(os.path.join(os.path.dirname(LTP_point_df_pkl_path), "summary",f"LTP_vs_laserpow_swarmplot_cut_extreme_{plotting_object}_vs_uncaging_pow_{uncaging_pow}.png"),
                    dpi=150,bbox_inches="tight", transparent=True) 
        plt.show()
    plt.close()


# %%
