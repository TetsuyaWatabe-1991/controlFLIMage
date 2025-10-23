import glob
import json
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

savefolder = r"G:\ImagingData\Tetsuya\20250902\analysis"

one_of_json_path_list = [
    r"G:\ImagingData\Tetsuya\20250829\auto1\plot\a1_pos1__highmag_1_007_F_F0_vs_power.json",
    r"G:\ImagingData\Tetsuya\20250902\plot\a3cont_pos1__highmag_1_005_F_F0_vs_power.json"
]

name_to_group = {
    "a1": "cont",
    'a2': 'TlnKO',
    'a3': 'cont',
    'a4': 'TlnKO',
}

json_list = []
for each_one_of_json_path in one_of_json_path_list:
    json_list += glob.glob(os.path.join(os.path.dirname(each_one_of_json_path), "*F_F0_vs_power.json"))


result_df = pd.DataFrame()
for each_json in json_list:
    basename = os.path.basename(each_json).replace(".json", "")
    group = name_to_group[basename.split("_")[0][:2]]
    
    with open(each_json, "r") as f:
        data = json.load(f)

    for each_key in data:
        power_mW = data[each_key]["power_mW"]
        spineF_F0 = data[each_key]["spineF_F0"]
        shaftF_F0 = data[each_key]["shaftF_F0"]
        result_dict = {
            "group": [group],
            "basename": [basename],
            "power_mW": [power_mW],
            "spine_f_f0": [spineF_F0],
            "shaft_f_f0": [shaftF_F0]
        }
        df = pd.DataFrame(result_dict)
        result_df = pd.concat([result_df, df], ignore_index=True)
        
result_df.to_csv(os.path.join(savefolder, "titration_json_analysis.csv"), index=False)

#omit inf and 0 in F/F0 record
result_df2 = result_df[(result_df["spine_f_f0"] != 0) &
                       (result_df["shaft_f_f0"] != 0) & 
                       (result_df["spine_f_f0"] != float("inf")) & 
                       (result_df["shaft_f_f0"] != float("inf"))]

mw_28df = result_df2[result_df2["power_mW"] == 2.8]

#arial 12
plt.rcParams["font.size"] = 12
plt.rcParams["font.family"] = "Arial"


plt.figure(figsize=(3,3))
p =sns.swarmplot(x="group",y="spine_f_f0",data=mw_28df, s=3)
plt.ylabel("Spine F/F0")
plt.xlabel("")
#show mean and std
sns.boxplot(showmeans=True,
            meanline=True,
            meanprops={'color': 'r', 'ls': '-', 'lw': 2},
            medianprops={'visible': False},
            whiskerprops={'visible': False},
            zorder=10,
            x="group",
            y="spine_f_f0",
            data=mw_28df,
            showfliers=False,
            showbox=False,
            showcaps=False,
            ax=p)
# plt.title("Power: 2.8 mW")
plt.xlabel("")
plt.ylim([-0.5,11])
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.savefig(os.path.join(savefolder, "spine_f_f0_2.8mW.png"), dpi=150, bbox_inches="tight",transparent=True)    
plt.show()

plt.figure(figsize=(3,3))
p = sns.swarmplot(x="group",y="shaft_f_f0",data=mw_28df, s=3)
plt.ylabel("Shaft F/F0")
plt.xlabel("")
#show mean and std
sns.boxplot(showmeans=True,
            meanline=True,
            meanprops={'color': 'r', 'ls': '-', 'lw': 2},
            medianprops={'visible': False},
            whiskerprops={'visible': False},
            zorder=10,
            x="group",
            y="shaft_f_f0",
            data=mw_28df,
            showfliers=False,
            showbox=False,
            showcaps=False,
            ax=p)
# plt.title("Power: 2.8 mW")
plt.xlabel("")
plt.ylim([-0.5,11])
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.savefig(os.path.join(savefolder, "shaft_f_f0_2.8mW.png"), dpi=150, bbox_inches="tight",transparent=True)
plt.show()




