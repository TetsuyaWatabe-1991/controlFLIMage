import json
import os
import glob
from collections import defaultdict
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd

json_path = r"\\RY-LAB-WS04\ImagingData\Tetsuya\20250701\auto1\plot\lowmag2__highmag_3_061_F_F0_vs_power.json"
plot_folder = os.path.join(os.path.dirname(json_path), "plot_F_F0_vs_power")
os.makedirs(plot_folder, exist_ok=True)
combined_plot_folder = os.path.join(os.path.dirname(plot_folder), "combined_plot")
os.makedirs(combined_plot_folder, exist_ok=True)

json_key = os.path.join(os.path.dirname(json_path), "*F_F0_vs_power.json")
json_list = glob.glob(json_key)

each_json_list = []
for each_json_path in json_list:
    with open(each_json_path, "r") as f:
        data = json.load(f)
    flim_path = list(data.keys())[0]
    print(flim_path)
    each_json_list.append(data)

power_mW_set = set()
spineF_F0_max = 0
shaftF_F0_max = 0
for each_json in each_json_list:
    for each_key in each_json.keys():
        power_mW_set.add(each_json[each_key]["power_mW"])
        if each_json[each_key]["spineF_F0"] > spineF_F0_max:
            spineF_F0_max = each_json[each_key]["spineF_F0"]
        if each_json[each_key]["shaftF_F0"] > shaftF_F0_max:
            shaftF_F0_max = each_json[each_key]["shaftF_F0"]

ymax = (max(spineF_F0_max, shaftF_F0_max)//2 +1) *2
ymin = 0

power_mW_list = list(power_mW_set)
power_mW_list.sort()
print(power_mW_list)
xmax = max(power_mW_list)//1 +1
xmin = min(power_mW_list)//1 -1

start_of_imaging = datetime(2025, 7, 1, 22, 0, 0)
result_df = pd.DataFrame()
for each_json in each_json_list:
    power_mW_list = []
    spine_F_F0_list = []
    shaft_F_F0_list = []
    flim_stemname_list = []
    flim_time_diff_hours_list = []
    for each_key in each_json.keys():
        flim_path = each_key
        flim_stemname = os.path.basename(flim_path).replace(".flim", "")
        flim_num = flim_stemname.split("_")[-1]
        power_mW = each_json[each_key]["power_mW"]
        spine_F_F0 = each_json[each_key]["spineF_F0"]
        shaft_F_F0 = each_json[each_key]["shaftF_F0"]

        power_mW_list.append(power_mW)
        spine_F_F0_list.append(spine_F_F0)
        shaft_F_F0_list.append(shaft_F_F0)
        flim_stemname_list.append(flim_stemname)
        
        #get the create date of the flimfile
        flim_create_date = os.path.getctime(flim_path)
        flim_create_date = datetime.fromtimestamp(flim_create_date)     
        time_diff = flim_create_date - start_of_imaging
        time_diff_seconds = time_diff.total_seconds()
        time_diff_minutes = time_diff_seconds / 60
        time_diff_hours = round(time_diff_minutes / 60, 1)
        flim_time_diff_hours_list.append(time_diff_hours)

    each_df = pd.DataFrame({
        "flim_stem_without_num": [flim_stemname_list[0][:-4]]*len(flim_stemname_list),
        "flim_stemname": flim_stemname_list,
        "power_mW": power_mW_list,
        "spine_F_F0": spine_F_F0_list,
        "shaft_F_F0": shaft_F_F0_list,
        "time_diff_hours": flim_time_diff_hours_list
    })
    result_df = pd.concat([result_df, each_df])

    #plot spine_F_F0 against power_mW in red circle
    plt.scatter(power_mW_list, spine_F_F0_list, color="red", marker="o")
    #plot shaft_F_F0 against power_mW in blue square
    plt.scatter(power_mW_list, shaft_F_F0_list, color="blue", marker="s")
    plt.title(f"{flim_stemname_list[0]},  {time_diff_hours} hours")
    plt.ylim(ymin, ymax)
    plt.xlim(xmin, xmax)
    plt.xlabel("Power (mW)")
    plt.ylabel("GCaMP F/F0 (a.u.)")
    plt.legend(["spine", "shaft"])
    plt.savefig(os.path.join(plot_folder, f"plot_{flim_stemname}.png"))
    
    plt.show()
    plt.close()

result_df.to_csv(os.path.join(plot_folder, "result_df.csv"), index=False)

hour_max = max(result_df["time_diff_hours"])
hour_x_max = hour_max//1 +2

for each_power_mW in power_mW_list:
    each_df = result_df[result_df["power_mW"] == each_power_mW]
    plt.scatter(each_df["time_diff_hours"], each_df["spine_F_F0"], color="red", marker="o")
    plt.scatter(each_df["time_diff_hours"], each_df["shaft_F_F0"], color="blue", marker="s")
    plt.title(f"GCaMP F/F0 vs Time, Power = {each_power_mW} mW")
    plt.xlabel("Time (hours)")
    plt.ylabel("GCaMP F/F0 (a.u.)")
    plt.legend(["spine", "shaft"])
    plt.xlim(0, hour_x_max)
    plt.ylim(ymin, ymax)
    plt.savefig(os.path.join(combined_plot_folder, f"plot_F_F0_vs_time_power_{each_power_mW}.png"))
    plt.show()
    plt.close()

for each_flim_stem_without_num in result_df["flim_stem_without_num"].unique():
    for each_power_mW in power_mW_list:
        each_df = result_df[(result_df["flim_stem_without_num"] == each_flim_stem_without_num) & (result_df["power_mW"] == each_power_mW)]
        plt.scatter(each_df["time_diff_hours"], each_df["spine_F_F0"], color="red", marker="o")
        plt.scatter(each_df["time_diff_hours"], each_df["shaft_F_F0"], color="blue", marker="s")
        plt.title(f"GCaMP F/F0 vs Time, {each_flim_stem_without_num}, Power = {each_power_mW} mW")
        plt.xlabel("Time (hours)")
        plt.ylabel("GCaMP F/F0 (a.u.)")
        plt.legend(["spine", "shaft"])
        plt.xlim(0, hour_x_max)
        plt.ylim(ymin, ymax)
        plt.savefig(os.path.join(combined_plot_folder, f"plot_F_F0_vs_time_flim_{each_flim_stem_without_num}_power_{each_power_mW}.png"))
        plt.show()
        plt.close()

