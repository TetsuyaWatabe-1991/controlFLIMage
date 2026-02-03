# %%
import pandas as pd
import os
import sys
sys.path.append("../../")
from custom_plot import plt
import seaborn as sns

combined_LTP_point_df_csv_path = r"Z:\User-Personal\Tetsuya_Zdrive\Data\202601\20260110\auto1\summary\LTP_point_df_after_analysis_combine0108_0110.csv"

combined_LTP_point_df = pd.read_csv(combined_LTP_point_df_csv_path)

#omit nan
combined_LTP_point_df = combined_LTP_point_df.dropna()


# %%
plot_y_of_interests_dict = {
    "norm_intensity": "Normalized $\Delta$volume (a.u.)",
    "GCaMP_Spine_F_F0": "GCaMP Spine F/F0",
    "GCaMP_DendriticShaft_F_F0": "GCaMP Dendritic Shaft F/F0",
}

each_y_of_interest_reject_range_dict = {
    "norm_intensity": (-0.5, 3.7),
    "GCaMP_Spine_F_F0": (0, 60),
    "GCaMP_DendriticShaft_F_F0": (0, 60),
}

# plot swamplot for each y_of_interest in plot_y_of_interests_dict
for each_y_of_interest in plot_y_of_interests_dict.keys():
    x_axis_order = ["Cont", "Indo", "IndoAP5"]
    combined_LTP_point_df["condition"] = pd.Categorical(combined_LTP_point_df["condition"], categories=x_axis_order, ordered=True)
    combined_LTP_point_df = combined_LTP_point_df.sort_values(by="condition")

    plt.figure(figsize=(5, 3))
    data_df = combined_LTP_point_df[(combined_LTP_point_df[each_y_of_interest] > each_y_of_interest_reject_range_dict[each_y_of_interest][0])
                                    & (combined_LTP_point_df[each_y_of_interest] < each_y_of_interest_reject_range_dict[each_y_of_interest][1])]
    p = sns.swarmplot(x="condition", y=each_y_of_interest, data=data_df, order=x_axis_order)
    plt.ylabel(plot_y_of_interests_dict[each_y_of_interest])
    sns.boxplot(showmeans=True,
                meanline=True,
                meanprops={'color': 'r', 'ls': '-', 'lw': 1},
                medianprops={'visible': False},
                whiskerprops={'visible': False},
                zorder=10,
                x="condition",
                y=each_y_of_interest,
                data=data_df.sort_values(by="condition"),
                showfliers=False,
                showbox=False,
                showcaps=False,
                ax=p)
    
    for x_pos, condition in zip(p.get_xticks(), data_df.sort_values(by="condition")["condition"].unique()):
        print(x_pos, condition)
        each_condition_df = data_df.sort_values(by="condition")[data_df.sort_values(by="condition")["condition"] == condition]
        print(each_condition_df[each_y_of_interest].mean())
        p.text(x_pos-0.3, float(each_condition_df[each_y_of_interest].mean()), f"{each_condition_df[each_y_of_interest].mean():.2f}", ha='center', va='bottom', fontsize=8, color="black")
        

    savepath = os.path.join(os.path.dirname(combined_LTP_point_df_csv_path),
                    f"{each_y_of_interest}_vs_x_condition.png")
    plt.savefig(savepath,
                dpi=150,bbox_inches="tight", transparent=True)
    plt.show()
    print(f"saved plot to \n{savepath}")


# %%
# %% plot for each date

# plot swamplot for each y_of_interest in plot_y_of_interests_dict for each date

for each_date in combined_LTP_point_df["date"].unique():
    for each_y_of_interest in plot_y_of_interests_dict.keys():
        plt.figure(figsize=(5, 3))
        data_df = combined_LTP_point_df[(combined_LTP_point_df["date"] == each_date)
                                        & (combined_LTP_point_df[each_y_of_interest] > each_y_of_interest_reject_range_dict[each_y_of_interest][0])
                                        & (combined_LTP_point_df[each_y_of_interest] < each_y_of_interest_reject_range_dict[each_y_of_interest][1])]

        #x axis order should be Cont, Indo, IndoAP5 (ABC order)
        # but on some date, not all of the conditions are present, so we need to check
        if "Cont" in data_df["condition"].unique():
            x_axis_order = ["Cont", "Indo", "IndoAP5"]
        else:
            x_axis_order = ["Indo", "IndoAP5"]

        p = sns.swarmplot(x="condition", y=each_y_of_interest, data=data_df, order=x_axis_order)
        plt.title(each_date)
        plt.ylabel(plot_y_of_interests_dict[each_y_of_interest])
        sns.boxplot(showmeans=True,
                    meanline=True,
                    meanprops={'color': 'r', 'ls': '-', 'lw': 1},
                    medianprops={'visible': False},
                    whiskerprops={'visible': False},
                    zorder=10,
                    x="condition",
                    y=each_y_of_interest,
                    data=data_df.sort_values(by="condition"),
                    showfliers=False,
                    showbox=False,
                    showcaps=False,
                    ax=p)
        
        for x_pos, condition in zip(p.get_xticks(), data_df.sort_values(by="condition")["condition"].unique()):
            print(x_pos, condition)
            each_condition_df = data_df.sort_values(by="condition")[data_df.sort_values(by="condition")["condition"] == condition]
            print(each_condition_df[each_y_of_interest].mean())
            p.text(x_pos-0.3, float(each_condition_df[each_y_of_interest].mean()), f"{each_condition_df[each_y_of_interest].mean():.2f}", ha='center', va='bottom', fontsize=8, color="black")
            

        savepath = os.path.join(os.path.dirname(combined_LTP_point_df_csv_path),
                        f"{each_date}_{each_y_of_interest}_vs_x_condition.png")
        plt.savefig(savepath,
                    dpi=150,bbox_inches="tight", transparent=True)
        plt.show()
        print(f"saved plot to \n{savepath}")

# %%
