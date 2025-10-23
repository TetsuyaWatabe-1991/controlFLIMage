import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


csv_path = r"\\RY-LAB-WS04\ImagingData\Tetsuya\20250718\auto1\summary\LTP_point_df_after_analysis.csv"

df = pd.read_csv(csv_path)

df["drug"] = " "
df.loc[df["label"].str.contains("Cont"), "drug"] = "Control"
df.loc[df["label"].str.contains("AP5"), "drug"] = "AP5"

print("Control: ",len(df[df["drug"]=="Control"]))
print("AP5: ",len(df[df["drug"]=="AP5"]))

#print mean and std of norm_intensity, 0.xx
print("Control: ",round(df[df["drug"]=="Control"]["norm_intensity"].mean(), 2), round(df[df["drug"]=="Control"]["norm_intensity"].std(), 2))
print("AP5: ",round(df[df["drug"]=="AP5"]["norm_intensity"].mean(), 2), round(df[df["drug"]=="AP5"]["norm_intensity"].std(), 2))

#SNR in control
df[df["drug"]=="Control"]["norm_intensity"].mean()/df[df["drug"]=="Control"]["norm_intensity"].std()

#static
from scipy import stats
stats.ttest_ind(df[df["drug"]=="Control"]["norm_intensity"], df[df["drug"]=="AP5"]["norm_intensity"])

df2_early = df[df["time_after_started_experiment_hour"]<10]
stats.ttest_ind(df2_early[df2_early["drug"]=="Control"]["norm_intensity"], df2_early[df2_early["drug"]=="AP5"]["norm_intensity"])

df2_late = df[df["time_after_started_experiment_hour"]>10]
stats.ttest_ind(df2_late[df2_late["drug"]=="Control"]["norm_intensity"], df2_late[df2_late["drug"]=="AP5"]["norm_intensity"])



#%%

plt.figure(figsize=(3, 4))
# beeswarm plot
p = sns.swarmplot(x="drug", y="norm_intensity", data=df)

# Draw mean and SD bars for each group
plt.title("Time 0-20h")
grouped = df.groupby("drug")["norm_intensity"]
x_labels = list(df["drug"].unique())  # Ensure x_labels is a list
for i, label in enumerate(x_labels):
    y = grouped.get_group(label)
    mean = y.mean()
    sd = y.std()
    # Mean bar (longer)
    p.hlines(mean, i-0.25, i+0.25, color='red', linewidth=2, zorder=20)
    # SD bars (shorter)
    p.hlines([mean-sd, mean+sd], i-0.12, i+0.12, color='red', linewidth=2, alpha=0.7, zorder=20)

# Show p-value (4 decimal places)
plt.text(0.5, 0.95, "p = " + str(round(stats.ttest_ind(df[df["drug"]=="Control"]["norm_intensity"], df[df["drug"]=="AP5"]["norm_intensity"])[1], 4)), ha="center", va="bottom", transform=plt.gca().transAxes)
plt.ylabel("Delta volume")
plt.show()





plt.figure(figsize=(3, 4))
# beeswarm plot
p = sns.swarmplot(x="drug", y="norm_intensity", data=df2_early)
plt.title("Time 0-10h")
# Draw mean and SD bars for each group
grouped = df2_early.groupby("drug")["norm_intensity"]
x_labels = list(df2_early["drug"].unique())  # Ensure x_labels is a list
for i, label in enumerate(x_labels):
    y = grouped.get_group(label)
    mean = y.mean()
    sd = y.std()
    # Mean bar (longer)
    p.hlines(mean, i-0.25, i+0.25, color='red', linewidth=2, zorder=20)
    # SD bars (shorter)
    p.hlines([mean-sd, mean+sd], i-0.12, i+0.12, color='red', linewidth=2, alpha=0.7, zorder=20)

# Show p-value (4 decimal places)
plt.text(0.5, 0.95, "p = " + str(round(stats.ttest_ind(df2_early[df2_early["drug"]=="Control"]["norm_intensity"], df2_early[df2_early["drug"]=="AP5"]["norm_intensity"])[1], 4)), ha="center", va="bottom", transform=plt.gca().transAxes)
plt.ylabel("Delta volume")
plt.show()


print("Number of data \nControl: ",len(df2_early[df2_early["drug"]=="Control"]), "AP5: ",len(df2_early[df2_early["drug"]=="AP5"]))

print("Mean value \nControl: ",df2_early[df2_early["drug"]=="Control"]["norm_intensity"].mean(), "AP5: ",df2_early[df2_early["drug"]=="AP5"]["norm_intensity"].mean())

print("Std value \nControl: ",df2_early[df2_early["drug"]=="Control"]["norm_intensity"].std(), "AP5: ",df2_early[df2_early["drug"]=="AP5"]["norm_intensity"].std())




plt.figure(figsize=(3, 4))
# beeswarm plot
p = sns.swarmplot(x="drug", y="norm_intensity", data=df2_late)
plt.title("Time 10-20h")
# Draw mean and SD bars for each group
grouped = df2_late.groupby("drug")["norm_intensity"]
x_labels = list(df2_late["drug"].unique())  # Ensure x_labels is a list
for i, label in enumerate(x_labels):
    y = grouped.get_group(label)
    mean = y.mean()
    sd = y.std()
    # Mean bar (longer)
    p.hlines(mean, i-0.25, i+0.25, color='red', linewidth=2, zorder=20)
    # SD bars (shorter)
    p.hlines([mean-sd, mean+sd], i-0.12, i+0.12, color='red', linewidth=2, alpha=0.7, zorder=20)

# Show p-value (4 decimal places)
plt.text(0.5, 0.95, "p = " + str(round(stats.ttest_ind(df2_late[df2_late["drug"]=="Control"]["norm_intensity"], df2_late[df2_late["drug"]=="AP5"]["norm_intensity"])[1], 4)), ha="center", va="bottom", transform=plt.gca().transAxes)
plt.ylabel("Delta volume")
plt.show()

print("Number of data \nControl: ",len(df2_late[df2_late["drug"]=="Control"]), "AP5: ",len(df2_late[df2_late["drug"]=="AP5"]))

print("Mean value \nControl: ",df2_late[df2_late["drug"]=="Control"]["norm_intensity"].mean(), "AP5: ",df2_late[df2_late["drug"]=="AP5"]["norm_intensity"].mean())

print("Std value \nControl: ",df2_late[df2_late["drug"]=="Control"]["norm_intensity"].std(), "AP5: ",df2_late[df2_late["drug"]=="AP5"]["norm_intensity"].std())


#%%


plt.figure(figsize=(3, 4))
#beeswarm plot
p=sns.swarmplot(x="drug", y="GCaMP_DendriticShaft_F_F0", data=df)
#show mean and std
# plot the mean line
sns.boxplot(showmeans=True,
            meanline=True,
            meanprops={'color': 'r', 'ls': '-', 'lw': 1},
            medianprops={'visible': False},
            whiskerprops={'visible': False},
            zorder=10,
            x="drug",
            y="GCaMP_DendriticShaft_F_F0",
            data=df,
            showfliers=False,
            showbox=False,
            showcaps=False,
            ax=p)
plt.ylabel("GCaMP_DendriticShaft_F_F0")
plt.show()




AP5_df = df[df["drug"]=="AP5"]


Control_df = df[df["drug"]=="Control"]

#plot norm_intensity vs GCaMP_DendriticShaft_F_F0
plt.figure(figsize=(4, 4))
plt.title("Control")
plt.scatter(Control_df["GCaMP_DendriticShaft_F_F0"], Control_df["norm_intensity"])
plt.ylabel("Delta volume")
plt.xlabel("GCaMP_DendriticShaft_F_F0")
plt.xlim(0, 11)
plt.ylim(-0.5, 2)
plt.show()

#plot norm_intensity vs GCaMP_DendriticShaft_F_F0
plt.figure(figsize=(4, 4))
plt.scatter(AP5_df["GCaMP_DendriticShaft_F_F0"], AP5_df["norm_intensity"])
plt.ylabel("Delta volume")
plt.xlabel("GCaMP_DendriticShaft_F_F0")
plt.xlim(0, 11)
plt.ylim(-0.5, 2)
plt.show()




#plot norm_intensity vs GCaMP_DendriticShaft_F_F0
plt.figure(figsize=(4, 4))
plt.title("Control")
plt.scatter(Control_df["GCaMP_Spine_F_F0"], Control_df["norm_intensity"])
plt.ylabel("Delta volume")
plt.xlabel("GCaMP_Spine_F_F0")
plt.xlim(0, 11)
plt.ylim(-0.5, 2)
plt.show()



#plot norm_intensity vs GCaMP_DendriticShaft_F_F0
plt.figure(figsize=(4, 4))
plt.title("AP5")
plt.scatter(AP5_df["GCaMP_Spine_F_F0"], AP5_df["norm_intensity"])
plt.ylabel("Delta volume")
plt.xlabel("GCaMP_Spine_F_F0")
plt.xlim(0, 18)
plt.ylim(-0.5, 2)
plt.show()


plt.figure(figsize=(4, 4))
plt.title("Control")
plt.scatter(Control_df["time_after_started_experiment_hour"], Control_df["norm_intensity"])
plt.ylabel("Delta volume")
plt.xlabel("time_after_started_experiment_hour")
plt.xlim(0, 24)
plt.ylim(-0.5, 2)
plt.show()



plt.figure(figsize=(4, 4))
plt.title("AP5")
plt.scatter(AP5_df["time_after_started_experiment_hour"], AP5_df["norm_intensity"])
plt.ylabel("Delta volume")
plt.xlabel("time_after_started_experiment_hour")
plt.xlim(0, 24)
plt.ylim(-0.5, 2)
plt.show()


# moving average of norm_intensity

AP5_df.sort_values(by="time_after_started_experiment_hour", inplace=True)
Control_df.sort_values(by="time_after_started_experiment_hour", inplace=True)

AP5_df["norm_intensity_moving_average"] = AP5_df["norm_intensity"].rolling(window=6).mean()
# Control_df["norm_intensity_moving_average"] = Control_df["norm_intensity"].rolling(window=6).mean()

#plot norm_intensity vs time_after_started_experiment_hour
plt.figure(figsize=(4, 4))
plt.title("AP5")
plt.scatter(AP5_df["time_after_started_experiment_hour"], AP5_df["norm_intensity"], color="k")
plt.plot(AP5_df["time_after_started_experiment_hour"], AP5_df["norm_intensity_moving_average"],"r-")
plt.ylabel("Delta volume")
plt.xlabel("time_after_started_experiment_hour")
plt.xlim(0, 24)
plt.ylim(-0.5, 2)
plt.show()



Control_df["norm_intensity_moving_average"] = Control_df["norm_intensity"].rolling(window=6).mean()


#plot norm_intensity vs time_after_started_experiment_hour
plt.figure(figsize=(4, 4))
plt.title("Control")
plt.scatter(Control_df["time_after_started_experiment_hour"], Control_df["norm_intensity"], color="k")
plt.plot(Control_df["time_after_started_experiment_hour"], Control_df["norm_intensity_moving_average"],"r-")
plt.ylabel("Delta volume")
plt.xlabel("time_after_started_experiment_hour")
plt.xlim(0, 24)
plt.ylim(-0.5, 2)
plt.show()





#plot norm_intensity vs time_after_started_experiment_hour
plt.figure(figsize=(4, 4))
plt.title("Control")
plt.scatter(Control_df["time_after_started_experiment_hour"], Control_df["GCaMP_Spine_F_F0"], color="k")
plt.ylabel("GCaMP_Spine_F_F0")
plt.xlabel("time_after_started_experiment_hour")
plt.xlim(0, 24)
plt.ylim(0, 18)
plt.show()






#plot norm_intensity vs time_after_started_experiment_hour
plt.figure(figsize=(4, 4))
plt.title("AP5")
plt.scatter(AP5_df["time_after_started_experiment_hour"], AP5_df["GCaMP_Spine_F_F0"], color="k")
plt.ylabel("GCaMP_Spine_F_F0")
plt.xlabel("time_after_started_experiment_hour")
plt.xlim(0, 24)
plt.ylim(0, 18)
plt.show()





# %%
