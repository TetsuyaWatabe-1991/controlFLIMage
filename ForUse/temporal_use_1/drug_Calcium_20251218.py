import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


csv_path = r"\\RY-LAB-WS04\ImagingData\Tetsuya\20251217\auto1\GCaMP_F_F0_in_ROI_combined.csv"

df = pd.read_csv(csv_path)


plt.figure(figsize=(10, 4))
# beeswarm plot
plot_order = [ 'Water 0.2%', 'DMSO 0.1%', 'D-AP5','Gavestinel', 'MK801']
p = sns.swarmplot(x="condition", y="F_F0", data=df, order=plot_order, palette=["k", "k", "k", "k", "k"],size=5)

# Draw mean and SD bars for each group
plt.title("GCaMP F/F0")
grouped = df.groupby("condition")["F_F0"]
for i, label in enumerate(plot_order):
    y = grouped.get_group(label)
    mean = y.mean()
    sd = y.std()
    # Mean bar (longer)
    p.hlines(mean, i-0.25, i+0.25, color='red', linewidth=2, zorder=20)
    # SD bars (shorter)
    p.hlines([mean-sd, mean+sd], i-0.12, i+0.12, color='red', linewidth=2, alpha=0.7, zorder=20)

# Show p-value (4 decimal places)

plt.show()

