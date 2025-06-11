# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt




# %%
# load csv
csv_path = r"\\RY-LAB-WS04\ImagingData\Tetsuya\20250606\plot\titration_result - Copy.csv"

df = pd.read_csv(csv_path)

# %%
#Try to find integer and um in um, and extract integer and use it as um
df["um"] = df["group"].str.extract(r"_(\d+)um").astype(int)


df["two_groups"] = ""

two_group_dict = {"Cont":"Cont",

                "Grin1":"Grin1",
                "cont":"Cont"
                }

for ind, each_row in df.iterrows():
    group = each_row["group"]
    for each_key in two_group_dict:
        # print(each_key)
        if each_key in group:
            df.loc[ind, "two_groups"] = two_group_dict[each_key]
            break


# %%

#only 2.8 mW
for each_pow in df["pow_mw_round"].unique():
    df_each_pow = df[df["pow_mw_round"] == each_pow]

    #welch"

    plt.figure(figsize=(3,3))
    plt.title(f"Power: {each_pow} mW")
    sns.swarmplot(x="two_groups",y="spine_F_F0",data=df_each_pow)
    #delete right and upper axis
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.xlabel("")
    plt.ylabel("Spine F/F0")
    plt.tight_layout()
    plt.show()


# %%
