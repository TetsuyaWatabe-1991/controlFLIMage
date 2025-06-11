import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt


file_list = [
r"G:\ImagingData\Tetsuya\20250227\plot\result - Copy202505152110.csv",
r"G:\ImagingData\Tetsuya\20250227\automation\highmag_highmag_list\tpem\plot\result - Copy202505152047.csv",
r"G:\ImagingData\Tetsuya\20250325\B6wt_GC6stdTom_tony_cut0303_2ndslice\tpem\plot\result - Copy.csv",
r"G:\ImagingData\Tetsuya\20250331\B6_cut0319_FlxGC6sTom_0322\highmag_RFP50ms100p\tpem2\plot\result - Copy.csv",
r"G:\ImagingData\Tetsuya\20250408\tpem2\plot\result - Copy.csv",
r"G:\ImagingData\Tetsuya\20250416\B6_cut0326_FlxGC6s_tdTomato0330\highmag_Trans5ms\tpem\plot\result - Copy.csv"
]

combined_df = pd.DataFrame()
for file in file_list:
    date = file.split("\\")[3]
    df = pd.read_csv(file)
    if ' dend_F_F0' in df.columns:
        print(df[' dend_F_F0'].mean())
        print("dend")
        df['shaft_f_f0'] = df[' dend_F_F0']
    elif 'shaft_f_f0' in df.columns:
        print(df['shaft_f_f0'].mean())
        print("shaft")
    else:
        print(file)
    df['date'] = date
    combined_df = pd.concat([combined_df,df])

print("--------------------------------")
for date in combined_df['date'].unique():
    each_df = combined_df[combined_df['date'] == date]
    print(date)
    print(len(each_df[each_df['shaft_f_f0'] > 10])/len(each_df))

combined_df.to_csv(r"G:\ImagingData\Tetsuya\20250415\combined_df.csv", index=False)

# sns.violinplot(x='date', y='shaft_f_f0', data=combined_df)
# sns.swarmplot(x='date', y='shaft_f_f0', data=combined_df)
# plt.show()