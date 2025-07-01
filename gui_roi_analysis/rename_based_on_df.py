#rename based on the rename_combined_df.csv

import os
import glob
import pandas as pd

csv_path = r"C:\Users\WatabeT\Desktop\20250626_Copy\rename_combined_df.csv"

combined_df = pd.read_csv(csv_path)

#rename the files based on the combined_df
while True:
    yes_or_no = input("Press y to continue...")
    if yes_or_no == "y":
        break
    else:
        continue

for ind, rows in combined_df.iterrows():
    before_rename_path = combined_df.at[ind, 'file_path']
    after_rename_path = os.path.dirname(before_rename_path) + "\\" + combined_df.at[ind, 'stem_name_renamed']

    if before_rename_path == after_rename_path:
        continue

    os.rename(before_rename_path, after_rename_path)
    print(f"Renamed {before_rename_path} to {after_rename_path}")

    
