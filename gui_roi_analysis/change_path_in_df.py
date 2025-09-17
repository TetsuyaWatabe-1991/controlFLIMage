import pandas as pd
pkl_path = r"\\RY-LAB-WS04\ImagingData\Tetsuya\20250904\auto1\combined_df1.pkl"
df = pd.read_pickle(pkl_path)

query_str = r"G:\ImagingData\Tetsuya"
replace_to = r"\\RY-LAB-WS04\ImagingData\Tetsuya"

# in any cells, if cells contain query_str, replace it with replace_to
df = df.applymap(lambda x: x.replace(query_str, replace_to) if isinstance(x, str) else x)

save_pkl_path = r"\\RY-LAB-WS04\ImagingData\Tetsuya\20250904\auto1\combined_df1_changed.pkl"
df.to_pickle(save_pkl_path)