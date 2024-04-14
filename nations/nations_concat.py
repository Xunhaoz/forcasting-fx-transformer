from pathlib import Path
import pandas as pd

csvs = Path(".").glob("*.csv")

dfs = []
for csv in csvs:
    df = pd.read_csv(csv, index_col='Date')
    dfs.append(df)

total_df = pd.concat(dfs, axis=1)


# 2004-07-26
# 2024-04-05
total_df = total_df['2004-07-26': '2024-04-05']
print(total_df)
print(total_df.isna().sum().sum())
print(total_df.shape)
