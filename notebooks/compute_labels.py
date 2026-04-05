import sys
sys.path.append(r'D:\thumbnailiq')

import pandas as pd
import numpy as np

df = pd.read_csv(r'D:\thumbnailiq\data\videos.csv')
print(f"Videos loaded: {len(df)}")

# Compute CTR proxy
df["ctr_proxy"] = df["view_count"] / df["days_old"]

# Remove outliers — videos with 0 views or less than 1 day old
df = df[df["view_count"] > 0]
df = df[df["days_old"] >= 1]

# Log transform to handle skew
df["ctr_proxy_log"] = np.log1p(df["ctr_proxy"])

# Drop rows with missing thumbnails
df = df[df["thumbnail_path"].notna()]

df.to_csv(r'D:\thumbnailiq\data\videos.csv', index=False)

print(f"Clean videos remaining: {len(df)}")
print(f"\nCTR Proxy stats:")
print(df["ctr_proxy"].describe().round(2))
print(f"\nSample:")
print(df[["title", "view_count", "days_old", "ctr_proxy", "ctr_proxy_log"]].head(5).to_string())