import sys
sys.path.append(r'D:\thumbnailiq')

import pandas as pd
import requests
import os
from tqdm import tqdm

df = pd.read_csv(r'D:\thumbnailiq\data\videos.csv')
print(f"Total videos to download: {len(df)}")

THUMBNAIL_FOLDER = r'D:\thumbnailiq\thumbnails'

def download_thumbnail(video_id, url):
    save_path = os.path.join(THUMBNAIL_FOLDER, f"{video_id}.jpg")
    
    if os.path.exists(save_path):
        return save_path
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            return save_path
        else:
            print(f"  Failed {video_id}: status {response.status_code}")
            return None
    except Exception as e:
        print(f"  Error {video_id}: {e}")
        return None

paths = []
for _, row in tqdm(df.iterrows(), total=len(df), desc="Downloading thumbnails"):
    path = download_thumbnail(row["video_id"], row["thumbnail_url"])
    paths.append(path)

df["thumbnail_path"] = paths
df.to_csv(r'D:\thumbnailiq\data\videos.csv', index=False)

success = sum(1 for p in paths if p is not None)
print(f"\nDownloaded {success}/{len(df)} thumbnails successfully")
print("CSV updated with thumbnail paths")