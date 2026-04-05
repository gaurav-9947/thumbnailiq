import sys
sys.path.append(r'D:\thumbnailiq')

import pandas as pd

print("Loading all feature files...")
base_df = pd.read_csv(r'D:\thumbnailiq\data\videos.csv')
color_df = pd.read_csv(r'D:\thumbnailiq\data\color_features.csv')
face_df = pd.read_csv(r'D:\thumbnailiq\data\face_features.csv')
text_df = pd.read_csv(r'D:\thumbnailiq\data\text_features.csv')
clip_df = pd.read_csv(r'D:\thumbnailiq\data\clip_features.csv')

print(f"Base: {len(base_df)} rows")
print(f"Color: {len(color_df)} rows, {len(color_df.columns)} cols")
print(f"Face: {len(face_df)} rows, {len(face_df.columns)} cols")
print(f"Text: {len(text_df)} rows, {len(text_df.columns)} cols")
print(f"CLIP: {len(clip_df)} rows, {len(clip_df.columns)} cols")

# Drop non-numeric face column before merging
face_df = face_df.drop(columns=["dominant_emotion"])

# Merge everything on video_id
df = base_df.merge(color_df, on="video_id", how="inner")
df = df.merge(face_df, on="video_id", how="inner")
df = df.merge(text_df, on="video_id", how="inner")
df = df.merge(clip_df, on="video_id", how="inner")

# Keep only what we need
df = df.dropna()

print(f"\nFinal merged shape: {df.shape}")
print(f"Total features: {df.shape[1]} columns")

df.to_csv(r'D:\thumbnailiq\data\features.csv', index=False)
print("Saved to data/features.csv")