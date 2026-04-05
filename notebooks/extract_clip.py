import sys
sys.path.append(r'D:\thumbnailiq')

import pandas as pd
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
from tqdm import tqdm

df = pd.read_csv(r'D:\thumbnailiq\data\videos.csv')
print(f"Processing {len(df)} thumbnails for CLIP embeddings...")
print("Loading CLIP model (first time downloads ~600MB)...")

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()
print("CLIP model loaded!")

def extract_clip_features(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model.vision_model(**inputs)
            features = outputs.pooler_output
        features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze().numpy()
    except Exception as e:
        print(f"Error {image_path}: {e}")
        return np.zeros(512)

results = []
for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting CLIP embeddings"):
    embedding = extract_clip_features(row["thumbnail_path"])
    row_dict = {"video_id": row["video_id"]}
    for i, val in enumerate(embedding):
        row_dict[f"clip_{i}"] = val
    results.append(row_dict)

clip_df = pd.DataFrame(results)
clip_df.to_csv(r'D:\thumbnailiq\data\clip_features.csv', index=False)
print(f"\nSaved {len(clip_df)} rows with {clip_df.shape[1]-1} CLIP features")
print(f"Sample columns: {list(clip_df.columns[:5])}")