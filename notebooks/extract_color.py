import sys
sys.path.append(r'D:\thumbnailiq')

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

df = pd.read_csv(r'D:\thumbnailiq\data\videos.csv')
print(f"Processing {len(df)} thumbnails for color features...")

def extract_color_features(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None

        # Convert to different color spaces
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Brightness — mean of grayscale
        brightness = float(np.mean(gray))

        # Contrast — std deviation of grayscale
        contrast = float(np.std(gray))

        # Saturation — mean and std of S channel in HSV
        saturation_mean = float(np.mean(hsv[:, :, 1]))
        saturation_std = float(np.std(hsv[:, :, 1]))

        # Hue — mean of H channel
        hue_mean = float(np.mean(hsv[:, :, 0]))

        # Colorfulness score
        r, g, b = img[:,:,2], img[:,:,1], img[:,:,0]
        rg = np.abs(r.astype(int) - g.astype(int))
        yb = np.abs(0.5*(r.astype(int) + g.astype(int)) - b.astype(int))
        colorfulness = float(np.sqrt(np.mean(rg**2) + np.mean(yb**2)))

        # Dominant colors via k-means (k=3)
        pixels = img.reshape(-1, 3).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, _, centers = cv2.kmeans(pixels, 3, None, criteria, 3, cv2.KMEANS_RANDOM_CENTERS)
        centers = centers.astype(int)
        dom1_b, dom1_g, dom1_r = centers[0]
        dom2_b, dom2_g, dom2_r = centers[1]
        dom3_b, dom3_g, dom3_r = centers[2]

        return {
            "brightness": brightness,
            "contrast": contrast,
            "saturation_mean": saturation_mean,
            "saturation_std": saturation_std,
            "hue_mean": hue_mean,
            "colorfulness": colorfulness,
            "dom1_r": dom1_r, "dom1_g": dom1_g, "dom1_b": dom1_b,
            "dom2_r": dom2_r, "dom2_g": dom2_g, "dom2_b": dom2_b,
            "dom3_r": dom3_r, "dom3_g": dom3_g, "dom3_b": dom3_b,
        }

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

results = []
for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting color features"):
    features = extract_color_features(row["thumbnail_path"])
    if features:
        features["video_id"] = row["video_id"]
        results.append(features)

color_df = pd.DataFrame(results)
color_df.to_csv(r'D:\thumbnailiq\data\color_features.csv', index=False)
print(f"\nSaved {len(color_df)} rows to data/color_features.csv")
print(color_df.head(3).to_string())