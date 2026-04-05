import sys
sys.path.append(r'D:\thumbnailiq')

import pandas as pd
from deepface import DeepFace
from tqdm import tqdm

df = pd.read_csv(r'D:\thumbnailiq\data\videos.csv')
print(f"Processing {len(df)} thumbnails for face features...")

def extract_face_features(image_path):
    try:
        results = DeepFace.analyze(
            img_path=image_path,
            actions=["emotion", "gender"],
            enforce_detection=False,
            silent=True
        )

        if isinstance(results, list):
            result = results[0]
        else:
            result = results

        face_count = len(results) if isinstance(results, list) else 1
        dominant_emotion = result.get("dominant_emotion", "unknown")
        emotions = result.get("emotion", {})

        return {
            "face_count": face_count,
            "dominant_emotion": dominant_emotion,
            "emotion_happy": emotions.get("happy", 0),
            "emotion_surprise": emotions.get("surprise", 0),
            "emotion_neutral": emotions.get("neutral", 0),
            "emotion_angry": emotions.get("angry", 0),
            "has_face": 1
        }

    except Exception as e:
        return {
            "face_count": 0,
            "dominant_emotion": "none",
            "emotion_happy": 0,
            "emotion_surprise": 0,
            "emotion_neutral": 0,
            "emotion_angry": 0,
            "has_face": 0
        }

results = []
for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting face features"):
    features = extract_face_features(row["thumbnail_path"])
    features["video_id"] = row["video_id"]
    results.append(features)

face_df = pd.DataFrame(results)
face_df.to_csv(r'D:\thumbnailiq\data\face_features.csv', index=False)
print(f"\nSaved {len(face_df)} rows to data/face_features.csv")
print(face_df.head(3).to_string())