import sys
sys.path.append(r'D:\thumbnailiq')

import cv2
import pytesseract
import pandas as pd
import numpy as np
from tqdm import tqdm

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

df = pd.read_csv(r'D:\thumbnailiq\data\videos.csv')
print(f"Processing {len(df)} thumbnails for text features...")

def extract_text_features(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        total_pixels = img.shape[0] * img.shape[1]

        # Get OCR data
        data = pytesseract.image_to_data(
            gray,
            output_type=pytesseract.Output.DICT,
            config='--psm 11'
        )

        # Extract words with confidence > 40
        words = [
            data['text'][i] for i in range(len(data['text']))
            if int(data['conf'][i]) > 40 and data['text'][i].strip() != ''
        ]

        word_count = len(words)
        text_joined = ' '.join(words)

        # Text area percentage
        text_area = 0
        for i in range(len(data['text'])):
            if int(data['conf'][i]) > 40 and data['text'][i].strip() != '':
                w = data['width'][i]
                h = data['height'][i]
                text_area += w * h

        text_area_pct = (text_area / total_pixels) * 100

        # Has numbers
        has_numbers = int(any(char.isdigit() for char in text_joined))

        # Has exclamation or question mark
        has_exclamation = int('!' in text_joined)
        has_question = int('?' in text_joined)

        return {
            "word_count": word_count,
            "text_area_pct": round(text_area_pct, 4),
            "has_numbers": has_numbers,
            "has_exclamation": has_exclamation,
            "has_question": has_question,
            "has_text": int(word_count > 0)
        }

    except Exception as e:
        print(f"Error {image_path}: {e}")
        return {
            "word_count": 0,
            "text_area_pct": 0,
            "has_numbers": 0,
            "has_exclamation": 0,
            "has_question": 0,
            "has_text": 0
        }

results = []
for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting text features"):
    features = extract_text_features(row["thumbnail_path"])
    features["video_id"] = row["video_id"]
    results.append(features)

text_df = pd.DataFrame(results)
text_df.to_csv(r'D:\thumbnailiq\data\text_features.csv', index=False)
print(f"\nSaved {len(text_df)} rows to data/text_features.csv")
print(text_df.head(3).to_string())