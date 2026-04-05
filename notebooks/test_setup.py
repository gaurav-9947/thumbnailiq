import sys
sys.path.append(r'D:\thumbnailiq')

import cv2
import pytesseract
import pandas as pd
import xgboost
import shap
import torch
import gradio
from googleapiclient.discovery import build
from config import YOUTUBE_API_KEY

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

print("OpenCV:", cv2.__version__)
print("Pandas:", pd.__version__)
print("XGBoost:", xgboost.__version__)
print("Torch:", torch.__version__)
print("Gradio:", gradio.__version__)

youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
print("YouTube API: connected")

print("\nAll good! Ready for Step 2.")