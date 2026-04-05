import sys
sys.path.append(r'D:\thumbnailiq')

import cv2
import numpy as np
import pandas as pd
import torch
import joblib
import shap
import pytesseract
import gradio as gr
from PIL import Image
from deepface import DeepFace
from transformers import CLIPProcessor, CLIPModel

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

print("Loading models...")
model = joblib.load(r'D:\thumbnailiq\models\xgboost_model.pkl')
feature_names = joblib.load(r'D:\thumbnailiq\models\feature_names.pkl')
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()
print("All models loaded!")

def extract_all_features(image_path):
    features = {}

    # Color features
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features['brightness'] = float(np.mean(gray))
    features['contrast'] = float(np.std(gray))
    features['saturation_mean'] = float(np.mean(hsv[:,:,1]))
    features['saturation_std'] = float(np.std(hsv[:,:,1]))
    features['hue_mean'] = float(np.mean(hsv[:,:,0]))
    r, g, b = img[:,:,2], img[:,:,1], img[:,:,0]
    rg = np.abs(r.astype(int) - g.astype(int))
    yb = np.abs(0.5*(r.astype(int)+g.astype(int)) - b.astype(int))
    features['colorfulness'] = float(np.sqrt(np.mean(rg**2) + np.mean(yb**2)))
    pixels = img.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, _, centers = cv2.kmeans(pixels, 3, None, criteria, 3, cv2.KMEANS_RANDOM_CENTERS)
    centers = centers.astype(int)
    for i, (b, g, r) in enumerate(centers):
        features[f'dom{i+1}_r'] = r
        features[f'dom{i+1}_g'] = g
        features[f'dom{i+1}_b'] = b

    # Face features
    try:
        results = DeepFace.analyze(image_path, actions=['emotion','gender'],
                                   enforce_detection=False, silent=True)
        result = results[0] if isinstance(results, list) else results
        features['face_count'] = len(results) if isinstance(results, list) else 1
        emotions = result.get('emotion', {})
        features['emotion_happy'] = emotions.get('happy', 0)
        features['emotion_surprise'] = emotions.get('surprise', 0)
        features['emotion_neutral'] = emotions.get('neutral', 0)
        features['emotion_angry'] = emotions.get('angry', 0)
        features['has_face'] = 1
    except:
        features['face_count'] = 0
        features['emotion_happy'] = 0
        features['emotion_surprise'] = 0
        features['emotion_neutral'] = 0
        features['emotion_angry'] = 0
        features['has_face'] = 0

    # Text features
    try:
        pil_img = Image.open(image_path).convert('RGB')
        img_array = np.array(pil_img)
        gray2 = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        total_pixels = gray2.shape[0] * gray2.shape[1]
        data = pytesseract.image_to_data(gray2,
                output_type=pytesseract.Output.DICT, config='--psm 11')
        words = [data['text'][i] for i in range(len(data['text']))
                 if int(data['conf'][i]) > 40 and data['text'][i].strip() != '']
        text_area = sum(data['width'][i] * data['height'][i]
                       for i in range(len(data['text']))
                       if int(data['conf'][i]) > 40 and data['text'][i].strip() != '')
        text_joined = ' '.join(words)
        features['word_count'] = len(words)
        features['text_area_pct'] = (text_area / total_pixels) * 100
        features['has_numbers'] = int(any(c.isdigit() for c in text_joined))
        features['has_exclamation'] = int('!' in text_joined)
        features['has_question'] = int('?' in text_joined)
        features['has_text'] = int(len(words) > 0)
    except:
        features['word_count'] = 0
        features['text_area_pct'] = 0
        features['has_numbers'] = 0
        features['has_exclamation'] = 0
        features['has_question'] = 0
        features['has_text'] = 0

    # CLIP features
    try:
        pil_img = Image.open(image_path).convert('RGB')
        inputs = clip_processor(images=pil_img, return_tensors='pt')
        with torch.no_grad():
            outputs = clip_model.vision_model(**inputs)
            embedding = outputs.pooler_output
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        embedding = embedding.squeeze().numpy()
        for i, val in enumerate(embedding):
            features[f'clip_{i}'] = val
    except:
        for i in range(768):
            features[f'clip_{i}'] = 0.0

    return features

def get_suggestions(features, shap_vals, feature_names):
    suggestions = []

    if features.get('saturation_mean', 0) < 80:
        suggestions.append("🎨 Boost color saturation — dull thumbnails get fewer clicks. Use vibrant, punchy colors.")
    if features.get('has_face', 0) == 0:
        suggestions.append("👤 Add a human face — thumbnails with faces consistently outperform those without.")
    if features.get('has_text', 0) == 0:
        suggestions.append("✏️ Add bold text overlay — a short hook phrase increases curiosity and clicks.")
    if features.get('brightness', 0) < 80:
        suggestions.append("💡 Increase brightness — dark thumbnails get overlooked in crowded feeds.")
    if features.get('contrast', 0) < 40:
        suggestions.append("⚡ Increase contrast — high contrast thumbnails stand out more on screen.")
    if features.get('has_numbers', 0) == 0:
        suggestions.append("🔢 Consider adding a number — '5 Ways to...' or '3 Tips...' thumbnails perform well.")
    if features.get('colorfulness', 0) < 30:
        suggestions.append("🌈 Make it more colorful — low colorfulness reduces visual impact.")

    return suggestions[:3] if len(suggestions) >= 3 else suggestions

def score_thumbnail(image):
    if image is None:
        return "Please upload a thumbnail image.", "", ""

    # Save temp image
    temp_path = r'D:\thumbnailiq\data\temp_thumbnail.jpg'
    if isinstance(image, np.ndarray):
        cv2.imwrite(temp_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    else:
        image.save(temp_path)

    # Extract features
    features = extract_all_features(temp_path)

    # Build feature vector
    feature_vector = [features.get(name, 0.0) for name in feature_names]
    feature_df = pd.DataFrame([feature_vector], columns=feature_names)

    # Predict
    pred_log = model.predict(feature_df)[0]
    pred_ctr = np.expm1(pred_log)

    # Convert to 0-100 score
    min_log, max_log = 0, 15
    score = int(np.clip((pred_log - min_log) / (max_log - min_log) * 100, 0, 100))

    # SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(feature_df)

    # Score color
    if score >= 70:
        score_color = "🟢"
    elif score >= 40:
        score_color = "🟡"
    else:
        score_color = "🔴"

    # Suggestions
    suggestions = get_suggestions(features, shap_values, feature_names)

    # Format score output
    score_output = f"{score_color} CTR Score: {score}/100"

    # Format analysis
    analysis = f"""
📊 Thumbnail Analysis
━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎨 Color
   Brightness:  {features.get('brightness', 0):.1f}/255
   Saturation:  {features.get('saturation_mean', 0):.1f}/255
   Contrast:    {features.get('contrast', 0):.1f}
   Colorfulness:{features.get('colorfulness', 0):.1f}

👤 Face Detection
   Faces found: {int(features.get('face_count', 0))}
   Happy score: {features.get('emotion_happy', 0):.1f}%
   Surprise:    {features.get('emotion_surprise', 0):.1f}%

✏️ Text Detection
   Words found: {int(features.get('word_count', 0))}
   Has numbers: {'Yes' if features.get('has_numbers') else 'No'}
   Text area:   {features.get('text_area_pct', 0):.2f}%
"""

    # Format suggestions
    if suggestions:
        suggestion_text = "💡 Top Suggestions to Improve\n━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        for i, s in enumerate(suggestions, 1):
            suggestion_text += f"{i}. {s}\n\n"
    else:
        suggestion_text = "✅ Your thumbnail looks great! No major improvements needed."

    return score_output, analysis, suggestion_text

# Custom CSS
css = """
.gradio-container {
    font-family: 'Segoe UI', sans-serif;
    max-width: 1200px;
    margin: auto;
}
.score-box {
    font-size: 2em;
    font-weight: bold;
    text-align: center;
    padding: 20px;
    border-radius: 12px;
    background: #1a1a2e;
}
#title {
    text-align: center;
    font-size: 2.5em;
    font-weight: bold;
    margin-bottom: 5px;
}
#subtitle {
    text-align: center;
    color: #888;
    margin-bottom: 30px;
}
"""

with gr.Blocks(css=css, theme=gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate"
)) as demo:

    gr.HTML("<div id='title'>ThumbnailIQ</div>")
    gr.HTML("<div id='subtitle'>Upload a YouTube thumbnail and get an instant CTR score + improvement tips</div>")

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                label="Upload Thumbnail",
                type="numpy",
                height=300
            )
            analyze_btn = gr.Button(
                "Analyze Thumbnail",
                variant="primary",
                size="lg"
            )
            gr.Examples(
                examples=[],
                inputs=image_input
            )

        with gr.Column(scale=1):
            score_output = gr.Textbox(
                label="CTR Score",
                lines=1,
                elem_classes=["score-box"]
            )
            analysis_output = gr.Textbox(
                label="Thumbnail Analysis",
                lines=12
            )
            suggestions_output = gr.Textbox(
                label="Improvement Suggestions",
                lines=6
            )

    analyze_btn.click(
        fn=score_thumbnail,
        inputs=image_input,
        outputs=[score_output, analysis_output, suggestions_output]
    )

    gr.HTML("""
    <div style='text-align:center; margin-top:20px; color:#888; font-size:0.9em;'>
        Built with XGBoost + CLIP + DeepFace + Gradio &nbsp;|&nbsp; ThumbnailIQ
    </div>
    """)

if __name__ == "__main__":
    demo.launch(share=False)