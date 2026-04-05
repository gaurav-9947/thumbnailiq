import sys
sys.path.append(r'D:\thumbnailiq')

import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt

print("Loading model and data...")
model = joblib.load(r'D:\thumbnailiq\models\xgboost_model.pkl')
feature_names = joblib.load(r'D:\thumbnailiq\models\feature_names.pkl')

df = pd.read_csv(r'D:\thumbnailiq\data\features.csv')

drop_cols = ['video_id', 'title', 'published_at', 'thumbnail_url',
             'thumbnail_path', 'ctr_proxy', 'category',
             'ctr_proxy_log', 'view_count', 'like_count', 'days_old']
drop_cols = [c for c in drop_cols if c in df.columns]

X = df.drop(columns=drop_cols)
y = df['ctr_proxy_log']

print("Calculating SHAP values...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Beeswarm plot — top 20 features
print("Generating beeswarm plot...")
plt.figure(figsize=(12, 8))
shap.summary_plot(
    shap_values, X,
    max_display=20,
    show=False
)
plt.title("Top 20 Features by SHAP Importance")
plt.tight_layout()
plt.savefig(r'D:\thumbnailiq\data\shap_beeswarm.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved shap_beeswarm.png")

# Bar plot — mean absolute SHAP values
print("Generating bar plot...")
plt.figure(figsize=(12, 8))
shap.summary_plot(
    shap_values, X,
    plot_type="bar",
    max_display=20,
    show=False
)
plt.title("Top 20 Features — Mean SHAP Value")
plt.tight_layout()
plt.savefig(r'D:\thumbnailiq\data\shap_bar.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved shap_bar.png")

# Print top 10 most important features
print("\nTop 10 most important features:")
shap_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': np.abs(shap_values).mean(axis=0)
})
shap_importance = shap_importance.sort_values('importance', ascending=False)
print(shap_importance.head(10).to_string(index=False))