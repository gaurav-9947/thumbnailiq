import sys
sys.path.append(r'D:\thumbnailiq')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
from scipy.stats import spearmanr
from xgboost import XGBRegressor

print("Loading data and model...")
model = joblib.load(r'D:\thumbnailiq\models\xgboost_model.pkl')

df = pd.read_csv(r'D:\thumbnailiq\data\features.csv')
drop_cols = ['video_id', 'title', 'published_at', 'thumbnail_url',
             'thumbnail_path', 'ctr_proxy', 'category',
             'ctr_proxy_log', 'view_count', 'like_count', 'days_old']
drop_cols = [c for c in drop_cols if c in df.columns]

X = df.drop(columns=drop_cols)
y = df['ctr_proxy_log']

# Cross validation
print("Running 5-fold cross validation...")
cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
print(f"CV R² scores: {cv_scores.round(3)}")
print(f"Mean CV R²:   {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Baseline comparison
baseline_pred = np.full(len(y), y.mean())
baseline_mae = mean_absolute_error(y, baseline_pred)
print(f"\nBaseline MAE (mean predictor): {baseline_mae:.4f}")
print(f"Model MAE:                     1.2681")
print(f"Improvement over baseline:     {((baseline_mae - 1.2681) / baseline_mae * 100):.1f}%")

# Actual vs predicted plot
predictions = pd.read_csv(r'D:\thumbnailiq\data\predictions.csv')
plt.figure(figsize=(8, 6))
plt.scatter(predictions['actual_ctr_log'], 
            predictions['predicted_ctr_log'], 
            alpha=0.7, color='steelblue', edgecolors='white', linewidth=0.5)
plt.plot([predictions['actual_ctr_log'].min(), predictions['actual_ctr_log'].max()],
         [predictions['actual_ctr_log'].min(), predictions['actual_ctr_log'].max()],
         'r--', linewidth=2, label='Perfect prediction')
plt.xlabel('Actual CTR (log)')
plt.ylabel('Predicted CTR (log)')
plt.title('Actual vs Predicted CTR')
plt.legend()
plt.tight_layout()
plt.savefig(r'D:\thumbnailiq\data\actual_vs_predicted.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved actual_vs_predicted.png")

print("\nEvaluation complete!")