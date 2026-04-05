import sys
sys.path.append(r'D:\thumbnailiq')

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import spearmanr
import joblib

print("Loading features...")
df = pd.read_csv(r'D:\thumbnailiq\data\features.csv')

# Drop non-feature columns
drop_cols = ['video_id', 'title', 'published_at', 'thumbnail_url', 
             'thumbnail_path', 'ctr_proxy', 'category']
drop_cols = [c for c in drop_cols if c in df.columns]

X = df.drop(columns=drop_cols)
y = df['ctr_proxy_log']

print(f"Feature matrix: {X.shape}")
print(f"Target: {y.shape}")

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")

# Train XGBoost
print("\nTraining XGBoost...")
model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=0
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
spearman = spearmanr(y_test, y_pred).statistic

print(f"\nResults:")
print(f"MAE:           {mae:.4f}")
print(f"R² Score:      {r2:.4f}")
print(f"Spearman Rank: {spearman:.4f}")

# Save model and feature names
joblib.dump(model, r'D:\thumbnailiq\models\xgboost_model.pkl')
feature_names = list(X.columns)
joblib.dump(feature_names, r'D:\thumbnailiq\models\feature_names.pkl')

# Save predictions
results_df = pd.DataFrame({
    'actual_ctr_log': y_test.values,
    'predicted_ctr_log': y_pred,
    'actual_ctr': np.expm1(y_test.values),
    'predicted_ctr': np.expm1(y_pred)
})
results_df.to_csv(r'D:\thumbnailiq\data\predictions.csv', index=False)

print(f"\nModel saved to models/xgboost_model.pkl")
print(f"Predictions saved to data/predictions.csv")