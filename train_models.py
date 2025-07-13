import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import joblib

# Load data
df = pd.read_csv("reg.csv")
df.dropna(inplace=True)

# Add new numeric features: average counts by city and by cause subcategory
df['City Avg Count'] = df.groupby('Million Plus Cities')['Count'].transform('mean')
df['Cause Avg Count'] = df.groupby('Cause Subcategory')['Count'].transform('mean')

# Features and target
categorical_features = ['Million Plus Cities', 'Cause category', 'Cause Subcategory', 'Outcome of Incident']
numeric_features = ['City Avg Count', 'Cause Avg Count']
target = 'Count'

X = df[categorical_features + numeric_features]
y = np.log1p(df[target])  # log(1 + y) transform

# Categorical encoder + numeric scaler
ct = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
    ('num', StandardScaler(), numeric_features)
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit transformer and encode training data
X_train_enc = ct.fit_transform(X_train)

# Save the transformer
joblib.dump(ct, "column_transformer.joblib")

# Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train_enc, y_train)
joblib.dump(rf, "random_forest_model.joblib")

# XGBoost
xg = xgb.XGBRegressor(n_estimators=100, random_state=42)
xg.fit(X_train_enc, y_train)
joblib.dump(xg, "xgboost_model.joblib")

print("✅ All models and transformer saved successfully!")
