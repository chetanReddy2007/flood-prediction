# Steps 3, 4 and 5 - Load, Preprocess and Train Models

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import joblib
import time

# ----------------------------------------
# STEP 3 - Load the Dataset
# ----------------------------------------
print("Loading dataset...")
df = pd.read_csv('dataset/train.csv')
print(f"Full Dataset Shape: {df.shape}")

# ----------------------------------------
# USE ONLY 50,000 ROWS FOR FASTER TRAINING
# ----------------------------------------
df = df.sample(n=50000, random_state=42)
print(f"Using Sample of: {df.shape[0]} rows")

# ----------------------------------------
# STEP 4 - Preprocess the Dataset
# ----------------------------------------
print("\nPreprocessing data...")

df = df.drop('id', axis=1)
df['Flood'] = (df['FloodProbability'] >= 0.5).astype(int)
df = df.drop('FloodProbability', axis=1)

X = df.drop('Flood', axis=1)
y = df['Flood']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set: {X_train.shape[0]} rows")
print(f"Testing set: {X_test.shape[0]} rows")

# ----------------------------------------
# STEP 5 - Train All 4 Models
# ----------------------------------------
print("\n--- Training Models ---\n")

results = {}

# 1. Decision Tree
print("Training Decision Tree...")
start = time.time()
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
dt_acc = accuracy_score(y_test, dt_pred)
results['Decision Tree'] = dt_acc
print(f"Decision Tree Accuracy: {dt_acc * 100:.2f}%")
print(f"Time: {time.time() - start:.2f} seconds")

# 2. Random Forest
print("\nTraining Random Forest...")
start = time.time()
rf = RandomForestClassifier(n_estimators=50, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
results['Random Forest'] = rf_acc
print(f"Random Forest Accuracy: {rf_acc * 100:.2f}%")
print(f"Time: {time.time() - start:.2f} seconds")

# 3. KNN
print("\nTraining KNN...")
start = time.time()
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)
knn_acc = accuracy_score(y_test, knn_pred)
results['KNN'] = knn_acc
print(f"KNN Accuracy: {knn_acc * 100:.2f}%")
print(f"Time: {time.time() - start:.2f} seconds")

# 4. XGBoost
print("\nTraining XGBoost...")
start = time.time()
xgb = XGBClassifier(random_state=42, eval_metric='logloss')
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)
xgb_acc = accuracy_score(y_test, xgb_pred)
results['XGBoost'] = xgb_acc
print(f"XGBoost Accuracy: {xgb_acc * 100:.2f}%")
print(f"Time: {time.time() - start:.2f} seconds")

# ----------------------------------------
# STEP 6 - Compare and Save Best Model
# ----------------------------------------
print("\n--- Model Comparison ---")
for model, acc in results.items():
    print(f"{model}: {acc * 100:.2f}%")

best_model_name = max(results, key=results.get)
print(f"\nBest Model: {best_model_name}")

if best_model_name == 'Decision Tree':
    best_model = dt
elif best_model_name == 'Random Forest':
    best_model = rf
elif best_model_name == 'KNN':
    best_model = knn
else:
    best_model = xgb

joblib.dump(best_model, 'model/flood_model.pkl')
print(f"Best model saved to model/flood_model.pkl")
print("\nAll done!")