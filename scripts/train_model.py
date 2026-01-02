import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
from micromlgen import port

# --- PHASE 1: ENVIRONMENT SETUP & DATA UNDERSTANDING ---

# 1. Load Dataset
file_path = r"C:\Users\USER\Desktop\cooking_activity_detection\data\Dataset.csv"
df = pd.read_csv(file_path)

# 2. Statistical Representation
print("--- Dataset Statistics ---")
print(df.describe())
print("\n--- Missing Values Summary ---")
print(df.isnull().sum())

# 3. Handle Activity Column (NaN -> 0, "Cooking" -> 1)
print("\n--- Class Counts Before Preprocessing ---")
print(df['Activity'].value_counts(dropna=False))

# Visualization: Class Distribution Before
plt.figure(figsize=(8, 5))
sns.countplot(x=df['Activity'].fillna('NaN'))
plt.title('Class Distribution Before Preprocessing')
plt.show()

df['Activity'] = df['Activity'].fillna(0)
df['Activity'] = df['Activity'].replace('Cooking', 1).astype(int)

print("\n--- Class Counts After Preprocessing ---")
print(df['Activity'].value_counts())

# 4. Feature Selection
base_features = ['Temperature', 'Relative Humidity', 'TVOC', 'CO', 'CO2']
X = df[base_features]
y = df['Activity']

# Visualization: Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df[base_features + ['Activity']].corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()

# 5. Baseline Evaluation (Logistic Regression)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
baseline = LogisticRegression(max_iter=1000)
baseline.fit(X_train, y_train)
y_pred_base = baseline.predict(X_test)

print("\n--- Phase 1: Baseline Logistic Regression Report ---")
print(classification_report(y_test, y_pred_base))

# --- PHASE 2: MODEL DEVELOPMENT & IMBALANCE HANDLING ---

# 1. Feature Engineering
df['Temp_TVOC'] = df['Temperature'] * df['TVOC']
extended_features = base_features + ['Temp_TVOC']

X_ext = df[extended_features]
y_ext = df['Activity']

X_train_ext, X_test_ext, y_train_ext, y_test_ext = train_test_split(
    X_ext, y_ext, test_size=0.2, random_state=42, stratify=y_ext
)

# 2. Scaling & SMOTE
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_ext)
X_test_scaled = scaler.transform(X_test_ext)

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train_scaled, y_train_ext)

print("\n--- Class Distribution After SMOTE ---")
unique, counts = np.unique(y_resampled, return_counts=True)
print(dict(zip(unique, counts)))

# 3. Random Forest Training
param_grid = {'n_estimators': [50, 100], 'max_depth': [10, 20]}
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='f1', n_jobs=-1)
grid_search.fit(X_resampled, y_resampled)
best_rf = grid_search.best_estimator_

# 4. Final Evaluation & Visualizations
y_pred_rf = best_rf.predict(X_test_scaled)
print("\n--- Phase 2: Optimized Random Forest Report ---")
print(classification_report(y_test_ext, y_pred_rf))

# Visualization: Confusion Matrix
cm = confusion_matrix(y_test_ext, y_pred_rf)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Random Forest Confusion Matrix')
plt.show()

# Visualization: Feature Importance
importances = best_rf.feature_importances_
indices = np.argsort(importances)
plt.figure(figsize=(10, 6))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [extended_features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()

# 5. Export Artifacts
if not os.path.exists('models'):
    os.makedirs('models')

joblib.dump(best_rf, 'models/cooking_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

with open("models/model.h", "w") as f:
    f.write(port(best_rf))

print("\nArtifacts and visualizations generated successfully.")
