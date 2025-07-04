import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import pickle
import os

# Step 1: Load dataset
print("📥 Loading dataset...")
df = pd.read_csv("../data/lending_club_full.csv", low_memory=False)
print("✅ Dataset loaded!")

# Step 2: Preprocess target
df = df[df['loan_status'].isin(['Fully Paid', 'Charged Off'])]
df = df.sample(5000, random_state=42)
df['loan_status'] = df['loan_status'].map({'Fully Paid': 0, 'Charged Off': 1})

# Step 3: Clean features
def clean_emp_length(val):
    if pd.isnull(val): return np.nan
    elif val == '< 1 year': return 0
    elif val == '10+ years': return 10
    else:
        try:
            return int(val.strip().split()[0])
        except:
            return np.nan

df['emp_length'] = df['emp_length'].apply(clean_emp_length)

# Select features used in Streamlit UI
features = ['loan_amnt', 'int_rate', 'emp_length', 'term', 'grade', 'home_ownership', 'verification_status', 'purpose']
df = df[features + ['loan_status']]

# Encode categoricals
for col in df.select_dtypes(include='object').columns:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

df.fillna(0, inplace=True)

# Step 4: SMOTE + split
X = df.drop('loan_status', axis=1)
y = df['loan_status']

print("Class distribution before SMOTE:")
print(y.value_counts())

smote = SMOTE(random_state=42, k_neighbors=1)
X_resampled, y_resampled = smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

# Step 5: Train and save models
models = {
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "logistic_regression": LogisticRegression(max_iter=1000),
    "decision_tree": DecisionTreeClassifier(),
    "gradient_boosting": GradientBoostingClassifier()
}

model_dir = "../models"
os.makedirs(model_dir, exist_ok=True)

for name, clf in models.items():
    print(f"\n🚀 Training {name}...")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    model_path = os.path.join(model_dir, f"model_{name}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)
    print(f"✅ Saved: {model_path}")
