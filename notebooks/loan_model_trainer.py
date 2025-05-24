import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 📥 Load dataset
print("📥 Loading dataset...")
df = pd.read_csv("../data/lending_club_full.csv", low_memory=False)
print("✅ Dataset loaded!")

# 🎯 Filter target classes
df = df[df['loan_status'].isin(['Fully Paid', 'Charged Off'])]
df = df.sample(5000, random_state=42)
df['loan_status'] = df['loan_status'].map({'Fully Paid': 0, 'Charged Off': 1})

# 🧹 Select only used features
selected_features = [
    'loan_amnt', 'int_rate', 'emp_length',
    'term', 'grade', 'home_ownership',
    'verification_status', 'purpose'
]
df = df[selected_features + ['loan_status']]

# 🧼 Clean employment length
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

# 🔠 Label encoding
for col in ['term', 'grade', 'home_ownership', 'verification_status', 'purpose']:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# 🩹 Fill missing values
df.fillna(0, inplace=True)

# 🔀 Train-test split
X = df.drop('loan_status', axis=1)
y = df['loan_status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 📦 Algorithms to train
models = {
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "logistic_regression": LogisticRegression(max_iter=1000),
    "decision_tree": DecisionTreeClassifier(),
    "gradient_boosting": GradientBoostingClassifier()
}

# 📁 Model save directory
os.makedirs("../models", exist_ok=True)

# 🚀 Train and save
for name, model in models.items():
    print(f"\n🚀 Training: {name.replace('_', ' ').title()}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("🧪 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("📊 Classification Report:")
    print(classification_report(y_test, y_pred))

    model_path = f"../models/model_{name}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"✅ Saved model to {model_path}")
