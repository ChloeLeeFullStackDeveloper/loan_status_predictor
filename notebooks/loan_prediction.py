import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Step 1: Load dataset
print("📥 Loading dataset...")
df = pd.read_csv("../data/lending_club_full.csv", low_memory=False)
print("✅ Dataset loaded!")

# Step 2: Data overview
print("\n📊 Dataset Info:")
print(df.info())
print("\n🔎 Class Distribution (original):")
print(df['loan_status'].value_counts(dropna=False))

# Step 3: Filter and encode target variable
print("\n🧹 Cleaning and Encoding Data...")
df = df[df['loan_status'].isin(['Fully Paid', 'Charged Off'])]  
print("\nAfter filtering loan_status:")
print(df['loan_status'].value_counts())

df = df.sample(5000, random_state=42)

df['loan_status'] = df['loan_status'].map({'Fully Paid': 0, 'Charged Off': 1})

def clean_emp_length(val):
    if pd.isnull(val): return np.nan
    elif val == '< 1 year': return 0
    elif val == '10+ years': return 10
    else:
        try:
            return int(val.strip().split()[0])
        except:
            return np.nan

if 'emp_length' in df.columns:
    df['emp_length'] = df['emp_length'].apply(clean_emp_length)

# Drop high-cardinality or unnecessary columns
columns_to_drop = [
    'emp_title', 'title', 'zip_code', 'addr_state', 'url', 'desc', 'earliest_cr_line',
    'last_pymnt_d', 'next_pymnt_d', 'last_credit_pull_d', 'application_type',
    'initial_list_status', 'policy_code', 'pymnt_plan', 'settlement_status',
    'debt_settlement_flag_date', 'settlement_date', 'settlement_amount',
    'settlement_percentage', 'settlement_term', 'issued_d'
]
df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)

# Drop rows with essential features missing
df.dropna(subset=['loan_amnt', 'int_rate', 'term', 'loan_status'], inplace=True)

# One-hot or label encode categorical columns
categorical_cols = ['term', 'grade', 'home_ownership', 'verification_status', 'purpose']
for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].astype(str)
        df[col] = LabelEncoder().fit_transform(df[col])

df_encoded = df.copy()

# Drop remaining object columns
remaining_obj_cols = df_encoded.select_dtypes(include=['object']).columns
df_encoded.drop(columns=remaining_obj_cols, inplace=True)

# ✅ Key fix: fill remaining NaNs instead of dropping everything
df_encoded.fillna(0, inplace=True)

print("✅ Preprocessing complete!")
print("✅ Rows remaining after cleaning:", len(df_encoded))

# Step 4: Train/test split
print("\n📊 Splitting data...")
X = df_encoded.drop('loan_status', axis=1)
y = df_encoded['loan_status']

print("Class Distribution Before SMOTE:")
print(y.value_counts())

if y.nunique() < 2 or y.value_counts().min() < 2:
    raise ValueError("❌ Not enough samples or class diversity for SMOTE.")

# SMOTE
smote = SMOTE(random_state=42, k_neighbors=1)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Final split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

# Step 5: Train multiple models
print("\n🚀 Training models...")
models = {
    "Random Forest": RandomForestClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Gradient Boosting": GradientBoostingClassifier()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\n🔎 {name} Results")
    print(classification_report(y_test, y_pred))

# Step 6: Hyperparameter tuning (example)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 5, 10]
}
grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=3)
grid.fit(X_train, y_train)
print("Best Parameters (Random Forest):", grid.best_params_)

# Final model eval
print("\n📈 Final Evaluation:")
final_model = RandomForestClassifier(n_estimators=100, random_state=42)
final_model.fit(X_train, y_train)
final_pred = final_model.predict(X_test)
print(confusion_matrix(y_test, final_pred))
print(classification_report(y_test, final_pred))
