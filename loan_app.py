import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Label mappings
purpose_mapping = {
    'credit_card': 0,
    'debt_consolidation': 1,
    'home_improvement': 2,
    'major_purchase': 3,
    'small_business': 4
}

verification_mapping = {
    'Verified': 0,
    'Source Verified': 1,
    'Not Verified': 2
}

grade_mapping = {
    'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6
}

term_mapping = {
    '36 months': 0,
    '60 months': 1
}

home_ownership_mapping = {
    'RENT': 0,
    'OWN': 1,
    'MORTGAGE': 2,
    'OTHER': 3
}

model_files = {
    "Random Forest": "model_random_forest.pkl",
    "Logistic Regression": "model_logistic_regression.pkl",
    "Decision Tree": "model_decision_tree.pkl",
    "Gradient Boosting": "model_gradient_boosting.pkl"
}

st.title("💸 Loan Status Predictor")
st.markdown("Predict whether a loan is **Fully Paid** or **Charged Off** based on input features.")

algorithm = st.selectbox("Choose Algorithm", list(model_files.keys()))

st.header("📋 Loan Application Info")
loan_amnt = st.slider("Loan Amount", 1000, 40000, 15000)
int_rate = st.slider("Interest Rate (%)", 5.0, 30.0, 12.0)
emp_length = st.slider("Employment Length (Years)", 0, 10, 3)

term = st.selectbox("Term", list(term_mapping.keys()))
grade = st.selectbox("Credit Grade", list(grade_mapping.keys()))
home_ownership = st.selectbox("Home Ownership", list(home_ownership_mapping.keys()))
verification_status = st.selectbox("Verification Status", list(verification_mapping.keys()))
purpose = st.selectbox("Loan Purpose", list(purpose_mapping.keys()))

if st.button("🔍 Predict Loan Status"):
    try:
        model_path = os.path.join("models", model_files[algorithm])
        if not os.path.exists(model_path):
            st.error(f"❌ Model file not found: {model_files[algorithm]}")
        else:
            with open(model_path, "rb") as file:
                model = pickle.load(file)

            input_data = pd.DataFrame({
                'loan_amnt': [loan_amnt],
                'int_rate': [int_rate],
                'emp_length': [emp_length],
                'term': [term_mapping[term]],
                'grade': [grade_mapping[grade]],
                'home_ownership': [home_ownership_mapping[home_ownership]],
                'verification_status': [verification_mapping[verification_status]],
                'purpose': [purpose_mapping[purpose]]
            })

            prediction = model.predict(input_data)[0]
            proba = model.predict_proba(input_data)[0][prediction]

            status = "✅ Fully Paid" if prediction == 0 else "❌ Charged Off"
            st.subheader("🔮 Prediction Result:")
            st.success(f"{status} (Confidence: {proba*100:.2f}%)")

    except Exception as e:
        st.error(f"❌ Prediction failed: {str(e)}")
