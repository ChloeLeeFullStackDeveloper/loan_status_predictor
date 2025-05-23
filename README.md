# 🧠 Loan Status Predictor

This project builds and compares machine learning models to predict the **loan repayment status** (Fully Paid or Charged Off) using a Lending Club dataset. It is developed as part of the **CPSC 3750: Artificial Intelligence** course.

---

## 🔍 Project Overview

- **Problem**: Predict whether a loan will be fully paid or charged off using applicant and loan details.
- **AI Topics**: Supervised learning, class imbalance handling (SMOTE), model evaluation, and hyperparameter tuning.
- **Goal**: Build a reliable, interpretable, and optimized loan classification system.

---

## 📦 Dataset

- Source: Lending Club loan dataset (2.2M+ rows)
- Filtered only loans with: `Fully Paid` or `Charged Off` statuses
- Cleaned and encoded key features including: `term`, `grade`, `home_ownership`, `purpose`, etc.
- Downsampled to 5000 rows for fast development/testing

### 🔧 Preprocessing steps:
- Dropped irrelevant and high-cardinality columns
- Cleaned `emp_length` values
- Label encoded categorical features
- Handled missing values with `fillna(0)`
- Applied **SMOTE** to handle class imbalance

---

## 🤖 Models Used

1. **Random Forest** ✅ Best performer
2. **Logistic Regression**
3. **Decision Tree**
4. **Gradient Boosting**

All models are trained using the same data split and evaluated using:
- Accuracy
- Precision / Recall / F1-score
- Confusion Matrix

---

## 🧪 Sample Output (Random Forest)

```
accuracy: 1.00
precision: 1.00
recall: 1.00
f1-score: 1.00
```

> *Note: These metrics are based on the 5,000-row sample dataset with SMOTE applied.*

---

## 🚀 How to Run

### 1. Clone this repo:
```bash
git clone https://github.com/your-username/loan_status_predictor.git
cd loan_status_predictor
```

### 2. Install dependencies:
```bash
pip install -r requirements.txt
```

### 3. Add data:
Place `lending_club_full.csv` in the `/data` folder  
> ⚠️ This file is ignored via `.gitignore` for privacy

### 4. Run the model:
```bash
cd notebooks
python3 loan_prediction.py
```

---

## 👥 Team Member Guide

### If you're a team member:
- Ensure you have Python 3.8+ installed
- Clone the project and install requirements
- Review the notebook or .py file for logic and edits
- Do **not commit the CSV dataset**
- Contribute to documentation and improvement ideas in `README.md` or `report.md`

---

## 🔮 Future Work Suggestions

- Add feature importance visualizations and SHAP explainability
- Deploy as a web app (Streamlit or Flask)
- Incorporate more advanced models (XGBoost, CatBoost)
- Evaluate with ROC-AUC, PR-AUC
- Add error analysis and fairness audit
- Run hyperparameter optimization (e.g., Optuna)

---

## 📚 Citations (APA 7th Edition)

- Dua, D., & Graff, C. (2019). *Lending Club Dataset*. University of California, Irvine. https://www.lendingclub.com/info/download-data.action
- Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research, 12, 2825–2830.
- Chawla, N. V., et al. (2002). *SMOTE: Synthetic Minority Over-sampling Technique*. Journal of Artificial Intelligence Research, 16, 321–357.

---

## 📄 License

This project is for educational use in CPSC 3750 - AI (University of Lethbridge).