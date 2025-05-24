# 🧠 Loan Status Predictor

This project builds and compares machine learning models to predict the **loan repayment status** (Fully Paid or Charged Off) using Lending Club loan data. It supports **algorithm selection** through a **Streamlit web app**, built for **CPSC 3750: Artificial Intelligence** coursework.

---

## 🔍 Project Overview

- **Goal**: Predict loan status (Fully Paid or Charged Off) from borrower and loan features.
- **AI Concepts**: Supervised learning, data preprocessing, class imbalance handling (SMOTE), model evaluation, Streamlit app deployment.
- **Features**:  
  ✅ Train multiple algorithms  
  ✅ Interactive prediction web app  
  ✅ Algorithm selection dropdown in UI  
  ✅ Evaluation reports + visual diagrams

---

## 📦 Dataset

- **Source**: Lending Club full dataset (~2.2M+ rows)
- **Filtered** to only: `Fully Paid` and `Charged Off`
- **Used Features**:  
  `loan_amnt`, `int_rate`, `emp_length`, `term`, `grade`, `home_ownership`, `verification_status`, `purpose`

### 🔧 Preprocessing steps:

- Cleaned `emp_length` values
- Dropped irrelevant/high-cardinality columns
- Label encoded categorical features
- Filled missing values (`fillna(0)`)
- Applied **SMOTE** to handle class imbalance
- Sampled ~5,000 rows for fast development/testing

---

## 🤖 Models Used

1. **Random Forest** ✅ Best performer
2. **Logistic Regression**
3. **Decision Tree**
4. **Gradient Boosting**

## 🤖 Models Trained

Models are trained on the same dataset split:

| Algorithm              | Status   |
| ---------------------- | -------- |
| ✅ Random Forest       | Included |
| ✅ Logistic Regression | Included |
| ✅ Decision Tree       | Included |
| ✅ Gradient Boosting   | Included |

All models are saved as `.pkl` files in the `/models` folder:

---

## 🧪 Sample Output (Random Forest)

```
Example metrics shown from `loan_model_trainer.py`

🔎 Random Forest Results
accuracy: 0.87
precision: 0.84
recall: 0.86
f1-score: 0.85

🔎 Gradient Boosting Results
accuracy: 0.89
precision: 0.87
recall: 0.88
f1-score: 0.87
accuracy: 1.00
precision: 1.00
recall: 1.00
f1-score: 1.00
```

> _Note: These metrics are based on the 5,000-row sample dataset with SMOTE applied._

---

## 🌐 Streamlit Web App

You can predict loan outcomes interactively using a dropdown for algorithm selection.

### 🔮 Features:

- Choose algorithm (Random Forest, Logistic Regression, etc.)
- Enter loan details via sliders/selectboxes
- Displays prediction result with icon

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
1. Clone the repo
- git clone https://github.com/ChloeLeeFullStackDeveloper/loan_status_predictor.git

2. Install Dependencies
- pip install -r requirements.txt

3. Add the dataset
- /data/lending_club_full.csv (in the keggle site: https://www.kaggle.com/datasets/wordsforthewise/lending-club)

4. Train models
- cd notebooks
- python3 loan_prediction.py

5. Run the Streamlit app
- streamlit run loan_app.py
```

---

### 🛠 File Structure'

```
loan_status_predictor/
├── data/
│   └── lending_club_full.csv
├── images/
│   ├── confusion_matrix.png
│   └── model_pipeline.png
├── models/
│   ├── model_random_forest.pkl
│   ├── model_logistic_regression.pkl
│   ├── model_decision_tree.pkl
│   └── model_gradient_boosting.pkl
├── notebooks/
│   ├── loan_model_trainer.py
│   └── loan_prediction.py
├── loan_app.py
├── requirements.txt
└── README.md
```

## 👥 Team Member Guide

### If you're a team member:

- Ensure you have Python 3.8+ installed
- Clone the project and install requirements
- Review the notebook or .py file for logic and edits
- Do **not commit the CSV dataset**
- Contribute to documentation and improvement ideas in `README.md` or `report.md`

---

## 🔮 Future Work Suggestions

- Add SHAP/feature importance explanations

- ROC-AUC / PR-AUC metrics

- Deploy to cloud (Streamlit Share, Render, etc.)

- Add user authentication and prediction logging

- Hyperparameter tuning with Optuna or GridSearchCV

---

## 📚 References (APA 7th Edition)

- Dua, D., & Graff, C. (2019). _Lending Club Dataset_. University of California, Irvine. https://www.lendingclub.com/info/download-data.action
- Pedregosa, F., et al. (2011). _Scikit-learn: Machine Learning in Python_. Journal of Machine Learning Research, 12, 2825–2830.
- Chawla, N. V., et al. (2002). _SMOTE: Synthetic Minority Over-sampling Technique_. Journal of Artificial Intelligence Research, 16, 321–357.

---

## 📄 License

This project is for educational use in CPSC 3750 - AI (University of Lethbridge).
