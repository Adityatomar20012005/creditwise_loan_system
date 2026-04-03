# creditwise_loan_system# CreditWise Loan Approval System 🏦

A machine learning project that predicts loan approval outcomes based on applicant financial and demographic data. Built using Python and scikit-learn, this project covers the full ML pipeline — from data preprocessing and EDA to model training and evaluation.

---

## 📌 Problem Statement

Banks and financial institutions receive thousands of loan applications daily. Manually evaluating each application is time-consuming and inconsistent. This project automates the loan approval decision by training classification models on historical applicant data.

---

## 📂 Dataset

- **File:** `loan_approval_data.csv`
- **Samples:** 1000 applicants
- **Features:** 20 columns (12 numerical, 8 categorical)

### Key Features:
| Feature | Type | Description |
|---|---|---|
| `Applicant_Income` | Numerical | Monthly income of the applicant |
| `Coapplicant_Income` | Numerical | Monthly income of co-applicant |
| `Credit_Score` | Numerical | Applicant's credit score |
| `DTI_Ratio` | Numerical | Debt-to-income ratio |
| `Loan_Amount` | Numerical | Requested loan amount |
| `Loan_Term` | Numerical | Loan repayment duration |
| `Savings` | Numerical | Applicant's savings |
| `Collateral_Value` | Numerical | Value of collateral offered |
| `Employment_Status` | Categorical | Employed / Self-employed / Unemployed |
| `Marital_Status` | Categorical | Married / Single |
| `Education_Level` | Categorical | Education qualification |
| `Loan_Purpose` | Categorical | Home / Car / Personal / Business, etc. |
| `Property_Area` | Categorical | Urban / Rural / Semiurban |
| `Gender` | Categorical | Male / Female |
| `Loan_Approved` | Target | Yes / No |

---

## ⚙️ Project Pipeline

### 1. Data Preprocessing
- Loaded dataset using pandas
- Handled missing values using `SimpleImputer`:
  - **Numerical columns** → Mean imputation
  - **Categorical columns** → Most frequent (mode) imputation

### 2. Exploratory Data Analysis (EDA)
Visualized key patterns in the data:
- **Loan Approval Rate** — Pie chart of approved vs. rejected loans
- **Gender Distribution** — Male vs. Female applicant ratio
- **Loan Purpose Breakdown** — Bar chart of loan categories
- **Property Area** — Distribution across Urban, Rural, Semiurban

### 3. Feature Engineering & Encoding
- **Label Encoding** → `Education_Level`, `Loan_Approved`
- **One-Hot Encoding** (drop first) → `Employment_Status`, `Marital_Status`, `Loan_Purpose`, `Property_Area`, `Gender`, `Employer_Category`

### 4. Correlation Analysis
- Generated a heatmap of feature correlations
- Ranked features by their correlation with `Loan_Approved`

### 5. Train-Test Split
- 80% training / 20% testing
- `random_state=42` for reproducibility

### 6. Feature Scaling
- Applied `StandardScaler` on both train and test sets (fit only on train)

---

## 🤖 Models Trained

| Model | Accuracy | Precision | Recall |
|---|---|---|---|
| Logistic Regression | **86.5%** | 78.33% | 77.05% |
| K-Nearest Neighbors (K=5) | 76.0% | 62.75% | 52.46% |
| Gaussian Naive Bayes | **86.5%** | 80.36% | 73.77% |

> **Best Models:** Logistic Regression and Gaussian Naive Bayes tied on accuracy. Naive Bayes edges out on precision.

---

## 🛠️ Tech Stack

- **Language:** Python 3
- **Libraries:** pandas, scikit-learn, matplotlib, seaborn
- **Environment:** JupyterLab / Jupyter Notebook

---

## 🚀 How to Run

1. Clone the repository:
```bash
   git clone https://github.com/your-username/creditwise-loan-system.git
   cd creditwise-loan-system
```

2. Install dependencies:
```bash
   pip install pandas scikit-learn matplotlib seaborn
```

3. Place `loan_approval_data.csv` in the project directory.

4. Open the notebook:
```bash
   jupyter notebook creditwise_loan_system.ipynb
```

5. Run all cells in order (Kernel → Restart & Run All).

---

---

## 📊 Key Insights

- Credit score, DTI ratio, and savings are among the strongest predictors of loan approval
- Logistic Regression performs well despite its simplicity, suggesting the decision boundary is largely linear
- KNN underperforms — likely sensitive to the high-dimensional feature space after one-hot encoding

---

## 🙋 Author

**Aditya**  
B.Tech Engineering Student | ML & AI Enthusiast  
[GitHub](https://github.com/Adityatomar20012005) • [LinkedIn](https://www.linkedin.com/in/aditya-tomar-431421387)
