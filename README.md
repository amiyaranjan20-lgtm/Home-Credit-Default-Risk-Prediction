
# Home Credit Default Risk – Individual Portfolio (Amiya Ranjan Sahoo)

Can we use historical loan data to estimate how likely a new applicant is to default, especially when they don’t have a traditional credit history?

This repository contains my individual analysis and modeling work for the Home Credit Default Risk project. It focuses on building and comparing several machine learning models in R to predict default probability and interpreting what those results mean from a business perspective.

---

## 1. Business Context and Objective

Home Credit lends to customers with thin or no credit files. This makes risk assessment challenging and can lead to:

- rejecting low-risk borrowers without documented history, and  
- approving borrowers who may be at high risk of default.

The guiding business question is:

> How accurately can we estimate a client’s probability of default using available demographic, financial, and behavioral data?

The core objective of this project is to:

- Build predictive models that outperform a naïve “always repaid” baseline  
- Create a transparent, reproducible modeling workflow  
- Interpret results in a way that supports lending decisions, risk management, and operational policy

---

## 2. Modeling Workflow Overview

My analysis follows a structured, reproducible workflow:

### **2.1 Data Understanding and Initial Cleaning**
- Standardized variable names and validated important columns  
- Handled the known 365243 sentinel in `days_employed`  
- Created derived, interpretable numeric features such as `age_years` and `emp_years`  
- Performed a detailed missingness audit and removed high-missing and zero-variance predictors

### **2.2 Preprocessing Pipeline (recipes)**
A complete preprocessing system was developed using the `recipes` package, incorporating:

- Median and mode imputation  
- Rare level grouping for categorical variables  
- One-hot encoding  
- Yeo–Johnson transformation for skewed variables  
- Centering and scaling  
- Removal of highly correlated numeric predictors

This ensures consistent, clean input for all downstream models.

### **2.3 Train/Validation Split**
- A **70/30 stratified split** preserved the ~8% default rate  
- A majority-class baseline established the minimum standard (AUC = 0.50)

### **2.4 Model Development**
I trained and compared four supervised learning models:

1. **Penalized Logistic Regression (glmnet)**  
2. **Decision Tree (CART)**  
3. **Random Forest (ranger)**  
4. **Gradient Boosting (XGBoost)**  

All models were trained using:

- 5-fold cross-validation  
- ROC/AUC as the main performance metric  
- Identical preprocessing and evaluation methodology  

### **2.5 Model Performance**
- CART: AUC ≈ 0.59  
- Logistic Regression: AUC ≈ 0.74  
- Random Forest: AUC ≈ 0.73  
- **XGBoost: AUC ≈ 0.75 (best model)**  

The XGBoost model provided the strongest generalization and ranking capability.

### **2.6 Kaggle Submission Workflow**
The final prediction pipeline:

- Applies the trained recipe to the Kaggle test set  
- Generates predicted probabilities for each applicant  
- Produces `submission_xgb.csv` according to Kaggle format  

---

## 3. My Individual Contributions

This repository documents the work I carried out individually for the project, including:

- Complete data exploration, cleaning, and feature engineering  
- A full preprocessing system using `recipes`  
- Implementation and evaluation of four different ML models  
- Development of consistent evaluation tools (AUC, ROC, confusion matrices, threshold analysis)  
- The full pipeline for generating a Kaggle-ready submission file  

The modeling logic, evaluation routines, and workflow decisions reflect my own approach to solving the credit-risk prediction task.

---

## 4. Business Value of the Model

A model with an AUC around 0.75 has practical value for a lender like Home Credit:

- **Lower default rates** through early identification of high-risk borrowers  
- **Higher approval rates** for reliable customers with limited credit history  
- **Improved pricing and loan-limit decisions** through risk segmentation  
- **Greater operational efficiency** for underwriting and customer review processes  

Essentially, the model helps balance risk and growth by improving the lender’s ability to distinguish between low- and high-risk applicants.

---

## 5. Challenges Encountered

A few key challenges influenced the project:

1. **Severe class imbalance** (~8% default)  
2. **Highly skewed numeric distributions and high-missing variables**  
3. **Need for preprocessing reproducibility across training and test sets**  
4. **Ensuring fair, cross-validated model comparisons**  
5. **Managing computational cost for ensemble models**

Working through these challenges improved both the model reliability and my understanding of applied ML workflows.

---

## 6. What I Learned

This project strengthened several practical skills:

- Reproducible data science using R Markdown, recipes, and caret  
- Handling imbalanced data and interpreting AUC, precision, recall, and F1  
- Building and evaluating multiple machine learning models fairly  
- Designing structured preprocessing pipelines  
- Communicating technical results in business language  

The experience reflects a complete end-to-end machine learning workflow, similar to what would occur in a real credit-risk modeling environment.

---

## 7. Repository Contents

- **`Home_Credit_Default_Risk_Prediction.Rmd`**  
  Full notebook containing data cleaning, feature engineering, model development, evaluation, and the final submission workflow.

- **`submission_xgb.csv`**  
  Kaggle submission file generated from the final XGBoost model.


- **`Home Credit Default Risk.pdf`** 
  The presentation used to summarize the business problem, modeling approach, and key insights from the project.

- **`kaggle_score_screenshot.png`**  
  Screenshot of the Kaggle Public Leaderboard score for the submitted XGBoost model.
