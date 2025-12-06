[Home Credit Default Risk Prediction : Project Overview](https://github.com/amiyaranjan20-lgtm/Home-Credit-Default-Risk-Prediction)

# Introduction

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

### 2.2 Exploratory Data Analysis (EDA)

Before model building, I performed an exploratory analysis to understand the structure, quality, and behavior of the variables in the dataset. This included:

- Reviewing dataset dimensions, variable types, and initial distributions using `skimr`, `summary()`, and `str()`.  
- Checking class imbalance in the target variable (≈ 8% default, 92% repaid), which influenced the choice of metrics and modeling approach.  
- Examining missing patterns across features and identifying variables with extremely high missingness, which later guided feature pruning.  
- Investigating key numerical features such as age, employment duration, income, and credit ratios to detect outliers, skewness, and transformation needs.  
- Exploring relationships using pair plots, correlations, and simple aggregations to understand which predictors showed early signal for default risk.  

The EDA phase provided the foundation for informed preprocessing decisions and helped shape the modeling strategy, especially around handling missingness, class imbalance, and nonlinear feature behavior.

![](/Images/Visual%20Interpretation%20of%20taget%20variable-1.png)

![](/Images/unnamed-chunk-5-7.png)

![](/Images/unnamed-chunk-5-11.png)

![](/Images/unnamed-chunk-6-6.png)

![](/Images/unnamed-chunk-7-1.png)


### **2.3 Preprocessing Pipeline (recipes)**
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

### 2.5 Overall Project Solution (Team-Level Summary)

As a team, we aligned on a common goal: build a supervised machine learning system that predicts the probability of loan default for Home Credit applicants. Our collective solution included:

- Using the Kaggle dataset to engineer predictive features related to demographics, employment, and financial behavior.  
- Establishing a consistent preprocessing framework so all team members’ models trained on comparable, high-quality inputs.  
- Testing four models — Logistic Regression, Decision Tree, Random Forest, and XGBoost — using identical evaluation criteria (5-fold CV and AUC).  
- Comparing performance across the models to identify the approach that delivered the best ranking accuracy and stability.  
- Agreeing that **XGBoost** was the strongest model overall and recommending it as the final risk-scoring solution for the project.

This team-level solution provided the shared foundation for the final presentation and served as the unified answer to the business problem.

### **2.6 Model Performance**
- CART: AUC ≈ 0.59  
- Logistic Regression: AUC ≈ 0.74  
- Random Forest: AUC ≈ 0.73  
- **XGBoost: AUC ≈ 0.75 (best model)**  

The XGBoost model provided the strongest generalization and ranking capability.

![](/Images/Picture1.png)

### **2.7 Kaggle Submission Workflow**
The final prediction pipeline:

- Applies the trained recipe to the Kaggle test set  
- Generates predicted probabilities for each applicant  
- Produces `submission_xgb.csv` according to Kaggle format
- Submits the file to the Kaggle competition portal for evaluation  

After uploading the submission, the model achieved a **public leaderboard AUC score of 0.74203**, which closely matches the validation performance of the XGBoost model.  
A screenshot of the Kaggle score (`kaggle_score_screenshot.png`) is included in the repository for reference.  

![](/Images/kaggle_score_screenshot.png)

---

## 3. My Individual Contributions

This repository documents the work I carried out individually for the project, including:

- Complete data exploration, cleaning, and feature engineering  
- A full preprocessing system using `recipes`  
- Implementation and evaluation of different ML models  
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

## 7. Interpretation of Results and Key Takeaways

The final XGBoost model achieved a validation AUC of approximately **0.75**, which indicates strong discriminatory power for distinguishing between borrowers who are likely to repay versus those who are likely to default. In credit-risk modeling, any AUC above 0.70 is considered practically useful, and scores in the 0.75 range are commonly used in real underwriting systems.

### How to Interpret These Values
- **AUC ≈ 0.75** means the model correctly ranks a randomly chosen defaulter above a randomly chosen non-defaulter about 75% of the time.  
- This is significantly better than the baseline (AUC = 0.50), which cannot differentiate risk at all.  
- The model captures complex borrower patterns that simple rules or linear scoring systems would miss.  
- Consistency between validation AUC and Kaggle score indicates that the model generalizes well and is not overfitting.

### What These Results Tell Us

1. **Default risk is predictable**  
   Key engineered features (income ratios, age, employment duration, external scores) show measurable separation between high-risk and low-risk groups.

2. **Tree-based models capture financial behavior better**  
   Logistic regression provides a solid baseline, but nonlinear relationships dominate this dataset.  
   XGBoost captures these interactions more effectively.

3. **Missingness and skewness directly affect credit-risk modeling**  
   Addressing anomalies, extreme skew, and high-missing features improved stability and reduced noise in the model.

4. **Imbalanced data must be handled carefully**  
   With defaults at only ~8%, accuracy alone is misleading.  
   Shifting focus to AUC, Recall, and thresholding supported more meaningful evaluation.

### How This Justifies the Business Case
The project results directly support the original business problem:

- **Better identification of risky borrowers** reduces credit losses.  
- **More confident approvals for low-risk applicants** improves financial inclusion and loan-volume growth.  
- **Data-driven decision thresholds** (approve / review / decline) align the model with business risk appetite.  
- **Consistent ranking of applicants** supports underwriting, pricing, and portfolio monitoring.

In practical terms, a model with this level of performance gives Home Credit a reliable early-warning system that can materially reduce default rates while allowing good clients without traditional credit histories to be approved.  
This aligns with the company’s mission and the operational goals of responsible lending.

Together, the results confirm that machine learning–based risk scoring is an effective solution to the business problem and provides a foundation for further model refinement, monitoring, and eventual deployment.

## 8. Repository Contents

- **`Home_Credit_Default_Risk_Prediction.Rmd`**  
  Full notebook containing data cleaning, feature engineering, model development, evaluation, and the final submission workflow.

- **`submission_xgb.csv`**  
  Kaggle submission file generated from the final XGBoost model.

- **`Home Credit Default Risk.pdf`** 
  The presentation used to summarize the business problem, modeling approach, and key insights from the project.

- **`kaggle_score_screenshot.png`**  
  Screenshot of the Kaggle Public Leaderboard score for the submitted XGBoost model.

- **`/Images/` folder**  
  Contains all Exploratory Data Analysis (EDA) visuals used to understand data structure, distribution patterns, correlations, missingness, and early signals that informed preprocessing and modeling decisions.
