# ml-assessment-Sona K S

Machine Learning Fundamentals — Graded Assessment

## What's in this project

```
ml-assessment-Sona K S/
├── part_a/
│   ├── q1_supervised.ipynb          # Heart disease classification (28 marks)
│   ├── q2_unsupervised.ipynb        # Customer segmentation with K-Means + PCA (22 marks)
│   └── q3_feature_engineering.ipynb # Retail regression pipeline (20 marks)
├── part_b/
│   └── business_analysis.md         # Business case analysis (30 marks)
└── data/
    ├── q1_heart_disease.csv
    ├── q2_customers.csv
    └── q3_retail_promotions.csv
```

## Quick Overview

### Part A: Coding Section (70 marks)

- **Q1 (28 marks):** Supervised learning classification pipeline for predicting heart disease. Implemented Decision Tree, Random Forest, and Gradient Boosting classifiers with comprehensive exploratory data analysis, preprocessing, model evaluation, and GridSearchCV hyperparameter tuning.
- **Q2 (22 marks):** Unsupervised learning customer segmentation using K-Means clustering. Applied the elbow method for optimal cluster selection, used PCA for dimensionality reduction and visualization, and provided cluster interpretation based on customer behavior patterns.
- **Q3 (20 marks):** Feature engineering and regression analysis for retail sales prediction. Implemented scikit-learn pipelines with both Linear Regression and Random Forest models, applied temporal train-test splitting, and performed feature importance analysis.

### Part B: Business Analysis (30 marks)

Comprehensive business case analysis for optimizing promotion effectiveness at a fashion retail chain. Covers ML problem formulation, data strategy with EDA planning, class imbalance handling, temporal evaluation methodology, feature importance communication, and end-to-end deployment with monitoring.

## What you need to run this

```
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
```

## How to get it running

1. Put the CSV files in the `data/` folder (the code looks for them there).
2. Open the notebooks in Jupyter or VS Code and run all the cells.
3. Everything's already been run and the outputs are saved, so you can see the results right away.

## How the marks break down

### Part A — Python Coding (70 marks)

#### Q1 — Supervised Learning: Heart Disease Classification (28 marks)
- **Task 1 — Data Loading & Inspection (3 marks):** Load and inspect the heart disease dataset, examining shape, data types, and missing value distribution.
- **Task 2 — Exploratory Data Analysis (5 marks):** Visualize target distribution, feature correlations, age distribution by disease status, and heart disease rates by chest pain type.
- **Task 3 — Data Preprocessing (5 marks):** Handle missing values using median imputation for numeric features and mode imputation for categorical variables. Apply one-hot encoding and standardization.
- **Task 4 — Model Training (5 marks):** Train Decision Tree, Random Forest, and Gradient Boosting classifiers on the preprocessed data.
- **Task 5 — Model Evaluation (6 marks):** Evaluate models using confusion matrices and classification metrics including precision, recall, and F1-score.
- **Task 6 — Hyperparameter Tuning (4 marks):** Use GridSearchCV to optimize Gradient Boosting hyperparameters (learning_rate, max_depth, n_estimators).

**Key Results:**
- Gradient Boosting achieved highest performance: ~90.8% accuracy, ~0.923 F1-score
- Random Forest: ~89.1% accuracy, ~0.893 F1-score
- Decision Tree: ~81.5% accuracy (baseline comparison)
- Tuned Gradient Boosting improved performance to ~91.85% accuracy with optimal hyperparameters (learning_rate=0.1, max_depth=4, n_estimators=200)

#### Q2 — Unsupervised Learning: Customer Segmentation (22 marks)
- **Task 1 — Data Loading & Inspection (3 marks):** Load and examine the customer dataset, reviewing feature types and distributions.
- **Task 2 — Exploratory Data Analysis (4 marks):** Analyze feature distributions and correlations to understand customer behavior patterns.
- **Task 3 — Data Preprocessing (3 marks):** Clean data and standardize all features (required for K-Means distance-based clustering).
- **Task 4 — Choosing K with the Elbow Method (4 marks):** Apply the elbow method to determine optimal number of clusters.
- **Task 5 — K-Means Clustering (4 marks):** Execute K-Means algorithm and assign customers to clusters.
- **Task 6 — PCA Visualisation (4 marks):** Apply PCA for dimensionality reduction and create 2D scatter plots colored by cluster.
- **Task 7 — Cluster Interpretation (3 marks):** Analyze cluster characteristics and provide business interpretation.

**Key Results:**
- Identified 3 optimal clusters: low-value, mid-value, and high-value customer segments
- PCA captured approximately 89% of variance in 2 principal components
- Cluster characteristics clearly differentiate customer segments by spending patterns and frequency

#### Q3 — Feature Engineering & Regression: Retail Promotions (20 marks)
- **Task 1 — Data Loading & Inspection (2 marks):** Load and inspect the retail sales dataset with feature overview.
- **Task 2 — Exploratory Data Analysis (3 marks):** Visualize sales distributions and promotion performance patterns.
- **Task 3 — Feature Engineering (3 marks):** Create temporal features and encode categorical variables for modeling.
- **Task 4 — Train-Test Split with Temporal Grain (2 marks):** Apply time-based data splitting to reflect real-world sequential prediction scenarios.
- **Task 5 — Model Training with scikit-learn Pipeline (4 marks):** Implement pipelines for Linear Regression and Random Forest Regressor models.
- **Task 6 — Model Evaluation & Comparison (4 marks):** Compare model performance using RMSE, MAE, R² metrics and generate parity plots.
- **Task 7 — Feature Importance Analysis (3 marks):** Identify and analyze most influential features in model predictions.

**Key Results:**
- Linear Regression outperformed Random Forest: R² = 0.69 vs 0.67
- Location and temporal features (festival days) emerged as most important predictors
- Temporal train-test split approach validates model performance on future-looking predictions

### Part B — Business Case Analysis (30 marks)

#### B1 — Problem Formulation (8 marks)
- **B1(a) — ML Problem Type (3 marks):** Define the problem as supervised regression for predicting items sold based on store features and promotion type.
- **B1(b) — Target Variable Justification (3 marks):** Justify selection of items_sold as target variable instead of revenue, aligned with business objective of volume maximization.
- **B1(c) — Global vs Location-Stratified Modelling (2 marks):** Discuss trade-offs between single global model and location-specific models for capturing regional promotion sensitivity.

#### B2 — Data & EDA Strategy (10 marks)
- **B2(a) — Data Joining & Grain Definition (4 marks):** Specify data integration approach and define store-month as the analytical grain.
- **B2(b) — EDA Before Modelling (4 marks):** Outline four key exploratory analyses: promotion effectiveness by type, temporal trends, feature correlations, and location-specific promotion performance.
- **B2(c) — Class Imbalance Handling (2 marks):** Address imbalanced data issue (80% no-promotion) with stratified sampling and weighted loss techniques.

#### B3 — Model Evaluation & Deployment (12 marks)
- **B3(a) — Train-Test Split & Metrics (4 marks):** Define temporal train-test split strategy and select appropriate evaluation metrics (RMSE, MAE, R², promotion ranking accuracy).
- **B3(b) — Explaining Different Recommendations (4 marks):** Demonstrate model interpretability using feature importance and SHAP values to explain varying promotion recommendations across store-months.
- **B3(c) — End-to-End Deployment Process (4 marks):** Detail production deployment workflow including model storage, monthly inference pipeline, monitoring framework, and retraining triggers.

---

## Submission Checklist

- [x] **Part A: Q1 Supervised Learning** — All 6 tasks done with working code and results
- [x] **Part A: Q2 Unsupervised Learning** — All 7 tasks including clustering and visualization
- [x] **Part A: Q3 Feature Engineering** — All 7 tasks with pipelines and feature analysis
- [x] **Part B: Business Case Analysis** — All the business thinking covered
- [x] **Data Files** — All three CSV files in the data folder
- [x] **Reproducibility** — Notebooks run and outputs saved
- [x] **Documentation** — Explained what I did and why

## Total Score: 100 marks (70 + 30)
