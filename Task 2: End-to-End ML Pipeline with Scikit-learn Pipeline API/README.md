# Task 2: End-to-End ML Pipeline with Scikit-learn Pipeline API

Build a reusable, production-ready machine learning pipeline that predicts customer churn using the Telco Customer Churn dataset. The project compares Logistic Regression and Random Forest, tunes each with `GridSearchCV`, and exports the best-performing pipeline with `joblib`.

## Objective

Predict whether a telecom customer will churn (leave the service) based on their account and service details, while following ML engineering best practices — clean preprocessing, no data leakage, hyperparameter tuning, and a reusable exported model.

## Dataset

**Telco Customer Churn Dataset** — customer account information including contract type, monthly charges, tenure, internet service, and other service subscriptions, with a binary `Churn` label (Yes/No).

## Approach

1. **Data cleaning** — converted `TotalCharges` to numeric, dropped invalid/missing rows, removed the non-predictive `customerID` column
2. **Train/test split** — stratified split to preserve the churn/non-churn ratio in both sets
3. **Preprocessing** — built with `ColumnTransformer`:
   - Numeric features scaled with `StandardScaler`
   - Categorical features encoded with `OneHotEncoder`
4. **Pipelines** — built two separate `Pipeline` objects (preprocessing + model), one for each algorithm:
   - Logistic Regression
   - Random Forest
5. **Hyperparameter tuning** — used `GridSearchCV` (5-fold cross-validation) to tune each pipeline independently
6. **Evaluation** — compared both tuned pipelines on the held-out test set using precision, recall, F1-score, and ROC-AUC (not accuracy alone, since churn is an imbalanced class)
7. **Model selection & export** — selected the better-performing pipeline and saved it with `joblib` for reuse on new, raw data

## Results

| Metric (Churn class) | Logistic Regression | Random Forest |
|---|---|---|
| Precision | 0.64 | 0.63 |
| Recall | **0.57** | 0.51 |
| F1-score | **0.60** | 0.57 |
| ROC-AUC | **0.8351** | 0.8316 |
| Accuracy | 0.80 | 0.79 |

**Logistic Regression was selected as the final model**, outperforming Random Forest on recall, F1-score, and ROC-AUC — the metrics that matter most for catching at-risk customers in an imbalanced churn dataset.

## Tech Stack

- Python
- scikit-learn (`Pipeline`, `ColumnTransformer`, `GridSearchCV`)
- pandas / numpy
- joblib

## Project Structure

```
.
├── Task2_ML_Pipeline_Churn.ipynb   # Full annotated notebook (Google Colab)
├── logistic_regression_pipeline.pkl # Exported final pipeline
└── README.md
```

## How to Run

1. Open `Task2_ML_Pipeline_Churn.ipynb` in Google Colab
2. Run all cells top to bottom (no GPU required — CPU runtime is sufficient)
3. The dataset is loaded automatically from a public URL; see the notebook for a manual-upload fallback if needed
4. The final trained pipeline is saved as a `.pkl` file, ready to reload with `joblib.load()` and use on new customer data

## Key Learning Outcomes

- Built a leak-free preprocessing + modeling workflow using `Pipeline` and `ColumnTransformer`
- Tuned and compared two distinct models fairly, using multiple evaluation metrics
- Learned to select a model based on evidence rather than assuming a more complex model performs better
- Exported a fully reusable pipeline for real-world, unprocessed input data
