# Customer Segmentation & Purchase Prediction

End-to-end data science project on e-commerce session data: statistical analysis, EDA, customer
segmentation (clustering), and purchase prediction (classification).

**Pipeline:** Data → Statistics → EDA → Preprocessing → Clustering → Classification → Evaluation → Validation → Business Insights

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/YOUR_REPO/blob/main/notebooks/Customer_Segmentation_Purchase_Prediction.ipynb)

> Replace `YOUR_USERNAME/YOUR_REPO` above with your actual GitHub path once pushed, and the badge
> will let anyone open the notebook directly in Colab.

## Problem Statement

An e-commerce company wants to understand its customer base and predict purchasing behavior in order
to improve marketing targeting, retention, and conversion. This project segments website sessions into
behavioral customer groups and builds a classification model to predict whether a session will result
in a purchase (`Revenue`).

## Dataset

- **Name:** Online Shoppers Purchasing Intention Dataset
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention)
- **Size:** 12,330 sessions, 18 features
- **Target:** `Revenue` (boolean — did the session end in a purchase)

The raw CSV is included at `data/online_shoppers_intention.csv` for reproducibility. It is also
downloaded automatically by the notebook itself if not present, so the repo works even if you don't
clone the data file (e.g. via Git LFS exclusion or a shallow clone).

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── data/
│   └── online_shoppers_intention.csv
└── notebooks/
    └── Customer_Segmentation_Purchase_Prediction.ipynb
```

## Project Approach

1. **Data Understanding** — shape, dtypes, missing values, duplicates, unique values, target balance
2. **Statistical Analysis** — mean/median/mode/variance/IQR, distribution shape, correlation, hypothesis tests (Mann-Whitney U, Kruskal-Wallis)
3. **EDA** — univariate, bivariate, and multivariate visualizations
4. **Preprocessing** — encoding categorical variables, feature scaling
5. **Customer Segmentation** — K-Means (Elbow Method + Silhouette Score for K selection), Hierarchical (Ward) clustering for comparison, business-named segments
6. **Purchase Prediction** — Logistic Regression, Decision Tree, Random Forest, AdaBoost, KNN (class-weighted for imbalance)
7. **Model Evaluation** — Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrices, ROC/PR curves
8. **Validation** — train/test overfitting check, feature importance, error analysis, segment validation
9. **Business Insights & Recommendations**

## Key Results

- Dataset has no missing values; a small number of duplicate rows were removed.
- Target is imbalanced (~85% No-Purchase / ~15% Purchase), so **Recall** (supported by F1 and
  ROC-AUC) was used as the primary evaluation metric rather than accuracy.
- `PageValues`, `ProductRelated` engagement, and low bounce/exit rates are the strongest behavioral
  signals of purchase intent.
- K-Means (K=4, chosen via Elbow Method + Silhouette Score, cross-checked against Hierarchical
  clustering) produced four behaviorally distinct, business-nameable customer segments with clearly
  different purchase rates — e.g. High-Value/High-Intent, Engaged Browsers, Window Shoppers, and
  Low-Engagement Visitors.
- Full model comparison, metrics, and business recommendations are documented inside the notebook.

## Getting Started

### Option 1 — Google Colab (recommended, zero setup)
Click the "Open in Colab" badge above, or upload `notebooks/Customer_Segmentation_Purchase_Prediction.ipynb`
manually. Then **Runtime → Run all**. The dataset is downloaded automatically if not already present —
no file upload or `pip install` needed.

### Option 2 — Run locally
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
pip install -r requirements.txt
jupyter notebook notebooks/Customer_Segmentation_Purchase_Prediction.ipynb
```
Then run all cells (`Kernel → Restart & Run All`).

## Tools & Technologies

Python · NumPy · Pandas · Matplotlib · Seaborn · SciPy · Scikit-learn · Jupyter Notebook

## Limitations & Future Work

- Sessions aren't linked to individual customers across visits, so segments describe session-level,
  not long-term customer-level, behavior.
- Class imbalance could be addressed more directly with SMOTE / resampling.
- Models could be hyperparameter-tuned with `GridSearchCV` and cross-validation for a fairer comparison.

## License

This project is released under the [MIT License](LICENSE). The dataset is provided by the UCI Machine
Learning Repository and is subject to its own terms.
