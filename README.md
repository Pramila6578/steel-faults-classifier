# Steel Plate Fault Classification

## a. Problem Statement
<!-- Write in your own words. Suggested points: steel plates can have 7 fault types;
manual inspection is slow; goal is to classify the fault type from 27 measured
geometric/luminosity features and compare 5 classifiers. -->

## b. Dataset Description
- **Source:** UCI Machine Learning Repository – Steel Plates Faults
- **Instances:** <!-- verify, e.g. 1941 -->
- **Features:** <!-- verify, e.g. 27 numeric -->
- **Target:** `Fault`, 7 classes: Pastry, Z_Scratch, K_Scatch, Stains, Dirtiness, Bumps, Other_Faults
- **Class distribution:** <!-- paste output of y.value_counts() -->
- **Preprocessing:** one-hot fault columns collapsed into a single target; stratified 80/20 split (random_state=42); StandardScaler fitted on the training set only.
<!-- Add anything else you did or noticed. -->

## c. GitHub Repository Link
<!-- paste your repo URL here -->

## d. Models Used

### Comparison Table
<!-- Paste the markdown table printed by train_models.py (or model/results.csv) -->

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | | | | | | |
| Decision Tree | | | | | | |
| kNN | | | | | | |
| Naive Bayes | | | | | | |
| Support Vector Machine | | | | | | |

Precision, Recall and F1 use weighted averaging; AUC uses one-vs-rest (weighted).

### Observations
<!-- Write 2-3 sentences per model from YOUR results and confusion matrices.
Ideas: which classes were confused, effect of scaling, over/underfitting of the tree,
why NB's independence assumption may fail on correlated features, effect of k in kNN,
class imbalance impact on accuracy vs MCC. -->

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | |
| Decision Tree | |
| kNN | |
| Naive Bayes | |
| Support Vector Machine | |
| **Overall winner** | |

## How to Run
```
pip install -r requirements.txt
python model/train_models.py
streamlit run app.py
```

**Live app:** <!-- paste Streamlit Community Cloud link -->
