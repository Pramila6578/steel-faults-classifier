# Steel Plate Fault Classification

## a. Problem Statement
Steel plates can develop surface and shape faults such as scratches, bumps, stains and pastry defects, and finding them by manual inspection is slow and inconsistent. This project classifies the fault type of a plate from 27 measurements taken from it (geometry, luminosity, steel type and indices). Five classification models (Logistic Regression, Decision Tree, kNN, Naive Bayes and SVM) are trained on the same data, compared using six evaluation metrics, and demonstrated in an interactive Streamlit app.

## b. Dataset Description
- **Source:** UCI Machine Learning Repository, Steel Plates Faults (Buscema, Terzi & Tastle, 2010). https://archive.ics.uci.edu/dataset/198/steel+plates+faults
- **Instances:** 1,941 | **Features:** 27 numeric | **Missing values:** none
- **Target:** `Fault`, 7 classes (multi-class classification)
- **Class distribution:** Other_Faults 673, Bumps 402, K_Scratch 391, Z_Scratch 190, Pastry 158, Stains 72, Dirtiness 55 (imbalanced)
- **Preprocessing:** the 7 one-hot fault columns were collapsed into one target column; stratified 80/20 train/test split (random_state=42, 389 test rows); StandardScaler fitted on the training set only.

## c. GitHub Repository Link
https://github.com/Pramila6578/steel-faults-classifier

## d. Models Used

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7275 | 0.9066 | 0.7329 | 0.7275 | 0.7277 | 0.6487 |
| Decision Tree | 0.7429 | 0.8283 | 0.7455 | 0.7429 | 0.7429 | 0.6688 |
| kNN | 0.7275 | 0.8941 | 0.7318 | 0.7275 | 0.7245 | 0.6548 |
| Naive Bayes | 0.6093 | 0.8840 | 0.6700 | 0.6093 | 0.5883 | 0.5484 |
| Support Vector Machine | 0.7506 | 0.8974 | 0.7578 | 0.7506 | 0.7521 | 0.6769 |

Precision, Recall and F1 use weighted averaging and AUC uses one-vs-rest (weighted), since this is a multi-class problem. Weighted recall is mathematically equal to accuracy, so those two columns are identical.

**Tuning experiment:** GridSearchCV (5-fold, scored on weighted F1) was run on kNN (best: k=3, distance weighting), Decision Tree (best: max_depth=8, min_samples_leaf=3) and SVM (best: C=5). Results are in `model/tuning_comparison.csv`. Gains were small (for example SVM accuracy 0.7506 to 0.7558), which is within noise for a 389-row test set. The table above and the app use the default models.

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best AUC of all models (0.9066) with accuracy 0.7275. It does well on well-defined faults such as K_Scratch (recall 0.910) and Stains (0.857) but struggles with Bumps (0.568) and Pastry (0.562), probably because these classes are not linearly separable. |
| Decision Tree | Second-best accuracy (0.7429) and the best recall on Dirtiness (0.909) and Z_Scratch (0.921), but the lowest AUC (0.8283), since an unpruned tree gives almost 0/1 probability estimates. Limiting depth during tuning raised AUC to 0.898, which suggests the default tree overfits. |
| kNN | Same accuracy as Logistic Regression (0.7275) with AUC 0.8941. Very good on K_Scratch (recall 0.962) but worst on Pastry (0.375), whose samples likely overlap with neighbours of other classes. Tuning (k=3, distance weighting) improved accuracy to 0.7404. |
| Naive Bayes | Weakest model (accuracy 0.6093, F1 0.5883, MCC 0.5484). Recall on Other_Faults is only 0.252, although it is best on Bumps (0.753). Its independence assumption is violated: 7 feature pairs have correlation above 0.9 (for example X_Minimum/X_Maximum and Pixels_Areas/Sum_of_Luminosity). Its AUC (0.884) is close to the others, so its probability ranking is reasonable but its hard predictions are poor. |
| Support Vector Machine | Best accuracy (0.7506), precision (0.7578), recall, F1 (0.7521) and MCC (0.6769). The RBF kernel can model non-linear boundaries on the scaled features. Its weakest classes are Pastry (recall 0.531) and Bumps (0.642). Tuning (C=5) gave only a small gain. |
| **Overall winner** | **Support Vector Machine**, as it leads on five of the six metrics. The margin over the Decision Tree (0.7506 vs 0.7429) is small on a 389-row test set, and Logistic Regression has the best AUC. Pastry and Bumps were the hardest classes for every model. |

## Streamlit App
**Live app:** https://steel-faults-classifier-yfwgirsbrkavvqdzmy6t97.streamlit.app/

Features: CSV upload (test data), model selection dropdown, evaluation metrics, confusion matrix and classification report.

## How to Run
```
pip install -r requirements.txt
jupyter notebook model/train_models.ipynb   # run all cells: trains the models and creates test_data.csv
streamlit run app.py
```
