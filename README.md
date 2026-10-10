# Cardiovascular Disease Prediction Using Machine Learning

A final year project that compares **seven machine learning models** for predicting cardiovascular disease from 11 patient examination and lifestyle measurements. All models are trained, tuned and evaluated on the same train / validation / test split of a real 70,000-record dataset.

> **Disclaimer:** This is an educational project. It is not a medical device and must not be used for diagnosis or treatment decisions.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Models](#models)
- [Results](#results)
- [Key Findings](#key-findings)
- [Limitations](#limitations)
- [Repository Structure](#repository-structure)
- [How to Run](#how-to-run)
- [Requirements](#requirements)
- [Author](#author)

## Overview

| | |
|---|---|
| **Task** | Binary classification: does the patient have cardiovascular disease? |
| **Target** | `cardio` (1 = disease, 0 = no disease) |
| **Models** | Logistic Regression, Decision Tree, Linear Regression (baseline), SVM, XGBoost, K-Nearest Neighbors, 1D CNN |
| **Metrics** | Accuracy, Precision, Recall, F1, ROC-AUC |
| **Environment** | Python, Google Colab |

## Dataset

The project uses the **Cardiovascular Disease dataset** published on Kaggle by Svetlana Ulianova.

- Source: https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset
- 70,000 patient records, 11 input features and 1 target (plus an `id` column)
- Classes are almost perfectly balanced (34,979 with disease, 35,021 without)
- No missing values, but the data contains entry errors that must be cleaned
- The CSV uses a **semicolon (`;`) separator**

| Feature | Description |
|---|---|
| `age` | Age in days (converted to years during cleaning) |
| `gender` | 1 = women, 2 = men |
| `height` | Height in cm |
| `weight` | Weight in kg |
| `ap_hi` | Systolic blood pressure |
| `ap_lo` | Diastolic blood pressure |
| `cholesterol` | 1 = normal, 2 = above normal, 3 = well above normal |
| `gluc` | 1 = normal, 2 = above normal, 3 = well above normal |
| `smoke` | Smoker (0/1) |
| `alco` | Alcohol intake (0/1) |
| `active` | Physical activity (0/1) |
| `cardio` | **Target**: cardiovascular disease (0/1) |

> The dataset file is not redistributed here. Download `cardio_train.csv` from Kaggle and place it in the project folder (see [How to Run](#how-to-run)).

## Methodology

1. **Exploratory data analysis**: class balance, age distribution, correlation heatmap.
2. **Data cleaning**
   - Drop `id`; convert `age` from days to years.
   - Remove impossible values: `ap_hi` outside 70 to 250, `ap_lo` outside 40 to 200, `ap_hi` not greater than `ap_lo`, `height` outside 120 to 220 cm, `weight` outside 30 to 200 kg.
   - Remove duplicate rows.
   - Result: **70,000 rows reduced to 68,591** (1,409 removed).
3. **Stratified split (70 / 15 / 15)**: 48,013 train, 10,289 validation, 10,289 test (`random_state=42`).
4. **Scaling**: `StandardScaler` fitted on the training set only, then applied to validation and test data (no data leakage). Scaled data is used by Logistic Regression, Linear Regression, SVM, KNN and the CNN. Decision Tree and XGBoost use unscaled data.
5. **Tuning on the validation set** and a **single final evaluation on the test set**.

## Models

| Model | Notes |
|---|---|
| Logistic Regression | `max_iter=1000` |
| Decision Tree | `max_depth` tuned from 1 to 20 on validation accuracy (best: 8) |
| Linear Regression | Baseline only: regression output thresholded at 0.5 |
| SVM | RBF kernel, `C=1.0`, `gamma='scale'` |
| XGBoost | 500 estimators max, learning rate 0.1, `max_depth=4`, early stopping (20 rounds) on validation log loss |
| KNN | K tuned over {3, 5, 9, 15, 21, 25, 31, 41, 51} on validation accuracy (best: 51, the largest value tried) |
| 1D CNN | Conv1D(32) - MaxPool - Conv1D(64) - MaxPool - Flatten - Dense(64) - Dropout(0.3) - Dense(1, sigmoid); Adam, binary cross-entropy, early stopping (patience 5, up to 50 epochs) |

## Results

Test-set results (10,289 patients), sorted by ROC-AUC. These are the values saved by the notebook in `model_comparison.csv`, rounded to four decimals.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| XGBoost | 0.7322 | 0.7557 | 0.6781 | 0.7148 | **0.8033** |
| 1D CNN | 0.7315 | 0.7513 | 0.6836 | 0.7158 | 0.8006 |
| Logistic Regression | 0.7246 | 0.7524 | 0.6608 | 0.7036 | 0.7939 |
| Linear Regression (baseline) | 0.7215 | 0.7542 | 0.6486 | 0.6974 | 0.7934 |
| KNN (K = 51) | 0.7230 | 0.7463 | 0.6669 | 0.7044 | 0.7920 |
| SVM (RBF) | 0.7300 | 0.7598 | 0.6643 | 0.7089 | 0.7917 |
| Decision Tree (depth 8) | 0.7297 | 0.7536 | 0.6741 | 0.7117 | 0.7896 |

Training accuracy is about 0.73 to 0.74 for every model, close to its test accuracy, so there is no sign of serious overfitting.

## Key Findings

- **All seven models perform almost identically**: 72 to 73% accuracy and ROC-AUC between 0.79 and 0.80. XGBoost and the CNN are marginally ahead, but the gap is only about one percentage point, so the ranking should not be over-interpreted.
- With only 11 features, the information in the data limits what any algorithm can learn. A simple Logistic Regression is almost as good as XGBoost or a CNN.
- Strongest correlations with the target (cleaned data): systolic blood pressure (0.43), diastolic blood pressure (0.34), age (0.24), cholesterol (0.22) and weight (0.18).
- Recall is about 65 to 68% across the models, so roughly one in three patients with the disease is missed. In a medical setting this false-negative rate matters as much as accuracy.

## Limitations

- Only 11 measurements are available; richer clinical features would likely help.
- Results come from one dataset and one random split; they are not clinically validated.
- Data-entry errors were handled with simple range rules, not clinical review.
- KNN's best K was the largest value in the search grid, so a larger K might do slightly better.
- Linear Regression predicts numbers rather than classes and is included only as a baseline.

**Possible future work:** add features such as BMI, use cross-validation and grid search, tune the decision threshold for higher recall, and test on an independent dataset.

## Repository Structure

```
.
├── Cardio_Disease_ML_Colab.ipynb   # Full pipeline: EDA, cleaning, 7 models, comparison
├── model_comparison.csv            # Test-set metrics for all models
├── README.md
└── cardio_train.csv                # Dataset (download from Kaggle, see below)
```

## How to Run

### Option 1: Google Colab (recommended)

1. Download `cardio_train.csv` from the [Kaggle dataset page](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset).
2. Open [Google Colab](https://colab.research.google.com/) and choose **File > Upload notebook** to upload `Cardio_Disease_ML_Colab.ipynb`.
3. Upload `cardio_train.csv` to the Colab session (or place it in Google Drive and mount Drive), and set `DATA_PATH` in the loading cell accordingly.
4. Choose **Runtime > Run all**. SVM, KNN and the CNN each take a few minutes.

### Option 2: Run locally

```bash
git clone <your-repository-url>
cd <repository-folder>
pip install pandas numpy scikit-learn xgboost tensorflow matplotlib seaborn jupyter
jupyter notebook Cardio_Disease_ML_Colab.ipynb
```

The notebook was written for Colab. When running locally, remove or skip the cells that use `google.colab` (`files.upload()` and `files.download()`), and set `DATA_PATH = 'cardio_train.csv'` with the CSV in the same folder.

## Requirements

- Python 3
- pandas, numpy
- scikit-learn
- xgboost
- tensorflow (Keras)
- matplotlib, seaborn

## Author

**Md Abu Said**
Department of Computer Science and Engineering (CSE)
National Institute of Textile Engineering and Research (NITER)

Supervisor: Tanvir Ahmed
