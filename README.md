# 🛒 DSN Mart Sales Prediction

A regression project predicting total product-store sales for DSN Mart's retail chain across
Nigeria — built for the **DSN Bootcamp Qualification Hackathon 2026 (ML Track)** on Kaggle,
comparing **Linear Regression**, **Random Forest**, and **XGBoost**, scored on **RMSE**.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-data%20wrangling-150458?logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-modeling-F7931E?logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-gradient%20boosting-EB5E28)
![License](https://img.shields.io/badge/license-MIT-green)

## 📌 Project Overview

DSN Mart wants to understand what drives product sales across its different store formats
and locations. This project builds a full pipeline that:

1. Cleans messy real-world retail data — inconsistent category casing, informative missing
   values, zero-inflated visibility readings
2. Explores which product and store attributes actually track with sales
3. Trains and compares **three regression models** — Linear Regression, Random Forest, and
   XGBoost — each in its own notebook, scored on **RMSE** (the competition's own metric)
4. Generates a Kaggle-ready submission file from the best-performing model

## 🗂️ Repo Structure

```
dsn-sales-prediction/
├── data/
│   ├── train.csv / test.csv / sample_submission.csv  # competition data
│   ├── X_train.csv / X_val.csv / y_train.csv / y_val.csv  # shared local split
│   ├── test_wrangled.csv                  # cleaned competition test set
│   └── *_results.json                     # validation RMSE per model
├── notebooks/
│   ├── 1_eda.ipynb                        # cleaning + exploratory data analysis
│   ├── 2_model_linear_regression.ipynb    # Model 1: Linear Regression
│   ├── 3_model_random_forest.ipynb        # Model 2: Random Forest
│   ├── 4_model_xgboost.ipynb              # Model 3: XGBoost
│   └── 5_model_comparison.ipynb           # compares all 3, prepares final Kaggle submission
├── submissions/                           # one CSV per model + submission_final.csv
├── images/                                # chart exports
├── requirements.txt
├── LICENSE
└── README.md
```

## 🧹 Data Cleaning Summary

| Issue in raw data | How it was handled |
|---|---|
| `product_category` has 48 distinct raw values for only 16 real categories (`"Snack Foods"` / `"snack foods"` / `"SNACK FOODS"`, etc.) | Standardized to title case, collapsing duplicates |
| `store_size` missing ~28% of rows, and missingness is *not* random (Corner Shops and Standard Supermarkets are missing size far more often than Flagship Hypermarkets/Superstores) | Kept as its own `"Unknown"` category rather than guessed — the missingness itself is informative |
| `product_weight_kg` missing ~18% of rows, and turns out inconsistent even *within* the same product (same `product_code` shows multiple different weights) | Imputed with the median weight for that product's *category*, since per-product grouping wasn't reliable |
| `shelf_visibility` has a cluster of exact zeros — a product can't realistically have *zero* shelf space | Treated as missing, imputed the same way as weight (median within product category) |
| `product_code` is high-cardinality (1,555 unique values) | Dropped — category, price, and weight already describe what matters about a product |
| `store_code` (only 10 stores, shared identically between train/test) | Dropped from the model features — store format, size, tier, and age already capture what matters about a store, without risking the model just memorizing per-store averages |

## 🤖 Why Three Models?

| Model | Notebook |
|---|---|
| **Linear Regression** | `2_model_linear_regression.ipynb` |
| **Random Forest** | `3_model_random_forest.ipynb` |
| **XGBoost** | `4_model_xgboost.ipynb` |

A linear baseline plus two tree ensembles — one that builds trees independently (Random
Forest) and one that builds them sequentially, each correcting the last (XGBoost) — to see
whether the extra model complexity is actually earning its keep on this data.

## 📊 Results

| Model | Validation RMSE |
|---|---|
| Baseline (predict the mean) | 1716.87 |
| Linear Regression | 1122.64 |
| Random Forest | 1121.97 |
| **XGBoost** | **1081.03** |

*(Exact values are in `5_model_comparison.ipynb`.)*

Linear Regression and Random Forest essentially tie — a reasonable sign that most of the
signal in this data is close to additive, without much complex non-linear interaction for
Random Forest's extra flexibility to exploit over a plain linear model. XGBoost's sequential,
error-correcting boosting picks up a bit more of the pattern than either.

**Final model: XGBoost** — lowest validation RMSE, so its predictions are what
`5_model_comparison.ipynb` copies to `submissions/submission_final.csv`, ready to upload to
the Kaggle leaderboard.

## 🚀 How to Run This Yourself

```bash
git clone https://github.com/<your-username>/dsn-sales-prediction.git
cd dsn-sales-prediction
pip install -r requirements.txt
jupyter notebook notebooks/1_eda.ipynb
```

Run notebooks in numeric order — `1_eda.ipynb` generates the local train/validation split
and the cleaned competition test set every model notebook depends on. Each model notebook
writes its own validation-RMSE result and a submission CSV; `5_model_comparison.ipynb` reads
all three results and copies the best model's submission to `submission_final.csv`.

## 🏆 Submitting to Kaggle

1. Take `submissions/submission_final.csv` (or any individual model's submission file)
2. On the [competition page](https://www.kaggle.com/competitions/dsn-bootcamp-qualification-hackathon-2026-ml-track),
   click **"Submit Predictions"** and upload it
3. Format matches `sample_submission.csv` exactly — `id`, `total_sales` columns, one row per
   `test.csv` row

## 📁 Data Source

Competition data from the **DSN Bootcamp Qualification Hackathon 2026 — ML Track**, hosted
on Kaggle by the DSN Community. Used here under the competition's own terms for a portfolio
write-up of the modeling approach and reasoning.

## 🔭 Possible Next Steps

- Feature-engineer a `price_per_kg` ratio (price ÷ weight) — a natural interaction the raw
  columns don't capture directly
- Try target-encoding `product_category` instead of one-hot, given its moderate cardinality
- Blend the three models' predictions rather than picking a single winner outright

## 📄 License

This project's code is licensed under the MIT License — see [LICENSE](LICENSE) for details.
The competition data itself remains subject to the DSN Community / Kaggle competition's own
terms.

---

*Sixth project in my Data Science / Machine Learning portfolio — see my
[GitHub profile](https://github.com/<your-username>) for the rest.*
