# Mule Account Detection — RBIH × IIT Delhi
### National Fraud Prevention Challenge | March 2026

> **Detect money mule accounts from 400M banking transactions using behavioural feature engineering + LightGBM + personalised suspicious window detection.**

---

## Results

| Metric | Score |
|---|---|
| AUC-ROC (Public) | **0.963** |
| F1 Score (Mule class) | **0.776** |
| Temporal IoU | **0.724** |
| Features engineered | **155** |

---

## Problem

Identify mule accounts (used to launder illicit funds) from ~160K banking accounts spanning 5 years of transactions.

- **~400M transactions**, 35 channels (UPI, NEFT, IMPS, ATM, …)
- **2,683 confirmed mules** out of 96,091 labelled accounts (1.67% — severely imbalanced)
- Labels contain noise; dataset includes deliberate red-herring features
- Evaluation: AUC-ROC (primary), F1, Temporal IoU, Red-Herring Avoidance (private)

---

## Repository Structure

```
.
├── 10_transaction_additional.ipynb   # Geo + IP feature extraction from transactions_additional
├── 11_itegrated_transactions.ipynb   # Merging transaction tables
├── 12_feature_extraction.ipynb       # DuckDB: MCC, channel, frequency features (400M rows)
├── 13_amount_time.ipynb              # Amount stats, counterparty features, temporal aggregates
├── 14_transaction_timeline.ipynb     # Monthly timeline table for Temporal IoU
├── 15_transaction.py                 # Transaction processing utilities
├── 16_window.py                      # Suspicious window finder (z-score based)
├── 17_checking.ipynb                 # Data quality checks
├── 18_preprocesing.ipynb             # Account + branch + customer table merges
├── 19_preprocessing.ipynb            # Additional preprocessing passes
├── 20_preprocess.ipynb               # Final preprocessing consolidation
├── 21_pre_encoding.ipynb             # Categorical encoding (Y/N → 0/1, channel risk groups)
├── 22_model.ipynb                    # LightGBM training, evaluation, inference
├── 23_window2.py                     # Window finder v2 (refined gap bridging)
└── README.md
```

**Intermediate artefacts produced (not included — too large):**
`account_features.parquet`, `transaction_features.parquet`, `transaction_final.parquet`, `df_final2.parquet`, `transaction_timeline.parquet`, `prediction.csv`

---

## Environment Setup

```bash
# Python 3.10+
pip install duckdb pandas numpy lightgbm scikit-learn imbalanced-learn pyarrow fastparquet

# Optional (experiments only)
pip install xgboost catboost shap
```

---

## How to Reproduce

### Step 1 — Feature Engineering (requires raw Kaggle data)

Run notebooks **in order**. Each saves an intermediate Parquet file.

```
18_preprocesing.ipynb        →  f_a.parquet, f_c.parquet
12_feature_extraction.ipynb  →  transaction_features.parquet
13_amount_time.ipynb         →  transaction_features_2.parquet, _3.parquet
10_transaction_additional.ipynb → transaction_features_4.parquet
14_transaction_timeline.ipynb   → transaction_timeline.parquet (for Temporal IoU)
19_preprocessing.ipynb / 20_preprocess.ipynb  →  df_final.parquet
21_pre_encoding.ipynb        →  df_final2.parquet   ← model-ready input
```

> **DuckDB is required for notebooks 12–14.** These process the full 400M-row transaction dataset via streaming SQL. Set `memory_limit='10GB'` and `threads=4` minimum.

### Step 2 — Train & Predict

Open and run **`22_model.ipynb`** end-to-end:

1. Loads `df_final2.parquet`
2. Drops correlated/redundant columns (`FINAL_DROP` list)
3. Splits 80/20 stratified on `is_mule`
4. Trains LightGBM with early stopping
5. Evaluates on holdout
6. Merges with `test_accounts.parquet` → generates `prediction.csv`

### Step 3 — Suspicious Window Detection

```bash
# Requires: outputs/transaction_timeline.parquet + prediction.csv
python 16_window.py

# Output: outputs/suspicious_windows.csv
# Columns: account_id, is_mule, suspicious_start, suspicious_end,
#          suspicious_months, peak_zscore, confidence

# Inspect a specific account
python 16_window.py inspect ACCT_000003
```

### Step 4 — Build Submission CSV

```python
import pandas as pd

preds   = pd.read_csv("prediction.csv")          # account_id, is_mule
windows = pd.read_csv("outputs/suspicious_windows.csv")

submission = preds.merge(
    windows[["account_id", "suspicious_start", "suspicious_end"]],
    on="account_id", how="left"
)
submission.to_csv("submission.csv", index=False)
# Columns: account_id, is_mule, suspicious_start, suspicious_end
```

---

## Approach Summary

### Feature Engineering (155 features across 8 groups)

| Group | Count | Key Signals |
|---|---|---|
| Transaction behavioural | 38 | `rapid_passthrough_ratio`, `monthly_structuring_count`, `night_txn_ratio` |
| Balance & liquidity | 22 | `near_zero_balance_count`, `balance_volatility`, `min_balance` |
| Temporal & timeline | 35 | Monthly aggregates, `month_first_txn`, `month_last_txn` |
| IP & device | 12 | `max_unique_ip_day`, `ip_change_count`, `top_ip_share` |
| ATM & channel | 10 | `atm_deposit_frac`, `uses_cdm/crm`, channel risk encoding |
| Geo-spatial | 8 | `geo_bbox_km`, `unique_geo_count` (Haversine distance) |
| Counterparty network | 16 | Unique counterparties, fan-in/fan-out ratios |
| Account lifecycle | 14 | `active_tenure_days`, `days_since_account_open`, freeze flag |

### Missing Value Strategy

Three-type classification — each treated differently to **preserve fraud signal in missingness**:

- **Type 1** (feature not applicable, e.g. `unfreeze_date`): Fill `0` + binary flag
- **Type 2** (random technical gap, <0.01% rows): Median within `product_family` group
- **Type 3** (unknown state, e.g. `rapid_passthrough_count`): Fill `0` + `_not_computed` indicator

### Model: LightGBM

```python
LGBMClassifier(
    n_estimators=1000, learning_rate=0.05, num_leaves=63,
    min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=0.1,
    class_weight="balanced",   # handles 1.67% mule rate without SMOTE
    random_state=42
)
# Early stopping: patience=50, monitor val AUC
```

**Why not SMOTE?** SMOTETomek dropped AUC from 0.963 → 0.948. Synthetic samples fell in feature overlap regions between high-volume legitimate accounts and mules. `class_weight='balanced'` achieves equivalent recall without noise.

### Suspicious Window Detection (`16_window.py`)

For each confirmed mule account:

1. Compute monthly **composite fraud score** (14 features, weighted: passthrough=3.0, structuring=2.5, …)
2. Use **first 3 months as personalised baseline** (μ, σ per account)
3. Flag months where `z-score = (score − μ) / σ ≥ 2.0`
4. **Bridge single-month gaps** (1 quiet month between suspicious months → included)
5. Extract `suspicious_start` / `suspicious_end` timestamps
6. Assign confidence: High (≥3 months + peak z ≥ 5), Medium, Low

> Personalised baselines prevent high-volume legitimate businesses from being falsely flagged — each account is compared to its own history.

---

## Red Herring Avoidance

Key design decisions to survive the private leaderboard:

| Trap | Mitigation |
|---|---|
| Branch-level metrics (turnover, employee count) | Dropped entirely — not behavioural |
| Festival transaction spikes | Monthly features normalised against personal baseline |
| Freeze/unfreeze timing | Used only as binary flag; duration/timing excluded |
| Correlated duplicate features | 20+ columns removed (`FINAL_DROP` list in `22_model.ipynb`) |
| High-volume legitimate businesses | Personalised z-score baseline per account |
| Post-activity tails | Gap bridging capped at 1 month |

---

## Key Findings

- **Pass-through ratio** is the strongest single discriminator (92% in mules vs. 18% in legitimate accounts)
- **85% of mules** show a dormant-then-activate pattern — months of inactivity followed by sudden burst
- Mule accounts use **4.2× more unique IPs per day** and have **2.8× larger geographic spread**
- Night-time transactions are **3× more frequent** in mule accounts
- Near-zero balance events occur **7× more often** post-credit in mule accounts

---

## Submission Format

```
account_id, is_mule, suspicious_start, suspicious_end
ACCT_000001, 0.023, , 
ACCT_000003, 0.981, 2022-03-15, 2023-01-28
...
```

- `is_mule`: float probability 0.0–1.0 (LightGBM `predict_proba`)
- Windows provided only for accounts predicted as mules
- Optimal classification threshold: **0.42**

---

## What Didn't Work

| Experiment | Outcome |
|---|---|
| SMOTETomek | AUC −0.015; noise in overlap zones |
| CatBoost | AUC 0.937; slower; excluded |
| Branch features included | Train AUC inflated; private phase risk |
| Global z-score threshold | IoU ~0.48; personalised approach scores 0.724 |
| Negative `days_since_open` as numeric | Introduced noise; converted to flag |

---

*National Fraud Prevention Challenge · RBIH × IIT Delhi · March 2026*
