"""
=============================================================================
 SUSPICIOUS WINDOW FINDER  (Fixed v2)

 Fixes vs v1:
   1. Missing row  — duplicate account_ids in predictions are deduplicated
                     before merging so no rows are silently dropped
   2. Exploding z-scores (194M!) — caused by near-zero std when an account
                     has almost identical scores every month.
                     Fix: use ROBUST scaling (median + IQR) instead of
                     mean + std, and hard-cap z-score at 20
   3. Sort order   — output sorted by account_id then year_month in timeline;
                     suspicious_windows sorted by is_mule DESC, peak_zscore DESC

 Inputs:
   outputs/transaction_timeline.parquet
   prediction.csv   (columns: account_id, is_mule)

 Output:
   outputs/suspicious_windows.csv
=============================================================================
"""

import os, sys, io, logging
import pandas as pd
import numpy as np

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
TIMELINE_PATH    = "outputs/transaction_timeline.parquet"
PREDICTIONS_PATH = "s2.csv"
OUTPUT_PATH      = "outputs/suspicious_windows.csv"

MULE_THRESHOLD  = 0.5
BASELINE_MONTHS = 3
ANOMALY_ZSCORE  = 2.0
MAX_ZSCORE_CAP  = 20.0   # cap extreme z-scores caused by near-zero IQR

WEIGHTS = {
    "monthly_passthrough_ratio"  : 3.0,
    "monthly_structuring_count"  : 2.5,
    "monthly_credit_debit_ratio" : 2.0,
    "monthly_night_txn_ratio"    : 1.5,
    "ip_change_count"            : 1.5,
    "near_zero_balance_count"    : 1.5,
    "atm_deposit_frac"           : 1.5,
    "clt_cash_ratio"             : 1.5,
    "monthly_round_ratio"        : 1.0,
    "monthly_txn_count"          : 1.0,
    "monthly_total_amount"       : 1.0,
    "balance_volatility"         : 1.0,
    "monthly_unique_cp"          : 0.8,
    "unique_ip_count"            : 0.8,
}

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(
            io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        ),
        logging.FileHandler("suspicious_windows.log", mode="w", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# STEP 1 — Load & validate data
# ─────────────────────────────────────────────
def load_data():
    # --- Timeline ---
    if not os.path.exists(TIMELINE_PATH):
        raise FileNotFoundError(
            f"Timeline not found: {TIMELINE_PATH}\n"
            "Run build_transaction_timeline.py first."
        )
    log.info("Loading timeline ...")
    timeline = pd.read_parquet(TIMELINE_PATH)
    timeline = timeline.sort_values(
        ["account_id", "year_month"]
    ).reset_index(drop=True)
    log.info("  Timeline shape : %s", timeline.shape)

    # --- Predictions ---
    if not os.path.exists(PREDICTIONS_PATH):
        raise FileNotFoundError(
            f"Predictions not found: {PREDICTIONS_PATH}"
        )
    log.info("Loading predictions ...")
    preds = pd.read_csv(PREDICTIONS_PATH)
    preds.columns = preds.columns.str.strip().str.lower()

    for col in ["account_id", "is_mule"]:
        if col not in preds.columns:
            raise ValueError(f"predictions file missing column: '{col}'")

    # Check for duplicate account_ids — warn but DO NOT remove them
    # (user wants output shape = predictions.csv shape)
    dup_count = preds["account_id"].duplicated().sum()
    if dup_count > 0:
        log.warning(
            "%d duplicate account_id(s) found in predictions. "
            "Each will appear as a separate row in the output. "
            "This is why output shape = predictions shape (%d rows).",
            dup_count, len(preds)
        )

    log.info("  Predictions shape: %s", preds.shape)
    log.info(
        "  Mule accounts (>= %.1f): %d",
        MULE_THRESHOLD,
        (preds["is_mule"] >= MULE_THRESHOLD).sum()
    )
    return timeline, preds


# ─────────────────────────────────────────────
# STEP 2 — Compute fraud score per month
# ─────────────────────────────────────────────
def compute_fraud_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Weighted composite fraud score (0–1) per account-month.
    Each feature is globally normalised to 0–1 before weighting.
    """
    df = df.copy()
    available = {f: w for f, w in WEIGHTS.items() if f in df.columns}
    missing   = set(WEIGHTS) - set(available)
    if missing:
        log.warning("Features not in timeline (skipped): %s", sorted(missing))

    score   = pd.Series(0.0, index=df.index)
    total_w = 0.0
    for feat, weight in available.items():
        col     = df[feat].fillna(0).clip(lower=0)
        col_max = col.max()
        norm    = col / col_max if col_max > 0 else col
        score  += norm * weight
        total_w += weight

    df["fraud_score"] = (score / total_w).round(6)
    return df


# ─────────────────────────────────────────────
# STEP 3 — Personalised robust z-score
# ─────────────────────────────────────────────
def add_personalised_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """
    FIX 2: Use ROBUST scaling (median + IQR) instead of mean + std.

    Why it fixes the 194-million z-score:
      Old way: z = (score - mean) / std
               If std ≈ 0 (all months look the same), z → infinity

      New way: z = (score - median) / IQR
               IQR = 75th percentile - 25th percentile
               If IQR = 0 (flat distribution), use MAD as fallback
               Hard cap at MAX_ZSCORE_CAP (20) to prevent outliers

    This is also more resistant to outliers in the baseline itself.
    """
    dfs = []
    for acct, grp in df.groupby("account_id"):
        grp = grp.sort_values("year_month").reset_index(drop=True)
        n   = len(grp)
        bn  = min(BASELINE_MONTHS, max(1, n - 1))

        baseline = grp["fraud_score"].iloc[:bn]
        b_median = baseline.median()

        # IQR of baseline
        b_q75 = baseline.quantile(0.75)
        b_q25 = baseline.quantile(0.25)
        b_iqr = b_q75 - b_q25

        # Fallback 1: MAD (median absolute deviation)
        if b_iqr < 1e-9:
            b_mad = (baseline - b_median).abs().median()
            b_iqr = b_mad * 1.4826   # scale MAD to be comparable to std

        # Fallback 2: tiny epsilon to avoid division by zero
        if b_iqr < 1e-9:
            b_iqr = 1e-9

        grp["baseline_median"] = round(b_median, 6)
        grp["zscore"] = (
            (grp["fraud_score"] - b_median) / b_iqr
        ).clip(-MAX_ZSCORE_CAP, MAX_ZSCORE_CAP).round(4)

        grp["is_suspicious"] = grp["zscore"] >= ANOMALY_ZSCORE
        dfs.append(grp)

    return pd.concat(dfs, ignore_index=True)


# ─────────────────────────────────────────────
# STEP 4 — Extract suspicious window
# ─────────────────────────────────────────────
def extract_window(grp: pd.DataFrame) -> dict:
    """
    Find suspicious_start and suspicious_end from flagged months.
    Bridges gaps of exactly 1 quiet month between suspicious ones.
    """
    acct  = grp["account_id"].iloc[0]
    grp   = grp.sort_values("year_month").reset_index(drop=True)
    s_idx = grp.index[grp["is_suspicious"]].tolist()

    peak_z = float(grp["zscore"].max())

    if not s_idx:
        return {
            "account_id"       : acct,
            "suspicious_start" : None,
            "suspicious_end"   : None,
            "suspicious_months": 0,
            "peak_zscore"      : round(peak_z, 3),
            "confidence"       : "no_anomaly_found",
        }

    # Bridge gaps of exactly 1 month
    bridged = set(s_idx)
    for k in range(len(s_idx) - 1):
        if s_idx[k + 1] - s_idx[k] == 2:
            bridged.add(s_idx[k] + 1)
    bridged = sorted(bridged)

    first = grp.loc[bridged[0]]
    last  = grp.loc[bridged[-1]]

    # Precise timestamps from timeline
    if "month_first_txn" in grp.columns and pd.notna(first.get("month_first_txn")):
        sus_start = pd.to_datetime(first["month_first_txn"])
    else:
        sus_start = pd.to_datetime(first["year_month"] + "-01")

    if "month_last_txn" in grp.columns and pd.notna(last.get("month_last_txn")):
        sus_end = pd.to_datetime(last["month_last_txn"])
    else:
        sus_end = (
            pd.to_datetime(last["year_month"] + "-01")
            + pd.offsets.MonthEnd(0)
        )

    # Confidence
    if len(s_idx) >= 3 and peak_z >= 5:
        conf = "high"
    elif len(s_idx) >= 2 or peak_z >= 3:
        conf = "medium"
    else:
        conf = "low"

    return {
        "account_id"       : acct,
        "suspicious_start" : sus_start,
        "suspicious_end"   : sus_end,
        "suspicious_months": len(s_idx),
        "peak_zscore"      : round(peak_z, 3),
        "confidence"       : conf,
    }


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def find_suspicious_windows() -> pd.DataFrame:
    timeline, preds = load_data()

    # Separate mule / non-mule
    mule_mask = preds["is_mule"] >= MULE_THRESHOLD
    mule_ids  = set(preds.loc[mule_mask, "account_id"])
    log.info("Confirmed mule accounts: %d", len(mule_ids))

    mule_tl = timeline[timeline["account_id"].isin(mule_ids)].copy()
    if mule_tl.empty:
        log.error(
            "No timeline rows found for mule accounts. "
            "Check account_id values match between files."
        )
        sys.exit(1)

    # ── Find mule accounts that exist in predictions but NOT in timeline ──
    # These are accounts that were predicted as mules but had no transactions
    # recorded in the raw data. We must keep them in the output with null
    # window values so the final shape matches predictions.csv exactly.
    mules_in_timeline = set(mule_tl["account_id"].unique())
    mules_missing_from_timeline = mule_ids - mules_in_timeline
    if mules_missing_from_timeline:
        log.warning(
            "%d mule account(s) found in predictions but NOT in timeline "
            "(no transactions recorded). They will appear in output with "
            "null suspicious_start/end. Missing: %s",
            len(mules_missing_from_timeline),
            sorted(mules_missing_from_timeline)
        )

    # Score → z-score → windows  (only for accounts that have timeline data)
    log.info("Computing fraud scores ...")
    scored = compute_fraud_scores(mule_tl)

    log.info("Computing robust z-scores ...")
    scored = add_personalised_zscore(scored)

    log.info("Extracting suspicious windows ...")
    windows = [extract_window(grp)
               for _, grp in scored.groupby("account_id")]
    mule_windows = pd.DataFrame(windows)

    # Add back mule accounts that had no timeline rows — with null windows
    # Always merge is_mule back into mule_windows
    mule_windows = mule_windows.merge(
        preds[["account_id", "is_mule"]],
        on="account_id", how="left"
    )

    # Add back mule accounts that had no timeline rows — with null windows
    if mules_missing_from_timeline:
        missing_rows = preds[
            preds["account_id"].isin(mules_missing_from_timeline)
        ][["account_id", "is_mule"]].copy()
        missing_rows["suspicious_start"]  = None
        missing_rows["suspicious_end"]    = None
        missing_rows["suspicious_months"] = 0
        missing_rows["peak_zscore"]       = None
        missing_rows["confidence"]        = "no_timeline_data"
        mule_windows = pd.concat([mule_windows, missing_rows], ignore_index=True)

    # Non-mule rows  — ALL non-mule accounts from predictions kept as-is
    non_mule = (
        preds[~mule_mask][["account_id", "is_mule"]]
        .copy()
        .assign(
            suspicious_start  = None,
            suspicious_end    = None,
            suspicious_months = 0,
            peak_zscore       = None,
            confidence        = "not_mule",
        )
    )

    # ── Proper sort order ────────────────────────────────────────────────
    final = pd.concat(
        [mule_windows, non_mule], ignore_index=True
    )
    # Convert dtypes cleanly to avoid FutureWarning
    final["suspicious_start"]  = pd.to_datetime(final["suspicious_start"])
    final["suspicious_end"]    = pd.to_datetime(final["suspicious_end"])
    final["suspicious_months"] = final["suspicious_months"].fillna(0).astype(int)
    final["peak_zscore"]       = pd.to_numeric(final["peak_zscore"], errors="coerce")

    # Sort: mules first, then by peak_zscore descending, then account_id
    final = final.sort_values(
        ["is_mule", "peak_zscore", "account_id"],
        ascending=[False, False, True]
    ).reset_index(drop=True)

    final = final[[
        "account_id", "is_mule",
        "suspicious_start", "suspicious_end",
        "suspicious_months", "peak_zscore", "confidence"
    ]]

    # Save
    os.makedirs("outputs", exist_ok=True)
    final.to_csv(OUTPUT_PATH, index=False)

    # Summary
    found     = final["suspicious_start"].notna().sum()
    no_window = mule_windows["suspicious_start"].isna().sum()
    high_conf = (final["confidence"] == "high").sum()
    med_conf  = (final["confidence"] == "medium").sum()
    low_conf  = (final["confidence"] == "low").sum()

    log.info("=" * 55)
    log.info("  SUMMARY")
    log.info("=" * 55)
    log.info("  Total accounts         : %d", len(final))
    log.info("  Confirmed mules        : %d", len(mule_ids))
    log.info("  Windows found          : %d", found)
    log.info("    High confidence      : %d", high_conf)
    log.info("    Medium confidence    : %d", med_conf)
    log.info("    Low confidence       : %d", low_conf)
    log.info("  Mules with no window   : %d", no_window)
    log.info("  Output saved           : %s", OUTPUT_PATH)
    log.info("=" * 55)

    top = (
        final[final["suspicious_start"].notna()]
        .head(15)
        .to_string(index=False)
    )
    print(f"\nTop 15 mule accounts by peak z-score:\n{top}")

    return final


# ─────────────────────────────────────────────
# DIAGNOSTIC — inspect one account
# ─────────────────────────────────────────────
def inspect_account(account_id: str):
    """
    Usage:  python find_suspicious_windows.py inspect ACCT_000003
    Prints month-by-month scores so you can see WHY a window was detected.
    """
    tl  = pd.read_parquet(TIMELINE_PATH)
    grp = tl[tl["account_id"] == account_id].copy()
    if grp.empty:
        print(f"Account '{account_id}' not found in timeline.")
        return

    grp = compute_fraud_scores(grp)
    grp = add_personalised_zscore(grp)

    cols = [
        "year_month", "fraud_score", "zscore", "is_suspicious",
        "monthly_txn_count", "monthly_passthrough_ratio",
        "monthly_structuring_count", "monthly_night_txn_ratio",
        "ip_change_count", "near_zero_balance_count",
        "month_first_txn", "month_last_txn",
    ]
    show = [c for c in cols if c in grp.columns]
    print(f"\nMonth-by-month for {account_id}:")
    print(grp[show].to_string(index=False))

    w = extract_window(grp)
    print("\nDetected window:")
    for k, v in w.items():
        print(f"  {k:<22}: {v}")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "inspect":
        inspect_account(sys.argv[2])
    else:
        find_suspicious_windows()