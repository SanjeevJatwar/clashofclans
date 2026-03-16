"""
=============================================================================
 SUSPICIOUS WINDOW FINDER
 Run this AFTER you have:
   1. outputs/transaction_timeline.parquet  (from build_transaction_timeline.py)
   2. predictions.csv                       (your mule predictions)

 predictions.csv must have at minimum:
   account_id   → e.g. ACCT_000003
   is_mule      → probability 0.0–1.0  OR  binary 0/1

 Output:
   outputs/suspicious_windows.csv
   columns: account_id, is_mule, suspicious_start, suspicious_end,
            suspicious_months, peak_zscore, confidence
=============================================================================
"""

import os, sys, io, logging
import pandas as pd
import numpy as np

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
TIMELINE_PATH    = "outputs/transaction_timeline.parquet"
PREDICTIONS_PATH = "prediction.csv"
OUTPUT_PATH      = "outputs/suspicious_windows.csv"

MULE_THRESHOLD   = 0.5   # is_mule >= this → confirmed mule
BASELINE_MONTHS  = 3     # first N months = "normal" baseline per account
ANOMALY_ZSCORE   = 2.0   # months with z-score above this = suspicious

# Fraud signal weights
# Higher weight = more important for detecting suspicious period
WEIGHTS = {
    "monthly_passthrough_ratio"  : 3.0,  # money in = money out → mule
    "monthly_structuring_count"  : 2.5,  # deposits just below 2L threshold
    "monthly_credit_debit_ratio" : 2.0,  # sudden credit surge
    "monthly_night_txn_ratio"    : 1.5,  # late-night automated activity
    "ip_change_count"            : 1.5,  # IP hopping = account sharing
    "near_zero_balance_count"    : 1.5,  # cash-out after credits
    "atm_deposit_frac"           : 1.5,  # cash stuffing via ATM
    "clt_cash_ratio"             : 1.5,  # CLT cash layering
    "monthly_round_ratio"        : 1.0,  # round-amount layering
    "monthly_txn_count"          : 1.0,  # volume spike
    "monthly_total_amount"       : 1.0,  # amount spike
    "balance_volatility"         : 1.0,  # erratic balance
    "monthly_unique_cp"          : 0.8,  # many counterparties
    "unique_ip_count"            : 0.8,  # many devices
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
# STEP 1 — Load data
# ─────────────────────────────────────────────
def load_data():
    # Timeline
    if not os.path.exists(TIMELINE_PATH):
        raise FileNotFoundError(
            f"Timeline not found: {TIMELINE_PATH}\n"
            f"Run build_transaction_timeline.py first."
        )
    log.info("Loading timeline from %s ...", TIMELINE_PATH)
    timeline = pd.read_parquet(TIMELINE_PATH)
    timeline = timeline.sort_values(["account_id", "year_month"]).reset_index(drop=True)
    log.info("  Timeline shape: %s", timeline.shape)

    # Predictions
    if not os.path.exists(PREDICTIONS_PATH):
        raise FileNotFoundError(
            f"Predictions not found: {PREDICTIONS_PATH}\n"
            f"Add your mule predictions file and re-run."
        )
    log.info("Loading predictions from %s ...", PREDICTIONS_PATH)
    preds = pd.read_csv(PREDICTIONS_PATH)
    preds.columns = preds.columns.str.strip().str.lower()

    # Validate required columns
    for col in ["account_id", "is_mule"]:
        if col not in preds.columns:
            raise ValueError(
                f"predictions.csv missing column: '{col}'\n"
                f"Required columns: account_id, is_mule"
            )

    log.info("  Predictions shape: %s", preds.shape)
    log.info("  Mule accounts (>= %.1f): %d",
             MULE_THRESHOLD,
             (preds["is_mule"] >= MULE_THRESHOLD).sum())

    return timeline, preds


# ─────────────────────────────────────────────
# STEP 2 — Score each month per account
# ─────────────────────────────────────────────
def compute_fraud_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a weighted composite fraud score for each account-month row.
    Each feature is normalised 0-1 (global max), then weighted and averaged.
    Result: df["fraud_score"]  in range 0–1
    """
    df = df.copy()
    available = {f: w for f, w in WEIGHTS.items() if f in df.columns}
    missing   = set(WEIGHTS) - set(available)
    if missing:
        log.warning("Features not in timeline (skipped): %s", missing)

    score    = pd.Series(0.0, index=df.index)
    total_w  = 0.0

    for feat, weight in available.items():
        col     = df[feat].fillna(0).clip(lower=0)
        col_max = col.max()
        norm    = col / col_max if col_max > 0 else col
        score  += norm * weight
        total_w += weight

    df["fraud_score"] = (score / total_w).round(6)
    return df


# ─────────────────────────────────────────────
# STEP 3 — Personalised z-score per account
# ─────────────────────────────────────────────
def add_personalised_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each account compare each month's fraud_score against
    that account's own FIRST N months as the 'normal' baseline.

    This is personalised — a high-volume legitimate business account
    won't be falsely flagged just because it does many transactions.
    """
    dfs = []
    for acct, grp in df.groupby("account_id"):
        grp  = grp.sort_values("year_month").reset_index(drop=True)
        n    = len(grp)
        bn   = min(BASELINE_MONTHS, max(1, n - 1))

        bmu  = grp["fraud_score"].iloc[:bn].mean()
        bsd  = grp["fraud_score"].iloc[:bn].std()
        if pd.isna(bsd) or bsd < 1e-9:
            bsd = 1e-9

        grp["baseline_mean"] = round(bmu, 6)
        grp["zscore"]        = ((grp["fraud_score"] - bmu) / bsd).round(4)
        grp["is_suspicious"] = grp["zscore"] >= ANOMALY_ZSCORE
        dfs.append(grp)

    return pd.concat(dfs, ignore_index=True)


# ─────────────────────────────────────────────
# STEP 4 — Extract window per account
# ─────────────────────────────────────────────
def extract_window(grp: pd.DataFrame) -> dict:
    """
    From a single account's monthly rows (sorted, with is_suspicious flag),
    return the suspicious start and end timestamps.

    Gap bridging: if there is exactly 1 normal month sandwiched between
    suspicious months, bridge the gap (avoids splitting one fraud event
    into two windows due to a quiet month).
    """
    acct   = grp["account_id"].iloc[0]
    grp    = grp.sort_values("year_month").reset_index(drop=True)
    s_idx  = grp.index[grp["is_suspicious"]].tolist()

    if not s_idx:
        return {
            "account_id"       : acct,
            "suspicious_start" : None,
            "suspicious_end"   : None,
            "suspicious_months": 0,
            "peak_zscore"      : round(grp["zscore"].max(), 3),
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

    # Use exact timestamps from timeline if available
    if "month_first_txn" in grp.columns and pd.notna(first.get("month_first_txn")):
        sus_start = pd.to_datetime(first["month_first_txn"])
    else:
        sus_start = pd.to_datetime(first["year_month"] + "-01")

    if "month_last_txn" in grp.columns and pd.notna(last.get("month_last_txn")):
        sus_end = pd.to_datetime(last["month_last_txn"])
    else:
        sus_end = (pd.to_datetime(last["year_month"] + "-01")
                   + pd.offsets.MonthEnd(0))

    # Confidence based on number of suspicious months and peak z-score
    peak_z = grp["zscore"].max()
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
    # Load
    timeline, preds = load_data()

    # Split mule / non-mule
    mule_ids  = set(preds.loc[preds["is_mule"] >= MULE_THRESHOLD, "account_id"])
    log.info("Processing %d confirmed mule accounts ...", len(mule_ids))

    mule_tl = timeline[timeline["account_id"].isin(mule_ids)].copy()

    if mule_tl.empty:
        log.error(
            "No timeline rows found for mule accounts.\n"
            "Check that account_id values match between timeline and predictions."
        )
        sys.exit(1)

    # Score + z-score
    log.info("Computing fraud scores ...")
    scored = compute_fraud_scores(mule_tl)

    log.info("Computing personalised z-scores ...")
    scored = add_personalised_zscore(scored)

    # Extract window per account
    log.info("Extracting suspicious windows ...")
    windows = []
    for acct, grp in scored.groupby("account_id"):
        windows.append(extract_window(grp))
    mule_windows = pd.DataFrame(windows)

    # Merge is_mule probability back
    mule_windows = mule_windows.merge(
        preds[["account_id", "is_mule"]], on="account_id", how="left"
    )

    # Non-mule accounts — no window
    non_mule = preds[preds["is_mule"] < MULE_THRESHOLD][["account_id","is_mule"]].copy()
    non_mule["suspicious_start"]  = None
    non_mule["suspicious_end"]    = None
    non_mule["suspicious_months"] = 0
    non_mule["peak_zscore"]       = None
    non_mule["confidence"]        = "not_mule"

    # Combine and sort
    final = pd.concat([mule_windows, non_mule], ignore_index=True)
    final = final.sort_values(
        ["is_mule", "peak_zscore"],
        ascending=[False, False]
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
    not_found = (final["is_mule"] >= MULE_THRESHOLD) & final["suspicious_start"].isna()
    high_conf = (final["confidence"] == "high").sum()
    med_conf  = (final["confidence"] == "medium").sum()

    log.info("=" * 55)
    log.info("  SUMMARY")
    log.info("=" * 55)
    log.info("  Total accounts         : %d", len(final))
    log.info("  Confirmed mules        : %d", len(mule_ids))
    log.info("  Windows found          : %d", found)
    log.info("    High confidence      : %d", high_conf)
    log.info("    Medium confidence    : %d", med_conf)
    log.info("  Mules with no window   : %d", not_found.sum())
    log.info("  Output saved           : %s", OUTPUT_PATH)
    log.info("=" * 55)

    print("\nTop 15 mule accounts by peak z-score:")
    print(
        final[final["suspicious_start"].notna()]
        .head(15)
        .to_string(index=False)
    )

    return final


# ─────────────────────────────────────────────
# DIAGNOSTIC — inspect one account month by month
# ─────────────────────────────────────────────
def inspect_account(account_id: str):
    """
    Shows month-by-month breakdown for one account.
    Useful for understanding WHY a window was detected.

    Usage:
        python find_suspicious_windows.py inspect ACCT_000003
    """
    timeline = pd.read_parquet(TIMELINE_PATH)
    grp = timeline[timeline["account_id"] == account_id].copy()

    if grp.empty:
        print(f"Account {account_id} not found in timeline.")
        return

    grp = compute_fraud_scores(grp)
    grp = add_personalised_zscore(grp)

    cols = [
        "year_month", "fraud_score", "zscore", "is_suspicious",
        "monthly_txn_count", "monthly_passthrough_ratio",
        "monthly_structuring_count", "monthly_night_txn_ratio",
        "ip_change_count", "near_zero_balance_count",
        "month_first_txn", "month_last_txn"
    ]
    show = [c for c in cols if c in grp.columns]

    print(f"\nMonth-by-month breakdown for {account_id}:")
    print(grp[show].to_string(index=False))

    w = extract_window(grp)
    print(f"\nDetected window:")
    for k, v in w.items():
        print(f"  {k:<22}: {v}")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "inspect":
        inspect_account(sys.argv[2])
    else:
        find_suspicious_windows()