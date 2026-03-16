"""
=============================================================================
 FRAUD DETECTION — TRANSACTION TIMELINE BUILDER  (Hash-Partition v6)

 STRATEGY: Hash Partition → Process Buckets → Merge
 ─────────────────────────────────────────────────────
 Phase 1  Read 706 raw files ONCE
          Write partitioned by hash(account_id) % N_BUCKETS
          → duckdb_tmp/buckets/bucket_0.parquet
          → duckdb_tmp/buckets/bucket_1.parquet  ...bucket_N
          Each bucket = all txns for ~1/N of accounts
          No account ever split across buckets (LAG is safe)

 Phase 2  For each bucket file (small, ~400 MB each):
          Run full feature SQL (JOIN + LAG + GROUP BY) in seconds
          Append result rows to transaction_timeline.parquet

 WHY THIS IS FASTEST
 ─────────────────────────────────────────────────────
 v4 batching  : re-reads 706 files per batch  (93 × 706 file opens)
 v5 two-phase : IN (account list) filter per batch  (slow predicate)
 v6 this file : each bucket file is pre-partitioned, no filter needed
                DuckDB reads one small file and processes it entirely

 ESTIMATED RUNTIME ON HP VICTUS 16 GB
 ─────────────────────────────────────────────────────
 Phase 1  ~25–35 min  (one full scan, done once, skipped on re-run)
 Phase 2  ~20–30 min  (50 buckets × 30 sec each)
 TOTAL    ~50–65 min

 PEAK RAM ~3–4 GB per bucket  (safe on 16 GB)
=============================================================================
"""

import os, sys, io, time, logging, glob, math
import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
TRANSACTIONS_GLOB     = "transactions/*/*.parquet"
TRANS_ADDITIONAL_GLOB = "transactions_additional/*/*.parquet"

OUTPUT_DIR   = "outputs"
OUTPUT_FILE  = "transaction_timeline.parquet"

BUCKET_DIR   = "duckdb_tmp/buckets"
N_BUCKETS    = 50        # 40 crore / 50 = ~80 lakh rows per bucket
                         # increase to 100 if still OOM in Phase 2

DUCKDB_MEMORY_LIMIT = "8GB"
DUCKDB_THREADS      = 2
TEMP_DIR            = "duckdb_tmp"

SKIP_PHASE1_IF_EXISTS = True   # set False to force rebuild of buckets

# ─────────────────────────────────────────────
# LOGGING  (UTF-8 — no Windows cp1252 crash)
# ─────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR,   exist_ok=True)
os.makedirs(BUCKET_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(
            io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        ),
        logging.FileHandler("timeline_build.log", mode="w", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def elapsed(s):
    d = time.time() - s
    return f"{d/60:.1f} min" if d > 90 else f"{d:.1f}s"

def fwd(p):
    """Forward-slash path — DuckDB needs this on Windows."""
    return p.replace("\\", "/")

def validate_paths(*globs):
    for pattern in globs:
        files = glob.glob(pattern, recursive=True)
        if not files:
            raise FileNotFoundError(
                f"No parquet files for: '{pattern}'  cwd={os.getcwd()}"
            )
        log.info("  Found %d files  '%s'", len(files), pattern)

def fresh_con():
    con = duckdb.connect(":memory:")
    con.execute(f"SET memory_limit='{DUCKDB_MEMORY_LIMIT}';")
    con.execute(f"SET threads={DUCKDB_THREADS};")
    con.execute(f"SET temp_directory='{TEMP_DIR}';")
    con.execute("SET preserve_insertion_order=false;")
    return con

def bucket_path(i):
    return fwd(os.path.join(BUCKET_DIR, f"bucket_{i:04d}.parquet"))

def all_buckets_exist():
    return all(os.path.exists(
        os.path.join(BUCKET_DIR, f"bucket_{i:04d}.parquet")
    ) for i in range(N_BUCKETS))


# ═══════════════════════════════════════════════════════════
# PHASE 1 — Read all raw files ONCE, write N bucket files
#
#  hash(account_id) % N_BUCKETS  guarantees every transaction
#  for the same account lands in the same bucket file.
#  LAG() within a bucket is therefore always correct.
# ═══════════════════════════════════════════════════════════
def phase1_partition():
    if SKIP_PHASE1_IF_EXISTS and all_buckets_exist():
        sizes = sum(
            os.path.getsize(os.path.join(BUCKET_DIR, f"bucket_{i:04d}.parquet"))
            for i in range(N_BUCKETS)
        ) / 1024**3
        log.info("Phase 1 SKIPPED — %d bucket files already exist (%.1f GB total)",
                 N_BUCKETS, sizes)
        return

    log.info("=" * 65)
    log.info("PHASE 1 — Partitioning raw data into %d bucket files", N_BUCKETS)
    log.info("Reads all 706 files ONCE.  Expected: ~25-35 min")
    log.info("=" * 65)
    t0  = time.time()
    con = fresh_con()

    txn_fwd = fwd(TRANSACTIONS_GLOB)
    add_fwd = fwd(TRANS_ADDITIONAL_GLOB)

    # Write one bucket at a time to keep memory low
    # DuckDB COPY with PARTITION BY would be ideal but requires
    # hive partition dirs; instead we write each bucket via WHERE filter
    # on the hash.  All 706 files are opened once per bucket but
    # DuckDB's metadata cache makes subsequent bucket scans fast.

    # Better: write all at once using a single COPY + partition expression
    # DuckDB supports COPY ... PARTITION BY (col) since v0.10
    # We add a bucket_id column and let DuckDB split the files.

    try:
        # Try modern PARTITION BY COPY (DuckDB >= 0.10)
        con.execute(f"""
            COPY (
                SELECT
                    -- bucket column (will NOT be in the feature output)
                    (hash(t.account_id) & 2147483647) % {N_BUCKETS} AS bucket_id,

                    t.transaction_id,
                    t.account_id,
                    TRY_CAST(t.transaction_timestamp AS TIMESTAMP)          AS ts,
                    STRFTIME(
                        TRY_CAST(t.transaction_timestamp AS TIMESTAMP),
                        '%Y-%m')                                             AS year_month,
                    DATE_TRUNC('day',
                        TRY_CAST(t.transaction_timestamp AS TIMESTAMP))     AS txn_date,
                    UPPER(TRIM(t.channel))                                  AS channel,
                    COALESCE(TRY_CAST(t.amount AS DOUBLE), 0.0)             AS amount,
                    UPPER(TRIM(t.txn_type))                                 AS txn_type,
                    t.counterparty_id,
                    a.ip_address,
                    TRY_CAST(a.balance_after_transaction AS DOUBLE)         AS balance_after,
                    UPPER(TRIM(a.part_transaction_type))                    AS part_txn_type,
                    COALESCE(UPPER(TRIM(a.atm_deposit_channel_code)), '')   AS atm_ch,
                    UPPER(TRIM(a.transaction_sub_type))                     AS sub_type,

                    -- geo cell
                    CASE WHEN TRY_CAST(a.latitude  AS DOUBLE) IS NOT NULL
                          AND TRY_CAST(a.longitude AS DOUBLE) IS NOT NULL
                         THEN CAST(FLOOR(TRY_CAST(a.latitude  AS DOUBLE)) AS VARCHAR)
                              || '_'
                              || CAST(FLOOR(TRY_CAST(a.longitude AS DOUBLE)) AS VARCHAR)
                         ELSE NULL END                                      AS geo_cell,

                    -- boolean flags (stored as BOOLEAN = 1 bit each)
                    (UPPER(TRIM(t.txn_type)) = 'C')                        AS is_credit,
                    (UPPER(TRIM(t.txn_type)) = 'D')                        AS is_debit,
                    (EXTRACT(hour FROM
                        TRY_CAST(t.transaction_timestamp AS TIMESTAMP)) >= 22
                     OR EXTRACT(hour FROM
                        TRY_CAST(t.transaction_timestamp AS TIMESTAMP)) < 6) AS is_night,
                    (UPPER(TRIM(t.channel)) IN ('UPI','UPC','UPD'))        AS is_upi,
                    (UPPER(TRIM(t.channel)) IN ('ATM','CDM','CRM','CASH')) AS is_cash,
                    (COALESCE(UPPER(TRIM(a.atm_deposit_channel_code)),'')
                        IN ('CDM','CRM'))                                   AS is_atm_deposit,
                    (UPPER(TRIM(a.transaction_sub_type)) = 'CLT_CASH')     AS is_clt_cash,
                    (UPPER(TRIM(a.part_transaction_type)) = 'CI')          AS is_ci,
                    (COALESCE(TRY_CAST(t.amount AS DOUBLE), 0) % 100 = 0) AS is_round,
                    (UPPER(TRIM(t.txn_type)) = 'C'
                     AND COALESCE(TRY_CAST(t.amount AS DOUBLE), 0)
                         BETWEEN 180000 AND 200000)                        AS is_structuring

                FROM read_parquet('{txn_fwd}',
                                  hive_partitioning=true, union_by_name=true) t
                LEFT JOIN read_parquet('{add_fwd}',
                                       hive_partitioning=true, union_by_name=true) a
                       ON a.transaction_id = t.transaction_id
                WHERE TRY_CAST(t.transaction_timestamp AS TIMESTAMP) IS NOT NULL
                  AND t.account_id    IS NOT NULL
                  AND t.transaction_id IS NOT NULL
            )
            TO '{fwd(BUCKET_DIR)}'
            (FORMAT PARQUET, COMPRESSION SNAPPY,
             PARTITION_BY (bucket_id),
             OVERWRITE_OR_IGNORE true,
             ROW_GROUP_SIZE 200000)
        """)
        # DuckDB writes bucket_id=0/data_0.parquet etc — rename to our format
        _rename_hive_to_flat(BUCKET_DIR)

    except Exception as e:
        log.warning("PARTITION BY COPY failed (%s) — falling back to loop", e)
        con.close()
        _phase1_loop_fallback(txn_fwd, add_fwd)
        return

    con.close()
    sizes = sum(
        os.path.getsize(os.path.join(BUCKET_DIR, f"bucket_{i:04d}.parquet"))
        for i in range(N_BUCKETS) if os.path.exists(
            os.path.join(BUCKET_DIR, f"bucket_{i:04d}.parquet"))
    ) / 1024**3
    log.info("Phase 1 done in %s — %d buckets, %.1f GB total",
             elapsed(t0), N_BUCKETS, sizes)


def _rename_hive_to_flat(bucket_dir):
    """
    DuckDB PARTITION_BY writes hive-style dirs:
      bucket_id=0/data_0.parquet
    Rename to our flat naming:
      bucket_0000.parquet
    """
    import re, shutil
    renamed = 0
    for entry in os.scandir(bucket_dir):
        if entry.is_dir():
            m = re.match(r"bucket_id=(\d+)", entry.name)
            if m:
                bucket_num = int(m.group(1))
                # find the data file inside
                for sub in os.scandir(entry.path):
                    if sub.name.endswith(".parquet"):
                        dest = os.path.join(bucket_dir,
                                            f"bucket_{bucket_num:04d}.parquet")
                        shutil.move(sub.path, dest)
                        renamed += 1
                        break
                # remove empty dir
                try: os.rmdir(entry.path)
                except Exception: pass
    log.info("  Renamed %d hive partition dirs to flat bucket files", renamed)


def _phase1_loop_fallback(txn_fwd, add_fwd):
    """
    Fallback for older DuckDB versions: write each bucket with a WHERE filter.
    Slower (N passes over the data) but always works.
    """
    log.info("Using loop fallback — writing %d buckets individually", N_BUCKETS)
    t0 = time.time()
    for i in range(N_BUCKETS):
        bp = bucket_path(i)
        if SKIP_PHASE1_IF_EXISTS and os.path.exists(bp):
            continue
        con = fresh_con()
        con.execute(f"""
            COPY (
                SELECT
                    t.transaction_id, t.account_id,
                    TRY_CAST(t.transaction_timestamp AS TIMESTAMP)          AS ts,
                    STRFTIME(TRY_CAST(t.transaction_timestamp AS TIMESTAMP), '%Y-%m') AS year_month,
                    DATE_TRUNC('day', TRY_CAST(t.transaction_timestamp AS TIMESTAMP)) AS txn_date,
                    UPPER(TRIM(t.channel)) AS channel,
                    COALESCE(TRY_CAST(t.amount AS DOUBLE), 0.0) AS amount,
                    UPPER(TRIM(t.txn_type)) AS txn_type,
                    t.counterparty_id,
                    a.ip_address,
                    TRY_CAST(a.balance_after_transaction AS DOUBLE) AS balance_after,
                    UPPER(TRIM(a.part_transaction_type)) AS part_txn_type,
                    COALESCE(UPPER(TRIM(a.atm_deposit_channel_code)),'') AS atm_ch,
                    UPPER(TRIM(a.transaction_sub_type)) AS sub_type,
                    CASE WHEN TRY_CAST(a.latitude AS DOUBLE) IS NOT NULL
                          AND TRY_CAST(a.longitude AS DOUBLE) IS NOT NULL
                         THEN CAST(FLOOR(TRY_CAST(a.latitude AS DOUBLE)) AS VARCHAR)
                              || '_' || CAST(FLOOR(TRY_CAST(a.longitude AS DOUBLE)) AS VARCHAR)
                         ELSE NULL END AS geo_cell,
                    (UPPER(TRIM(t.txn_type))='C') AS is_credit,
                    (UPPER(TRIM(t.txn_type))='D') AS is_debit,
                    (EXTRACT(hour FROM TRY_CAST(t.transaction_timestamp AS TIMESTAMP))>=22
                     OR EXTRACT(hour FROM TRY_CAST(t.transaction_timestamp AS TIMESTAMP))<6) AS is_night,
                    (UPPER(TRIM(t.channel)) IN ('UPI','UPC','UPD')) AS is_upi,
                    (UPPER(TRIM(t.channel)) IN ('ATM','CDM','CRM','CASH')) AS is_cash,
                    (COALESCE(UPPER(TRIM(a.atm_deposit_channel_code)),'') IN ('CDM','CRM')) AS is_atm_deposit,
                    (UPPER(TRIM(a.transaction_sub_type))='CLT_CASH') AS is_clt_cash,
                    (UPPER(TRIM(a.part_transaction_type))='CI') AS is_ci,
                    (COALESCE(TRY_CAST(t.amount AS DOUBLE),0) % 100 = 0) AS is_round,
                    (UPPER(TRIM(t.txn_type))='C'
                     AND COALESCE(TRY_CAST(t.amount AS DOUBLE),0) BETWEEN 180000 AND 200000) AS is_structuring
                FROM read_parquet('{txn_fwd}', hive_partitioning=true, union_by_name=true) t
                LEFT JOIN read_parquet('{add_fwd}', hive_partitioning=true, union_by_name=true) a
                       ON a.transaction_id = t.transaction_id
                WHERE (hash(t.account_id) & 2147483647) % {N_BUCKETS} = {i}
                  AND TRY_CAST(t.transaction_timestamp AS TIMESTAMP) IS NOT NULL
                  AND t.account_id IS NOT NULL
                  AND t.transaction_id IS NOT NULL
            ) TO '{bp}' (FORMAT PARQUET, COMPRESSION SNAPPY, ROW_GROUP_SIZE 200000)
        """)
        con.close()
        log.info("  Bucket %d/%d done  %s", i+1, N_BUCKETS, elapsed(t0))


# ═══════════════════════════════════════════════════════════
# PHASE 2 — Process each bucket → compute all 50 features
# ═══════════════════════════════════════════════════════════
FEATURE_SQL = """
WITH
-- Window functions in their own CTE (DuckDB rule)
bal_win AS (
    SELECT account_id, year_month, balance_after,
           LAG(balance_after) OVER (PARTITION BY account_id ORDER BY ts) AS prev_bal,
           CASE txn_type WHEN 'C' THEN amount ELSE -amount END           AS signed_amt
    FROM read_parquet('{BUCKET}')
),
ip_win AS (
    SELECT account_id, year_month, ip_address,
           LAG(ip_address) OVER (PARTITION BY account_id ORDER BY ts)   AS prev_ip
    FROM read_parquet('{BUCKET}')
),

-- Main monthly aggregation
base AS (
    SELECT
        account_id, year_month,
        COUNT(*)                                                    AS monthly_txn_count,
        SUM(ABS(amount))                                            AS monthly_total_amount,
        AVG(ABS(amount))                                            AS monthly_avg_amount,
        MAX(ABS(amount))                                            AS monthly_max_amount,
        SUM(ABS(amount)) FILTER (WHERE is_credit)                   AS monthly_credit,
        SUM(ABS(amount)) FILTER (WHERE is_debit)                    AS monthly_debit,
        COUNT(*) FILTER (WHERE is_credit)                           AS monthly_credit_count,
        COUNT(*) FILTER (WHERE is_debit)                            AS monthly_debit_count,
        MAX(ABS(amount)) FILTER (WHERE is_credit)                   AS monthly_max_credit_amount,
        SUM(CASE WHEN is_credit THEN ABS(amount)
                 ELSE -ABS(amount) END)                             AS monthly_net_flow,
        COUNT(DISTINCT counterparty_id)                             AS monthly_unique_cp,
        COUNT(DISTINCT counterparty_id) FILTER (WHERE is_credit)    AS monthly_unique_credit_cp,
        COUNT(DISTINCT counterparty_id) FILTER (WHERE is_debit)     AS monthly_unique_debit_cp,
        COUNT(*) FILTER (WHERE is_structuring)                      AS monthly_structuring_count,
        COUNT(*) FILTER (WHERE is_round)                            AS monthly_round_count,
        COUNT(*) FILTER (WHERE is_night)                            AS monthly_night_txn_count,
        COUNT(DISTINCT txn_date)                                    AS monthly_active_days,
        COUNT(*) FILTER (WHERE is_upi)                              AS monthly_upi_count,
        COUNT(*) FILTER (WHERE is_cash)                             AS monthly_cash_count,
        COUNT(DISTINCT channel)                                     AS monthly_unique_channels,
        COUNT(DISTINCT geo_cell)                                    AS unique_geo_count,
        COUNT(*) FILTER (WHERE geo_cell IS NOT NULL)                AS geo_tagged_count,
        COUNT(*) FILTER (WHERE is_atm_deposit)                      AS atm_deposit_count,
        COUNT(*) FILTER (WHERE is_ci)                               AS ci_count,
        COUNT(*) FILTER (WHERE is_clt_cash)                         AS clt_cash_count,
        MIN(ts)                                                     AS month_first_txn,
        MAX(ts)                                                     AS month_last_txn
    FROM read_parquet('{BUCKET}')
    GROUP BY account_id, year_month
),

-- Daily max (needs two-level GROUP BY — separate CTE)
daily_max AS (
    SELECT account_id, year_month,
           MAX(dc) AS monthly_max_daily_txn_count,
           MAX(da) AS monthly_max_daily_amount
    FROM (
        SELECT account_id, year_month, txn_date,
               COUNT(*)         AS dc,
               SUM(ABS(amount)) AS da
        FROM read_parquet('{BUCKET}')
        GROUP BY account_id, year_month, txn_date
    ) d
    GROUP BY account_id, year_month
),

-- Balance aggregation (from pre-windowed CTE)
bal_agg AS (
    SELECT account_id, year_month,
           AVG(balance_after)                                       AS avg_balance,
           MIN(balance_after)                                       AS min_balance,
           MAX(balance_after)                                       AS max_balance,
           STDDEV_POP(balance_after)                                AS balance_volatility,
           COUNT(*) FILTER (WHERE balance_after IS NOT NULL
                             AND balance_after < 500)               AS near_zero_balance_count,
           COUNT(*) FILTER (WHERE balance_after IS NOT NULL
                             AND prev_bal IS NOT NULL
                             AND ABS(balance_after - prev_bal - signed_amt) > 1.0)
                                                                    AS balance_mismatch_count
    FROM bal_win
    GROUP BY account_id, year_month
),

-- IP aggregation (from pre-windowed CTE)
ip_agg AS (
    SELECT account_id, year_month,
           COUNT(DISTINCT ip_address) FILTER (WHERE ip_address IS NOT NULL) AS unique_ip_count,
           SUM(CASE WHEN ip_address IS NOT NULL
                     AND prev_ip IS NOT NULL
                     AND ip_address <> prev_ip THEN 1 ELSE 0 END)           AS ip_change_count
    FROM ip_win
    GROUP BY account_id, year_month
)

-- Final output with all ratio columns
SELECT
    b.account_id,
    b.year_month,
    b.monthly_txn_count,
    b.monthly_total_amount,
    b.monthly_avg_amount,
    b.monthly_max_amount,
    COALESCE(b.monthly_credit, 0)                                   AS monthly_credit,
    COALESCE(b.monthly_debit,  0)                                   AS monthly_debit,
    b.monthly_credit_count,
    b.monthly_debit_count,
    CASE WHEN b.monthly_debit_count > 0
         THEN ROUND(b.monthly_credit_count * 1.0 / b.monthly_debit_count, 4)
         ELSE NULL END                                              AS monthly_credit_debit_ratio,
    COALESCE(b.monthly_max_credit_amount, 0)                        AS monthly_max_credit_amount,
    b.monthly_net_flow,
    CASE WHEN GREATEST(COALESCE(b.monthly_credit,0),
                       COALESCE(b.monthly_debit,0)) > 0
         THEN ROUND(LEAST(COALESCE(b.monthly_credit,0),
                          COALESCE(b.monthly_debit,0))
                  / GREATEST(COALESCE(b.monthly_credit,0),
                              COALESCE(b.monthly_debit,0)), 4)
         ELSE 0 END                                                 AS monthly_passthrough_ratio,
    b.monthly_unique_cp,
    b.monthly_unique_credit_cp,
    b.monthly_unique_debit_cp,
    b.monthly_structuring_count,
    b.monthly_round_count,
    CASE WHEN b.monthly_txn_count > 0
         THEN ROUND(b.monthly_round_count * 1.0 / b.monthly_txn_count, 4)
         ELSE 0 END                                                 AS monthly_round_ratio,
    b.monthly_night_txn_count,
    CASE WHEN b.monthly_txn_count > 0
         THEN ROUND(b.monthly_night_txn_count * 1.0 / b.monthly_txn_count, 4)
         ELSE 0 END                                                 AS monthly_night_txn_ratio,
    b.monthly_active_days,
    CASE WHEN b.monthly_active_days > 0
         THEN ROUND(b.monthly_txn_count * 1.0 / b.monthly_active_days, 4)
         ELSE 0 END                                                 AS monthly_txn_per_active_day,
    CASE WHEN b.monthly_txn_count > 0
         THEN ROUND(b.monthly_upi_count * 1.0 / b.monthly_txn_count, 4)
         ELSE 0 END                                                 AS monthly_upi_ratio,
    CASE WHEN b.monthly_txn_count > 0
         THEN ROUND(b.monthly_cash_count * 1.0 / b.monthly_txn_count, 4)
         ELSE 0 END                                                 AS monthly_cash_ratio,
    b.monthly_unique_channels,
    CASE WHEN b.monthly_txn_count > 0
         THEN ROUND(b.geo_tagged_count * 1.0 / b.monthly_txn_count, 4)
         ELSE 0 END                                                 AS geo_frac,
    b.unique_geo_count,
    COALESCE(ip.unique_ip_count, 0)                                 AS unique_ip_count,
    COALESCE(ip.ip_change_count, 0)                                 AS ip_change_count,
    b.atm_deposit_count,
    CASE WHEN b.monthly_txn_count > 0
         THEN ROUND(b.atm_deposit_count * 1.0 / b.monthly_txn_count, 4)
         ELSE 0 END                                                 AS atm_deposit_frac,
    bal.avg_balance,
    bal.min_balance,
    bal.max_balance,
    bal.balance_volatility,
    COALESCE(bal.near_zero_balance_count, 0)                        AS near_zero_balance_count,
    COALESCE(bal.balance_mismatch_count,  0)                        AS balance_mismatch_count,
    CASE WHEN b.monthly_txn_count > 0
         THEN ROUND(b.ci_count * 1.0 / b.monthly_txn_count, 4)
         ELSE 0 END                                                 AS ci_ratio,
    CASE WHEN b.monthly_txn_count > 0
         THEN ROUND(b.clt_cash_count * 1.0 / b.monthly_txn_count, 4)
         ELSE 0 END                                                 AS clt_cash_ratio,
    dm.monthly_max_daily_txn_count,
    dm.monthly_max_daily_amount,
    b.month_first_txn,
    b.month_last_txn

FROM base b
LEFT JOIN bal_agg  bal ON bal.account_id=b.account_id AND bal.year_month=b.year_month
LEFT JOIN ip_agg    ip ON  ip.account_id=b.account_id AND  ip.year_month=b.year_month
LEFT JOIN daily_max dm ON  dm.account_id=b.account_id AND  dm.year_month=b.year_month
ORDER BY b.account_id, b.year_month
"""


def phase2_compute_features():
    log.info("=" * 65)
    log.info("PHASE 2 — Computing features per bucket")
    log.info("Each bucket: ~30 sec   Total: ~%d min", N_BUCKETS // 2)
    log.info("=" * 65)

    out_path  = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    tmp_path  = out_path + ".building"
    pq_writer = None
    total_rows = 0
    failed    = []
    t_p2      = time.time()

    for i in range(N_BUCKETS):
        bp = bucket_path(i)
        if not os.path.exists(bp):
            log.warning("Bucket %d missing — skip", i)
            failed.append(i)
            continue

        t0  = time.time()
        sql = FEATURE_SQL.replace("{BUCKET}", bp)
        try:
            con = fresh_con()
            df  = con.execute(sql).df()
            con.close()

            if df.empty:
                log.warning("Bucket %d — 0 rows, skipping", i)
                continue

            table = pa.Table.from_pandas(df, preserve_index=False)
            if pq_writer is None:
                pq_writer = pq.ParquetWriter(tmp_path, table.schema,
                                             compression="snappy")
            pq_writer.write_table(table)
            total_rows += len(df)

            # ETA every 5 buckets
            if (i + 1) % 5 == 0 or i == 0:
                rate    = (i + 1) / (time.time() - t_p2) * 60
                eta_min = (N_BUCKETS - i - 1) / rate if rate > 0 else 0
                pct     = (i + 1) / N_BUCKETS * 100
                log.info(
                    "  Bucket %d/%d  %.0f%%  rows_this=%d  total=%d  "
                    "time=%s  ETA=%.0f min",
                    i + 1, N_BUCKETS, pct, len(df),
                    total_rows, elapsed(t0), eta_min
                )

        except duckdb.OutOfMemoryException:
            log.error(
                "Bucket %d OOM — increase N_BUCKETS (currently %d) "
                "or set DUCKDB_MEMORY_LIMIT lower",
                i, N_BUCKETS
            )
            failed.append(i)
            try: con.close()
            except Exception: pass

        except Exception as ex:
            log.error("Bucket %d error: %s", i, ex)
            failed.append(i)
            try: con.close()
            except Exception: pass

    if pq_writer is None:
        raise RuntimeError("All buckets failed — no output written")

    pq_writer.close()
    if os.path.exists(out_path):
        os.remove(out_path)
    os.rename(tmp_path, out_path)

    log.info("Phase 2 done in %s  total_rows=%d  output=%s",
             elapsed(t_p2), total_rows, out_path)

    if failed:
        log.warning("%d buckets failed: %s — increase N_BUCKETS=%d and re-run",
                    len(failed), failed, N_BUCKETS * 2)

    return out_path


# ═══════════════════════════════════════════════════════════
# PHASE 3 — Load final parquet as DataFrame
#            Ready for suspicious window detection
# ═══════════════════════════════════════════════════════════
def load_timeline(out_path: str) -> pd.DataFrame:
    """
    Load the completed timeline into pandas.
    Safe on 16 GB RAM — the timeline is much smaller than raw data.
    """
    log.info("Loading transaction_timeline into DataFrame ...")
    df = pd.read_parquet(out_path)
    df = df.sort_values(["account_id", "year_month"]).reset_index(drop=True)
    log.info("DataFrame shape: %s", df.shape)
    log.info("Columns: %s", list(df.columns))
    return df


# ═══════════════════════════════════════════════════════════
# BONUS — Suspicious window finder
#         Run this after loading predictions
# ═══════════════════════════════════════════════════════════
def find_suspicious_windows(
    timeline: pd.DataFrame,
    predictions: pd.DataFrame,    # columns: account_id, is_mule
    mule_threshold: float = 0.5,
    baseline_months: int  = 3,
    anomaly_zscore: float = 2.0,
) -> pd.DataFrame:
    """
    For each confirmed mule account, find suspicious_start and suspicious_end.

    Algorithm:
      1. Score each month with a weighted composite fraud score
      2. Compare to the account's own first N months as baseline (z-score)
      3. suspicious_start = first month where z-score > threshold
      4. suspicious_end   = last  month where z-score > threshold
    """
    WEIGHTS = {
        "monthly_passthrough_ratio"  : 3.0,
        "monthly_credit_debit_ratio" : 2.0,
        "monthly_structuring_count"  : 2.5,
        "monthly_night_txn_ratio"    : 1.5,
        "monthly_round_ratio"        : 1.0,
        "ip_change_count"            : 1.5,
        "near_zero_balance_count"    : 1.5,
        "atm_deposit_frac"           : 1.5,
        "monthly_txn_count"          : 1.0,
        "monthly_total_amount"       : 1.0,
        "clt_cash_ratio"             : 1.5,
        "balance_volatility"         : 1.0,
    }

    preds = predictions.copy()
    preds.columns = preds.columns.str.strip().str.lower()
    mules = set(preds.loc[preds["is_mule"] >= mule_threshold, "account_id"])

    df = timeline[timeline["account_id"].isin(mules)].copy()
    df = df.sort_values(["account_id", "year_month"]).reset_index(drop=True)

    # Compute weighted score (0-1 per feature, globally normalised)
    score = pd.Series(0.0, index=df.index)
    total_w = 0
    for feat, w in WEIGHTS.items():
        if feat not in df.columns:
            continue
        col     = df[feat].fillna(0).clip(lower=0)
        col_max = col.max()
        norm    = col / col_max if col_max > 0 else col
        score  += norm * w
        total_w += w
    df["_score"] = score / total_w

    # Personalised z-score vs account's own baseline
    results = []
    for acct, grp in df.groupby("account_id"):
        grp = grp.sort_values("year_month").reset_index(drop=True)
        n   = len(grp)
        bn  = min(baseline_months, max(1, n - 1))

        bmu = grp["_score"].iloc[:bn].mean()
        bsd = grp["_score"].iloc[:bn].std()
        if pd.isna(bsd) or bsd < 1e-9:
            bsd = 1e-9

        grp["_z"]     = (grp["_score"] - bmu) / bsd
        grp["_susp"]  = grp["_z"] >= anomaly_zscore

        susp_rows = grp[grp["_susp"]]
        if susp_rows.empty:
            sus_start = sus_end = None
        else:
            first_row = susp_rows.iloc[0]
            last_row  = susp_rows.iloc[-1]
            sus_start = first_row.get("month_first_txn",
                            pd.to_datetime(first_row["year_month"] + "-01"))
            sus_end   = last_row.get("month_last_txn",
                            pd.to_datetime(last_row["year_month"] + "-01")
                            + pd.offsets.MonthEnd(0))

        results.append({
            "account_id"       : acct,
            "suspicious_start" : sus_start,
            "suspicious_end"   : sus_end,
            "suspicious_months": len(susp_rows),
            "peak_zscore"      : round(grp["_z"].max(), 3),
            "avg_score"        : round(grp["_score"].mean(), 4),
        })

    result_df = pd.DataFrame(results)

    # Merge is_mule back in
    result_df = result_df.merge(
        preds[["account_id", "is_mule"]], on="account_id", how="left"
    )

    # Add non-mule accounts
    non_mule = preds[preds["is_mule"] < mule_threshold][["account_id","is_mule"]].copy()
    non_mule["suspicious_start"]  = None
    non_mule["suspicious_end"]    = None
    non_mule["suspicious_months"] = 0
    non_mule["peak_zscore"]       = None
    non_mule["avg_score"]         = None

    final = pd.concat([result_df, non_mule], ignore_index=True)
    final = final.sort_values(["is_mule","account_id"],
                               ascending=[False, True]).reset_index(drop=True)

    final = final[[
        "account_id", "is_mule",
        "suspicious_start", "suspicious_end",
        "suspicious_months", "peak_zscore", "avg_score"
    ]]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = os.path.join(OUTPUT_DIR, "suspicious_windows.csv")
    final.to_csv(out, index=False)
    log.info("Suspicious windows saved: %s  (%d mule accounts with windows)",
             out, final["suspicious_start"].notna().sum())
    return final


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    t_total = time.time()
    try:
        log.info("Starting Transaction Timeline Builder v6 (Hash-Partition)")
        validate_paths(TRANSACTIONS_GLOB, TRANS_ADDITIONAL_GLOB)

        # Phase 1 — partition raw data into bucket files (one-time ~30 min)
        phase1_partition()

        # Phase 2 — compute all features per bucket (~20-30 min)
        out_path = phase2_compute_features()

        # Load into pandas DataFrame
        timeline_df = load_timeline(out_path)

        log.info("transaction_timeline ready — shape=%s", timeline_df.shape)
        log.info("Total runtime: %s", elapsed(t_total))

        # ── Optional: find suspicious windows if predictions exist ────────────
        preds_path = "predictions.csv"
        if os.path.exists(preds_path):
            log.info("predictions.csv found — running suspicious window detection")
            preds = pd.read_csv(preds_path)
            windows_df = find_suspicious_windows(timeline_df, preds)
            log.info("\nTop 10 mule accounts:\n%s",
                windows_df[windows_df["suspicious_start"].notna()]
                .sort_values("peak_zscore", ascending=False)
                .head(10)
                .to_string(index=False)
            )
        else:
            log.info(
                "No predictions.csv found — timeline is ready.\n"
                "Call find_suspicious_windows(timeline_df, predictions_df) "
                "when you have predictions."
            )

        return timeline_df

    except FileNotFoundError as e:
        log.error("PATH ERROR: %s", e)
        sys.exit(1)
    except Exception as e:
        log.exception("Fatal error: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()