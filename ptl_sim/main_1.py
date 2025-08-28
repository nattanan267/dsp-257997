# main.py
import pandas as pd
from config import DATA_DIR
from scenario_runner import run_policy_scenario
from metrics import compute_realdata_summary
import numpy as np
from policy import (
    baseline_policy,
    shared_ptl_equal_policy,
    greedy_policy,
    # greedy_policy_with_distance,
    # capacity_weighted_policy,
    # capacity_weighted_policy_with_distance,
    complexity_balanced_policy,
    fee_biased_policy,
    neutral_fee_policy,
    # profit_maximizing_policy,
    ml_balanced_policy,
)
from tqdm import tqdm
import random

R = 10                 
BASE_SEED = 42         
SCENARIO_ID = "base"   

def _mean_ci_normal(vals, ci=0.95):
    """คืน (mean, half_CI) ใช้ normal approx; ถ้า n<2 -> half_CI=0"""
    x = pd.to_numeric(pd.Series(vals), errors="coerce").dropna().values
    n = len(x)
    if n == 0:
        return np.nan, np.nan
    m = float(np.mean(x))
    if n < 2:
        return m, 0.0
    s = float(np.std(x, ddof=1))
    z = 1.96 if abs(ci - 0.95) < 1e-9 else 1.96  
    half = z * s / np.sqrt(n)
    return m, half

def aggregate_with_ci(df_summaries, by=["policy"]):
    """
    รวมผลสรุปข้าม replications → สร้างคอลัมน์ mean และ ci_half
    KPI ที่ใช้: avg_wait, pct_seen_within_90_days, avg_travel_distance_km, avg_utilisation, avg_tariff_all
    """
    kpis = [
        "avg_wait",
        "pct_seen_within_90_days",
        "avg_travel_distance_km",
        "avg_utilisation",
        "avg_tariff_all", 
    ]
    rows = []
    for keys, g in df_summaries.groupby(by):
        row = {}
        if isinstance(keys, tuple):
            for k, v in zip(by, keys):
                row[k] = v
        else:
            row[by[0]] = keys
        for k in kpis:
            m, half = _mean_ci_normal(g[k])
            row[f"{k}_mean"] = round(m, 2) if pd.notna(m) else np.nan
            row[f"{k}_ci_half"] = round(half, 2) if pd.notna(half) else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(by)

def mean_ci(df: pd.DataFrame, col: str, ci: float = 0.95):
    vals = pd.to_numeric(df[col], errors="coerce").dropna().values
    n = len(vals)
    if n == 0:
        return np.nan, (np.nan, np.nan)
    m = float(np.mean(vals))
    s = float(np.std(vals, ddof=1)) if n > 1 else 0.0
    # normal approx:
    z = 1.96 if ci == 0.95 else 1.96  # ปรับเองถ้าอยากได้ระดับอื่น
    half = z * s / np.sqrt(n) if n > 1 else 0.0
    return m, (m - half, m + half)


df_patients = pd.read_csv(DATA_DIR / "patient_lat_long.csv", low_memory=False)
df_providers = pd.read_csv(DATA_DIR / "provider_lat_long.csv")

df_all = df_patients.copy()


realdata_summary = compute_realdata_summary(df_patients, df_providers)
realdata_summary["description"] = "Real data baseline from recorded wait time"
realdata_summary.to_csv(DATA_DIR / "summary_results/summary_realdata.csv", index=False)

summaries = [realdata_summary]

# ----- Policy list -----
policy_configs = [
    ("Baseline", baseline_policy, "Nearest hospital; High -> Independent sector ~10% (constrained); GA -> NHS only", False),
    ("Shared Equal", shared_ptl_equal_policy, "Distribute all cases equally; GA -> NHS only", False),
    ("Fastest First", greedy_policy, "Assign earliest feasible slot across providers; GA -> NHS only", False),
    # # ("Greedy Fastest Distance", greedy_policy_with_distance, "Assign to shortest queue and nearest", False),
    # # ("Capacity Weighted", capacity_weighted_policy, "Based on provider capacity; GA -> NHS only", False),
    # # ("Capacity Weighted Distance", capacity_weighted_policy_with_distance, "Based on provider capacity and distance; GA -> NHS only", False),
    ("Complexity Balanced", complexity_balanced_policy, "Evenly distribute cases by complexity", False),
    ("Fee Biased", fee_biased_policy, "Private providers prefer high-tariff cases", False),
    ("Neutral Fee", neutral_fee_policy, "No tariff-based bias", False),
    # # ("Profit Maximizing Policy", profit_maximizing_policy, "No tariff-based bias", False),
    (
        "ML Balanced (wait+dist+util)",
        lambda patients, providers, use_priority=False: ml_balanced_policy(
            patients,
            providers,
            use_priority=use_priority,
            model_path=str(DATA_DIR / "models" / "ml_wait_regressor_v2.pkl"),
            lambda_d=0.15,
            lambda_u=0.10,
            top_k=3,
            lookahead_days=30,
            max_days_forward=365,
        ),
        "ML v2: predict wait + blend distance/utilization; Top-3 with short lookahead booking",
        False,
    ),
]

# ----- Run -----
for use_priority in [False]:
    suffix = " +Priority" if use_priority else ""
    for name, policy_fn, description, simulate_after_policy in tqdm(
        policy_configs, desc=f"Running policy simulations{suffix}"
    ):
        policy_name = name + suffix

        for r in range(R):
            mix = abs(hash(policy_name)) % 1_000_000
            seed = BASE_SEED + mix + r
            random.seed(seed)
            np.random.seed(seed)

            df_result, summary = run_policy_scenario(
                df_all,
                df_providers,
                policy_fn,
                policy_name,
                use_priority=use_priority,
                simulate_after_policy=simulate_after_policy,
            )

            out_tag = f"{policy_name.lower().replace(' ', '_')}_r{r}"
            df_result.to_csv(
                DATA_DIR / f"simulation_output/sim_{out_tag}.csv",
                index=False,
            )

            if "policy" not in summary.columns:
                summary["policy"] = policy_name
            summary["description"] = description + suffix
            summary["replicate"] = r
            summary["scenario_id"] = SCENARIO_ID

            summary.to_csv(
                DATA_DIR / f"summary_results/summary_{out_tag}.csv",
                index=False,
            )
            summaries.append(summary)

summary_all = pd.concat(summaries, ignore_index=True)
summary_all.to_csv(DATA_DIR / "summary_results/summary_all_policies.csv", index=False)

try:
    df_ci = aggregate_with_ci(
        summary_all.loc[summary_all["scenario_id"] == SCENARIO_ID],  # ตัด realdata ออก
        by=["policy"]
    )
    df_ci.to_csv(DATA_DIR / "summary_results/summary_ci_policies.csv", index=False)
except Exception as e:
    print("CI aggregation skipped due to error:", e)

