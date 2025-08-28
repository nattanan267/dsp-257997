# main.py
import random
from typing import List, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import DATA_DIR
from scenario_runner import run_policy_scenario
from metrics import compute_realdata_summary
from policy import target_share_balanced_policy


R = 10                
BASE_SEED = 42
SCENARIO_ID = "base"


FRONTIER_SCENARIO_ID_2 = "overall_share_frontier"
OVERALL_TARGETS = [round(x, 1) for x in np.linspace(0.3, 0.9, 7)]  # 0.3..0.9
FRONTIER_FALLBACK_2 = 60
INCLUDE_GA_IN_TARGET = True   # True = รับ % ภาพรวมทั้งระบบตามที่อาจารย์ถาม

rows_overall = []

# ----- Load data -----
df_patients = pd.read_csv(DATA_DIR / "patient_lat_long.csv", low_memory=False)
df_providers = pd.read_csv(DATA_DIR / "provider_lat_long.csv")
df_all = df_patients.copy()

def _compute_overall_nhs_share_pct(df_result: pd.DataFrame) -> float:
    """
    % ไป NHS (รวม GA/non-GA) ในบรรดาเคสที่ถูก assign ภายใน 365 วัน
    """
    if "assigned_is_nhs" not in df_result.columns:
        return float("nan")
    ok = df_result["wait_time"].notnull() & (pd.to_numeric(df_result["wait_time"], errors="coerce") <= 365)
    if ok.sum() == 0:
        return float("nan")
    nhs_flag = df_result.loc[ok, "assigned_is_nhs"].astype(str).str.lower().isin(["true", "1", "yes"])
    return float(nhs_flag.mean() * 100.0)

total_runs = len(OVERALL_TARGETS) * R
master_bar = tqdm(total=total_runs, desc="Total runs", position=0)

for target_share in OVERALL_TARGETS:
    policy_label = f"TargetShare (overall={target_share:.1f})"

    rep_bar = tqdm(range(R), desc=f"Replicates @ target={target_share:.1f}", position=1, leave=False)

    for r in rep_bar:
        mix = abs(hash((policy_label, FRONTIER_SCENARIO_ID_2))) % 1_000_000
        seed = BASE_SEED + mix + r
        random.seed(seed); np.random.seed(seed)

        def overall_share_policy(patients, providers, use_priority=False, _t=target_share):
            return target_share_balanced_policy(
                patients, providers, use_priority=use_priority,
                target_overall_nhs_share=float(_t),
                include_ga_in_target=INCLUDE_GA_IN_TARGET,
                fallback_after_days=FRONTIER_FALLBACK_2,
                slack_days=14,
                max_days_forward=365,
            )

        df_result, summary = run_policy_scenario(
            df_all, df_providers, overall_share_policy,
            policy_name=policy_label, use_priority=False, simulate_after_policy=False
        )

        try:
            avg_wait_val = float(pd.to_numeric(summary["avg_wait"], errors="coerce").iloc[0])
        except Exception:
            avg_wait_val = float("nan")
        try:
            avg_dist_val = float(pd.to_numeric(summary["avg_travel_distance_km"], errors="coerce").iloc[0])
        except Exception:
            avg_dist_val = float("nan")

        rows_overall.append({
            "scenario_id": FRONTIER_SCENARIO_ID_2,
            "target_overall_nhs_share": target_share,
            "replicate": r,
            "achieved_overall_nhs_share_pct": round(_compute_overall_nhs_share_pct(df_result), 2),
            "avg_wait": round(avg_wait_val, 2) if pd.notna(avg_wait_val) else np.nan,
            "avg_travel_distance_km": round(avg_dist_val, 2) if pd.notna(avg_dist_val) else np.nan,
        })

        master_bar.update(1)

    rep_bar.close()

master_bar.close()

df_overall_raw = pd.DataFrame(rows_overall)
df_overall_raw.to_csv(DATA_DIR / "summary_results/overall_nhs_share_frontier_raw.csv", index=False)

def _agg_ci_simple(vals: np.ndarray):
    vals = pd.to_numeric(pd.Series(vals), errors="coerce").dropna().values
    if len(vals) == 0:
        return np.nan, np.nan
    m = float(np.mean(vals))
    if len(vals) < 2:
        return m, 0.0
    s = float(np.std(vals, ddof=1))
    return m, 1.96 * s / np.sqrt(len(vals))

rows_ci2 = []
for t, g in df_overall_raw.groupby("target_overall_nhs_share"):
    mw, hw = _agg_ci_simple(g["avg_wait"])
    ms, hs = _agg_ci_simple(g["achieved_overall_nhs_share_pct"])
    md, hd = _agg_ci_simple(g["avg_travel_distance_km"])
    rows_ci2.append({
        "target_overall_nhs_share": t,
        "avg_wait_mean": round(mw, 2) if pd.notna(mw) else np.nan,
        "avg_wait_ci_half": round(hw, 2) if pd.notna(hw) else np.nan,
        "achieved_overall_nhs_share_mean": round(ms, 2) if pd.notna(ms) else np.nan,
        "achieved_overall_nhs_share_ci_half": round(hs, 2) if pd.notna(hs) else np.nan,
        "avg_travel_distance_mean": round(md, 2) if pd.notna(md) else np.nan,
        "avg_travel_distance_ci_half": round(hd, 2) if pd.notna(hd) else np.nan,
    })

df_overall_ci = pd.DataFrame(rows_ci2).sort_values("target_overall_nhs_share")
df_overall_ci.to_csv(DATA_DIR / "summary_results/overall_nhs_share_frontier_ci.csv", index=False)
