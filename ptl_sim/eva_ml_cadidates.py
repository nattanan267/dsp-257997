# eval_ml_candidates.py
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple
from config import DATA_DIR
from ml_utils import load_wait_regressor, FEATURE_COLS, rank_eval_hit_rate_at_k

LAMBDA_D = 0.20   
LAMBDA_U = 0.10   
USE_BLEND = True  

PROV_CSV = Path(DATA_DIR) / "provider_lat_long.csv"  # fallback capacity สำหรับเติม prov_*

def _prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    X = df[FEATURE_COLS].copy()
    for c in FEATURE_COLS:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    mask = X.notnull().all(axis=1)
    return X.loc[mask], mask

def ensure_v2_capacity_features(df: pd.DataFrame) -> pd.DataFrame:
    need = {"prov_throughput_week", "prov_demand_week", "prov_utilization"}
    if need.issubset(df.columns):
        return df

    df = df.copy()

    if "prov_throughput_week" not in df.columns:
        prov = pd.read_csv(PROV_CSV, low_memory=False)
        if "is_nhs" in prov.columns:
            prov["is_nhs"] = prov["is_nhs"].astype(str).str.lower().map(
                {"true":1, "false":0, "1":1, "0":0}
            ).fillna(0).astype(int)
        else:
            prov["is_nhs"] = 0

        if "SlotsPerDay" in prov.columns:
            prov["SlotsPerDay"] = pd.to_numeric(prov["SlotsPerDay"], errors="coerce").fillna(0)
            slot_min = np.where(prov["is_nhs"]==1, 20.0, 15.0)
            minutes_per_day_cap = prov["SlotsPerDay"] * slot_min
        else:
            slot_min = np.where(prov["is_nhs"]==1, 20.0, 15.0)
            minutes_per_day_cap = 8*60

        cases_per_day_cap = minutes_per_day_cap / slot_min
        prov["prov_throughput_week"] = cases_per_day_cap * 5
        prov_cap = prov[["provider","prov_throughput_week"]].rename(columns={"provider":"provider"})
        df = df.merge(prov_cap, on="provider", how="left")

    if "prov_demand_week" not in df.columns:
        prov_counts = df.groupby("provider").size().rename("cnt").reset_index()
        weeks = 52
        prov_counts["prov_demand_week"] = prov_counts["cnt"] / weeks
        df = df.merge(prov_counts[["provider","prov_demand_week"]], on="provider", how="left", suffixes=("", "_est"))
        if "prov_demand_week_est" in df.columns:
            df["prov_demand_week"] = df["prov_demand_week"].fillna(df["prov_demand_week_est"])
            df = df.drop(columns=["prov_demand_week_est"])

    for c in ["prov_throughput_week","prov_demand_week"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").replace([np.inf,-np.inf], np.nan)

    med_tp = df["prov_throughput_week"].median() if "prov_throughput_week" in df.columns else np.nan
    med_dm = df["prov_demand_week"].median() if "prov_demand_week" in df.columns else np.nan
    if not np.isfinite(med_tp): med_tp = 1.0
    if not np.isfinite(med_dm): med_dm = 1.0

    if "prov_throughput_week" not in df.columns: df["prov_throughput_week"] = med_tp
    else: df["prov_throughput_week"] = df["prov_throughput_week"].fillna(med_tp)
    if "prov_demand_week" not in df.columns: df["prov_demand_week"] = med_dm
    else: df["prov_demand_week"] = df["prov_demand_week"].fillna(med_dm)

    if "prov_utilization" not in df.columns:
        df["prov_utilization"] = (df["prov_demand_week"] / df["prov_throughput_week"])\
            .replace([np.inf,-np.inf], np.nan).fillna(0.5)

    return df

def _compute_topk_fallback_delta(df: pd.DataFrame, k: int = 3) -> Tuple[float, float, float]:
    best = df.loc[df.groupby("case_id")["actual_wait"].idxmin(), ["case_id", "provider", "actual_wait"]]
    best = best.rename(columns={"provider": "best_provider", "actual_wait": "best_wait"})

    df_sorted = df.sort_values(["case_id", "rank_score"], ascending=[True, True]).copy()
    df_sorted["rank_pred"] = df_sorted.groupby("case_id").cumcount() + 1
    topk = df_sorted[df_sorted["rank_pred"] <= k].copy()

    agg = topk.groupby("case_id").agg(
        providers=("provider", list),
        actuals=("actual_wait", list),
    ).reset_index()

    merged = best.merge(agg, on="case_id", how="left")

    def choose_wait(row):
        provs = row["providers"]; actuals = row["actuals"]
        if isinstance(provs, list) and len(provs) > 0:
            if row["best_provider"] in provs:
                return actuals[provs.index(row["best_provider"])]
            else:
                return actuals[0]
        return row["best_wait"]

    merged["chosen_wait"] = merged.apply(choose_wait, axis=1)
    best_avg = float(merged["best_wait"].mean())
    chosen_avg = float(merged["chosen_wait"].mean())
    return best_avg, chosen_avg, chosen_avg - best_avg

if __name__ == "__main__":
    MODEL = Path(DATA_DIR) / "models" / "ml_wait_regressor_v2.pkl"
    CAND  = Path(DATA_DIR) / "ml_rank_candidates_v2.csv"

    df = pd.read_csv(CAND, low_memory=False, dtype={"provider":"string"})

    df = ensure_v2_capacity_features(df)

    model = load_wait_regressor(MODEL)

    X, mask = _prepare_features(df)
    df = df.loc[mask].copy()
    df["pred_wait_time"] = model.predict(X)


    if USE_BLEND:
        def z(s):
            s = pd.to_numeric(s, errors="coerce")
            mu = s.mean()
            sd = s.std(ddof=0)
            return (s - (0.0 if not np.isfinite(mu) else mu)) / (1e-9 if not np.isfinite(sd) or sd==0 else sd)

        d_norm = z(df["distance_km"])
        u_norm = z(df["prov_utilization"])
        df["rank_score"] = df["pred_wait_time"] + LAMBDA_D * d_norm + LAMBDA_U * u_norm
    else:
        df["rank_score"] = df["pred_wait_time"]


    hit1, mrr1 = rank_eval_hit_rate_at_k(
        model=model,
        candidates_df=df.assign(pred_wait=df["rank_score"]),  
        k=1, case_col="case_id", provider_col="provider", actual_wait_col="actual_wait",
    )
    hit3, mrr3 = rank_eval_hit_rate_at_k(
        model=model,
        candidates_df=df.assign(pred_wait=df["rank_score"]),
        k=3, case_col="case_id", provider_col="provider", actual_wait_col="actual_wait",
    )
    print(f"Hit@1: {hit1:.3f} | MRR@1: {mrr1:.3f}")
    print(f"Hit@3: {hit3:.3f} | MRR@3: {mrr3:.3f}")


    best_avg, chosen_avg, delta = _compute_topk_fallback_delta(df, k=3)
    print(f"\nBest possible avg wait  : {best_avg:.2f} days")
    print(f"ML Top-3+fallback avg   : {chosen_avg:.2f} days")
    print(f"Δ wait (Top-3 - Best)   : {delta:.2f} days")
