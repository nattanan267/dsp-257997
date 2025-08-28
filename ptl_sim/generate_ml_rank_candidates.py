# generate_ml_rank_candidates_v2.py
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List
from config import DATA_DIR

FEATURE_COLS = [
    "need_ga", "priority", "complexity_enc", "tariff",
    "patient_lat", "patient_long",
    "lat_prov", "long_prov",
    "is_nhs",
    "distance_km",
    "prov_throughput_week",
    "prov_demand_week",
    "prov_utilization",
]

def _haversine_vec(lat1, lon1, lat2, lon2, radius_km: float = 6371.0):
    lat1 = np.radians(np.asarray(lat1, dtype=float))
    lon1 = np.radians(np.asarray(lon1, dtype=float))
    lat2 = np.radians(np.asarray(lat2, dtype=float))
    lon2 = np.radians(np.asarray(lon2, dtype=float))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    return radius_km * (2.0 * np.arcsin(np.sqrt(a)))

def _detect_case_id_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["case_id", "patient_id", "id", "local_patient_identifier"]:
        if c in df.columns:
            return c
    return None

def _coerce_boolish_to_int(s: pd.Series) -> pd.Series:
    return (s.astype(str).str.strip().str.lower()
            .map({"true":1, "false":0, "1":1, "0":0})
            .fillna(0).astype(int))

def build_ml_rank_candidates_v2(sim_outputs_pattern,
                                providers_csv,
                                out_csv,
                                min_candidates_per_case: int = 2,
                                aggregate: str = "median",
                                verbose: bool = True):
    sim_outputs_pattern = Path(sim_outputs_pattern)
    providers_csv = Path(providers_csv)
    out_csv = Path(out_csv)


    df_prov = pd.read_csv(providers_csv)
    if "is_nhs" in df_prov.columns:
        df_prov["is_nhs"] = _coerce_boolish_to_int(df_prov["is_nhs"])
    else:
        df_prov["is_nhs"] = 0

    need_cols = ["provider", "lat", "long", "is_nhs"]
    missing = [c for c in need_cols if c not in df_prov.columns]
    if missing:
        raise ValueError(f"providers_csv missing columns: {missing}")
    df_prov = df_prov.rename(columns={"lat":"lat_prov", "long":"long_prov"})

    # capacity guess (SlotsPerDay -> cases/week; ไม่มีให้ fallback 8 ชม./วัน)
    if "SlotsPerDay" in df_prov.columns:
        df_prov["SlotsPerDay"] = pd.to_numeric(df_prov["SlotsPerDay"], errors="coerce").fillna(0)
        slot_min = np.where(df_prov["is_nhs"]==1, 20.0, 15.0)
        minutes_per_day_cap = df_prov["SlotsPerDay"] * slot_min
    else:
        slot_min = np.where(df_prov["is_nhs"]==1, 20.0, 15.0)
        minutes_per_day_cap = 8 * 60

    cases_per_day_cap = minutes_per_day_cap / slot_min
    df_prov["cases_per_week_cap"] = cases_per_day_cap * 5
    df_prov_use = df_prov[["provider","lat_prov","long_prov","is_nhs","cases_per_week_cap"]]


    csv_paths = sorted(Path(sim_outputs_pattern.parent).glob(sim_outputs_pattern.name))
    if verbose:
        print(f"Found {len(csv_paths)} simulation files")

    frames: List[pd.DataFrame] = []
    for p in csv_paths:
        try:
            df = pd.read_csv(p)
        except Exception as e:
            if verbose: print(f"Skip {p}: read error {e}")
            continue

        if "wait_time" not in df.columns or "assigned_provider" not in df.columns:
            if verbose: print(f"Skip {p}: missing wait_time or assigned_provider")
            continue

        case_col = _detect_case_id_col(df)
        if case_col is None:
            if verbose: print(f"Skip {p}: cannot detect case id column")
            continue

        use = df.copy()
        use = use[use["wait_time"].notnull()].copy()

        # need_ga / needs_ga
        if "need_ga" not in use.columns:
            if "needs_ga" in use.columns:
                use["need_ga"] = pd.to_numeric(use["needs_ga"], errors="coerce").fillna(0).astype(int)
            else:
                use["need_ga"] = 0
        else:
            use["need_ga"] = pd.to_numeric(use["need_ga"], errors="coerce").fillna(0).astype(int)

        # priority / tariff
        use["priority"] = pd.to_numeric(use.get("priority", 3), errors="coerce").fillna(3).astype(int)
        use["tariff"] = pd.to_numeric(use.get("tariff", 0.0), errors="coerce").fillna(0.0)

        # complexity_enc
        if "complexity_enc" not in use.columns:
            comp_map = {"Low":0, "Medium":1, "High":2}
            if "complexity" in use.columns:
                use["complexity_enc"] = use["complexity"].astype(str).str.strip().str.title().map(comp_map)
            else:
                use["complexity_enc"] = np.nan
        use["complexity_enc"] = pd.to_numeric(use["complexity_enc"], errors="coerce")

        # patient coords
        if {"patient_lat","patient_long"}.issubset(use.columns):
            pass
        elif {"lat","long"}.issubset(use.columns):
            use = use.rename(columns={"lat":"patient_lat","long":"patient_long"})
        else:
            if verbose: print(f"Skip {p}: missing patient lat/long")
            continue

        # ensure is_nhs in 'use' (ถ้าไฟล์มี assigned_is_nhs)
        if "is_nhs" not in use.columns:
            if "assigned_is_nhs" in use.columns:
                use["is_nhs"] = _coerce_boolish_to_int(use["assigned_is_nhs"])
            else:
                use["is_nhs"] = np.nan
        else:
            use["is_nhs"] = _coerce_boolish_to_int(use["is_nhs"])

        # join provider coords & is_nhs & capacity
        use = use.merge(df_prov_use, left_on="assigned_provider", right_on="provider", how="left")

        # fix duplicate is_nhs (_x/_y)
        if "is_nhs_x" in use.columns and "is_nhs_y" in use.columns:
            use["is_nhs"] = use["is_nhs_y"].fillna(use["is_nhs_x"])
            use.drop(columns=["is_nhs_x","is_nhs_y"], inplace=True)

        # numeric & dropna coords
        for c in ["patient_lat","patient_long","lat_prov","long_prov"]:
            use[c] = pd.to_numeric(use[c], errors="coerce")
        use = use.dropna(subset=["patient_lat","patient_long","lat_prov","long_prov"])

        # distance
        use["distance_km"] = _haversine_vec(use["patient_lat"], use["patient_long"],
                                            use["lat_prov"], use["long_prov"])

        use = use.rename(columns={case_col:"case_id"})
        use["provider"] = use["provider"].astype(str)

        frames.append(
            use[["case_id","assigned_provider","wait_time","need_ga","priority","complexity_enc",
                 "tariff","patient_lat","patient_long","lat_prov","long_prov","is_nhs",
                 "distance_km","cases_per_week_cap"]].rename(
                     columns={"assigned_provider":"provider",
                              "cases_per_week_cap":"prov_throughput_week"}
                 )
        )

    if not frames:
        raise ValueError("No usable simulation outputs.")
    df_all = pd.concat(frames, ignore_index=True)


    prov_counts = df_all.groupby("provider").size().rename("cnt").reset_index()
    weeks = 52
    prov_counts["prov_demand_week"] = prov_counts["cnt"] / weeks
    df_all = df_all.merge(prov_counts[["provider","prov_demand_week"]], on="provider", how="left")

    # utilization
    df_all["prov_utilization"] = (df_all["prov_demand_week"] / df_all["prov_throughput_week"])\
        .replace([np.inf,-np.inf], np.nan).fillna(0.5)


    df_agg = (
        df_all
        .groupby(["case_id","provider"], as_index=False)
        .agg(
            actual_wait=("wait_time", aggregate),   # 'median' หรือ 'mean'
            need_ga=("need_ga","last"),
            priority=("priority","last"),
            complexity_enc=("complexity_enc","last"),
            tariff=("tariff","last"),
            patient_lat=("patient_lat","last"),
            patient_long=("patient_long","last"),
            lat_prov=("lat_prov","last"),
            long_prov=("long_prov","last"),
            is_nhs=("is_nhs","last"),
            distance_km=("distance_km","last"),
            prov_throughput_week=("prov_throughput_week","last"),
            prov_demand_week=("prov_demand_week","last"),
            prov_utilization=("prov_utilization","last"),
        )
    )


    vc = df_agg["case_id"].value_counts()
    valid_cases = set(vc[vc >= min_candidates_per_case].index)
    df_out = df_agg[df_agg["case_id"].isin(valid_cases)].copy()


    must_have = ["patient_lat","patient_long","lat_prov","long_prov","distance_km","actual_wait"]
    df_out = df_out.dropna(subset=must_have)

    int_like = ["need_ga","priority","complexity_enc","is_nhs"]
    for c in int_like:
        df_out[c] = pd.to_numeric(df_out[c], errors="coerce").fillna(0).astype(int)
    float_like = ["tariff","patient_lat","patient_long","lat_prov","long_prov",
                  "distance_km","prov_throughput_week","prov_demand_week","prov_utilization","actual_wait"]
    for c in float_like:
        df_out[c] = pd.to_numeric(df_out[c], errors="coerce")

    # columns order
    df_out = df_out[["case_id","provider","actual_wait"] + FEATURE_COLS].reset_index(drop=True)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_csv, index=False)

    if verbose:
        n_cases = df_out["case_id"].nunique()
        avg_cands = df_out.groupby("case_id")["provider"].nunique().mean()
        print(f"✅ Saved {out_csv}")
        print(f"- cases: {n_cases:,} | avg candidates per case: {avg_cands:.2f}")
        print(f"- shape: {df_out.shape}")

if __name__ == "__main__":
    build_ml_rank_candidates_v2(
        sim_outputs_pattern= Path(DATA_DIR) / "simulation_output" / "*.csv",
        providers_csv= Path(DATA_DIR) / "provider_lat_long.csv",
        out_csv= Path(DATA_DIR) / "ml_rank_candidates_v2.csv",
        min_candidates_per_case=2,
        aggregate="median",
        verbose=True,
    )
