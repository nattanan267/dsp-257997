import pandas as pd
import numpy as np
from pathlib import Path
import glob

DATA_DIR = Path(__file__).resolve().parent / "data"

def build_wait_dataset_v2(
    sim_outputs_pattern,
    provider_csv,
    output_csv,
    verbose=True
):
    df_prov = pd.read_csv(provider_csv)
    df_prov["is_nhs"] = pd.to_numeric(df_prov["is_nhs"], errors="coerce").fillna(0).astype(int)

    if "SlotsPerDay" in df_prov.columns:
        df_prov["SlotsPerDay"] = pd.to_numeric(df_prov["SlotsPerDay"], errors="coerce").fillna(0)
    else:
        df_prov["SlotsPerDay"] = np.nan


    df_prov["slot_minutes_guess"] = np.where(df_prov["is_nhs"] == 1, 20.0, 15.0)


    df_prov["minutes_per_day_cap"] = np.where(
        df_prov["SlotsPerDay"].notnull() & (df_prov["SlotsPerDay"] > 0),
        df_prov["SlotsPerDay"] * df_prov["slot_minutes_guess"],
        8 * 60
    )


    df_prov["cases_per_day_cap"] = df_prov["minutes_per_day_cap"] / df_prov["slot_minutes_guess"]
    df_prov["cases_per_week_cap"] = df_prov["cases_per_day_cap"] * 5
    df_prov_caps = df_prov[["provider", "cases_per_week_cap"]].rename(columns={"provider": "assigned_provider"})


    files = sorted(glob.glob(str(sim_outputs_pattern)))
    if verbose:
        print(f"Found {len(files)} simulation files")

    df_all = []
    for f in files:
        df = pd.read_csv(f)
        df_all.append(df)
    df_feat = pd.concat(df_all, ignore_index=True)


    expected_cols = [
        "need_ga", "priority", "complexity_enc", "tariff", "is_nhs",
        "patient_lat", "patient_long", "lat_prov", "long_prov",
        "distance_km", "wait_time", "assigned_provider", "case_id", "year_week",
        "prov_throughput_week", "prov_demand_week"
    ]
    for col in expected_cols:
        if col not in df_feat.columns:
            df_feat[col] = np.nan


    # throughput/week
    df_feat["prov_throughput_week"] = pd.to_numeric(df_feat["prov_throughput_week"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    if df_feat["prov_throughput_week"].isna().all():
        df_feat = df_feat.merge(df_prov_caps, on="assigned_provider", how="left")
        df_feat["prov_throughput_week"] = df_feat["prov_throughput_week"].fillna(df_feat["cases_per_week_cap"])
        df_feat = df_feat.drop(columns=["cases_per_week_cap"])
    df_feat["prov_throughput_week"] = df_feat["prov_throughput_week"].fillna(df_feat["prov_throughput_week"].median())

    # demand/week
    df_feat["prov_demand_week"] = pd.to_numeric(df_feat["prov_demand_week"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    if df_feat["prov_demand_week"].isna().all():
        weeks = max(1, df_feat["year_week"].nunique()) if ("year_week" in df_feat.columns and df_feat["year_week"].notnull().any()) else 52
        prov_counts = df_feat.groupby("assigned_provider").size().rename("cnt").reset_index()
        prov_counts["prov_demand_week"] = prov_counts["cnt"] / weeks
        df_feat = df_feat.merge(prov_counts[["assigned_provider", "prov_demand_week"]], on="assigned_provider", how="left", suffixes=("", "_est"))
        if "prov_demand_week_est" in df_feat.columns:
            df_feat["prov_demand_week"] = df_feat["prov_demand_week"].fillna(df_feat["prov_demand_week_est"])
            df_feat = df_feat.drop(columns=["prov_demand_week_est"])
    df_feat["prov_demand_week"] = df_feat["prov_demand_week"].fillna(df_feat["prov_demand_week"].median())

    # utilization
    df_feat["prov_utilization"] = (df_feat["prov_demand_week"] / df_feat["prov_throughput_week"]).replace([np.inf, -np.inf], np.nan).fillna(0.5)

    df_feat.to_csv(output_csv, index=False)
    if verbose:
        print(f"Saved {output_csv} | rows={len(df_feat):,} providers={df_feat['assigned_provider'].nunique()}")
        print(f"Columns: {list(df_feat.columns)}")
        print(df_feat[["prov_throughput_week", "prov_demand_week", "prov_utilization"]].describe())


if __name__ == "__main__":
    build_wait_dataset_v2(
        sim_outputs_pattern=Path(DATA_DIR) / "simulation_output" / "*.csv",
        provider_csv=Path(DATA_DIR) / "provider.csv",
        output_csv=Path(DATA_DIR) / "ml_wait_regression_dataset_v2.csv",
        verbose=True
    )
