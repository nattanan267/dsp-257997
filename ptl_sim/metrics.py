# metrics.py
import pandas as pd
import numpy as np
from utils import haversine_distance


def compute_summary(
    df_result: pd.DataFrame,
    providers: dict,
    policy_name: str,
    use_priority: bool = False,
    replicate: int | None = None,  
    scenario_id: str | None = None 
) -> pd.DataFrame:


    if "complexity" in df_result.columns:
        df_result["complexity"] = df_result["complexity"].astype(str).str.lower()
    else:
        df_result["complexity"] = "unknown"

    if "assigned_is_nhs" not in df_result.columns:
        df_result["assigned_is_nhs"] = False
    df_result["assigned_is_nhs"] = (
        df_result["assigned_is_nhs"].astype(str).str.lower().isin(["true", "1", "yes"])
    )

    if "needs_ga" in df_result.columns:
        df_result["needs_ga"] = df_result["needs_ga"].astype(bool)
    elif "need_ga" in df_result.columns:
        df_result.rename(columns={"need_ga": "needs_ga"}, inplace=True)
        df_result["needs_ga"] = df_result["needs_ga"].astype(bool)
    else:
        df_result["needs_ga"] = False

    if "priority" not in df_result.columns:
        df_result["priority"] = 3

    if "tariff" not in df_result.columns:
        df_result["tariff"] = np.nan
    df_result["tariff"] = pd.to_numeric(df_result["tariff"], errors="coerce")

    assigned_mask = df_result["wait_time"].notnull() & (df_result["wait_time"] <= 365)

    avg_wait = float(df_result.loc[assigned_mask, "wait_time"].mean())
    max_wait = df_result.loc[assigned_mask, "wait_time"].max()
    std_wait = float(df_result.loc[assigned_mask, "wait_time"].std())

    pct_within_7 = float((df_result.loc[assigned_mask, "wait_time"] <= 7).mean() * 100)
    pct_within_30 = float((df_result.loc[assigned_mask, "wait_time"] <= 30).mean() * 100)
    pct_within_90 = float((df_result.loc[assigned_mask, "wait_time"] <= 90).mean() * 100)

    avg_wait_by_complexity = df_result.loc[assigned_mask].groupby("complexity")["wait_time"].mean().to_dict()
    avg_wait_by_type = df_result.loc[assigned_mask].groupby("assigned_is_nhs")["wait_time"].mean().to_dict()
    avg_wait_by_ga = df_result.loc[assigned_mask].groupby("needs_ga")["wait_time"].mean().to_dict()

    wait_weeks = df_result.loc[assigned_mask, "wait_time"] / 7.0
    avg_wait_w = wait_weeks.mean()
    max_wait_w = wait_weeks.max()
    std_wait_w = wait_weeks.std()

    pct_within_2w  = (df_result.loc[assigned_mask, "wait_time"] <= 14).mean()  * 100
    pct_within_4w  = (df_result.loc[assigned_mask, "wait_time"] <= 28).mean()  * 100
    pct_within_12w = (df_result.loc[assigned_mask, "wait_time"] <= 84).mean()  * 100
    travel_distances = []
    for _, row in df_result.iterrows():
        assigned = row.get("assigned_provider")
        if assigned and assigned in providers:
            p = providers[assigned]
            travel_distances.append(haversine_distance(row["lat"], row["long"], p.lat, p.long))
    avg_travel_distance = float(np.mean(travel_distances)) if travel_distances else 0.0

    utilisation_by_provider = calculate_provider_utilisation(providers)
    avg_utilisation = (
        sum(utilisation_by_provider.values()) / len(utilisation_by_provider)
        if utilisation_by_provider else 0.0
    )

    df_tmp = df_result.copy()
    df_tmp["assigned_sector"] = df_tmp["assigned_is_nhs"].map({True: "nhs", False: "private"})
    complexity_sector_dist = (
        df_tmp.loc[assigned_mask]
        .groupby(["complexity", "assigned_sector"])
        .size()
        .unstack(fill_value=0)
    )
    complexity_sector_pct = (
        complexity_sector_dist.div(complexity_sector_dist.sum(axis=1), axis=0) * 100
        if len(complexity_sector_dist) else pd.DataFrame()
    )

    df_valid = df_result[assigned_mask & df_result["tariff"].notna()]
    total_tariff_all = float(df_valid["tariff"].sum())
    total_tariff_nhs = float(df_valid[df_valid["assigned_is_nhs"]]["tariff"].sum())
    total_tariff_private = float(df_valid[~df_valid["assigned_is_nhs"]]["tariff"].sum())
    avg_tariff_all = float(df_valid["tariff"].mean()) if len(df_valid) else np.nan
    avg_tariff_nhs = float(df_valid[df_valid["assigned_is_nhs"]]["tariff"].mean()) if len(df_valid) else np.nan
    avg_tariff_private = float(df_valid[~df_valid["assigned_is_nhs"]]["tariff"].mean()) if len(df_valid) else np.nan

    summary = {
        "policy": policy_name,
        "use_priority": use_priority,
        "replicate": replicate,         
        "scenario_id": scenario_id,
        "total_patients": int(len(df_result)),
        "avg_wait": round(avg_wait, 2) if not np.isnan(avg_wait) else None,
        "max_wait": int(max_wait) if pd.notna(max_wait) else None,
        "std_wait_time": round(std_wait, 2) if not np.isnan(std_wait) else None,
        "pct_seen_within_7_days": round(pct_within_7, 2),
        "pct_seen_within_30_days": round(pct_within_30, 2),
        "pct_seen_within_90_days": round(pct_within_90, 2),
        "avg_travel_distance_km": round(avg_travel_distance, 2),
        "avg_utilisation": round(avg_utilisation * 100, 2),
        "avg_wait_low": round(float(avg_wait_by_complexity.get("low", np.nan)), 2) if "low" in avg_wait_by_complexity else None,
        "avg_wait_medium": round(float(avg_wait_by_complexity.get("medium", np.nan)), 2) if "medium" in avg_wait_by_complexity else None,
        "avg_wait_high": round(float(avg_wait_by_complexity.get("high", np.nan)), 2) if "high" in avg_wait_by_complexity else None,
        "avg_wait_nhs": round(float(avg_wait_by_type.get(True, np.nan)), 2) if True in avg_wait_by_type else None,
        "avg_wait_private": round(float(avg_wait_by_type.get(False, np.nan)), 2) if False in avg_wait_by_type else None,
        "avg_wait_ga": round(float(avg_wait_by_ga.get(True, np.nan)), 2) if True in avg_wait_by_ga else None,
        "avg_wait_non_ga": round(float(avg_wait_by_ga.get(False, np.nan)), 2) if False in avg_wait_by_ga else None,
        "avg_wait_weeks": round(avg_wait_w, 2),
        "max_wait_weeks": round(max_wait_w, 2),
        "std_wait_weeks": round(std_wait_w, 2),
        "pct_seen_within_2_weeks": round(pct_within_2w, 2),
        "pct_seen_within_4_weeks": round(pct_within_4w, 2),
        "pct_seen_within_12_weeks": round(pct_within_12w, 2),
        "num_unassigned": int(len(df_result) - assigned_mask.sum()),
        "total_tariff_all": round(total_tariff_all, 2),
        "total_tariff_nhs": round(total_tariff_nhs, 2),
        "total_tariff_private": round(total_tariff_private, 2),
        "avg_tariff_all": round(avg_tariff_all, 2) if pd.notna(avg_tariff_all) else None,
        "avg_tariff_nhs": round(avg_tariff_nhs, 2) if pd.notna(avg_tariff_nhs) else None,
        "avg_tariff_private": round(avg_tariff_private, 2) if pd.notna(avg_tariff_private) else None,
    }

    for group in ["low", "medium", "high"]:
        summary[f"pct_{group}_to_nhs"] = (
            float(complexity_sector_pct.get("nhs", {}).get(group, 0.0))
            if isinstance(complexity_sector_pct, pd.DataFrame) else 0.0
        )
        summary[f"pct_{group}_to_private"] = (
            float(complexity_sector_pct.get("private", {}).get(group, 0.0))
            if isinstance(complexity_sector_pct, pd.DataFrame) else 0.0
        )

    if "priority" in df_result.columns:
        for level in sorted(pd.unique(df_result["priority"].dropna())):
            try:
                level_mask = (df_result["priority"] == level) & assigned_mask
                summary[f"avg_wait_priority_{level}"] = round(
                    float(df_result.loc[level_mask, "wait_time"].mean()), 2
                )
                summary[f"pct_priority_{level}_seen_within_7"] = round(
                    float((df_result.loc[level_mask, "wait_time"] <= 7).mean() * 100), 2
                )
            except Exception:
                pass

    return pd.DataFrame([summary])


def calculate_provider_utilisation(providers: dict) -> dict:
    utilisation = {}
    for name, provider in providers.items():
        cap = float(getattr(provider, "minutes_per_day", 0.0)) or 0.0
        if cap <= 0 or not getattr(provider, "schedule", None):
            utilisation[name] = 0.0
            continue

        daily_utils = []
        for _, entries in provider.schedule.items():
            used = sum(float(m) for _, m in entries)
            day_util = used / cap if cap > 0 else 0.0
            day_util = min(max(day_util, 0.0), 1.0)  # clamp
            daily_utils.append(day_util)

        utilisation[name] = float(np.mean(daily_utils)) if daily_utils else 0.0

    return utilisation


def export_csv(df: pd.DataFrame, path):
    df.to_csv(path, index=False)


def compute_realdata_summary(df_patient: pd.DataFrame, df_providers: pd.DataFrame) -> pd.DataFrame:
    df = df_patient.copy()
    df = df[df["waiting_time"].notnull()].copy()

    df["wait_time"] = pd.to_numeric(df["waiting_time"], errors="coerce")
    df["assigned_provider"] = df["provider"].astype(str)
    df["assigned_is_nhs"] = df["is_nhs"].astype(str).str.lower().isin(["true", "1", "yes"])
    df["needs_ga"] = (
        df["needs_ga"].astype(bool) if "needs_ga" in df.columns
        else (df["need_ga"].astype(bool) if "need_ga" in df.columns else False)
    )
    df["patient_id"] = df["local_patient_identifier"].astype(str)
    df["arrival_time"] = pd.to_datetime(df["start_clock_date"], errors="coerce", dayfirst=True,).dt.strftime("%Y-%m-%d")
    df["complexity"] = df.get("complexity", "Medium")

    from provider import Provider
    providers_dict = {
        str(row["provider"]): Provider(
            name=str(row["provider"]),
            lat=float(row.get("lat", 0.0)),
            long=float(row.get("long", 0.0)),
            is_nhs=row.get("is_nhs", True),
            minutes_per_day=0.0, 
        )
        for _, row in df_providers.iterrows()
    }

    return compute_summary(df, providers=providers_dict, policy_name="RealData", use_priority=False)
