import pandas as pd
import numpy as np
from pathlib import Path
from utils import haversine_distance  # ยัง import ไว้เผื่อใช้ที่อื่น
from config import DATA_DIR

def _haversine_vec(lat1, lon1, lat2, lon2, radius_km: float = 6371.0):
    lat1 = np.radians(np.asarray(lat1, dtype=float))
    lon1 = np.radians(np.asarray(lon1, dtype=float))
    lat2 = np.radians(np.asarray(lat2, dtype=float))
    lon2 = np.radians(np.asarray(lon2, dtype=float))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return radius_km * (2.0 * np.arcsin(np.sqrt(a)))

def build_wait_regression_dataset(sim_outputs_pattern, providers_csv, out_csv):
    sim_outputs_pattern = Path(sim_outputs_pattern)
    providers_csv = Path(providers_csv)
    out_csv = Path(out_csv)

    df_prov = pd.read_csv(providers_csv)
    df_prov["is_nhs"] = df_prov["is_nhs"].astype(str).str.lower() == "true"
    if "SlotsPerDay" in df_prov.columns:
        df_prov["minutes_per_day"] = np.where(
            df_prov["is_nhs"], df_prov["SlotsPerDay"] * 20, df_prov["SlotsPerDay"] * 15
        )
    else:
        df_prov["minutes_per_day"] = 8 * 60

    file_dir = sim_outputs_pattern.parent
    file_pat = sim_outputs_pattern.name
    csv_paths = sorted(file_dir.glob(file_pat))

    all_rows = []
    for path in csv_paths:
        df = pd.read_csv(path)
        use = df[df["wait_time"].notnull()].copy()

        use = use.merge(
            df_prov[["provider", "lat", "long", "is_nhs", "minutes_per_day"]],
            left_on="assigned_provider",
            right_on="provider",
            how="left",
            suffixes=("", "_prov")
        )

        comp_map = {"Low": 0, "Medium": 1, "High": 2}
        use["complexity_enc"] = use["complexity"].map(comp_map)
        use["need_ga"] = use["need_ga"].astype(int)
        use["priority"] = pd.to_numeric(use["priority"], errors="coerce").fillna(3).astype(int)
        use["tariff"] = pd.to_numeric(use["tariff"], errors="coerce").fillna(0.0)

        # แปลงให้เป็นตัวเลขและตัดแถวที่ไม่มีพิกัด
        for col in ["lat", "long", "lat_prov", "long_prov"]:
            use[col] = pd.to_numeric(use[col], errors="coerce")
        use = use.dropna(subset=["lat", "long", "lat_prov", "long_prov"])

        # คำนวณระยะทางแบบเวกเตอร์
        use["distance_km"] = _haversine_vec(
            use["lat"], use["long"], use["lat_prov"], use["long_prov"]
        )

        out = use[[
            "need_ga", "priority", "complexity_enc", "tariff",
            "lat", "long",                  # patient coords
            "lat_prov", "long_prov",        # provider coords
            "is_nhs", "minutes_per_day",
            "distance_km",
            "wait_time",
            "assigned_provider"
        ]].rename(columns={"lat": "patient_lat", "long": "patient_long"})

        all_rows.append(out)

    if not all_rows:
        raise ValueError("Not found simulation outputs")

    final_df = pd.concat(all_rows, ignore_index=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(out_csv, index=False)
    print(f"✅ Saved regression dataset to {out_csv} with shape {final_df.shape}")

if __name__ == "__main__":
    build_wait_regression_dataset(
        sim_outputs_pattern= Path(DATA_DIR) / "simulation_output" / "*.csv",
        providers_csv= Path(DATA_DIR) / "provider_lat_long.csv",
        out_csv= Path(DATA_DIR) / "ml_wait_regression_dataset.csv",
    )
