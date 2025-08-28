import pandas as pd
import numpy as np
import joblib
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr, kendalltau

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
TARGET_COL = "wait_time"


def _coerce_and_fix_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "need_ga" not in df.columns and "needs_ga" in df.columns:
        df = df.rename(columns={"needs_ga": "need_ga"})
    for c in FEATURE_COLS + [TARGET_COL]:
        if c not in df.columns:
            df[c] = np.nan

    int_like = ["need_ga", "priority", "complexity_enc", "is_nhs"]
    float_like = ["tariff", "distance_km", "prov_throughput_week", "prov_demand_week",
                  "prov_utilization", "patient_lat","patient_long","lat_prov","long_prov", TARGET_COL]
    for c in int_like:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in float_like:
        df[c] = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)

    df = df.dropna(subset=[TARGET_COL])
    for c in int_like:
        df[c] = df[c].fillna(0).astype(int)

    def _safe_fill_median(s, fallback=0.0):
        return s.fillna(s.median()) if s.notna().any() else s.fillna(fallback)

    for c in ["tariff","distance_km","prov_throughput_week","prov_demand_week",
              "prov_utilization","patient_lat","patient_long","lat_prov","long_prov"]:
        df[c] = _safe_fill_median(df[c], fallback=0.0)

    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0)
    return df


def train_wait_regressor(dataset_csv: str, model_path: str, random_state: int = 42):
    df = pd.read_csv(dataset_csv)
    df = _coerce_and_fix_columns(df)
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    model = RandomForestRegressor(n_estimators=300, random_state=random_state, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    sp_corr, sp_p = spearmanr(y_test, y_pred)
    kd_tau, kd_p = kendalltau(y_test, y_pred, variant="b")
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    print(f"Spearman ρ: {sp_corr:.3f} (p={sp_p:.1e}) | Kendall τ: {kd_tau:.3f} (p={kd_p:.1e})")
    print(f"MAE: {mae:.3f} | RMSE: {rmse:.3f}")


def load_wait_regressor(model_path: str):
    return joblib.load(model_path)
