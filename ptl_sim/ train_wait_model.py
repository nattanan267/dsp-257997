# train_wait_model_hgbr.py
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from config import DATA_DIR
from ml_utils import FEATURE_COLS, TARGET_COL, _coerce_and_fix_columns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr, kendalltau

if __name__ == "__main__":
    CSV = Path(DATA_DIR) / "ml_wait_regression_dataset_v2.csv"
    OUT = Path(DATA_DIR) / "models" / "ml_wait_regressor_v2_hgbr.pkl"

    df = pd.read_csv(CSV, low_memory=False)
    df = _coerce_and_fix_columns(df)

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    # log1p target
    y_log = np.log1p(y.clip(lower=0))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_log, test_size=0.2, random_state=42
    )

    model = HistGradientBoostingRegressor(
        random_state=42,
        max_depth=None,
        learning_rate=0.05,
        max_iter=500,
        l2_regularization=0.0,
        early_stopping=True,
        validation_fraction=0.1,
    )
    model.fit(X_train, y_train)

    # predict & revert
    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)

    # metrics
    sp_corr, sp_p = spearmanr(np.expm1(y_test), y_pred)
    kd_tau, kd_p = kendalltau(np.expm1(y_test), y_pred, variant="b")
    mae = mean_absolute_error(np.expm1(y_test), y_pred)
    rmse = np.sqrt(mean_squared_error(np.expm1(y_test), y_pred))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, OUT)

    print(f"✅ Model saved to {OUT}")
    print(f"Spearman ρ: {sp_corr:.3f} (p={sp_p:.1e})")
    print(f"Kendall τ : {kd_tau:.3f} (p={kd_p:.1e})")
    print(f"MAE: {mae:.3f} | RMSE: {rmse:.3f}")
