from pathlib import Path

SIM_CONFIG = {
    "n_patients": 1000,
    "current_time": 0,
    "ga_rate": 0.02,
    "rejection_prob_by_complexity": {
        "High": 0.6,
        "Low": 0.0
    },
    "training_overhead_nhs": 1.5,
    "random_seed": 42
}

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"