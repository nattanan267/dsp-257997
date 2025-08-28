# eval_rank.py
import pandas as pd
from pathlib import Path
from config import DATA_DIR
from ml_utils import load_wait_regressor, rank_eval_hit_rate_at_k

if __name__ == "__main__":
    MODEL = Path(DATA_DIR) / "models" / "ml_wait_regressor.pkl"
    CAND  = Path(DATA_DIR) / "ml_rank_candidates.csv"

    df = pd.read_csv(CAND)
    model = load_wait_regressor(MODEL)


    hit1, mrr1 = rank_eval_hit_rate_at_k(model, df, k=1,
                                         case_col="case_id",
                                         provider_col="provider",
                                         actual_wait_col="actual_wait")
    print(f"Hit@1: {hit1:.3f} | MRR@1: {mrr1:.3f}")


    hit3, mrr3 = rank_eval_hit_rate_at_k(model, df, k=3,
                                         case_col="case_id",
                                         provider_col="provider",
                                         actual_wait_col="actual_wait")
    print(f"Hit@3: {hit3:.3f} | MRR@3: {mrr3:.3f}")
