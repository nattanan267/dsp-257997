# policy_ml_wait_plus_distance_balanced.py
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np
from utils import haversine_distance
from ml_utils import load_wait_regressor, make_feature_row

def ml_wait_plus_distance_balanced_policy(
    patients, providers, model_path: str, beta: float = 1.0, use_priority: bool = False
):
    model = load_wait_regressor(model_path)

    total_by_complexity = defaultdict(int)
    for p in patients:
        total_by_complexity[p.complexity] += 1

    target_quota = defaultdict(int)
    for comp in ["Low", "Medium", "High"]:
        half = total_by_complexity[comp] // 2
        target_quota[("NHS", comp)] = half
        target_quota[("Private", comp)] = total_by_complexity[comp] - half

    if use_priority:
        patients.sort(key=lambda p: (p.arrival_time, p.priority))
    else:
        patients.sort(key=lambda p: p.arrival_time)

    provider_list = list(providers.values())

    for patient in patients:
        nhs_only = bool(getattr(patient, "need_ga", False))

        assigned = False
        arrival_dt = datetime.strptime(patient.arrival_time, "%Y-%m-%d")

        for delta in range(365):
            day = arrival_dt + timedelta(days=delta)
            date_str = day.strftime("%Y-%m-%d")

            candidates = [
                prov for prov in provider_list
                if (str(prov.is_nhs).lower() == "true" or not nhs_only) and prov.can_accept(patient, date_str)
            ]
            if not candidates:
                continue

            scored = []
            for prov in candidates:
                # ตรวจ quota ก่อนคำนวณคะแนน (กันแต้มสวยแต่โควต้าเต็ม)
                grp = "NHS" if str(prov.is_nhs).lower() == "true" else "Private"
                if target_quota[(grp, patient.complexity)] <= 0:
                    continue

                dist = haversine_distance
