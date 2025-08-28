import pandas as pd
from datetime import datetime, timedelta
from math import radians, sin, cos, sqrt, atan2
import numpy as np
from typing import List, Dict
from patient import Patient
from provider import Provider
from tqdm import tqdm
import random
import datetime as dt


def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


def seed_backlog(
    providers: dict,
    occupancy_low: float = 0.45,
    occupancy_high: float = 0.75,
    days: int = 30,
    start_date: str = "2024-04-01"
):
    """
    Pre-fill backlog ที่ realistic: ใช้สัดส่วนของ minutes_per_day
    และไม่ให้เกิน capacity ของวันนั้น ๆ
    """
    start = dt.datetime.strptime(start_date, "%Y-%m-%d").date()
    for p in providers.values():
        cap = float(getattr(p, "minutes_per_day", 0.0)) or 0.0
        if cap <= 0:
            continue
        for d in range(days):
            date_str = (start + dt.timedelta(days=d)).strftime("%Y-%m-%d")
            frac = random.uniform(occupancy_low, occupancy_high)
            used = int(min(cap, max(0.0, frac * cap)))
            if used > 0:
                p.schedule.setdefault(date_str, []).append(("backlog", used))


def get_week_str(date_str: str) -> str:
    dt_ = datetime.strptime(date_str, "%Y-%m-%d")
    return f"{dt_.isocalendar().year}-{dt_.isocalendar().week:02d}"


def estimate_slots_per_day_from_patient_data(df_patient):
    df = df_patient.copy()
    daily_counts = (
        df.groupby(['provider', 'end_clock_date', 'complexity'])
        .size()
        .reset_index(name='daily_cases')
    )
    median_per_complexity = (
        daily_counts.groupby(['provider', 'complexity'])['daily_cases']
        .median()
        .reset_index()
    )
    slots_wide = median_per_complexity.pivot(index='provider', columns='complexity', values='daily_cases')
    slots_wide = slots_wide.fillna(0)
    slots_wide['SlotsPerDay'] = slots_wide.sum(axis=1)
    slots_wide['SlotsPerWeek'] = slots_wide['SlotsPerDay'] * 5
    slots_wide['SlotsPerYear'] = slots_wide['SlotsPerWeek'] * 52
    return slots_wide[['SlotsPerDay', 'SlotsPerWeek', 'SlotsPerYear']].reset_index()


def simulate(patients: List["Patient"], providers: Dict[str, "Provider"], wait_time_model=None) -> List["Patient"]:
    """
    ใช้เฉพาะกรณี “policy ไม่ได้จองจริง”
    เพราะฟังก์ชันนี้จะ clear_schedule() แล้วจองใหม่ทั้งหมด
    """
    for provider in providers.values():
        provider.clear_schedule()

    patients.sort(key=lambda p: p.arrival_time)

    for patient in tqdm(patients, desc="Simulating", unit="patient"):
        provider_name = getattr(patient, "assigned_provider", None)
        if not provider_name:
            continue

        provider = providers.get(provider_name)
        if not provider:
            continue

        arrival = datetime.strptime(patient.arrival_time, "%Y-%m-%d")
        max_days = 365
        assigned = False

        # พยายามจองกับ provider ที่เลือกก่อน
        for delta in range(max_days):
            assign_date = arrival + timedelta(days=delta)
            date_str = assign_date.strftime("%Y-%m-%d")
            if provider.can_accept(patient, date_str):
                provider.assign_patient(patient, date_str)
                patient.wait_time = delta
                assigned = True
                break

        # (optional) fallback ด้วยโมเดล
        if (not assigned) and wait_time_model and ("get_fallback_provider" in globals()):
            try:
                fallback_provider, fallback_delta = get_fallback_provider(patient, providers, wait_time_model)  # noqa: F821
            except Exception:
                fallback_provider, fallback_delta = (None, None)

            if fallback_provider and isinstance(fallback_delta, (int, float)) and fallback_delta >= 0:
                assign_date = arrival + timedelta(days=int(fallback_delta))
                date_str = assign_date.strftime("%Y-%m-%d")
                if fallback_provider.can_accept(patient, date_str):
                    fallback_provider.assign_patient(patient, date_str)
                    patient.assigned_provider = fallback_provider.name
                    patient.wait_time = int(fallback_delta)
                    patient.was_fallback = True
                    assigned = True

        if not assigned:
            patient.wait_time = None

    return patients


def get_fallback_provider(patient, providers, wait_time_model, max_days=365):
    fallback_candidates = []
    original_provider = patient.assigned_provider
    arrival = datetime.strptime(patient.arrival_time, "%Y-%m-%d")

    for name, provider in providers.items():
        if name == original_provider:
            continue
        if patient.need_ga and str(provider.is_nhs).lower() != "true":
            continue

        for delta in range(max_days):
            date_str = (arrival + timedelta(days=delta)).strftime("%Y-%m-%d")
            if provider.can_accept(patient, date_str):
                dist = haversine_distance(patient.lat, patient.long, provider.lat, provider.long)
                complexity = {"low": 0, "medium": 1, "high": 2}.get(patient.complexity.lower(), 1)
                features = pd.DataFrame([{
                    "complexity": complexity,
                    "needs_ga": int(patient.need_ga),
                    "distance_to_provider": dist
                }])
                est_wait = wait_time_model.predict(features)[0]
                fallback_candidates.append((provider, delta, est_wait))
                break

    if not fallback_candidates:
        return None, None

    fallback_candidates.sort(key=lambda x: (x[1], x[2]))
    return fallback_candidates[0][0], fallback_candidates[0][1]

def _is_nhs(provider) -> bool:
    return str(getattr(provider, "is_nhs", "false")).lower() == "true"