import pandas as pd
from tqdm import tqdm

from patient import Patient
from provider import Provider
from metrics import compute_summary
from utils import simulate, seed_backlog
from config import DATA_DIR


# — ปรับ capacity สมจริงขึ้น (จูนได้)
CAPACITY_SCALE = 1       # ลดความจุรวมของระบบลงให้เกิดคิว (0.5–0.8)
AVG_MIN_NHS = 35           # นาทีต่อสล็อตของ NHS
AVG_MIN_PRIVATE = 25       # นาทีต่อสล็อตของเอกชน


def _to_date_iso(s):
    """แปลงสตริงวันที่ให้เป็น YYYY-MM-DD (รับ dayfirst=True)"""
    dt = pd.to_datetime(s, dayfirst=True, errors="coerce")
    return None if pd.isna(dt) else dt.date().isoformat()


def _to_bool_s(val) -> bool:
    """รับค่าหลากหลายรูปแบบ → bool"""
    if isinstance(val, (int, float)):
        return bool(int(val))
    if isinstance(val, str):
        return val.strip().lower() in ("true", "1", "yes", "y", "t")
    return bool(val)


def run_policy_scenario(
    df_patients, df_providers, policy_fn, policy_name="Baseline",
    use_priority=False, simulate_after_policy=False, wait_time_model=None
):
    """
    รัน policy หนึ่งรายการ:
      - สร้าง Patient/Provider objects
      - seed backlog ให้ทุก provider (สำคัญ!)
      - เรียก policy (นโยบายจะจองจริง → ตั้ง wait_time)
      - (ถ้าจำเป็น) simulate หลัง policy
      - สร้าง df_result และ summary
    """
    patients = []
    for _, row in df_patients.iterrows():
        arr_iso = _to_date_iso(row.get("start_clock_date"))
        if arr_iso is None:
            continue
        patient = Patient(
            patient_id=str(row.get("local_patient_identifier")),
            arrival_time=arr_iso,
            complexity=str(row.get("complexity", "Medium")),
            lat=float(row.get("lat", 0.0)),
            long=float(row.get("long", 0.0)),
            original_provider=str(row.get("provider")),
            need_ga=_to_bool_s(row.get("needs_ga", False)),
            priority=int(row.get("priority", 3)),
            hrg_code=row.get("hrg_code"),
            tariff=float(row.get("tariff", 0.0)) if row.get("tariff") is not None else None
        )
        patients.append(patient)

    providers = {}
    for _, row in df_providers.iterrows():
        name = str(row["provider"])
        slots_per_day = float(row.get("SlotsPerDay", 0))
        is_nhs_str = str(row.get("is_nhs", "true")).lower()
        is_nhs_bool = (is_nhs_str == "true")

        avg_minutes = AVG_MIN_NHS if is_nhs_bool else AVG_MIN_PRIVATE
        minutes_per_day = slots_per_day * avg_minutes * CAPACITY_SCALE

        providers[name] = Provider(
            name=name,
            lat=float(row.get("lat", 0.0)),
            long=float(row.get("long", 0.0)),
            is_nhs=str(row.get("is_nhs", "true")),  
            minutes_per_day=float(minutes_per_day)
        )

    start_dates = pd.to_datetime(df_patients["start_clock_date"], dayfirst=True, errors="coerce").dropna()
    
    seed_backlog(providers, occupancy_low=0.05, occupancy_high=0.1, days=30, start_date="2024-04-01")

    demand_per_day = float(start_dates.dt.date.value_counts().mean()) if not start_dates.empty else 0.0
    total_minutes_per_day = sum(p.minutes_per_day for p in providers.values())
    equiv_cases_per_day = total_minutes_per_day / 60.0
    print(f"[SANITY] demand/day ≈ {demand_per_day:.1f}, capacity/day ≈ {equiv_cases_per_day:.1f} (60-min eq)")

    patients = policy_fn(patients, providers, use_priority=use_priority)

    if simulate_after_policy:
        patients = simulate(patients, providers, wait_time_model=wait_time_model)

    rows = []
    for p in patients:
        assigned_name = getattr(p, "assigned_provider", None)
        assigned_is_nhs = None
        if assigned_name in providers:
            assigned_is_nhs = (str(providers[assigned_name].is_nhs).lower() == "true")
        rows.append({
            "patient_id": p.patient_id,
            "arrival_time": p.arrival_time,
            "complexity": p.complexity,
            "need_ga": bool(getattr(p, "need_ga", False)),
            "priority": getattr(p, "priority", None),
            "original_provider": p.original_provider,
            "assigned_provider": assigned_name,
            "assigned_is_nhs": assigned_is_nhs,
            "wait_time": p.wait_time,
            "lat": p.lat,
            "long": p.long,
            "was_fallback": getattr(p, "was_fallback", False),
            "tariff": getattr(p, "tariff", None),
            "has_tariff": getattr(p, "has_tariff", None),
        })
    df_result = pd.DataFrame(rows)

    summary = compute_summary(df_result, providers, policy_name, use_priority=use_priority)

    null_waits = df_result["wait_time"].isna().sum()
    if null_waits:
        print(f"⚠️ {null_waits} patients have no wait_time in policy '{policy_name}'")

    return df_result, summary
