# policy.py
import random
from collections import defaultdict, Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import haversine_distance, seed_backlog
from patient import Patient
from provider import Provider
from geo_utils import get_service_time
from ml_utils import load_wait_regressor, FEATURE_COLS
from config import DATA_DIR

def provider_accepts_strict(patient: Patient, provider: Provider) -> bool:
    """
    Strict rule:
      - needs_ga -> NHS เท่านั้น
      - ไม่ต้อง GA -> NHS/Private ได้ทั้งคู่
    """
    is_nhs = str(getattr(provider, "is_nhs", "false")).lower() == "true"
    if getattr(patient, "need_ga", False) and not is_nhs:
        return False
    return True

def _zscore_safe_vec(v):
    v = np.asarray(pd.to_numeric(v, errors="coerce"), dtype=float)
    mu = np.nanmean(v) if np.isfinite(np.nanmean(v)) else 0.0
    sd = np.nanstd(v) if np.isfinite(np.nanstd(v)) and np.nanstd(v) > 0 else 1.0
    return (v - mu) / sd


def _is_nhs(provider: Provider) -> bool:
    return str(getattr(provider, "is_nhs", "false")).lower() == "true"


def _is_nhs_int(provider) -> int:
    return 1 if _is_nhs(provider) else 0


def _complexity_enc(patient) -> int:
    return {"Low": 0, "Medium": 1, "High": 2}.get(getattr(patient, "complexity", "Medium"), 1)


def _slot_minutes(is_nhs_int: int) -> float:
    return 20.0 if is_nhs_int == 1 else 15.0


def _provider_static(provider: Provider):
    is_nhs = _is_nhs_int(provider)
    slot_min = _slot_minutes(is_nhs)
    minutes_per_day = float(getattr(provider, "minutes_per_day", 8 * 60))
    slots_per_day = max(1.0, minutes_per_day / slot_min)
    throughput_week = slots_per_day * 5.0
    return {
        "is_nhs": is_nhs,
        "slot_min": slot_min,
        "minutes_per_day": minutes_per_day,
        "slots_per_day": slots_per_day,
        "throughput_week": throughput_week,
    }


def _build_feature_rows_for_candidates(
    patient: Patient, candidate_providers: List[Provider], date_str: str
) -> pd.DataFrame:
    rows = []
    plat = float(getattr(patient, "lat"))
    plon = float(getattr(patient, "long"))
    need_ga = int(getattr(patient, "need_ga", False))
    priority = int(getattr(patient, "priority", 3))
    comp_enc = _complexity_enc(patient)
    tariff = float(getattr(patient, "tariff", 0.0))
    

    for prov in candidate_providers:
        stat = _provider_static(prov)
        dist_km = float(haversine_distance(plat, plon, float(prov.lat), float(prov.long)))
        try:
            qlen_today = float(prov.get_queue_length(date_str))
        except Exception:
            qlen_today = 0.0
        util_day = qlen_today / max(1.0, stat["slots_per_day"])
        util_day = float(np.clip(util_day, 0.0, 2.0))
        prov_demand_week = util_day * stat["throughput_week"]

        rows.append({
            "need_ga": need_ga,
            "priority": priority,
            "complexity_enc": comp_enc,
            "tariff": tariff,
            "patient_lat": plat,
            "patient_long": plon,
            "lat_prov": float(prov.lat),
            "long_prov": float(prov.long),
            "is_nhs": stat["is_nhs"],
            "distance_km": dist_km,
            "prov_throughput_week": stat["throughput_week"],
            "prov_demand_week": prov_demand_week,
            "prov_utilization": util_day,
            "_provider_obj": prov,
        })

    df = pd.DataFrame(rows)
    for c in FEATURE_COLS:
        if c not in df.columns:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df



def baseline_policy(
    patients: List[Patient],
    providers: Dict[str, Provider],
    use_priority: bool = False,
    high_to_private_share: float = 0.10,
    fallback_after_days: int = 60,   
    max_days_forward: int = 365,
) -> List[Patient]:
    def _is_nhs(pr: Provider) -> bool:
        return str(getattr(pr, "is_nhs", "false")).lower() == "true"

    total_high = sum(
        1 for p in patients
        if str(getattr(p, "complexity", "")) == "High" and not getattr(p, "need_ga", False)
    )
    allow_private_high_quota = int(round(total_high * max(0.0, min(1.0, high_to_private_share))))
    private_high_used = 0

    patients.sort(key=lambda p: (p.arrival_time, p.priority) if use_priority else p.arrival_time)

    for patient in tqdm(patients, desc="Baseline Policy (NHS-first + grace)"):
        arrival = datetime.strptime(patient.arrival_time, "%Y-%m-%d")
        comp = str(getattr(patient, "complexity", "Medium"))
        needs_ga = bool(getattr(patient, "need_ga", False))
        is_high = (comp == "High")

        assigned = False

        for delta in range(min(fallback_after_days, max_days_forward)):
            date_str = (arrival + timedelta(days=delta)).strftime("%Y-%m-%d")
            nhs_elig = [pr for pr in providers.values() if _is_nhs(pr) and pr.can_accept(patient, date_str)]
            if nhs_elig:
                nearest = min(nhs_elig, key=lambda pr: haversine_distance(pr.lat, pr.long, patient.lat, patient.long))
                nearest.assign_patient(patient, date_str)
                assigned = True
                break

        if assigned:
            continue

        for delta in range(fallback_after_days, max_days_forward):
            date_str = (arrival + timedelta(days=delta)).strftime("%Y-%m-%d")

            if needs_ga:
                nhs_elig = [pr for pr in providers.values() if _is_nhs(pr) and pr.can_accept(patient, date_str)]
                if nhs_elig:
                    nearest = min(nhs_elig, key=lambda pr: haversine_distance(pr.lat, pr.long, patient.lat, patient.long))
                    nearest.assign_patient(patient, date_str)
                    assigned = True
                break

            nhs_elig = [pr for pr in providers.values() if _is_nhs(pr) and pr.can_accept(patient, date_str)]
            priv_elig = [pr for pr in providers.values() if not _is_nhs(pr) and pr.can_accept(patient, date_str)]

            if nhs_elig:
                nearest = min(nhs_elig, key=lambda pr: haversine_distance(pr.lat, pr.long, patient.lat, patient.long))
                nearest.assign_patient(patient, date_str)
                assigned = True
                break

            if comp in ("Low", "Medium"):
                if priv_elig:
                    nearest = min(priv_elig, key=lambda pr: haversine_distance(pr.lat, pr.long, patient.lat, patient.long))
                    nearest.assign_patient(patient, date_str)
                    assigned = True
                    break
            else:  
                if private_high_used < allow_private_high_quota and priv_elig:
                    nearest = min(priv_elig, key=lambda pr: haversine_distance(pr.lat, pr.long, patient.lat, patient.long))
                    nearest.assign_patient(patient, date_str)
                    private_high_used += 1
                    assigned = True
                    break

    return patients


def greedy_policy(
    patients: List[Patient],
    providers: Dict[str, Provider],
    use_priority: bool = False,
    max_days_forward: int = 365,
) -> List[Patient]:
    patients.sort(key=lambda p: (p.arrival_time, p.priority) if use_priority else p.arrival_time)

    for patient in tqdm(patients, desc="Fastest First (pure earliest)"):
        arrival = datetime.strptime(patient.arrival_time, "%Y-%m-%d")
        needs_ga = bool(getattr(patient, "need_ga", False))

        best = None  

        for prov in providers.values():
            if needs_ga and str(getattr(prov, "is_nhs", "false")).lower() != "true":
                continue

            for delta in range(max_days_forward):
                date_str = (arrival + timedelta(days=delta)).strftime("%Y-%m-%d")
                if prov.can_accept(patient, date_str):
                    if best is None or delta < best[0]:
                        best = (delta, prov, date_str)
                    break

        if best is not None:
            _, prov, dstr = best
            prov.assign_patient(patient, dstr)

    return patients

def shared_ptl_equal_policy(
    patients: List[Patient],
    providers: Dict[str, Provider],
    use_priority: bool = False
) -> List[Patient]:
    assigned_count = Counter({name: 0 for name in providers.keys()})

    def _pick_provider(cands: List[Provider], patient: Patient, date_str: str) -> Optional[Provider]:
        if not cands:
            return None
        ordered = sorted(
            cands,
            key=lambda pr: (assigned_count[pr.name], haversine_distance(pr.lat, pr.long, patient.lat, patient.long))
        )
        for pr in ordered:
            if pr.can_accept(patient, date_str):
                return pr
        return None

    patients.sort(key=lambda p: (p.arrival_time, p.priority) if use_priority else p.arrival_time)

    for patient in tqdm(patients, desc="Shared PTL Equal"):
        arrival = datetime.strptime(patient.arrival_time, "%Y-%m-%d")
        for delta in range(365):
            date_str = (arrival + timedelta(days=delta)).strftime("%Y-%m-%d")
            candidates = [pr for pr in providers.values() if provider_accepts_strict(patient, pr)]
            chosen = _pick_provider(candidates, patient, date_str)
            if chosen:
                chosen.assign_patient(patient, date_str)
                assigned_count[chosen.name] += 1
                break

    return patients


def _earliest_day_with_distance_tiebreak(
    prov_list: List["Provider"],
    p: "Patient",
    start_dt: datetime,
    max_days_forward: int,
) -> Tuple[int, Optional[str], Optional["Provider"]]:
    best_delta = 10**9
    best_date: Optional[str] = None
    best_prov: Optional["Provider"] = None

    for d in range(max_days_forward):
        ds = (start_dt + timedelta(days=d)).strftime("%Y-%m-%d")
        same_day = [pr for pr in prov_list if pr.can_accept(p, ds)]
        if not same_day:
            continue
        pr_best = min(same_day, key=lambda pr: haversine_distance(pr.lat, pr.long, p.lat, p.long))
        best_delta, best_date, best_prov = d, ds, pr_best
        break
    return best_delta, best_date, best_prov


def complexity_balanced_policy(
    patients: List["Patient"],
    providers: Dict[str, "Provider"],
    use_priority: bool = False,
    max_days_forward: int = 365,
    slack_days: int = 14,
    strict_final_balance: bool = False,
) -> List["Patient"]:

    nhs_provs = [pr for pr in providers.values() if _is_nhs(pr)]
    pri_provs = [pr for pr in providers.values() if not _is_nhs(pr)]

    def _ord(lst): lst.sort(key=lambda x: (x.arrival_time, x.priority) if use_priority else x.arrival_time)

    non_ga = [p for p in patients if not getattr(p, "need_ga", False)]
    _ord(non_ga)

    by_comp: Dict[str, List["Patient"]] = {"Low": [], "Medium": [], "High": []}
    for p in non_ga:
        by_comp[str(getattr(p, "complexity", "Medium"))].append(p)

    targets: Dict[str, Dict[str, int]] = {}
    for comp, lst in by_comp.items():
        n = len(lst)
        priv = n // 2
        nhs = n - priv
        targets[comp] = {"private": priv, "nhs": nhs}

    remain: Dict[str, Dict[str, int]] = {c: dict(targets[c]) for c in targets}

    all_in_order = sorted(patients, key=lambda p: (p.arrival_time, p.priority) if use_priority else p.arrival_time)

    for p in tqdm(all_in_order, desc="Complexity Balanced (one-pass)"):
        start = datetime.strptime(p.arrival_time, "%Y-%m-%d")

        if getattr(p, "need_ga", False):
            d_nhs, ds_nhs, pr_nhs = _earliest_day_with_distance_tiebreak(nhs_provs, p, start, max_days_forward)
            if pr_nhs is not None:
                pr_nhs.assign_patient(p, ds_nhs)
            continue

        comp = str(getattr(p, "complexity", "Medium"))

        d_priv, ds_priv, pr_priv = _earliest_day_with_distance_tiebreak(pri_provs, p, start, max_days_forward)
        d_nhs,  ds_nhs,  pr_nhs  = _earliest_day_with_distance_tiebreak(nhs_provs, p, start, max_days_forward)

        if pr_priv is None and pr_nhs is None:
            continue  

        if pr_priv is None:
            if not strict_final_balance or remain[comp]["nhs"] > 0:
                pr_nhs.assign_patient(p, ds_nhs); remain[comp]["nhs"] -= 1
            continue
        if pr_nhs is None:
            if not strict_final_balance or remain[comp]["private"] > 0:
                pr_priv.assign_patient(p, ds_priv); remain[comp]["private"] -= 1
            continue

        want = "private" if remain[comp]["private"] > remain[comp]["nhs"] else "nhs"

        if strict_final_balance:
            if want == "private" and remain[comp]["private"] <= 0: want = "nhs"
            elif want == "nhs" and remain[comp]["nhs"] <= 0:       want = "private"

        if want == "private" and (d_nhs + slack_days < d_priv): want = "nhs"
        elif want == "nhs" and (d_priv + slack_days < d_nhs):    want = "private"

        if want == "private":
            pr_priv.assign_patient(p, ds_priv); remain[comp]["private"] -= 1
        else:
            pr_nhs.assign_patient(p, ds_nhs);   remain[comp]["nhs"] -= 1

    return patients


COST_PER_MINUTE = {"Low": 6, "Medium": 10, "High": 16}

def _profit_per_minute(patient: Patient, provider: Provider) -> float:
    try:
        service_time = get_service_time(patient, provider)
        if service_time <= 0:
            return -1e9
    except Exception:
        return -1e9
    cpm = COST_PER_MINUTE.get(getattr(patient, "complexity", "Medium"), 10)
    revenue = float(getattr(patient, "tariff", 0.0) or 0.0)
    profit = revenue - cpm * service_time
    return profit / service_time


def fee_biased_policy(
    patients: List[Patient],
    providers: Dict[str, Provider],
    use_priority: bool = False,
    min_margin_per_minute: float = 0.5,
) -> List[Patient]:
    patients.sort(
        key=lambda p: (-float(p.tariff or 0), p.arrival_time, p.priority) if use_priority
        else (-float(p.tariff or 0), p.arrival_time)
    )

    for patient in tqdm(patients, desc="Fee-Biased (private prefers low/medium)"):
        arrival = datetime.strptime(patient.arrival_time, "%Y-%m-%d")
        needs_ga = bool(getattr(patient, "need_ga", False))
        for delta in range(365):
            date_str = (arrival + timedelta(days=delta)).strftime("%Y-%m-%d")
            candidates = [pr for pr in providers.values()
                          if provider_accepts_strict(patient, pr) and pr.can_accept(patient, date_str)]
            if not candidates:
                continue

            if not needs_ga:
                privs = [pr for pr in candidates if not _is_nhs(pr)]
                nhss  = [pr for pr in candidates if _is_nhs(pr)]

                profitable_privs: List[Tuple[Provider, float]] = []
                for pr in privs:
                    ppm = _profit_per_minute(patient, pr)
                    if ppm >= min_margin_per_minute:
                        bonus = 0.2 if str(getattr(patient, "complexity", "Medium")) in ("Low", "Medium") else 0.0
                        profitable_privs.append((pr, ppm + bonus))

                if profitable_privs:
                    profitable_privs.sort(key=lambda x: (-x[1], haversine_distance(
                        x[0].lat, x[0].long, patient.lat, patient.long)))
                    best_priv = profitable_privs[0][0]
                    best_priv.assign_patient(patient, date_str)
                    break
                else:
                    if nhss:
                        nearest_nhs = min(nhss, key=lambda pr: haversine_distance(
                            pr.lat, pr.long, patient.lat, patient.long))
                        nearest_nhs.assign_patient(patient, date_str)
                        break
            else:
                nhs_only = [pr for pr in candidates if _is_nhs(pr)]
                if nhs_only:
                    nearest = min(nhs_only, key=lambda pr: haversine_distance(
                        pr.lat, pr.long, patient.lat, patient.long))
                    nearest.assign_patient(patient, date_str)
                    break

    return patients


def neutral_fee_policy(
    patients: List[Patient],
    providers: Dict[str, Provider],
    use_priority: bool = False
) -> List[Patient]:
    """กลางจริง: strict GA rule + ใกล้ที่สุดในวันแรกที่ว่าง"""
    patients.sort(key=lambda p: (p.arrival_time, p.priority) if use_priority else p.arrival_time)
    for patient in tqdm(patients, desc="Neutral Fee Policy"):
        arrival = datetime.strptime(patient.arrival_time, "%Y-%m-%d")
        for delta in range(365):
            date_str = (arrival + timedelta(days=delta)).strftime("%Y-%m-%d")
            candidates = [pr for pr in providers.values()
                          if provider_accepts_strict(patient, pr) and pr.can_accept(patient, date_str)]
            if not candidates:
                continue
            chosen = min(candidates, key=lambda pr: haversine_distance(pr.lat, pr.long, patient.lat, patient.long))
            chosen.assign_patient(patient, date_str)
            break
    return patients

_ML_ASSIGN_MODEL = None
def _get_assign_model(model_path: str):
    global _ML_ASSIGN_MODEL
    if _ML_ASSIGN_MODEL is None:
        _ML_ASSIGN_MODEL = load_wait_regressor(model_path)
    return _ML_ASSIGN_MODEL


def ml_balanced_policy(
    patients: List["Patient"],
    providers: Dict[str, "Provider"],
    use_priority: bool = False,
    model_path: str = None,
    lambda_d: float = 0.15,
    lambda_u: float = 0.10,
    top_k: int = 3,
    lookahead_days: int = 30,
    max_days_forward: int = 365,
) -> List["Patient"]:
    if model_path is None:
        model_path = str(Path(DATA_DIR) / "models" / "ml_wait_regressor_v2.pkl")
    model = _get_assign_model(model_path)

    patients.sort(key=lambda p: (p.arrival_time, p.priority) if use_priority else p.arrival_time)

    for patient in tqdm(patients, desc="ML Balanced (Capacity-Aware)", unit="patient"):
        arrival_dt = datetime.strptime(patient.arrival_time, "%Y-%m-%d")
        assigned = False

        for d in range(max_days_forward):
            date_str = (arrival_dt + timedelta(days=d)).strftime("%Y-%m-%d")
            eligible = [prov for prov in providers.values()
                        if provider_accepts_strict(patient, prov) and prov.can_accept(patient, date_str)]
            if not eligible:
                continue

            df_cand = _build_feature_rows_for_candidates(patient, eligible, date_str)
            X = df_cand[FEATURE_COLS].apply(pd.to_numeric, errors="coerce").fillna(0.0)
            pred_wait = model.predict(X)

            d_norm = _zscore_safe_vec(df_cand["distance_km"].values)
            u_norm = _zscore_safe_vec(df_cand["prov_utilization"].values)
            rank_score = pred_wait + lambda_d * d_norm + lambda_u * u_norm

            order = np.lexsort((df_cand["distance_km"].values, rank_score))
            df_cand = df_cand.iloc[order].reset_index(drop=True)

            for i in range(min(top_k, len(df_cand))):
                prov = df_cand.iloc[i]["_provider_obj"]
                for look_ahead in range(lookahead_days):
                    check_date = arrival_dt + timedelta(days=d + look_ahead)
                    check_str = check_date.strftime("%Y-%m-%d")
                    if prov.can_accept(patient, check_str):
                        prov.assign_patient(patient, check_str)
                        assigned = True
                        break
                if assigned:
                    break
            if assigned:
                break

        if not assigned:
            patient.assigned_provider = None
            patient.wait_time = None

    return patients

def target_share_balanced_policy(
    patients: List["Patient"],
    providers: Dict[str, "Provider"],
    use_priority: bool = False,
    target_overall_nhs_share: float = 0.50,   
    include_ga_in_target: bool = True,        
    fallback_after_days: int = 60,            
    slack_days: int = 14,                     
    max_days_forward: int = 365,
) -> List["Patient"]:

    nhs_provs = [pr for pr in providers.values() if _is_nhs(pr)]
    pri_provs = [pr for pr in providers.values() if not _is_nhs(pr)]

    total_assigned = 0
    nhs_assigned = 0

    def _update_share_counts(chosen_provider: "Provider", counted: bool):
        nonlocal total_assigned, nhs_assigned
        if not counted:
            return
        total_assigned += 1
        if _is_nhs(chosen_provider):
            nhs_assigned += 1

    patients.sort(key=lambda p: (p.arrival_time, p.priority) if use_priority else p.arrival_time)

    for p in tqdm(patients, desc="Target-Share Balanced (overall)"):
        start = datetime.strptime(p.arrival_time, "%Y-%m-%d")
        needs_ga = bool(getattr(p, "need_ga", False))

        if needs_ga:
            d_nhs, ds_nhs, pr_nhs = _earliest_day_with_distance_tiebreak(nhs_provs, p, start, max_days_forward)
            if pr_nhs is not None:
                pr_nhs.assign_patient(p, ds_nhs)
                _update_share_counts(pr_nhs, counted=include_ga_in_target)
            continue

        assigned = False

        for delta in range(min(fallback_after_days, max_days_forward)):
            date_str = (start + timedelta(days=delta)).strftime("%Y-%m-%d")
            nhs_elig = [pr for pr in nhs_provs if pr.can_accept(p, date_str)]
            if nhs_elig:
                chosen = min(nhs_elig, key=lambda pr: haversine_distance(pr.lat, pr.long, p.lat, p.long))
                chosen.assign_patient(p, date_str)
                _update_share_counts(chosen, counted=True)
                assigned = True
                break
        if assigned:
            continue

        d_nhs, ds_nhs, pr_nhs = _earliest_day_with_distance_tiebreak(nhs_provs, p, start, max_days_forward)
        d_pri, ds_pri, pr_pri = _earliest_day_with_distance_tiebreak(pri_provs, p, start, max_days_forward)

        if pr_nhs is None and pr_pri is None:
            continue

        if pr_nhs is None and pr_pri is not None:
            pr_pri.assign_patient(p, ds_pri); _update_share_counts(pr_pri, counted=True); continue
        if pr_pri is None and pr_nhs is not None:
            pr_nhs.assign_patient(p, ds_nhs); _update_share_counts(pr_nhs, counted=True); continue

        current_share = (nhs_assigned / total_assigned) if total_assigned > 0 else 0.0
        want = "nhs" if current_share < float(target_overall_nhs_share) else "private"

        if want == "nhs" and (d_pri + slack_days < d_nhs):
            want = "private"
        elif want == "private" and (d_nhs + slack_days < d_pri):
            want = "nhs"

        if want == "nhs":
            pr_nhs.assign_patient(p, ds_nhs); _update_share_counts(pr_nhs, counted=True)
        else:
            pr_pri.assign_patient(p, ds_pri); _update_share_counts(pr_pri, counted=True)

    return patients