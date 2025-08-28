from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
from patient import Patient
from geo_utils import get_service_time


@dataclass
class Provider:
    name: str
    lat: float
    long: float
    is_nhs: str          
    minutes_per_day: float
    schedule: dict = field(default_factory=dict)

    def get_queue_length(self, date_str: str) -> int:
        return len(self.schedule.get(date_str, []))

    def can_accept(self, patient: Patient, date_str: str) -> bool:
        time_needed = get_service_time(patient, self)
        if time_needed <= 0:
            return False
        used = sum(m for _, m in self.schedule.get(date_str, []))
        noise = getattr(self, "daily_noise", 0)
        return used + noise + time_needed <= self.minutes_per_day

    def assign_patient(self, patient: Patient, date_str: str):
        time_needed = get_service_time(patient, self)
        self.schedule.setdefault(date_str, []).append((patient, time_needed))
        wait_days = self._wait_days(patient.arrival_time, date_str)
        patient.assign(self.name, wait_days)

    @staticmethod
    def _wait_days(arrival_date: str, assigned_date: str) -> int:
        arrival = datetime.strptime(arrival_date, "%Y-%m-%d")
        assigned = datetime.strptime(assigned_date, "%Y-%m-%d")
        return max(0, (assigned - arrival).days)

    def clear_schedule(self):
        self.schedule = {}
