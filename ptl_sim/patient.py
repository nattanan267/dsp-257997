from dataclasses import dataclass
from typing import Optional


@dataclass
class Patient:
    patient_id: str
    arrival_time: str
    complexity: str
    lat: float
    long: float
    original_provider: str
    need_ga: bool = False
    priority: int = 3
    hrg_code: Optional[str] = None
    tariff: Optional[float] = None
    has_tariff: Optional[bool] = False

    assigned_provider: Optional[str] = None
    wait_time: Optional[int] = None

    def assign(self, provider_name: str, wait_days: int):
        self.assigned_provider = provider_name
        self.wait_time = wait_days

    def __repr__(self):
        return (f"Patient({self.patient_id}, {self.arrival_time}, {self.complexity}, "
                f"Provider={self.original_provider} → {self.assigned_provider}, Wait={self.wait_time})")
