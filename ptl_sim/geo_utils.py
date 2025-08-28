import pandas as pd
from config import DATA_DIR
import os
import random
import numpy as np

NHS_COMPLEXITY_TIME = {
    "low": (15, 20),
    "medium": (20, 25),
    "high": (35, 60)
}

PRIVATE_COMPLEXITY_TIME = {
    "low": (10, 15),
    "medium": (15, 20),
    "high": (30, 45)
}

def load_lsoa_coordinates():
    cache_path = DATA_DIR / "lsoa_coordinates.pkl"
    if os.path.exists(cache_path):
        return pd.read_pickle(cache_path)
    
    df = pd.read_csv(DATA_DIR / "lsoa_coordinates.csv", usecols=["LSOA01", "LSOA11", "LAT", "LONG"])
    df.to_pickle(cache_path)
    return df

def get_service_time(patient, provider):
    comp = getattr(patient, "complexity", "Medium")
    base = {"Low": 30, "Medium": 60, "High": 120}.get(comp, 60)
    if bool(getattr(patient, "need_ga", False)):
        base += 60
    is_nhs = str(getattr(provider, "is_nhs", "false")).lower() == "true"
    if not is_nhs:
        base = int(base * 0.9)
    return int(base)
    
def get_patient_lat_long(df, lsoa_coords=None):
    if lsoa_coords is None:
        lsoa_coords = load_lsoa_coordinates()

    df = df.copy()

    lsoa01_map = (
        lsoa_coords[['LSOA01', 'LAT', 'LONG']]
        .drop_duplicates(subset='LSOA01')
        .set_index('LSOA01')
        .to_dict(orient='index')
    )

    lsoa11_map = (
        lsoa_coords[['LSOA11', 'LAT', 'LONG']]
        .drop_duplicates(subset='LSOA11')
        .set_index('LSOA11')
        .to_dict(orient='index')
    )

    def resolve_coords(row):
        lsoa = row['lsoa']
        if lsoa in lsoa01_map:
            return pd.Series([lsoa01_map[lsoa]['LAT'], lsoa01_map[lsoa]['LONG']])
        elif lsoa in lsoa11_map:
            return pd.Series([lsoa11_map[lsoa]['LAT'], lsoa11_map[lsoa]['LONG']])
        return pd.Series([None, None])

    df[['lat', 'long']] = df.apply(resolve_coords, axis=1)

    # synthetic location
    # Calculate missing after fill LSOA
    # missing = df['lat'].isnull() | df['long'].isnull()
    # df['synthetic_location'] = missing

    # if missing lat/lon → fill with random known locations (synthetic), to avoid bias
    # if missing.sum() > 0:
    #     known_coords = df[~missing][['lat', 'long']].dropna()
    #     print("Known coords shape:", known_coords.shape)
    #     synthetic_coords = known_coords.sample(n=missing.sum(), replace=True, random_state=42).reset_index(drop=True)
    #     missing_idx = df[missing].index
    #     df.loc[missing_idx, 'lat'] = synthetic_coords['lat'].values
    #     df.loc[missing_idx, 'long'] = synthetic_coords['long'].values

    return df
