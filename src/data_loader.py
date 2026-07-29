"""
data_loader.py
---------------
Loads battery cycling data for the fault monitoring / RUL project.

Two modes:
1. load_real_nasa_data(path) -> loads the real NASA Li-ion Battery Aging
   dataset (https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository)
   if you have downloaded the .mat files locally. Point `path` at the folder
   containing them (e.g. B0005.mat, B0006.mat, ...).

2. generate_synthetic_data() -> generates a synthetic but realistic dataset
   that mirrors the structure and degradation physics of the NASA dataset
   (capacity fade, voltage/current/temperature curves, occasional anomalous
   cycles). Used here because this environment has no internet access to
   fetch the real dataset. Swap to load_real_nasa_data() once you have the
   real files locally -- the rest of the pipeline (features, models, plots)
   does not need to change.
"""

import numpy as np
import pandas as pd

RANDOM_SEED = 42


def generate_synthetic_data(n_batteries: int = 4, n_cycles: int = 168,
                             seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Generate synthetic Li-ion battery cycling data.

    Models capacity fade as a combination of:
      - slow linear/exponential degradation (normal aging)
      - a "knee point" where degradation accelerates (real battery behavior)
      - random anomalous cycles (sudden capacity drops / voltage sag) to
        simulate faults for the anomaly detection model to catch

    Returns a long-format DataFrame: one row per (battery_id, cycle).
    """
    rng = np.random.default_rng(seed)
    rows = []

    for b in range(n_batteries):
        battery_id = f"B{5 + b:04d}"
        nominal_capacity = 2.0  # Ah, matches NASA dataset's ~2Ah cells
        # each battery ages at a slightly different rate
        decay_rate = rng.uniform(0.0022, 0.0032)
        knee_cycle = rng.integers(90, 130)
        knee_severity = rng.uniform(1.5, 2.5)

        # inject 2-4 random fault cycles per battery
        n_faults = rng.integers(2, 5)
        fault_cycles = set(rng.choice(np.arange(10, n_cycles - 5),
                                       size=n_faults, replace=False))

        capacity = nominal_capacity
        for cycle in range(1, n_cycles + 1):
            # base exponential-ish fade
            base_fade = decay_rate * cycle
            # accelerated fade after knee point
            if cycle > knee_cycle:
                base_fade += decay_rate * knee_severity * (cycle - knee_cycle)

            capacity = nominal_capacity - base_fade + rng.normal(0, 0.01)

            is_fault = cycle in fault_cycles
            if is_fault:
                # sudden anomalous drop simulating an internal fault
                capacity -= rng.uniform(0.08, 0.18)

            capacity = max(capacity, 0.3)

            # discharge voltage/current/temperature curves (simplified summary stats
            # per cycle, mirroring what you'd extract from the raw NASA time series)
            voltage_mean = 3.7 - 0.15 * (nominal_capacity - capacity) + rng.normal(0, 0.02)
            voltage_min = voltage_mean - rng.uniform(0.6, 0.9)
            current_mean = -2.0 + rng.normal(0, 0.05)
            temperature_max = 32 + 6 * (nominal_capacity - capacity) + rng.normal(0, 1.0)
            if is_fault:
                temperature_max += rng.uniform(3, 8)
                voltage_min -= rng.uniform(0.1, 0.3)

            internal_resistance = 0.05 + 0.02 * (nominal_capacity - capacity) + rng.normal(0, 0.002)
            discharge_time = 3600 * (capacity / nominal_capacity) + rng.normal(0, 30)

            rows.append({
                "battery_id": battery_id,
                "cycle": cycle,
                "capacity_Ah": round(capacity, 4),
                "voltage_mean_V": round(voltage_mean, 4),
                "voltage_min_V": round(voltage_min, 4),
                "current_mean_A": round(current_mean, 4),
                "temperature_max_C": round(temperature_max, 3),
                "internal_resistance_ohm": round(max(internal_resistance, 0.01), 5),
                "discharge_time_s": round(max(discharge_time, 100), 1),
                "is_synthetic_fault": is_fault,
            })

    df = pd.DataFrame(rows)
    return df


def load_real_nasa_data(path: str) -> pd.DataFrame:
    """
    Placeholder loader for the real NASA .mat files. Requires scipy.io.
    Left here so you can swap this in once you have internet / the real
    dataset downloaded locally (see README for the download link).
    """
    raise NotImplementedError(
        "Point this at your local NASA .mat files and implement parsing with "
        "scipy.io.loadmat(). See README.md for the dataset link and structure."
    )


if __name__ == "__main__":
    df = generate_synthetic_data()
    df.to_csv("data/battery_cycles.csv", index=False)
    print(f"Generated {len(df)} rows across {df['battery_id'].nunique()} batteries")
    print(df.head())
