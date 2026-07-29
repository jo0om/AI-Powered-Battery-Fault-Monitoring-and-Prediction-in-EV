"""
features.py
------------
Feature engineering for battery fault monitoring / RUL prediction.
"""

import pandas as pd
import numpy as np


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds engineered features per battery:
      - capacity_fade_pct: % capacity lost vs. first cycle (nominal)
      - capacity_delta: cycle-over-cycle capacity change (spikes = anomalies)
      - rolling_capacity_std: rolling volatility of capacity
      - voltage_sag: voltage_mean - voltage_min
      - cycles_to_eol: cycles remaining until capacity crosses the 80% SOH
        threshold (used as the RUL regression target)
    """
    df = df.sort_values(["battery_id", "cycle"]).reset_index(drop=True)
    out = []

    for battery_id, g in df.groupby("battery_id"):
        g = g.copy()
        nominal_capacity = g["capacity_Ah"].iloc[0]

        g["capacity_fade_pct"] = (1 - g["capacity_Ah"] / nominal_capacity) * 100
        g["capacity_delta"] = g["capacity_Ah"].diff().fillna(0)
        g["rolling_capacity_std"] = g["capacity_Ah"].rolling(5, min_periods=1).std().fillna(0)
        g["voltage_sag"] = g["voltage_mean_V"] - g["voltage_min_V"]

        # End of Life (EOL) commonly defined as 80% of nominal capacity (SOH = 80%)
        eol_threshold = 0.8 * nominal_capacity
        eol_cycles = g.loc[g["capacity_Ah"] <= eol_threshold, "cycle"]
        eol_cycle = eol_cycles.min() if not eol_cycles.empty else g["cycle"].max()

        g["cycles_to_eol"] = (eol_cycle - g["cycle"]).clip(lower=0)

        out.append(g)

    result = pd.concat(out, ignore_index=True)
    return result


if __name__ == "__main__":
    df = pd.read_csv("data/battery_cycles.csv")
    feat_df = engineer_features(df)
    feat_df.to_csv("data/battery_features.csv", index=False)
    print(feat_df[["battery_id", "cycle", "capacity_fade_pct", "cycles_to_eol"]].head(10))
