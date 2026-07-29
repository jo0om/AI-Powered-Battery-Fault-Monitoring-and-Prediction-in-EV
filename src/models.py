"""
models.py
----------
Two models:
1. Anomaly detection (Isolation Forest) - flags abnormal battery cycles
   (sudden capacity drops, voltage sag, temperature spikes) that indicate
   a developing fault.
2. RUL regression (Random Forest) - predicts cycles remaining until the
   battery reaches End of Life (80% of nominal capacity).
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

ANOMALY_FEATURES = [
    "capacity_delta", "rolling_capacity_std", "voltage_sag",
    "temperature_max_C", "internal_resistance_ohm",
]

RUL_FEATURES = [
    "cycle", "capacity_Ah", "capacity_fade_pct", "voltage_mean_V",
    "voltage_sag", "internal_resistance_ohm", "temperature_max_C",
]


def detect_anomalies(df: pd.DataFrame, contamination: float = 0.05):
    """Fits an Isolation Forest and returns df with an `is_anomaly` column."""
    X = df[ANOMALY_FEATURES].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=200, contamination=contamination, random_state=42
    )
    preds = model.fit_predict(X_scaled)  # -1 = anomaly, 1 = normal

    df = df.copy()
    df["is_anomaly"] = preds == -1
    df["anomaly_score"] = model.decision_function(X_scaled)
    return df, model, scaler


def train_rul_model(df: pd.DataFrame):
    """Trains a Random Forest regressor to predict cycles_to_eol (RUL)."""
    X = df[RUL_FEATURES].fillna(0)
    y = df["cycles_to_eol"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=300, max_depth=8, random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    metrics = {"mae": mae, "r2": r2}
    return model, metrics, (X_test, y_test, y_pred)


if __name__ == "__main__":
    df = pd.read_csv("data/battery_features.csv")

    df_anom, iso_model, scaler = detect_anomalies(df)
    n_flagged = df_anom["is_anomaly"].sum()
    n_true_faults = df_anom["is_synthetic_fault"].sum()
    n_caught = ((df_anom["is_anomaly"]) & (df_anom["is_synthetic_fault"])).sum()
    print(f"Flagged {n_flagged} anomalous cycles out of {len(df_anom)}")
    print(f"Of {n_true_faults} injected faults, caught {n_caught} "
          f"({n_caught / n_true_faults * 100:.1f}% recall)")

    rul_model, metrics, _ = train_rul_model(df)
    print(f"RUL model -> MAE: {metrics['mae']:.2f} cycles, R^2: {metrics['r2']:.3f}")

    df_anom.to_csv("data/battery_with_anomalies.csv", index=False)
