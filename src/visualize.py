"""
visualize.py
-------------
Generates the three key plots for the project:
1. Capacity degradation over cycles (per battery)
2. Anomaly detection results overlaid on capacity curve
3. Predicted vs actual RUL (regression performance)
"""

import pandas as pd
import matplotlib.pyplot as plt
from models import detect_anomalies, train_rul_model

plt.style.use("seaborn-v0_8-whitegrid")


def plot_degradation(df: pd.DataFrame, save_path: str):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for battery_id, g in df.groupby("battery_id"):
        ax.plot(g["cycle"], g["capacity_Ah"], label=battery_id, linewidth=1.8)

    ax.axhline(y=0.8 * 2.0, color="red", linestyle="--", linewidth=1,
               label="EOL threshold (80% SOH)")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Capacity (Ah)")
    ax.set_title("Battery Capacity Degradation over Charge/Discharge Cycles")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_anomalies(df_anom: pd.DataFrame, save_path: str, battery_id: str = None):
    if battery_id is None:
        battery_id = df_anom["battery_id"].unique()[0]
    g = df_anom[df_anom["battery_id"] == battery_id]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(g["cycle"], g["capacity_Ah"], color="steelblue", linewidth=1.5,
            label="Capacity")
    anomalies = g[g["is_anomaly"]]
    ax.scatter(anomalies["cycle"], anomalies["capacity_Ah"], color="red",
               s=60, zorder=5, label="Flagged anomaly")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Capacity (Ah)")
    ax.set_title(f"Anomaly Detection on Battery {battery_id} "
                 f"(Isolation Forest)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_rul_predictions(y_test, y_pred, save_path: str):
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(y_test, y_pred, alpha=0.6, color="darkorange", edgecolor="k",
               linewidth=0.3)
    lims = [0, max(y_test.max(), y_pred.max()) + 5]
    ax.plot(lims, lims, "k--", linewidth=1, label="Perfect prediction")
    ax.set_xlabel("Actual RUL (cycles)")
    ax.set_ylabel("Predicted RUL (cycles)")
    ax.set_title("Remaining Useful Life: Predicted vs Actual")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    df = pd.read_csv("data/battery_features.csv")

    plot_degradation(df, "plots/capacity_degradation.png")

    df_anom, _, _ = detect_anomalies(df)
    plot_anomalies(df_anom, "plots/anomaly_detection.png", battery_id="B0005")

    _, metrics, (X_test, y_test, y_pred) = train_rul_model(df)
    plot_rul_predictions(y_test, y_pred, "plots/rul_prediction.png")

    print("Saved plots to plots/")
