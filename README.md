# AI Powered Battery Fault Monitoring and Prediction in EV

**Repo:** https://github.com/jo0om/AI-Powered-Battery-Fault-Monitoring-and-Prediction-in-EV

An end-to-end machine learning pipeline for monitoring the health of
lithium-ion EV batteries: detecting anomalous cycles that indicate a
developing fault, and predicting Remaining Useful Life (RUL) to support
predictive maintenance.

## Overview

Li-ion batteries in electric vehicles degrade with every charge/discharge
cycle. Catching abnormal degradation early (a fault) and forecasting how
many cycles remain before end-of-life lets fleet operators schedule
maintenance proactively instead of reactively. This project builds:

1. **Anomaly detection** (Isolation Forest) — flags individual cycles where
   capacity, voltage, or temperature behavior deviates sharply from the
   expected degradation trend.
2. **RUL prediction** (Random Forest Regressor) — predicts the number of
   cycles remaining until the battery crosses 80% of nominal capacity (the
   standard End-of-Life / SOH threshold used in battery health research).

## Results

| Model | Metric | Result |
|---|---|---|
| Anomaly detection | Recall on injected fault cycles | 100% (11/11) |
| RUL prediction | MAE | ~5.4 cycles |
| RUL prediction | R² | 0.947 |

See `plots/` for the capacity degradation curves, anomaly detection overlay,
and predicted-vs-actual RUL scatter plot.

## A note on the dataset

This project uses a **synthetic dataset generator** (`src/data_loader.py`)
that mirrors the structure and degradation physics of NASA's
[Li-ion Battery Aging Dataset](https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository) —
the standard public dataset for this kind of project. It models:
- realistic capacity fade curves (slow degradation + an accelerated "knee
  point," matching real battery aging behavior)
- per-cycle voltage/current/temperature summary statistics
- randomly injected fault cycles (sudden capacity drops, temperature spikes,
  voltage sag) for the anomaly detector to catch

This was a deliberate choice to keep the project fully reproducible without
requiring a large external download. **To use the real NASA dataset**:
download the `.mat` files from the link above, and implement
`load_real_nasa_data()` in `src/data_loader.py` using `scipy.io.loadmat()` —
the rest of the pipeline (feature engineering, models, plots) works
unchanged on real data, since it expects the same column structure.

## Project Structure

```
AI-Powered-Battery-Fault-Monitoring-and-Prediction-in-EV/
├── data/                   # generated/loaded CSV data
├── notebooks/
│   └── battery_analysis.ipynb   # main walkthrough notebook
├── plots/                  # generated result plots
├── src/
│   ├── data_loader.py      # synthetic data generator / real data loader
│   ├── features.py         # feature engineering
│   ├── models.py           # anomaly detection + RUL models
│   └── visualize.py        # plot generation
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Run the full pipeline from the project root:

```bash
python src/data_loader.py     # generates data/battery_cycles.csv
python src/features.py        # generates data/battery_features.csv
python src/models.py          # trains models, prints metrics
python src/visualize.py       # generates plots/*.png
```

Or open `notebooks/battery_analysis.ipynb` for the full walkthrough with
explanations and inline visualizations.

## Tech Stack

Python, Pandas, NumPy, Scikit-learn, Matplotlib, Jupyter

## Future Work

- Validate against the real NASA dataset
- Try gradient boosting (XGBoost/LightGBM) for RUL prediction
- Extend to streaming/online fault detection for real-time monitoring
- Package as a simple dashboard (Streamlit) for demo purposes
