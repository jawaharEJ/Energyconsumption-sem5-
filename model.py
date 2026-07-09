"""
model.py
--------
Generates a realistic synthetic household energy-consumption dataset,
trains a RandomForestRegressor to predict daily energy consumption (kWh),
and saves the trained model + historical dataset to disk for the Flask
backend to use.

Run once with:  python model.py
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import json
import os

np.random.seed(42)

DAYS = 730  # 2 years of daily data
START_DATE = pd.Timestamp("2024-01-01")


def generate_synthetic_data(days=DAYS, start=START_DATE):
    dates = pd.date_range(start=start, periods=days, freq="D")

    records = []
    for d in dates:
        month = d.month
        day_of_week = d.dayofweek  # 0=Mon ... 6=Sun
        is_weekend = 1 if day_of_week >= 5 else 0

        # --- synthetic outdoor temperature (seasonal, Celsius) ---
        seasonal = 15 + 12 * np.sin((month - 3) / 12 * 2 * np.pi)
        temperature = seasonal + np.random.normal(0, 2.5)

        # --- base household load ---
        base_load = 12.0  # kWh/day baseline (lighting, fridge, standby devices)

        # heating demand when cold, cooling demand when hot
        heating = max(0, 18 - temperature) * 0.55
        cooling = max(0, temperature - 24) * 0.65

        # weekend effect: more people home -> slightly higher usage
        weekend_effect = 2.2 if is_weekend else 0

        # random appliance usage noise
        noise = np.random.normal(0, 1.3)

        consumption = base_load + heating + cooling + weekend_effect + noise

        # inject occasional anomalies (e.g., faulty appliance / AC left on)
        is_anomaly = 0
        if np.random.rand() < 0.03:
            consumption += np.random.uniform(8, 15)
            is_anomaly = 1

        consumption = max(3.0, consumption)  # can't go below minimal standby load

        records.append({
            "date": d.strftime("%Y-%m-%d"),
            "month": month,
            "day_of_week": day_of_week,
            "is_weekend": is_weekend,
            "temperature": round(temperature, 2),
            "consumption_kwh": round(consumption, 2),
            "is_anomaly_actual": is_anomaly,
        })

    return pd.DataFrame(records)


def train_model(df):
    features = ["month", "day_of_week", "is_weekend", "temperature"]
    X = df[features]
    y = df["consumption_kwh"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=200, max_depth=8, random_state=42
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"Model trained. MAE: {mae:.3f} kWh | R2: {r2:.3f}")

    return model, {"mae": round(mae, 3), "r2": round(r2, 3)}


def compute_residual_stats(df, model):
    """Used later for anomaly detection thresholds on historical data."""
    features = ["month", "day_of_week", "is_weekend", "temperature"]
    df["predicted_kwh"] = model.predict(df[features])
    df["residual"] = df["consumption_kwh"] - df["predicted_kwh"]
    residual_std = df["residual"].std()
    df["is_anomaly_detected"] = (df["residual"].abs() > 2 * residual_std).astype(int)
    return df, residual_std


if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))

    df = generate_synthetic_data()
    model, metrics = train_model(df)
    df, residual_std = compute_residual_stats(df, model)

    joblib.dump(model, os.path.join(out_dir, "energy_model.pkl"))
    df.to_csv(os.path.join(out_dir, "historical_data.csv"), index=False)

    with open(os.path.join(out_dir, "model_meta.json"), "w") as f:
        json.dump({
            "metrics": metrics,
            "residual_std": round(float(residual_std), 3),
            "features": ["month", "day_of_week", "is_weekend", "temperature"],
        }, f, indent=2)

    print("Saved: energy_model.pkl, historical_data.csv, model_meta.json")
