"""
app.py
------
Flask backend for the AI Energy Consumption Predictor.

Endpoints:
    GET  /api/history            -> last N days of historical usage + detected anomalies
    POST /api/predict            -> predict consumption for a single day
    GET  /api/forecast?days=7    -> forecast the next N days
    GET  /api/model-info         -> model performance metrics

Run with:  python app.py
Server runs on http://localhost:5000
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import datetime, timedelta

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "frontend"))
COST_PER_KWH_USD = 0.15  # illustrative electricity rate ($/kWh)
USD_TO_INR = 83.0
COST_PER_KWH_INR = round(COST_PER_KWH_USD * USD_TO_INR, 2)

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
CORS(app)

# ---------- Load model + data at startup ----------
model = joblib.load(os.path.join(BASE_DIR, "energy_model.pkl"))
history_df = pd.read_csv(os.path.join(BASE_DIR, "historical_data.csv"))
with open(os.path.join(BASE_DIR, "model_meta.json")) as f:
    model_meta = json.load(f)

FEATURES = model_meta["features"]
RESIDUAL_STD = model_meta["residual_std"]

CONSUMPTION_QUANTILES = history_df["consumption_kwh"].quantile([0.25, 0.75]).to_dict()


def percent_change(current, previous):
    if previous == 0:
        return None
    return round(((current - previous) / previous) * 100, 2)


def compare_periods(current_period, previous_period):
    current_avg = current_period["consumption_kwh"].mean() if len(current_period) else 0
    previous_avg = previous_period["consumption_kwh"].mean() if len(previous_period) else 0
    return {
        "current_average": round(current_avg, 2),
        "previous_average": round(previous_avg, 2),
        "change_pct": percent_change(current_avg, previous_avg),
    }


def format_cost(kwh):
    usd = round(kwh * COST_PER_KWH_USD, 2)
    return {
        "usd": usd,
        "inr": round(usd * USD_TO_INR, 2),
    }


def classify_usage(value):
    if value >= CONSUMPTION_QUANTILES[0.75]:
        return "High"
    if value <= CONSUMPTION_QUANTILES[0.25]:
        return "Low"
    return "Moderate"


def build_energy_tip(predicted, category, temperature):
    if category == "High":
        if temperature >= 30:
            return "High usage predicted in hot weather. Use fans, set AC at a warmer temperature, and close curtains during peak sun hours."
        if temperature <= 15:
            return "High usage predicted in cold weather. Use energy-efficient heating and insulate windows to save on electricity."
        return "High usage predicted. Turn off unused lights, unplug idle devices, and schedule heavy appliances during off-peak hours."

    if category == "Moderate":
        return "Moderate usage expected. Maintain efficient habits by using LED lighting and energy-saving appliance modes."

    return "Low usage predicted. Keep up the good work and continue minimizing standby power and unnecessary loads."


def build_features(date_str, temperature):
    """Build a single-row feature DataFrame from a date + temperature."""
    d = pd.Timestamp(date_str)
    row = {
        "month": d.month,
        "day_of_week": d.dayofweek,
        "is_weekend": 1 if d.dayofweek >= 5 else 0,
        "temperature": float(temperature),
    }
    return pd.DataFrame([row])[FEATURES]


@app.route("/api/history", methods=["GET"])
def get_history():
    days = int(request.args.get("days", 60))
    subset = history_df.tail(days).copy()

    records = []
    for _, row in subset.iterrows():
        cost = format_cost(row["consumption_kwh"])
        records.append({
            "date": row["date"],
            "consumption_kwh": row["consumption_kwh"],
            "predicted_kwh": row["predicted_kwh"],
            "temperature": row["temperature"],
            "is_anomaly_detected": bool(row["is_anomaly_detected"]),
            "consumption_category": row.get("consumption_category", classify_usage(row["consumption_kwh"])),
            "estimated_cost_usd": cost["usd"],
            "estimated_cost_inr": cost["inr"],
        })

    return jsonify({
        "count": len(records),
        "data": records,
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True)

    date_str = payload.get("date")
    temperature = payload.get("temperature")

    if date_str is None or temperature is None:
        return jsonify({"error": "date and temperature are required"}), 400

    try:
        X = build_features(date_str, temperature)
    except Exception as e:
        return jsonify({"error": f"invalid input: {e}"}), 400

    predicted = float(model.predict(X)[0])
    cost = format_cost(predicted)
    category = classify_usage(predicted)
    energy_tip = build_energy_tip(predicted, category, float(temperature))
    is_high_usage = predicted > (history_df["consumption_kwh"].mean() + 2 * RESIDUAL_STD)

    return jsonify({
        "date": date_str,
        "temperature": temperature,
        "predicted_consumption_kwh": round(predicted, 2),
        "estimated_cost_usd": cost["usd"],
        "estimated_cost_inr": cost["inr"],
        "predicted_category": category,
        "energy_tip": energy_tip,
        "is_high_usage_alert": bool(is_high_usage),
    })


@app.route("/api/forecast", methods=["GET"])
def forecast():
    n_days = int(request.args.get("days", 7))

    last_date = pd.Timestamp(history_df["date"].max())
    # simple seasonal temperature estimate: average temp for that month historically
    monthly_avg_temp = history_df.groupby("month")["temperature"].mean().to_dict()

    forecasts = []
    for i in range(1, n_days + 1):
        future_date = last_date + timedelta(days=i)
        month = future_date.month
        approx_temp = monthly_avg_temp.get(month, history_df["temperature"].mean())
        # small random-free daily variation for smoother forecast display
        approx_temp += np.sin(i) * 0.8

        X = build_features(future_date.strftime("%Y-%m-%d"), approx_temp)
        predicted = float(model.predict(X)[0])

        cost = format_cost(predicted)
        category = classify_usage(predicted)
        forecasts.append({
            "date": future_date.strftime("%Y-%m-%d"),
            "estimated_temperature": round(approx_temp, 2),
            "predicted_consumption_kwh": round(predicted, 2),
            "estimated_cost_usd": cost["usd"],
            "estimated_cost_inr": cost["inr"],
            "predicted_category": category,
        })

    total_kwh = sum(f["predicted_consumption_kwh"] for f in forecasts)
    total_cost = round(total_kwh * COST_PER_KWH, 2)

    return jsonify({
        "forecast": forecasts,
        "total_predicted_kwh": round(total_kwh, 2),
        "total_estimated_cost_usd": total_cost,
        "total_estimated_cost_inr": round(total_kwh * COST_PER_KWH_INR, 2),
    })


@app.route("/api/model-info", methods=["GET"])
def model_info():
    anomaly_count = int(history_df["is_anomaly_detected"].sum())
    monthly_summary = history_df.groupby("month")["consumption_kwh"].mean().round(2).to_dict()
    high_days = history_df.nlargest(5, "consumption_kwh")["date"].tolist()

    return jsonify({
        "metrics": model_meta["metrics"],
        "residual_std": RESIDUAL_STD,
        "total_historical_days": len(history_df),
        "anomalies_detected": anomaly_count,
        "cost_per_kwh_usd": COST_PER_KWH_USD,
        "cost_per_kwh_inr": COST_PER_KWH_INR,
        "monthly_average_kwh": monthly_summary,
        "top_high_usage_days": high_days,
    })


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/api/insights", methods=["GET"])
def insights():
    anomalies = int(history_df["is_anomaly_detected"].sum())
    high_usage_days = history_df.nlargest(5, "consumption_kwh")[["date", "consumption_kwh"]].to_dict(orient="records")
    monthly = history_df.groupby("month")["consumption_kwh"].mean().round(2).to_dict()

    last_week = history_df.tail(7)
    previous_week = history_df.iloc[-14:-7] if len(history_df) >= 14 else history_df.head(0)
    last_month = history_df.tail(30)
    previous_month = history_df.iloc[-60:-30] if len(history_df) >= 60 else history_df.head(0)

    week_comparison = compare_periods(last_week, previous_week)
    month_comparison = compare_periods(last_month, previous_month)

    return jsonify({
        "anomalies_detected": anomalies,
        "monthly_average_kwh": monthly,
        "high_usage_days": high_usage_days,
        "current_energy_rate_usd": COST_PER_KWH_USD,
        "current_energy_rate_inr": COST_PER_KWH_INR,
        "week_comparison": week_comparison,
        "month_comparison": month_comparison,
    })


@app.route("/", methods=["GET"])
def index():
    return app.send_static_file("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
