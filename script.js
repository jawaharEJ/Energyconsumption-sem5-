const API_BASE = "http://localhost:5000";

let historyChart, forecastChart;

async function init() {
  try {
    await fetch(`${API_BASE}/api/health`);
    document.getElementById("model-status").textContent = "model connected";
  } catch (e) {
    document.getElementById("model-status").textContent = "backend offline";
    document.getElementById("model-chip").style.borderColor = "#ef5350";
    return;
  }

  loadModelInfo();
  loadHistory();
  loadForecast();
  loadInsights();
  setupPlanner();

  document.getElementById("forecast-days").addEventListener("change", loadForecast);

  // default date field to tomorrow
  const tomorrow = new Date();
  tomorrow.setDate(tomorrow.getDate() + 1);
  document.getElementById("input-date").value = tomorrow.toISOString().split("T")[0];
}

function setupPlanner() {
  const customRate = document.getElementById("custom-rate");
  const goalReduction = document.getElementById("goal-reduction");

  customRate.addEventListener("input", updatePlanner);
  goalReduction.addEventListener("input", updatePlanner);
  updatePlanner();
}

function updatePlanner() {
  const average = parseFloat(document.getElementById("stat-avg").textContent) || 0;
  const customRate = parseFloat(document.getElementById("custom-rate").value) || 0;
  const goalReduction = parseFloat(document.getElementById("goal-reduction").value) || 0;

  const monthlyUsage = average * 30;
  const monthlyCost = monthlyUsage * customRate;
  const targetSavings = Math.min(Math.max(goalReduction, 0), 50) / 100;
  const targetCost = monthlyCost * (1 - targetSavings);
  const dailySavings = ((monthlyCost - targetCost) / 30).toFixed(2);

  document.getElementById("planner-monthly-cost").textContent =
    `₹${monthlyCost.toFixed(2)} / target ₹${targetCost.toFixed(2)}`;
  document.getElementById("planner-savings-goal").textContent =
    `Save ₹${dailySavings} per day (${goalReduction}% target)`;

  const advice = goalReduction >= 20
    ? "Aim to shift heavy appliance use to off-peak hours and replace old bulbs with LEDs."
    : "Keep lights off in empty rooms and minimize standby power for the best results.";

  document.getElementById("planner-advice").textContent = advice;
}

async function loadModelInfo() {
  const res = await fetch(`${API_BASE}/api/model-info`);
  const data = await res.json();

  document.getElementById("stat-anomalies").textContent = data.anomalies_detected;
  document.getElementById("stat-r2").textContent = data.metrics.r2.toFixed(2);
}

async function loadInsights() {
  const res = await fetch(`${API_BASE}/api/insights`);
  const data = await res.json();

  document.getElementById("current-rate").textContent =
    `$${data.current_energy_rate_usd.toFixed(2)} / ₹${data.current_energy_rate_inr.toFixed(2)} per kWh`;

  const highDays = document.getElementById("insight-high-days");
  highDays.innerHTML = "";
  data.high_usage_days.forEach(day => {
    const li = document.createElement("li");
    li.textContent = `${day.date}: ${day.consumption_kwh} kWh`;
    highDays.appendChild(li);
  });

  const monthly = document.getElementById("insight-monthly-average");
  monthly.innerHTML = "";
  Object.entries(data.monthly_average_kwh).forEach(([month, value]) => {
    const li = document.createElement("li");
    li.textContent = `Month ${month}: ${value} kWh`;
    monthly.appendChild(li);
  });

  const weekCmp = document.getElementById("insight-week-comparison");
  weekCmp.textContent = data.week_comparison.change_pct === null
    ? "No prior week data"
    : `Current ${data.week_comparison.current_average} kWh vs prior ${data.week_comparison.previous_average} kWh (${data.week_comparison.change_pct}% change)`;

  const monthCmp = document.getElementById("insight-month-comparison");
  monthCmp.textContent = data.month_comparison.change_pct === null
    ? "No prior month data"
    : `Current ${data.month_comparison.current_average} kWh vs prior ${data.month_comparison.previous_average} kWh (${data.month_comparison.change_pct}% change)`;
}

async function loadHistory() {
  const res = await fetch(`${API_BASE}/api/history?days=60`);
  const { data } = await res.json();

  const labels = data.map(d => d.date.slice(5)); // MM-DD
  const actual = data.map(d => d.consumption_kwh);
  const predicted = data.map(d => d.predicted_kwh);
  const pointColors = data.map(d => d.is_anomaly_detected ? "#ef5350" : "#2dd4bf");
  const pointRadii = data.map(d => d.is_anomaly_detected ? 5 : 2);

  const avg = (actual.reduce((a, b) => a + b, 0) / actual.length).toFixed(2);
  document.getElementById("stat-avg").textContent = avg;

  const ctx = document.getElementById("history-chart");
  if (historyChart) historyChart.destroy();

  historyChart = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "Actual",
          data: actual,
          borderColor: "#2dd4bf",
          backgroundColor: "transparent",
          borderWidth: 2,
          pointBackgroundColor: pointColors,
          pointRadius: pointRadii,
          tension: 0.3,
        },
        {
          label: "Model fit",
          data: predicted,
          borderColor: "#7a5a1f",
          backgroundColor: "transparent",
          borderWidth: 1.5,
          borderDash: [4, 3],
          pointRadius: 0,
          tension: 0.3,
        },
      ],
    },
    options: chartOptions("kWh"),
  });
}

async function loadForecast() {
  const days = parseInt(document.getElementById("forecast-days").value, 10);
  const res = await fetch(`${API_BASE}/api/forecast?days=${days}`);
  const data = await res.json();

  document.getElementById("forecast-total").textContent =
    `${data.total_predicted_kwh} kWh total · $${data.total_estimated_cost_usd} / ₹${data.total_estimated_cost_inr}`;
  updatePlanner();

  const labels = data.forecast.map(f => f.date.slice(5));
  const values = data.forecast.map(f => f.predicted_consumption_kwh);

  const ctx = document.getElementById("forecast-chart");
  if (forecastChart) forecastChart.destroy();

  forecastChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [{
        label: "Forecasted kWh",
        data: values,
        backgroundColor: "#f5a623",
        borderRadius: 4,
        maxBarThickness: 34,
      }],
    },
    options: chartOptions("kWh"),
  });
}

function chartOptions(unit) {
  return {
    responsive: true,
    plugins: {
      legend: { display: false },
      tooltip: {
        callbacks: {
          label: (ctx) => `${ctx.dataset.label}: ${ctx.parsed.y} ${unit}`,
        },
      },
    },
    scales: {
      x: {
        ticks: { color: "#8b95a1", font: { family: "JetBrains Mono", size: 10 } },
        grid: { color: "#263340" },
      },
      y: {
        ticks: { color: "#8b95a1", font: { family: "JetBrains Mono", size: 10 } },
        grid: { color: "#263340" },
      },
    },
  };
}

document.getElementById("predict-form").addEventListener("submit", async (e) => {
  e.preventDefault();

  const date = document.getElementById("input-date").value;
  const temperature = parseFloat(document.getElementById("input-temp").value);

  const res = await fetch(`${API_BASE}/api/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ date, temperature }),
  });

  if (!res.ok) {
    alert("Prediction failed. Check your inputs.");
    return;
  }

  const data = await res.json();

  const readout = document.getElementById("readout");
  readout.hidden = false;
  document.getElementById("readout-digits").textContent =
    data.predicted_consumption_kwh.toFixed(2);
  document.getElementById("readout-cost").textContent =
    `$${data.estimated_cost_usd.toFixed(2)} / ₹${data.estimated_cost_inr.toFixed(2)} estimated cost`;
  document.getElementById("prediction-category").textContent =
    `Category: ${data.predicted_category}`;
  document.getElementById("readout-tip-text").textContent = data.energy_tip;
  document.getElementById("readout-tip").hidden = false;

  const alertBadge = document.getElementById("readout-alert");
  alertBadge.hidden = !data.is_high_usage_alert;
});

init();
