import pandas as pd
import streamlit as st

st.set_page_config(page_title="Turbine Monitoring Dashboard", layout="wide")

# ----------------------------
# Load data
# ----------------------------
snapshot_path = "engine_health_snapshot.csv"
full_path = "test_predictions_full.csv"

snapshot = pd.read_csv(snapshot_path)
full = pd.read_csv(full_path)

# Safety: ensure types
snapshot["engine_id"] = snapshot["engine_id"].astype(int)
full["engine_id"] = full["engine_id"].astype(int)

# ----------------------------
# Sidebar controls
# ----------------------------
st.sidebar.title("Controls")
st.sidebar.caption("Filter the fleet and drill into engines.")

risk_levels = ["All"] + sorted(snapshot["RiskLevel"].unique().tolist())
selected_level = st.sidebar.selectbox("Risk Level", risk_levels, index=0)

top_n = st.sidebar.slider("Top N risky engines", 5, 50, 10)

engine_list = sorted(full["engine_id"].unique().tolist())
selected_engine = st.sidebar.selectbox("Engine drill-down", engine_list)

show_only_latest = st.sidebar.checkbox("Show latest snapshot table only", value=True)

st.sidebar.divider()
st.sidebar.subheader("Maintenance rules")
critical_rul_threshold = st.sidebar.number_input("Critical if Predicted RUL ≤", value=30, min_value=1, max_value=200)

# ----------------------------
# Apply filters
# ----------------------------
filtered = snapshot.copy()
if selected_level != "All":
    filtered = filtered[filtered["RiskLevel"] == selected_level]

# Always rank by risk score
filtered_sorted = filtered.sort_values("RiskScore_0_1", ascending=False)

top_risky = filtered_sorted.head(top_n).copy()

# ----------------------------
# Header
# ----------------------------
st.title("Turbine Predictive Maintenance Monitoring")
st.caption("Fleet health, risk ranking, and engine degradation drill-down (NASA C-MAPSS FD001)")

# ----------------------------
# KPI Row
# ----------------------------
col1, col2, col3, col4, col5 = st.columns(5)

avg_rul = snapshot["Predicted_RUL"].mean()
critical_count = (snapshot["Predicted_RUL"] <= critical_rul_threshold).sum()
high_count = (snapshot["RiskLevel"] == "High").sum() if "High" in snapshot["RiskLevel"].unique() else 0
avg_health = snapshot["HealthScore_0_100"].mean()
avg_risk = snapshot["RiskScore_0_1"].mean()

col1.metric("Avg Predicted RUL", f"{avg_rul:.1f} cycles")
col2.metric("Critical Engines", f"{critical_count}")
col3.metric("Avg HealthScore", f"{avg_health:.1f}/100")
col4.metric("Avg RiskScore", f"{avg_risk:.3f}")
col5.metric("Fleet Size", f"{snapshot.shape[0]} engines")

st.divider()

# ----------------------------
# Alerts panel
# ----------------------------
st.subheader("Alerts")

critical_engines = snapshot[snapshot["Predicted_RUL"] <= critical_rul_threshold].sort_values("Predicted_RUL").head(8)

if len(critical_engines) > 0:
    st.warning("Critical engines detected. Prioritise inspection/maintenance scheduling.")
    st.dataframe(
        critical_engines[["engine_id", "cycle", "Predicted_RUL", "HealthScore_0_100", "RiskScore_0_1", "RiskLevel"]]
        .sort_values("Predicted_RUL"),
        use_container_width=True
    )
else:
    st.success("No engines currently meet the critical threshold based on your settings.")

st.divider()

# ----------------------------
# Fleet ranking
# ----------------------------
st.subheader("Top Risk Ranking")

left, right = st.columns([1.2, 1])

with left:
    st.write("Top risky engines by RiskScore (higher = riskier)")
    st.bar_chart(top_risky.set_index("engine_id")["RiskScore_0_1"])

with right:
    st.write("RiskLevel distribution (latest snapshot)")
    risk_counts = snapshot["RiskLevel"].value_counts()
    st.bar_chart(risk_counts)

st.divider()

# ----------------------------
# Table view
# ----------------------------
st.subheader("Fleet Snapshot Table")

if show_only_latest:
    st.write("Latest snapshot per engine (already one row per engine):")
    st.dataframe(
        filtered_sorted[[
            "engine_id",
            "cycle",
            "Predicted_RUL",
            "HealthScore_0_100",
            "RiskScore_0_1",
            "RiskLevel",
            "FailureRisk_Prob"
        ]].head(50),
        use_container_width=True
    )
else:
    st.write("Showing time-series rows from the full dataset (first 50 rows):")
    st.dataframe(full.head(50), use_container_width=True)

st.divider()

# ----------------------------
# Engine drill-down charts
# ----------------------------
st.subheader(f"Engine Drill-down: Engine {selected_engine}")

engine_data = full[full["engine_id"] == selected_engine].sort_values("cycle")

c1, c2 = st.columns(2)

with c1:
    st.write("Predicted RUL over time")
    st.line_chart(engine_data.set_index("cycle")[["Predicted_RUL"]])

with c2:
    st.write("Failure risk probability over time")
    if "FailureRisk_Prob" in engine_data.columns:
        st.line_chart(engine_data.set_index("cycle")[["FailureRisk_Prob"]])
    else:
        st.info("FailureRisk_Prob column not found in the full predictions file.")

# ----------------------------
# Maintenance recommendation
# ----------------------------
latest_row = snapshot[snapshot["engine_id"] == selected_engine].iloc[0]
pred_rul = float(latest_row["Predicted_RUL"])
risk_level = str(latest_row["RiskLevel"])
risk_prob = float(latest_row.get("FailureRisk_Prob", 0.0))

st.divider()
st.subheader("Maintenance Recommendation")

if pred_rul <= critical_rul_threshold or risk_level == "Critical":
    st.error(f"Action: IMMEDIATE attention recommended (Predicted RUL ≈ {pred_rul:.1f} cycles, RiskLevel={risk_level}).")
elif risk_level == "High":
    st.warning(f"Action: Schedule maintenance soon (Predicted RUL ≈ {pred_rul:.1f} cycles, RiskLevel={risk_level}).")
elif risk_level == "Medium":
    st.info(f"Action: Monitor closely and plan maintenance (Predicted RUL ≈ {pred_rul:.1f} cycles, RiskLevel={risk_level}).")
else:
    st.success(f"Action: Normal operation (Predicted RUL ≈ {pred_rul:.1f} cycles, RiskLevel={risk_level}).")

st.caption(
    "Note: This is a demo monitoring dashboard using benchmark data. "
    "In real deployments, thresholds are tuned based on safety requirements and operational constraints."
)