import pandas as pd
import streamlit as st

st.set_page_config(page_title="Turbine Predictive Maintenance Dashboard", layout="wide")

st.title("Turbine Predictive Maintenance Dashboard")
st.caption("RUL prediction, failure risk, and fleet health monitoring (NASA C-MAPSS FD001)")

# Load data
snapshot_path = "engine_health_snapshot.csv"
full_path = "test_predictions_full.csv"

snapshot = pd.read_csv(snapshot_path)
full = pd.read_csv(full_path)

# ----------------------------
# Fleet KPI Section
# ----------------------------
st.subheader("Fleet Overview")

col1, col2, col3, col4 = st.columns(4)

avg_rul = snapshot["Predicted_RUL"].mean()
critical_count = (snapshot["RiskLevel"] == "Critical").sum()
high_count = (snapshot["RiskLevel"] == "High").sum()
avg_risk = snapshot["RiskScore_0_1"].mean()

col1.metric("Avg Predicted RUL (cycles)", f"{avg_rul:.1f}")
col2.metric("Critical Engines", f"{critical_count}")
col3.metric("High Risk Engines", f"{high_count}")
col4.metric("Avg RiskScore (0–1)", f"{avg_risk:.3f}")

st.divider()

# ----------------------------
# Risk Ranking + Table
# ----------------------------
st.subheader("Top Risk Engines")

top_n = st.slider("Show Top N Risky Engines", 5, 30, 10)

top_risky = snapshot.sort_values("RiskScore_0_1", ascending=False).head(top_n)

st.bar_chart(top_risky.set_index("engine_id")["RiskScore_0_1"])

st.write("Latest snapshot per engine (sorted by risk):")
st.dataframe(top_risky, use_container_width=True)

st.divider()

# ----------------------------
# Engine Drill-down
# ----------------------------
st.subheader("Engine Drill-down")

engine_list = sorted(full["engine_id"].unique())
selected_engine = st.selectbox("Select engine_id", engine_list)

engine_data = full[full["engine_id"] == selected_engine].sort_values("cycle")

left, right = st.columns(2)

with left:
    st.write("Predicted RUL over time")
    chart_df = engine_data[["cycle", "Predicted_RUL"]].set_index("cycle")
    st.line_chart(chart_df)

with right:
    st.write("Failure Risk Probability over time")
    if "FailureRisk_Prob" in engine_data.columns:
        chart_df2 = engine_data[["cycle", "FailureRisk_Prob"]].set_index("cycle")
        st.line_chart(chart_df2)
    else:
        st.info("FailureRisk_Prob column not found in test_predictions_full.csv")

st.divider()

# ----------------------------
# Health Distribution
# ----------------------------
st.subheader("Fleet Health Distribution")

st.write("HealthScore (0–100) distribution")
st.bar_chart(snapshot["HealthScore_0_100"].value_counts().sort_index())

st.write("RiskLevel counts")
risk_counts = snapshot["RiskLevel"].value_counts()
st.bar_chart(risk_counts)

st.caption("Tip: Deploy this app later to Streamlit Community Cloud by connecting your GitHub repository.")