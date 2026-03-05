import pandas as pd
import numpy as np

# Config: what counts as "critical"
CRITICAL_RUL = 30  # cycles

def clamp(series, low, high):
    return series.clip(lower=low, upper=high)

if __name__ == "__main__":
    # We use the full test predictions file you already exported
    df = pd.read_csv("test_predictions_full.csv")

    # ---- 1) Pick "latest reading" per engine (monitoring snapshot) ----
    latest = (
        df.sort_values(["engine_id", "cycle"])
          .groupby("engine_id")
          .tail(1)
          .copy()
    )

    # ---- 2) Health Score (0–100) based on predicted RUL ----
    # Define a reasonable max RUL scale from predictions (robust: 95th percentile)
    max_rul_scale = float(np.percentile(df["Predicted_RUL"], 95))
    if max_rul_scale <= 0:
        max_rul_scale = float(df["Predicted_RUL"].max())

    latest["HealthScore_0_100"] = clamp((latest["Predicted_RUL"] / max_rul_scale) * 100, 0, 100).round(1)

    # ---- 3) Risk Score (0–1): higher means more risky ----
    # Risk increases as predicted RUL decreases
    latest["RiskScore_0_1"] = clamp(1 - (latest["HealthScore_0_100"] / 100), 0, 1).round(3)

    # ---- 4) Risk Level label (easy to read for stakeholders) ----
    # You can tune these thresholds later
    def risk_level(row):
        if row["Predicted_RUL"] <= CRITICAL_RUL or row.get("FailureRisk", 0) == 1:
            return "Critical"
        if row["RiskScore_0_1"] >= 0.60:
            return "High"
        if row["RiskScore_0_1"] >= 0.35:
            return "Medium"
        return "Low"

    latest["RiskLevel"] = latest.apply(risk_level, axis=1)

    # ---- 5) Export a clean monitoring table ----
    out_cols = [
        "engine_id",
        "cycle",
        "Predicted_RUL",
        "True_RUL",
        "HealthScore_0_100",
        "RiskScore_0_1",
        "RiskLevel",
        "FailureRisk",
        "FailureRisk_Prob",
    ]

    # Some columns may not exist depending on your export; keep only what's available
    out_cols = [c for c in out_cols if c in latest.columns]

    latest = latest[out_cols].sort_values(["RiskLevel", "RiskScore_0_1"], ascending=[True, False])

    latest.to_csv("engine_health_snapshot.csv", index=False)

    print("✅ Exported monitoring snapshot → engine_health_snapshot.csv")
    print(f"Scale used for HealthScore (95th percentile Predicted_RUL): {max_rul_scale:.2f}")
    print("\nTop 10 riskiest engines:")
    print(latest.head(10).to_string(index=False))