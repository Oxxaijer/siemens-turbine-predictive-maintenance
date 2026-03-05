import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score

# Load dataset
df = pd.read_csv("data/train_FD001.txt", sep=r"\s+", header=None)

columns = (
    ["engine_id", "cycle"] +
    [f"setting_{i}" for i in range(1,4)] +
    [f"sensor_{i}" for i in range(1,22)]
)

df = df.iloc[:, :26]
df.columns = columns

# Compute RUL
max_cycle = df.groupby("engine_id")["cycle"].max()
df["RUL"] = df["engine_id"].map(max_cycle) - df["cycle"]

# Failure classification target
df["FailureSoon"] = (df["RUL"] <= 30).astype(int)

# Drop identifiers
df_model = df.drop(columns=["engine_id","cycle"])

X = df_model.drop(columns=["RUL","FailureSoon"])
y_reg = df_model["RUL"]
y_class = df_model["FailureSoon"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_reg, test_size=0.2, random_state=42
)

# Regression model
reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
reg_model.fit(X_train, y_train)

predictions = reg_model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)

print("\nRegression Model")
print("MAE:", mae)

# Classification model
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X, y_class, test_size=0.2, random_state=42
)

clf_model = RandomForestClassifier(n_estimators=100, random_state=42)
clf_model.fit(X_train_c, y_train_c)

pred_c = clf_model.predict(X_test_c)
acc = accuracy_score(y_test_c, pred_c)

print("\nFailure Prediction Model")
print("Accuracy:", acc)

# Feature importance
importances = pd.Series(
    reg_model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nTop Important Sensors:")
print(importances.head(10))

# -------- Export dashboard dataset (Power BI ready) --------

# Build a clean feature matrix with the SAME columns used in training
X_all = df_model.drop(columns=["RUL", "FailureSoon"])  # df_model already has engine_id/cycle removed

# Predict for every row
df_dashboard = df.copy()
df_dashboard["Predicted_RUL"] = reg_model.predict(X_all)
df_dashboard["FailureRisk"] = clf_model.predict(X_all)

# (Optional) Also export probability (better for dashboards)
df_dashboard["FailureRisk_Prob"] = clf_model.predict_proba(X_all)[:, 1]

df_dashboard.to_csv("dashboard_data.csv", index=False)
print("\n✅ Dashboard dataset exported → dashboard_data.csv")