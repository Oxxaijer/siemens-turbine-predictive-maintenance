import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# ---------------------------
# column names
# ---------------------------

COLUMNS = (
    ["engine_id", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

FEATURE_COLS = (
    [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

# ---------------------------
# load dataset
# ---------------------------

df = pd.read_csv("data/train_FD001.txt", sep=r"\s+", header=None)
df = df.iloc[:, :26]
df.columns = COLUMNS

# ---------------------------
# compute RUL
# ---------------------------

max_cycle = df.groupby("engine_id")["cycle"].max()

df["RUL"] = df["engine_id"].map(max_cycle) - df["cycle"]

# ---------------------------
# Feature Engineering
# Rolling sensor averages
# ---------------------------

for col in FEATURE_COLS:
    df[col + "_rolling"] = (
        df.groupby("engine_id")[col]
        .rolling(window=5)
        .mean()
        .reset_index(0, drop=True)
    )

# fill missing values
df = df.fillna(method="bfill")

# ---------------------------
# features
# ---------------------------

feature_columns = [
    c for c in df.columns
    if c not in ["engine_id", "cycle", "RUL"]
]

X = df[feature_columns]
y = df["RUL"]

# ---------------------------
# train test split
# ---------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------
# train model
# ---------------------------

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ---------------------------
# evaluate
# ---------------------------

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)

print("\nImproved Model Results")
print("MAE:", mae)