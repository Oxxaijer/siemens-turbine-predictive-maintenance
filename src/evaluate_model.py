import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---------- Common column names ----------
COLUMNS = (
    ["engine_id", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

# Use ONLY these columns as ML features (important!)
FEATURE_COLS = (
    [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

def load_cmapss(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df = df.iloc[:, :26]  # keep real columns
    df.columns = COLUMNS
    return df

def add_rul_for_training(df_train: pd.DataFrame) -> pd.DataFrame:
    # In train set, engines run until failure so we can compute RUL directly
    max_cycle = df_train.groupby("engine_id")["cycle"].max()
    out = df_train.copy()
    out["RUL"] = out["engine_id"].map(max_cycle) - out["cycle"]
    return out

def build_test_truth(df_test: pd.DataFrame, rul_path: str) -> pd.DataFrame:
    """
    Test engines stop BEFORE failure.
    RUL_FD001.txt gives remaining cycles AFTER the last observed cycle of each engine.

    True_RUL_at_each_row = (last_cycle - current_cycle) + RUL_at_last_cycle
    """
    rul_last = pd.read_csv(rul_path, header=None, names=["RUL_at_last_cycle"])
    rul_last["engine_id"] = np.arange(1, len(rul_last) + 1)

    last_cycle = df_test.groupby("engine_id")["cycle"].max().reset_index()
    last_cycle = last_cycle.rename(columns={"cycle": "last_cycle"})

    truth = last_cycle.merge(rul_last, on="engine_id", how="inner")

    out = df_test.merge(truth, on="engine_id", how="left")
    out["True_RUL"] = (out["last_cycle"] - out["cycle"]) + out["RUL_at_last_cycle"]
    return out

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    # Always use only sensor + setting columns (prevents sklearn feature mismatch)
    return df[FEATURE_COLS].copy()

if __name__ == "__main__":
    train_path = "data/train_FD001.txt"
    test_path  = "data/test_FD001.txt"
    rul_path   = "data/RUL_FD001.txt"

    # Load train/test
    train_df = load_cmapss(train_path)
    test_df  = load_cmapss(test_path)

    # Add RUL to train
    train_df = add_rul_for_training(train_df)

    # Create classification label for failure soon (<= 30 cycles)
    train_df["FailureSoon"] = (train_df["RUL"] <= 30).astype(int)

    # Prepare training data
    X_train = make_features(train_df)
    y_rul   = train_df["RUL"]
    y_fail  = train_df["FailureSoon"]

    # Train models
    reg_model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
    clf_model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    reg_model.fit(X_train, y_rul)
    clf_model.fit(X_train, y_fail)

    # Build true RUL for test set
    test_df = build_test_truth(test_df, rul_path)

    # Predict on test
    X_test = make_features(test_df)
    test_df["Predicted_RUL"] = reg_model.predict(X_test)
    test_df["FailureRisk"] = clf_model.predict(X_test)
    test_df["FailureRisk_Prob"] = clf_model.predict_proba(X_test)[:, 1]

    # Evaluate RUL prediction on test (official benchmark)
    mae = mean_absolute_error(test_df["True_RUL"], test_df["Predicted_RUL"])
    rmse = np.sqrt(mean_squared_error(test_df["True_RUL"], test_df["Predicted_RUL"]))

    print("✅ Official NASA Test Evaluation (FD001)")
    print(f"MAE  (cycles): {mae:.3f}")
    print(f"RMSE (cycles): {rmse:.3f}")

    # Export a clean per-engine summary (latest cycle only)
    latest = (
        test_df.sort_values(["engine_id", "cycle"])
              .groupby("engine_id")
              .tail(1)
              .copy()
    )

    latest = latest[[
        "engine_id",
        "cycle",
        "Predicted_RUL",
        "True_RUL",
        "FailureRisk",
        "FailureRisk_Prob"
    ]].sort_values("FailureRisk_Prob", ascending=False)

    latest.to_csv("test_engine_summary.csv", index=False)
    test_df.to_csv("test_predictions_full.csv", index=False)

    print("\n✅ Exported:")
    print(" - test_engine_summary.csv   (latest snapshot per engine)")
    print(" - test_predictions_full.csv (all rows, time-series)")