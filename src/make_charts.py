import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor

# ---------- column names ----------
COLUMNS = (
    ["engine_id", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

FEATURE_COLS = (
    [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

def load_cmapss(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df = df.iloc[:, :26]
    df.columns = COLUMNS
    return df

def add_rul_train(df_train: pd.DataFrame) -> pd.DataFrame:
    max_cycle = df_train.groupby("engine_id")["cycle"].max()
    out = df_train.copy()
    out["RUL"] = out["engine_id"].map(max_cycle) - out["cycle"]
    return out

def build_test_truth(df_test: pd.DataFrame, rul_path: str) -> pd.DataFrame:
    rul_last = pd.read_csv(rul_path, header=None, names=["RUL_at_last_cycle"])
    rul_last["engine_id"] = np.arange(1, len(rul_last) + 1)

    last_cycle = df_test.groupby("engine_id")["cycle"].max().reset_index()
    last_cycle = last_cycle.rename(columns={"cycle": "last_cycle"})

    truth = last_cycle.merge(rul_last, on="engine_id", how="inner")

    out = df_test.merge(truth, on="engine_id", how="left")
    out["True_RUL"] = (out["last_cycle"] - out["cycle"]) + out["RUL_at_last_cycle"]
    return out

def add_rolling_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    out = df.copy()
    for col in FEATURE_COLS:
        out[col + "_rolling"] = (
            out.groupby("engine_id")[col]
            .rolling(window=window)
            .mean()
            .reset_index(0, drop=True)
        )
    out = out.bfill()
    return out

if __name__ == "__main__":
    train_path = "data/train_FD001.txt"
    test_path = "data/test_FD001.txt"
    rul_path = "data/RUL_FD001.txt"

    # Load train/test
    train_df = load_cmapss(train_path)
    test_df = load_cmapss(test_path)

    # Add RUL to train
    train_df = add_rul_train(train_df)

    # Add rolling features (same idea as your v2 model)
    train_df = add_rolling_features(train_df, window=5)

    # Build training features
    feature_columns = [c for c in train_df.columns if c not in ["engine_id", "cycle", "RUL"]]
    X_train = train_df[feature_columns]
    y_train = train_df["RUL"]

    # Train a regressor (for feature importance + predictions)
    model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # Prepare test truth + rolling features
    test_df = build_test_truth(test_df, rul_path)
    test_df = add_rolling_features(test_df, window=5)

    X_test = test_df[feature_columns]
    test_df["Predicted_RUL"] = model.predict(X_test)

    # ---------- Chart 1: Predicted vs True RUL (snapshot per engine) ----------
    latest = (
        test_df.sort_values(["engine_id", "cycle"])
        .groupby("engine_id")
        .tail(1)
        .copy()
    )

    plt.figure(figsize=(7, 7))
    plt.scatter(latest["True_RUL"], latest["Predicted_RUL"])
    max_val = max(latest["True_RUL"].max(), latest["Predicted_RUL"].max())
    plt.plot([0, max_val], [0, max_val])
    plt.title("Predicted vs True RUL (Test Engines)")
    plt.xlabel("True RUL (cycles)")
    plt.ylabel("Predicted RUL (cycles)")
    plt.tight_layout()
    plt.savefig("assets/pred_vs_true_rul.png", dpi=200)
    plt.close()

    # ---------- Chart 2: Top 10 risky engines (lowest predicted RUL) ----------
    top_risky = latest.sort_values("Predicted_RUL").head(10)

    plt.figure(figsize=(10, 5))
    plt.bar(top_risky["engine_id"].astype(str), top_risky["Predicted_RUL"])
    plt.title("Top 10 Engines at Risk (Lowest Predicted RUL)")
    plt.xlabel("Engine ID")
    plt.ylabel("Predicted RUL (cycles)")
    plt.tight_layout()
    plt.savefig("assets/top10_risky_engines.png", dpi=200)
    plt.close()

    # ---------- Chart 3: Feature Importance (Top 15) ----------
    importances = pd.Series(model.feature_importances_, index=feature_columns).sort_values(ascending=False).head(15)
    plt.figure(figsize=(10, 6))
    plt.bar(importances.index, importances.values)
    plt.xticks(rotation=75, ha="right")
    plt.title("Top 15 Feature Importances (Random Forest)")
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.savefig("assets/top_feature_importance.png", dpi=200)
    plt.close()

    print("✅ Charts saved in assets/:")
    print(" - assets/pred_vs_true_rul.png")
    print(" - assets/top10_risky_engines.png")
    print(" - assets/top_feature_importance.png")