import pandas as pd

# Column names based on NASA C-MAPSS format
columns = (
    ["engine_id", "cycle"] +
    [f"setting_{i}" for i in range(1, 4)] +
    [f"sensor_{i}" for i in range(1, 22)]
)

# Load the dataset (multiple spaces -> use \s+)
df = pd.read_csv("data/train_FD001.txt", sep=r"\s+", header=None)

# Keep first 26 columns (some files include extra blanks at the end)
df = df.iloc[:, :26]
df.columns = columns

# Calculate Remaining Useful Life (RUL)
max_cycle_per_engine = df.groupby("engine_id")["cycle"].max()
df["RUL"] = df["engine_id"].map(max_cycle_per_engine) - df["cycle"]

print("✅ Dataset loaded with column names + RUL added")
print("Shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nRUL statistics:")
print(df["RUL"].describe())

print("\nUnique engines:", df["engine_id"].nunique())