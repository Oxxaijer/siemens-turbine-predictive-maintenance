import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("engine_health_snapshot.csv")

plt.figure(figsize=(10,5))
plt.hist(df["HealthScore_0_100"], bins=15)
plt.title("Fleet HealthScore Distribution (Latest Snapshot)")
plt.xlabel("HealthScore (0–100)")
plt.ylabel("Number of engines")
plt.tight_layout()
plt.savefig("assets/fleet_health_distribution.png", dpi=200)
plt.close()

print("✅ Saved: assets/fleet_health_distribution.png")