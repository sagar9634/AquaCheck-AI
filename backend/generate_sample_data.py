"""
generate_sample_data.py
-----------------------
Creates a SYNTHETIC water-quality dataset with the same columns as the
Kaggle "Water Potability" dataset, so the project runs out of the box.

>>> For your real project, put the real Kaggle file at:
>>>     backend/data/water_potability.csv
>>> and train_model.py will use it instead (this script is then not needed).
"""
import os
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 3300

# Feature distributions roughly match the public Kaggle dataset.
df = pd.DataFrame({
    "ph":              np.clip(rng.normal(7.08, 1.59, N), 0, 14),
    "Hardness":        np.clip(rng.normal(196, 33, N), 47, 324),
    "Solids":          np.clip(rng.normal(22000, 8800, N), 320, 61000),
    "Chloramines":     np.clip(rng.normal(7.1, 1.58, N), 0.35, 13.1),
    "Sulfate":         np.clip(rng.normal(333, 41, N), 129, 481),
    "Conductivity":    np.clip(rng.normal(426, 80, N), 181, 753),
    "Organic_carbon":  np.clip(rng.normal(14.3, 3.3, N), 2.2, 28.3),
    "Trihalomethanes": np.clip(rng.normal(66.4, 16.2, N), 0.7, 124),
    "Turbidity":       np.clip(rng.normal(3.97, 0.78, N), 1.45, 6.74),
})

# A made-up "risk score": bad pH, high turbidity, high THM, high solids... = riskier
z = (df - df.mean()) / df.std()
score = (
    -0.60 * z["ph"].abs()
    - 0.50 * z["Turbidity"].clip(lower=0)
    - 0.50 * z["Trihalomethanes"].clip(lower=0)
    - 0.40 * z["Solids"].clip(lower=0)
    - 0.30 * z["Chloramines"].clip(lower=0)
    - 0.25 * z["Organic_carbon"].clip(lower=0)
    - 0.20 * z["Sulfate"].abs()
    + rng.normal(0, 0.55, N)           # real life is noisy
)
df["Potability"] = (score > np.quantile(score, 0.60)).astype(int)

# Real data has missing values too (pH, Sulfate, Trihalomethanes)
for col, frac in [("ph", 0.15), ("Sulfate", 0.24), ("Trihalomethanes", 0.05)]:
    df.loc[rng.random(N) < frac, col] = np.nan

out_dir = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(out_dir, exist_ok=True)
path = os.path.join(out_dir, "water_potability.csv")
df.round(4).to_csv(path, index=False)
print(f"Saved {len(df)} rows -> {path}")
print(df["Potability"].value_counts().to_string())
