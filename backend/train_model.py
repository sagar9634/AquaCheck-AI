"""
train_model.py
--------------
Trains machine-learning models FROM SCRATCH on the water-quality CSV,
compares them, and saves the best one for the web app.

Run:  python train_model.py
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "data", "water_potability.csv")
MODEL_DIR = os.path.join(HERE, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURES = ["ph", "Hardness", "Solids", "Chloramines", "Sulfate",
            "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"]
TARGET = "Potability"

# Friendly info the website uses to build the sliders
INFO = {
    "ph":              {"label": "pH",                "unit": "",        "hint": "How acidic or basic the water is"},
    "Hardness":        {"label": "Hardness",          "unit": "mg/L",    "hint": "Calcium & magnesium dissolved in water"},
    "Solids":          {"label": "Total Dissolved Solids", "unit": "ppm", "hint": "Minerals and salts dissolved in water"},
    "Chloramines":     {"label": "Chloramines",       "unit": "ppm",     "hint": "Disinfectant used in public water"},
    "Sulfate":         {"label": "Sulfate",           "unit": "mg/L",    "hint": "Naturally occurring mineral"},
    "Conductivity":    {"label": "Conductivity",      "unit": "μS/cm",   "hint": "How well water conducts electricity"},
    "Organic_carbon":  {"label": "Organic Carbon",    "unit": "ppm",     "hint": "Decaying natural matter"},
    "Trihalomethanes": {"label": "Trihalomethanes",   "unit": "μg/L",    "hint": "By-products of chlorination"},
    "Turbidity":       {"label": "Turbidity",         "unit": "NTU",     "hint": "Cloudiness of the water"},
}


def make_presets(df, model, n=40):
    """Median of the n samples the model is most sure are safe / unsafe."""
    prob = model.predict_proba(df[FEATURES])[:, 1]
    order = np.argsort(prob)
    risky, good = df.iloc[order[:n]], df.iloc[order[-n:]]
    pick = lambda d: {f: round(float(d[f].median()), 2) for f in FEATURES}
    return {"Good sample": pick(good), "Risky sample": pick(risky)}


def main():
    if not os.path.exists(DATA_PATH):
        raise SystemExit(f"Dataset not found at {DATA_PATH}\n"
                         "Put the Kaggle CSV there, or run generate_sample_data.py first.")

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows, {df[FEATURES].isna().sum().sum()} missing values")

    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # Two candidate models. Both fill missing values with the column median.
    candidates = {
        "Logistic Regression": make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced")),
        "Random Forest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                   class_weight="balanced", random_state=42, n_jobs=-1)),
    }

    results, best_name, best_f1 = {}, None, -1
    for name, model in candidates.items():
        cv = cross_val_score(model, X_train, y_train, cv=5, scoring="f1").mean()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        f1 = f1_score(y_test, pred)
        results[name] = {
            "accuracy": round(accuracy_score(y_test, pred), 4),
            "precision": round(precision_score(y_test, pred), 4),
            "recall": round(recall_score(y_test, pred), 4),
            "f1": round(f1, 4),
            "cv_f1": round(cv, 4),
            "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        }
        print(f"{name:20s} acc={results[name]['accuracy']}  f1={results[name]['f1']}  cv_f1={results[name]['cv_f1']}")
        if f1 > best_f1:
            best_name, best_f1 = name, f1

    best = candidates[best_name]
    print(f"\nBest model: {best_name}")
    joblib.dump(best, os.path.join(MODEL_DIR, "model.joblib"))

    # Feature importance (for the chatbot + charts)
    final = best[-1]
    if hasattr(final, "feature_importances_"):
        imp = final.feature_importances_
    else:
        imp = np.abs(final.coef_[0]) / np.abs(final.coef_[0]).sum()
    importance = {f: round(float(v), 4) for f, v in zip(FEATURES, imp)}

    # Per-feature ranges used by the website
    potable = df[df[TARGET] == 1]
    features_meta = []
    for f in FEATURES:
        col = df[f].dropna()
        features_meta.append({
            "key": f, **INFO[f],
            "min": round(float(col.quantile(0.005)), 2),
            "max": round(float(col.quantile(0.995)), 2),
            "median": round(float(col.median()), 2),
            # the "comfort zone": where most potable samples in the data fall
            "safe_low": round(float(potable[f].quantile(0.10)), 2),
            "safe_high": round(float(potable[f].quantile(0.90)), 2),
        })

    meta = {
        "best_model": best_name,
        "dataset_rows": int(len(df)),
        "potable_share": round(float(y.mean()), 3),
        "metrics": results[best_name],
        "all_metrics": results,
        "importance": importance,
        "features": features_meta,
        # one-click example samples for the demo (built from the model's most
        # confident "safe" and "unsafe" samples, so the demo shows clear contrast)
        "presets": make_presets(df, best),
    }
    with open(os.path.join(MODEL_DIR, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print("Saved model/model.joblib and model/meta.json")


if __name__ == "__main__":
    main()
