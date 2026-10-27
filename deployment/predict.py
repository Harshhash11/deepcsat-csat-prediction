"""Score new Shopzilla support tickets locally.

Usage:   python predict.py tickets.csv scored.csv
Input :  a CSV with the same raw columns as eCommerce_Customer_support_data.csv (CSAT Score is NOT needed).
Output:  the input row ids + predicted CSAT, P(each score), P(dissatisfied) and a risk tier.
"""
import os, sys, json
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)

import numpy as np, pandas as pd, joblib
from tensorflow import keras
from csat_utils import engineer, AnnPipeline, RAW_INPUT_COLS

_CACHE = {}

def load_pipeline():
    """Load (and cache) preprocessor + model + metadata."""
    if not _CACHE:
        meta = json.load(open(os.path.join(HERE, "meta.json")))
        pre = joblib.load(os.path.join(HERE, "preprocessor.joblib"))
        model = keras.models.load_model(os.path.join(HERE, "csat_model.keras"), compile=False)
        _CACHE["pipe"] = AnnPipeline(pre, model, "embed", meta["n_cat"], meta["class_weights"], meta["offsets"])
        _CACHE["meta"] = meta
    return _CACHE["pipe"], _CACHE["meta"]

def score_dataframe(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw ticket rows in -> scored DataFrame out. Missing optional columns are treated as 'not available'."""
    pipe, meta = load_pipeline()
    raw = raw.copy()
    for c in RAW_INPUT_COLS:
        if c not in raw.columns:
            raw[c] = np.nan
    proba = pipe.predict_proba(engineer(raw))
    risk = proba[:, 0] + proba[:, 1]
    th = meta["risk_thresholds"]
    out = pd.DataFrame(proba, columns=[f"p_csat_{k}" for k in range(1, 6)], index=raw.index).round(4)
    out.insert(0, "predicted_csat", pipe.predict(engineer(raw)))
    out.insert(1, "expected_csat", (proba * np.arange(1, 6)).sum(1).round(2))
    out["p_dissatisfied"] = risk.round(4)
    out["risk_tier"] = np.where(risk >= th["high"], "HIGH", np.where(risk >= th["medium"], "MEDIUM", "LOW"))
    if "Unique id" in raw.columns:
        out.insert(0, "Unique id", raw["Unique id"].values)
    return out

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    scored = score_dataframe(pd.read_csv(sys.argv[1]))
    scored.to_csv(sys.argv[2], index=False)
    print(scored.head(10).to_string(index=False))
    print(f"\nScored {len(scored):,} tickets -> {sys.argv[2]}  |  risk tiers: {scored.risk_tier.value_counts().to_dict()}")
