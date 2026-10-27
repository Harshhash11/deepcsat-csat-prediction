"""csat_utils.py - feature engineering + preprocessing for the Shopzilla CSAT project.

Kept in a real module (not notebook-only code) so that the fitted preprocessor
saved with joblib can be loaded again in a separate process (e.g. predict.py).
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

# --------------------------------------------------------------------------
# Column groups
# --------------------------------------------------------------------------
CAT_COLS = ["channel_name", "category", "Sub-category", "Product_category",
            "Tenure Bucket", "Agent Shift", "Manager", "Supervisor", "Customer_City"]
NUM_COLS = ["response_min", "log_response_min", "resp_negative_flag", "rep_hour", "rep_dow",
            "rep_day", "is_weekend", "survey_lag_days", "has_order_info",
            "order_to_issue_days", "log_item_price", "has_order_id"]
AGENT_COL = "Agent_name"
RAW_INPUT_COLS = ["channel_name", "category", "Sub-category", "Customer Remarks", "Order_id",
                  "order_date_time", "Issue_reported at", "issue_responded", "Survey_response_Date",
                  "Customer_City", "Product_category", "Item_price", "Agent_name", "Supervisor",
                  "Manager", "Tenure Bucket", "Agent Shift"]


# --------------------------------------------------------------------------
# Feature engineering (row-level only -> safe for real-time scoring)
# --------------------------------------------------------------------------
def engineer(raw: pd.DataFrame) -> pd.DataFrame:
    """Turn raw ticket rows into model-ready features. No target information is used."""
    d = pd.DataFrame(index=raw.index)
    rep = pd.to_datetime(raw["Issue_reported at"], format="%d/%m/%Y %H:%M", errors="coerce")
    resp = pd.to_datetime(raw["issue_responded"], format="%d/%m/%Y %H:%M", errors="coerce")
    survey = pd.to_datetime(raw["Survey_response_Date"], format="%d-%b-%y", errors="coerce")
    order = pd.to_datetime(raw["order_date_time"], format="%d/%m/%Y %H:%M", errors="coerce")

    # Responsiveness: the single most important operational driver (see EDA).
    rt = (resp - rep).dt.total_seconds() / 60.0
    d["resp_negative_flag"] = (rt < 0).astype(int)          # data-entry anomaly (responded before reported)
    d["response_min"] = rt.clip(lower=0).fillna(0)
    d["log_response_min"] = np.log1p(d["response_min"])

    # Calendar features
    d["rep_hour"] = rep.dt.hour.fillna(12)
    d["rep_dow"] = rep.dt.dayofweek.fillna(0)
    d["rep_day"] = rep.dt.day.fillna(15)
    d["is_weekend"] = (d["rep_dow"] >= 5).astype(int)
    d["survey_lag_days"] = (survey - rep.dt.normalize()).dt.days.clip(-5, 31).fillna(0)

    # Order information (missing for ~80% of tickets = not order-linked)
    d["has_order_info"] = order.notna().astype(int)
    d["order_to_issue_days"] = ((rep - order).dt.total_seconds() / 86400).clip(0, 400).fillna(0)
    d["log_item_price"] = np.log1p(pd.to_numeric(raw["Item_price"], errors="coerce").fillna(0).clip(lower=0))
    d["has_order_id"] = raw["Order_id"].notna().astype(int)

    # Categoricals (missing -> explicit 'Unknown' level)
    for c in CAT_COLS:
        d[c] = raw[c].fillna("Unknown").astype(str).str.strip()
    d[AGENT_COL] = raw[AGENT_COL].fillna("Unknown").astype(str)
    return d


# --------------------------------------------------------------------------
# Agent-level target encoding (out-of-fold, smoothed) - 1,371 agents is too many for one-hot
# --------------------------------------------------------------------------
class AgentStats(BaseEstimator, TransformerMixin):
    """Smoothed historical agent performance.

    Outputs [agent_low_rate, agent_mean_csat, log_agent_volume].
    fit_transform uses K-fold out-of-fold encoding so a row never sees its own label (no leakage).
    """

    def __init__(self, smoothing=20, n_splits=5, random_state=42):
        self.smoothing = smoothing
        self.n_splits = n_splits
        self.random_state = random_state

    def _stats(self, agents, y):
        df = pd.DataFrame({"a": np.asarray(agents).ravel(), "low": (y <= 2).astype(float), "s": y.astype(float)})
        g = df.groupby("a").agg(n=("low", "size"), low=("low", "sum"), s=("s", "sum"))
        return g

    def _encode(self, agents, g, prior_low, prior_s):
        a = pd.Series(np.asarray(agents).ravel())
        n = a.map(g["n"]).fillna(0).to_numpy()
        low = a.map(g["low"]).fillna(0).to_numpy()
        s = a.map(g["s"]).fillna(0).to_numpy()
        m = self.smoothing
        return np.c_[(low + m * prior_low) / (n + m), (s + m * prior_s) / (n + m), np.log1p(n)]

    def fit(self, X, y):
        y = np.asarray(y)
        self.prior_low_ = float((y <= 2).mean())
        self.prior_s_ = float(y.mean())
        self.stats_ = self._stats(X, y)
        return self

    def fit_transform(self, X, y=None, **kw):
        y = np.asarray(y)
        self.fit(X, y)
        X = np.asarray(X).ravel()
        out = np.zeros((len(X), 3))
        for tr, va in KFold(self.n_splits, shuffle=True, random_state=self.random_state).split(X):
            g = self._stats(X[tr], y[tr])
            out[va] = self._encode(X[va], g, self.prior_low_, self.prior_s_)
        return out

    def transform(self, X):
        return self._encode(X, self.stats_, self.prior_low_, self.prior_s_)


def _shift_one(x):
    """OrdinalEncoder gives -1 to unseen levels; shift so 0 = unknown/rare, 1.. = real levels."""
    return x + 1


# --------------------------------------------------------------------------
# Preprocessors
# --------------------------------------------------------------------------
def make_dense_preprocessor(extra_num=()):
    """One-hot (rare levels pooled) + scaled numerics + agent stats -> dense matrix for sklearn / MLP."""
    num = list(NUM_COLS) + list(extra_num)
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=100, sparse_output=False), CAT_COLS),
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
        ("agent", Pipeline([("enc", AgentStats()), ("sc", StandardScaler())]), [AGENT_COL]),
    ])


def make_embed_preprocessor(extra_num=()):
    """Integer codes for embedding layers (first len(CAT_COLS) columns) + scaled numerics + agent stats."""
    num = list(NUM_COLS) + list(extra_num)
    return ColumnTransformer([
        ("cat", Pipeline([("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                                                 encoded_missing_value=-1, min_frequency=100)),
                          ("shift", FunctionTransformer(_shift_one))]), CAT_COLS),
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
        ("agent", Pipeline([("enc", AgentStats()), ("sc", StandardScaler())]), [AGENT_COL]),
    ])


# --------------------------------------------------------------------------
# Wrapper that couples a fitted preprocessor with a Keras model -> predict_proba(DataFrame)
# --------------------------------------------------------------------------
class AnnPipeline:
    def __init__(self, pre, model, kind="dense", n_cat=0, class_weights=None, offsets=None):
        self.pre, self.model, self.kind, self.n_cat = pre, model, kind, n_cat
        self.offsets = None if offsets is None else np.asarray(offsets, dtype="int32")
        self.class_weights = np.ones(5) if class_weights is None else np.asarray(class_weights)

    def _inputs(self, X):
        Xt = self.pre.transform(X)
        if self.kind == "embed":
            cats = Xt[:, :self.n_cat].astype("int32")
            if self.offsets is not None:          # one shared embedding table -> shift each column into its own id range
                cats = cats + self.offsets
            return [cats, Xt[:, self.n_cat:].astype("float32")]
        return Xt.astype("float32")

    def predict_proba(self, X):
        return np.asarray(self.model.predict(self._inputs(X), batch_size=4096, verbose=0))

    def predict(self, X):
        """Predicted CSAT (1-5) after optional per-class probability re-weighting."""
        return (self.predict_proba(X) * self.class_weights).argmax(1) + 1
