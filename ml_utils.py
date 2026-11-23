"""Shared helpers for the Sri Lankan vehicle price ensemble project.

Used by both the notebook (training) and app.py (dashboard), so that the
dashboard builds features exactly the same way the models were trained.
"""

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

# Listings were collected Dec 2024 - Feb 2025, so ages are measured from 2025.
REF_YEAR = 2025
# Sri Lanka's import ban on personal vehicles (since March 2020) ended on this date.
BAN_LIFT_DATE = "2025-02-01"

RENAME = {
    "Brand": "brand",
    "Model": "model",
    "YOM": "yom",
    "Engine (cc)": "engine_cc",
    "Gear": "gear",
    "Fuel Type": "fuel_type",
    "Millage(KM)": "mileage_km",
    "Town": "town",
    "Date": "date",
    "Leasing": "leasing",
    "Condition": "condition",
    "AIR CONDITION": "air_condition",
    "POWER STEERING": "power_steering",
    "POWER MIRROR": "power_mirror",
    "POWER WINDOW": "power_window",
    "Price": "price",
}

EQUIPMENT = ["air_condition", "power_steering", "power_mirror", "power_window"]

# Feature groups used by the models
MODEL_COL = ["model"]  # 1,555 distinct values -> target encoded
CAT_COLS = ["brand", "gear", "fuel_type", "leasing", "condition", "town"]  # one-hot
# mileage_km is NOT used: in this dataset it is always exactly age x 11,000 km (filled in by the
# dataset creator), so it carries no information beyond age. See notebook section 4.
NUM_COLS = ["age", "engine_cc", "listed_after_ban"] + EQUIPMENT
FEATURES = MODEL_COL + CAT_COLS + NUM_COLS
CATBOOST_CAT_COLS = MODEL_COL + CAT_COLS


def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns, tidy text values and drop exact duplicate listings."""
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")
    df = df.rename(columns=RENAME)
    for col in ["brand", "model", "town", "gear", "fuel_type", "leasing", "condition"] + EQUIPMENT:
        df[col] = df[col].astype(str).str.strip()
    df["brand"] = df["brand"].str.upper()
    df["model"] = df["model"].str.upper()
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering (Lesson 4, step 5). Works on training data and on dashboard input."""
    df = df.copy()
    df["age"] = (REF_YEAR - df["yom"]).clip(lower=1)
    if "listed_after_ban" not in df.columns:
        df["listed_after_ban"] = (pd.to_datetime(df["date"]) >= BAN_LIFT_DATE).astype(int)
    for col in EQUIPMENT:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = (df[col] == "Available").astype(int)
    return df


def make_preprocessor(scale: bool) -> ColumnTransformer:
    """Preprocessing used by every sklearn-style model.

    - model: target encoding (average log-price per model, learned with internal
      cross-fitting so it does not leak the row's own price)
    - other categories: one-hot, with rare values (< 10 listings) grouped together
    - numbers: standardised for ANN / SVR / linear models, left as-is for trees
    """
    num = StandardScaler() if scale else "passthrough"
    te = TargetEncoder(target_type="continuous", cv=KFold(5, shuffle=True, random_state=42))
    model_enc = Pipeline([("te", te), ("scale", StandardScaler())]) if scale else te
    return ColumnTransformer(
        [
            ("model_te", model_enc, MODEL_COL),
            (
                "cat_ohe",
                OneHotEncoder(min_frequency=10, handle_unknown="infrequent_if_exist", sparse_output=False),
                CAT_COLS,
            ),
            ("num", num, NUM_COLS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


class RFANNHybrid(RegressorMixin, BaseEstimator):
    """Residual hybrid: Random Forest first, then an ANN learns what the forest got wrong.

    fit:
      1. Out-of-fold RF predictions on the training set (cross_val_predict), so the ANN
         sees realistic RF errors rather than the near-perfect in-sample fit.
      2. Refit the RF on all training data.
      3. Train the ANN on [features + RF prediction] to predict the residual y - rf_pred.
    predict:
      rf_pred + ann_correction
    """

    def __init__(self, rf_pipeline, ann_pipeline, cv=5):
        self.rf_pipeline = rf_pipeline
        self.ann_pipeline = ann_pipeline
        self.cv = cv

    def fit(self, X, y):
        y = np.asarray(y)
        self.rf_ = clone(self.rf_pipeline)
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=42)
        rf_oof = cross_val_predict(self.rf_, X, y, cv=kf)
        self.rf_.fit(X, y)
        X_ann = X.copy()
        X_ann["rf_pred"] = rf_oof
        self.ann_ = clone(self.ann_pipeline)
        self.ann_.fit(X_ann, y - rf_oof)
        return self

    def predict(self, X):
        rf_pred = self.rf_.predict(X)
        X_ann = X.copy()
        X_ann["rf_pred"] = rf_pred
        return rf_pred + self.ann_.predict(X_ann)


def make_ann_preprocessor() -> ColumnTransformer:
    """Like make_preprocessor(scale=True) but also scales the RF prediction column."""
    model_enc = Pipeline([("te", TargetEncoder(target_type="continuous", cv=KFold(5, shuffle=True, random_state=42))), ("scale", StandardScaler())])
    return ColumnTransformer(
        [
            ("model_te", model_enc, MODEL_COL),
            (
                "cat_ohe",
                OneHotEncoder(min_frequency=10, handle_unknown="infrequent_if_exist", sparse_output=False),
                CAT_COLS,
            ),
            ("num", StandardScaler(), NUM_COLS + ["rf_pred"]),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


class CatBoostModel(RegressorMixin, BaseEstimator):
    """Thin sklearn wrapper around CatBoostRegressor.

    CatBoost reads the raw text categories itself (no one-hot / target encoding needed),
    using its own leakage-safe "ordered target statistics". The wrapper only exists so
    that GridSearchCV and StackingRegressor can clone it like any sklearn model.
    """

    def __init__(self, iterations=1000, learning_rate=0.08, depth=6, l2_leaf_reg=3.0, random_seed=42):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.random_seed = random_seed

    def fit(self, X, y):
        self.model_ = CatBoostRegressor(
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            depth=self.depth,
            l2_leaf_reg=self.l2_leaf_reg,
            random_seed=self.random_seed,
            cat_features=CATBOOST_CAT_COLS,
            verbose=0,
            thread_count=2,
        )
        self.model_.fit(X[FEATURES], y)
        return self

    def predict(self, X):
        return self.model_.predict(X[FEATURES])

    @property
    def feature_importances_(self):
        return self.model_.get_feature_importance()
