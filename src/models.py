"""
Modelos requeridos: Regresión Logística, Random Forest, XGBoost (con fallback a
GradientBoosting de sklearn si el paquete xgboost no está disponible en el entorno),
y un modelo de marcador exacto (regresión, objetivo Poisson cuando es posible).
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from src import config as cfg

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("[aviso] paquete 'xgboost' no disponible en este entorno; "
          "se usa GradientBoosting de sklearn como sustituto funcionalmente equivalente. "
          "Instala 'pip install xgboost' en tu máquina para usar XGBoost real.")


def make_preprocessor():
    return ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), cfg.NUMERIC_FEATURES),
        ("cat", SimpleImputer(strategy="most_frequent"), cfg.CATEGORICAL_FEATURES),
    ])


def build_classifiers():
    models = {
        "logreg": Pipeline([
            ("prep", make_preprocessor()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]),
        "random_forest": Pipeline([
            ("prep", make_preprocessor()),
            ("clf", RandomForestClassifier(
                n_estimators=500, max_depth=5, min_samples_leaf=2,
                class_weight="balanced", random_state=cfg.RANDOM_STATE)),
        ]),
    }
    if HAS_XGBOOST:
        models["xgboost"] = Pipeline([
            ("prep", make_preprocessor()),
            ("clf", XGBClassifier(
                n_estimators=200, max_depth=3, learning_rate=0.08,
                subsample=0.8, colsample_bytree=0.8,
                eval_metric="logloss", random_state=cfg.RANDOM_STATE)),
        ])
    else:
        models["xgboost"] = Pipeline([
            ("prep", make_preprocessor()),
            ("clf", GradientBoostingClassifier(
                n_estimators=200, max_depth=3, learning_rate=0.08,
                random_state=cfg.RANDOM_STATE)),
        ])
    return models


def build_score_regressors():
    if HAS_XGBOOST:
        reg_a = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.08,
                              objective="count:poisson", random_state=cfg.RANDOM_STATE)
        reg_b = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.08,
                              objective="count:poisson", random_state=cfg.RANDOM_STATE)
    else:
        reg_a = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.08,
                                           loss="squared_error", random_state=cfg.RANDOM_STATE)
        reg_b = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.08,
                                           loss="squared_error", random_state=cfg.RANDOM_STATE)
    model_score_a = Pipeline([("prep", make_preprocessor()), ("reg", reg_a)])
    model_score_b = Pipeline([("prep", make_preprocessor()), ("reg", reg_b)])
    return model_score_a, model_score_b


def build_penalty_model():
    return Pipeline([
        ("prep", make_preprocessor()),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
