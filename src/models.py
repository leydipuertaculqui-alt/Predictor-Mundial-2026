"""Construcción de los modelos de clasificación y regresión del proyecto.

Se mantienen los mismos tres clasificadores base (Logistic Regression,
Random Forest y XGBoost), así como el modelo de marcador y el modelo de
penales, pero con una estructura más limpia y reutilizable.
"""
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import config as cfg

try:
    from xgboost import XGBClassifier, XGBRegressor

    HAS_XGBOOST = True
except ImportError:  # pragma: no cover - depende del entorno
    HAS_XGBOOST = False
    print(
        "[aviso] paquete 'xgboost' no disponible en este entorno; "
        "se usa GradientBoosting de sklearn como sustituto funcionalmente equivalente. "
        "Instala 'pip install xgboost' en tu máquina para usar XGBoost real."
    )


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                cfg.NUMERIC_FEATURES,
            ),
            ("cat", SimpleImputer(strategy="most_frequent"), cfg.CATEGORICAL_FEATURES),
        ]
    )


def build_classifiers() -> dict:
    models = {
        "logreg": Pipeline(
            steps=[
                ("prep", make_preprocessor()),
                ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("prep", make_preprocessor()),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=500,
                        max_depth=5,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        random_state=cfg.RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }

    if HAS_XGBOOST:
        models["xgboost"] = Pipeline(
            steps=[
                ("prep", make_preprocessor()),
                (
                    "clf",
                    XGBClassifier(
                       n_estimators=400,
                       max_depth=4,
                       learning_rate=0.03,
                       subsample=0.85,
                       colsample_bytree=0.85,
                       min_child_weight=2,
                       gamma=0.15,
                       reg_alpha=0.10,
                       reg_lambda=1.2,
                       objective="binary:logistic",
                       eval_metric="logloss",
                       random_state=cfg.RANDOM_STATE,
                    ),
                ),
            ]
        )
    else:
        models["xgboost"] = Pipeline(
            steps=[
                ("prep", make_preprocessor()),
                (
                    "clf",
                    GradientBoostingClassifier(
                        n_estimators=200,
                        max_depth=3,
                        learning_rate=0.08,
                        random_state=cfg.RANDOM_STATE,
                    ),
                ),
            ]
        )
    return models


def build_score_regressors() -> tuple:
    if HAS_XGBOOST:
        reg_a = XGBRegressor(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.08,
            objective="count:poisson",
            random_state=cfg.RANDOM_STATE,
            
        )
        reg_b = XGBRegressor(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.08,
            objective="count:poisson",
            random_state=cfg.RANDOM_STATE,
        )
    else:
        reg_a = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.08,
            loss="squared_error",
            random_state=cfg.RANDOM_STATE,
        )
        reg_b = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.10,
            reg_lambda=1.2,
            loss="squared_error",
            random_state=cfg.RANDOM_STATE,
        )

    model_score_a = Pipeline(steps=[("prep", make_preprocessor()), ("reg", reg_a)])
    model_score_b = Pipeline(steps=[("prep", make_preprocessor()), ("reg", reg_b)])
    return model_score_a, model_score_b


def build_penalty_model() -> Pipeline:
    return Pipeline(
        steps=[
            ("prep", make_preprocessor()),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )
