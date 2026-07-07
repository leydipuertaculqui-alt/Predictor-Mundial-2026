"""Núcleo del esquema iterativo del predictor.

run_stage() ejecuta una única fase y nunca intenta encadenar predicciones entre fases.
La salvaguarda advance_to_next_stage_guard garantiza que las fases posteriores solo
se predicen cuando sus equipos ya vienen definidos por resultados reales.
"""
import numpy as np
import pandas as pd

from src import config as cfg, data_loader as dl, evaluation as ev, feature_engineering as fe, models as mdl
from src.logging_utils import get_logger
from src import model_manager
from src import metrics_history as mh
from src import prediction_history as ph
from src import explainability

logger = get_logger(__name__)


def _load_runtime_inputs() -> tuple:
    cfg.ensure_directories()
    teams_df = dl.load_teams_static()
    players_df = dl.load_players()
    h2h_df = dl.load_historical_h2h()
    historical_df = dl.load_historical_matches()
    bracket_map = dl.load_bracket_map()
    real_results = dl.load_matches_played()
    return teams_df, players_df, h2h_df, historical_df, bracket_map, real_results


def train_all_models(train_df: pd.DataFrame):
    X_train = train_df[cfg.ALL_FEATURES]
    y_clf = train_df["winner"]

    logger.info("Entrenando clasificadores y regresores para la fase actual")
    classifiers = mdl.build_classifiers()
    for _, pipe in classifiers.items():
        pipe.fit(X_train, y_clf)

    reg_a, reg_b = mdl.build_score_regressors()
    reg_a.fit(X_train, train_df["score_a_90"])
    reg_b.fit(X_train, train_df["score_b_90"])

    pen_df = train_df[train_df["went_to_penalties"] == True]
    penalty_model = None
    if len(pen_df) >= 6:
        penalty_model = mdl.build_penalty_model()
        penalty_model.fit(pen_df[cfg.ALL_FEATURES], pen_df["winner"])

    return {
    "classifiers": classifiers,
    "score_models": {
        "team_a": reg_a,
        "team_b": reg_b,
    },
    "penalty_model": penalty_model,
}

def predict_stage(fixtures_feat: pd.DataFrame, classifiers, reg_a, reg_b, penalty_model):
    X = fixtures_feat[cfg.ALL_FEATURES]
    results = {}
    for name, pipe in classifiers.items():
        proba = pipe.predict_proba(X)[:, 1]
        results[name] = proba

    score_a = np.clip(np.round(reg_a.predict(X)).astype(int), 0, None)
    score_b = np.clip(np.round(reg_b.predict(X)).astype(int), 0, None)

    if penalty_model is not None:
        pen_proba = penalty_model.predict_proba(X)[:, 1]
    else:
        pen_proba = np.full(len(X), 0.5)

    return results, score_a, score_b, pen_proba

def weighted_probability(clf_results, weights, index):
    """
    Calcula la probabilidad ponderada usando el rendimiento histórico
    de cada clasificador.
    """
    total = 0.0

    for model_name in clf_results:
        total += (
            clf_results[model_name][index]
            * weights.get(model_name, 0)
        )

    return float(total)

def assemble_report(
    stage,
    fixtures,
    clf_results,
    score_a,
    score_b,
    pen_proba,
    model_weights,
    real_results=None,
):
    matches_out = []
    for i, match in enumerate(fixtures):
        proba_avg = weighted_probability(clf_results,model_weights,i,)
        entry = {
            "match_id": match["match_id"],
            "team_a": match["team_a"],
            "team_b": match["team_b"],
            "predicted_score_90": {"a": int(score_a[i]), "b": int(score_b[i])},
            "advance_probability": {name: round(float(clf_results[name][i]), 3) for name in clf_results},
            "advance_probability_avg_a": round(float(proba_avg), 3),
            "penalty_scenario_probability": round(float(pen_proba[i]), 3) if score_a[i] == score_b[i] else None,
            "predicted_winner": match["team_a"] if proba_avg >= 0.5 else match["team_b"],
        }
        if real_results is not None:
            real = real_results[real_results["match_id"] == match["match_id"]]
            if not real.empty:
                result = real.iloc[0]
                entry["actual_result"] = {
                    "score_90": {"a": int(result["score_a_90"]), "b": int(result["score_b_90"])},
                    "went_to_penalties": bool(result["went_to_penalties"]),
                    "winner": result["winner_final"],
                }
                entry["hit"] = bool(entry["predicted_winner"] == result["winner_final"])
        matches_out.append(entry)
    return {"stage": stage, "matches": matches_out}


def add_validation_summary(report, fixtures, fixtures_feat, clf_results, score_a, score_b, real_results):
    ids = [match["match_id"] for match in fixtures]
    known_ids = set(real_results["match_id"])
    mask = [match_id in known_ids for match_id in ids]
    n_validable = sum(mask)
    report["n_matches_total"] = len(ids)
    report["n_matches_validated"] = n_validable
    if n_validable == 0:
        report["validation_note"] = (
            "Ningún partido de esta fase tiene resultado real todavía; "
            "todas las predicciones están pendientes de confirmación."
        )
        return report

    real = real_results.set_index("match_id").loc[[match_id for match_id in ids if match_id in known_ids]]
    idx_mask = np.array(mask)
    y_true = (real["winner_final"].values == fixtures_feat.loc[idx_mask, "team_a"].values).astype(int)

    validation = {}
    for name, proba in clf_results.items():
        y_pred = (proba[idx_mask] >= 0.5).astype(int)
        validation[name] = ev.evaluate_classifier(y_true, y_pred)
    validation["score_model"] = ev.evaluate_score_model(
        real["score_a_90"].values,
        score_a[idx_mask],
        real["score_b_90"].values,
        score_b[idx_mask],
    )
    if n_validable < len(ids):
        validation["_note"] = f"Validado solo con {n_validable}/{len(ids)} partidos ya jugados."
    report["validation_summary"] = validation
    return report


def feature_importance(classifiers) -> dict:
    out = {}
    for name, pipe in classifiers.items():
        clf = pipe.named_steps["clf"]
        if hasattr(clf, "feature_importances_"):
            importance = clf.feature_importances_
            pairs = sorted(zip(cfg.ALL_FEATURES, importance), key=lambda item: -item[1])[:8]
            out[name] = [(feature, round(float(value), 4)) for feature, value in pairs]
        elif hasattr(clf, "coef_"):
            importance = clf.coef_[0]
            pairs = sorted(zip(cfg.ALL_FEATURES, importance), key=lambda item: -abs(item[1]))[:8]
            out[name] = [(feature, round(float(value), 4)) for feature, value in pairs]
    return out


def run_stage(stage: str, validate: bool = True) -> dict:
    """Ejecuta una fase completa de predicción sin encadenar otras fases."""
    history_file = None
    stage = cfg.validate_stage(stage)
    logger.info("Ejecutando predicción para la fase %s", stage)

    teams_df, players_df, h2h_df, historical_df, bracket_map, real_results = _load_runtime_inputs()

    train_df = fe.build_training_table(
        historical_df,
        teams_df,
        players_df,
        h2h_df,
    )

    if train_df.empty:
        raise RuntimeError(
            "Tabla de entrenamiento vacía: revisa historical_matches.csv y teams_static.csv"
        )

    # -----------------------------------------------------
    # Cargar modelos si ya existen; si no, entrenarlos.
    # -----------------------------------------------------

    if model_manager.models_exist():

        logger.info("Cargando modelos previamente entrenados...")

        models = model_manager.load_models()

    else:

        logger.info("No existen modelos guardados. Entrenando modelos...")

        models = train_all_models(train_df)

    classifiers = models["classifiers"]
    reg_a = models["score_models"]["team_a"]
    reg_b = models["score_models"]["team_b"]
    penalty_model = models["penalty_model"]

    # -----------------------------------------------------
    
    predicted_file = cfg.DATA_PROCESSED / f"{stage}_predicted_fixtures.csv"

    if predicted_file.exists():

     logger.info(
        "Usando fixtures predichos para %s",
        stage
     )

     fixtures = pd.read_csv(predicted_file).to_dict("records")

    else:

     fixtures = dl.get_stage_fixtures(
        stage,
        bracket_map,
        real_results,
    )


    fixtures_feat = fe.build_features(
        fixtures,
        teams_df,
        players_df,
        h2h_df,
        historical_df,
        stage,
    )
    
    clf_results, score_a, score_b, pen_proba = predict_stage(
        fixtures_feat,
        classifiers,
        reg_a,
        reg_b,
        penalty_model,
    )

    # ==========================================================
    # Pesos iniciales (todos iguales antes de la validación)
    # ==========================================================

    model_weights = {
        "logreg": 1 / 3,
        "random_forest": 1 / 3,
        "xgboost": 1 / 3,
    }

    # ==========================================================
    # Generar reporte inicial
    # ==========================================================

    report = assemble_report(
        stage,
        fixtures,
        clf_results,
        score_a,
        score_b,
        pen_proba,
        model_weights,
        real_results=real_results if validate else None,
    )

    # ==========================================================
    # Validación contra resultados reales
    # ==========================================================

    if validate:

        report = add_validation_summary(
            report,
            fixtures,
            fixtures_feat,
            clf_results,
            score_a,
            score_b,
            real_results,
        )

        # Calcular pesos usando el accuracy histórico
        if "validation_summary" in report:

            metrics = report["validation_summary"]

            scores = {}

            for model in ("logreg", "random_forest", "xgboost"):

                if model in metrics:
                    scores[model] = metrics[model]["accuracy"]

            total = sum(scores.values())

            if total > 0:

                model_weights = {
                    model: value / total
                    for model, value in scores.items()
                }

        # Buscar el mejor clasificador SOLO si existe validación
        best = None

        if "validation_summary" in report:

          best = ev.select_best_classifier(report["validation_summary"])
          ranking = ev.rank_models(report["validation_summary"])

          report["model_ranking"] = ranking

          if best is not None:
           report["best_classifier"] = best["best_model"]
           report["best_classifier_metrics"] = best["metrics"]

        # ======================================================
        # Volver a construir el reporte usando los nuevos pesos
        # ======================================================

        report = assemble_report(
            stage,
            fixtures,
            clf_results,
            score_a,
            score_b,
            pen_proba,
            model_weights,
            real_results=real_results,
        )

        report = add_validation_summary(
            report,
            fixtures,
            fixtures_feat,
            clf_results,
            score_a,
            score_b,
            real_results,
        )
        if best is not None:
         report["best_classifier"] = best["best_model"]
         report["best_classifier_metrics"] = best["metrics"]

    
    report["top_features"] = feature_importance(classifiers)

    if "best_classifier" in report:

        report["decision_support"] = explainability.decision_support(
          classifiers,
          report["best_classifier"],
          cfg.ALL_FEATURES,
        )
        
    report["train_set_size"] = len(train_df)
    report["uses_xgboost_native"] = mdl.HAS_XGBOOST
    
    report["model_weights"] = {
    k: round(v, 3)
    for k, v in model_weights.items()
    }
    
    history_file = None
    best = None 
    
    if validate and "validation_summary" in report:

     history_file = mh.save_metrics_history(
         stage,
         report["validation_summary"]
        )

     report["metrics_history_file"] = str(history_file)

    print(">>> Historial guardado:", history_file)
    
    
    prediction_file = ph.save_predictions(
     stage,
     report["matches"]
    )

    report["prediction_history_file"] = str(prediction_file)
    
    return report
    
def advance_to_next_stage_guard(stage: str, next_stage: str):
    """Impide predecir una fase posterior sin resultados reales para todos sus partidos."""
    bracket_map = dl.load_bracket_map()
    real_results = dl.load_matches_played()
    try:
        dl.get_stage_fixtures(next_stage, bracket_map, real_results)
    except RuntimeError as error:
        raise RuntimeError(f"No se puede avanzar de {stage} a {next_stage} todavía: {error}") from error

    logger.info("Todos los partidos de %s están resueltos con resultados reales", next_stage)
    print(f"[ok] Todos los partidos de {next_stage} están resueltos con resultados reales. Se puede predecir.")
