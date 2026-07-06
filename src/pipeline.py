"""
Núcleo del esquema iterativo (sección 7 del diseño). run_stage() ejecuta UNA fase;
nunca llama a otra fase. advance_requires_real_results() es la salvaguarda contra
la predicción en cadena.
"""
import json
import numpy as np
import pandas as pd
from src import config as cfg, data_loader as dl, feature_engineering as fe, models as mdl, evaluation as ev


def train_all_models(train_df: pd.DataFrame):
    X_train = train_df[cfg.ALL_FEATURES]
    y_clf = train_df["winner"]

    classifiers = mdl.build_classifiers()
    for name, pipe in classifiers.items():
        pipe.fit(X_train, y_clf)

    reg_a, reg_b = mdl.build_score_regressors()
    reg_a.fit(X_train, train_df["score_a_90"])
    reg_b.fit(X_train, train_df["score_b_90"])

    pen_df = train_df[train_df["went_to_penalties"] == True]
    penalty_model = None
    if len(pen_df) >= 6:  # umbral mínimo para no sobreajustar un modelo casi vacío
        penalty_model = mdl.build_penalty_model()
        penalty_model.fit(pen_df[cfg.ALL_FEATURES], pen_df["winner"])

    return classifiers, reg_a, reg_b, penalty_model


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
        pen_proba = np.full(len(X), 0.5)  # sin datos suficientes -> supuesto neutro documentado

    return results, score_a, score_b, pen_proba


def assemble_report(stage, fixtures, clf_results, score_a, score_b, pen_proba,
                     real_results: pd.DataFrame = None):
    matches_out = []
    for i, m in enumerate(fixtures):
        proba_avg = np.mean([clf_results[name][i] for name in clf_results])
        entry = {
            "match_id": m["match_id"],
            "team_a": m["team_a"], "team_b": m["team_b"],
            "predicted_score_90": {"a": int(score_a[i]), "b": int(score_b[i])},
            "advance_probability": {name: round(float(clf_results[name][i]), 3) for name in clf_results},
            "advance_probability_avg_a": round(float(proba_avg), 3),
            "penalty_scenario_probability": round(float(pen_proba[i]), 3) if score_a[i] == score_b[i] else None,
            "predicted_winner": m["team_a"] if proba_avg >= 0.5 else m["team_b"],
        }
        if real_results is not None:
            real = real_results[real_results["match_id"] == m["match_id"]]
            if not real.empty:
                r = real.iloc[0]
                entry["actual_result"] = {
                    "score_90": {"a": int(r["score_a_90"]), "b": int(r["score_b_90"])},
                    "went_to_penalties": bool(r["went_to_penalties"]),
                    "winner": r["winner_final"],
                }
                entry["hit"] = bool(entry["predicted_winner"] == r["winner_final"])
        matches_out.append(entry)
    return {"stage": stage, "matches": matches_out}


def add_validation_summary(report, fixtures, fixtures_feat, clf_results, score_a, score_b, real_results):
    ids = [m["match_id"] for m in fixtures]
    known_ids = set(real_results["match_id"])
    # Solo se valida el subconjunto de partidos de esta fase que YA tiene resultado real;
    # el resto son predicciones genuinas pendientes de confirmación (no se "inventan" resultados).
    mask = [mid in known_ids for mid in ids]
    n_validable = sum(mask)
    report["n_matches_total"] = len(ids)
    report["n_matches_validated"] = n_validable
    if n_validable == 0:
        report["validation_note"] = (
            "Ningún partido de esta fase tiene resultado real todavía; "
            "todas las predicciones están pendientes de confirmación."
        )
        return report

    real = real_results.set_index("match_id").loc[[mid for mid in ids if mid in known_ids]]
    idx_mask = np.array(mask)
    y_true = (real["winner_final"].values == fixtures_feat.loc[idx_mask, "team_a"].values).astype(int)

    validation = {}
    for name, proba in clf_results.items():
        y_pred = (proba[idx_mask] >= 0.5).astype(int)
        validation[name] = ev.evaluate_classifier(y_true, y_pred)
    validation["score_model"] = ev.evaluate_score_model(
        real["score_a_90"].values, score_a[idx_mask], real["score_b_90"].values, score_b[idx_mask]
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
            imp = clf.feature_importances_
            pairs = sorted(zip(cfg.ALL_FEATURES, imp), key=lambda x: -x[1])[:8]
            out[name] = [(f, round(float(v), 4)) for f, v in pairs]
        elif hasattr(clf, "coef_"):
            imp = clf.coef_[0]
            pairs = sorted(zip(cfg.ALL_FEATURES, imp), key=lambda x: -abs(x[1]))[:8]
            out[name] = [(f, round(float(v), 4)) for f, v in pairs]
    return out


def run_stage(stage: str, validate: bool = True) -> dict:
    """
    Ejecuta UNA fase completa. Esta es la única función que se invoca desde fuera;
    NO llama a run_stage de otra fase (evita predicción en cadena / simulación de bracket).
    """
    teams_df = dl.load_teams_static()
    players_df = dl.load_players()
    h2h_df = dl.load_historical_h2h()
    historical_df = dl.load_historical_matches()
    bracket_map = dl.load_bracket_map()
    real_results = dl.load_matches_played()

    # --- entrenamiento con historial de mundiales pasados (2014/2018/2022) ---
    train_df = fe.build_training_table(historical_df, teams_df, players_df, h2h_df)
    if train_df.empty:
        raise RuntimeError("Tabla de entrenamiento vacía: revisa historical_matches.csv y teams_static.csv")
    classifiers, reg_a, reg_b, penalty_model = train_all_models(train_df)

    # --- fixtures reales de la fase (resueltos con resultados reales previos, nunca predicciones) ---
    fixtures = dl.get_stage_fixtures(stage, bracket_map, real_results)
    fixtures_feat = fe.build_features(fixtures, teams_df, players_df, h2h_df, historical_df, stage)

    clf_results, score_a, score_b, pen_proba = predict_stage(
        fixtures_feat, classifiers, reg_a, reg_b, penalty_model
    )

    report = assemble_report(stage, fixtures, clf_results, score_a, score_b, pen_proba,
                              real_results=real_results if validate else None)

    if validate:
        report = add_validation_summary(report, fixtures, fixtures_feat, clf_results, score_a, score_b, real_results)

    report["top_features"] = feature_importance(classifiers)
    report["train_set_size"] = len(train_df)
    report["uses_xgboost_native"] = mdl.HAS_XGBOOST
    return report


def advance_to_next_stage_guard(stage: str, next_stage: str):
    """
    Salvaguarda explícita (sección 7): lanza error si se intenta predecir next_stage
    sin que TODOS sus partidos tengan los equipos resueltos por resultados reales.
    """
    bracket_map = dl.load_bracket_map()
    real_results = dl.load_matches_played()
    try:
        dl.get_stage_fixtures(next_stage, bracket_map, real_results)
    except RuntimeError as e:
        raise RuntimeError(
            f"No se puede avanzar de {stage} a {next_stage} todavía: {e}"
        )
    print(f"[ok] Todos los partidos de {next_stage} están resueltos con resultados reales. Se puede predecir.")
