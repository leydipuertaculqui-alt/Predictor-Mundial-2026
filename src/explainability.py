"""
Explainable AI
Genera una explicación sencilla del modelo ganador.
"""

FEATURE_NAMES = {

    "fifa_ranking_pos_diff": "Diferencia Ranking FIFA",

    "fifa_ranking_pts_diff": "Diferencia Puntos FIFA",

    "market_value_diff": "Diferencia Valor del Plantel",

    "recent_form_diff": "Forma reciente",

    "wc_titles_diff": "Diferencia de títulos mundiales",

    "wc_best_stage_diff": "Mejor participación histórica",

    "group_position_diff": "Posición en fase de grupos",

    "goals_for_group_diff": "Goles anotados",

    "goals_against_group_diff": "Goles recibidos",

    "goal_difference_group_diff": "Diferencia de goles",

    "group_strength_score_diff": "Fortaleza del grupo",

    "key_players_avg_age_diff": "Edad promedio jugadores clave",

    "key_players_prev_wc_goals_diff": "Goles históricos jugadores clave",

    "ko_stage_historical_win_rate_a": "Historial KO Equipo A",

    "ko_stage_historical_win_rate_b": "Historial KO Equipo B",

    "stage_ordinal": "Etapa del torneo",

}


def feature_label(feature):

    return FEATURE_NAMES.get(feature, feature)


def decision_support(classifiers, best_model, feature_names):

    if best_model not in classifiers:
        return None

    pipe = classifiers[best_model]

    clf = pipe.named_steps["clf"]

    if hasattr(clf, "feature_importances_"):

        values = clf.feature_importances_

    elif hasattr(clf, "coef_"):

        values = abs(clf.coef_[0])

    else:

        return None

    pairs = sorted(
        zip(feature_names, values),
        key=lambda x: -x[1]
    )[:5]

    factors = []

    for feature, importance in pairs:

        factors.append({

            "feature": feature_label(feature),

            "importance": round(float(importance * 100),2)

        })

    return {

        "selected_model": best_model,

        "selection_reason":
            "Mayor Accuracy durante la validación.",

        "main_factors": factors

    }