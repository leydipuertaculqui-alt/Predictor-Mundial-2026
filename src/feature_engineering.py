"""
Construye la matriz de features (sección 5 del diseño) para un conjunto de partidos,
a partir de las tablas reales cargadas por data_loader.
"""
import numpy as np
import pandas as pd
from src import config as cfg


def _team_row(team_id: str, teams_df: pd.DataFrame) -> pd.Series:
    row = teams_df.loc[teams_df["team_id"] == team_id]
    if row.empty:
        raise KeyError(f"Equipo '{team_id}' no encontrado en teams_static.csv")
    return row.iloc[0]


def _key_player_stats(team_id: str, players_df: pd.DataFrame) -> dict:
    sub = players_df[players_df["team_id"] == team_id]
    if sub.empty:
        return dict(avail=1.0, top_goals=0, top_rating=6.5, dependency=0.2, avg_age=26, prev_wc_goals=0)
    key = sub[sub["is_key_player"] == True]
    if key.empty:
        key = sub
    avail = 1.0 - key["suspension_next_match"].astype(bool).mean()
    if (key["injury_status"] != "healthy").any():
        avail *= 0.85
    return dict(
        avail=float(avail),
        top_goals=float(key["goals_tournament"].max()),
        top_rating=float(key["individual_rating_avg"].max()),
        dependency=float(key["team_dependency_index"].max()),
        avg_age=float(key["age"].mean()),
        prev_wc_goals=float(key["goals_prev_wc"].max()),
    )


def _h2h_stats(team_a: str, team_b: str, h2h_df: pd.DataFrame) -> dict:
    row = h2h_df[
        ((h2h_df["team_a"] == team_a) & (h2h_df["team_b"] == team_b))
        | ((h2h_df["team_a"] == team_b) & (h2h_df["team_b"] == team_a))
    ]
    if row.empty:
        return dict(wins_a=0, wins_b=0, draws=0)
    r = row.iloc[0]
    if r["team_a"] == team_a:
        return dict(wins_a=r["wins_a"], wins_b=r["wins_b"], draws=r["draws"])
    return dict(wins_a=r["wins_b"], wins_b=r["wins_a"], draws=r["draws"])


def _ko_historical_win_rate(team_id: str, historical_df: pd.DataFrame) -> float:
    """% de partidos eliminatorios ganados en Mundiales anteriores (2014/2018/2022)."""
    played = historical_df[(historical_df["team_a"] == team_id) | (historical_df["team_b"] == team_id)]
    if played.empty:
        return 0.5  # sin historial reciente en eliminatorias -> valor neutro
    wins = (played["winner_final"] == team_id).sum()
    return float(wins / len(played))


def build_features(
    fixtures: list,
    teams_df: pd.DataFrame,
    players_df: pd.DataFrame,
    h2h_df: pd.DataFrame,
    historical_df: pd.DataFrame,
    stage: str,
) -> pd.DataFrame:
    rows = []
    for m in fixtures:
        a, b = m["team_a"], m["team_b"]
        ta, tb = _team_row(a, teams_df), _team_row(b, teams_df)
        kpa, kpb = _key_player_stats(a, players_df), _key_player_stats(b, players_df)
        h2h = _h2h_stats(a, b, h2h_df)
        row = {
            "match_id": m["match_id"], "team_a": a, "team_b": b,
            "fifa_ranking_pts_diff": ta["fifa_ranking_pts"] - tb["fifa_ranking_pts"],
            "fifa_ranking_pos_diff": tb["fifa_ranking_pos"] - ta["fifa_ranking_pos"],  # invertido: pos baja = mejor
            "wc_appearances_diff": ta["wc_appearances"] - tb["wc_appearances"],
            "wc_best_stage_diff": ta["wc_best_stage"] - tb["wc_best_stage"],
            "wc_titles_diff": ta["wc_titles"] - tb["wc_titles"],
            "goals_for_group_diff": ta["group_gf"] - tb["group_gf"],
            "goals_against_group_diff": tb["group_ga"] - ta["group_ga"],  # menos GA propio es mejor
            "goals_for_ko_cumulative_diff": 0,   # se actualiza fase a fase con matches_played reales
            "goals_against_ko_cumulative_diff": 0,
            "possession_avg_diff": 0.0,          # placeholder: requiere feed Opta/FBref con acceso a internet
            "shots_on_target_avg_diff": 0.0,
            "defensive_efficiency_diff": 0.0,
            "clean_sheets_ko_diff": 0,
            "days_since_last_match_diff": 0,
            "group_strength_score_diff": ta["group_pts"] - tb["group_pts"],
            "group_position_a": ta["group_position"],
            "group_position_b": tb["group_position"],
            "key_player_availability_diff": kpa["avail"] - kpb["avail"],
            "top_scorer_goals_tournament_diff": kpa["top_goals"] - kpb["top_goals"],
            "top_scorer_individual_rating_diff": kpa["top_rating"] - kpb["top_rating"],
            "team_dependency_index_diff": kpa["dependency"] - kpb["dependency"],
            "key_players_avg_age_diff": kpa["avg_age"] - kpb["avg_age"],
            "key_players_prev_wc_goals_diff": kpa["prev_wc_goals"] - kpb["prev_wc_goals"],
            "h2h_wins_a": h2h["wins_a"], "h2h_wins_b": h2h["wins_b"], "h2h_draws": h2h["draws"],
            "h2h_goal_diff_avg": 0.0,
            "h2h_penalties_a_win_rate": 0.5,
            "ko_stage_historical_win_rate_a": _ko_historical_win_rate(a, historical_df),
            "ko_stage_historical_win_rate_b": _ko_historical_win_rate(b, historical_df),
            "same_confederation": int(ta["confederation"] == tb["confederation"]),
            "neutral_venue": int(a not in ("USA", "MEX", "CAN") and b not in ("USA", "MEX", "CAN")),
            "host_advantage": int(a in ("USA", "MEX", "CAN")) - int(b in ("USA", "MEX", "CAN")),
            "stage_ordinal": cfg.STAGE_ORDINAL[stage],
        }
        rows.append(row)
    return pd.DataFrame(rows)


def build_training_table(historical_df: pd.DataFrame, teams_df: pd.DataFrame,
                          players_df: pd.DataFrame, h2h_df: pd.DataFrame) -> pd.DataFrame:
    """
    Tabla de entrenamiento a partir de partidos eliminatorios de Mundiales pasados.
    NOTA: para equipos históricos que no jugaron 2026 (ej. Alemania 2014 vs Argelia),
    usamos las filas de teams_static.csv disponibles; si un equipo histórico no está
    en teams_static (p.ej. Bélgica 2018 no cambia, pero Rusia 2018 no clasificó a 2026),
    esa fila se descarta automáticamente para no introducir datos inventados.
    """
    known_teams = set(teams_df["team_id"])
    rows = []
    for _, m in historical_df.iterrows():
        if m["team_a"] not in known_teams or m["team_b"] not in known_teams:
            continue
        fixture = [{"match_id": f"{m['wc_year']}_{m['team_a']}_{m['team_b']}",
                    "team_a": m["team_a"], "team_b": m["team_b"]}]
        feat = build_features(fixture, teams_df, players_df, h2h_df, historical_df, stage=m["stage"])
        feat["winner"] = int(m["winner_final"] == m["team_a"])
        feat["score_a_90"] = m["score_a_90"]
        feat["score_b_90"] = m["score_b_90"]
        feat["went_to_penalties"] = m["went_to_penalties"]
        rows.append(feat)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)
