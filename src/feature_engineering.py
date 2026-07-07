"""Construye la matriz de features para los partidos del torneo.

La lógica de negocio se conserva intacta; la refactorización se centra en
hacer el código más legible, reutilizable y fácil de mantener.
"""
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
    record = row.iloc[0]
    if record["team_a"] == team_a:
        return dict(wins_a=record["wins_a"], wins_b=record["wins_b"], draws=record["draws"])
    return dict(wins_a=record["wins_b"], wins_b=record["wins_a"], draws=record["draws"])


def _ko_historical_win_rate(team_id: str, historical_df: pd.DataFrame) -> float:
    """Porcentaje de partidos eliminatorios ganados en Mundiales anteriores (2014/2018/2022)."""
    played = historical_df[(historical_df["team_a"] == team_id) | (historical_df["team_b"] == team_id)]
    if played.empty:
        return 0.5
    wins = (played["winner_final"] == team_id).sum()
    return float(wins / len(played))


def _build_feature_row(match: dict, teams_df: pd.DataFrame, players_df: pd.DataFrame,
                       h2h_df: pd.DataFrame, historical_df: pd.DataFrame, stage: str) -> dict:
    team_a, team_b = match["team_a"], match["team_b"]
    team_a_stats, team_b_stats = _team_row(team_a, teams_df), _team_row(team_b, teams_df)
    key_a, key_b = _key_player_stats(team_a, players_df), _key_player_stats(team_b, players_df)
    head_to_head = _h2h_stats(team_a, team_b, h2h_df)

    return {
        "match_id": match["match_id"],
        "team_a": team_a,
        "team_b": team_b,
        "fifa_ranking_pts_diff": team_a_stats["fifa_ranking_pts"] - team_b_stats["fifa_ranking_pts"],
        "fifa_ranking_pos_diff": team_b_stats["fifa_ranking_pos"] - team_a_stats["fifa_ranking_pos"],
        "wc_appearances_diff": team_a_stats["wc_appearances"] - team_b_stats["wc_appearances"],
        "wc_best_stage_diff": team_a_stats["wc_best_stage"] - team_b_stats["wc_best_stage"],
        "wc_titles_diff": team_a_stats["wc_titles"] - team_b_stats["wc_titles"],
        "goals_for_group_diff": team_a_stats["group_gf"] - team_b_stats["group_gf"],
        "goals_against_group_diff": team_b_stats["group_ga"] - team_a_stats["group_ga"],
        "goals_for_ko_cumulative_diff": 0,
        "goals_against_ko_cumulative_diff": 0,
        "possession_avg_diff": 0.0,
        "shots_on_target_avg_diff": 0.0,
        "defensive_efficiency_diff": 0.0,
        "clean_sheets_ko_diff": 0,
        "days_since_last_match_diff": 0,
        "group_strength_score_diff": team_a_stats["group_pts"] - team_b_stats["group_pts"],
        "group_position_a": team_a_stats["group_position"],
        "group_position_b": team_b_stats["group_position"],
        "key_player_availability_diff": key_a["avail"] - key_b["avail"],
        "top_scorer_goals_tournament_diff": key_a["top_goals"] - key_b["top_goals"],
        "top_scorer_individual_rating_diff": key_a["top_rating"] - key_b["top_rating"],
        "team_dependency_index_diff": key_a["dependency"] - key_b["dependency"],
        "key_players_avg_age_diff": key_a["avg_age"] - key_b["avg_age"],
        "key_players_prev_wc_goals_diff": key_a["prev_wc_goals"] - key_b["prev_wc_goals"],
        "h2h_wins_a": head_to_head["wins_a"],
        "h2h_wins_b": head_to_head["wins_b"],
        "h2h_draws": head_to_head["draws"],
        "h2h_goal_diff_avg": 0.0,
        "h2h_penalties_a_win_rate": 0.5,
        "ko_stage_historical_win_rate_a": _ko_historical_win_rate(team_a, historical_df),
        "ko_stage_historical_win_rate_b": _ko_historical_win_rate(team_b, historical_df),
        "same_confederation": int(team_a_stats["confederation"] == team_b_stats["confederation"]),
        "neutral_venue": int(team_a not in cfg.HOST_TEAMS and team_b not in cfg.HOST_TEAMS),
        "host_advantage": int(team_a in cfg.HOST_TEAMS) - int(team_b in cfg.HOST_TEAMS),
        "stage_ordinal": cfg.STAGE_ORDINAL[stage],
    }


def build_features(
    fixtures: list,
    teams_df: pd.DataFrame,
    players_df: pd.DataFrame,
    h2h_df: pd.DataFrame,
    historical_df: pd.DataFrame,
    stage: str,
) -> pd.DataFrame:
    rows = [_build_feature_row(match, teams_df, players_df, h2h_df, historical_df, stage) for match in fixtures]
    return pd.DataFrame(rows)


def build_training_table(historical_df: pd.DataFrame, teams_df: pd.DataFrame,
                          players_df: pd.DataFrame, h2h_df: pd.DataFrame) -> pd.DataFrame:
    """Tabla de entrenamiento a partir de partidos eliminatorios de Mundiales pasados."""
    known_teams = set(teams_df["team_id"])
    rows = []
    for _, match in historical_df.iterrows():
        if match["team_a"] not in known_teams or match["team_b"] not in known_teams:
            continue
        fixture = [{
            "match_id": f"{match['wc_year']}_{match['team_a']}_{match['team_b']}",
            "team_a": match["team_a"],
            "team_b": match["team_b"],
        }]
        feature_frame = build_features(fixture, teams_df, players_df, h2h_df, historical_df, stage=match["stage"])
        feature_frame["winner"] = int(match["winner_final"] == match["team_a"])
        feature_frame["score_a_90"] = match["score_a_90"]
        feature_frame["score_b_90"] = match["score_b_90"]
        feature_frame["went_to_penalties"] = match["went_to_penalties"]
        rows.append(feature_frame)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)
