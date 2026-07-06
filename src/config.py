"""
Configuración central del proyecto.
Ajusta rutas y listas de features desde aquí.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
REPORTS = OUTPUTS / "reports"

STAGES = ["R32", "R16", "QF", "SF", "3P", "F"]
STAGE_ORDINAL = {"R32": 1, "R16": 2, "QF": 3, "SF": 4, "3P": 5, "F": 6}

# --- Features usadas por los modelos (sección 5 del diseño) ---
NUMERIC_FEATURES = [
    "fifa_ranking_pts_diff", "fifa_ranking_pos_diff",
    "wc_appearances_diff", "wc_best_stage_diff", "wc_titles_diff",
    "goals_for_group_diff", "goals_against_group_diff",
    "goals_for_ko_cumulative_diff", "goals_against_ko_cumulative_diff",
    "possession_avg_diff", "shots_on_target_avg_diff",
    "defensive_efficiency_diff", "clean_sheets_ko_diff",
    "days_since_last_match_diff", "group_strength_score_diff",
    "group_position_a", "group_position_b",
    "key_player_availability_diff",
    "top_scorer_goals_tournament_diff",
    "top_scorer_individual_rating_diff",
    "team_dependency_index_diff",
    "key_players_avg_age_diff",
    "key_players_prev_wc_goals_diff",
    "h2h_wins_a", "h2h_wins_b", "h2h_draws",
    "h2h_goal_diff_avg",
    "h2h_penalties_a_win_rate",
    "ko_stage_historical_win_rate_a", "ko_stage_historical_win_rate_b",
    "stage_ordinal",
]

CATEGORICAL_FEATURES = ["same_confederation", "neutral_venue", "host_advantage"]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

RANDOM_STATE = 42
