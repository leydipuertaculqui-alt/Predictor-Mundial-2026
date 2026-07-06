"""
Carga los datos crudos reales (recolectados manualmente vía búsqueda web,
verificados contra FIFA/ESPN/Wikipedia/Yahoo Sports/CBS Sports/CNN, julio 2026).
"""
import json
import pandas as pd
from src import config as cfg


def load_teams_static() -> pd.DataFrame:
    return pd.read_csv(cfg.DATA_RAW / "teams_static.csv")


def load_players() -> pd.DataFrame:
    return pd.read_csv(cfg.DATA_RAW / "players.csv")


def load_historical_matches() -> pd.DataFrame:
    """Partidos eliminatorios reales de Mundiales 2014/2018/2022 (para entrenamiento)."""
    return pd.read_csv(cfg.DATA_RAW / "historical_matches.csv")


def load_historical_h2h() -> pd.DataFrame:
    return pd.read_csv(cfg.DATA_RAW / "historical_h2h.csv")


def load_matches_played() -> pd.DataFrame:
    """Resultados reales YA jugados del Mundial 2026 (R32 completo + R16 parcial)."""
    return pd.read_csv(cfg.DATA_RAW / "matches_played.csv")


def load_bracket_map() -> dict:
    with open(cfg.DATA_RAW / "bracket_map.json") as f:
        return json.load(f)


def resolve_slot(slot: str, real_results: pd.DataFrame) -> str:
    """
    Resuelve un slot tipo 'winner_M73' o 'loser_M101' a un team_id real,
    usando SOLO resultados reales ya jugados (nunca predicciones).
    """
    if not (slot.startswith("winner_") or slot.startswith("loser_")):
        return slot  # ya es un team_id directo (ej. equipos de R32)

    kind, match_id = slot.split("_", 1)
    row = real_results.loc[real_results["match_id"] == match_id]
    if row.empty:
        raise RuntimeError(
            f"No hay resultado real todavía para {match_id}. "
            f"No se puede resolver el slot '{slot}' (evita predicción en cadena)."
        )
    row = row.iloc[0]
    if kind == "winner":
        return row["winner_final"]
    else:  # loser
        return row["team_b"] if row["winner_final"] == row["team_a"] else row["team_a"]


def get_stage_fixtures(stage: str, bracket_map: dict, real_results: pd.DataFrame) -> list:
    """
    Devuelve la lista de partidos de una fase con los team_id YA resueltos
    (a partir de resultados reales, nunca de predicciones).
    """
    fixtures = []
    for m in bracket_map[stage]:
        team_a = resolve_slot(m["team_a"], real_results)
        team_b = resolve_slot(m["team_b"], real_results)
        fixtures.append({**m, "team_a": team_a, "team_b": team_b})
    return fixtures
