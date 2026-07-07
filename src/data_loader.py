"""Carga de datos crudos y utilidades de resolución del bracket.

Los datos se leen de forma robusta y se validan antes de usarlos para evitar
fallos silenciosos en la pipeline.
"""
import json
from pathlib import Path

import pandas as pd

from src import config as cfg


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo requerido: {path}")
    return pd.read_csv(path)


def _read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo requerido: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_teams_static() -> pd.DataFrame:
    return _read_csv(cfg.DATA_RAW / "teams_static.csv")


def load_players() -> pd.DataFrame:
    return _read_csv(cfg.DATA_RAW / "players.csv")


def load_historical_matches() -> pd.DataFrame:
    """Partidos eliminatorios reales de Mundiales 2014/2018/2022 (para entrenamiento)."""
    return _read_csv(cfg.DATA_RAW / "historical_matches.csv")


def load_historical_h2h() -> pd.DataFrame:
    return _read_csv(cfg.DATA_RAW / "historical_h2h.csv")


def load_matches_played() -> pd.DataFrame:
    """Resultados reales YA jugados del Mundial 2026 (R32 completo + R16 parcial)."""
    return _read_csv(cfg.DATA_RAW / "matches_played.csv")


def load_bracket_map() -> dict:
    return _read_json(cfg.DATA_RAW / "bracket_map.json")


def resolve_slot(slot: str, real_results: pd.DataFrame) -> str:
    """Resuelve un slot del bracket a un team_id real usando solo resultados ya jugados."""
    if not (slot.startswith("winner_") or slot.startswith("loser_")):
        return slot

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
    return row["team_b"] if row["winner_final"] == row["team_a"] else row["team_a"]


def get_stage_fixtures(stage: str, bracket_map: dict, real_results: pd.DataFrame) -> list:
    """Devuelve los partidos de una fase con los equipos ya resueltos por resultados reales."""
    cfg.validate_stage(stage)
    if stage not in bracket_map:
        raise KeyError(f"No existe la fase '{stage}' en el bracket map")

    fixtures = []
    for match in bracket_map[stage]:
        team_a = resolve_slot(match["team_a"], real_results)
        team_b = resolve_slot(match["team_b"], real_results)
        fixtures.append({**match, "team_a": team_a, "team_b": team_b})
    return fixtures
