"""
Sincroniza los resultados reales desde data/raw/matches_played.csv
hacia outputs/history/predictions_history.csv

No modifica el pipeline ni los modelos.
"""

from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent

MATCHES = ROOT / "data" / "raw" / "matches_played.csv"
HISTORY = ROOT / "outputs" / "history" / "predictions_history.csv"


def main():

    if not MATCHES.exists():
        raise FileNotFoundError(MATCHES)

    if not HISTORY.exists():
        raise FileNotFoundError(HISTORY)

    real = pd.read_csv(MATCHES)
    history = pd.read_csv(HISTORY)

    actualizados = 0

    for _, row in real.iterrows():

        if pd.isna(row["winner_final"]):
            continue

        mask = history["match_id"] == row["match_id"]

        if not mask.any():
            continue

        history.loc[mask, "real_score_a"] = row["score_a_90"]
        history.loc[mask, "real_score_b"] = row["score_b_90"]
        history.loc[mask, "real_winner"] = row["winner_final"]

        actualizados += 1

    history.to_csv(HISTORY, index=False)

    print("=" * 40)
    print("SINCRONIZACIÓN COMPLETADA")
    print("=" * 40)
    print(f"Partidos actualizados: {actualizados}")
    print(f"Archivo: {HISTORY}")


if __name__ == "__main__":
    main()