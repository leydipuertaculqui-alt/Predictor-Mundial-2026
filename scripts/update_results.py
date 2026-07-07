"""
Actualización de resultados reales del Mundial 2026.

Uso:
python scripts/update_results.py MATCH_ID TEAM_A TEAM_B SCORE_A SCORE_B WINNER

Ejemplo:
python scripts/update_results.py M91 BRA NOR 2 0 BRA
"""

from pathlib import Path
import sys
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent

MATCHES_FILE = ROOT / "data" / "raw" / "matches_played.csv"


def update_result(
    match_id,
    team_a,
    team_b,
    score_a,
    score_b,
    winner
):

    if not MATCHES_FILE.exists():
        raise FileNotFoundError(
            f"No existe {MATCHES_FILE}"
        )

    df = pd.read_csv(MATCHES_FILE)


    # Buscar partido
    mask = df["match_id"] == match_id


    if not mask.any():
        raise ValueError(
            f"No existe el partido {match_id}"
        )


    # Actualizar datos

    df.loc[mask, "team_a"] = team_a
    df.loc[mask, "team_b"] = team_b

    df.loc[mask, "score_a_90"] = int(score_a)
    df.loc[mask, "score_b_90"] = int(score_b)

    df.loc[mask, "winner_final"] = winner


    # Determinar penales

    if int(score_a) == int(score_b):
        df.loc[mask, "went_to_penalties"] = True
    else:
        df.loc[mask, "went_to_penalties"] = False


    # Guardar

    df.to_csv(
        MATCHES_FILE,
        index=False
    )


    print("================================")
    print("RESULTADO ACTUALIZADO")
    print("================================")
    print(f"Partido : {match_id}")
    print(f"{team_a} {score_a} - {score_b} {team_b}")
    print(f"Ganador : {winner}")
    print("Archivo actualizado correctamente")


def main():

    if len(sys.argv) != 7:

        print(
            "Uso:"
            "\npython scripts/update_results.py "
            "MATCH_ID TEAM_A TEAM_B SCORE_A SCORE_B WINNER"
        )

        sys.exit(1)


    _, match_id, team_a, team_b, score_a, score_b, winner = sys.argv


    update_result(
        match_id,
        team_a,
        team_b,
        score_a,
        score_b,
        winner
    )


if __name__ == "__main__":
    main()