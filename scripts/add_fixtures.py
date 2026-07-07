"""
Agrega partidos pendientes del bracket al archivo matches_played.csv

Uso:
python scripts/add_fixtures.py R16
python scripts/add_fixtures.py QF
"""

from pathlib import Path
import json
import sys
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent

MATCHES_FILE = ROOT / "data" / "raw" / "matches_played.csv"
BRACKET_FILE = ROOT / "data" / "raw" / "bracket_map.json"


COLUMNS = [
    "match_id",
    "stage",
    "team_a",
    "team_b",
    "score_a_90",
    "score_b_90",
    "went_to_penalties",
    "penalty_winner",
    "winner_final"
]


def add_fixtures(stage):

    df = pd.read_csv(MATCHES_FILE)

    with open(BRACKET_FILE, "r", encoding="utf-8") as f:
        bracket = json.load(f)


    if stage not in bracket:
        raise ValueError(
            f"La fase {stage} no existe en bracket_map"
        )


    existing = set(df["match_id"])


    new_rows = []


    for match in bracket[stage]:

        match_id = match["match_id"]


        if match_id not in existing:

            new_rows.append({

                "match_id": match_id,
                "stage": stage,
                "team_a": match["team_a"],
                "team_b": match["team_b"],
                "score_a_90": None,
                "score_b_90": None,
                "went_to_penalties": None,
                "penalty_winner": None,
                "winner_final": None

            })


    if not new_rows:
        print("No hay partidos nuevos para agregar.")
        return


    df_new = pd.DataFrame(new_rows)

    df = pd.concat(
        [df, df_new],
        ignore_index=True
    )


    df.to_csv(
        MATCHES_FILE,
        index=False
    )


    print("==============================")
    print("PARTIDOS AGREGADOS")
    print("==============================")

    for row in new_rows:
        print(
            row["match_id"],
            row["team_a"],
            "vs",
            row["team_b"]
        )


def main():

    if len(sys.argv) != 2:
        print(
            "Uso: python scripts/add_fixtures.py FASE"
        )
        sys.exit(1)


    stage = sys.argv[1]

    add_fixtures(stage)



if __name__ == "__main__":
    main()