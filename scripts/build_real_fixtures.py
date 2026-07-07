"""
Construye cruces de una fase futura usando
ganadores predichos de la fase anterior.

Ejemplo:
python scripts/build_predicted_fixtures.py QF
"""

import json
from pathlib import Path
import sys
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent


BRACKET = ROOT / "data" / "raw" / "bracket_map.json"
PRED_HISTORY = ROOT / "outputs" / "history" / "predictions_history.csv"

OUTPUT = ROOT / "data" / "processed"


def resolve_winner(slot, predictions):

    if not slot.startswith("winner_"):
        return slot


    match_id = slot.replace("winner_", "")


    row = predictions[
        predictions["match_id"] == match_id
    ]


    if row.empty:
        raise RuntimeError(
            f"No existe predicción para {match_id}"
        )


    return row.iloc[-1]["predicted_winner"]



def main(stage):


    with open(BRACKET, encoding="utf-8") as f:
        bracket = json.load(f)


    predictions = pd.read_csv(PRED_HISTORY)


    fixtures = []


    for match in bracket[stage]:

        fixtures.append({

            "match_id": match["match_id"],

            "stage": stage,

            "team_a": resolve_winner(
                match["team_a"],
                predictions
            ),

            "team_b": resolve_winner(
                match["team_b"],
                predictions
            ),

            "date": match["date"],

            "venue": match["venue"]

        })


    df = pd.DataFrame(fixtures)


    OUTPUT.mkdir(
        parents=True,
        exist_ok=True
    )


    outfile = OUTPUT / f"{stage}_predicted_fixtures.csv"


    df.to_csv(
        outfile,
        index=False
    )


    print("\n==============================")
    print("CRUCES PREDICHOS")
    print("==============================")

    print(df)


    print("\nGuardado en:")
    print(outfile)



if __name__ == "__main__":


    if len(sys.argv)!=2:

        print(
            "Uso: python scripts/build_predicted_fixtures.py QF"
        )

        sys.exit()


    main(sys.argv[1])