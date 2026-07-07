from pathlib import Path
from datetime import datetime
import pandas as pd

# Carpeta donde se almacenará el historial
HISTORY_DIR = Path("outputs/history")
HISTORY_FILE = HISTORY_DIR / "predictions_history.csv"


def save_predictions(stage, matches):

    HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    rows = []

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for match in matches:

        row = {

            "date": now,

            "stage": stage,

            "match_id": match["match_id"],

            "team_a": match["team_a"],

            "team_b": match["team_b"],

            "predicted_score_a": match["predicted_score_90"]["a"],

            "predicted_score_b": match["predicted_score_90"]["b"],

            "predicted_winner": match["predicted_winner"],

            "probability": round(
                match["advance_probability_avg_a"], 4
            )
        }

        # Si existen resultados reales también se guardan
        if "actual_result" in match:

            actual = match["actual_result"]

            row["real_score_a"] = actual["score_90"]["a"]
            row["real_score_b"] = actual["score_90"]["b"]
            row["real_winner"] = actual["winner"]

        rows.append(row)

    df_new = pd.DataFrame(rows)

    if HISTORY_FILE.exists():

        df_old = pd.read_csv(HISTORY_FILE)

        df = pd.concat(
            [df_old, df_new],
            ignore_index=True
        )

    else:

        df = df_new

    df.to_csv(
        HISTORY_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    return HISTORY_FILE
