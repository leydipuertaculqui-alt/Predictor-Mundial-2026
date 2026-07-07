from pathlib import Path
from datetime import datetime
import pandas as pd


HISTORY_DIR = Path("outputs/history")
HISTORY_FILE = HISTORY_DIR / "metrics_history.csv"


def save_metrics_history(stage, validation_summary):

    HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    rows = []

    score_metrics = validation_summary.get("score_model", {})

    for model in ["logreg", "random_forest", "xgboost"]:

        if model not in validation_summary:
            continue

        metrics = validation_summary[model]

        rows.append({
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "stage": stage,
            "model": model,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "mae_score": score_metrics.get("MAE"),
            "rmse_score": score_metrics.get("RMSE")
        })

    df_new = pd.DataFrame(rows)

    if HISTORY_FILE.exists():

        df_old = pd.read_csv(HISTORY_FILE)

        df = pd.concat([df_old, df_new], ignore_index=True)

    else:

        df = df_new

    df.to_csv(HISTORY_FILE, index=False)

    return HISTORY_FILE