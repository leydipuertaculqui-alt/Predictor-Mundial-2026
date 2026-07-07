from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error
import numpy as np


def evaluate_classifier(y_true, y_pred):
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 3),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 3),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 3),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 3),
    }


def evaluate_score_model(y_true_a, y_pred_a, y_true_b, y_pred_b):
    mae = (mean_absolute_error(y_true_a, y_pred_a) + mean_absolute_error(y_true_b, y_pred_b)) / 2
    rmse = np.sqrt((mean_squared_error(y_true_a, y_pred_a) + mean_squared_error(y_true_b, y_pred_b)) / 2)
    return {"MAE": round(float(mae), 3), "RMSE": round(float(rmse), 3)}

def select_best_classifier(validation_summary):
    """
    Selecciona automáticamente el mejor clasificador usando el siguiente criterio:

    1. Accuracy
    2. F1
    3. Precision
    4. Recall

    Devuelve:
    {
        "best_model": "...",
        "metrics": {...}
    }
    """

    candidates = {
        name: metrics
        for name, metrics in validation_summary.items()
        if name not in ("score_model", "_note")
    }

    if not candidates:
        return None

    best_name = max(
        candidates,
        key=lambda name: (
            candidates[name]["accuracy"],
            candidates[name]["f1"],
            candidates[name]["precision"],
            candidates[name]["recall"],
        ),
    )

    return {
        "best_model": best_name,
        "metrics": candidates[best_name],
    }


def rank_models(validation_summary):
    """
    Ordena los clasificadores según Accuracy y F1.
    """

    ranking = []

    for model in ["logreg", "random_forest", "xgboost"]:

        if model not in validation_summary:
            continue

        metrics = validation_summary[model]

        ranking.append({
            "model": model,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"]
        })

    ranking.sort(
        key=lambda x: (x["accuracy"], x["f1"]),
        reverse=True
    )

    return ranking