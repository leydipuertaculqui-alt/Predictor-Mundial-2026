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
