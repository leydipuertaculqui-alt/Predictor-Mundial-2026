from pathlib import Path
import joblib

ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = ROOT / "saved_models"


def models_exist() -> bool:
    """
    Verifica si los modelos principales ya existen.
    """
    required = [
        "logreg.joblib",
        "random_forest.joblib",
        "xgboost.joblib",
        "score_team_a.joblib",
        "score_team_b.joblib",
    ]

    return all((MODEL_DIR / file).exists() for file in required)


def load_models() -> dict:
    """
    Carga todos los modelos previamente entrenados.
    """

    classifiers = {
        "logreg": joblib.load(MODEL_DIR / "logreg.joblib"),
        "random_forest": joblib.load(MODEL_DIR / "random_forest.joblib"),
        "xgboost": joblib.load(MODEL_DIR / "xgboost.joblib"),
    }

    score_models = {
        "team_a": joblib.load(MODEL_DIR / "score_team_a.joblib"),
        "team_b": joblib.load(MODEL_DIR / "score_team_b.joblib"),
    }

    penalty_model = None

    penalty_file = MODEL_DIR / "penalty.joblib"

    if penalty_file.exists():
        penalty_model = joblib.load(penalty_file)

    return {
        "classifiers": classifiers,
        "score_models": score_models,
        "penalty_model": penalty_model,
    }