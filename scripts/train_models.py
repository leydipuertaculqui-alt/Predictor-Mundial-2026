"""
Entrenamiento de los modelos del predictor del Mundial 2026.

Este script:

1. Carga toda la información histórica.
2. Construye la tabla de entrenamiento.
3. Entrena todos los modelos.
4. Compara su rendimiento.
5. Selecciona automáticamente el mejor clasificador.
6. Guarda todos los modelos entrenados.
7. Guarda información del entrenamiento.
"""
from pathlib import Path
import sys
import json
import joblib

# Permite importar la carpeta src
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from src import config as cfg
from src import data_loader as dl
from src import feature_engineering as fe
from src import pipeline


def main():

    print("=" * 60)
    print(" ENTRENAMIENTO DE MODELOS ".center(60))
    print("=" * 60)

    cfg.ensure_directories()

    print("\nCargando datos...")

    teams_df = dl.load_teams_static()
    players_df = dl.load_players()
    h2h_df = dl.load_historical_h2h()
    historical_df = dl.load_historical_matches()

    train_df = fe.build_training_table(
        historical_df,
        teams_df,
        players_df,
        h2h_df,
    )

    print(f"Partidos históricos: {len(train_df)}")

    print("\nEntrenando modelos...")

    model_manager = pipeline.train_all_models(train_df)

    save_dir = ROOT / "saved_models"
    save_dir.mkdir(exist_ok=True)

    # Clasificadores
    for nombre, modelo in model_manager["classifiers"].items():
        ruta = save_dir / f"{nombre}.joblib"
        joblib.dump(modelo, ruta)
        print(f"✔ {nombre} guardado")

    # Regresores
    joblib.dump(
        model_manager["score_models"]["team_a"],
        save_dir / "score_team_a.joblib"
    )

    joblib.dump(
        model_manager["score_models"]["team_b"],
        save_dir / "score_team_b.joblib"
    )

    print("✔ Modelos de score guardados")

    # Penales
    if model_manager["penalty_model"] is not None:
        joblib.dump(
            model_manager["penalty_model"],
            save_dir / "penalty.joblib"
        )
        print("✔ Modelo de penales guardado")

    metadata = {
        "train_size": len(train_df),
        "features": cfg.ALL_FEATURES,
        "uses_xgboost": pipeline.mdl.HAS_XGBOOST,
    }

    with open(save_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print("\nEntrenamiento finalizado correctamente.")
    print(f"Modelos guardados en:\n{save_dir}")


if __name__ == "__main__":
    main()