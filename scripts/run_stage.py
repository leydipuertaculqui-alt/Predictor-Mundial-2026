"""
CLI para ejecutar una fase del predictor.

Uso:
    python scripts/run_stage.py R32
    python scripts/run_stage.py R16
    python scripts/run_stage.py QF
"""
 
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config as cfg
from src import pipeline as pl


def main():

    # ----------------------------------------------------
    # Validación de argumentos
    # ----------------------------------------------------

    if len(sys.argv) != 2:
        print(f"Uso: python {sys.argv[0]} <fase>")
        print(f"Fases válidas: {cfg.STAGES}")
        sys.exit(1)

    try:
        stage = cfg.validate_stage(sys.argv[1])
    except ValueError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    # ----------------------------------------------------
    # Verificar si puede avanzarse a la siguiente fase
    # ----------------------------------------------------
 
    idx = cfg.STAGES.index(stage)

    if idx > 0:

        prev_stage = cfg.STAGES[idx - 1] if stage != "F" else "SF"

        try:
            pl.advance_to_next_stage_guard(prev_stage, stage)

        except RuntimeError as e:
            print(f"[DETENIDO] {e}")
            sys.exit(1)

    # ----------------------------------------------------
    # Ejecutar predicción
    # ----------------------------------------------------

    report = pl.run_stage(stage, validate=True)

    # ----------------------------------------------------
    # Guardar reporte JSON
    # ----------------------------------------------------

    cfg.ensure_directories()

    out_path = cfg.REPORTS / f"reporte_{stage}.json"

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False
        )

    # ====================================================
    # RESULTADOS
    # ====================================================

    print(f"\n=== Reporte de fase {stage} ===")

    for match in report["matches"]:

        hit_str = ""

        if "hit" in match:
            hit_str = " ✅ ACIERTO" if match["hit"] else " ❌ FALLO"

        real_str = ""

        if "actual_result" in match:

            actual = match["actual_result"]

            real_str = (
                f" | Real: "
                f"{actual['score_90']['a']}-"
                f"{actual['score_90']['b']} "
                f"(ganó {actual['winner']})"
            )

        print(
            f"{match['team_a']:>4} "
            f"{match['predicted_score_90']['a']}-"
            f"{match['predicted_score_90']['b']} "
            f"{match['team_b']:<4}"
            f" | Predicho avanza: "
            f"{match['predicted_winner']:<4}"
            f" (p={match['advance_probability_avg_a']:.2%})"
            f"{real_str}"
            f"{hit_str}"
        )

    # ====================================================
    # VALIDACIÓN
    # ====================================================

    validation = report.get("validation_summary")

    if validation:

        print("\n--- Validación contra resultados reales ---")

        for name, metrics in validation.items():
            print(f"  {name}: {metrics}")

    # ====================================================
    # RANKING
    # ====================================================

    ranking = report.get("model_ranking", [])

    if ranking:

        print("\n--- Ranking de modelos ---")

        for i, item in enumerate(ranking, start=1):

            print(
                f"{i}. "
                f"{item['model']:15}"
                f" Accuracy={item['accuracy']:.3f}"
                f" F1={item['f1']:.3f}"
            )

    # ====================================================
    # MEJOR MODELO
    # ====================================================

    if "best_classifier" in report:

        print(f"\nModelo recomendado: {report['best_classifier']}")

    # ====================================================
    # PESOS DEL ENSEMBLE
    # ====================================================

    if "model_weights" in report:

        print("\n--- Pesos del Ensemble ---")

        for model, weight in report["model_weights"].items():

            print(f"{model:15}: {weight:.3f}")

    # ====================================================
    # INFORMACIÓN DEL ENTRENAMIENTO
    # ====================================================

    if "train_set_size" in report:

        print(
            f"\nPartidos históricos utilizados: "
            f"{report['train_set_size']}"
        )

    if "uses_xgboost_native" in report:

        print(
            "Motor XGBoost:",
            "Sí" if report["uses_xgboost_native"] else "No (GradientBoosting)"
        )

    # ====================================================
    # HISTORIAL
    # ====================================================

    if "metrics_history_file" in report:

        print(
            f"\nHistorial actualizado en:"
            f" {report['metrics_history_file']}"
        )

    # ====================================================
    # REPORTE
    # ====================================================

    print(f"\nReporte completo guardado en:")
    print(out_path)
    
    if "prediction_history_file" in report:

     print(
        f"Historial de predicciones: "
        f"{report['prediction_history_file']}"
     )


if __name__ == "__main__":
    main()
