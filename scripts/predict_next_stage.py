"""
Predicción de fases futuras del Mundial.

Usa:
- resultados reales disponibles
- predicciones previas
- modelos entrenados

No modifica pipeline.py
"""
import sys
from pathlib import Path
import pandas as pd


# ============================================
# Configurar raíz del proyecto
# ============================================

ROOT = Path(__file__).resolve().parent.parent

sys.path.append(str(ROOT))


from src import pipeline


MATCHES_REAL = ROOT / "data" / "raw" / "matches_played.csv"
BRACKET = ROOT / "data" / "raw" / "bracket_map.json"
PRED_HISTORY = ROOT / "outputs" / "history" / "predictions_history.csv"


def load_real_results():

    return pd.read_csv(MATCHES_REAL)



def load_predictions():

    if not PRED_HISTORY.exists():
        return pd.DataFrame()

    return pd.read_csv(PRED_HISTORY)



def resolve_winner(match_id, real_results, predictions):

    # primero busca resultado real
    real = real_results[
        real_results["match_id"] == match_id
    ]

    if not real.empty:
        return real.iloc[0]["winner_final"]


    # luego busca predicción
    pred = predictions[
        predictions["match_id"] == match_id
    ]

    if not pred.empty:
        return pred.iloc[-1]["predicted_winner"]


    raise RuntimeError(
        f"No existe ganador para {match_id}"
    )



def main(stage):

    import json

    with open(BRACKET,encoding="utf-8") as f:
        bracket=json.load(f)


    real_results = load_real_results()
    predictions = load_predictions()


    print("\n==============================")
    print(f"FASE {stage}")
    print("==============================\n")


    for match in bracket[stage]:

        team_a = match["team_a"]
        team_b = match["team_b"]


        if team_a.startswith("winner_"):
            team_a = resolve_winner(
                team_a.replace("winner_",""),
                real_results,
                predictions
            )


        if team_b.startswith("winner_"):
            team_b = resolve_winner(
                team_b.replace("winner_",""),
                real_results,
                predictions
            )

        
            print("\nEjecutando modelo...\n")

    report = pipeline.run_stage(
        stage,
        validate=False
    )

    print("\n==============================")
    print("PREDICCIONES")
    print("==============================\n")


    predicted_rows = []


    for match in report["matches"]:

        score = match["predicted_score_90"]

        prob = match["advance_probability_avg_a"]


        print("--------------------------------")

        print(
            match["match_id"],
            ":",
            match["team_a"],
            "vs",
            match["team_b"]
        )


        print(
            "Marcador predicho:",
            f'{score["a"]}-{score["b"]}'
        )


        print(
            "Clasifica:",
            match["predicted_winner"]
        )


        print(
            "Probabilidad equipo A:",
            f"{prob*100:.1f}%"
        )


        if match["penalty_scenario_probability"] is not None:

            print(
                "Probabilidad de penales:",
                f"{match['penalty_scenario_probability']*100:.1f}%"
            )


        print("--------------------------------")


        predicted_rows.append({

            "match_id": match["match_id"],
            "stage": stage,

            "team_a": match["team_a"],
            "team_b": match["team_b"],

            "predicted_score_a": score["a"],

            "predicted_score_b": score["b"],

            "predicted_winner": match["predicted_winner"]

        })


    print("\nPredicciones generadas correctamente.")
    
    output = pd.DataFrame(predicted_rows)

    output_path = ROOT / "data" / "processed" / f"{stage}_predicted_results.csv"

    output.to_csv(
        output_path,
        index=False
    )

    print("\nArchivo generado:")
    print(output_path) 
    
if __name__ == "__main__":

    if len(sys.argv)!=2:
        print(
            "Uso: python scripts/predict_next_stage.py R16"
        )
        sys.exit()


    main(sys.argv[1])