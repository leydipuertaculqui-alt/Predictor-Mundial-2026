from pathlib import Path
import json
import streamlit as st
import pandas as pd


st.title("⚽ Predicciones del Mundial")


# Buscar reporte más reciente

reports_dir = Path(
    "outputs/reports"
)


reports = sorted(
    reports_dir.glob(
        "reporte_*.json"
    )
)


if not reports:

    st.warning(
        "No existen reportes."
    )

    st.stop()


latest = reports[-1]


with open(
    latest,
    encoding="utf-8"
) as f:

    report=json.load(f)



stage = report.get(
    "stage",
    "-"
)


st.success(
    f"Fase actual: {stage}"
)



# Convertir partidos a tabla


rows=[]


for match in report["matches"]:

    score = match["predicted_score_90"]

    actual = match["actual_result"]["score_90"]


    rows.append({

        "Partido":
        f'{match["team_a"]} vs {match["team_b"]}',


        "Marcador Predicho":
        f'{score["a"]} - {score["b"]}',


        "Ganador Predicho":
        match["predicted_winner"],


        "Probabilidad":
        f'{match["advance_probability_avg_a"]*100:.1f}%',


        "Resultado Real":
        f'{actual["a"]} - {actual["b"]}',


        "Estado":
        "✅ Acierto"
        if match["hit"]
        else
        "❌ Fallo"

    })


df=pd.DataFrame(rows)



st.dataframe(

    df,

    use_container_width=True,

    hide_index=True

)