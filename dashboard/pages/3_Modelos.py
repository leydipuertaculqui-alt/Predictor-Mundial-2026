from pathlib import Path
import json
import streamlit as st
import pandas as pd
import plotly.express as px


st.title("🤖 Comparación de Modelos")


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

    report = json.load(f)



st.success(
    f"Reporte analizado: {latest.name}"
)



# Obtener métricas

summary = report.get(
    "validation_summary",
    {}
)



data=[]


for model, metrics in summary.items():

    if model == "score_model":
        continue


    data.append({

        "Modelo": model.replace("_"," ").title(),

        "Accuracy":
        metrics.get(
            "accuracy",
            0
        ),

        "Precision":
        metrics.get(
            "precision",
            0
        ),

        "Recall":
        metrics.get(
            "recall",
            0
        ),

        "F1":
        metrics.get(
            "f1",
            0
        )

    })



df=pd.DataFrame(data)



# Mostrar tabla

st.subheader(
    "📊 Métricas por modelo"
)


st.dataframe(

    df.style.format(
        {
            "Accuracy":"{:.3f}",
            "Precision":"{:.3f}",
            "Recall":"{:.3f}",
            "F1":"{:.3f}"
        }
    ),

    use_container_width=True

)



# Gráfico accuracy

st.subheader(
    "Accuracy comparativo"
)



fig = px.bar(

    df,

    x="Modelo",

    y="Accuracy",

    text="Accuracy",

    title="Comparación de Accuracy"

)



fig.update_traces(

    texttemplate="%{text:.1%}",

    textposition="outside"

)


st.plotly_chart(

    fig,

    use_container_width=True

)



# Mejor modelo

best = report.get(
    "best_classifier",
    "-"
)



st.success(

    f"🏆 Modelo seleccionado: {best}"

)