from pathlib import Path
import json
import streamlit as st
import pandas as pd
import plotly.express as px


st.title("🧠 Explainable AI")


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



decision = report.get(
    "decision_support",
    {}
)



# Información general

col1, col2 = st.columns(2)


with col1:

    st.metric(

        "Modelo seleccionado",

        decision.get(
            "selected_model",
            "-"
        )

    )


with col2:

    st.metric(

        "Razón",

        decision.get(
            "selection_reason",
            "-"
        )

    )



st.divider()



st.subheader(
    "Factores principales de decisión"
)



factors = decision.get(
    "main_factors",
    []
)



if not factors:

    st.warning(
        "No existen factores."
    )

    st.stop()



df = pd.DataFrame(
    factors
)



df = df.sort_values(
    "importance"
)



fig = px.bar(

    df,

    x="importance",

    y="feature",

    orientation="h",

    text="importance",

    title=
    "Importancia de variables del modelo"

)



fig.update_traces(

    texttemplate="%{text:.2f}",

    textposition="outside"

)



st.plotly_chart(

    fig,

    use_container_width=True

)



st.subheader(
    "Detalle de factores"
)



st.dataframe(

    df,

    use_container_width=True,

    hide_index=True

)