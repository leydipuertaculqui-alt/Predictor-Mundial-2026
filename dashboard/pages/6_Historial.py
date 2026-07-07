from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px


st.title("🕒 Historial del Sistema")


history_file = Path(
    "outputs/history/predictions_history.csv"
)


if not history_file.exists():

    st.warning(
        "No existe predictions_history.csv"
    )

    st.stop()



# =========================
# Cargar datos
# =========================

df = pd.read_csv(history_file)



df["date"] = pd.to_datetime(
    df["date"]
)



# =========================
# Última ejecución
# =========================

latest = df["date"].max()


latest_data = df[
    df["date"] == latest
]



col1,col2,col3,col4 = st.columns(4)



with col1:

    st.metric(
        "Última ejecución",
        latest.strftime("%d/%m/%Y %H:%M")
    )



with col2:

    st.metric(
        "Fase actual",
        latest_data["stage"].iloc[0]
    )



with col3:

    st.metric(
        "Partidos procesados",
        len(latest_data)
    )



with col4:

    st.metric(
        "Total histórico",
        len(df)
    )



st.divider()



# =========================
# Línea temporal
# =========================


st.subheader(
    "📅 Evolución de ejecuciones"
)



timeline = (

    df.groupby(
        ["date","stage"]
    )
    .size()
    .reset_index(
        name="matches"
    )

)



fig = px.line(

    timeline,

    x="date",

    y="matches",

    color="stage",

    markers=True,

    title="Partidos procesados por ejecución"

)



st.plotly_chart(

    fig,

    use_container_width=True

)



# =========================
# Resultados históricos
# =========================


st.subheader(
    "⚽ Historial de predicciones"
)



show = latest_data.copy()



show["partido"] = (

    show["team_a"]

    +" vs "

    +show["team_b"]

)



show["prediccion"] = (

    show["predicted_score_a"].astype(str)

    +" - "

    +show["predicted_score_b"].astype(str)

)



show["resultado_real"] = (

    show["real_score_a"]
    .fillna("-")
    .astype(str)

    +" - "

    +show["real_score_b"]
    .fillna("-")
    .astype(str)

)



show["probability"] = (

    show["probability"]*100

).round(1).astype(str)+"%"



table = show[

[
"partido",
"prediccion",
"resultado_real",
"predicted_winner",
"real_winner",
"probability"
]

]



st.dataframe(

table,

use_container_width=True,

hide_index=True

)



# =========================
# Filtros
# =========================


st.subheader(
    "🔎 Buscar historial"
)



stage = st.selectbox(

    "Seleccionar fase",

    ["Todas"] + list(df.stage.unique())

)



filtered=df.copy()



if stage!="Todas":

    filtered=filtered[
        filtered.stage==stage
    ]



st.write(
    f"Registros encontrados: {len(filtered)}"
)


st.dataframe(

filtered,

use_container_width=True,

hide_index=True

)