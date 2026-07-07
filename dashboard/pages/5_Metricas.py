from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px


st.title("📈 Métricas del Modelo")


# Ruta del historial

metrics_file = Path(
    "outputs/history/metrics_history.csv"
)


if not metrics_file.exists():

    st.warning(
        "No existe metrics_history.csv"
    )

    st.stop()



# Leer datos

df = pd.read_csv(
    metrics_file
)



# Convertir fecha

df["date"] = pd.to_datetime(
    df["date"]
)



st.success(
    f"Registros cargados: {len(df)}"
)



# ---------------------------
# Filtros
# ---------------------------

st.sidebar.subheader(
    "Filtros de métricas"
)


stages = st.sidebar.multiselect(

    "Fase",

    options=df["stage"].unique(),

    default=df["stage"].unique()

)



models = st.sidebar.multiselect(

    "Modelo",

    options=df["model"].unique(),

    default=df["model"].unique()

)



filtered = df[

    df["stage"].isin(stages)

    &
    
    df["model"].isin(models)

]



# ---------------------------
# Tarjetas resumen
# ---------------------------


col1,col2,col3,col4 = st.columns(4)



with col1:

    st.metric(

        "Accuracy máximo",

        f"{filtered['accuracy'].max()*100:.1f}%"

    )



with col2:

    st.metric(

        "Mejor F1",

        f"{filtered['f1'].max()*100:.1f}%"

    )



with col3:

    st.metric(

        "MAE Score",

        f"{filtered['mae_score'].min():.3f}"

    )



with col4:

    st.metric(

        "RMSE Score",

        f"{filtered['rmse_score'].min():.3f}"

    )



st.divider()



# ---------------------------
# Accuracy evolución
# ---------------------------


st.subheader(
    "📊 Evolución del Accuracy"
)



fig = px.line(

    filtered,

    x="stage",

    y="accuracy",

    color="model",

    markers=True,

    title="Accuracy por fase"

)


fig.update_yaxes(

    tickformat=".0%"

)



st.plotly_chart(

    fig,

    use_container_width=True

)



# ---------------------------
# Tabla completa
# ---------------------------


st.subheader(
    "📋 Historial completo"
)



display = filtered.copy()


display["accuracy"] = (
    display["accuracy"]*100
).round(1).astype(str)+"%"



display["f1"] = (
    display["f1"]*100
).round(1).astype(str)+"%"



st.dataframe(

    display,

    use_container_width=True,

    hide_index=True

)