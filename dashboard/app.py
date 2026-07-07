import streamlit as st


st.set_page_config(
    page_title="Mundial Predictor ML",
    page_icon="🏆",
    layout="wide"
)


# ===============================
# CSS PROFESIONAL
# ===============================

st.markdown(
"""
<style>

body {
    font-family: 'Arial', sans-serif;
}


/* TITULO PRINCIPAL */

.main-title {
    font-size: 42px;
    font-weight: 800;
    color: #0066CC;
    text-align: center;
    margin-bottom: 10px;
}


/* SUBTITULO */

.subtitle {

    font-size: 20px;
    text-align:center;
    color: #555555;

}


/* TARJETAS */

.card {

    background: rgba(255,255,255,0.95);

    border-radius:15px;

    padding:25px;

    margin:10px;

    border:1px solid #dddddd;

    box-shadow:
    0px 4px 10px rgba(0,0,0,0.15);

}


/* TITULOS DE PAGINA */

h1 {

    color:#0066CC !important;

}


/* METRICAS */

.metric-title {

    font-size:18px;

    color:#555555;

}


.metric-value {

    font-size:32px;

    font-weight:bold;

    color:#111111;

}


/* SIDEBAR */

section[data-testid="stSidebar"] {

    background-color:#0B1F3A;

}


section[data-testid="stSidebar"] * {

    color:white;

}


</style>
""",
unsafe_allow_html=True
)



# ===============================
# PAGINA PRINCIPAL
# ===============================


st.markdown(
"""
<div class="main-title">

🏆 MUNDIAL PREDICTOR ML

</div>


<div class="subtitle">

Centro de Analítica y Predicción del Mundial 2026

</div>

<br>

""",
unsafe_allow_html=True
)



st.markdown(
"""
<div class="card">

<h2>
📊 Sistema Inteligente de Predicción
</h2>

<p>
Dashboard profesional para analizar:
</p>

<ul>

<li>Predicciones por fase del Mundial</li>

<li>Rendimiento de modelos Machine Learning</li>

<li>Métricas actualizadas automáticamente</li>

<li>Explicabilidad del modelo</li>

<li>Evolución histórica</li>

</ul>

</div>

""",
unsafe_allow_html=True
)



st.info(
"""
Seleccione una sección desde el menú lateral.
Los datos se cargan automáticamente desde:

outputs/reports

outputs/history

"""
)