from pathlib import Path
import streamlit as st
import json
from datetime import datetime


st.title("⚙️ Configuración del Sistema")


# =====================
# Estado general
# =====================

st.subheader("🟢 Estado del Dashboard")


now = datetime.now()


col1,col2,col3 = st.columns(3)



with col1:

    st.metric(
        "Estado",
        "Operativo"
    )


with col2:

    st.metric(
        "Hora actual",
        now.strftime("%H:%M")
    )


with col3:

    st.metric(
        "Fecha",
        now.strftime("%d/%m/%Y")
    )


st.divider()



# =====================
# Directorios
# =====================


st.subheader(
    "📁 Verificación de archivos"
)



paths = {


"Reportes":
"outputs/reports",


"Historial":
"outputs/history",


"Modelos":
"models",


"Datos":
"data"

}



for name,path in paths.items():


    folder = Path(path)


    if folder.exists():

        st.success(
            f"✔ {name}: disponible"
        )

    else:

        st.error(
            f"✖ {name}: no encontrado"
        )



st.divider()



# =====================
# Reportes disponibles
# =====================


st.subheader(
    "📄 Reportes generados"
)



reports = Path(
    "outputs/reports"
).glob(
    "*.json"
)



reports=list(reports)



for r in reports:

    st.write(
        f"✔ {r.name}"
    )



st.divider()



# =====================
# Modelos ML
# =====================


st.subheader(
    "🤖 Modelos utilizados"
)


models=[

"Logistic Regression",

"Random Forest",

"XGBoost"

]


for m in models:

    st.write(
        f"✔ {m}"
    )



st.divider()



st.info(
"""
Este dashboard está diseñado para actualizarse
automáticamente durante las diferentes fases
del Mundial 2026.

Solo es necesario actualizar los resultados
reales y ejecutar nuevamente el pipeline.
"""
)