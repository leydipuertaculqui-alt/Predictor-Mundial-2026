from pathlib import Path
import json
import streamlit as st


from components.cards import metric_card



st.title("📊 Resumen General")


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
        "No existen reportes generados"
    )

    st.stop()



latest = reports[-1]


with open(
    latest,
    encoding="utf-8"
) as f:

    report=json.load(f)



st.success(
    f"Última actualización: {latest.name}"
)



metrics = report.get(
    "best_classifier_metrics",
    {}
)



col1,col2,col3,col4 = st.columns(4)



with col1:

    metric_card(
        "Modelo",
        report.get(
            "best_classifier",
            "-"
        ),
        "🤖"
    )



with col2:

    metric_card(
        "Accuracy",
        metrics.get(
            "accuracy",
            "-"
        ),
        "📈"
    )



with col3:

    metric_card(
        "Partidos",
        len(
            report.get(
                "matches",
                []
            )
        ),
        "⚽"
    )



with col4:

    metric_card(
        "Fase",
        "R32",
        "🏆"
    )