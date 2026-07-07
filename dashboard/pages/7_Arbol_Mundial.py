from pathlib import Path
import json
import streamlit as st


st.title("🌍 Árbol del Mundial 2026")


reports_dir = Path(
    "outputs/reports"
)



# =========================
# cargar reportes
# =========================

reports = sorted(
    reports_dir.glob("reporte_*.json")
)


if not reports:

    st.warning(
        "No existen reportes."
    )

    st.stop()



# =========================
# mostrar fases
# =========================


for report_file in reports:

    with open(
        report_file,
        encoding="utf-8"
    ) as f:

        data=json.load(f)



    stage=data["stage"]


    st.divider()


    st.subheader(
        f"🏆 {stage}"
    )



    for match in data["matches"]:


        team_a = match["team_a"]

        team_b = match["team_b"]


        score = match.get(
            "predicted_score_90",
            {}
        )


        winner = match.get(
            "actual_result",
            {}
        ).get(
            "winner"
        )


        predicted = match.get(
            "predicted_winner"
        )



        col1,col2,col3 = st.columns(
            [2,1,2]
        )



        with col1:

            st.markdown(
                f"""
                ### {team_a}

                """
            )



        with col2:

            st.markdown(
                "### ⚔️"
            )



        with col3:

            st.markdown(
                f"""
                ### {team_b}

                """
            )



        st.write(
            f"""
            Predicción:
            **{score.get('a','-')} - {score.get('b','-')}**

            """
        )



        if winner:


            st.success(
                f"Clasificado: {winner}"
            )


        else:


            st.info(
                f"Predicción de avance: {predicted}"
            )