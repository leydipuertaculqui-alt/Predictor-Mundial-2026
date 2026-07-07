import pandas as pd

from src import pipeline


def test_assemble_report_includes_match_entries():
    fixtures = [{"match_id": "M1", "team_a": "A", "team_b": "B"}]
    clf_results = {"logreg": [0.75], "random_forest": [0.6], "xgboost": [0.8]}
    report = pipeline.assemble_report(
        "R32",
        fixtures,
        clf_results,
        score_a=[2],
        score_b=[1],
        pen_proba=[0.2],
    )

    assert report["stage"] == "R32"
    assert len(report["matches"]) == 1
    assert report["matches"][0]["predicted_winner"] == "A"
    assert report["matches"][0]["predicted_score_90"] == {"a": 2, "b": 1}


def test_add_validation_summary_handles_no_real_results():
    report = {"stage": "R16"}
    fixtures = [{"match_id": "M1", "team_a": "A", "team_b": "B"}]
    fixtures_feat = pd.DataFrame([{"team_a": "A"}])
    clf_results = {"logreg": [0.7]}
    real_results = pd.DataFrame(columns=["match_id", "winner_final", "score_a_90", "score_b_90"])

    updated = pipeline.add_validation_summary(
        report,
        fixtures,
        fixtures_feat,
        clf_results,
        score_a=[1],
        score_b=[0],
        real_results=real_results,
    )

    assert updated["n_matches_validated"] == 0
    assert "validation_note" in updated
