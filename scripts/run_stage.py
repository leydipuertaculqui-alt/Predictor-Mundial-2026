"""
Uso:
    python scripts/run_stage.py R32
    python scripts/run_stage.py R16
    python scripts/run_stage.py QF
    ... etc.

Cada invocación es independiente: NO encadena predicciones de una fase a otra.
Antes de correr una fase, el script exige (vía advance_to_next_stage_guard) que
todos los partidos de esa fase tengan sus equipos resueltos por resultados reales
(bracket fijo + resultados ya jugados), nunca por predicciones previas.
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config as cfg, pipeline as pl


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in cfg.STAGES:
        print(f"Uso: python {sys.argv[0]} <fase>   (fases válidas: {cfg.STAGES})")
        sys.exit(1)

    stage = sys.argv[1]

    # Salvaguarda: solo se corre si el bracket ya tiene equipos reales para esta fase.
    idx = cfg.STAGES.index(stage)
    if idx > 0:
        prev_stage = cfg.STAGES[idx - 1] if stage != "F" else "SF"
        try:
            pl.advance_to_next_stage_guard(prev_stage, stage)
        except RuntimeError as e:
            print(f"[detenido] {e}")
            sys.exit(1)

    report = pl.run_stage(stage, validate=True)

    out_path = cfg.REPORTS / f"reporte_{stage}.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n=== Reporte de fase {stage} ===")
    for m in report["matches"]:
        hit_str = ""
        if "hit" in m:
            hit_str = "  ✅ ACIERTO" if m["hit"] else "  ❌ FALLO"
        real_str = ""
        if "actual_result" in m:
            r = m["actual_result"]
            real_str = f"  | Real: {r['score_90']['a']}-{r['score_90']['b']} (ganó {r['winner']})"
        print(f"{m['team_a']:>4} {m['predicted_score_90']['a']}-{m['predicted_score_90']['b']} {m['team_b']:<4}"
              f"  | Predicho avanza: {m['predicted_winner']:<4} (p={m['advance_probability_avg_a']:.2f})"
              f"{real_str}{hit_str}")

    if "validation_summary" in report:
        print("\n--- Validación contra resultados reales ---")
        for name, metrics in report["validation_summary"].items():
            print(f"  {name}: {metrics}")

    print(f"\nReporte completo guardado en: {out_path}")


if __name__ == "__main__":
    main()
