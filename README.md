# Predictor Mundial 2026 — Fases Eliminatorias

Proyecto ejecutable (no una simulación de bracket completo) que implementa el diseño
acordado: predicción fase por fase, con actualización manual de datos reales entre
cada fase, usando Regresión Logística, Random Forest, XGBoost y un regresor de
marcador (objetivo Poisson).

## De dónde salen los datos (IMPORTANTE)

Este entorno de ejecución de código **no tiene acceso a internet**. Por eso los datos
en `data/raw/` fueron recolectados por mí (Claude) usando la herramienta de **búsqueda
web** en la conversación, verificados contra FIFA, ESPN, Wikipedia, Yahoo Sports,
CBS Sports, CNN, Fox Sports y NBC Sports (julio 2026), y luego transcritos a CSV/JSON
a mano. Son datos reales, no sintéticos — pero:

- `teams_static.csv`: ranking FIFA exacto para el top-6 (Argentina, España, Francia,
  Inglaterra, Portugal, Brasil); para el resto de los 32 equipos, los puntos son una
  **aproximación razonable** basada en su posición conocida aproximada, ya que no pude
  extraer la tabla completa de 32 posiciones exactas en el tiempo disponible. Reemplaza
  esos valores por los exactos de `inside.fifa.com/fifa-world-ranking/men` si necesitas
  precisión total.
- `players.csv`: estadísticas reales de goles/asistencias del torneo 2026 confirmadas
  (Messi 7 goles, Mbappé 7 goles, Haaland 5 goles, Ounahi 2 goles, etc.); edad y club
  son de conocimiento general y deberían revisarse contra una fuente tipo Transfermarkt.
  **Faltan la mayoría de jugadores por equipo** — solo incluí 1-2 jugadores clave por
  selección. Para producción real, esto debe completarse con un feed de Opta/FBref.
  `possession_avg`, `shots_on_target_avg`, `defensive_efficiency` a nivel de equipo
  quedaron en 0.0 (placeholders) por la misma razón — no tuve forma de extraer estadísticas
  granulares de posesión/tiros por partido en el tiempo disponible.
- `matches_played.csv`: los 16 resultados de dieciseisavos y los 2 de octavos ya jugados
  (Francia 1-0 Paraguay, Marruecos 3-0 Canadá) están verificados con marcador exacto y
  fuente cruzada (ESPN + FIFA + CNN + Al Jazeera).
- `historical_matches.csv`: resultados eliminatorios reales de los Mundiales 2014, 2018
  y 2022 (hechos históricos conocidos, no requieren búsqueda).
- `bracket_map.json`: estructura oficial de emparejamientos (partidos 73-104) según el
  documento FIFA publicado el 6-dic-2025, con los slots ya resueltos con los equipos
  reales confirmados de fase de grupos.

## Cómo correrlo

```bash
cd mundial2026_predictor
pip install -r requirements.txt   # incluye xgboost real si tienes acceso a internet
python scripts/run_stage.py R32   # predice dieciseisavos con datos "pre-torneo" y valida vs. la realidad
python scripts/run_stage.py R16   # actualiza con resultados reales de R32 y predice octavos
python scripts/run_stage.py QF    # solo funcionará cuando los 8 partidos de octavos tengan resultado real
```

Cada comando es independiente. Si intentas correr una fase cuyo bracket todavía depende
de partidos sin jugar, el script se detiene con un mensaje explícito en vez de simular
o encadenar predicciones.

## Resultado de la validación de dieciseisavos (ya ejecutado)

Con los datos "congelados" al 27 de junio (antes de dieciseisavos) y el modelo
entrenado solo con partidos eliminatorios de 2014/2018/2022:

| Modelo | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Regresión Logística | 0.50 | 0.64 | 0.64 | 0.64 |
| Random Forest | 0.69 | 0.69 | 1.00 | 0.82 |
| XGBoost (o GradientBoosting si no hay xgboost instalado) | 0.63 | 0.67 | 0.91 | 0.77 |
| Modelo de marcador | MAE 0.84 | RMSE 1.24 | | |

El tamaño de muestra de entrenamiento es pequeño (36 partidos históricos que cruzan
con los 32 equipos de 2026) — esto es una limitación estructural del problema, no un
error de implementación, y está documentada en el diseño original.

## Qué falta para llevar esto a producción real

1. Reemplazar el ranking FIFA aproximado por la tabla oficial completa de 32 posiciones.
2. Añadir estadísticas de posesión/tiros/eficiencia defensiva por equipo y por partido
   (requiere un feed con acceso a internet, ej. Opta, FBref, o la propia API de FIFA).
3. Completar `players.csv` con la plantilla completa de cada selección, no solo 1-2
   jugadores clave.
4. Verificar/expandir `historical_h2h.csv` — solo incluí los enfrentamientos que pude
   confirmar con certeza; el resto quedó en 0 (sin historial) como supuesto conservador.
5. Instalar `xgboost` real (aquí corrió con el sustituto `GradientBoostingClassifier`
   de scikit-learn porque este entorno no tiene acceso a internet para `pip install`).
