# ⚽ Predictor Mundial FIFA 2026

## Descripción general

Este proyecto implementa un flujo de predicción progresiva para las fases eliminatorias del Mundial FIFA 2026. La metodología conserva el enfoque original:

- Entrenamiento con resultados históricos de los Mundiales 2014, 2018 y 2022.
- Uso de tres modelos de clasificación: Logistic Regression, Random Forest y XGBoost.
- Uso de dos regresores para estimar el marcador antes de los penales.
- Predicción fase por fase, usando únicamente resultados reales para resolver los partidos de fases posteriores.
- Actualización automática de reportes y validación cuando ya existen resultados oficiales.

## Arquitectura

- [src/config.py](src/config.py): configuración central, rutas y listas de features.
- [src/data_loader.py](src/data_loader.py): carga de datos y resolución del bracket.
- [src/feature_engineering.py](src/feature_engineering.py): ingeniería de features para entrenamiento y predicción.
- [src/models.py](src/models.py): construcción de clasificadores y regresores.
- [src/pipeline.py](src/pipeline.py): orquestación del flujo de predicción y validación.
- [scripts/run_stage.py](scripts/run_stage.py): punto de entrada CLI para ejecutar una fase.

## Ejecución

```bash
python scripts/run_stage.py R32
python scripts/run_stage.py R16
```

## Pruebas

```bash
python -m pytest -q
```
