# High-Collision-Risk-Predictor

Predicts which Ottawa road segments are likely to be in the top 20% collision-risk band for the next year. 

## Data (in `data/raw/`)
- `Traffic_Collision_Data.csv` � collisions with date/time, coords, severity.
- `Road_Centrelines___Lignes_m%C3%A9dianes_de_route.shp` � road segments & geometry.
- `Upcoming_Construction_facilities%2C_culverts%2C_parks%2C_bridges.csv` � planned works.
- `Transportation_Midblock_Volumes_2024.csv` � volume proxy (optional in features).
- `OC_Transpo_Stops.csv` � transit stops (optional density feature).

## Repo layout
- `src/data/` � ingest, clean, spatial joins.
- `src/features/` � feature builder (`build.py`).
- `src/models/` � training (`train.py`), feature importance, interpretability.
- `src/utils/geo.py` � CRS + spatial helpers.
- `src/viz/map_results.py` � render interactive risk map.
- `data/processed/` � generated parquet tables.
- `models/` � trained artifacts, metrics, SHAP/importance outputs.
- `reports/` � figures, map (for Pages).
- `docs/` � deployed `index.html` (GitHub Pages).

## Quickstart
```
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Build features
```
python -m src.features.build --collisions data/raw/Traffic_Collision_Data.csv \
  --roads data/raw/Road_Centrelines___Lignes_m%C3%A9dianes_de_route.shp \
  --construction data/raw/Upcoming_Construction_facilities%2C_culverts%2C_parks%2C_bridges.csv \
  --buffer-m 12 --out data/processed/segment_year_features.parquet
```

## Train models
```
python -m src.models.train --config configs/logreg.yaml   # baseline
python -m src.models.train --config configs/lightgbm.yaml # best performer
```
Artifacts: `models/*_model.joblib`, `models/*_metrics.json`, `models/*_feature_importances.csv`, `models/*_shap_bar.png` (tree models).

## Model comparison (test year 2024)
| Model | ROC-AUC | PR-AUC | Brier | Recall@5% | Recall@10% | Notes |
|-------|--------:|-------:|------:|----------:|-----------:|-------|
| LightGBM | 0.936 | 0.842 | 0.075 | 0.238 | 0.459 | Best overall; tree-based, calibrated via prob outputs |
| Logistic Regression | 0.923 | 0.824 | 0.122 | 0.238 | 0.455 | Strong linear baseline |

Best model: **LightGBM** (see `models/lgbm_metrics.json`, `models/lgbm_model.joblib`).

## Interpretability
- **Feature importance (gain):** `models/lgbm_feature_importances.csv` (sorted).
- **Global SHAP (bar):** `models/lgbm_shap_bar.png` (top 20 drivers).
- **Permutation importance:** `python -m src.models.feature_importance --features data/processed/segment_year_features.parquet --models-dir models --out reports/feature_importance.png`.

## Map (GitHub Pages)
- Deployed map: https://ms-codess.github.io/High-Collision-Risk-Predictor/
