
# 🚦 High Collision Risk Predictor – Ottawa

Predicts which **Ottawa road segments** are likely to fall into the **top 20% collision-risk band** for the upcoming year using spatial + temporal machine learning features.

🔗 **Live Risk Map (GitHub Pages):**
[https://ms-codess.github.io/High-Collision-Risk-Predictor/](https://ms-codess.github.io/High-Collision-Risk-Predictor/)

---

## 🎯 Project Objective

This project builds a **segment-level risk classification model** to help:

* Identify high-risk road segments proactively
* Support urban safety planning
* Prioritize infrastructure or enforcement interventions

The target is binary:

> **1 = Segment in top 20% collision-risk band next year**
> **0 = Otherwise**

---

##  Data Sources (`data/raw/`)

###  Traffic Collision Data
* `Traffic_Collision_Data.csv`

  * Date / time
  * Coordinates
  * Severity

---

### Road Network (Base Geometry)
* `Road_Centrelines___Lignes_médianes_de_route.shp`

  * Road segment geometry
  * Segment identifiers

---

###  Upcoming Construction

* `Upcoming_Construction_facilities,culverts,parks,bridges.csv`

  * Planned works
  * Spatial proximity features

---

### 🚦 Optional Feature Enhancers

* `Transportation_Midblock_Volumes_2024.csv` → traffic volume proxy
* `OC_Transpo_Stops.csv` → transit stop density

---

## 🏗 Repository Structure

```
High-Collision-Risk-Predictor/
│
├── data/
│   ├── raw/              # Original datasets
│   └── processed/        # Feature tables (parquet)
│
├── src/
│   ├── data/             # Cleaning + spatial joins
│   ├── features/         # Feature engineering (build.py)
│   ├── models/           # Training + evaluation
│   ├── utils/geo.py      # CRS + geospatial helpers
│   └── viz/              # Interactive risk map
│
├── models/               # Saved models + metrics
├── reports/              # Figures + exported outputs
├── docs/                 # GitHub Pages site
└── configs/              # Model configs (YAML)
```

This follows a **production-style ML pipeline**:

1. Ingest
2. Clean
3. Spatial join
4. Feature engineering
5. Model training
6. Interpretability
7. Deployment (map)

---

## ⚙️ Quickstart

Create environment:

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

---

## 🧮 Build Features

```bash
python -m src.features.build \
  --collisions data/raw/Traffic_Collision_Data.csv \
  --roads data/raw/Road_Centrelines___Lignes_médianes_de_route.shp \
  --construction data/raw/Upcoming_Construction_facilities,culverts,parks,bridges.csv \
  --buffer-m 12 \
  --out data/processed/segment_year_features.parquet
```

Output:
`segment_year_features.parquet`

Includes:

* Historical collision counts (rolling windows)
* Severity-weighted risk
* Distance to construction
* Transit stop density
* Traffic volume proxy
* Spatial lag features

---

##  Model Training

Baseline:

```bash
python -m src.models.train --config configs/logreg.yaml
```

Best model:

```bash
python -m src.models.train --config configs/lightgbm.yaml
```

Saved artifacts:

* `*_model.joblib`
* `*_metrics.json`
* `*_feature_importances.csv`
* `*_shap_bar.png`

---

## 📊 Model Performance (Test Year: 2024)

| Model               |   ROC-AUC |    PR-AUC |     Brier | Recall@5% | Recall@10% |
| ------------------- | --------: | --------: | --------: | --------: | ---------: |
| **LightGBM**        | **0.936** | **0.842** | **0.075** |     0.238 |      0.459 |
| Logistic Regression |     0.923 |     0.824 |     0.122 |     0.238 |      0.455 |

### Best Model: LightGBM

Why it wins:

* Handles non-linearity
* Captures interaction effects
* Strong calibration
* Robust with tabular + spatial features

---


