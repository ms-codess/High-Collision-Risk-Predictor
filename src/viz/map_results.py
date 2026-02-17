"""Build an interactive map of best classification model risk predictions."""
from __future__ import annotations

import sys
import pathlib
import argparse
import json

import pandas as pd
import geopandas as gpd
from joblib import load

# Ensure project root on path
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.utils.geo import WGS84


def get_best_model_and_predict(
    features_path: pathlib.Path,
    models_dir: pathlib.Path,
    test_year: int = 2024,
    metric: str = "pr_auc",
) -> tuple[str, pd.DataFrame]:
    """Load all model metrics, pick best by metric, load model and predict on test year.

    Returns
    -------
    (best_model_name, dataframe with RD_SEGMENT_ID, year, high_risk, proba, predicted)
    """
    metrics_files = list(models_dir.glob("*_metrics.json"))
    if not metrics_files:
        raise FileNotFoundError(f"No *_metrics.json found in {models_dir}. Train models first.")

    best_name = None
    best_value = -1.0
    for p in metrics_files:
        with open(p) as f:
            m = json.load(f)
        if m.get("test_year") != test_year:
            continue
        v = m.get(metric)
        if v is not None and v > best_value:
            best_value = v
            best_name = p.stem.replace("_metrics", "")

    if best_name is None:
        raise ValueError(f"No metrics for test_year={test_year}. Train with test_year={test_year}.")

    df = pd.read_parquet(features_path)
    test_mask = df["year"] == test_year
    if not test_mask.any():
        raise ValueError(f"No rows for year {test_year} in {features_path}")

    # Drop label and same-year outcome cols (must match train.py so pipeline gets same features)
    leak_cols = ["high_risk", "collisions_total", "fatal_major", "injuries_total"]
    drop_cols = [c for c in leak_cols if c in df.columns]
    X_test = df.loc[test_mask].drop(columns=drop_cols)
    y_test = df.loc[test_mask, "high_risk"].astype(int)
    segment_year = df.loc[test_mask, ["RD_SEGMENT_ID", "year"]].copy()

    pipe = load(models_dir / f"{best_name}_model.joblib")
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = pipe.predict(X_test)

    out = segment_year.copy()
    out["high_risk"] = y_test.values
    out["proba"] = proba
    out["predicted"] = pred
    return best_name, out


def build_risk_map(
    predictions_df: pd.DataFrame,
    roads_path: pathlib.Path,
    output_html: pathlib.Path,
    *,
    segment_id_col: str = "RD_SEGMENT_ID",
    proba_col: str = "proba",
    title: str = "Ottawa Traffic Collision Risk",
    model_name: str | None = None,
    test_year: int | None = None,
) -> None:
    """Create a Folium map of road segments colored by predicted risk probability."""
    import folium

    roads = gpd.read_file(roads_path)
    # Shapefile truncates column names to 10 chars
    if "RD_SEGMENT" in roads.columns and segment_id_col not in roads.columns:
        roads = roads.rename(columns={"RD_SEGMENT": segment_id_col})
    if roads.crs and roads.crs != WGS84:
        roads = roads.to_crs(WGS84)

    # One row per segment for map (use max proba if multiple years)
    seg = (
        predictions_df.groupby(segment_id_col)[proba_col]
        .max()
        .reset_index()
        .rename(columns={proba_col: "risk_proba"})
    )

    roads = roads.merge(seg, left_on=segment_id_col, right_on=segment_id_col, how="inner")
    roads["risk_proba"] = roads["risk_proba"].fillna(0).clip(0, 1).astype(float)
    roads["risk_pct"] = (roads["risk_proba"] * 100).round(0).astype(int).astype(str) + "%"
    # Keep only columns needed for map (avoid Timestamp/non-JSON in __geo_interface__)
    roads = roads[[segment_id_col, "risk_proba", "risk_pct", "geometry"]].copy()

    # Ottawa centre
    centre = roads.geometry.union_all().centroid
    m = folium.Map(
        location=[centre.y, centre.x],
        zoom_start=11,
        tiles="CartoDB positron",
        control_scale=True,
    )

    # Risk bands (align with style_fn); label with percentage range
    RISK_BANDS = [
        (0.0, 0.2, "#91cf60", "Low (0–20%)"),
        (0.2, 0.4, "#fee08b", "Medium (20–40%)"),
        (0.4, 0.7, "#fc8d59", "High (40–70%)"),
        (0.7, 1.01, "#d73027", "Very high (70%+)"),
    ]

    def style_fn(feature):
        p = feature.get("properties", {}).get("risk_proba") or 0
        if p >= 0.7:
            color, weight = "#d73027", 4
        elif p >= 0.4:
            color, weight = "#fc8d59", 3
        elif p >= 0.2:
            color, weight = "#fee08b", 2
        else:
            color, weight = "#91cf60", 2
        return {"color": color, "weight": weight, "opacity": 0.85}

    folium.GeoJson(
        roads.__geo_interface__,
        style_function=lambda x: style_fn(x),
        name="Risk",
        tooltip=folium.GeoJsonTooltip(
            fields=[segment_id_col, "risk_pct"],
            aliases=["Segment ID", "Risk"],
            localize=True,
            style=(
                "background-color: rgba(255,255,255,0.95); border: 1px solid #e5e7eb; "
                "border-radius: 6px; padding: 8px 12px; font-size: 12px; font-family: 'Segoe UI', sans-serif;"
            ),
        ),
    ).add_to(m)

    folium.LayerControl(position="topright").add_to(m)

    # Title block (top of map)
    title_html = f"""
    <div style="
      position: fixed; top: 10px; left: 50%; transform: translateX(-50%);
      z-index: 9999; padding: 14px 28px; background: rgba(255,255,255,0.96);
      border-radius: 8px; box-shadow: 0 2px 12px rgba(0,0,0,0.15);
      font-family: 'Segoe UI', system-ui, sans-serif; text-align: center;
    ">
      <div style="font-size: 20px; font-weight: 600; color: #1a1a2e;">{title}</div>
      <div style="font-size: 12px; color: #6b7280; margin-top: 4px;">Predicted collision risk by road segment · Hover for details</div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    # Info panel (bottom-left): model, year, data source, methodology
    n_segments = len(roads)
    info_model = f"Model: {model_name}" if model_name else "Model: best by PR-AUC"
    info_year_line = f'<div style="margin-bottom: 4px;">Prediction year: {test_year}</div>' if test_year is not None else ""
    info_html = f"""
    <div style="
      position: fixed; bottom: 30px; left: 10px; z-index: 9999; max-width: 300px;
      padding: 14px 16px; background: rgba(255,255,255,0.95);
      border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.15);
      font-family: 'Segoe UI', system-ui, sans-serif; font-size: 12px; line-height: 1.5; color: #374151;
    ">
      <div style="font-weight: 600; color: #1a1a2e; margin-bottom: 8px; border-bottom: 1px solid #e5e7eb; padding-bottom: 6px;">About this map</div>
      <div style="margin-bottom: 4px;">{info_model}</div>
      {info_year_line}
      <div style="margin-bottom: 4px;">Segments shown: {n_segments:,}</div>
      <div style="margin-top: 8px; font-size: 11px; color: #6b7280;">
        <strong>Risk</strong> = predicted probability that a segment is in the <strong>top 20%</strong> of collision counts for that year. Uses road attributes, construction, and prior-year collision history. <strong>Data:</strong> City of Ottawa Open Data (collisions, road centrelines, construction).
      </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(info_html))

    # Legend (bottom-right)
    legend_items = "".join(
        f'<li><span style="background:{color}; width:18px; height:12px; display:inline-block; border-radius:2px; margin-right:8px; vertical-align:middle;"></span>{label}</li>'
        for _lo, _hi, color, label in RISK_BANDS
    )
    legend_html = f"""
    <div style="
      position: fixed; bottom: 30px; right: 10px; z-index: 9999;
      padding: 14px 18px; background: rgba(255,255,255,0.95);
      border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.15);
      font-family: 'Segoe UI', system-ui, sans-serif; font-size: 13px;
    ">
      <div style="font-weight: 600; color: #1a1a2e; margin-bottom: 8px; border-bottom: 1px solid #e5e7eb; padding-bottom: 6px;">Risk level</div>
      <ul style="list-style:none; margin:0; padding:0; line-height: 1.8;">{legend_items}</ul>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    output_html.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output_html))
    print(f"Map saved to {output_html}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build risk map from best classifier")
    parser.add_argument("--features", type=pathlib.Path, default=pathlib.Path("data/processed/segment_year_features.parquet"))
    parser.add_argument("--models-dir", type=pathlib.Path, default=pathlib.Path("models"))
    parser.add_argument("--roads", type=pathlib.Path, default=pathlib.Path("data/raw/Road_Centrelines___Lignes_médianes_de_route.shp"))
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("reports/risk_map.html"))
    parser.add_argument("--test-year", type=int, default=2024)
    parser.add_argument("--metric", default="pr_auc", choices=["pr_auc", "roc_auc"])
    args = parser.parse_args()

    best_name, pred_df = get_best_model_and_predict(
        args.features, args.models_dir, test_year=args.test_year, metric=args.metric
    )
    print(f"Best model: {best_name} (by {args.metric})")

    build_risk_map(
        pred_df,
        args.roads,
        args.out,
        title="Ottawa Traffic Collision Risk",
        model_name=best_name,
        test_year=args.test_year,
    )

    reports_dir = args.out.parent
    with open(reports_dir / "best_model.txt", "w") as f:
        f.write(best_name)
    print(f"Best model name written to {reports_dir / 'best_model.txt'}")


if __name__ == "__main__":
    main()
