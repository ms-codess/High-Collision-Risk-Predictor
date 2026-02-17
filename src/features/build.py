"""Feature building pipeline to create segment-year dataset and high-risk label."""
from __future__ import annotations

import sys
import pathlib
import argparse
from typing import Optional, Iterable

import pandas as pd
import geopandas as gpd

# Ensure project root on path for direct execution
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data import ingest, clean, spatial_join  # noqa: E402
from src.utils.geo import to_projected, DEFAULT_CRS  # noqa: E402


def _safe_sum(df: pd.DataFrame, cols: Iterable[str]) -> pd.Series:
    present = [c for c in cols if c in df.columns]
    if not present:
        return pd.Series([0] * len(df), index=df.index)
    return df[present].fillna(0).sum(axis=1)


def build_features(
    collision_path: pathlib.Path,
    roads_path: pathlib.Path,
    construction_path: Optional[pathlib.Path] = None,
    buffer_m: float = 12.0,
) -> pd.DataFrame:
    """Create segment-year feature table with high-risk label."""
    # Collisions
    raw_collisions = ingest.read_collisions(str(collision_path))
    collisions = clean.clean_collisions(raw_collisions)

    # Roads
    roads_gdf = ingest.read_roads(str(roads_path))
    roads = clean.clean_roads(roads_gdf)

    # Collision -> segment join
    joined = spatial_join.collisions_to_segments(collisions, roads, road_id="RD_SEGMENT_ID", buffer_m=buffer_m)

    # Construction flag (optional)
    construction_flag = None
    if construction_path:
        construction_raw = ingest.read_construction(str(construction_path))
        construction = clean.clean_construction(construction_raw)
        # spatial join to flag segments with any construction point
        construction = construction[["geometry"]].copy()
        construction = to_projected(construction, DEFAULT_CRS)
        construction["tmp"] = 1
        construction_buffered = construction.copy()
        construction_buffered["geometry"] = construction_buffered.geometry.buffer(20)
        cjoin = gpd.sjoin(roads[["RD_SEGMENT_ID", "geometry"]], construction_buffered, how="left", predicate="intersects")
        construction_flag = (
            cjoin.groupby("RD_SEGMENT_ID")["tmp"]
            .max()
            .rename("construction_flag")
            .fillna(0)
            .astype(int)
            .reset_index()
        )

    # Aggregations per segment-year (fatal/injuries optional)
    agg = (
        joined.assign(collisions=1)
        .groupby(["RD_SEGMENT_ID", "year"], as_index=False)
        .agg(collisions_total=("collisions", "sum"))
    )
    for col, src in [("fatal_major", "num_of_fatal"), ("injuries_total", "num_of_injuries")]:
        if src in joined.columns:
            sums = joined.groupby(["RD_SEGMENT_ID", "year"])[src].sum().reset_index().rename(columns={src: col})
            agg = agg.merge(sums, on=["RD_SEGMENT_ID", "year"], how="left")
        if col not in agg.columns:
            agg[col] = 0

    # Merge road attributes
    road_cols = [
        "RD_SEGMENT_ID",
        "length_m",
        "SUBTYPE_TEXT",
        "SUBCLASS",
        "OWNERSHIP",
        "FLOW",
        "GRADE_SEPARATED",
    ]
    roads_small = roads[[c for c in road_cols if c in roads.columns]].drop_duplicates("RD_SEGMENT_ID")
    df = agg.merge(roads_small, on="RD_SEGMENT_ID", how="left")

    # Add construction flag
    if construction_flag is not None:
        df = df.merge(construction_flag, on="RD_SEGMENT_ID", how="left")
    if "construction_flag" not in df:
        df["construction_flag"] = 0

    # Fill missing numeric values with zeros
    num_cols = [c for c in df.columns if c not in {"RD_SEGMENT_ID", "year", "SUBTYPE_TEXT", "SUBCLASS", "OWNERSHIP", "FLOW", "GRADE_SEPARATED"}]
    df[num_cols] = df[num_cols].fillna(0)

    # Lagged features (previous year) — usable at predict time without leaking same-year outcome
    df = df.sort_values(["RD_SEGMENT_ID", "year"])
    for col, lag_col in [
        ("collisions_total", "collisions_prev_year"),
        ("fatal_major", "fatal_prev_year"),
        ("injuries_total", "injuries_prev_year"),
    ]:
        df[lag_col] = df.groupby("RD_SEGMENT_ID")[col].shift(1)
    df["collisions_prev_year"] = df["collisions_prev_year"].fillna(0).astype(int)
    if "fatal_prev_year" in df.columns:
        df["fatal_prev_year"] = df["fatal_prev_year"].fillna(0).astype(int)
    if "injuries_prev_year" in df.columns:
        df["injuries_prev_year"] = df["injuries_prev_year"].fillna(0).astype(int)

    # Label: top 20% collisions per year (from current-year counts only)
    labels = []
    for y, g in df.groupby("year"):
        threshold = g["collisions_total"].quantile(0.8)
        lab = (g["collisions_total"] >= threshold).astype(int)
        labels.append(lab)
    df["high_risk"] = pd.concat(labels).sort_index()

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build segment-year feature table")
    parser.add_argument("--collisions", default="data/raw/Traffic_Collision_Data.csv", type=pathlib.Path)
    parser.add_argument("--roads", default="data/raw/Road_Centrelines___Lignes_médianes_de_route.shp", type=pathlib.Path)
    parser.add_argument("--construction", default="data/raw/Upcoming_Construction_facilities%2C_culverts%2C_parks%2C_bridges.csv", type=pathlib.Path)
    parser.add_argument("--buffer-m", default=12.0, type=float)
    parser.add_argument("--out", default="data/processed/segment_year_features.parquet", type=pathlib.Path)
    args = parser.parse_args()

    out_df = build_features(args.collisions, args.roads, args.construction, buffer_m=args.buffer_m)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.out, index=False)
    print(f"Wrote features to {args.out} (rows={len(out_df)})")


if __name__ == "__main__":
    main()
