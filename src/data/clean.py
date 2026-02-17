"""Cleaning and standardization steps for raw datasets."""
from __future__ import annotations

import sys
import pathlib

import pandas as pd
import geopandas as gpd
from pandas.api.types import CategoricalDtype

from shapely import wkt


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.utils.geo import df_to_points, to_projected, DEFAULT_CRS, WGS84


def clean_collisions(df: pd.DataFrame) -> gpd.GeoDataFrame:
    df = df.copy()
    df["Accident_Date"] = pd.to_datetime(df["Accident_Date"], errors="coerce")
    df["year"] = df["Accident_Date"].dt.year
    df["month"] = df["Accident_Date"].dt.month
    df["dow"] = df["Accident_Date"].dt.dayofweek
    df["hour"] = pd.to_numeric(df.get("Hour", pd.NA), errors="coerce")

    # Normalize select categoricals
    for col in ["Light", "Traffic_Control", "Classification_Of_Accident"]:
        if col in df:
            df[col] = df[col].astype("string").str.strip().str.upper()

    # Fatal/injury counts for aggregation (support common Ottawa/open-data column names)
    for target, candidates in [
        ("num_of_fatal", ["Fatalities", "NUM_FATAL", "Fatal", "num_of_fatal"]),
        ("num_of_injuries", ["Injuries", "NUM_INJURIES", "Injury_Count", "num_of_injuries", "Injuries_Total"]),
    ]:
        if target not in df.columns:
            found = next((c for c in candidates if c in df.columns), None)
            df[target] = pd.to_numeric(df[found], errors="coerce").fillna(0).astype(int) if found else 0

    gdf = df_to_points(df, lon_col="Long", lat_col="Lat", crs=WGS84)
    gdf = to_projected(gdf, DEFAULT_CRS)
    return gdf


# Shapefile truncates column names to 10 chars; map to expected names
_ROAD_COL_RENAME = {
    "RD_SEGMENT": "RD_SEGMENT_ID",
    "SHAPE_Leng": "SHAPE_Length",
    "SUBTYPE_TE": "SUBTYPE_TEXT",
    "GRADE_SEPA": "GRADE_SEPARATED",
}


def clean_roads(df: pd.DataFrame | gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Standardize road centreline GeoDataFrame, ensuring geometry + length (meters)."""
    # If it's already a GeoDataFrame with geometry, keep it
    if isinstance(df, gpd.GeoDataFrame):
        gdf = df.copy()
    else:
        df = df.copy()
        if "geometry" in df.columns:
            gdf = gpd.GeoDataFrame(df, geometry=gpd.GeoSeries.from_wkt(df["geometry"]), crs=WGS84)
        else:
            raise ValueError("Road data must include geometry (shapefile/geojson or CSV with WKT 'geometry').")

    # Normalize shapefile-truncated column names
    rename = {k: v for k, v in _ROAD_COL_RENAME.items() if k in gdf.columns and v not in gdf.columns}
    if rename:
        gdf = gdf.rename(columns=rename)

    # Set CRS if missing, then project
    if gdf.crs is None:
        gdf.set_crs(WGS84, inplace=True)
    gdf = to_projected(gdf, DEFAULT_CRS)

    # Length in meters
    if "SHAPE_Length" in gdf:
        gdf["length_m"] = gdf["SHAPE_Length"]
    else:
        gdf["length_m"] = gdf.geometry.length
    return gdf


def clean_construction(df: pd.DataFrame) -> gpd.GeoDataFrame:
    df = df.copy()
    if "TARGETED_START" in df.columns:
        df["TARGETED_START"] = pd.to_datetime(df["TARGETED_START"], errors="coerce")
    # Support X/Y or Longitude/Latitude
    lon_col = next((c for c in ["X", "Longitude", "LONG", "lon"] if c in df.columns), None)
    lat_col = next((c for c in ["Y", "Latitude", "LAT", "lat"] if c in df.columns), None)
    if lon_col is None or lat_col is None:
        raise ValueError("Construction data must have X/Y or Longitude/Latitude columns")
    gdf = df_to_points(df, lon_col=lon_col, lat_col=lat_col, crs=WGS84)
    gdf = to_projected(gdf, DEFAULT_CRS)
    return gdf

__all__ = [
    "clean_collisions",
    "clean_roads",
    "clean_construction",
]
