"""Lightweight geospatial helpers for Ottawa collision project."""
from __future__ import annotations

import sys
import pathlib

import geopandas as gpd
from shapely.geometry import Point

# Ensure project root on sys.path when run directly
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Default CRS used for project (Ottawa)
DEFAULT_CRS = "EPSG:26918"  # NAD83 / UTM 18N
WGS84 = "EPSG:4326"

def df_to_points(df, lon_col: str = "Long", lat_col: str = "Lat", crs: str = WGS84) -> gpd.GeoDataFrame:
    """Convert lon/lat columns to a GeoDataFrame with given CRS."""
    missing = df[lon_col].isna() | df[lat_col].isna()
    gdf = gpd.GeoDataFrame(
        df.loc[~missing].copy(),
        geometry=gpd.points_from_xy(df.loc[~missing, lon_col], df.loc[~missing, lat_col]),
        crs=crs,
    )
    return gdf

def to_projected(gdf: gpd.GeoDataFrame, crs: str = DEFAULT_CRS) -> gpd.GeoDataFrame:
    """Project geometries to the project CRS if needed."""
    if gdf.crs is None:
        raise ValueError("GeoDataFrame missing CRS; cannot project")
    if gdf.crs == crs:
        return gdf
    return gdf.to_crs(crs)

def spatial_join_points_to_lines(points: gpd.GeoDataFrame, lines: gpd.GeoDataFrame, buffer_m: float = 12.0, how: str = "left") -> gpd.GeoDataFrame:
    """Join points (e.g., collisions) to nearest line segments with optional buffer.

    Parameters
    ----------
    points : gpd.GeoDataFrame
        Point geometries in project CRS.
    lines : gpd.GeoDataFrame
        Line geometries in project CRS with an ID column (e.g., RD_SEGMENT_ID).
    buffer_m : float
        Buffer distance in meters to catch near-miss mappings.
    how : str
        Passed to geopandas.sjoin ("left"/"inner").
    """
    if points.crs != lines.crs:
        raise ValueError("points and lines must share CRS before joining")
    if buffer_m and buffer_m > 0:
        buffered = lines.copy()
        buffered["geometry"] = buffered.geometry.buffer(buffer_m)
    else:
        buffered = lines
    return gpd.sjoin(points, buffered, how=how, predicate="intersects")

__all__ = [
    "DEFAULT_CRS",
    "WGS84",
    "df_to_points",
    "to_projected",
    "spatial_join_points_to_lines",
]
