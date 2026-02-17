"""Tests for src.utils.geo."""
import pytest
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString

from src.utils.geo import (
    WGS84,
    DEFAULT_CRS,
    df_to_points,
    to_projected,
    spatial_join_points_to_lines,
)


def test_df_to_points():
    df = pd.DataFrame({"Long": [-75.7, -75.8], "Lat": [45.4, 45.5]})
    gdf = df_to_points(df, lon_col="Long", lat_col="Lat", crs=WGS84)
    assert len(gdf) == 2
    assert gdf.crs == WGS84
    assert gdf.geometry.iloc[0].x == pytest.approx(-75.7)
    assert gdf.geometry.iloc[0].y == pytest.approx(45.4)


def test_df_to_points_drops_missing():
    df = pd.DataFrame({"Long": [-75.7, None, -75.8], "Lat": [45.4, 45.5, None]})
    gdf = df_to_points(df, lon_col="Long", lat_col="Lat", crs=WGS84)
    assert len(gdf) == 1


def test_to_projected():
    gdf = gpd.GeoDataFrame(
        {"id": [1]},
        geometry=[Point(-75.7, 45.4)],
        crs=WGS84,
    )
    out = to_projected(gdf, DEFAULT_CRS)
    assert out.crs == DEFAULT_CRS
    assert out.geometry.iloc[0].x != -75.7  # projected coords differ


def test_spatial_join_points_to_lines():
    points = gpd.GeoDataFrame(
        {"id": [1, 2]},
        geometry=[Point(0, 0), Point(10, 10)],
        crs=DEFAULT_CRS,
    )
    line = LineString([(0, 0), (5, 0), (10, 0)])
    lines = gpd.GeoDataFrame(
        {"RD_SEGMENT_ID": [100]},
        geometry=[line],
        crs=DEFAULT_CRS,
    )
    joined = spatial_join_points_to_lines(points, lines, buffer_m=5, how="left")
    assert "RD_SEGMENT_ID" in joined.columns
    assert len(joined) >= 1
