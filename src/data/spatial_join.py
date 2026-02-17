"""Spatial joins between cleaned datasets."""
from __future__ import annotations

import sys
import pathlib

import geopandas as gpd

# Ensure project root on sys.path when run directly
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.utils.geo import spatial_join_points_to_lines


def collisions_to_segments(collisions: gpd.GeoDataFrame, roads: gpd.GeoDataFrame, road_id: str = "RD_SEGMENT_ID", buffer_m: float = 12.0) -> gpd.GeoDataFrame:
    joined = spatial_join_points_to_lines(collisions, roads[[road_id, "geometry"]], buffer_m=buffer_m, how="left")
    return joined

__all__ = ["collisions_to_segments"]
