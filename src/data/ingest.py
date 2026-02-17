"""Raw data ingestion helpers."""
from __future__ import annotations

import sys
import pathlib

import pandas as pd
import geopandas as gpd

# Ensure project root on sys.path when run directly
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


def read_collisions(path: str) -> pd.DataFrame:
    dtype = {
        "Accident_Year": "Int64",
        "Accident_Date": "string",
        "Light": "string",
        "Traffic_Control": "string",
    }
    df = pd.read_csv(path, dtype=dtype)
    return df


def read_roads(path: str) -> pd.DataFrame:
    """Read road centreline dataset (shapefile/geojson/fgdb/CSV with geometry)."""
    return gpd.read_file(path)


def read_construction(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def read_midblock_volumes(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def read_transit_stops(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

__all__ = [
    "read_collisions",
    "read_roads",
    "read_construction",
    "read_midblock_volumes",
    "read_transit_stops",
]
