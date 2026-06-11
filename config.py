"""
Device-specific path configuration.

Edit the block matching your device's hostname to point at the correct data
directories. All notebooks import from here instead of hardcoding paths.
"""
import re
import socket
from pathlib import Path


def tilecodes_in_dir(directory, pattern="*.laz"):
    """Return sorted list of tilecodes found in *directory*.

    Scans for files matching *pattern* and extracts the tilecode
    (``XXXXXX_YYYYYY``) from each filename.  Files whose names don't
    contain a tilecode are silently skipped.
    """
    tc_re = re.compile(r'(\d{6}_\d{6})')
    codes = sorted({
        m.group(1)
        for p in Path(directory).rglob(pattern)
        for m in [tc_re.search(p.stem)]
        if m
    })
    return codes

HOSTNAME = socket.gethostname()

# ── External device ────────────────────────────────────────────────────────────
# Update hostname and paths to match the external device's setup.
if HOSTNAME == "external-device":
    TILE_DIR     = Path("/mnt/storage/pointclouds/amsterdam")
    AHN_RAW_DIR  = Path("/mnt/storage/ahn/raw")
    AHN_NPZ_DIR  = Path("/mnt/storage/ahn/npz")
    AHN_GRID_SHP = Path("/mnt/storage/ahn/ahn_units_shapefile/AHN_subunits_GeoTiles.shp")
    BGT_DIR      = Path("/mnt/storage/bgt")
    BOMEN_DIR    = Path("/mnt/storage/bomen")
    AFVAL_DIR    = Path("/mnt/storage/afvalbakken")
    OSM_DIR      = Path("/mnt/storage/osm")
    TERRAS_DIR   = Path("/mnt/storage/terras")
    BBOX_DIR     = Path("/mnt/storage/bbox_polygons")
    LABELED_DIR  = Path("/mnt/storage/labeled_pointcloud")
    CLUSTERS_DIR = Path("/mnt/storage/clusters")
    DATASET_DIR  = Path("/mnt/storage/dataset")
    MODELS_DIR   = Path("/mnt/storage/models")

# ── Local Mac (default) ────────────────────────────────────────────────────────
else:
    TILE_DIR     = Path("data/input/pointcloud/raw")
    AHN_RAW_DIR  = Path("data/input/ahn")
    AHN_NPZ_DIR  = Path("data/output/ahn")
    AHN_GRID_SHP = Path("data/input/ahn/ahn_units_shapefile/AHN_subunits_GeoTiles.shp")
    BGT_DIR      = Path("data/input/bgt")
    BOMEN_DIR    = Path("data/input/bomen")
    AFVAL_DIR    = Path("data/input/afvalbakken")
    OSM_DIR      = Path("data/input/osm")
    TERRAS_DIR   = Path("data/input/terras")
    BBOX_DIR     = Path("data/input/bbox_polygons")
    LABELED_DIR  = Path("data/output/labeled_pointcloud")
    CLUSTERS_DIR = Path("data/output/clusters")
    DATASET_DIR  = Path("data/output/dataset")
    MODELS_DIR   = Path("data/models")

# ── A10 ring bounding box (RD New / EPSG:28992) ────────────────────────────────
# Used by data setup and tile discovery notebooks.
A10_BBOX_RD = (117000, 482000, 126000, 492000)  # xmin, ymin, xmax, ymax

# ── Tilecodes to set up data for (notebook 0.5) ───────────────────────────────
# BGT road layers and AHN data are downloaded only for these tiles.
# Update this list before running notebook 0.5 on each device.
SETUP_TILECODES = [
    # Already found
    "122700_487700",
    "122700_485700",
    "119500_488900",
    "119900_490100",
    "120300_490100",
    "121100_484500",
    "121900_484900",
    "121900_485300",
    "123100_487700",
    "123500_487700",
    "122300_486900",
    # Expanding from 119500_488900
    "119100_488900",
    "119100_488500",
    "119500_488500",
    # Expanding from 119900/120300_490100
    "119500_490100",
    "120700_490100",
    "121100_490100",
    "119900_490500",
    "120300_490500",
    "120700_490500",
    # Expanding from 121100_484500
    "121100_484100",
    "120700_484500",
    # Expanding from 121900_484900/485300
    "121900_484500",
    "121900_484100",
    "122300_484500",
    "122300_485700",
    # Expanding from 123100/123500_487700
    "123100_488100",
    "123500_488100",
    "123900_487700",
    "124300_487700",
    "123100_487300",
    "123500_487300",
    "123900_488100",
    # Expanding from 122300_486900
    "121500_486900",
    "120700_486900",
    "119900_486900",
    "123100_486500",
    "123100_486100",
    # New unexplored
    "123900_485700",
    "118700_487700",
    "118300_487300",
]
