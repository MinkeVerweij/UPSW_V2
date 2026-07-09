"""
Sidewalk usable-width computation.

Uses BGT voetpad polygons (fetched from PDOK) and obstacle footprints
(convex hulls of LiDAR clusters) to compute the remaining passable width
at regular intervals along each sidewalk segment.
"""

import logging
from pathlib import Path
from io import StringIO

import numpy as np
import pandas as pd
import requests
import geopandas as gpd
from shapely.geometry import LineString, Point, MultiLineString
from shapely.ops import unary_union, split, nearest_points

logger = logging.getLogger(__name__)

_PDOK_BGT_URL = (
    "https://api.pdok.nl/lv/bgt/ogc/v1_0/collections/wegdeel/items"
)

# BGT bgt-functie values that represent walkable footpaths
VOETPAD_TYPES = {"voetpad", "voetpad op rijbaan", "trottoir"}

# UPSW label codes that represent permanent, space-occupying obstacles
# (exclude ground, road, building, noise, cables)
OBSTACLE_LABELS = {30, 39, 40, 44, 45, 46, 47, 60, 61, 62, 65, 80, 81, 83,
                   85, 88, 89, 90, 91}


# ---------------------------------------------------------------------------
# Voetpad fetching from PDOK
# ---------------------------------------------------------------------------

def fetch_voetpad_polygons(bbox_rd, page_size=200):
    """
    Fetch BGT voetpad polygons from PDOK OGC API for a given bounding box.

    Parameters
    ----------
    bbox_rd : (xmin, ymin, xmax, ymax) in EPSG:28992
    page_size : int
        Items per API page.  PDOK does not support offset-based pagination;
        pages are followed via the ``next`` link in each response.

    Returns
    -------
    GeoDataFrame with columns: geometry (Polygon/MultiPolygon), bgt_functie
        CRS: EPSG:28992
    """
    xmin, ymin, xmax, ymax = bbox_rd
    features = []

    # First request — PDOK OGC API does not accept an `offset` parameter;
    # subsequent pages are reached via the `next` link in the response.
    next_url = _PDOK_BGT_URL
    params = {
        "f": "json",
        "bbox": f"{xmin},{ymin},{xmax},{ymax}",
        "bbox-crs": "http://www.opengis.net/def/crs/EPSG/0/28992",
        "crs": "http://www.opengis.net/def/crs/EPSG/0/28992",
        "limit": page_size,
    }

    while next_url:
        if next_url == _PDOK_BGT_URL:
            resp = requests.get(next_url, params=params, timeout=30)
        else:
            resp = requests.get(next_url, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        batch = data.get("features", [])
        features.extend(batch)

        # Follow the `next` link if present
        next_url = next(
            (lnk["href"] for lnk in data.get("links", []) if lnk.get("rel") == "next"),
            None,
        )

    if not features:
        logger.warning("No BGT wegdeel features returned for bbox.")
        return gpd.GeoDataFrame(columns=["geometry", "bgt_functie"], crs="EPSG:28992")

    gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:28992")
    bgt_col = next((c for c in gdf.columns if "functie" in c.lower()), None)
    if bgt_col:
        gdf = gdf.rename(columns={bgt_col: "bgt_functie"})
    else:
        gdf["bgt_functie"] = "unknown"

    voetpad_gdf = gdf[gdf["bgt_functie"].str.lower().isin(VOETPAD_TYPES)].copy()
    voetpad_gdf = voetpad_gdf[["geometry", "bgt_functie"]].reset_index(drop=True)
    logger.info(f"Fetched {len(voetpad_gdf)} voetpad polygons.")
    return voetpad_gdf


# ---------------------------------------------------------------------------
# Obstacle footprints from cluster inventory
# ---------------------------------------------------------------------------

def obstacle_footprints_from_inventory(cluster_inventory_df, clusters_dir,
                                       buffer_m=0.15):
    """
    Build obstacle footprint polygons from cluster inventory NPZ files.

    Uses the centroid + area_m2 from the inventory to create circular
    footprints (no need to load full NPZ unless higher accuracy is needed).

    Parameters
    ----------
    cluster_inventory_df : pd.DataFrame
        Must have columns: centroid_x, centroid_y, area_m2, cluster_id,
        and either final_label or auto_label.
    clusters_dir : Path
        Root of the cluster NPZ directory.
    buffer_m : float
        Extra buffer around each footprint (metres).

    Returns
    -------
    GeoDataFrame with columns: cluster_id, label, geometry (Polygon)
        CRS: EPSG:28992
    """
    inv = cluster_inventory_df.copy()
    label_col = "final_label" if "final_label" in inv.columns else "auto_label"
    inv = inv[inv[label_col].isin(OBSTACLE_LABELS)].reset_index(drop=True)

    rows = []
    for _, row in inv.iterrows():
        cx, cy = row["centroid_x"], row["centroid_y"]
        r = max(np.sqrt(row.get("area_m2", 0.25) / np.pi), 0.2) + buffer_m
        footprint = Point(cx, cy).buffer(r, resolution=8)
        rows.append({
            "cluster_id": row.get("cluster_id", ""),
            "label": row[label_col],
            "geometry": footprint,
        })

    if not rows:
        return gpd.GeoDataFrame(columns=["cluster_id", "label", "geometry"],
                                crs="EPSG:28992")
    return gpd.GeoDataFrame(rows, crs="EPSG:28992")


# ---------------------------------------------------------------------------
# Width computation
# ---------------------------------------------------------------------------

def _sidewalk_centerline(voetpad_polygon):
    """
    Approximate the centreline of a voetpad polygon as the medial axis midline.

    Uses a simplified approach: the polygon's longest axis via skeleton
    approximation from the boundary midpoints at equal-spaced angles.
    Falls back to the polygon's centroid spine if the polygon is near-rectangular.
    """
    # Simple approach: use the boundary of the polygon's skeleton via
    # Shapely's representative point along the polygon's elongation direction.
    # For narrow elongated polygons (typical sidewalk shape), we approximate
    # the centreline by sampling the interior at regular spacing.
    try:
        from shapely.ops import polylabel
    except ImportError:
        pass

    # Compute approximate major axis from PCA of boundary coords
    coords = np.array(voetpad_polygon.exterior.coords)
    centroid = coords.mean(axis=0)
    cov = np.cov((coords - centroid).T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    major_axis = eigvecs[:, np.argmax(eigvals)]

    # Project polygon extent along major axis to get line endpoints
    proj = (coords - centroid) @ major_axis
    p_min = centroid + major_axis * proj.min()
    p_max = centroid + major_axis * proj.max()

    # Shrink slightly to stay inside polygon
    dx = p_max - p_min
    p_min = p_min + dx * 0.05
    p_max = p_max - dx * 0.05

    line = LineString([p_min, p_max])
    # Clip to voetpad polygon
    clipped = line.intersection(voetpad_polygon)
    if clipped.is_empty:
        return line
    if isinstance(clipped, MultiLineString):
        # Pick the longest segment
        clipped = max(clipped.geoms, key=lambda g: g.length)
    return clipped


def _cross_section_width(voetpad_polygon, obstacles_union, point, direction,
                         max_width=20.0):
    """
    Compute total and usable cross-section width at a point along the centreline.

    Casts a perpendicular line across the voetpad polygon, measures total width,
    then subtracts obstacles intersecting the cross-section.
    """
    perp = np.array([-direction[1], direction[0]])
    perp /= np.linalg.norm(perp)

    # Cross-section line across the full voetpad (max_width on each side)
    p = np.array([point.x, point.y])
    p1 = Point(p - perp * max_width)
    p2 = Point(p + perp * max_width)
    cross_line = LineString([p1, p2])

    section = cross_line.intersection(voetpad_polygon)
    if section.is_empty:
        return None, None

    if isinstance(section, MultiLineString):
        width_total = sum(seg.length for seg in section.geoms)
    else:
        width_total = section.length

    # Subtract obstacle width
    if obstacles_union is None or obstacles_union.is_empty:
        return width_total, width_total

    obs_on_section = cross_line.intersection(obstacles_union)
    if obs_on_section.is_empty:
        obs_width = 0.0
    elif obs_on_section.geom_type in ("LineString", "MultiLineString"):
        if isinstance(obs_on_section, MultiLineString):
            obs_width = sum(seg.length for seg in obs_on_section.geoms)
        else:
            obs_width = obs_on_section.length
    else:
        obs_width = 0.0

    width_usable = max(width_total - obs_width, 0.0)
    return width_total, width_usable


def compute_usable_width(voetpad_gdf, obstacles_gdf, step_m=0.5):
    """
    Compute remaining usable sidewalk width at regular intervals.

    Parameters
    ----------
    voetpad_gdf : GeoDataFrame
        BGT voetpad polygons (EPSG:28992).
    obstacles_gdf : GeoDataFrame
        Obstacle footprints (EPSG:28992).
    step_m : float
        Interval between cross-section measurements in metres.

    Returns
    -------
    GeoDataFrame with columns:
        geometry (LineString segment between measurement points),
        width_total_m, width_usable_m, n_obstacles
        CRS: EPSG:28992
    """
    if obstacles_gdf is not None and len(obstacles_gdf) > 0:
        obstacles_union = unary_union(obstacles_gdf.geometry)
    else:
        obstacles_union = None

    rows = []
    for _, voetpad in voetpad_gdf.iterrows():
        poly = voetpad.geometry
        if poly is None or poly.is_empty:
            continue

        centreline = _sidewalk_centerline(poly)
        total_length = centreline.length
        if total_length < step_m:
            continue

        n_steps = max(int(total_length / step_m), 1)
        distances = np.linspace(0, total_length, n_steps + 1)

        prev_point = None
        prev_width_total = None
        prev_width_usable = None

        for dist in distances:
            pt = centreline.interpolate(dist)

            # Tangent direction at this point
            eps = min(0.1, total_length * 0.01)
            pt_ahead = centreline.interpolate(min(dist + eps, total_length))
            direction = np.array([pt_ahead.x - pt.x, pt_ahead.y - pt.y])
            if np.linalg.norm(direction) < 1e-9:
                continue
            direction /= np.linalg.norm(direction)

            width_total, width_usable = _cross_section_width(
                poly, obstacles_union, pt, direction
            )

            # Count obstacles within 2*step_m of this point
            if obstacles_gdf is not None and len(obstacles_gdf) > 0:
                dists_to_obs = obstacles_gdf.geometry.distance(pt)
                n_obs = int((dists_to_obs < step_m * 2).sum())
            else:
                n_obs = 0

            if prev_point is not None and width_total is not None:
                seg = LineString([prev_point, pt])
                avg_total = (prev_width_total + width_total) / 2
                avg_usable = (prev_width_usable + width_usable) / 2
                rows.append({
                    "geometry": seg,
                    "width_total_m": round(avg_total, 3),
                    "width_usable_m": round(avg_usable, 3),
                    "n_obstacles": n_obs,
                })

            prev_point = pt
            prev_width_total = width_total
            prev_width_usable = width_usable

    if not rows:
        return gpd.GeoDataFrame(
            columns=["geometry", "width_total_m", "width_usable_m", "n_obstacles"],
            crs="EPSG:28992",
        )
    return gpd.GeoDataFrame(rows, crs="EPSG:28992")


def width_to_geojson(width_gdf, out_path):
    """Write width GeoDataFrame to GeoJSON (EPSG:4326 for display compatibility)."""
    out = width_gdf.to_crs("EPSG:4326")
    out.to_file(str(out_path), driver="GeoJSON")
    logger.info(f"Saved sidewalk width to {out_path}")


def width_category(w):
    """Classify usable width (metres) into accessibility category."""
    if w is None or np.isnan(w):
        return "unknown"
    if w >= 1.8:
        return "good"      # ≥1.8m: two wheelchairs can pass
    if w >= 1.2:
        return "adequate"  # ≥1.2m: single wheelchair comfortable
    if w >= 0.9:
        return "narrow"    # ≥0.9m: single wheelchair just fits
    return "blocked"       # <0.9m: inaccessible


# ---------------------------------------------------------------------------
# OSM sidewalk network
# ---------------------------------------------------------------------------

_OVERPASS_INTERPRETER_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]

_OVERPASS_HEADERS = {
    "User-Agent": "UPSW-SidewalkWidth/1.0 (research; contact: see project README)",
    "Accept": "*/*",
}


def _overpass_raw(query, timeout=25):
    """
    Execute an Overpass QL query via direct requests, trying multiple mirrors.

    Returns the parsed JSON dict on success, raises ConnectionError if all
    endpoints fail.
    """
    import time

    last_exc = None
    for url in _OVERPASS_INTERPRETER_URLS:
        for attempt in range(2):  # one retry per endpoint on 429
            try:
                resp = requests.post(
                    url,
                    data={"data": query},
                    headers=_OVERPASS_HEADERS,
                    timeout=timeout,
                )
                if resp.status_code == 200:
                    logger.info(f"Overpass OK: {url}")
                    return resp.json()
                if resp.status_code == 429:
                    logger.warning(f"429 rate-limited by {url}, waiting 10 s …")
                    time.sleep(10)
                    continue
                logger.warning(f"{url} returned {resp.status_code}")
                break  # non-200, non-429 → try next mirror
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout ({timeout} s) on {url}")
                last_exc = TimeoutError(f"{url} timed out after {timeout} s")
                break
            except Exception as exc:
                logger.warning(f"{url} error: {exc}")
                last_exc = exc
                break

    raise ConnectionError(
        "All Overpass mirrors failed.\n"
        "Possible causes: corporate firewall, VPN, or server overload.\n"
        "Workarounds: (1) try again later, (2) use a VPN, (3) call "
        "fetch_osm_sidewalks() from a different network."
    ) from last_exc


def _overpass_json_to_gdf(data, target_crs="EPSG:28992"):
    """Convert a raw Overpass JSON response (nodes + ways) to a LineString GDF."""
    from shapely.geometry import LineString
    from pyproj import Transformer
    _wgs2rd = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)

    elements = data.get("elements", [])
    nodes = {
        e["id"]: _wgs2rd.transform(e["lon"], e["lat"])
        for e in elements
        if e["type"] == "node" and "lon" in e and "lat" in e
    }

    rows = []
    for e in elements:
        if e["type"] != "way":
            continue
        coords = [nodes[nid] for nid in e.get("nodes", []) if nid in nodes]
        if len(coords) < 2:
            continue
        tags = e.get("tags", {})
        rows.append({
            "osmid":   e["id"],
            "name":    tags.get("name"),
            "highway": tags.get("highway"),
            "surface": tags.get("surface"),
            "width":   tags.get("width"),
            "geometry": LineString(coords),
        })

    if not rows:
        return gpd.GeoDataFrame(
            columns=["osmid", "name", "highway", "surface", "width", "geometry"],
            crs=target_crs,
        )
    return gpd.GeoDataFrame(rows, crs=target_crs)


def fetch_osm_sidewalks(bbox_rd, crs="EPSG:28992"):
    """
    Download OSM pedestrian edges for a bounding box.

    Fetches ``highway=footway|pedestrian|path|living_street`` ways via direct
    Overpass API calls (bypasses osmnx's rate-limit retry logic).  Tries
    multiple public mirrors; fails fast (25 s per endpoint) and cycles to the
    next one.

    Parameters
    ----------
    bbox_rd : (xmin, ymin, xmax, ymax) in EPSG:28992
    crs : str
        Output CRS (default EPSG:28992).

    Returns
    -------
    GeoDataFrame of LineString edges with columns:
        osmid, name, highway, surface, width (OSM tag, may be NaN),
        geometry
    """
    from pyproj import Transformer
    _rd2wgs = Transformer.from_crs("EPSG:28992", "EPSG:4326", always_xy=True)
    xmin, ymin, xmax, ymax = bbox_rd
    lon_w, lat_s = _rd2wgs.transform(xmin, ymin)
    lon_e, lat_n = _rd2wgs.transform(xmax, ymax)

    # Overpass bbox syntax: (south,west,north,east)
    bb = f"{lat_s:.6f},{lon_w:.6f},{lat_n:.6f},{lon_e:.6f}"

    query = f"""
[out:json][timeout:25];
(
  way["highway"~"footway|pedestrian|path|living_street"]({bb});
  way["sidewalk"~"left|right|both"]["highway"]({bb});
  >;
);
out;
"""
    data = _overpass_raw(query, timeout=30)
    edges = _overpass_json_to_gdf(data, target_crs=crs)

    edges = edges.drop_duplicates(subset=["osmid"]).reset_index(drop=True)
    logger.info(f"Fetched {len(edges)} OSM sidewalk edges.")
    return edges


def compute_width_on_osm_network(
    osm_edges_gdf,
    voetpad_gdf,
    obstacles_gdf,
    step_m=1.0,
    default_half_width_m=1.2,
):
    """
    Compute usable sidewalk width for each OSM edge.

    For each edge the function:
    1. Looks for an overlapping BGT voetpad polygon. If found, cross-sections
       are clipped to that polygon boundary (accurate total width).
    2. If no BGT polygon is found, a synthetic polygon is created by buffering
       the edge by ``default_half_width_m`` on each side (estimated total width).
    3. Obstacle footprints are subtracted from the perpendicular cross-section
       using the same logic as ``compute_usable_width``.

    Parameters
    ----------
    osm_edges_gdf : GeoDataFrame
        OSM LineString edges (EPSG:28992), from :func:`fetch_osm_sidewalks`.
    voetpad_gdf : GeoDataFrame
        BGT voetpad polygons (EPSG:28992).
    obstacles_gdf : GeoDataFrame
        Obstacle footprints (EPSG:28992).
    step_m : float
        Sampling interval along each edge (metres).
    default_half_width_m : float
        Half-width used when no BGT voetpad polygon is found (metres).
        A value of 1.2 gives a 2.4 m total, typical for Amsterdam sidewalks.

    Returns
    -------
    GeoDataFrame
        One row per OSM edge with added columns:
        ``width_total_m`` (mean), ``width_usable_m`` (mean),
        ``width_usable_min_m`` (worst cross-section),
        ``category`` (worst cross-section category),
        ``bgt_matched`` (bool — True if a BGT polygon was used).
    """
    if obstacles_gdf is not None and len(obstacles_gdf) > 0:
        obs_union = unary_union(obstacles_gdf.geometry)
    else:
        obs_union = None

    vp_sindex = voetpad_gdf.sindex if len(voetpad_gdf) > 0 else None

    rows = []
    for edge_i, edge_row in osm_edges_gdf.iterrows():
        line = edge_row.geometry
        if line is None or line.is_empty or line.length < step_m:
            continue

        # Find overlapping BGT voetpad polygon
        poly = None
        bgt_matched = False
        if vp_sindex is not None:
            cands = list(vp_sindex.intersection(line.bounds))
            for ci in cands:
                vp = voetpad_gdf.iloc[ci].geometry
                if vp.intersects(line.buffer(0.5)):
                    poly = vp
                    bgt_matched = True
                    break

        if poly is None:
            poly = line.buffer(default_half_width_m, cap_style=2)

        # Sample cross-sections along the line
        total_length = line.length
        n_steps = max(int(total_length / step_m), 1)
        distances = np.linspace(0, total_length, n_steps + 1)

        widths_total, widths_usable = [], []
        for dist in distances:
            pt = line.interpolate(dist)
            eps = min(0.1, total_length * 0.01)
            pt_ahead = line.interpolate(min(dist + eps, total_length))
            direction = np.array([pt_ahead.x - pt.x, pt_ahead.y - pt.y])
            norm = np.linalg.norm(direction)
            if norm < 1e-9:
                continue
            direction /= norm
            wt, wu = _cross_section_width(poly, obs_union, pt, direction)
            if wt is not None:
                widths_total.append(wt)
                widths_usable.append(wu)

        if not widths_usable:
            continue

        mean_total  = float(np.mean(widths_total))
        mean_usable = float(np.mean(widths_usable))
        min_usable  = float(np.min(widths_usable))

        row = dict(edge_row)
        row["width_total_m"]    = round(mean_total, 3)
        row["width_usable_m"]   = round(mean_usable, 3)
        row["width_usable_min_m"] = round(min_usable, 3)
        row["category"]         = width_category(min_usable)
        row["bgt_matched"]      = bgt_matched
        rows.append(row)

    if not rows:
        return gpd.GeoDataFrame(crs="EPSG:28992")

    result = gpd.GeoDataFrame(rows, crs="EPSG:28992")
    logger.info(
        f"Width computed for {len(result)}/{len(osm_edges_gdf)} OSM edges."
    )
    return result
