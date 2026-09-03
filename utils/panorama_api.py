"""Amsterdam panorama API client."""

import json
import math
import logging
from pathlib import Path

import numpy as np
import requests
from PIL import Image
from io import BytesIO
from pyproj import Transformer
from shapely.geometry import Point, Polygon

logger = logging.getLogger(__name__)

_WGS84_TO_RD = Transformer.from_crs("EPSG:4326", "EPSG:28992", always_xy=True)
_RD_TO_WGS84 = Transformer.from_crs("EPSG:28992", "EPSG:4326", always_xy=True)

_BASE_URL = "https://api.data.amsterdam.nl/panorama/panoramas/"


def rd_to_wgs84(x, y):
    """Convert RD New (EPSG:28992) x, y to WGS84 lon, lat."""
    return _RD_TO_WGS84.transform(x, y)


def wgs84_to_rd(lon, lat):
    """Convert WGS84 lon, lat to RD New (EPSG:28992) x, y."""
    return _WGS84_TO_RD.transform(lon, lat)


def fetch_panoramas_near(lon, lat, radius_m=50, max_results=5, mission_year=None):
    """
    Fetch panorama metadata near a WGS84 coordinate.

    Parameters
    ----------
    lon, lat : float
        WGS84 longitude / latitude.
    radius_m : float
        Search radius in metres.
    max_results : int
        Maximum number of panoramas to return.
    mission_year : int or None
        If given, restrict results to this mission year (e.g. 2024).
        Uses the ``tags=mission-<year>`` API filter (the ``mission_year``
        query parameter is silently ignored by the API).

    Returns
    -------
    list of dict
        Raw panorama dicts from the API, ordered by distance.
    """
    params = {
        "near": f"{lon},{lat}",
        "radius": radius_m,
        "ordering": "_distance",
        "limit": max_results,
    }
    if mission_year is not None:
        params["tags"] = f"mission-{mission_year}"

    # The API silently caps each response to a server-side page size (observed
    # 25) regardless of the requested `limit` — a tile with two overlapping
    # driving passes can have 100+ panoramas within radius_m, and without
    # following `_links.next` only the closest-to-`(lon,lat)` page ever comes
    # back, silently starving coverage on the far side of the tile (confirmed:
    # a viaduct-side streetlight whose nearest actual panorama was 4.9m away
    # got matched to a cached one 72m away, because an entire second capture
    # date never made it past page 1).
    panoramas = []
    url, req_params = _BASE_URL, params
    while url and len(panoramas) < max_results:
        resp = requests.get(url, params=req_params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        panoramas.extend(data.get("_embedded", {}).get("panoramas", []))
        next_link = data.get("_links", {}).get("next")
        url = next_link.get("href") if next_link else None
        req_params = None  # the next href already carries all query params
    return panoramas[:max_results]


def _infer_heading_from_neighbors(pano_id, lon, lat):
    """
    Infer vehicle heading for panoramas with heading=0.0 by fetching the two
    adjacent panoramas from the same recording session and computing the
    direction of travel from their position difference.

    Returns heading in degrees (0=N, 90=E), or 0.0 if inference fails.
    """
    try:
        resp = requests.get(
            f"{_BASE_URL}{pano_id}/adjacencies/", timeout=10
        )
        resp.raise_for_status()
        adj = resp.json().get("_embedded", {}).get("adjacencies", [])
        # Filter for same-session neighbors (same pano_id prefix up to last segment)
        session = "_".join(pano_id.rsplit("_", 1)[:-1])
        same_session = [
            a for a in adj
            if a.get("pano_id", "").startswith(session)
            and a.get("pano_id") != pano_id
        ]
        if len(same_session) < 1:
            return 0.0
        # Use the first neighbor to compute direction
        neighbor = same_session[0]
        n_coords = neighbor.get("geometry", {}).get("coordinates", [])
        if len(n_coords) < 2:
            return 0.0
        n_lon, n_lat = n_coords[0], n_coords[1]
        # Direction from neighbor → current position gives travel direction
        dx = (lon - n_lon) * math.cos(math.radians(lat)) * 111320
        dy = (lat - n_lat) * 111320
        heading = math.degrees(math.atan2(dx, dy)) % 360.0
        return heading
    except Exception:
        return 0.0


def fetch_nearest_panorama(x_rd, y_rd, radius_m=100, mission_year=2024):
    """
    Return metadata dict for the nearest panorama to an RD New coordinate.

    Falls back to any year if no panorama found for *mission_year*.
    For 2024 panoramas that report heading=0.0, the heading is inferred
    from adjacent frame positions.
    """
    lon, lat = rd_to_wgs84(x_rd, y_rd)
    panoramas = fetch_panoramas_near(lon, lat, radius_m=radius_m,
                                     max_results=1, mission_year=mission_year)
    if not panoramas:
        logger.warning(f"No {mission_year} panorama within {radius_m}m — retrying without year filter.")
        panoramas = fetch_panoramas_near(lon, lat, radius_m=radius_m, max_results=1)
    if not panoramas:
        raise ValueError(f"No panorama found within {radius_m}m of ({x_rd}, {y_rd}).")

    pano = panoramas[0]

    # 2024 panoramas store heading=0.0; infer from neighbors
    if pano.get("heading", 0.0) == 0.0 and pano.get("mission_year") == "2024":
        coords = pano["geometry"]["coordinates"]
        pano = dict(pano)  # don't mutate the original
        pano["heading"] = _infer_heading_from_neighbors(
            pano["pano_id"], coords[0], coords[1]
        )
        logger.info(f"Inferred heading {pano['heading']:.1f}° for {pano['pano_id']}")

    return pano


def parse_pose(pano_dict):
    """
    Extract camera pose from a panorama API dict.

    Returns
    -------
    dict with keys:
        x_rd, y_rd : float  — camera position in EPSG:28992
        z_wgs84    : float  — ellipsoidal height (metres, WGS84)
        heading    : float  — compass bearing in degrees (0=N, 90=E)
        pitch      : float  — tilt in degrees (positive = up)
        roll       : float  — roll in degrees
        pano_id    : str
        timestamp  : str
        image_url_medium : str
        image_url_full   : str
    """
    coords = pano_dict["geometry"]["coordinates"]  # [lon, lat, elev]
    lon, lat, z = coords[0], coords[1], coords[2]
    x_rd, y_rd = wgs84_to_rd(lon, lat)

    links = pano_dict.get("_links", {})
    return {
        "pano_id": pano_dict["pano_id"],
        "timestamp": pano_dict.get("timestamp", ""),
        "x_rd": x_rd,
        "y_rd": y_rd,
        "z_wgs84": z,
        "heading": pano_dict["heading"],
        "pitch": pano_dict["pitch"],
        "roll": pano_dict.get("roll", 0.0),
        "image_url_medium": links.get("equirectangular_medium", {}).get("href", ""),
        "image_url_full": links.get("equirectangular_full", {}).get("href", ""),
        "mission_year": pano_dict.get("mission_year", ""),
    }


def download_panorama_image(pose_or_url, size="medium", cache_dir=None):
    """
    Download and return a PIL Image for the panorama.

    Parameters
    ----------
    pose_or_url : dict or str
        Either a pose dict (from :func:`parse_pose`) or a direct image URL string.
    size : {"small", "medium", "full"}
        Image resolution to fetch (ignored when pose_or_url is a URL string).
    cache_dir : Path or None
        If given, cache downloaded images here as ``<pano_id>.jpg``.

    Returns
    -------
    PIL.Image.Image
    """
    if isinstance(pose_or_url, str):
        url = pose_or_url
        pano_id = Path(url).stem
    else:
        key = f"image_url_{size}"
        url = pose_or_url[key]
        pano_id = pose_or_url["pano_id"]

    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{pano_id}.jpg"
        if cache_path.exists():
            logger.debug(f"Cache hit: {cache_path}")
            return Image.open(cache_path)

    logger.info(f"Downloading panorama: {url}")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    img = Image.open(BytesIO(resp.content)).convert("RGB")

    if cache_dir is not None:
        img.save(cache_path, "JPEG", quality=90)
        logger.debug(f"Cached to {cache_path}")

    return img


def tile_centroid_wgs84(tilecode):
    """
    Return (lon, lat) centroid of an XXXXXX_YYYYYY tilecode (EPSG:28992, 50m tiles).
    """
    parts = tilecode.split("_")
    x_min, y_min = float(parts[0]), float(parts[1])
    return rd_to_wgs84(x_min + 25.0, y_min + 25.0)


def _tile_polygon(tilecode, bbox_dir):
    """Return a Shapely Polygon (EPSG:28992) for the tile's point cloud footprint."""
    bbox_path = Path(bbox_dir) / f"bbox_{tilecode}.geojson"
    if bbox_path.exists():
        with open(bbox_path) as f:
            gj = json.load(f)
        coords = gj["features"][0]["geometry"]["coordinates"][0]
        return Polygon(coords)
    # Fallback: regular 50×50 m box
    parts = tilecode.split("_")
    x0, y0 = float(parts[0]), float(parts[1])
    return Polygon([(x0, y0), (x0+50, y0), (x0+50, y0+50), (x0, y0+50)])


def _infer_headings_from_list(pano_dicts):
    """
    For panoramas with heading=0.0 (2024 series), infer travel direction from
    sequential neighbours already present in *pano_dicts*.

    Mutates each dict in-place; returns the same list.
    """
    positions = {p["pano_id"]: p["geometry"]["coordinates"][:2]
                 for p in pano_dicts}

    for p in pano_dicts:
        if p.get("heading", 0.0) != 0.0:
            continue
        pano_id = p["pano_id"]
        prefix, _, num_str = pano_id.rpartition("_")
        try:
            num = int(num_str)
        except ValueError:
            continue

        p_lon, p_lat = p["geometry"]["coordinates"][:2]
        for delta in (1, -1):
            neighbour_id = f"{prefix}_{num + delta:05d}"
            if neighbour_id not in positions:
                continue
            n_lon, n_lat = positions[neighbour_id]
            # Direction of travel: from this frame toward the next (delta=+1)
            # or from the previous frame toward this one (delta=-1)
            if delta == 1:
                dx = (n_lon - p_lon) * math.cos(math.radians(p_lat)) * 111_320
                dy = (n_lat - p_lat) * 111_320
            else:
                dx = (p_lon - n_lon) * math.cos(math.radians(p_lat)) * 111_320
                dy = (p_lat - n_lat) * 111_320
            p["heading"] = math.degrees(math.atan2(dx, dy)) % 360.0
            break

    return pano_dicts


def _thin_panoramas(pano_dicts, min_spacing_m):
    """
    Greedy spatial thinning: keep only panoramas that are at least
    *min_spacing_m* metres from every already-kept panorama.

    Input panoramas should be ordered by distance from the tile centroid
    so the most central one is always kept first.
    """
    kept = []
    kept_xy = []
    for p in pano_dicts:
        lon, lat = p["geometry"]["coordinates"][:2]
        x_rd, y_rd = wgs84_to_rd(lon, lat)
        if not kept_xy or all(
            math.hypot(x_rd - kx, y_rd - ky) >= min_spacing_m
            for kx, ky in kept_xy
        ):
            kept.append(p)
            kept_xy.append((x_rd, y_rd))
    return kept


def fetch_panoramas_in_tile(tilecode, bbox_dir, mission_year=2024,
                             min_spacing_m=15.0, buffer_m=0.0):
    """
    Fetch all panoramas whose camera position falls inside the tile's
    LiDAR point cloud footprint (optionally expanded by *buffer_m*), then
    thin them spatially.

    Parameters
    ----------
    tilecode : str
        e.g. ``"120300_489300"``
    bbox_dir : str or Path
        Directory containing ``bbox_<tilecode>.geojson`` files.
    mission_year : int or None
        Year filter (uses ``tags=mission-<year>``).  Pass None for any year.
    min_spacing_m : float
        Minimum distance between returned panoramas (metres).
    buffer_m : float
        Grow the tile polygon by this many metres before testing camera
        positions against it. Panoramas are 360° equirectangular images, so
        a camera just outside the tile boundary can still see everything
        inside it — restricting to strictly-inside cameras (buffer_m=0, the
        historical default) starves multi-camera triangulation of viewing
        angles, since most tiles then only contain a handful of cameras
        clustered along whatever short street segment crosses the tile.

    Returns
    -------
    list of pose dicts (same format as :func:`parse_pose`), each with
    ``"tilecode"`` added.
    """
    tile_poly = _tile_polygon(tilecode, bbox_dir)
    search_poly = tile_poly.buffer(buffer_m) if buffer_m else tile_poly
    centroid = tile_poly.centroid
    cx_rd, cy_rd = centroid.x, centroid.y

    # Circumradius: large enough to cover the whole (buffered) search area
    # from the tile's centroid.
    verts = np.array(search_poly.exterior.coords)
    radius_m = float(np.hypot(verts[:, 0] - cx_rd,
                               verts[:, 1] - cy_rd).max()) + 5

    c_lon, c_lat = rd_to_wgs84(cx_rd, cy_rd)

    panos = fetch_panoramas_near(c_lon, c_lat, radius_m=radius_m,
                                  max_results=500, mission_year=mission_year)
    if not panos and mission_year is not None:
        logger.warning(f"{tilecode}: no {mission_year} panoramas — retrying without year filter.")
        panos = fetch_panoramas_near(c_lon, c_lat, radius_m=radius_m, max_results=500)

    # Keep only those whose camera position is inside the (buffered) tile polygon
    inside = []
    for p in panos:
        lon, lat = p["geometry"]["coordinates"][:2]
        x_rd, y_rd = wgs84_to_rd(lon, lat)
        if search_poly.contains(Point(x_rd, y_rd)):
            inside.append(p)

    if not inside:
        logger.warning(f"{tilecode}: no panoramas inside tile polygon.")
        return []

    # Infer headings from sequential neighbours before thinning
    _infer_headings_from_list(inside)

    # Thin spatially — keep the most central panoramas first
    inside.sort(key=lambda p: math.hypot(
        wgs84_to_rd(*p["geometry"]["coordinates"][:2])[0] - cx_rd,
        wgs84_to_rd(*p["geometry"]["coordinates"][:2])[1] - cy_rd,
    ))
    thinned = _thin_panoramas(inside, min_spacing_m)

    poses = []
    for p in thinned:
        pose = parse_pose(p)
        pose["tilecode"] = tilecode
        poses.append(pose)

    logger.info(f"{tilecode}: {len(inside)} panoramas inside tile → "
                f"{len(thinned)} kept (spacing ≥{min_spacing_m}m).")
    return poses
