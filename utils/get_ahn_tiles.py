#!/usr/bin/env python3

"""
Download AHN tiles corresponding to pointcloud tiles.

Workflow
--------
1. Parse coordinates from LAZ filenames
2. Convert to GeoDataFrame
3. Spatially join with AHN grid
4. Download required AHN tiles asynchronously
"""

import os
import re
import asyncio
import aiohttp
import aiofiles
import geopandas as gpd

from pathlib import Path
from shapely.geometry import Point
from tqdm.asyncio import tqdm_asyncio


# --------------------------------------------------
# UTILITIES
# --------------------------------------------------

def filenames_to_gdf(data_dir, regex_pattern, crs):
    """
    Convert filenames containing coordinates into a GeoDataFrame.

    Example filename:
        123456_456789.laz
    """

    pattern = re.compile(regex_pattern)

    records = []

    for fname in os.listdir(data_dir):

        match = pattern.match(fname)

        if not match:
            continue

        x_str, y_str = match.groups()

        try:
            x = float(x_str)
            y = float(y_str)
        except ValueError:
            print(f"Skipping invalid filename: {fname}")
            continue

        records.append(
            {
                "filename": fname,
                "geometry": Point(x, y)
            }
        )

    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=crs)

    return gdf


def get_required_ahn_tiles(
    data_dir,
    ahn_grid_path,
    regex_pattern=r"^(\d+)_(\d+)\.laz$",
    crs="EPSG:28992"
):
    """
    Determine which AHN tiles intersect the input pointcloud tiles.
    """

    print("Reading AHN grid...")
    gdf_ahn = gpd.read_file(ahn_grid_path)

    print("Parsing filenames...")
    gdf_tiles = filenames_to_gdf(data_dir, regex_pattern, crs)

    print("Running spatial join...")
    joined = gpd.sjoin(gdf_tiles, gdf_ahn)

    ahn_tiles = joined["GT_AHNSUB"].unique().tolist()

    print(f"Found {len(ahn_tiles)} AHN tiles")

    return ahn_tiles


# --------------------------------------------------
# DOWNLOAD FUNCTIONS
# --------------------------------------------------

async def download_tile(
    session,
    semaphore,
    code,
    output_dir,
    base_url,
    chunk_size,
    retries
):

    url = base_url.format(code=code)

    filename = Path(output_dir) / f"{code}.LAZ"

    if filename.exists() and filename.stat().st_size > 10_000_000:
        print(f"Skipping {code} (already exists)")
        return

    for attempt in range(1, retries + 1):

        try:

            async with semaphore:

                async with session.get(url, timeout=None) as resp:

                    if resp.status == 404:
                        print(f"{code}: 404 not found")
                        return

                    if resp.status != 200:
                        print(f"{code}: HTTP {resp.status}")
                        continue

                    async with aiofiles.open(filename, "wb") as f:

                        async for chunk in resp.content.iter_chunked(chunk_size):
                            await f.write(chunk)

                    size_mb = filename.stat().st_size / 1e6

                    print(f"{code}: saved ({size_mb:.1f} MB)")

                    return

        except asyncio.CancelledError:
            print(f"{code} cancelled")
            return

        except Exception as e:

            print(f"{code} attempt {attempt}: {e}")

            await asyncio.sleep(3)

    print(f"{code}: failed after {retries} retries")


async def download_all_tiles(
    tile_codes,
    output_dir,
    base_url,
    max_concurrent=4,
    chunk_size=1024 * 1024,
    retries=3
):

    semaphore = asyncio.Semaphore(max_concurrent)

    async with aiohttp.ClientSession() as session:

        tasks = [
            download_tile(
                session,
                semaphore,
                code,
                output_dir,
                base_url,
                chunk_size,
                retries
            )
            for code in tile_codes
        ]

        await tqdm_asyncio.gather(*tasks)


# --------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------

async def main(
    data_dir,
    ahn_grid_path,
    output_dir,
    crs="EPSG:28992",
    regex_pattern=r"^(\d+)_(\d+)\.laz$",
    dataset_folder="AHN5_T",
    max_concurrent=4
):
    """
    Main pipeline callable from Jupyter.
    """

    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    base_url = f"https://geotiles.citg.tudelft.nl/{dataset_folder}/{{code}}.LAZ"

    ahn_tiles = get_required_ahn_tiles(
        data_dir,
        ahn_grid_path,
        regex_pattern,
        crs
    )

    await download_all_tiles(
        ahn_tiles,
        output_dir,
        base_url,
        max_concurrent=max_concurrent
    )