# UPSW V2 — Amsterdam LiDAR Obstacle Classifier

Trains a PointNet++ classifier to label urban obstacles (trees, cars, bikes, bollards, benches, …) in Cyclomedia LiDAR point clouds of Amsterdam public space.

**Pipeline:** BGT/AHN auto-labeling → DBSCAN clustering → manual labeling review → PointNet++ training → inference on new tiles.

---

## Requirements

- Cyclomedia LiDAR tiles in LAZ format (one tile ≈ 50 × 50 m, EPSG:28992 / RD New)
- Internet access for the first run (downloads AHN elevation data and BGT road/object layers from PDOK and Amsterdam open data APIs)
- Python 3.11, conda recommended

---

## Installation

```bash
git clone https://github.com/MinkeVerweij/UPSW_V2.git
cd UPSW_V2

conda env create -f environment.yml
conda activate upsw
```

**GPU (optional but recommended for training):** replace `pytorch` with `pytorch-cuda=12.1` in `environment.yml` before creating the environment, or follow the [PyTorch installation guide](https://pytorch.org/get-started/locally/).

---

## Configuration

Edit `config.py` to point at your data directories. Add a block for your machine's hostname:

```python
elif HOSTNAME == "your-hostname":       # run: hostname
    TILE_DIR     = Path("/path/to/laz/tiles")   # folder containing your LAZ files
    LABELED_DIR  = Path("/path/to/labeled")
    CLUSTERS_DIR = Path("/path/to/clusters")
    # … other paths (copy the pattern from the existing blocks)
```

Then set the tilecodes you want to process at the bottom of `config.py`:

```python
SETUP_TILECODES = [
    "120300_489300",
    "120300_488900",
]
```

The tilecode is the `XXXXXX_YYYYYY` part of each LAZ filename.

---

## Notebooks — run in order

| # | Notebook | What it does |
|---|---|---|
| 0 | `0. Get AHN tiles.ipynb` | Download AHN4 elevation tiles from PDOK |
| 0.5 | `0.5. Data Setup.ipynb` | Download BGT road layers for your tilecodes |
| 1 | `1. AHN processing.ipynb` | Process AHN grids → NPZ elevation rasters |
| 1.6 | `1.6. Scrape BGT + Bomen Atlas.ipynb` | Scrape trees, poles, furniture, OSM objects |
| 2 | `2. Ground and Road fusion.ipynb` | Label ground / road / building points |
| 2.5 | `2.5. BGT Object Labeling.ipynb` | Label known objects (trees, cars, poles, …) |
| 4 | `4. Auto-Labeling Pipeline.ipynb` | DBSCAN clustering + auto-assign BGT labels |
| 4.5 | `4.5. Enhanced Labeling Review.ipynb` | Interactive UI to confirm / correct labels |
| 5 | `5. Dataset Preparation.ipynb` | Train / val / test split, RF baseline check |
| 6 | `6. Train PointNet++.ipynb` | Train PointNet++ SSG classifier |
| 7 | `7. Inference Pipeline.ipynb` | Apply model to new tiles, export GeoJSON |

**Comparison notebook:** `Cluster Comparison.ipynb` — side-by-side comparison of the 3D connected-component clustering (UPSW_neo) vs the 2D DBSCAN approach used here.

---

## Label taxonomy

| Code | Class | Code | Class |
|---|---|---|---|
| 30 | Tree | 65 | Bollard / slender pole |
| 40 | Car | 80 | City bench |
| 49 | Partial car | 81 | Rubbish bin |
| 44 | Single bike | 83 | Large container |
| 46 | Multi-bike | 91 | Terrace |
| 47 | Bike on pole | 99 | Noise |
| 45 | Scooter | 60 | Street light |

Custom classes (code ≥ 200) can be added at runtime in the labeling UI.

---

## Key design decisions

- **Physical scale preserved** — XY centered on cluster centroid, Z absolute. Ball query radii (0.2 m / 0.4 m) are physically meaningful. No unit-sphere normalisation.
- **Tile-based train/test split** — `GroupShuffleSplit` by tilecode prevents same-tile data leakage.
- **RF baseline gate** — macro-F1 < 70 % before PointNet++ training signals a label quality problem, not a model problem.
- **2D DBSCAN on unknown points** — only points not already identified by BGT/AHN are clustered, so labeled objects are never re-detected as obstacles.

---

## Data not included

All LAZ, NPZ, pkl, and model files are excluded from the repository (`.gitignore`). Transfer data between machines with `rsync`:

```bash
rsync -av --progress /path/to/data/ user@remote:/path/to/data/
```
