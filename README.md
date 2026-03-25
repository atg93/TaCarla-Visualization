# TaCarla-Visualization

A multi-sensor autonomous driving **data visualization toolkit** built around the [CARLA](https://carla.org/) simulator. It reads raw sensor recordings and structured annotation labels, renders annotated images for each frame (BEV detections, lane overlays, LiDAR, 6-camera surround view), and exposes an interactive Streamlit UI for browsing and playing back results.

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Data Layout](#data-layout)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Running the UI](#running-the-ui)
  - [Processing a Scenario (CLI)](#processing-a-scenario-cli)
  - [Choosing a Backend](#choosing-a-backend)
- [Output Structure](#output-structure)
- [Camera Rig](#camera-rig)
- [Shell Scripts](#shell-scripts)
- [Requirements](#requirements)

---

## Overview

TaCarla-Visualization processes driving episodes collected in CARLA. Each episode contains:

- **Sensor data** — RGB images from 6 surround cameras + LiDAR point clouds (`.las`/`.laz`)
- **Label data** — Parquet files with per-frame annotations (bounding boxes, lane lines, traffic lights, waypoints, BEV detection images, measurements)

The pipeline builds **sliding windows** of 16 consecutive frames, renders all visual modalities for each window, and saves the results as PNG images. A Streamlit UI lets you browse episodes, switch between sensor folders, and play back frames as a video.

---

## Repository Structure

```
TaCarla-Visualization/
├── _UI.py                              # Streamlit viewer application
├── run_multiprocess.py                 # CPU-parallel rendering pipeline
├── run_pyspark.py                      # PySpark-based rendering pipeline (cluster)
├── utils.py                            # Drawing & geometry helper functions
├── requirements.txt                    # Python dependencies
├── Visualize_Data_in_the_environment.sh  # Launcher (native environment)
└── Visualize_Data_using_sing.sh          # Launcher (Singularity container)
```

---

## Data Layout

Each Parquet file must contain these columns:

| Column | Description |
|---|---|
| `id` | Frame identifier (used for sorting) |
| `measurements` | Ego vehicle telemetry (sequence of 16 used as temporal context) |
| `lanes` | Lane line annotations |
| `new_boxes` | 3D bounding boxes |
| `detection_bev_image` | Relative path to BEV detection image |
| `lidar` | Relative path to LiDAR file |
| `front`, `front_left`, `front_right`, `back`, `back_left`, `back_right` | Relative paths to camera images |
| `text_annotations` | Caption |

---

## Installation

### Prerequisites

- Python 3.8+
- [CARLA Python egg](https://carla.readthedocs.io/en/latest/start_quickstart/) on `PYTHONPATH`
- `lazrs` backend for LiDAR decompression (installed via `pip` as part of `laspy[lazrs]`)

### Steps

```bash
git clone https://github.com/atg93/TaCarla-Visualization.git
cd TaCarla-Visualization

pip install -r requirements.txt

# Add the CARLA Python egg to your path (adjust version and path as needed)
export PYTHONPATH=$PYTHONPATH:/path/to/carla/PythonAPI/carla/dist/carla-*.egg
```

---

## Configuration

All paths are controlled via environment variables:

| Variable | Description | Example |
|---|---|---|
| `MAIN_PATH` | Root directory containing sensor and label folders | `/media/hdd/text_data` |
| `TOWN_FOLDER` | Town/scenario name prefix | `Town12` |
| `OUT_DIR` | Output folder name (relative to `MAIN_PATH`) | `Vis_output` |
| `PYSPARK_ENABLE` | Set to `true` to use PySpark backend instead of multiprocessing | `false` |

Example:

```bash
export MAIN_PATH=/media/hdd/text_data
export TOWN_FOLDER=Town12
export OUT_DIR=Vis_output
export PYSPARK_ENABLE=false
```

---

## Usage

### Running the UI

```bash
streamlit run _UI.py
```

The Streamlit app will open in your browser. From there you can:

- **Select an episode** from the dropdown
- **Browse sensor folders** (front, BEV, LiDAR, etc.) via folder buttons
- **Scrub frames** with the sample slider
- **Play back** frames as animation with adjustable FPS (1–30)
- **Navigate** with keyboard: `Space` / `→` = next frame, `←` = previous frame
- **Trigger rendering** for the current episode using the **"Visualize the current episode"** button

### Processing a Scenario (CLI)

To run the rendering pipeline directly without the UI:

```bash
python run_multiprocess.py
```

This processes all episode folders found under `SENSORS_PATH`, writing annotated PNGs to `OUT_DIR`.

### Choosing a Backend

| Backend | When to use |
|---|---|
| `run_multiprocess.py` | Default — works on any single machine, no Spark required |
| `run_pyspark.py` | Large-scale processing on a Spark/YARN cluster |

Switch via the `PYSPARK_ENABLE` environment variable (see [Configuration](#configuration)).

---

## Output Structure

For each processed episode the pipeline creates:

```
<OUT_DIR>/
└── <episode_name>/
    ├── bev/          # Bird's-eye view with detection/lane/waypoint overlays
    ├── label/        # Raw label renderings
    ├── lidar/        # LiDAR point cloud projected to BEV image
    ├── front/        # Front camera with lane, box, and traffic-light overlays
    ├── front_left/
    ├── front_right/
    ├── back/
    ├── back_left/
    └── back_right/
```

Each image is named `<frame_id>.png`.

---

## Camera Rig

Six RGB cameras at 1600 × 900 px, all at 70° FoV except the rear camera (110°):

| ID | X | Y | Z | Yaw | FoV |
|---|---|---|---|---|---|
| `front` | 0.701 | 0.016 | 1.511 | 0° | 70° |
| `front_right` | 0.551 | 0.493 | 1.496 | +55° | 70° |
| `front_left` | 0.524 | −0.495 | 1.509 | −55° | 70° |
| `back` | −1.528 | 0.003 | 1.579 | 180° | 110° |
| `back_left` | −0.536 | −0.485 | 1.591 | −110° | 70° |
| `back_right` | −0.515 | 0.481 | 1.562 | +110° | 70° |

---

## Shell Scripts

| Script | Description |
|---|---|
| `Visualize_Data_in_the_environment.sh` | Launches the pipeline in the current native environment |
| `Visualize_Data_using_sing.sh` | Launches the pipeline inside a Singularity container |

Edit these scripts to set your `MAIN_PATH`, `TOWN_FOLDER`, and `OUT_DIR` before running.

---

## Requirements

Core dependencies (see `requirements.txt` for pinned versions):

- `streamlit` — interactive UI
- `opencv-python` — image processing
- `numpy`, `Pillow` — array and image utilities
- `pyarrow` — Parquet I/O
- `laspy[lazrs]` — LiDAR `.las`/`.laz` reading
- `carla` — CARLA Python API (provided as `.egg` separately)
- `pyspark` *(optional)* — for the cluster-scale backend

---

## License

No license file is currently included in this repository.
