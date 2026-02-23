# run_mp_final.py
# Multiprocessing alternative to run_pyspark_final.py (no Spark).
# Safe under Streamlit: uses "spawn" start method and caps native threads.
# - Reads each <folder>/<folder>.parquet from DATA_PATH
# - Builds sliding windows of length SEQ_LEN over 'measurements' (starting at each row)
# - Processes rows in parallel and writes images under OUT_DIR/<folder>/{bev,label,lidar,cameras}
# - Preserves your drawing/helpers via imports from utils

# ------------------------------
# Thread caps (set BEFORE imports)
# ------------------------------
import os as _os
_os.environ.setdefault("OMP_DISPLAY_ENV", "FALSE")
_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
_os.environ.setdefault("MKL_NUM_THREADS", "1")
_os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
_os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# ------------------------------
# Standard lib
# ------------------------------
import os
import ast
import sys
import math
import time
import signal
import traceback
from contextlib import contextmanager
from typing import Iterator, Dict, Any, List, Optional

# ------------------------------
# 3rd-party
# ------------------------------
import numpy as np
import cv2
from PIL import Image
import laspy
from laspy import LazBackend
import pyarrow as pa
import pyarrow.parquet as pq
import carla  # ensure egg is on PYTHONPATH for subprocesses

# ------------------------------
# Multiprocessing (spawn-safe)
# ------------------------------
from multiprocessing import cpu_count, get_context

# ------------------------------
# Your helper module
# ------------------------------
from utils import (
    point_in_canvas, build_projection_matrix, get_image_point, world_from_transform,
    compose_camera_world_matrix, _cam_pose_from_ego_and_rel, _in_front_and_in_fov,
    visualize_lane, new_visualize, draw_lidar, draw_tl, inverse_get_relative_transform,
    draw_vehicle, plot_bounding_box_center, draw_label_raw, draw_on_bev,
    get_waypoints, get_virtual_lidar_to_vehicle_transform, get_vehicle_to_virtual_lidar_transform,
    transform_waypoints, draw_wp_of_ego, draw_text_on_bev
)

# =============================================================================
# CONFIG
# =============================================================================
try:
    MAIN_PATH = os.getenv("MAIN_PATH")  #"/media/hdd/text_data"
    TOWN_FOLDER = os.getenv("TOWN_FOLDER")
    SENSORS_PATH = MAIN_PATH + "/" + TOWN_FOLDER + "_sensors" #"/media/hdd/text_data/leaderboard_plant_pdm_Town12"
    LABELS_PATH = MAIN_PATH + "/" + TOWN_FOLDER + "_labels"
    OUT_DIR = MAIN_PATH + "/" + os.getenv("OUT_DIR")  #

except:
    MAIN_PATH = "/media/hdd/text_data/deneme_data"
    TOWN_FOLDER = "Town12"
    SENSORS_PATH = MAIN_PATH + "/" + TOWN_FOLDER + "_sensors" #"/media/hdd/text_data/leaderboard_plant_pdm_Town12"
    LABELS_PATH = MAIN_PATH + "/" + TOWN_FOLDER + "_labels"
    OUT_DIR = MAIN_PATH + "/" + "Vis_output"  #

assert os.path.exists(SENSORS_PATH) and os.path.exists(LABELS_PATH)


SEQ_LEN   = 16

PROCESSES     = max(1, cpu_count() - 1)
CHUNK_SIZE    = 32
BATCH_ROWS    = 200_000
MAX_TASKS_PER = 200  # recycle workers to limit leaks

os.makedirs(OUT_DIR, exist_ok=True)

# =============================================================================
# CAMERA RIG (if helpers rely on this global)
# =============================================================================
new_camera_dict: Dict[str, Dict[str, Any]] = {}
w = 1600
h = 900

def add_cam(cam): new_camera_dict[cam["id"]] = cam

add_cam({'type': 'sensor.camera.rgb', 'x': 0.70079118954, 'y': 0.0159456324149, 'z': 1.51095763913,
         'roll': 0.0, 'pitch': 0.0, 'yaw':   0.0, 'width': w, 'height': h, 'fov': 70,  'id': 'front'})
add_cam({'type': 'sensor.camera.rgb', 'x': 0.5508477543,  'y': 0.493404796419, 'z': 1.49574800619,
         'roll': 0.0, 'pitch': 0.0, 'yaw':  55.0, 'width': w, 'height': h, 'fov': 70,  'id': 'front_right'})
add_cam({'type': 'sensor.camera.rgb', 'x': 0.52387798135, 'y':-0.494631336551, 'z': 1.50932822144,
         'roll': 0.0, 'pitch': 0.0, 'yaw': -55.0, 'width': w, 'height': h, 'fov': 70,  'id': 'front_left'})
add_cam({'type': 'sensor.camera.rgb', 'x':-1.5283260309358,'y': 0.00345136761476,'z': 1.57910346144,
         'roll': 0.0, 'pitch': 0.0, 'yaw':-180.0, 'width': w, 'height': h, 'fov': 110, 'id': 'back'})
add_cam({'type': 'sensor.camera.rgb', 'x':-0.53569100218,  'y':-0.484795032713, 'z': 1.59097014818,
         'roll': 0.0, 'pitch': 0.0, 'yaw':-110.0, 'width': w, 'height': h, 'fov': 70,  'id': 'back_left'})
add_cam({'type': 'sensor.camera.rgb', 'x':-0.5148780988,   'y': 0.480568219723, 'z': 1.56239545128,
         'roll': 0.0, 'pitch': 0.0, 'yaw': 110.0, 'width': w, 'height': h, 'fov': 70,  'id': 'back_right'})

CAMERA_NAMES = ["back", "back_left", "back_right", "front", "front_left", "front_right"]

# =============================================================================
# UTILS
# =============================================================================
@contextmanager
def pushd(new_dir: str):
    prev = os.getcwd()
    os.makedirs(new_dir, exist_ok=True)
    os.chdir(new_dir)
    try:
        yield
    finally:
        os.chdir(prev)

def ensure_out_dirs(base_out_dir: str):
    for sd in ["bev", "label", "lidar"] + CAMERA_NAMES + ["_tmp"]:
        os.makedirs(os.path.join(base_out_dir, sd), exist_ok=True)

def _abs_from_main(rel_path: str) -> Optional[str]:
    rel_path = (rel_path or "").split(" ")[-1]
    if not rel_path:
        return None
    full = SENSORS_PATH + '/' + '/'.join(rel_path.split("/")[2:])
    #full = os.path.join(MAIN_PATH, rel_path.lstrip("/"))
    return full if os.path.exists(full) else None

def safe_read_image(rel_path: str):
    full = _abs_from_main(rel_path)
    if not full:
        return None
    try:
        return np.array(Image.open(full))
    except Exception:
        return None

def _arrow_cell_as_py(col: pa.ChunkedArray, i: int):
    try:
        return col[i].as_py()
    except Exception:
        return col.slice(i, 1).to_pylist()[0]

def _arrow_slice_as_py(col: pa.ChunkedArray, start: int, length: int):
    return col.slice(start, length).to_pylist()

# =============================================================================
# WORKER
# =============================================================================
def _init_worker():
    # Keep native threads per worker minimal
    try:
        import cv2 as _cv2
        _cv2.setNumThreads(1)
    except Exception:
        pass

#'detection_bev_image', 'lidar', 'back', 'back_left', 'back_right', 'front', 'front_left', 'front_right'

def process_row_dict(row_dict: dict) -> dict:
    base_out_dir = row_dict.get("out_dir", OUT_DIR) or OUT_DIR
    ensure_out_dirs(base_out_dir)

    _id = str(row_dict.get("id"))
    lanes = row_dict.get("lanes") or []
    new_boxes = row_dict.get("new_boxes") or []
    content = (new_boxes or []) + (lanes or [])
    meas_seq = row_dict.get("measurements_seq") or []
    text_annotations = ast.literal_eval(row_dict.get("text_annotations")) or {}
    idf_score = row_dict.get("idf_score") or []

    tmp_dir = os.path.join(base_out_dir, "_tmp", _id)

    # --- BEV detection image ---
    bev_path = os.path.join(base_out_dir, "bev", f"{_id}.png")
    bev_rel = row_dict.get("detection_bev_image")
    bev_img = safe_read_image(bev_rel)
    if bev_img is not None:
        try:
            bev_bgr = cv2.cvtColor(bev_img, cv2.COLOR_RGB2BGR) if bev_img.ndim == 3 else bev_img
            draw_on_bev(bev_bgr, lanes, new_boxes)
            draw_wp_of_ego(bev_bgr, meas_seq)
            bev_bgr = draw_text_on_bev(bev_bgr, text_annotations["caption"], idf_score)
            cv2.imwrite(bev_path, bev_bgr)
        except Exception:
            bev_path = ""
    else:
        bev_path = ""

    # --- Label raw rendering (draw_label_raw -> label_image.png in CWD) ---
    label_path = os.path.join(base_out_dir, "label", f"{_id}.png")
    with pushd(tmp_dir):
        try:
            _ = draw_label_raw(content, "detection", label_path)
            src = os.path.join(tmp_dir, "label_image.png")
            if os.path.exists(src):
                os.replace(src, label_path)
            else:
                label_path = ""
        except Exception:
            label_path = ""

    # --- LiDAR BEV ---
    lidar_path = os.path.join(base_out_dir, "lidar", f"{_id}.png")
    lidar_rel = (row_dict.get("lidar") or "").split(" ")[-1]
    if lidar_rel:
        full = SENSORS_PATH + '/' + '/'.join(lidar_rel.split("/")[2:])
        if os.path.exists(full):
            try:
                with laspy.open(full, laz_backend=LazBackend.Lazrs) as f:
                    las = f.read()
                draw_lidar(las.xyz, out_path=lidar_path)
            except Exception:
                try:
                    with laspy.open(full) as f:
                        las = f.read()
                    draw_lidar(las.xyz, out_path=lidar_path)
                except Exception:
                    lidar_path = ""
        else:
            lidar_path = ""
    else:
        lidar_path = ""

    # --- Camera overlays ---
    camera_outputs = {}
    for cname in CAMERA_NAMES:
        camera_outputs[cname] = ""
        rel = row_dict.get(cname)
        img = safe_read_image(rel)
        if img is None:
            continue

        dst = os.path.join(base_out_dir, cname, f"{_id}.png")
        with pushd(tmp_dir):
            try:
                if cname == "front":
                    _ = draw_tl(img, content)
            except Exception:
                pass

            try:
                img2 = visualize_lane(img, content, cname)
            except Exception:
                img2 = img

            try:
                new_visualize(img2, content, cname)
            except Exception:
                pass

            candidates = [
                f"{cname}.png",
                f"{cname}_image.png",
                f"{cname}_overlay.png",
                f"{cname}_vis.png",
            ]
            moved = False
            for cand in candidates:
                src = os.path.join(tmp_dir, cand)
                if os.path.exists(src):
                    try:
                        os.replace(src, dst)
                        moved = True
                        break
                    except FileNotFoundError:
                        moved = False

            if not moved:
                try:
                    to_write = img2
                    if isinstance(to_write, np.ndarray) and to_write.ndim == 3 and to_write.shape[2] == 3:
                        to_write = cv2.cvtColor(to_write, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(dst, to_write)
                except Exception:
                    dst = ""

        camera_outputs[cname] = dst if dst and os.path.exists(dst) else ""

    # cleanup (best-effort)
    try:
        if os.path.isdir(tmp_dir) and not os.listdir(tmp_dir):
            os.rmdir(tmp_dir)
    except Exception:
        pass

    return {
        "id": _id,
        "bev_path": bev_path,
        "label_path": label_path,
        "lidar_path": lidar_path,
        "front_path": camera_outputs.get("front", ""),
        "front_left_path": camera_outputs.get("front_left", ""),
        "front_right_path": camera_outputs.get("front_right", ""),
        "back_path": camera_outputs.get("back", ""),
        "back_left_path": camera_outputs.get("back_left", ""),
        "back_right_path": camera_outputs.get("back_right", ""),
    }

# =============================================================================
# PARQUET -> TASKS (streamed, sliding windows)
# =============================================================================
NEEDED_COLUMNS = [
    "id","measurements","lanes","new_boxes","detection_bev_image","lidar",
    "back","back_left","back_right","front","front_left","front_right", "text_annotations","idf_score"
]

def iter_tasks_from_parquet(info_path: str, seq_len: int, out_dir: str):
    meta = pq.ParquetFile(info_path)
    schema = meta.schema_arrow
    missing = [c for c in NEEDED_COLUMNS if c not in schema.names]
    if missing:
        raise KeyError(f"Missing columns in {info_path}: {missing}")

    table_ids = pq.read_table(info_path, columns=["id"])
    ids = np.array(table_ids["id"]).astype(float)
    order = np.argsort(ids, kind="mergesort")

    STREAMING = False

    if not STREAMING:
        table = pq.read_table(info_path, columns=NEEDED_COLUMNS)
        table = table.take(pa.array(order))
        n = table.num_rows

        for i in range(0, n - seq_len + 1):
            meas_seq = _arrow_slice_as_py(table["measurements"], i, seq_len)
            row = {name: _arrow_cell_as_py(table[name], i) for name in NEEDED_COLUMNS if name != "measurements"}
            row["measurements_seq"] = meas_seq
            row["out_dir"] = out_dir
            yield row
    else:
        table_full = pq.read_table(info_path, columns=NEEDED_COLUMNS)
        table_full = table_full.take(pa.array(order))
        n = table_full.num_rows

        start = 0
        while start < n:
            length = min(BATCH_ROWS, n - start)
            overlap = min(seq_len - 1, n - (start + length))
            end = start + length + overlap

            batch = table_full.slice(start, end - start)
            for i in range(0, min(length, batch.num_rows - seq_len + 1)):
                meas_seq = _arrow_slice_as_py(batch["measurements"], i, seq_len)
                row = {name: _arrow_cell_as_py(batch[name], i) for name in NEEDED_COLUMNS if name != "measurements"}
                row["measurements_seq"] = meas_seq
                row["out_dir"] = out_dir
                yield row

            start += length

# =============================================================================
# POOL DRIVER
# =============================================================================
def process_folder(folder, SENSORS_PATH, LABELS_PATH, OUT_DIR):
    folder_dir = os.path.join(SENSORS_PATH, folder)
    pq_name = f"{folder}.parquet"
    info_path = os.path.join(LABELS_PATH, pq_name)
    if not (os.path.isdir(folder_dir) and os.path.exists(info_path)):
        return 0, 0

    per_folder_out = os.path.join(OUT_DIR, folder)
    ensure_out_dirs(per_folder_out)

    produced = 0
    errors = 0

    tasks = iter_tasks_from_parquet(info_path, SEQ_LEN, per_folder_out)

    ctx = get_context("spawn")
    with ctx.Pool(
        processes=PROCESSES,
        initializer=_init_worker,
        maxtasksperchild=MAX_TASKS_PER
    ) as pool:
        try:
            for _ in pool.imap_unordered(process_row_dict, tasks, chunksize=CHUNK_SIZE):
                produced += 1
                if produced % 200 == 0:
                    print(f"[{folder}] processed {produced} rows...", flush=True)
        except KeyboardInterrupt:
            print("Interrupted by user. Terminating workers...", flush=True)
            pool.terminate(); pool.join()
            raise
        except Exception as e:
            print("Error in pool:", e, flush=True)
            traceback.print_exc()
            pool.terminate(); pool.join()
            raise
        else:
            pool.close(); pool.join()

    return produced, errors

# =============================================================================
# MAIN (optional CLI)
# =============================================================================
def main():
    if not os.path.isdir(SENSORS_PATH):
        raise FileNotFoundError(f"Data path not found: {SENSORS_PATH}")

    if not os.path.isdir(LABELS_PATH):
        raise FileNotFoundError(f"Data path not found: {LABELS_PATH}")

    os.makedirs(OUT_DIR, exist_ok=True)

    folders = sorted(os.listdir(SENSORS_PATH))
    total = 0
    started = time.time()
    for folder in folders:
        produced, _ = process_folder(folder, SENSORS_PATH, LABELS_PATH, OUT_DIR)
        total += produced
        if produced:
            print(f"[{folder}] done: {produced} windows", flush=True)
    dur = time.time() - started
    print(f"All done. Produced {total} windows in {dur:.1f}s.", flush=True)

if __name__ == "__main__":
    main()
