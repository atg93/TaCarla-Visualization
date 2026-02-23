# run_pyspark_final.py

import os
import ast
from typing import Iterator
from contextlib import contextmanager

from pyspark.sql import SparkSession, Row, Window
from pyspark.sql.functions import col, row_number, collect_list, size, lit
from pyspark.storagelevel import StorageLevel

# ==== 3rd-party (must exist on all executors) ====
import numpy as np
import cv2
from PIL import Image
import laspy
from laspy import LazBackend
import carla  # carla egg must be available on executors

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
MAIN_PATH = os.getenv("MAIN_PATH")  #"/media/hdd/text_data"
TOWN_FOLDER = os.getenv("TOWN_FOLDER")
SENSORS_PATH = MAIN_PATH + "/" + TOWN_FOLDER + "_sensors" #"/media/hdd/text_data/leaderboard_plant_pdm_Town12"
LABELS_PATH = MAIN_PATH + "/" + TOWN_FOLDER + "_labels" #"/media/hdd/text_data/leaderboard_plant_pdm_Town12"
OUT_DIR   = MAIN_PATH + "/" + os.getenv("OUT_DIR") #"/tmp/pyspark_vis_out"

SEQ_LEN   = 16
os.makedirs(OUT_DIR, exist_ok=True)

# =============================================================================
# CAMERA RIG
# =============================================================================
new_camera_dict = {}
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
    for sd in ["bev", "label", "lidar"] + CAMERA_NAMES:
        os.makedirs(os.path.join(base_out_dir, sd), exist_ok=True)
    os.makedirs(os.path.join(base_out_dir, "_tmp"), exist_ok=True)

def build_dataframe_with_sequences(spark, info_path: str, seq_len: int):
    base = (spark.read.parquet(info_path)
            .select("id","measurements","lanes","new_boxes",
                    "detection_bev_image","lidar",
                    "back","back_left","back_right","front","front_left","front_right", "text_annotations","idf_score"))
    w = Window.orderBy(col("id"))
    df = (base
          .withColumn("rn", row_number().over(w))
          .withColumn("measurements_seq",
                      collect_list(col("measurements")).over(w.rowsBetween(0, seq_len - 1)))
          .where(size(col("measurements_seq")) == seq_len))
    return df.persist(StorageLevel.MEMORY_AND_DISK)

def safe_read_image(rel_path: str):
    rel_path = (rel_path or "").split(" ")[-1]
    if not rel_path:
        return None
    #full = os.path.join(MAIN_PATH, rel_path.lstrip("/"))
    full = SENSORS_PATH + '/' + '/'.join(rel_path.split("/")[2:])
    if not os.path.exists(full):
        return None
    try:
        return np.array(Image.open(full))
    except Exception:
        return None



# =============================================================================
# WORKER
# =============================================================================
def process_row_dict(row_dict: dict, base_out_dir: str) -> dict:
    _id = str(row_dict.get("id"))
    lanes = row_dict.get("lanes") or []
    new_boxes = row_dict.get("new_boxes") or []
    content = new_boxes + lanes
    meas_seq = row_dict.get("measurements_seq") or []
    text_annotations = ast.literal_eval(row_dict.get("text_annotations")) or {}
    idf_score = row_dict.get("idf_score") or []

    ensure_out_dirs(base_out_dir)
    # UNIQUE temp directory per row to avoid races
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
    lidar_rel = row_dict.get("lidar").split(" ")[-1]
    if lidar_rel:
        full = SENSORS_PATH + '/' + '/'.join(lidar_rel.split("/")[2:])
        assert os.path.exists(full)

        if os.path.exists(full):
            with laspy.open(full, laz_backend=LazBackend.Lazrs) as f:
                las = f.read()
            draw_lidar(las.xyz, out_path=lidar_path)

        else:
            lidar_path = ""
    else:
        lidar_path = ""

    # --- Camera overlays (robust) ---
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

            # Try legacy writer; may or may not emit a file
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

def partition_worker(iter_rows: Iterator[Row]):
    for row in iter_rows:
        row_d = row.asDict(recursive=True)
        base_out_dir = row_d.get("out_dir", OUT_DIR) or OUT_DIR
        out = process_row_dict(row_d, base_out_dir)
        fields = {
            "id": "",
            "bev_path": "",
            "label_path": "",
            "lidar_path": "",
            "front_path": "",
            "front_left_path": "",
            "front_right_path": "",
            "back_path": "",
            "back_left_path": "",
            "back_right_path": "",
        }
        fields.update(out)
        yield Row(**fields)

# =============================================================================
# MAIN Function
# =============================================================================
def run_pyspark(folder, SENSORS_PATH, LABELS_PATH, OUT_DIR="/tmp/pyspark_vis_out"):
    if not os.path.isdir(SENSORS_PATH):
        raise FileNotFoundError(f"Data path not found: {SENSORS_PATH}")

    if not os.path.isdir(LABELS_PATH):
        raise FileNotFoundError(f"Data path not found: {LABELS_PATH}")

    pq_name = f"{folder}.parquet"
    folder_dir = os.path.join(SENSORS_PATH, folder)
    labels_list = os.listdir(LABELS_PATH)
    if folder not in os.listdir(OUT_DIR):
        if pq_name in labels_list:
            os.makedirs(OUT_DIR, exist_ok=True)

            spark = (
                SparkSession.builder
                .appName("pysparkized-bev-pipeline-final")
                .config("spark.driver.memory", "16g")
                .config("spark.executor.memory", "8g")
                .config("spark.executor.cores", "2")
                .config("spark.sql.shuffle.partitions", "200")
                .config("spark.sql.files.maxPartitionBytes", "64m")
                .config("spark.sql.execution.arrow.pyspark.enabled", "true")
                .getOrCreate()
            )

            folders = sorted(os.listdir(folder_dir))


            info_path = os.path.join(LABELS_PATH, pq_name)
            df_seq = build_dataframe_with_sequences(spark, info_path, SEQ_LEN).repartition(200)

            per_folder_out = os.path.join(OUT_DIR, folder)
            df_seq = df_seq.withColumn("out_dir", lit(per_folder_out))

            out_df = df_seq.rdd.mapPartitions(partition_worker).toDF()

            # trigger side-effects
            _ = out_df.count()

            # optional manifest:
            # out_df.coalesce(1).write.mode("overwrite").json(os.path.join(per_folder_out, "_manifest"))

            df_seq.unpersist(blocking=False)

            spark.stop()
        else:
            print(pq_name, " does not exist")
