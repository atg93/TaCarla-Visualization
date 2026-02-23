import math

from pyspark.sql import SparkSession
from pyspark.sql import Window
from pyspark.sql.functions import row_number, col
import copy
import os
from PIL import Image
import carla
import cv2
import numpy as np

import laspy
from laspy import LazBackend

#from run_pyspark_v1 import *

def read_from_parquet(df_window, index, name):
    assert name in key_list
    _data = (df_window.where(col("rn") == index + 1)  # pick the i-th row globally
    .select(name)
    .first()[name])
    if type(_data) == type([]):
        data = [r.asDict(recursive=True) for r in _data]
    elif type(_data) == type(''):
        data = _data.split(' ')[-1]
    else:
        data = _data.asDict(recursive=True)
    return data

# ----------------------------
# Camera rig
# ----------------------------
new_camera_dict = {}
w = 1600
h = 900

def add_cam(cam):
    new_camera_dict[cam["id"]] = cam

add_cam({'type': 'sensor.camera.rgb', 'x': 0.70079118954, 'y': 0.0159456324149, 'z': 1.51095763913,
         'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0, 'width': w, 'height': h, 'fov': 70, 'id': 'front'})
add_cam({'type': 'sensor.camera.rgb', 'x': 0.5508477543, 'y': 0.493404796419, 'z': 1.49574800619,
         'roll': 0.0, 'pitch': 0.0, 'yaw': 55.0, 'width': w, 'height': h, 'fov': 70, 'id': 'front_right'})
add_cam({'type': 'sensor.camera.rgb', 'x': 0.52387798135, 'y': -0.494631336551, 'z': 1.50932822144,
         'roll': 0.0, 'pitch': 0.0, 'yaw': -55.0, 'width': w, 'height': h, 'fov': 70, 'id': 'front_left'})
add_cam({'type': 'sensor.camera.rgb', 'x': -1.5283260309358, 'y': 0.00345136761476, 'z': 1.57910346144,
         'roll': 0.0, 'pitch': 0.0, 'yaw': -180.0, 'width': w, 'height': h, 'fov': 110, 'id': 'back'})
add_cam({'type': 'sensor.camera.rgb', 'x': -0.53569100218, 'y': -0.484795032713, 'z': 1.59097014818,
         'roll': 0.0, 'pitch': 0.0, 'yaw': -110.0, 'width': w, 'height': h, 'fov': 70, 'id': 'back_left'})
add_cam({'type': 'sensor.camera.rgb', 'x': -0.5148780988, 'y': 0.480568219723, 'z': 1.56239545128,
         'roll': 0.0, 'pitch': 0.0, 'yaw': 110.0, 'width': w, 'height': h, 'fov': 70, 'id': 'back_right'})

# ----------------------------
# Utils
# ----------------------------

def point_in_canvas(pos, img_h, img_w):
    return (0 <= pos[0] < img_w) and (0 <= pos[1] < img_h)

def build_projection_matrix(w, h, fov, is_behind_camera=False):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.eye(3, dtype=float)
    K[0, 0] = K[1, 1] = (-focal if is_behind_camera else focal)
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K

def get_image_point(loc, K, w2c):
    point = np.array([loc.x, loc.y, loc.z, 1.0], dtype=float)
    pc = np.dot(w2c, point)
    # UE (x,y,z) -> (y,-z,x)
    pc = np.array([pc[1], -pc[2], pc[0]], dtype=float)
    if abs(pc[2]) < 1e-6:
        pc[2] = 1e-6
    p = np.dot(K, pc)
    p[0] /= p[2]; p[1] /= p[2]
    return p[0:2]

def world_from_transform(tf: carla.Transform) -> np.ndarray:
    return np.array(tf.get_matrix(), dtype=float)

def compose_camera_world_matrix(ego_tf: carla.Transform, cam_rel_tf: carla.Transform) -> np.ndarray:
    """world_from_camera = world_from_ego @ ego_from_camRel"""
    ego_mat = world_from_transform(ego_tf)
    rel_mat = world_from_transform(cam_rel_tf)
    return ego_mat @ rel_mat

def _cam_pose_from_ego_and_rel(ego_tf: carla.Transform, cam_rel_tf: carla.Transform):
    """Return (cam_mat 4x4 world_from_camera, cam_loc (carla.Location), cam_fwd (carla.Vector3D, unit))."""
    cam_mat = compose_camera_world_matrix(ego_tf, cam_rel_tf)
    Rcw = cam_mat[:3, :3].astype(float)
    t_cw = cam_mat[:3, 3].astype(float)

    cam_loc = carla.Location(float(t_cw[0]), float(t_cw[1]), float(t_cw[2]))
    # UE forward is +X axis in camera frame -> first column of world_from_camera is cam +X in world
    cam_fwd = np.array([Rcw[0, 0], Rcw[1, 0], Rcw[2, 0]], dtype=float)
    n = np.linalg.norm(cam_fwd) + 1e-12
    cam_fwd /= n
    return cam_mat, cam_loc, carla.Vector3D(*cam_fwd)

def _in_front_and_in_fov(cam_loc: carla.Location,
                         cam_fwd: carla.Vector3D,
                         target_loc: carla.Location,
                         hfov_deg: float) -> bool:
    """Return True iff target is in front of camera and within horizontal FOV."""
    r = np.array([target_loc.x - cam_loc.x,
                  target_loc.y - cam_loc.y,
                  target_loc.z - cam_loc.z], dtype=float)
    r_norm = np.linalg.norm(r)
    if r_norm < 1e-9:
        return True
    r_unit = r / r_norm
    f = np.array([cam_fwd.x, cam_fwd.y, cam_fwd.z], dtype=float)
    dot = float(np.dot(f, r_unit))
    if dot <= 0.0:
        return False
    cos_half = np.cos(np.deg2rad(hfov_deg) / 2.0)
    return dot >= cos_half

# ----------------------------
# 3D -> 2D visualization (actors) with camera-centric gating
# ----------------------------
def visualize_lane(image_rgb, content, camera_name, actor_dist_thresh_m= 30.0):
    #img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    img_bgr = image_rgb
    edges = [[0,0]]

    fov = new_camera_dict[camera_name]["fov"]
    H, W, _ = img_bgr.shape
    K     = build_projection_matrix(W, H, fov, is_behind_camera=False)
    K_neg = build_projection_matrix(W, H, fov, is_behind_camera=True)

    # Ego pose
    ego = content[0]
    ego_loc = carla.Location(*ego["world_location"])
    ego_rot = carla.Rotation(pitch=ego["vehicle_rotation"][0],
                             roll=ego["vehicle_rotation"][1],
                             yaw=ego["vehicle_rotation"][2])
    ego_tf  = carla.Transform(ego_loc, ego_rot)

    # Camera pose
    cam_rel_tf = carla.Transform(
        carla.Location(x=new_camera_dict[camera_name]["x"],
                       y=new_camera_dict[camera_name]["y"],
                       z=new_camera_dict[camera_name]["z"]),
        carla.Rotation(roll=new_camera_dict[camera_name]["roll"],
                       pitch=new_camera_dict[camera_name]["pitch"],
                       yaw=new_camera_dict[camera_name]["yaw"])
    )
    cam_mat, cam_loc, cam_fwd = _cam_pose_from_ego_and_rel(ego_tf, cam_rel_tf)
    world_2_camera = np.linalg.inv(cam_mat)

    for idx, npc in enumerate(content):
        if idx == 0:
            continue
        if npc.get("class") != "Lane":
            continue

        npc_loc_1 = inverse_get_relative_transform(npc['position'], np.array(carla.Transform(ego_loc, ego_rot).get_matrix()))
        npc_loc = carla.Location(x=npc_loc_1[0],y=npc_loc_1[1],z=npc_loc_1[2])
        if npc_loc.distance(ego_loc) >= actor_dist_thresh_m:
            continue
        if not _in_front_and_in_fov(cam_loc, cam_fwd, npc_loc, fov):
            continue

        # Simple 1m cube to visualize the lane point
        bb = carla.BoundingBox()
        w_, l_, h_ = 0.0001, 0.0001, 0.0001
        bb.extent.x = float(l_) / 2.0  # forward
        bb.extent.y = float(w_) / 2.0  # right
        bb.extent.z = float(h_) / 2.0  # up

        verts = [npc_loc]
        for e0, e1 in edges:
            p1 = get_image_point(verts[e0], K, world_2_camera)

            # Behind-camera continuity
            if not point_in_canvas(p1, H, W):
                r0 = verts[e0] - cam_loc
                if cam_fwd.dot(r0) <= 0:
                    p1 = get_image_point(verts[e0], K_neg, world_2_camera)

            cv2.circle(img_bgr, (int(p1[0]), int(p1[1])), 5, (0, 0, 255), -1)

    return img_bgr

def new_visualize(image_rgb, content, camera_name, actor_dist_thresh_m=60.0):
    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    edges = [[0,1],[1,3],[3,2],[2,0],[0,4],[4,5],[5,1],[5,7],[7,6],[6,4],[6,2],[7,3]]

    fov = new_camera_dict[camera_name]["fov"]
    H, W, _ = img_bgr.shape
    K     = build_projection_matrix(W, H, fov, is_behind_camera=False)
    K_neg = build_projection_matrix(W, H, fov, is_behind_camera=True)

    # Ego pose
    ego = content[0]
    ego_loc = carla.Location(*ego["world_location"])
    ego_loc = carla.Location(x=ego_loc.x,
                       y=ego_loc.y,
                       z=ego_loc.z)
    ego_rot = carla.Rotation(pitch=ego["vehicle_rotation"][0],
                             roll=ego["vehicle_rotation"][1],
                             yaw=ego["vehicle_rotation"][2])
    ego_tf  = carla.Transform(ego_loc, ego_rot)

    # Camera pose
    cam_rel_tf = carla.Transform(
        carla.Location(x=new_camera_dict[camera_name]["x"],
                       y=new_camera_dict[camera_name]["y"],
                       z=new_camera_dict[camera_name]["z"]),
        carla.Rotation(roll=new_camera_dict[camera_name]["roll"],
                       pitch=new_camera_dict[camera_name]["pitch"],
                       yaw=new_camera_dict[camera_name]["yaw"])
    )
    cam_mat, cam_loc, cam_fwd = _cam_pose_from_ego_and_rel(ego_tf, cam_rel_tf)
    world_2_camera = np.linalg.inv(cam_mat)

    for idx, npc in enumerate(content):
        if idx == 0:
            continue
        if not check_car(npc["class"]):
            continue

        npc_loc = carla.Location(*npc['world_location'])
        if npc_loc.distance(ego_loc) >= actor_dist_thresh_m:
            continue
        if not _in_front_and_in_fov(cam_loc, cam_fwd, npc_loc, fov):
            continue

        # Build bbox
        bb_loc = carla.Location(*npc['bounding_box_location'])
        bb_rot = carla.Rotation(pitch=npc['bounding_box_rotation'][0],
                                roll =npc['bounding_box_rotation'][1],
                                yaw  =npc['bounding_box_rotation'][2])
        bb = carla.BoundingBox()
        bb.rotation = bb_rot
        # npc["extent"] assumed [width(y), length(x), height(z)]
        h_, l_, w_ = npc["extent"]
        bb.extent.x = float(l_) / 2.0  # forward
        bb.extent.y = float(w_) / 2.0  # right
        bb.extent.z = float(h_) / 2.0  # up
        bb.location = bb_loc

        npc_tf = carla.Transform(npc_loc,
                                 carla.Rotation(pitch=npc["vehicle_rotation"][0],
                                                roll =npc["vehicle_rotation"][1],
                                                yaw  =npc["vehicle_rotation"][2]))

        verts = [v for v in bb.get_world_vertices(npc_tf)]
        for e0, e1 in edges:
            p1 = get_image_point(verts[e0], K, world_2_camera)
            p2 = get_image_point(verts[e1], K, world_2_camera)

            # Behind-camera continuity
            if not point_in_canvas(p1, H, W) and not point_in_canvas(p2, H, W):
                r0 = verts[e0] - cam_loc
                r1 = verts[e1] - cam_loc
                if cam_fwd.dot(r0) <= 0:
                    p1 = get_image_point(verts[e0], K_neg, world_2_camera)
                if cam_fwd.dot(r1) <= 0:
                    p2 = get_image_point(verts[e1], K_neg, world_2_camera)

            cv2.line(img_bgr, (int(p1[0]), int(p1[1])),
                              (int(p2[0]), int(p2[1])), (0, 0, 255), 1)

    cv2.imwrite(f"{camera_name}.png", img_bgr)

# ----------------------------
# LiDAR BEV (simple raster)
# ----------------------------

def draw_lidar(lidar_xyz, lidar_range=200, image_size=400, out_path='lidar.png'):
    pts2d = (lidar_xyz[:, :2] * (image_size / lidar_range)).astype(int) * 2
    img = np.zeros((image_size, image_size), dtype=np.uint8)
    pts = pts2d + image_size // 2
    for x, y in pts:
        if 0 <= x < image_size and 0 <= y < image_size:
            img[y, x] = 255
    cv2.imwrite(out_path, img)

# ----------------------------
# Traffic light 2D poly overlay
# ----------------------------

def draw_tl(front_image_rgb, content):
    img_bgr = cv2.cvtColor(front_image_rgb, cv2.COLOR_RGB2BGR)
    for boxes in content:
        if boxes.get('class') == 'two_d_light_p':
            for edge in boxes['two_d_light_p']:
                p1_0, p1_1, p2_0, p2_1 = np.array(edge).flatten()
                cv2.line(img_bgr, (int(p1_0), int(p1_1)), (int(p2_0), int(p2_1)), (255, 0, 0), 2)
    return front_image_rgb

# ----------------------------
# Ego-relative -> world transform
# ----------------------------

def inverse_get_relative_transform(relative_pos, ego_matrix):
    """
    return the relative transform from ego_pose to vehicle pose
    """
    rot = np.eye(3)
    #relative_pos = rot @ (relative_pos)
    relative_pos = rot @ (relative_pos + np.array([1.3, 0.0, 2.5]))
    relative_pos[-1] = 0.0
    # transform from right handed system
    relative_pos[1] = - relative_pos[1]

    ###
    rot = ego_matrix[:3, :3].T
    rot = np.linalg.inv(rot)
    relative_pos = rot @ relative_pos

    position = relative_pos + ego_matrix[:3, 3]

    return position


# ----------------------------
# BEV helpers (200x200)
# ----------------------------

def draw_vehicle(center, orientation_rad, velocity, box_size, bbox, arrow_thick=1):
    mask_vehicle = np.zeros((200, 200), dtype=np.uint8)
    mask_arrow = np.zeros((200, 200), dtype=np.uint8)

    width, height = bbox[0] * 3, bbox[1] * 3
    tl = (int(center[1] - width / 2), int(center[0] - height / 2))
    br = (int(center[1] + width / 2), int(center[0] + height / 2))
    tr = (tl[0], br[1]); bl = (br[0], tl[1])
    pts = np.array([br, tr, tl, bl], dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask_vehicle, [pts], color=255)

    cxy = (int(center[1]), int(center[0]))
    base_len = 10
    line_len = int(np.clip(velocity * base_len, base_len, base_len * 2))
    endp = (int(cxy[0] - line_len * np.cos(orientation_rad + (np.pi/2))),
            int(cxy[1] - line_len * np.sin(orientation_rad + (np.pi/2))))
    cv2.arrowedLine(mask_arrow, cxy, endp, 255, arrow_thick)

    return mask_vehicle.astype(bool), mask_arrow.astype(bool)

def plot_bounding_box_center(center, width=4, height=8):
    mask = np.zeros((200, 200), dtype=np.uint8)
    tl = (int(center[1] - width / 2), int(center[0] - height / 2))
    br = (int(center[1] + width / 2), int(center[0] + height / 2))
    cv2.rectangle(mask, br, tl, 255, 1)
    return mask

def draw_label_raw(label_raw, name, out_path):
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    mask_vehicle_image = np.zeros((200, 200), dtype=np.uint8)
    special_vehicle_image = np.zeros((200, 200), dtype=np.uint8)
    bike_and_cons_vehicle_image = np.zeros((200, 200), dtype=np.uint8)
    lane_guidance_mask = np.zeros((200, 200), dtype=np.uint8)

    for index, sample in enumerate(label_raw):
        center = np.array(sample['position'], dtype=float)
        if name == 'detection':
            center = np.array([center[0], center[1]], dtype=float)
        center *= (-1)
        center = center * 4 + 100

        if sample['class'] == 'Route':
            sample['speed'] = 0

        if sample['class'] == 'Car':
            bbox = np.array(sample["extent"], dtype=float)
            mask_vehicle, mask_arrow = draw_vehicle(center, sample['yaw'], sample['speed'],
                                                    (bbox[0], bbox[1]), bbox)
            if index == 0:
                ego_front_mask, _ = draw_vehicle(np.array([86, 100]), sample['yaw'],
                                                 sample['speed'], (bbox[0]/2, bbox[1]/2), bbox/2)
                image[mask_vehicle] = (255, 255, 255)
                image[mask_arrow] = (255, 0, 0)
            else:
                image[mask_vehicle] = (0, 0, 255)
                image[mask_arrow] = (255, 0, 0)
                _, mask_arrow = draw_vehicle(center, sample['yaw'], sample['speed'], (bbox[0], bbox[1]), bbox)
                mask_vehicle_image[mask_vehicle] = 255
                mask_vehicle_image[mask_arrow] = 255

        elif sample['class'] in ('Police', 'Firetruck', 'Crossbike', 'Construction', 'Ambulance', 'Walker'):
            bbox = np.array(sample["extent"], dtype=float)
            if sample['class'] == 'Crossbike':
                bbox = np.array([2.0, 2.0, 2.0], dtype=float)
            mask_vehicle, mask_arrow = draw_vehicle(center, sample['yaw'], sample['speed'],
                                                    (bbox[0], bbox[1]), bbox)
            if index == 0:
                image[mask_vehicle] = (255, 255, 255)
                image[mask_arrow] = (0, 255, 0)
            else:
                image[mask_vehicle] = (0, 255, 0)
                image[mask_arrow] = (0, 0, 255)
                mask_vehicle_image[mask_vehicle] = 255
                special_vehicle_image[mask_vehicle] = 255
                special_vehicle_image[mask_arrow] = 255
                if sample['class'] in ('Crossbike', 'Construction', 'Walker'):
                    bike_and_cons_vehicle_image[mask_vehicle] = 255
                    bike_and_cons_vehicle_image[mask_arrow] = 255
                _, mask_arrow = draw_vehicle(center, sample['yaw'], sample['speed'], (bbox[0], bbox[1]), bbox)
                mask_vehicle_image[mask_arrow] = 255

        elif sample['class'] == 'Radar':
            bbox = np.array(sample["extent"], dtype=float)
            mask_vehicle, _ = draw_vehicle(center, sample['yaw'], sample['speed'], (bbox[0], bbox[1]), bbox)
            image[mask_vehicle] = (255, 0, 0)

        elif sample['class'] == 'lane_guidance':
            bbox = np.array(sample["extent"], dtype=float)
            position_center = np.array(sample['position'], dtype=float) * 4 + 100
            mask_lane, _ = draw_vehicle(position_center, sample['yaw'], 0.0, (bbox[0], bbox[1]), bbox)
            lane_guidance_mask |= mask_lane
            image[mask_lane] = (255, 255, 255)

        elif sample['class'] == "Lane":
            bbox = np.array(sample["extent"], dtype=float)
            mask_lane, _ = draw_vehicle(center, sample['yaw'], 0.0, (bbox[0], bbox[1]), bbox)
            lane_guidance_mask |= mask_lane
            image[mask_lane] = (255, 255, 0)

        elif sample['class'] in ('tl_bev_pixel', 'Stop_sign'):
            bbox = np.array(sample["extent"], dtype=float)
            mask = plot_bounding_box_center(center, bbox[0], bbox[1]).astype(bool)
            if sample['state'] == 2:
                image[mask] = (0, 255, 0)
                if sample['class'] == 'tl_bev_pixel':
                    tl_image = copy.deepcopy(mask)
            elif sample['state'] == 1:
                image[mask] = (0, 255, 255)
                if sample['class'] == 'tl_bev_pixel':
                    tl_image = copy.deepcopy(mask)
            elif sample['state'] == 0:
                image[mask] = (0, 0, 255)

        elif sample['class'] == "Lane_guidance_wp":
            bbox = np.array(sample["extent"], dtype=float)
            mask = plot_bounding_box_center(center, bbox[0], bbox[1]).astype(bool)
            image[mask] = (255, 255, 255)

        if sample['class'] == 'tl_bev_pixel':
            bbox = np.array(sample["extent"], dtype=float)
            mask = plot_bounding_box_center(center, bbox[0], bbox[1]).astype(bool)
            if np.linalg.norm(np.array(sample['tl_bev_pixel_coordinate']).mean(0)) < 25:
                if sample['state'] == 1:
                    image[mask] = (0, 255, 255)
                elif sample['state'] == 0:
                    image[mask] = (0, 0, 255)

    #cv2.imwrite("label_image.png", image)
    cv2.imwrite(out_path, image)
    return image

import numpy as np
import cv2

def world_to_bev_xy(x, y, scale, H, W):
    """
    Map CARLA world (x forward, y right) to BEV pixel coords.
    You use: u = -y*scale + W/2, v = -x*scale + H/2
    """
    u = -y * scale + W / 2.0
    v = -x * scale + H / 2.0
    return int(round(u)), int(round(v))

def vehicle_box_corners_xy(center_xy, yaw_deg, length_m, width_m):
    """
    Return 4 XY corners (world meters) of a rotated rectangle around center_xy.
    CARLA yaw is degrees; convert to radians.
    Corners order: front-left, front-right, rear-right, rear-left.
    """
    cx, cy = center_xy
    hl = length_m / 2.0  # half length (along +x forward in CARLA)
    hw = width_m  / 2.0  # half width  (along +y right in CARLA)

    # Box in vehicle local frame (x forward, y right)
    local = np.array([
        [ hl, -hw],
        [ hl,  hw],
        [-hl,  hw],
        [-hl, -hw],
    ], dtype=np.float32)  # shape (4,2)

    # Rotate around center by yaw (radians)
    # NOTE: with your screen mapping, a positive CARLA yaw needs a minus sign
    # to look correct on the image (because image axes are flipped). If it looks
    # mirrored, flip the sign on yaw_rad below.
    yaw_rad = -np.deg2rad(yaw_deg)
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    R = np.array([[c, -s],
                  [s,  c]], dtype=np.float32)

    rotated = (local @ R.T)
    corners = rotated + np.array([cx, cy], dtype=np.float32)
    return corners  # (4,2) in world meters

def draw_vehicle_bev(img, center_xyz, yaw_deg, extent_or_size, scale, color=(0,255,0), thickness=2, fill=False, dims_are_full=True, order="W,L,H"):
    """
    center_xyz: [x,y,z] in meters (CARLA)
    yaw_deg: vehicle yaw in degrees
    extent_or_size: either full sizes [W,L,H] or half sizes depending on dims_are_full.
    order: which order the input uses; default matches your sample [width, length, height].
    """
    H, W = img.shape[:2]
    x, y = float(center_xyz[0]), float(center_xyz[1])

    # Parse sizes
    W_in, L_in = None, None
    if order.upper() == "W,L,H":
        W_in, L_in = float(extent_or_size[0]), float(extent_or_size[1])
    elif order.upper() == "L,W,H":
        L_in, W_in = float(extent_or_size[0]), float(extent_or_size[1])
    else:
        # fallback assume [x_len, y_len, z] with x=length, y=width (CARLA bbox.extent style)
        L_in, W_in = float(extent_or_size[0]), float(extent_or_size[1])

    if dims_are_full:
        length_m, width_m = L_in, W_in
    else:
        length_m, width_m = 2.0 * L_in, 2.0 * W_in

    # 4 corners in world (meters)
    corners_xy = vehicle_box_corners_xy((x, y), yaw_deg, length_m, width_m)

    # Map to pixels
    pts = np.array([world_to_bev_xy(cx, cy, scale, H, W) for (cx, cy) in corners_xy], dtype=np.int32).reshape(-1,1,2)

    if fill:
        cv2.fillPoly(img, [pts], color=color)
        # Outline for visibility
        cv2.polylines(img, [pts], isClosed=True, color=(0,0,0), thickness=1, lineType=cv2.LINE_AA)
    else:
        cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)

    # Draw a small heading arrow (from center to front-center)
    # Front center point in world: (hl, 0) rotated + translated
    hl = length_m / 2.0
    yaw_rad = -np.deg2rad(yaw_deg)
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    fx, fy = x + c*hl, y + s*hl  # front direction in world

    u0, v0 = world_to_bev_xy(x,  y,  scale, H, W)
    u1, v1 = world_to_bev_xy(fx, fy, scale, H, W)
    _color = tuple(255 - np.array(color).astype(int))
    cv2.arrowedLine(img, (u0,v0), (u1,v1), (int(_color[0]),int(_color[1]),int(_color[2])), 2, tipLength=0.3)

    return img

def check_car(class_name):
    return class_name == "Car" or class_name == "Police" or class_name == "Ambulance" or class_name == "Firetruck" \
           or class_name == "Construction" or class_name == "Walker" or class_name == "Crossbike"

def draw_on_bev(bev_camera_image, lane_content, _content, scale=12.5):
    for ct in lane_content:
        center_point = np.array(ct['position']) + np.array([1.3, 0.0, 2.5])
        points_left = np.array(ct['points_left']) + np.array([1.3, 0.0, 2.5])
        points_right = np.array(ct['points_right']) + np.array([1.3, 0.0, 2.5])
        cv2.circle(bev_camera_image, (int(center_point[1]*(-1)*scale+bev_camera_image.shape[1]/2),int(center_point[0]*(-1)*scale+bev_camera_image.shape[0]/2)), 1, (255, 0, 0), -1)
        cv2.circle(bev_camera_image, (int(points_left[1]*(-1)*scale+bev_camera_image.shape[1]/2),int(points_left[0]*(-1)*scale+bev_camera_image.shape[0]/2)), 1, (0, 0, 255), -1)
        cv2.circle(bev_camera_image, (int(points_right[1]*(-1)*scale+bev_camera_image.shape[1]/2),int(points_right[0]*(-1)*scale+bev_camera_image.shape[0]/2)), 1, (0, 0, 255), -1)

    for ct_index, ct in enumerate(_content):
        if not check_car(ct['class']):
            continue

        color = (0, 0, 255)
        if ct_index == 0:
            color = (255,0, 0)


        center_point = np.array(ct['position']) + np.array([1.3, 0.0, 2.5])
        draw_vehicle_bev(bev_camera_image, center_point, math.degrees(ct['yaw']), ct['extent'], scale, color)


def get_waypoints(measurements):
    assert len(measurements) == 16
    num = 16
    waypoints = {"1": []}

    for i in range(0, num):
        waypoints["1"].append([measurements[i]["ego_matrix"], True])

    Identity = list(list(row) for row in np.eye(4))
    # padding here
    for k in waypoints.keys():
        while len(waypoints[k]) < num:
            waypoints[k].append([Identity, False])
    return waypoints

def get_virtual_lidar_to_vehicle_transform():
    # This is a fake lidar coordinate
    T = np.eye(4)
    T[0, 3] = 1.3
    T[1, 3] = 0.0
    T[2, 3] = 2.5
    return T

def get_vehicle_to_virtual_lidar_transform():
    return np.linalg.inv(get_virtual_lidar_to_vehicle_transform())

def transform_waypoints(waypoints):
    """transform waypoints to be origin at ego_matrix"""

    # TODO should transform to virtual lidar coordicate?
    T = get_vehicle_to_virtual_lidar_transform()

    for k in waypoints.keys():
        vehicle_matrix = np.array(waypoints[k][0][0])
        vehicle_matrix_inv = np.linalg.inv(vehicle_matrix)
        for i in range(1, len(waypoints[k])):
            matrix = np.array(waypoints[k][i][0])
            waypoints[k][i][0] = T @ vehicle_matrix_inv @ matrix

    return waypoints



def draw_wp_of_ego(bev_camera_image, loaded_measurements, scale=12.5):
    # ego car is always the first one in label file
    waypoints = get_waypoints(loaded_measurements)
    waypoints = transform_waypoints(waypoints)

    filtered_waypoints = []
    for id in ["1"]:
        waypoint = []
        for matrix, _ in waypoints[id][1:]:
            waypoint.append(matrix[:2, 3])
        filtered_waypoints.append(waypoint)
    ego_waypoint = np.array(filtered_waypoints).reshape(-1,2)

    for ct in ego_waypoint:
        ct_wp = np.array(ct) + np.array([1.3, 0.0])
        cv2.circle(bev_camera_image, (int(ct_wp[1]*scale+bev_camera_image.shape[1]/2),int(ct_wp[0]*(-1)*scale+bev_camera_image.shape[0]/2)), 3, (0, 255, 0), -1)

    #cv2.imwrite("bev_camera_image.png",bev_camera_image)

from typing import Optional, Tuple, Union, List

def _auto_font_scale(img_w: int, base_w: int = 1280, base_scale: float = 0.8) -> float:
    """
    Heuristic: scale the font with image width so text looks reasonable across sizes.
    """
    return max(0.4, base_scale * (img_w / float(base_w)))


def _text_size(text: str, font, font_scale: float, thickness: int) -> Tuple[int, int, int]:
    """
    Return (width, height, baseline) for a single-line text with OpenCV.
    """
    (w, h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    return w, h, baseline

def _wrap_text_to_width(
    text: str,
    max_w: int,
    font,
    font_scale: float,
    thickness: int,
) -> List[str]:
    """
    Greedy word-wrapping to fit text in max_w pixels (approx using cv2.getTextSize).
    Splits on whitespace; if a single token is longer than max_w, it will be hard-broken.
    """
    if not text:
        return [""]

    words = text.split()
    lines: List[str] = []
    cur: List[str] = []

    def width_of(s: str) -> int:
        w, _, _ = _text_size(s, font, font_scale, thickness)
        return w

    for wtoken in words:
        candidate = (" ".join(cur + [wtoken])).strip()
        if width_of(candidate) <= max_w or not cur:
            cur.append(wtoken)
        else:
            # push current line
            line = " ".join(cur).strip()
            if line:
                lines.append(line)
            else:
                lines.append("")
            cur = [wtoken]

        # Hard-break a single very long token
        while width_of(" ".join(cur)) > max_w and len(cur) == 1 and len(cur[0]) > 1:
            tok = cur[0]
            # binary search-ish: find largest prefix that fits
            lo, hi = 1, len(tok)
            best = 1
            while lo <= hi:
                mid = (lo + hi) // 2
                if width_of(tok[:mid]) <= max_w:
                    best = mid
                    lo = mid + 1
                else:
                    hi = mid - 1
            lines.append(tok[:best])
            cur[0] = tok[best:]

    if cur:
        lines.append(" ".join(cur).strip())

    return lines

def normalize(idf, min_val=1.278820592487134, max_val=2.8576686588437608):
    return np.clip((idf - min_val) / (max_val - min_val), 0, 1)

def draw_text_on_bev(
    image: Union[str, np.ndarray],
    annotation_text: str,
    rarity_text: str,
    out_path: Optional[str] = None,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    thickness: int = 2,
    pad_x: int = 16,
    pad_y: int = 10,
    line_gap: int = 8,
    para_gap: int = 6,
    bar_color: Tuple[int, int, int] = (0, 0, 0),
    text_color: Tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """
    Add a black bar to the bottom and draw wrapped multi-line text.
    - First paragraph: "text: {annotation_text}"
    - Second paragraph: "idf score: {idf_text}"
    """
    # Load image if a path is provided
    if isinstance(image, str):
        img = cv2.imread(image, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read image from path: {image}")
    else:
        img = image
        if not isinstance(img, np.ndarray) or img.ndim < 2:
            raise ValueError("image must be a valid numpy array (H x W x C or H x W).")
    rarity_float = float(rarity_text)
    rarity_float = normalize(rarity_float)
    rarity_text = str(round(rarity_float, 4))


    h, w = img.shape[:2]
    font_scale = _auto_font_scale(w)

    # Build labeled strings
    ann_full = f"text: {annotation_text}".strip()
    idf_full = f"rarity score: {rarity_text}".strip()

    # Maximum allowed text width inside the bar
    max_text_w = max(1, w - 2 * pad_x)

    # Wrap into lines
    ann_lines = _wrap_text_to_width(ann_full, max_text_w, font, font_scale, thickness)
    idf_lines = _wrap_text_to_width(idf_full, max_text_w, font, font_scale, thickness)

    # Measure line height
    _, line_h, base = _text_size("Ag", font, font_scale, thickness)  # representative height

    # Calculate total bar height
    # top pad + lines (with gaps) + paragraph gap + idf lines (with gaps) + bottom pad
    ann_block_h = len(ann_lines) * line_h + (len(ann_lines) - 1) * line_gap if ann_lines else 0
    idf_block_h = len(idf_lines) * line_h + (len(idf_lines) - 1) * line_gap if idf_lines else 0
    bar_h = pad_y + ann_block_h + (para_gap if ann_lines and idf_lines else 0) + idf_block_h + pad_y
    bar_h = 102 #max(bar_h, line_h + 2 * pad_y)  # always at least one line tall

    # Create bar
    bar = np.full((bar_h, w, 3), bar_color, dtype=np.uint8)

    # Draw lines
    y = pad_y + line_h  # baseline of first line
    for line in ann_lines:
        cv2.putText(bar, line, (pad_x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)
        y += line_h + line_gap

    if ann_lines and idf_lines:
        y += max(0, para_gap - line_gap)  # add extra space between paragraphs

    for line in idf_lines:
        cv2.putText(bar, line, (pad_x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)
        y += line_h + line_gap

    out = np.vstack([img, bar])

    return out



