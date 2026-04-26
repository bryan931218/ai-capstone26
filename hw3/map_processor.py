import cv2
import numpy as np
from typing import List, Tuple
from scipy import ndimage

SCALE_FACTOR = 10000.0 / 255.0
CEILING_COLOR = np.array([8, 255, 214])
FLOOR_COLOR = np.array([255, 194, 7])
MAP_RESOLUTION_M = 0.025  # meters / pixel
COLOR_TOL = 10.0


def load_and_filter_map(point_path: str, color_path: str):

    points = np.load(point_path)
    colors = np.load(color_path)

    # Convert to real-world meters
    coords = points * SCALE_FACTOR

    # =============== TODO 1-1 ===============
    # Hints: To get a good 2d map, filter ceiling/floor, project to 2D,
    # remove isolated points, inflate obstacles to get occupancy map, etc.
    # IMPORTANT: return map_img as float in value range [0, 1] for visualization downstream.
    # NOTE: in habitat sim, x z plane corresponds to world horizontal plane, and y is vertical.
    color_dist_floor = np.linalg.norm(colors - FLOOR_COLOR[None, :], axis=1)
    color_dist_ceiling = np.linalg.norm(colors - CEILING_COLOR[None, :], axis=1)
    non_floor_ceiling = (color_dist_floor > COLOR_TOL) & (color_dist_ceiling > COLOR_TOL)

    # Robust vertical filtering to remove most of the floor/ceiling leftovers.
    y = coords[:, 1]
    y_low, y_high = np.percentile(y, [8, 95])
    vertical_keep = (y > y_low) & (y < y_high)
    keep = non_floor_ceiling & vertical_keep

    coords_keep = coords[keep]
    colors_keep = colors[keep]
    if len(coords_keep) == 0:
        raise RuntimeError("All points were filtered out. Please check filtering thresholds.")

    # Project to x-z plane and rasterize.
    xs_world = coords_keep[:, 0]
    zs_world = coords_keep[:, 2]
    x_min, x_max = xs_world.min(), xs_world.max()
    z_min, z_max = zs_world.min(), zs_world.max()

    width = int(np.ceil((x_max - x_min) / MAP_RESOLUTION_M)) + 1
    height = int(np.ceil((z_max - z_min) / MAP_RESOLUTION_M)) + 1

    px = np.clip(((xs_world - x_min) / MAP_RESOLUTION_M).astype(np.int32), 0, width - 1)
    pz = np.clip(((zs_world - z_min) / MAP_RESOLUTION_M).astype(np.int32), 0, height - 1)

    # Build semantic/color map with mean color per pixel.
    color_sum = np.zeros((height, width, 3), dtype=np.float64)
    hit_count = np.zeros((height, width), dtype=np.float64)
    np.add.at(color_sum, (pz, px, slice(None)), colors_keep)
    np.add.at(hit_count, (pz, px), 1.0)

    valid = hit_count > 0
    map_img = np.ones((height, width, 3), dtype=np.float32)
    map_img[valid] = (color_sum[valid] / hit_count[valid, None]) / 255.0
    map_img = np.clip(map_img, 0.0, 1.0)

    # Occupancy: any hit is obstacle. Then clean isolated noise and inflate.
    obstacle_raw = valid.copy()
    structure = np.ones((3, 3), dtype=bool)
    obstacle_clean = ndimage.binary_opening(obstacle_raw, structure=structure)
    obstacle_clean = ndimage.binary_closing(obstacle_clean, structure=structure)

    # Inflate for robot safety margin.
    inflate_radius_px = int(np.ceil(0.15 / MAP_RESOLUTION_M))  # 15cm
    yy, xx = np.ogrid[-inflate_radius_px:inflate_radius_px + 1, -inflate_radius_px:inflate_radius_px + 1]
    disk = (xx * xx + yy * yy) <= (inflate_radius_px * inflate_radius_px)
    occupancy_map = ndimage.binary_dilation(obstacle_clean, structure=disk)
    occupancy_map = occupancy_map.astype(np.uint8)

    return map_img, occupancy_map, x_min, z_min, MAP_RESOLUTION_M


def select_start(map_img: np.ndarray) -> Tuple[int, int]:
    """Display map and return user-clicked start coordinate."""
    start_point = []

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            start_point.append((x, y))
            print(f"Start selected: ({x}, {y})")

    cv2.namedWindow("Select Start")
    cv2.setMouseCallback("Select Start", mouse_callback)
    print("Click on the map window to select a start location...")

    while True:
        cv2.imshow("Select Start", (map_img * 255).astype(np.uint8))
        key = cv2.waitKey(1) & 0xFF
        if start_point:
            break
        if key == ord("q"):
            raise RuntimeError("No start selected. Exiting.")

    cv2.destroyWindow("Select Start")
    return start_point[0]


def get_goal_pixels(map_img: np.ndarray, semantic_dict: dict, goal_name: str) -> List[Tuple[int, int]]:
    """function to find all pixels corresponding to the goal object based on color matching."""

    if goal_name.lower() not in semantic_dict:
        raise ValueError(f"Unknown semantic object: {goal_name}. Available options: {list(semantic_dict.keys())}")

    goal_colors = semantic_dict[goal_name.lower()]
    goal_pixels: List[Tuple[float, float]] = []

    for gc in goal_colors:
        gc_norm = np.array(gc) / 255.0
        mask_goal = np.all(np.isclose(map_img, gc_norm, atol=10/255.0), axis=2)
        zs, xs = np.where(mask_goal)
        goal_pixels.extend(list(zip(xs, zs)))

    if not goal_pixels:
        raise ValueError(f"No valid pixels found for '{goal_name}'.")

    return goal_pixels
