import cv2
import numpy as np
from typing import List, Tuple

SCALE_FACTOR = 10000.0 / 255.0
CEILING_COLOR = np.array([8, 255, 214])
FLOOR_COLOR = np.array([255, 194, 7])


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

    # return map_img, occupancy_map, ...


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
