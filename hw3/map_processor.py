from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

SCALE_FACTOR = 10000.0 / 255.0
CEILING_COLOR = np.array([8, 255, 214])
FLOOR_COLOR = np.array([255, 194, 7])
ALT_FLOOR_COLORS = (np.array([255, 184, 6]),)


@dataclass(frozen=True)
class MapMeta:
    """Affine metadata for converting between map pixels and Habitat x-z world."""

    min_x: float
    max_z: float
    resolution: float
    width: int
    height: int


def _color_mask(colors: np.ndarray, target: np.ndarray, tolerance: float = 1.0) -> np.ndarray:
    return np.all(np.abs(colors - target.reshape(1, 3)) <= tolerance, axis=1)


def _multi_color_mask(colors: np.ndarray, targets: List[np.ndarray], tolerance: float = 1.0) -> np.ndarray:
    mask = np.zeros(colors.shape[0], dtype=bool)
    for target in targets:
        mask |= _color_mask(colors, target, tolerance=tolerance)
    return mask


def _points_to_pixels(coords: np.ndarray, meta: MapMeta) -> Tuple[np.ndarray, np.ndarray]:
    px = np.rint((coords[:, 0] - meta.min_x) / meta.resolution).astype(np.int32)
    py = np.rint((meta.max_z - coords[:, 2]) / meta.resolution).astype(np.int32)
    px = np.clip(px, 0, meta.width - 1)
    py = np.clip(py, 0, meta.height - 1)
    return px, py


def load_and_filter_map(
    point_path: str,
    color_path: str,
    resolution: float = 0.02,
    obstacle_inflation: int = 1,
    min_obstacle_area: int = 2,
):
    points = np.load(point_path)
    colors = np.load(color_path)
    if colors.max() <= 1.0:
        colors = colors * 255.0

    # Convert to real-world meters. In Habitat, x-z is horizontal and y is vertical.
    coords = points * SCALE_FACTOR

    floor_mask = _multi_color_mask(colors, [FLOOR_COLOR, *ALT_FLOOR_COLORS])
    ceiling_mask = _color_mask(colors, CEILING_COLOR)
    obstacle_mask = ~(floor_mask | ceiling_mask)

    min_x = float(coords[:, 0].min())
    max_x = float(coords[:, 0].max())
    min_z = float(coords[:, 2].min())
    max_z = float(coords[:, 2].max())
    width = int(np.ceil((max_x - min_x) / resolution)) + 1
    height = int(np.ceil((max_z - min_z) / resolution)) + 1
    meta = MapMeta(min_x=min_x, max_z=max_z, resolution=resolution, width=width, height=height)

    floor_grid = np.zeros((height, width), dtype=np.uint8)
    obstacle_grid = np.zeros((height, width), dtype=np.uint8)
    map_img = np.ones((height, width, 3), dtype=np.float32)
    obstacle_color_img = np.zeros((height, width, 3), dtype=np.float32)

    fx, fy = _points_to_pixels(coords[floor_mask], meta)
    floor_grid[fy, fx] = 255

    ox, oy = _points_to_pixels(coords[obstacle_mask], meta)
    obstacle_grid[oy, ox] = 255
    obstacle_color_img[oy, ox] = colors[obstacle_mask].astype(np.float32) / 255.0

    close_kernel = np.ones((5, 5), dtype=np.uint8)
    floor_grid = cv2.morphologyEx(floor_grid, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    obstacle_grid = cv2.morphologyEx(obstacle_grid, cv2.MORPH_OPEN, np.ones((2, 2), dtype=np.uint8))
    obstacle_grid = cv2.morphologyEx(obstacle_grid, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8))

    # Remove tiny obstacle speckles from point-cloud noise. Those speckles can be
    # inflated into doorway blockers even though they are not real geometry.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(obstacle_grid, connectivity=8)
    cleaned_obstacles = np.zeros_like(obstacle_grid)
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_obstacle_area:
            cleaned_obstacles[labels == label] = 255
    obstacle_grid = cleaned_obstacles
    map_img[obstacle_grid > 0] = obstacle_color_img[obstacle_grid > 0]

    inflate_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (obstacle_inflation * 2 + 1, obstacle_inflation * 2 + 1)
    )
    inflated_obstacles = cv2.dilate(obstacle_grid, inflate_kernel, iterations=1)

    # True means blocked. Anything outside observed floor is blocked.
    occupancy_map = (floor_grid == 0) | (inflated_obstacles > 0)
    cv2.imwrite("semantic_map.png", cv2.cvtColor((map_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    return map_img, occupancy_map, meta


def pixel_to_world(pixel: Tuple[int, int], meta: MapMeta) -> Tuple[float, float]:
    x, y = pixel
    world_x = meta.min_x + float(x) * meta.resolution
    world_z = meta.max_z - float(y) * meta.resolution
    return world_x, world_z


def world_to_pixel(world: Tuple[float, float], meta: MapMeta) -> Tuple[int, int]:
    x, z = world
    px = int(round((x - meta.min_x) / meta.resolution))
    py = int(round((meta.max_z - z) / meta.resolution))
    return int(np.clip(px, 0, meta.width - 1)), int(np.clip(py, 0, meta.height - 1))


def nearest_free_pixel(pixel: Tuple[int, int], occupancy_map: np.ndarray, max_radius: int = 80) -> Tuple[int, int]:
    x, y = pixel
    h, w = occupancy_map.shape
    if 0 <= x < w and 0 <= y < h and not occupancy_map[y, x]:
        return pixel

    for radius in range(1, max_radius + 1):
        x0, x1 = max(0, x - radius), min(w - 1, x + radius)
        y0, y1 = max(0, y - radius), min(h - 1, y + radius)
        candidates = []
        for cx in range(x0, x1 + 1):
            candidates.append((cx, y0))
            candidates.append((cx, y1))
        for cy in range(y0 + 1, y1):
            candidates.append((x0, cy))
            candidates.append((x1, cy))

        free = [(cx, cy) for cx, cy in candidates if not occupancy_map[cy, cx]]
        if free:
            return min(free, key=lambda p: (p[0] - x) ** 2 + (p[1] - y) ** 2)

    raise ValueError(f"No free pixel found near {pixel}.")


def select_start(map_img: np.ndarray, occupancy_map: np.ndarray = None) -> Tuple[int, int]:
    """Display map and return user-clicked start coordinate."""
    start_point = []

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if occupancy_map is not None and occupancy_map[y, x]:
                try:
                    free = nearest_free_pixel((x, y), occupancy_map)
                    start_point.append(free)
                    print(f"Clicked point is blocked; using nearest free start: {free}")
                except ValueError:
                    print("Clicked point is blocked and no nearby free cell was found.")
            else:
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
    """Find all pixels corresponding to the goal object based on color matching."""

    if goal_name.lower() not in semantic_dict:
        raise ValueError(f"Unknown semantic object: {goal_name}. Available options: {list(semantic_dict.keys())}")

    goal_colors = semantic_dict[goal_name.lower()]
    goal_pixels: List[Tuple[int, int]] = []

    for gc in goal_colors:
        gc_norm = np.array(gc) / 255.0
        mask_goal = np.all(np.isclose(map_img, gc_norm, atol=10 / 255.0), axis=2)
        zs, xs = np.where(mask_goal)
        goal_pixels.extend(list(zip(xs.astype(int), zs.astype(int))))

    if not goal_pixels:
        raise ValueError(f"No valid pixels found for '{goal_name}'.")

    return goal_pixels
