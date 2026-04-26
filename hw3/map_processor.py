import cv2
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple

SCALE_FACTOR = 10000.0 / 255.0
CEILING_COLOR = np.array([8, 255, 214])
FLOOR_COLOR = np.array([255, 194, 7])
RESOLUTION = 40   # pixels per meter
PADDING = 20      # pixel padding around map edges


def load_and_filter_map(point_path: str, color_path: str):
    points = np.load(point_path)
    colors = np.load(color_path)

    # Convert to real-world meters
    coords = points * SCALE_FACTOR

    # =============== TODO 1-1 ===============
    # 1. Identify floor and ceiling by color (tolerance 5 per channel)
    floor_mask = np.all(np.abs(colors - FLOOR_COLOR) <= 5, axis=1)
    ceiling_mask = np.all(np.abs(colors - CEILING_COLOR) <= 5, axis=1)
    obj_mask = ~(floor_mask | ceiling_mask)

    # 2. Compute 2D grid bounds from ALL points (including floor) with padding
    #    In Habitat: x-z plane is horizontal, y is vertical.
    #    Mapping: z_world → col (image x-axis), x_world → row (image y-axis)
    all_x = coords[:, 0]
    all_z = coords[:, 2]
    x_min = all_x.min() - PADDING / RESOLUTION
    z_min = all_z.min() - PADDING / RESOLUTION
    x_max = all_x.max() + PADDING / RESOLUTION
    z_max = all_z.max() + PADDING / RESOLUTION

    H = int((x_max - x_min) * RESOLUTION) + 1
    W = int((z_max - z_min) * RESOLUTION) + 1

    # 3. Project non-floor/ceiling points to pixel grid
    obj_x = coords[obj_mask, 0]
    obj_z = coords[obj_mask, 2]
    obj_rgb = colors[obj_mask]

    col = np.clip(((obj_z - z_min) * RESOLUTION).astype(int), 0, W - 1)
    row = np.clip(((obj_x - x_min) * RESOLUTION).astype(int), 0, H - 1)

    # 4. Paint colored map image (white background, semantic colors for walls/furniture).
    #    Expand each projected point to a 2×2 patch so 1-pixel-wide wall lines remain
    #    visible after any subsequent filtering.
    map_img = np.ones((H, W, 3), dtype=np.float32)
    dr = np.array([0, 0, 1, 1])
    dc = np.array([0, 1, 0, 1])
    exp_row = np.clip((row[:, None] + dr[None, :]).ravel(), 0, H - 1)
    exp_col = np.clip((col[:, None] + dc[None, :]).ravel(), 0, W - 1)
    exp_rgb = np.repeat(obj_rgb, 4, axis=0)
    map_img[exp_row, exp_col] = exp_rgb / 255.0

    # 5. Remove noise pixels: require at least 4 occupied neighbours in a 5×5 window.
    #    This removes small stray clusters (1-3 px) while preserving real wall lines.
    occupied = (np.min(map_img, axis=2) < 0.99).astype(np.float32)
    neighbor_sum = cv2.filter2D(occupied, -1, np.ones((5, 5), dtype=np.float32))
    map_img[(neighbor_sum < 4) & (occupied > 0)] = 1.0

    # 6. Build occupancy map for path planning.
    # Derive obstacle mask directly from the cleaned visual map so that noise
    # already removed in step 5 is not re-introduced via point-cloud lookup.
    # Dilate obstacles by 1 px (3×3 ellipse) to make 1-pixel-wide walls thick
    # enough for reliable line-segment collision checks.
    # Floor footprint with generous dilation to fill entire rooms.
    floor_col = np.clip(((coords[floor_mask, 2] - z_min) * RESOLUTION).astype(int), 0, W - 1)
    floor_row = np.clip(((coords[floor_mask, 0] - x_min) * RESOLUTION).astype(int), 0, H - 1)
    floor_occ = np.zeros((H, W), dtype=np.uint8)
    floor_occ[floor_row, floor_col] = 1
    k_floor = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
    floor_dilated = cv2.dilate(floor_occ, k_floor)

    # Obstacle mask from cleaned visual map + 1-px dilation to thicken walls.
    occupied_vis = (np.min(map_img, axis=2) < 0.99).astype(np.uint8)
    k_wall = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    obs_occ = cv2.dilate(occupied_vis, k_wall)

    # Occupancy: 0 = free (inside floor area and not obstacle), 1 = obstacle
    occupancy_map = np.ones((H, W), dtype=np.uint8)
    occupancy_map[floor_dilated == 1] = 0
    occupancy_map[obs_occ == 1] = 1

    return map_img, occupancy_map, x_min, z_min, float(RESOLUTION)


def select_start(map_img: np.ndarray) -> Tuple[int, int]:
    """Display map with matplotlib and return user-clicked start coordinate."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(map_img)
    ax.set_title("Click to select start location, then press Enter")
    fig.tight_layout()

    coords_list = plt.ginput(1, timeout=0)
    plt.close(fig)

    if not coords_list:
        raise RuntimeError("No start selected. Exiting.")

    col = int(round(coords_list[0][0]))
    row = int(round(coords_list[0][1]))
    print(f"Start selected: ({col}, {row})")
    return col, row


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
        goal_pixels.extend(list(zip(xs.tolist(), zs.tolist())))

    if not goal_pixels:
        raise ValueError(f"No valid pixels found for '{goal_name}'.")

    return goal_pixels
