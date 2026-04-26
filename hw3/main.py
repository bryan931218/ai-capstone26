import random
import sys
from pathlib import Path
from typing import List, Tuple

import cv2

from map_processor import (
    get_goal_pixels,
    load_and_filter_map,
    nearest_free_pixel,
    pixel_to_world,
    select_start,
)
from navigator import init_sim, execute_waypoint_path
from rrt_planner import plan_path, visualize_path


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "semantic_3d_pointcloud"
if not DATA_DIR.exists():
    DATA_DIR = BASE_DIR.parent / "semantic_3d_pointcloud"

POINT_CLOUD_DATA = str(DATA_DIR / "point.npy")
COLOR_DATA = str(DATA_DIR / "color0255.npy")

# Colors come from color_coding_semantic_segmentation_classes.xlsx.
# Habitat semantic object ids are scene-specific. The provided template included
# ids for rack, cooktop, and sofa; cushion/stair still work for planning, but
# target highlighting needs hw0/apartment_0 metadata to determine exact ids.
SEMANTIC_DICTS = {
    "colors": {
        "rack": [[0, 255, 133]],
        "cushion": [[255, 9, 92]],
        "sofa": [[10, 0, 255]],
        "stair": [[173, 255, 0]],
        "cooktop": [[7, 255, 224]],
    },
    "indices": {
        "rack": 66,
        "cushion": 29,
        "sofa": 76,
        "stair": 77,
        "cooktop": 32,
    },
}


def pick_goal(map_img, occupancy_map, start: Tuple[int, int]) -> Tuple[str, Tuple[int, int]]:
    prompt = "Enter semantic destination (rack, cushion, sofa, stair, cooktop): "
    goal_prompt = input(prompt).strip().lower()
    if goal_prompt not in SEMANTIC_DICTS["colors"]:
        print(f"Goal '{goal_prompt}' is not valid.")
        sys.exit(1)

    goal_pixels = get_goal_pixels(map_img, SEMANTIC_DICTS["colors"], goal_prompt)
    component_count, labels = cv2.connectedComponents((~occupancy_map).astype("uint8"), 8)
    start_component = labels[start[1], start[0]]

    candidates = []
    for object_pixel in goal_pixels:
        try:
            approach = nearest_free_pixel(object_pixel, occupancy_map, max_radius=100)
        except ValueError:
            continue
        component = labels[approach[1], approach[0]]
        if component == start_component and component != 0:
            candidates.append((approach, object_pixel))

    if not candidates:
        print(f"No approach point for '{goal_prompt}' was found in the start component; using any reachable candidate.")
        for object_pixel in goal_pixels:
            try:
                approach = nearest_free_pixel(object_pixel, occupancy_map, max_radius=100)
            except ValueError:
                continue
            component = labels[approach[1], approach[0]]
            if 0 < component < component_count:
                candidates.append((approach, object_pixel))

    if candidates:
        candidates.sort(key=lambda item: (item[0][0] - start[0]) ** 2 + (item[0][1] - start[1]) ** 2)
        return goal_prompt, candidates[0][0]

    raise RuntimeError(f"Could not find a navigable approach point near '{goal_prompt}'.")


def run_in_sim(start_world: Tuple[float, float], world_path: List[Tuple[float, float]], goal_prompt: str):
    start_x, start_z = start_world
    print(f"Spawning Agent at world position: ({start_x:.3f}, {start_z:.3f})")

    sim, agent, _ = init_sim(start_x=start_x, start_z=start_z)
    goal_idx = SEMANTIC_DICTS["indices"].get(goal_prompt)
    if goal_idx is None:
        print(f"No Habitat semantic id is configured for '{goal_prompt}', so navigation runs without target mask.")
    execute_waypoint_path(world_path, sim, agent, goal_idx)


def main():
    """Entry point."""

    print("=== Step 1: Processing the 3D Map ===")
    map_img, occupancy_map, map_meta = load_and_filter_map(POINT_CLOUD_DATA, COLOR_DATA)
    print(f"Map size: {map_meta.width} x {map_meta.height} pixels, resolution: {map_meta.resolution:.3f} m/px")
    print("Saved semantic_map.png")

    print("=== Step 2: Selecting Agent Start and Goal Positions ===")
    start = select_start(map_img, occupancy_map)
    goal_prompt, goal = pick_goal(map_img, occupancy_map, start)
    print(f"Start pixel: {start}")
    print(f"Goal approach pixel selected at coordinates: {goal}")

    print("=== Step 3: Executing Path Planning (RRT) ===")
    path = plan_path(start, goal, occupancy_map)
    if not path:
        print("Planner could not find a path. Try another start point or increase max_iters.")
        sys.exit(1)
    print("Waypoints in pixel coordinates:")
    for point in path:
        print(point)

    print("=== Step 4: Visualizing the Planned Path ===")
    visualize_path(map_img, path, start, goal, out_path="rrt_path.png")
    print("Saved rrt_path.png")

    print("=== Step 5: Translating Path to Habitat Simulator ===")
    world_path = [pixel_to_world(point, map_meta) for point in path]
    print("Waypoints in Habitat x-z world coordinates:")
    for point in world_path:
        print(f"({point[0]:.3f}, {point[1]:.3f})")

    run_in_sim(world_path[0], world_path, goal_prompt)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
