import random
import sys
from typing import List, Tuple

from map_processor import load_and_filter_map, select_start, get_goal_pixels
from navigator import init_sim, execute_waypoint_path


POINT_CLOUD_DATA = "semantic_3d_pointcloud/point.npy"
COLOR_DATA = "semantic_3d_pointcloud/color0255.npy"

# Sample semantic color and index dictionaries for a few object categories. 
# Check hw0/replica_v1/apartment_0/habitat/info_semantic.json and 
# hw3/color_coding_semantic_segmentation_classes.xlsx for the full list of 
# categories and their corresponding colors and indices.
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


def pick_goal(map_img) -> Tuple[str, Tuple[int, int]]:
    prompt = "Enter semantic destination (ex: 'rack', 'cooktop', 'sofa'): "
    goal_prompt = input(prompt).strip().lower()
    if goal_prompt not in SEMANTIC_DICTS["colors"]:
        print(f"Goal '{goal_prompt}' is not valid.")
        sys.exit(1)

    goal_pixels = get_goal_pixels(map_img, SEMANTIC_DICTS["colors"], goal_prompt)
    goal = random.choice(goal_pixels)
    return goal_prompt, goal


def run_in_sim(start_world: Tuple[float, float], world_path: List[Tuple[float, float]], goal_prompt: str):
    start_x, start_z = start_world
    print(f"Spawning Agent at world position: ({start_x:.3f}, {start_z:.3f})")

    sim, agent, _ = init_sim(start_x=start_x, start_z=start_z)
    execute_waypoint_path(world_path, sim, agent, SEMANTIC_DICTS["indices"][goal_prompt])


def main():
    """Entry point."""

    print("=== Step 1: Processing the 3D Map ===")
    # =============== TODO 1-2 ===============
    # map_img, occupancy_map, ... = load_and_filter_map(POINT_CLOUD_DATA, COLOR_DATA)


    print("=== Step 2: Selecting Agent Start and Goal Positions ===")
    start = select_start(map_img)
    goal_prompt, goal = pick_goal(map_img)
    print(f"Goal pixel selected at coordinates: {goal}")


    print("=== Step 3: Executing Path Planning (RRT) ===")
    # =============== TODO 2 ===============
    # implement RRT path planner in plan_path()

    # path = plan_path(start, goal, occupancy_map)
    # if not path:
    #     print("Planner could not find a path.")
    #     sys.exit(1)


    print("=== Step 4: Visualizing the Planned Path ===")
    # =============== TODO 3 ===============
    # Visualize the planned path over the map

    # visualize_path(...)


    print("=== Step 5: Translating Path to Habitat Simulator ===")
    # =============== TODO 4 ===============
    # Convert pixel path to world coordinates
    # world_path is a list of tuples(float, float) representing waypoints in world coordinates

    # world_path = ... 

    run_in_sim(world_path[0], world_path, goal_prompt)


if __name__ == "__main__":
    main()
