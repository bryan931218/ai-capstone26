import random
import sys
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from map_processor import load_and_filter_map, select_start, get_goal_pixels
from navigator import init_sim, execute_waypoint_path


POINT_CLOUD_DATA = "semantic_3d_pointcloud/point.npy"
COLOR_DATA = "semantic_3d_pointcloud/color0255.npy"

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


def _is_free(occupancy_map: np.ndarray, node: Tuple[int, int]) -> bool:
    x, z = node
    h, w = occupancy_map.shape
    return (0 <= x < w) and (0 <= z < h) and (occupancy_map[z, x] == 0)


def _line_collision_free(p0: Tuple[int, int], p1: Tuple[int, int], occupancy_map: np.ndarray) -> bool:
    x0, z0 = p0
    x1, z1 = p1
    dist = int(max(abs(x1 - x0), abs(z1 - z0)))
    if dist == 0:
        return _is_free(occupancy_map, p0)
    xs = np.linspace(x0, x1, dist + 1).astype(np.int32)
    zs = np.linspace(z0, z1, dist + 1).astype(np.int32)
    h, w = occupancy_map.shape
    inside = (xs >= 0) & (xs < w) & (zs >= 0) & (zs < h)
    if not np.all(inside):
        return False
    return np.all(occupancy_map[zs, xs] == 0)


def _nearest_free(node: Tuple[int, int], occupancy_map: np.ndarray, max_radius: int = 40) -> Tuple[int, int]:
    if _is_free(occupancy_map, node):
        return node
    x0, z0 = node
    h, w = occupancy_map.shape
    for r in range(1, max_radius + 1):
        x_min, x_max = max(0, x0 - r), min(w - 1, x0 + r)
        z_min, z_max = max(0, z0 - r), min(h - 1, z0 + r)
        for x in range(x_min, x_max + 1):
            for z in (z_min, z_max):
                if _is_free(occupancy_map, (x, z)):
                    return (x, z)
        for z in range(z_min, z_max + 1):
            for x in (x_min, x_max):
                if _is_free(occupancy_map, (x, z)):
                    return (x, z)
    raise ValueError(f"Cannot find free space near point {node}.")


def plan_path(
    start: Tuple[int, int],
    goal: Tuple[int, int],
    occupancy_map: np.ndarray,
    max_iter: int = 12000,
    step_size: int = 10,
    goal_sample_rate: float = 0.2,
    goal_radius: int = 12,
) -> List[Tuple[int, int]]:
    start = _nearest_free(start, occupancy_map)
    goal = _nearest_free(goal, occupancy_map)

    nodes = [start]
    parents = [-1]
    rng = random.Random(42)
    h, w = occupancy_map.shape

    for _ in range(max_iter):
        if rng.random() < goal_sample_rate:
            sample = goal
        else:
            sample = (rng.randint(0, w - 1), rng.randint(0, h - 1))

        dists = [((n[0] - sample[0]) ** 2 + (n[1] - sample[1]) ** 2) for n in nodes]
        nearest_idx = int(np.argmin(dists))
        nx, nz = nodes[nearest_idx]
        dx, dz = sample[0] - nx, sample[1] - nz
        length = float(np.hypot(dx, dz))
        if length < 1e-6:
            continue
        scale = min(step_size, length) / length
        new_node = (int(round(nx + dx * scale)), int(round(nz + dz * scale)))

        if not _is_free(occupancy_map, new_node):
            continue
        if not _line_collision_free(nodes[nearest_idx], new_node, occupancy_map):
            continue

        nodes.append(new_node)
        parents.append(nearest_idx)
        new_idx = len(nodes) - 1

        if np.hypot(new_node[0] - goal[0], new_node[1] - goal[1]) <= goal_radius:
            if _line_collision_free(new_node, goal, occupancy_map):
                nodes.append(goal)
                parents.append(new_idx)
                break

    if parents[-1] == -1 or nodes[-1] != goal:
        return []

    # Backtrack
    path = []
    idx = len(nodes) - 1
    while idx != -1:
        path.append(nodes[idx])
        idx = parents[idx]
    path.reverse()
    return path


def visualize_path(map_img: np.ndarray, occupancy_map: np.ndarray, path: List[Tuple[int, int]], start, goal):
    plt.figure(figsize=(10, 7))
    plt.imshow(map_img)
    # occupancy map: occupied=1, free=0
    occ_vis = np.ma.masked_where(occupancy_map == 0, occupancy_map)
    plt.imshow(occ_vis, cmap="gray", alpha=0.28)

    path_np = np.array(path)
    plt.plot(path_np[:, 0], path_np[:, 1], "r-", linewidth=2.0, label="RRT path")
    plt.scatter([start[0]], [start[1]], c="lime", s=80, marker="o", label="start")
    plt.scatter([goal[0]], [goal[1]], c="magenta", s=90, marker="*", label="goal")
    plt.title("Planned Path on 2D Semantic Map")
    plt.axis("equal")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


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
    map_img, occupancy_map, x_min, z_min, map_resolution = load_and_filter_map(POINT_CLOUD_DATA, COLOR_DATA)


    print("=== Step 2: Selecting Agent Start and Goal Positions ===")
    start = select_start(map_img)
    goal_prompt, goal = pick_goal(map_img)
    print(f"Goal pixel selected at coordinates: {goal}")


    print("=== Step 3: Executing Path Planning (RRT) ===")
    # =============== TODO 2 ===============
    # implement RRT path planner in plan_path()
    path = plan_path(start, goal, occupancy_map)
    if not path:
        print("Planner could not find a path.")
        sys.exit(1)


    print("=== Step 4: Visualizing the Planned Path ===")
    # =============== TODO 3 ===============
    # Visualize the planned path over the map
    visualize_path(map_img, occupancy_map, path, start, goal)


    print("=== Step 5: Translating Path to Habitat Simulator ===")
    # =============== TODO 4 ===============
    # Convert pixel path to world coordinates
    # world_path is a list of tuples(float, float) representing waypoints in world coordinates
    world_path = [(x_min + px * map_resolution, z_min + pz * map_resolution) for (px, pz) in path]

    run_in_sim(world_path[0], world_path, goal_prompt)


if __name__ == "__main__":
    main()
