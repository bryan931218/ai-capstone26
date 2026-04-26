import random
import sys
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

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


def plan_path(
    start: Tuple[int, int],
    goal: Tuple[int, int],
    occupancy_map: np.ndarray,
    max_iter: int = 10000,
    step_size: int = 5,
) -> List[Tuple[int, int]]:
    """RRT path planner in pixel space. Returns path as list of (col, row) tuples."""
    # =============== TODO 2 ===============
    from collections import deque

    H, W = occupancy_map.shape

    def is_free(c: float, r: float) -> bool:
        ci, ri = int(round(c)), int(round(r))
        return 0 <= ci < W and 0 <= ri < H and occupancy_map[ri, ci] == 0

    def collision_free(p1: Tuple, p2: Tuple) -> bool:
        dist = float(np.linalg.norm(np.array(p2, float) - np.array(p1, float)))
        n = max(int(np.ceil(dist)) + 1, 2)  # ≤1 px spacing — guaranteed to hit 1-px walls
        for t in np.linspace(0, 1, n):
            c = p1[0] + t * (p2[0] - p1[0])
            r = p1[1] + t * (p2[1] - p1[1])
            if not is_free(c, r):
                return False
        return True

    # BFS from start to find all reachable free cells (8-connected)
    if not is_free(start[0], start[1]):
        free_all = np.argwhere(occupancy_map == 0)
        if len(free_all) == 0:
            return []
        d = np.sqrt((free_all[:, 1] - start[0]) ** 2 + (free_all[:, 0] - start[1]) ** 2)
        start = (int(free_all[np.argmin(d), 1]), int(free_all[np.argmin(d), 0]))

    reachable = np.zeros((H, W), dtype=bool)
    q: deque = deque([start])
    reachable[start[1], start[0]] = True
    while q:
        c, r = q.popleft()
        for dc, dr in [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]:
            nc, nr = c + dc, r + dr
            if 0 <= nc < W and 0 <= nr < H and not reachable[nr, nc] and occupancy_map[nr, nc] == 0:
                reachable[nr, nc] = True
                q.append((nc, nr))

    reachable_cells = np.argwhere(reachable)  # [[row, col], ...]
    if len(reachable_cells) == 0:
        return []

    def snap_to_reachable(pt: Tuple[int, int]) -> Tuple[int, int]:
        """Find the nearest reachable cell to pt with some open space around it.
        Prefers cells that have fewer obstacle neighbours so RRT can reach them."""
        if reachable[pt[1], pt[0]]:
            return pt
        dists = np.sqrt((reachable_cells[:, 1] - pt[0]) ** 2 + (reachable_cells[:, 0] - pt[1]) ** 2)
        # Among candidates within 2× the nearest distance, prefer cells in open areas
        min_d = dists.min()
        candidates = np.where(dists <= max(min_d * 2, 20))[0]
        # Score = distance + penalty for obstacle-dense neighbourhood
        best_score, best_idx = float("inf"), int(np.argmin(dists))
        k_size = 5
        for ci in candidates:
            cr, cc = int(reachable_cells[ci, 0]), int(reachable_cells[ci, 1])
            r0, r1 = max(0, cr - k_size), min(H, cr + k_size + 1)
            c0, c1 = max(0, cc - k_size), min(W, cc + k_size + 1)
            obs_density = occupancy_map[r0:r1, c0:c1].mean()
            score = dists[ci] + obs_density * 30
            if score < best_score:
                best_score, best_idx = score, ci
        return int(reachable_cells[best_idx, 1]), int(reachable_cells[best_idx, 0])

    goal = snap_to_reachable(goal)

    nodes = np.zeros((max_iter + 2, 2), dtype=float)
    nodes[0] = [start[0], start[1]]
    parents = np.full(max_iter + 2, -1, dtype=int)
    n_nodes = 1

    for _ in range(max_iter):
        # 10% goal bias, else sample a random free cell
        if random.random() < 0.1:
            sample = np.array([goal[0], goal[1]], dtype=float)
        else:
            fc_idx = random.randint(0, len(reachable_cells) - 1)
            sample = np.array([reachable_cells[fc_idx, 1], reachable_cells[fc_idx, 0]], dtype=float)

        # Nearest node in tree
        dists = np.linalg.norm(nodes[:n_nodes] - sample, axis=1)
        nearest_idx = int(np.argmin(dists))
        nearest = nodes[nearest_idx]

        # Steer toward sample by at most step_size
        direction = sample - nearest
        dist = float(np.linalg.norm(direction))
        if dist == 0:
            continue
        new_node = nearest + direction / dist * min(dist, float(step_size))

        if not is_free(new_node[0], new_node[1]):
            continue
        if not collision_free(tuple(nearest), tuple(new_node)):
            continue

        nodes[n_nodes] = new_node
        parents[n_nodes] = nearest_idx
        n_nodes += 1

        # Check if goal is directly reachable from new node
        goal_arr = np.array([goal[0], goal[1]], dtype=float)
        if np.linalg.norm(new_node - goal_arr) < step_size * 2:
            if collision_free(tuple(new_node), goal):
                nodes[n_nodes] = goal_arr
                parents[n_nodes] = n_nodes - 1
                n_nodes += 1
                # Backtrack to recover path
                path: List[Tuple[int, int]] = []
                idx = n_nodes - 1
                while idx != -1:
                    path.append((int(round(nodes[idx, 0])), int(round(nodes[idx, 1]))))
                    idx = int(parents[idx])
                return path[::-1]

    return []


def visualize_path(
    map_img: np.ndarray,
    path: List[Tuple[int, int]],
    start: Tuple[int, int],
    goal: Tuple[int, int],
    goal_name: str = "",
) -> None:
    """Visualize the planned RRT path over the 2D semantic map using matplotlib."""
    # =============== TODO 3 ===============
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(map_img)

    if path:
        cols = [p[0] for p in path]
        rows = [p[1] for p in path]
        ax.plot(cols, rows, color="red", linewidth=2, label="RRT Path")

    ax.plot(start[0], start[1], "go", markersize=10, label="Start")
    ax.plot(goal[0], goal[1], "r*", markersize=14, label=f"Goal ({goal_name})")
    ax.legend(loc="upper right")
    ax.set_title("RRT Planned Path")
    fig.tight_layout()
    plt.savefig("path_visualization.png", dpi=150)
    plt.show()


def main():
    """Entry point."""

    print("=== Step 1: Processing the 3D Map ===")
    # =============== TODO 1-2 ===============
    map_img, occupancy_map, x_min, z_min, resolution = load_and_filter_map(POINT_CLOUD_DATA, COLOR_DATA)
    print("Saved semantic_map.png")

    print("=== Step 2: Selecting Agent Start and Goal Positions ===")
    start = select_start(map_img)
    goal_prompt, goal = pick_goal(map_img)
    print(f"Goal pixel selected at coordinates: {goal}")

    print("=== Step 3: Executing Path Planning (RRT) ===")
    path = plan_path(start, goal, occupancy_map)
    if not path:
        print("Planner could not find a path.")
        sys.exit(1)

    print("=== Step 4: Visualizing the Planned Path ===")
    visualize_path(map_img, path, start, goal, goal_prompt)

    print("=== Step 5: Translating Path to Habitat Simulator ===")
    # =============== TODO 4 ===============
    # Pixel (col, row) → world (x, z):
    #   x_world = row / resolution + x_min
    #   z_world = col / resolution + z_min
    world_path = [(row / resolution + x_min, col / resolution + z_min) for (col, row) in path]

    run_in_sim(world_path[0], world_path, goal_prompt)


if __name__ == "__main__":
    main()
