import math
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


Point = Tuple[int, int]


@dataclass
class Node:
    point: Point
    parent: Optional[int]


def _distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _steer(src: Point, dst: Point, step_size: int) -> Point:
    dist = _distance(src, dst)
    if dist <= step_size:
        return dst
    ratio = step_size / dist
    return int(round(src[0] + (dst[0] - src[0]) * ratio)), int(round(src[1] + (dst[1] - src[1]) * ratio))


def _collision_free(a: Point, b: Point, occupancy_map: np.ndarray) -> bool:
    h, w = occupancy_map.shape
    dist = max(1, int(math.ceil(_distance(a, b))))
    xs = np.linspace(a[0], b[0], dist + 1).round().astype(np.int32)
    ys = np.linspace(a[1], b[1], dist + 1).round().astype(np.int32)
    if np.any(xs < 0) or np.any(xs >= w) or np.any(ys < 0) or np.any(ys >= h):
        return False
    return not np.any(occupancy_map[ys, xs])


def _sample_free(occupancy_map: np.ndarray) -> Point:
    ys, xs = np.where(~occupancy_map)
    idx = random.randrange(len(xs))
    return int(xs[idx]), int(ys[idx])


def _reconstruct(nodes: List[Node], idx: int) -> List[Point]:
    path: List[Point] = []
    while idx is not None:
        path.append(nodes[idx].point)
        idx = nodes[idx].parent
    path.reverse()
    return path


def _shortcut_path(path: List[Point], occupancy_map: np.ndarray) -> List[Point]:
    if len(path) <= 2:
        return path
    shortened = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not _collision_free(path[i], path[j], occupancy_map):
            j -= 1
        shortened.append(path[j])
        i = j
    return shortened


def plan_path(
    start: Point,
    goal: Point,
    occupancy_map: np.ndarray,
    max_iters: int = 8000,
    step_size: int = 12,
    goal_sample_rate: float = 0.15,
    goal_radius: int = 14,
) -> List[Point]:
    """Plan a pixel-coordinate path with RRT. occupancy_map True cells are blocked."""

    if occupancy_map[start[1], start[0]]:
        raise ValueError(f"Start {start} is not in free space.")
    if occupancy_map[goal[1], goal[0]]:
        raise ValueError(f"Goal {goal} is not in free space.")

    nodes = [Node(start, None)]
    for _ in range(max_iters):
        sample = goal if random.random() < goal_sample_rate else _sample_free(occupancy_map)
        nearest_idx = min(range(len(nodes)), key=lambda i: _distance(nodes[i].point, sample))
        new_point = _steer(nodes[nearest_idx].point, sample, step_size)
        if new_point == nodes[nearest_idx].point:
            continue
        if not _collision_free(nodes[nearest_idx].point, new_point, occupancy_map):
            continue

        nodes.append(Node(new_point, nearest_idx))
        new_idx = len(nodes) - 1
        if _distance(new_point, goal) <= goal_radius and _collision_free(new_point, goal, occupancy_map):
            nodes.append(Node(goal, new_idx))
            return _shortcut_path(_reconstruct(nodes, len(nodes) - 1), occupancy_map)

    return []


def visualize_path(map_img: np.ndarray, path: List[Point], start: Point, goal: Point, out_path: str = "rrt_path.png"):
    canvas = (np.clip(map_img, 0.0, 1.0) * 255).astype(np.uint8)
    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    for a, b in zip(path[:-1], path[1:]):
        cv2.line(canvas, a, b, (0, 0, 255), 2, lineType=cv2.LINE_AA)
    cv2.circle(canvas, start, 5, (0, 255, 0), -1)
    cv2.circle(canvas, goal, 5, (255, 0, 0), -1)
    cv2.imwrite(out_path, canvas)
    cv2.imshow("RRT Path", canvas)
    cv2.waitKey(1)
    return canvas
