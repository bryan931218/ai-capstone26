import os
import glob
import numpy as np
import open3d as o3d
import argparse
from scipy.spatial.transform import Rotation as R
import time

# ---------- Camera Intrinsics (Resolution 512x512, FOV 90) ----------
IMG_W, IMG_H = 512, 512
FOV = np.deg2rad(90.0)
FX = (IMG_W / 2.0) / np.tan(FOV / 2.0)
FY = (IMG_H / 2.0) / np.tan(FOV / 2.0)
CX, CY = IMG_W / 2.0, IMG_H / 2.0
DEPTH_SCALE = 1000.0


def sorted_frame_files(pattern):
    files = glob.glob(pattern)
    def frame_index(path):
        stem = os.path.splitext(os.path.basename(path))[0]
        return int(stem) if stem.isdigit() else stem

    return sorted(files, key=frame_index)


def decode_depth_to_meters(depth_image):
    """
    Decode saved depth image into metric depth (meters).
    Supports:
    - uint16 millimeter depth (recommended)
    - uint8 visualization depth saved by old load.py (0~10m mapped to 0~255)
    """
    depth_np = np.asarray(depth_image)
    if depth_np.dtype == np.uint16:
        return depth_np.astype(np.float32) / DEPTH_SCALE
    if depth_np.dtype == np.uint8:
        return depth_np.astype(np.float32) / 255.0 * 10.0
    return depth_np.astype(np.float32)


def depth_image_to_point_cloud(rgb_image, depth_image):
    """
    TASK 1: Geometric Unprojection
    Convert depth pixels (u, v, d) into 3D camera-frame points (x, y, z).
    """
    rgb = np.asarray(rgb_image, dtype=np.float32)
    depth = decode_depth_to_meters(depth_image)

    v_coords, u_coords = np.indices(depth.shape)
    # Ignore invalid and very far depth points to reduce outliers.
    valid = (depth > 1e-3) & (depth < 8.0)

    z = depth[valid]
    u = u_coords[valid]
    v = v_coords[valid]

    # Habitat camera faces -Z in camera frame.
    z = -z
    x = (u - CX) * z / FX
    y = (v - CY) * z / FY

    points_3d = np.stack([x, y, z], axis=1)
    colors_norm = rgb[valid] / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    pcd.colors = o3d.utility.Vector3dVector(colors_norm)
    return pcd


def preprocess_point_cloud(pcd, voxel_size):
    """
    Pre-processing: Voxelization and Normal Estimation.
    """
    pts = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors)
    if len(pts) == 0:
        return o3d.geometry.PointCloud(), o3d.pipelines.registration.Feature()

    # Custom voxel downsampling (no Open3D voxel_down_sample):
    # group points by voxel index and average xyz/rgb per voxel.
    voxel_idx = np.floor(pts / voxel_size).astype(np.int64)
    uniq, inv = np.unique(voxel_idx, axis=0, return_inverse=True)

    counts = np.bincount(inv).astype(np.float64)
    sum_pts = np.zeros((len(uniq), 3), dtype=np.float64)
    sum_cols = np.zeros((len(uniq), 3), dtype=np.float64)
    np.add.at(sum_pts, inv, pts)
    np.add.at(sum_cols, inv, cols)

    down_pts = sum_pts / counts[:, None]
    down_cols = sum_cols / counts[:, None]

    pcd_down = o3d.geometry.PointCloud()
    pcd_down.points = o3d.utility.Vector3dVector(down_pts)
    pcd_down.colors = o3d.utility.Vector3dVector(down_cols)

    radius_normal = voxel_size * 2.0
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )

    radius_feature = voxel_size * 5.0
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100),
    )
    return pcd_down, pcd_fpfh


def my_local_icp_algorithm(source_pcd, target_pcd, initial_transform):
    """
    TASK 2 (BONUS): Custom ICP placeholder.
    """
    result = o3d.pipelines.registration.RegistrationResult()
    result.transformation = initial_transform.copy()
    return result


def local_icp_algorithm(source_down, target_down, trans_init, threshold):
    """
    TASK 2: Open3D ICP Implementation (REQUIRED)
    """
    return o3d.pipelines.registration.registration_icp(
        source_down,
        target_down,
        threshold,
        trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    )


def visualize_and_evaluate(reconstructed_pcd, predicted_cam_poses, gt_poses, args):
    """
    TASK 3: Evaluation & Visualization
    """
    pred_xyz = np.array([pose[:3, 3] for pose in predicted_cam_poses])

    if len(gt_poses) > 0:
        n = min(len(pred_xyz), len(gt_poses))
        gt_xyz = gt_poses[:n, :3, 3]
        pred_xyz = pred_xyz[:n]

        lines = [[i, i + 1] for i in range(n - 1)]

        pred_line = o3d.geometry.LineSet()
        pred_line.points = o3d.utility.Vector3dVector(pred_xyz)
        pred_line.lines = o3d.utility.Vector2iVector(lines)
        pred_line.colors = o3d.utility.Vector3dVector(np.tile([1.0, 0.0, 0.0], (len(lines), 1)))

        gt_line = o3d.geometry.LineSet()
        gt_line.points = o3d.utility.Vector3dVector(gt_xyz)
        gt_line.lines = o3d.utility.Vector2iVector(lines)
        gt_line.colors = o3d.utility.Vector3dVector(np.tile([0.0, 0.0, 0.0], (len(lines), 1)))

        mean_l2_error = np.linalg.norm(pred_xyz - gt_xyz, axis=1).mean()
        draw_items = [reconstructed_pcd, pred_line, gt_line]
    else:
        mean_l2_error = 0.0
        draw_items = [reconstructed_pcd]

    print(f"Mean L2 distance: {mean_l2_error:.6f} meters")

    o3d.visualization.draw_geometries(
        draw_items, window_name=f"Floor {args.floor} Reconstruction"
    )
    return mean_l2_error


def reconstruct(args):
    voxel_size = 0.12
    icp_threshold = 0.30
    keyframe_update_interval = 5

    rgb_dir = os.path.join(args.data_root, "rgb")
    depth_dir = os.path.join(args.data_root, "depth")

    rgb_files = sorted_frame_files(os.path.join(rgb_dir, "*.png"))
    depth_files = sorted_frame_files(os.path.join(depth_dir, "*.png"))

    gt_pose_path = os.path.join(args.data_root, "GT_pose.npy")
    gt_poses = []
    if os.path.exists(gt_pose_path):
        gt_data = np.load(gt_pose_path)
        for p in gt_data:
            mat = np.eye(4)
            mat[:3, :3] = R.from_quat([p[4], p[5], p[6], p[3]]).as_matrix()
            mat[:3, 3] = [p[0], p[1], p[2]]
            gt_poses.append(mat)
        gt_poses = np.stack(gt_poses)

    if not rgb_files or not depth_files:
        raise RuntimeError(f"No RGB-D frames found in {args.data_root}")

    rgb0 = o3d.io.read_image(rgb_files[0])
    depth0 = o3d.io.read_image(depth_files[0])
    first_pcd = depth_image_to_point_cloud(rgb0, depth0)

    if len(gt_poses) > 0:
        init_pose = gt_poses[0].copy()
    else:
        init_pose = np.eye(4)

    camera_poses = [init_pose]
    accumulated_pcd = o3d.geometry.PointCloud(first_pcd)
    accumulated_pcd.transform(init_pose)
    map_down, map_fpfh = preprocess_point_cloud(accumulated_pcd, voxel_size)
    accepted_since_ref_update = 0

    frame_stride = max(1, int(args.frame_stride))
    for i in range(1, min(len(rgb_files), len(depth_files)), frame_stride):
        print(f"Processing Frame {i}...")

        rgb = o3d.io.read_image(rgb_files[i])
        depth = o3d.io.read_image(depth_files[i])
        cur_pcd = depth_image_to_point_cloud(rgb, depth)
        cur_down, cur_fpfh = preprocess_point_cloud(cur_pcd, voxel_size)

        if len(map_down.points) < 100:
            camera_poses.append(camera_poses[-1].copy())
            print(f"Warning: map too sparse at frame {i}, skip.")
            continue

        # Pipeline order (required):
        # Unprojection -> Voxelization -> Global RANSAC -> Local ICP
        distance_threshold = voxel_size * 1.5
        result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            cur_down,
            map_down,
            cur_fpfh,
            map_fpfh,
            mutual_filter=True,
            max_correspondence_distance=distance_threshold,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            ransac_n=4,
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(20000, 0.999),
        )

        if args.version == "my_icp":
            result_icp = my_local_icp_algorithm(cur_down, map_down, result_ransac.transformation)
        else:
            result_icp = local_icp_algorithm(
                cur_down,
                map_down,
                result_ransac.transformation,
                icp_threshold,
            )

        is_good_match = result_icp.fitness >= 0.20
        if not is_good_match:
            camera_poses.append(camera_poses[-1].copy())
            print(
                f"Warning: reject frame {i} (ICP fitness={result_icp.fitness:.4f}, rmse={result_icp.inlier_rmse:.4f})"
            )
            continue

        world_t_cur = result_icp.transformation
        camera_poses.append(world_t_cur)

        cur_world = o3d.geometry.PointCloud(cur_pcd)
        cur_world.transform(world_t_cur)
        accumulated_pcd += cur_world
        accepted_since_ref_update += 1

        if accepted_since_ref_update >= keyframe_update_interval:
            map_down, map_fpfh = preprocess_point_cloud(accumulated_pcd, voxel_size)
            accepted_since_ref_update = 0

    points = np.asarray(accumulated_pcd.points)
    colors = np.asarray(accumulated_pcd.colors)
    if len(points) > 0:
        # Habitat world up-axis is Y; remove top 5% points as ceiling.
        y_thresh = np.percentile(points[:, 1], 95)
        keep = points[:, 1] <= y_thresh
        filtered = o3d.geometry.PointCloud()
        filtered.points = o3d.utility.Vector3dVector(points[keep])
        filtered.colors = o3d.utility.Vector3dVector(colors[keep])
        accumulated_pcd = filtered.voxel_down_sample(voxel_size)

    return accumulated_pcd, camera_poses, gt_poses


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--floor', type=int, default=1)
    parser.add_argument('-v', '--version', type=str, default='open3d', help='open3d or my_icp')
    parser.add_argument('--frame_stride', type=int, default=1, help='Process every N-th frame')
    args = parser.parse_args()

    args.data_root = (
        "data_collection/first_floor/" if args.floor == 1 else "data_collection/second_floor/"
    )

    start_time = time.time()
    result_pcd, pred_poses, gt_poses = reconstruct(args)

    print(f"Total execution time: {time.time() - start_time:.2f}s")
    visualize_and_evaluate(result_pcd, pred_poses, gt_poses, args)
