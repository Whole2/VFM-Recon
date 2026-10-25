#!/usr/bin/env python3
# coding: utf-8
"""
compute_global_scale.py

只计算每个场景的 global scale，并将结果保存在场景文件夹下的 global_scale.txt 中。
"""

import os
import glob
import numpy as np
import cv2
import torch

# lightglue / superpoint 依赖
from lightglue import LightGlue, SuperPoint

# ---------------- Helper functions ----------------

def project_points(points, proj_mat):
    points = points.reshape(-1, 3)
    pcd_homo = cv2.convertPointsToHomogeneous(points)[:, 0]
    proj_0 = (proj_mat @ pcd_homo.T).T
    proj_z = proj_0[:, -1:]
    proj = proj_0[:, :-1] / proj_z
    return proj, proj_z

def remove_negative_depth(point_cloud, proj_matrices):
    # point_cloud: (N,3)
    _, proj_z0 = project_points(point_cloud, proj_matrices[0])
    _, proj_z1 = project_points(point_cloud, proj_matrices[1])
    inliers = (proj_z0 > 0) & (proj_z1 > 0)
    point_cloud = point_cloud[inliers[:, 0]]
    return point_cloud

def calibrate_intra_submap_scale(submaps):
    """
    输入: submaps: list of submap, 每个 submap 是 frames 列表；
           每个 frame = (image_path, depth, pose, intrinsic)
    返回: array of estimated scales (float) per submap
    """
    estimated_scalse = []

    # 初始化模型 (只加载一次)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
    matcher = LightGlue(features='superpoint').eval().to(device)

    for map_id, frames in enumerate(submaps):
        valid_ratios = []
        print(f"--- Calibrating Submap {map_id} (Total frames: {len(frames)}) ---")

        for i in range(len(frames) - 1):
            ref_frame_idx = i
            target_frame_idx = i + 1

            try:
                pair_frames = [frames[ref_frame_idx], frames[target_frame_idx]]
                img_paths = [f[0] for f in pair_frames]
                depths = [f[1] for f in pair_frames]
                poses = [f[2] for f in pair_frames]
                intrinsics = [f[3] for f in pair_frames]

                proj_matrices = [(intr @ np.linalg.inv(p))[:3] for (intr, p) in zip(intrinsics, poses)]

                images_tensor = []
                for imp in img_paths:
                    im = cv2.imread(imp, -1)
                    if im is None:
                        raise ValueError(f"Image not found: {imp}")
                    # to RGB
                    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                    im_tensor = torch.from_numpy(im).float().to(device).permute(2, 0, 1) / 255.
                    images_tensor.append(im_tensor)

                # extract features
                with torch.no_grad():
                    feat0 = extractor.extract(images_tensor[0], resize=None)
                    feat1 = extractor.extract(images_tensor[1], resize=None)
                    matches_raw = matcher({'image0': feat0, 'image1': feat1})

                # get keypoints and matches
                kpts0 = feat0["keypoints"][0]  # tensor (N0,2)
                kpts1 = feat1["keypoints"][0]  # tensor (N1,2)
                matches = matches_raw["matches"][0]  # (M,2) or (-1,-1) for unmatched

                # Filter out unmatched (if library uses -1)
                # Depending on LightGlue return format, matches may contain (-1,-1) rows; handle that.
                matches_np = matches.cpu().numpy()
                valid_mask = (matches_np[:,0] >= 0) & (matches_np[:,1] >= 0)
                if valid_mask.sum() == 0:
                    continue
                matches = matches[valid_mask]
                m_kpts0 = kpts0[matches[..., 0].long()]
                m_kpts1 = kpts1[matches[..., 1].long()]

                if len(m_kpts0) < 10:
                    continue

                # triangulate
                pts0 = m_kpts0.cpu().numpy().T.astype(np.float32)
                pts1 = m_kpts1.cpu().numpy().T.astype(np.float32)
                points_h = cv2.triangulatePoints(proj_matrices[0], proj_matrices[1], pts0, pts1)
                points = cv2.convertPointsFromHomogeneous(points_h.T)
                point_cloud = points[:, 0]  # (N,3)

                # remove negative depth
                point_cloud = remove_negative_depth(point_cloud, proj_matrices)
                if point_cloud.shape[0] < 5:
                    continue

                depth_im = depths[0]
                P = proj_matrices[0]
                pcd_homo = np.hstack([point_cloud, np.ones((len(point_cloud), 1))])
                proj_0 = (P @ pcd_homo.T).T
                proj_depth = proj_0[:, 2]
                proj_uv = proj_0[:, :2] / proj_depth[:, None]

                current_pair_ratios = []
                for k in range(len(point_cloud)):
                    u, v = proj_uv[k]
                    # nearest pixel sampling (same as original)
                    ui = int(np.round(u))
                    vi = int(np.round(v))
                    if (0 <= ui < depth_im.shape[1] and 0 <= vi < depth_im.shape[0]):
                        d_val = depth_im[vi, ui]
                        if d_val > 0.1:
                            current_pair_ratios.append(proj_depth[k] / d_val)

                if len(current_pair_ratios) > 0:
                    valid_ratios.append(np.median(current_pair_ratios))

            except Exception as e:
                # 忽略单张对计算错误，继续下一对
                # 可以打印调试信息（但避免过多输出）
                # print(f"pair error: {e}")
                continue

        if len(valid_ratios) == 0:
            print(f"WARNING: Submap {map_id} failed to calibrate. Defaulting scale to 1.0.")
            estimated_scalse.append(1.0)
        else:
            final_submap_scale = np.median(valid_ratios)
            print(f"Submap {map_id} Final Scale: {final_submap_scale:.4f}")
            estimated_scalse.append(final_submap_scale)

    return np.array(estimated_scalse)

# ---------------- Main logic ----------------

def compute_and_save_global_scale(scene_name, base_path):
    print(f"\nProcessing Scene: {scene_name}")

    path = os.path.join(base_path, scene_name)
    if not os.path.exists(path):
        print(f"Error: Scene path {path} does not exist. Skipping.")
        return

    images = sorted(glob.glob(os.path.join(path, "images/*.jpg")))
    if len(images) == 0:
        print(f"Error: No images found in {path}. Skipping.")
        return

    poses_path = os.path.join(path, "poses.npy")
    if not os.path.exists(poses_path):
        print(f"Error: poses.npy not found in {path}. Skipping.")
        return
    poses = np.load(poses_path)

    intrinsic_path = os.path.join(path, "intrinsic.txt")
    if not os.path.exists(intrinsic_path):
        print(f"Error: intrinsic.txt not found in {path}. Skipping.")
        return
    K = np.loadtxt(intrinsic_path)[None].repeat(len(poses), axis=0)

    frame_window = 16
    n_submaps = len(images) // (frame_window - 1) + 1

    # Build global_frames: first frame of each submap
    global_depth_dir = os.path.join(path, "vggt_depth_global")
    global_frames = []

    for i in range(n_submaps):
        index = i * (frame_window - 1)
        if index >= len(images):
            break
        g_depth_path = os.path.join(global_depth_dir, f"depth_{index:06d}.png")
        if not os.path.exists(g_depth_path):
            # skip missing global depth for this keyframe
            continue
        # read and normalize (same as original)
        g_depth = cv2.imread(g_depth_path, cv2.IMREAD_UNCHANGED)
        if g_depth is None:
            continue
        g_depth = g_depth / 5000.0
        g_depth = cv2.resize(g_depth, (640, 480)).astype(np.float32)

        pose = poses[index]
        image_path = images[index]
        intrinsic = K[index]
        global_frames.append((image_path, g_depth, pose, intrinsic))

    if not global_frames:
        print("Error: No global frames found. Cannot compute global scale.")
        # write fallback
        out_file = os.path.join(path, "global_scale.txt")
        with open(out_file, "w") as f:
            f.write("1.0\n")
        print(f"Wrote fallback global_scale=1.0 to {out_file}")
        return

    print("--- Calibrating Global Scale ---")
    try:
        global_scale_array = calibrate_intra_submap_scale([global_frames])
        global_scale = float(global_scale_array[0])
        print(f"Computed Global Scale: {global_scale:.6f}")
    except Exception as e:
        print(f"Error during global calibration: {e}")
        global_scale = 1.0
        print("Defaulting global scale to 1.0")

    # save to file in scene folder
    out_file = os.path.join(path, "global_scale.txt")
    try:
        with open(out_file, "w") as f:
            f.write(f"{global_scale:.6f}\n")
        print(f"Saved global scale to {out_file}")
    except Exception as e:
        print(f"Failed to write global scale file: {e}")

# ---------------- Configuration ----------------

BASE_PATH = "/mnt/archive/users/mingyuhang/organized_datasets/preprocessed/TanksAndTemples"

SCENE_NAMES = [
    "Barn",
    "Meetingroom",
    "Ignatius",
    "Courthouse",
    "Truck",
    "Caterpillar"
]

if __name__ == "__main__":
    for scene in SCENE_NAMES:
        compute_and_save_global_scale(scene, BASE_PATH)
    print("\nAll requested scenes processed.")
