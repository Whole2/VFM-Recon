#!/usr/bin/env python3
# coding: utf-8
"""
compute_submap_scales_strict.py

为每个场景计算每个 submap 的 scale（每个 submap 使用多对帧以减少噪声），
并将结果写入场景目录下的 submap_scale.txt （每行 "submap_index scale"）。
如果某个 submap 估计失败（有效对数 < min_valid_pairs），脚本会抛出异常并终止运行。
"""

import os
import glob
import numpy as np
import cv2
import torch
from lightglue import LightGlue, SuperPoint

# ---------------- util ----------------

def project_points(points, proj_mat):
    points = points.reshape(-1, 3)
    pcd_homo = cv2.convertPointsToHomogeneous(points)[:, 0]
    proj_0 = (proj_mat @ pcd_homo.T).T
    proj_z = proj_0[:, -1:]
    proj = proj_0[:, :-1] / proj_z
    return proj, proj_z

def remove_negative_depth(point_cloud, proj_matrices):
    _, proj_z0 = project_points(point_cloud, proj_matrices[0])
    _, proj_z1 = project_points(point_cloud, proj_matrices[1])
    inliers = (proj_z0 > 0) & (proj_z1 > 0)
    if inliers.shape[0] == 0:
        return np.zeros((0,3), dtype=np.float32)
    point_cloud = point_cloud[inliers[:, 0]]
    return point_cloud

# ---------------- scale estimation (multi-pair, strict) ----------------

def calibrate_intra_submap_scale(submaps, max_pairs=10, min_valid_pairs=3, device='cuda'):
    """
    对每个 submap 使用多对帧估计 scale。
    参数:
      - submaps: list of submap, 每个 submap 是 frames 列表；
                 每个 frame = (image_path, depth, pose, intrinsic)
      - max_pairs: 每个 submap 最多试验的帧对数量（优先使用连续对 0-1,1-2,...）
      - min_valid_pairs: 至少要有多少个成功的帧对（有有效 ratio）才视为 submap 成功
      - device: 'cuda' 或 'cpu'
    返回:
      - np.array of scales per submap
    如果任一 submap 失败（有效对 < min_valid_pairs），抛出 RuntimeError（并终止程序）。
    """
    estimated_scales = []

    # 初始化模型（只加载一次）
    device = device if torch.cuda.is_available() and device == 'cuda' else 'cpu'
    extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
    matcher = LightGlue(features='superpoint').eval().to(device)

    for map_id, frames in enumerate(submaps):
        print(f"[Scene] Calibrating submap {map_id}, frames={len(frames)} (max_pairs={max_pairs}, min_valid_pairs={min_valid_pairs})")

        if len(frames) < 2:
            raise RuntimeError(f"Submap {map_id} has fewer than 2 frames -> cannot estimate scale.")

        # 选择帧对策略：使用连续帧对 (i, i+1)，最多 max_pairs 对。
        pair_indices = list(range(0, max(1, len(frames)-1)))
        if len(pair_indices) > max_pairs:
            # 取等间隔的 max_pairs 个起始索引
            idxs = np.linspace(0, len(pair_indices)-1, max_pairs).astype(int)
            pair_indices = [pair_indices[i] for i in idxs.tolist()]

        per_pair_medians = []
        successful_pairs = 0

        # 对每个选定的帧对进行估计
        for start_idx in pair_indices:
            try:
                ref = frames[start_idx]
                tar = frames[start_idx + 1]

                img_path0, depth0, pose0, K0 = ref
                img_path1, depth1, pose1, K1 = tar

                # 读取并检查图像
                im0 = cv2.imread(img_path0, -1)
                im1 = cv2.imread(img_path1, -1)
                if im0 is None or im1 is None:
                    print(f"  Pair ({start_idx},{start_idx+1}): image missing, skip pair")
                    continue
                im0 = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
                im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)

                # 准备tensor
                t0 = torch.from_numpy(im0).float().to(device).permute(2,0,1) / 255.0
                t1 = torch.from_numpy(im1).float().to(device).permute(2,0,1) / 255.0

                with torch.no_grad():
                    feat0 = extractor.extract(t0, resize=None)
                    feat1 = extractor.extract(t1, resize=None)
                    matches_raw = matcher({'image0': feat0, 'image1': feat1})

                kpts0 = feat0["keypoints"][0]   # (N0,2)
                kpts1 = feat1["keypoints"][0]   # (N1,2)
                matches = matches_raw["matches"][0]  # (M,2) or with -1

                matches_np = matches.cpu().numpy()
                valid_mask = (matches_np[:,0] >= 0) & (matches_np[:,1] >= 0)
                if valid_mask.sum() == 0:
                    print(f"  Pair ({start_idx},{start_idx+1}): no valid matches, skip pair")
                    continue

                matches_filtered = matches[valid_mask]
                m_kpts0 = kpts0[matches_filtered[..., 0].long()]
                m_kpts1 = kpts1[matches_filtered[..., 1].long()]

                if len(m_kpts0) < 8:
                    print(f"  Pair ({start_idx},{start_idx+1}): too few matches ({len(m_kpts0)}), skip pair")
                    continue

                # projection matrices (3x4)
                P0 = (K0 @ np.linalg.inv(pose0))[:3]
                P1 = (K1 @ np.linalg.inv(pose1))[:3]

                pts0 = m_kpts0.cpu().numpy().T.astype(np.float32)
                pts1 = m_kpts1.cpu().numpy().T.astype(np.float32)

                pts_h = cv2.triangulatePoints(P0, P1, pts0, pts1)
                pts3d = cv2.convertPointsFromHomogeneous(pts_h.T)
                point_cloud = pts3d[:, 0]  # (N,3)

                # 过滤负深
                point_cloud = remove_negative_depth(point_cloud, (P0, P1))
                if point_cloud.shape[0] < 5:
                    print(f"  Pair ({start_idx},{start_idx+1}): not enough positive-depth points ({point_cloud.shape[0]}), skip pair")
                    continue

                # 投影回参考相机，计算深度比
                pcd_homo = np.hstack([point_cloud, np.ones((len(point_cloud), 1))])
                proj_0 = (P0 @ pcd_homo.T).T  # (N,3)
                proj_depth = proj_0[:, 2]
                proj_uv = proj_0[:, :2] / proj_depth[:, None]

                # 在参考深度图上采样，收集ratio
                depth_im = depth0
                ratios = []
                h, w = depth_im.shape[:2]
                for i_pt in range(len(point_cloud)):
                    u, v = proj_uv[i_pt]
                    ui = int(round(u))
                    vi = int(round(v))
                    if 0 <= ui < w and 0 <= vi < h:
                        d_val = depth_im[vi, ui]
                        if d_val > 0.0:
                            ratios.append(proj_depth[i_pt] / float(d_val))

                if len(ratios) == 0:
                    print(f"  Pair ({start_idx},{start_idx+1}): no valid depth samples, skip pair")
                    continue

                pair_median = float(np.median(ratios))
                per_pair_medians.append(pair_median)
                successful_pairs += 1
                print(f"  Pair ({start_idx},{start_idx+1}): median ratio = {pair_median:.6f} (samples={len(ratios)})")

            except Exception as e:
                print(f"  Pair ({start_idx},{start_idx+1}): exception {e}, skip pair")
                continue

        # 完成该 submap 的所有选定对处理后，检查成功对数
        if successful_pairs < min_valid_pairs:
            raise RuntimeError(f"Submap {map_id} failed: only {successful_pairs} valid pairs (need >= {min_valid_pairs})")

        # 最终 scale 取 per-pair median 的中位数（robust）
        final_scale = float(np.median(per_pair_medians))
        print(f"  Submap {map_id} final scale (median of {successful_pairs} pairs): {final_scale:.6f}")
        estimated_scales.append(final_scale)

    return np.array(estimated_scales, dtype=np.float32)

# ---------------- main ----------------

BASE_PATH = "/mnt/archive/users/mingyuhang/organized_datasets/preprocessed/TanksAndTemples"

# 可修改要处理的场景列表
SCENE_NAMES = [
    "Barn",
    "Meetingroom",
    "Ignatius",
    "Courthouse",
    "Truck",
    "Caterpillar"
]

def process_scene_and_save(scene_name, base_path, max_pairs=10, min_valid_pairs=3, device='cuda'):
    path = os.path.join(base_path, scene_name)
    print(f"\n=== Scene: {scene_name} ===")
    if not os.path.exists(path):
        raise RuntimeError(f"Scene path not found: {path}")

    images = sorted(glob.glob(os.path.join(path, "images/*.jpg")))
    if len(images) == 0:
        raise RuntimeError(f"No images in {path}")

    poses_path = os.path.join(path, "poses.npy")
    if not os.path.exists(poses_path):
        raise RuntimeError(f"poses.npy not found in {path}")
    poses = np.load(poses_path)

    intrinsic_path = os.path.join(path, "intrinsic.txt")
    if not os.path.exists(intrinsic_path):
        raise RuntimeError(f"intrinsic.txt not found in {path}")
    K = np.loadtxt(intrinsic_path)[None].repeat(len(poses), axis=0)

    frame_window = 16
    n_submaps = len(images) // (frame_window - 1) + 1

    # construct submaps
    submaps = []
    for i in range(0, n_submaps):
        frames = []
        for j in range(frame_window - 1):
            index = i * (frame_window - 1) + j
            if index >= len(images):
                break
            depth_path = os.path.join(path, "vggt_depth", f"{i:06d}_{j:02d}.png")
            if not os.path.exists(depth_path):
                continue
            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if depth is None:
                continue
            depth = (depth / 5000.0).astype(np.float32)
            depth = cv2.resize(depth, (640, 480)).astype(np.float32)
            pose = poses[index]
            image = images[index]
            intrinsic = K[index]
            frames.append((image, depth, pose, intrinsic))
        submaps.append(frames)

    # run calibration (this will raise RuntimeError if any submap fails)
    scales = calibrate_intra_submap_scale(submaps, max_pairs=max_pairs, min_valid_pairs=min_valid_pairs, device=device)

    # save to submap_scale.txt
    out_file = os.path.join(path, "submap_scale.txt")
    with open(out_file, "w") as f:
        for idx, s in enumerate(scales):
            f.write(f"{idx} {s:.6f}\n")
    print(f"Saved {len(scales)} submap scales to {out_file}")

if __name__ == "__main__":
    # 全局入口：处理多个场景，若任一场景失败，脚本会抛错并中止
    for scene in SCENE_NAMES:
        process_scene_and_save(scene, BASE_PATH, max_pairs=10, min_valid_pairs=3, device='cuda')
    print("\nAll scenes processed successfully.")
