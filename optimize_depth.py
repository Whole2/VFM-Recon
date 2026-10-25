import os
import glob
import re
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from scipy.optimize import least_squares
import open3d as o3d
import itertools
import traceback
import datetime
import json

# 引入 SuperPoint 和 LightGlue 
from lightglue import LightGlue, SuperPoint, viz2d
from lightglue.utils import load_image, rbd

# [系统设置] 静音 Open3D 的非致命日志
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

# ================= 配置参数 =================
CONFIG = {
    # --- 1. 场景与文件配置 ---
    "base_path": "/mnt/archive/users/mingyuhang/organized_datasets/preprocessed/ScanNet",
    
    # 是否使用 deepvmvs.json 指定关键帧
    "use_keyframe_file": True,
    "keyframe_file_path": "deepvmvs.json",
    
    # 场景选择模式: "LIST", "RANGE", "ALL"
    "scene_selection_mode": "RANGE",
    
    # [模式 LIST]
    "scene_list": [
         "scene0721_00" 
    ],
    # [模式 RANGE]
    "scene_range_start": "scene0707_00",
    "scene_range_end": "scene0806_00",

    
    
    
    # --- 2. 核心参数 ---
    
    "version": 3,
    "align_weight": 1.0,
    
    "frame_window": 32,
    "overlap_frames": 4,
    
    
    # --- 3. 深度阈值 ---
    "max_depth_estimation": 1000.0,      
    "max_depth_alignment": 1000.0,       
    "max_depth_superpoint": 1000.0,      
    "tsdf_min_weight": 0.0,           
    "min_cluster_triangles": 1000,    
    
    # --- 4. 算法参数 ---
    "min_inliers": 20,             
    "reproj_thresh": 2.0,          
    "voxel_size": 0.02,            
    "sdf_trunc_factor": 5.0,
    "pairs_per_submap": 10,         
    "overlap_min_points": 100,     
    "png_depth_scale": 1000.0      
}

# ==================== 0. 辅助函数 ====================

def load_keyframe_indices(json_path):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Keyframe file not found: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def get_target_scenes(base_path, mode, specific_list, range_start, range_end):
    all_scenes_in_dir = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])
    
    if mode == "ALL":
        return all_scenes_in_dir
    elif mode == "LIST":
        return [s for s in specific_list if s in all_scenes_in_dir]
    elif mode == "RANGE":
        valid_scenes = []
        for s in all_scenes_in_dir:
            if s >= range_start and s <= range_end:
                valid_scenes.append(s)
        return valid_scenes
    else:
        raise ValueError(f"Unknown SCENE_SELECTION_MODE: {mode}")

# ==================== 1. 基础几何工具函数 ====================

def project_points(points, proj_mat):
    points = points.reshape(-1, 3)
    pcd_homo = cv2.convertPointsToHomogeneous(points)[:, 0]
    proj_0 = (proj_mat @ pcd_homo.T).T
    proj_z = proj_0[:, -1:]
    proj_z[proj_z == 0] = 1e-6
    proj = proj_0[:, :-1] / proj_z
    return proj, proj_z

def reprojection_error(point_cloud, P0, P1, m_kpts0, m_kpts1):
    point_cloud = point_cloud.reshape(-1, 3)
    pcd_homo = cv2.convertPointsToHomogeneous(point_cloud)[:, 0]
    proj_0 = P0 @ pcd_homo.T
    proj_1 = P1 @ pcd_homo.T
    proj_0 = proj_0[:2] / proj_0[2]
    proj_1 = proj_1[:2] / proj_1[2]
    error0 = np.linalg.norm(proj_0.T - m_kpts0, axis=1)
    error1 = np.linalg.norm(proj_1.T - m_kpts1, axis=1)
    return np.stack([error0, error1], axis=1).mean(axis=-1)

def remove_outliers(point_cloud, P0, P1, m_kpts0, m_kpts1, max_depth_thresh, threshold=2.0):
    reproj_error = reprojection_error(point_cloud, P0, P1, m_kpts0, m_kpts1)
    inliers_reproj = reproj_error < threshold
    _, proj_z0 = project_points(point_cloud, P0)
    _, proj_z1 = project_points(point_cloud, P1)
    inliers_depth = (proj_z0 > 0.1) & (proj_z1 > 0.1) & \
                    (proj_z0 < max_depth_thresh) & (proj_z1 < max_depth_thresh)
    inliers = inliers_reproj & inliers_depth[:, 0]
    return point_cloud[inliers], inliers

# ==================== 2. 核心功能：单Submap尺度估计 ====================

def calibrate_intra_submap_scale(submaps):
    limit_depth_sp = CONFIG["max_depth_superpoint"]
    print(f"  > [步骤 1] SuperPoint Scale估计 (Feature MaxDepth={limit_depth_sp})...")
    
    extractor = SuperPoint(max_num_keypoints=2048).eval().cuda() 
    matcher = LightGlue(features='superpoint').eval().cuda()

    estimated_scales = []
    
    for map_id, frames in enumerate(submaps):
        num_frames = len(frames)
        if num_frames < 2:
            estimated_scales.append(None)
            continue
            
        pairs_to_test = []
        for i in range(0, num_frames - 1):
            pairs_to_test.append((i, i+1))
        for i in range(0, num_frames - 2):
            pairs_to_test.append((i, i+2))
            
        if len(pairs_to_test) > CONFIG["pairs_per_submap"]:
            indices = np.linspace(0, len(pairs_to_test)-1, CONFIG["pairs_per_submap"], dtype=int)
            pairs_to_test = [pairs_to_test[i] for i in indices]
            
        submap_ratios = []
        
        for (idx1, idx2) in pairs_to_test:
            try:
                img1_path, depth1, pose1, K1, _ = frames[idx1] 
                img2_path, depth2, pose2, K2, _ = frames[idx2]
                
                P1 = (K1 @ np.linalg.inv(pose1))[:3]
                P2 = (K2 @ np.linalg.inv(pose2))[:3]

                im1 = cv2.imread(img1_path, -1)
                im2 = cv2.imread(img2_path, -1)
                im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
                im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
                
                image1_tensor = torch.from_numpy(im1).float().cuda().permute(2, 0, 1) / 255.0
                image2_tensor = torch.from_numpy(im2).float().cuda().permute(2, 0, 1) / 255.0
                
                feats0 = extractor.extract(image1_tensor, resize=None)
                feats1 = extractor.extract(image2_tensor, resize=None)
                matches01 = matcher({'image0': feats0, 'image1': feats1})
                feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]
                
                kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
                m_kpts0 = kpts0[matches[..., 0]]
                m_kpts1 = kpts1[matches[..., 1]]
                
                np_kpts0 = m_kpts0.cpu().numpy()
                np_kpts1 = m_kpts1.cpu().numpy()
                
                if len(np_kpts0) < CONFIG["min_inliers"]:
                    continue
                
                points_4d = cv2.triangulatePoints(P1, P2, np_kpts0.T, np_kpts1.T)
                points_3d = cv2.convertPointsFromHomogeneous(points_4d.T)[:, 0]

                points_clean, inliers = remove_outliers(
                    points_3d, P1, P2, np_kpts0, np_kpts1, 
                    max_depth_thresh=limit_depth_sp, 
                    threshold=CONFIG["reproj_thresh"]
                )
                
                if points_clean.shape[0] < CONFIG["min_inliers"]:
                    continue

                pcd_homo = np.hstack([points_clean, np.ones((len(points_clean), 1))])
                proj_0 = (P1 @ pcd_homo.T).T 
                proj_depth = proj_0[:, 2] 
                proj_uv = proj_0[:, :2] / proj_depth[:, None]
                
                current_pair_ratios = []
                h_im, w_im = depth1.shape
                
                for k in range(len(points_clean)):
                    u, v = proj_uv[k]
                    if 0 <= u < w_im and 0 <= v < h_im:
                        d_pred = depth1[int(v), int(u)]
                        if d_pred > 1e-3:
                            current_pair_ratios.append(proj_depth[k] / d_pred)
                
                if len(current_pair_ratios) > 10:
                    submap_ratios.append(np.median(current_pair_ratios))

            except Exception:
                pass
        
        if len(submap_ratios) > 0:
            final_scale = np.median(submap_ratios)
            estimated_scales.append(final_scale)
        else:
            print(f"    Submap {map_id}: Scale估计失败 (特征点不足或过远)")
            estimated_scales.append(None)
            
    return estimated_scales

# ==================== 3. 核心功能：全局尺度图优化 ====================

def compute_relative_ratios(submaps, overlap_frames):
    ratios = [] 
    valid_edges = [] 
    limit_depth = CONFIG["max_depth_alignment"]
    
    for i in range(len(submaps) - 1):
        edge_ratios = []
        for k in range(overlap_frames):
            idx_prev = -(overlap_frames - k) 
            idx_curr = k                    
            try:
                # frames: (path, depth, pose, K, real_idx)
                depth_prev = submaps[i][idx_prev][1]
                depth_curr = submaps[i+1][idx_curr][1]
                
                # 校验: 确保重叠帧是同一个真实帧
                id_prev = submaps[i][idx_prev][4]
                id_curr = submaps[i+1][idx_curr][4]
                if id_prev != id_curr:
                    continue

                mask = (depth_prev > 1e-3) & (depth_curr > 1e-3) & \
                       (depth_prev < limit_depth) & (depth_curr < limit_depth)
                if np.sum(mask) > CONFIG["overlap_min_points"]:
                    val = depth_prev[mask] / depth_curr[mask]
                    edge_ratios.append(val)
            except IndexError:
                continue

        if len(edge_ratios) == 0:
            ratios.append(1.0)
            valid_edges.append(False)
        else:
            all_vals = np.concatenate(edge_ratios)
            ratio = np.median(all_vals)
            ratios.append(ratio)
            valid_edges.append(True)
    return np.array(ratios), valid_edges

def optimize_global_scales(submaps, initial_abs_scales, overlap_frames):
    print(f"  > [步骤 2] 全局图优化 (Align MaxDepth={CONFIG['max_depth_alignment']})...")
    n_nodes = len(submaps)
    rel_ratios, rel_valid = compute_relative_ratios(submaps, overlap_frames)
    rel_targets = np.log(rel_ratios + 1e-6) 
    
    abs_indices = []
    abs_targets = []
    for i, s in enumerate(initial_abs_scales):
        if s is not None and not np.isnan(s) and s > 0:
            abs_indices.append(i)
            abs_targets.append(np.log(s))
            
    if len(abs_indices) == 0:
        print("    警告: 无有效 Scale，回退到 1.0")
        abs_indices = [0]
        abs_targets = [0.0]

    def loss_func(x):
        residuals = []
        w_rel = CONFIG["align_weight"]
        for i in range(n_nodes - 1):
            if rel_valid[i]:
                diff = x[i+1] - x[i]
                target = rel_targets[i]
                residuals.append(w_rel * (diff - target))
        w_abs = 1.0
        for idx, target in zip(abs_indices, abs_targets):
            residuals.append(w_abs * (x[idx] - target))
        return np.array(residuals)

    avg_val = np.median(abs_targets) if abs_targets else 0.0
    x0 = np.full(n_nodes, avg_val)
    
    res = least_squares(loss_func, x0, verbose=0)
    optimized_scales = np.exp(res.x)
    return optimized_scales

# ==================== 4. 融合功能 (FORCE CPU) ====================

def depth_fusion(submaps, scales=None, output="tsdf.ply"):
    voxel_size = CONFIG["voxel_size"]
    sdf_trunc_factor = CONFIG["sdf_trunc_factor"] 
    limit_depth = CONFIG["max_depth_estimation"]
    min_weight = CONFIG.get("tsdf_min_weight", 5.0)

    device = o3d.core.Device("CPU:0")
    
    vbg = o3d.t.geometry.VoxelBlockGrid(
        attr_names=('tsdf', 'weight', 'color'),
        attr_dtypes=(o3d.core.Dtype.Float32, o3d.core.Dtype.Float32, o3d.core.Dtype.Float32),
        attr_channels=((1), (1), (3)),
        voxel_size=voxel_size,
        block_resolution=16,
        block_count=50000,
        device=device
    )

    for map_id, frames in enumerate(submaps):
        current_scale = 1.0
        if scales is not None:
            if isinstance(scales, (list, np.ndarray)) and map_id < len(scales):
                s = scales[map_id]
                if s is not None and np.isfinite(s):
                    current_scale = float(s)
            
        for i in range(len(frames)):
            # frames item: (img_path, depth_map, pose, K, real_idx)
            _, depth_map, pose, K, _ = frames[i]
            
            depth = depth_map * current_scale
            H, W = depth.shape
            
            depth_tensor = o3d.core.Tensor(depth.astype(np.float32), device=device).reshape((H, W, 1))
            color_tensor = o3d.core.Tensor(np.ones((H, W, 3), dtype=np.float32), device=device)
            
            depth_img = o3d.t.geometry.Image(depth_tensor)
            color_img = o3d.t.geometry.Image(color_tensor)
            
            intrinsic_t = o3d.core.Tensor(K[:3, :3].astype(np.float64), device=device)
            extrinsic_t = o3d.core.Tensor(np.linalg.inv(pose).astype(np.float64), device=device)
            
            depth_max_val = float(limit_depth)
            trunc_val = float(sdf_trunc_factor)

            frustum_block_coords = vbg.compute_unique_block_coordinates(
                depth_img, intrinsic_t, extrinsic_t, 
                depth_scale=1.0, depth_max=depth_max_val, trunc_voxel_multiplier=trunc_val
            )
            
            vbg.integrate(
                frustum_block_coords, depth_img, color_img, 
                intrinsic_t, extrinsic_t, 
                depth_scale=1.0, depth_max=depth_max_val, trunc_voxel_multiplier=trunc_val
            )

    try:
        mesh_t = vbg.extract_triangle_mesh(weight_threshold=min_weight)
    except TypeError:
        mesh_t = vbg.extract_triangle_mesh()
    
    mesh = mesh_t.to_legacy()
    
    if len(mesh.vertices) > 0:
        mesh.remove_duplicated_vertices()
        mesh.remove_duplicated_triangles()
        mesh.remove_degenerate_triangles()
        try:
            triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
            triangle_clusters = np.asarray(triangle_clusters)
            cluster_n_triangles = np.asarray(cluster_n_triangles)
            min_cluster_triangles = CONFIG["min_cluster_triangles"]
            triangles_to_remove = cluster_n_triangles[triangle_clusters] < min_cluster_triangles
            mesh.remove_triangles_by_mask(triangles_to_remove)
            mesh.remove_unreferenced_vertices()
        except Exception:
            pass

    o3d.io.write_triangle_mesh(output, mesh)

# ==================== 5. 投影保存 (FORCE CPU) ====================

def save_optimized_depth_maps(submaps, optimized_scales, output_dir, mesh_path: str = None):
    print(f"  > [步骤 3] 保存深度图到: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # 取第一帧获取尺寸
    _, d_sample, _, _, _ = submaps[0][0]
    H, W = d_sample.shape
    png_scale = CONFIG["png_depth_scale"]

    if mesh_path is None or not os.path.exists(mesh_path):
        raise RuntimeError(f"mesh_path not found: {mesh_path}")

    legacy_mesh = o3d.io.read_triangle_mesh(mesh_path)
    if legacy_mesh is None or len(legacy_mesh.vertices) == 0:
        raise RuntimeError(f"Mesh is empty: {mesh_path}")

    device = o3d.core.Device("CPU:0")
    
    try:
        mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(legacy_mesh).to(device)
        mesh_t.vertex.positions = mesh_t.vertex.positions.to(o3d.core.Dtype.Float32)
    except Exception as e:
        raise RuntimeError(f"Failed to create TensorMesh on CPU: {e}")

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh_t)

    # 收集需要保存的帧 (Real Index -> Pose/K)
    # 目的：确保每一帧只保存一次，且覆盖整个场景
    tasks = {}
    for map_id, frames in enumerate(submaps):
        for item in frames:
            # item: (path, depth, pose, K, real_idx)
            pose, K, real_idx = item[2], item[3], item[4]
            if real_idx not in tasks:
                tasks[real_idx] = (pose.copy(), K.copy())

    print(f"    共需保存 {len(tasks)} 帧优化后的深度图...")
    saved_count = 0
    
    try:
        # 按真实ID排序保存，确保输出文件是 depth_000000.png, depth_000001.png 等（对应原始序号）
        sorted_indices = sorted(tasks.keys())
        
        for idx in sorted_indices:
            pose, K = tasks[idx]
            
            c2w_pose = pose.astype(np.float64)
            final_w2c = np.linalg.inv(c2w_pose)

            intrinsic_t = o3d.core.Tensor(K[:3, :3], dtype=o3d.core.Dtype.Float32, device=device)
            extrinsic_t = o3d.core.Tensor(final_w2c.astype(np.float32), dtype=o3d.core.Dtype.Float32, device=device)

            rays = scene.create_rays_pinhole(intrinsic_t, extrinsic_t, width_px=W, height_px=H)
            ans = scene.cast_rays(rays)
            depth_np = ans['t_hit'].numpy()

            finite_mask = np.isfinite(depth_np)
            depth_np[~finite_mask] = 0.0
            
            scaled = np.rint(depth_np * png_scale)
            scaled = np.clip(scaled, 0, 65535).astype(np.uint16)

            # [命名关键点] 必须使用原始图像序号命名
            out_path = os.path.join(output_dir, f"{idx:06d}.png")
            cv2.imwrite(out_path, scaled)
            saved_count += 1

            if saved_count % 50 == 0:
                print(f"    Saved count: {saved_count}")

        print(f"    Done. Saved {saved_count} depth maps.")

    except Exception as e:
        raise RuntimeError("Raycasting loop failed: " + str(e))

# ==================== 主入口 ====================

def process_single_scene(scene_name, keyframe_indices=None):
    print(f"\n{'='*50}")
    print(f"正在处理场景: {scene_name}")
    if keyframe_indices:
        print(f"Keyframe Mode: ON ({len(keyframe_indices)} frames)")
    else:
        print(f"Keyframe Mode: OFF (All frames)")
    print(f"{'='*50}")
    
    scene_path = os.path.join(CONFIG["base_path"], scene_name)
    if not os.path.exists(scene_path):
        print(f"错误: 路径不存在 {scene_path}")
        return

    frame_window = CONFIG["frame_window"]
    overlap_frames = CONFIG["overlap_frames"]
    stride = frame_window - overlap_frames

    # 1. 确定输入的 depth 文件夹名称
    # 规则：vggt_depth_{suffix}_{window}_{overlap}
    suffix = "kf" if keyframe_indices is not None else "full"
    depth_folder_name = f"vggt_depth_{frame_window}_{overlap_frames}"
    full_depth_path = os.path.join(scene_path, depth_folder_name)
    
    if not os.path.exists(full_depth_path):
        print(f"  错误: 找不到深度文件夹 {full_depth_path}")
        return

    # 2. 准备基础数据 (Images, Poses, Intrinsic)
    all_images = sorted(glob.glob(os.path.join(scene_path, "images/*.jpg")))
    poses_path = os.path.join(scene_path, "poses.npy")
    intr_path = os.path.join(scene_path, "intrinsic.txt")
    
    if not os.path.exists(poses_path): return
    all_poses = np.load(poses_path)
    
    if not os.path.exists(intr_path): return
    K_base = np.loadtxt(intr_path)
    # 广播 K
    all_K = K_base[None].repeat(len(all_poses), axis=0)

    # 3. 确定要处理的帧索引列表 (Real Indices)
    target_indices = []
    if keyframe_indices is not None:
        target_indices = [idx for idx in keyframe_indices if idx < len(all_images)]
        target_indices.sort()
    else:
        target_indices = list(range(len(all_images)))

    if len(target_indices) == 0:
        print("  警告: 无有效帧需要处理")
        return

    # 4. 构建 Submaps (核心修改部分)
    # 需要重建 VGGT 代码中的切分逻辑来找到对应的深度文件名
    submaps = []
    
    submap_idx = 0  # Submap 计数器，对应文件名中的 000000
    curr_start = 0  # 列表索引游标
    
    while curr_start < len(target_indices):
        curr_end = min(curr_start + frame_window, len(target_indices))
        
        # 获取当前窗口内的 真实索引 (用来找 RGB 和 Pose)
        batch_real_indices = target_indices[curr_start : curr_end]
        
        frames_in_submap = []
        
        # 遍历当前 submap 内的每一帧
        # local_id 对应文件名中的 _00, _01, _02
        for local_id, real_idx in enumerate(batch_real_indices):
            
            # [文件名修正] 按照 SubmapID_FrameInSubmapID.png 读取
            # 例如: 000000_00.png, 000000_01.png
            depth_filename = f"{submap_idx:06d}_{local_id:02d}.png"
            depth_path = os.path.join(full_depth_path, depth_filename)
            
            if not os.path.exists(depth_path):
                print(f"    Missing depth: {depth_filename} (Mapped to RealFrame {real_idx})")
                continue
                
            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED) / 5000.0
            depth = cv2.resize(depth, (640, 480)).astype(np.float32)
            
            # 收集数据: (ImgPath, Depth, Pose, K, RealIdx)
            frames_in_submap.append((
                all_images[real_idx], 
                depth, 
                all_poses[real_idx], 
                all_K[real_idx],
                real_idx 
            ))
            
        if len(frames_in_submap) > 0:
            submaps.append(frames_in_submap)
        
        submap_idx += 1  # 下一个 submap
        if curr_end == len(target_indices):
            break
        curr_start += stride

    print(f"  构建了 {len(submaps)} 个 Submaps (源自 {len(target_indices)} 帧)")
    
    out_ply_before = os.path.join(scene_path, f"{scene_name}_{suffix}_BeforeScale.ply")
    depth_fusion(submaps, scales=None, output=out_ply_before)

    # 5. Scale 估计
    raw_scales = calibrate_intra_submap_scale(submaps)

    # # 6. Fusion (Before Opt)
    # out_ply_before = os.path.join(scene_path, f"{scene_name}_{suffix}_BeforeOpt.ply")
    # depth_fusion(submaps, scales=raw_scales, output=out_ply_before)
    
    # out_ply_before = os.path.join(scene_path, f"{scene_name}_BeforeScale.ply")
    # depth_fusion(submaps, scales=None, output=out_ply_before)

    # 7. 图优化
    optimized_scales = optimize_global_scales(submaps, raw_scales, overlap_frames)
    
    # print("\n  === 优化报告 (前5个) ===")
    # for i in range(min(5, len(submaps))):
    #     r_val = raw_scales[i]
    #     ratio = f"{optimized_scales[i]/r_val:.2f}x" if (r_val is not None) else "Filled"
    #     print(f"    ID {i}: Raw={r_val:.3f} -> Opt={optimized_scales[i]:.3f} ({ratio})")

    # 8. Fusion (After Opt)
    out_ply_after = os.path.join(scene_path, f"{scene_name}_AfterOpt_{CONFIG['version']}.ply")
    depth_fusion(submaps, scales=optimized_scales, output=out_ply_after)
    
    
    # out_ply_after = os.path.join("debug", f"{scene_name}_AfterOpt_{CONFIG['version']}.ply")
    # depth_fusion(submaps, scales=optimized_scales, output=out_ply_after)
    

    # 9. 保存深度图
    out_depth_dir = os.path.join(scene_path, f"vggt_optimize_depth_superpoint_project_{frame_window}_{overlap_frames}")
    save_optimized_depth_maps(submaps, optimized_scales, out_depth_dir, out_ply_after)

if __name__ == "__main__":
    
    # 1. 准备关键帧数据
    kf_data = {}
    if CONFIG["use_keyframe_file"]:
        print(f"Loading keyframes from {CONFIG['keyframe_file_path']}...")
        try:
            kf_data = load_keyframe_indices(CONFIG['keyframe_file_path'])
            print(f"Loaded keyframe config for {len(kf_data)} scenes.")
        except Exception as e:
            print(f"Failed to load keyframe file: {e}")
            exit(1)

    # 2. 筛选场景
    try:
        target_scenes = get_target_scenes(
            CONFIG["base_path"], 
            CONFIG["scene_selection_mode"], 
            CONFIG["scene_list"], 
            CONFIG["scene_range_start"], 
            CONFIG["scene_range_end"]
        )
    except Exception as e:
        print(f"Error in scene selection: {e}")
        exit(1)

    print(f"任务列表 ({len(target_scenes)}): {target_scenes}")
    error_log_file = "error_log_fusion.txt"
    
    for scene in target_scenes:
        try:
            # 获取当前场景的关键帧列表
            current_kf = None
            if CONFIG["use_keyframe_file"]:
                if scene in kf_data:
                    current_kf = sorted(kf_data[scene])
                else:
                    print(f"Warning: Scene {scene} not in json. Using ALL frames.")
                    current_kf = None 

            process_single_scene(scene, keyframe_indices=current_kf)
            
        except Exception as e:
            error_msg = f"[{datetime.datetime.now()}] Scene '{scene}' Failed: {str(e)}\n"
            full_traceback = traceback.format_exc()
            with open(error_log_file, "a") as f:
                f.write(error_msg)
                f.write(full_traceback)
                f.write("-" * 50 + "\n")
            print(f"!!! 场景 {scene} 处理出错 (详情已保存至 {error_log_file})")
            
    print("\n所有场景处理完毕。")