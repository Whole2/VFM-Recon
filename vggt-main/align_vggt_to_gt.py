import os
import glob
import numpy as np
import cv2
import open3d as o3d
from tqdm import tqdm

# ================= 配置区域 =================

# 1. 路径与场景配置
BASE_PATH = "/mnt/archive/users/mingyuhang/organized_datasets/preprocessed/TanksAndTemples/"

SCENE_LIST = [
    "Barn",
    "Caterpillar",
    "Courthouse",
    "Ignatius",
    "Meetingroom",
    "Truck" 
]

# 2. 窗口与重叠配置 (必须与 VGGT 生成脚本的配置完全一致!)
SUBMAP_WINDOW = 8
USE_OVERLAP = False
SUBMAP_OVERLAP_VAL = 4 if USE_OVERLAP else 0
STRIDE = SUBMAP_WINDOW - SUBMAP_OVERLAP_VAL
VGGT_FOLDER_NAME = f"vggt_out_{SUBMAP_WINDOW}_{SUBMAP_OVERLAP_VAL}"

# 3. 融合策略配置
FUSE_OVERLAP_FRAMES = True 
MIN_CLUSTER_TRIANGLES = 2000 

# 4. 输出目录
OUTPUT_DIR = "align_output_mesh"

# 5. TSDF 核心参数
VOXEL_SIZE = 0.04   
TRUNC_MARGIN = 0.20 
DEPTH_SCALE_FACTOR = 5000.0 
MAX_DEPTH = 3.0     

# =================================================

def compute_umeyama_scale(pred_pts, gt_pts):
    """仅计算尺度 s (Scale)，使得 s * pred ~ gt"""
    n = pred_pts.shape[0]
    if n < 3: return 1.0

    mu_pred = np.mean(pred_pts, axis=0)
    mu_gt = np.mean(gt_pts, axis=0)
    pred_centered = pred_pts - mu_pred
    gt_centered = gt_pts - mu_gt

    var_pred = np.mean(np.sum(pred_centered**2, axis=1))
    H = pred_centered.T @ gt_centered / n

    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(U @ Vt)
    S_mat = np.eye(3)
    if d < 0: S_mat[2, 2] = -1
    R = U @ S_mat @ Vt

    trace_RH = np.trace(R @ H.T)
    scale = trace_RH / var_pred
    return scale

def get_submap_ids(vggt_path):
    pose_files = glob.glob(os.path.join(vggt_path, "pose", "*.npy"))
    submap_ids = set()
    for f in pose_files:
        basename = os.path.basename(f)
        try:
            sid = int(basename.split('_')[0])
            submap_ids.add(sid)
        except:
            continue
    return sorted(list(submap_ids))

def remove_small_clusters(mesh, min_triangles):
    if min_triangles <= 0: return mesh
    print(f"正在清理 Mesh: 移除面数小于 {min_triangles} 的碎片...")
    cluster_index_map, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
    cluster_index_map = np.asarray(cluster_index_map)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    valid_clusters_indices = np.where(cluster_n_triangles >= min_triangles)[0]
    print(f"共发现 {len(cluster_n_triangles)} 个碎片，保留其中 {len(valid_clusters_indices)} 个大区域。")
    if len(valid_clusters_indices) == 0: return mesh

    triangles_to_remove = np.ones(len(cluster_index_map), dtype=bool)
    for valid_idx in valid_clusters_indices:
        triangles_to_remove[cluster_index_map == valid_idx] = False
    mesh.remove_triangles_by_mask(triangles_to_remove)
    mesh.remove_unreferenced_vertices()
    return mesh

def load_intrinsics_from_txt(txt_path):
    """
    从 txt 文件加载内参矩阵。
    支持格式：
    1. 纯数字矩阵 (4x4 或 3x3)，空格或逗号分隔
    2. ScanNet / TanksAndTemples 常见格式
    """
    try:
        # 读取所有行，过滤掉注释和空行
        with open(txt_path, 'r') as f:
            lines = [l.strip() for l in f.readlines() if l.strip() and not l.startswith('#')]
        
        # 将所有数字解析出来
        data = []
        for line in lines:
            # 替换逗号为 空格，防止 csv 格式报错
            parts = line.replace(',', ' ').split()
            data.extend([float(x) for x in parts])
        
        data = np.array(data)

        # 判断形状
        if data.size == 16:
            K = data.reshape(4, 4)[:3, :3] # 取左上角 3x3
        elif data.size == 9:
            K = data.reshape(3, 3)
        else:
            print(f"[错误] intrinsic.txt 包含 {data.size} 个数字，无法解析为 3x3 或 4x4 矩阵。")
            return None
            
        print(f"成功从 {os.path.basename(txt_path)} 加载内参:\n{K}")
        return K

    except Exception as e:
        print(f"[错误] 解析 intrinsic.txt 失败: {e}")
        return None

def process_single_scene(scene_name):
    print(f"\n{'='*50}")
    print(f"正在处理场景: {scene_name}")
    print(f"Submap 模式: {'有重叠 (Overlapping)' if USE_OVERLAP else '无重叠 (Block/Non-Overlapping)'}")
    
    scene_dir = os.path.join(BASE_PATH, scene_name)
    vggt_dir = os.path.join(scene_dir, VGGT_FOLDER_NAME)
    gt_pose_path = os.path.join(scene_dir, "poses.npy")
    
    # 优先寻找 intrinsic.txt
    gt_intrinsics_txt = os.path.join(scene_dir, "intrinsic.txt") # 注意：通常单数 intrinsic.txt
    gt_intrinsics_npy = os.path.join(scene_dir, "intrinsics.npy")

    if not os.path.exists(vggt_dir):
        print(f"[错误] 未找到 VGGT 输出目录: {vggt_dir}")
        return
    if not os.path.exists(gt_pose_path):
        print(f"[错误] 未找到 GT Pose 文件: {gt_pose_path}")
        return

    # 加载 GT Pose
    print("正在加载 GT Pose...")
    full_gt_poses = np.load(gt_pose_path)
    
    # 加载内参逻辑
    K_gt_all = None
    
    # 1. 尝试 txt
    if os.path.exists(gt_intrinsics_txt):
        K_gt_all = load_intrinsics_from_txt(gt_intrinsics_txt)
    
    # 2. 尝试 npy (如果 txt 失败或不存在)
    if K_gt_all is None and os.path.exists(gt_intrinsics_npy):
        print(f"正在加载 numpy 内参: {gt_intrinsics_npy}")
        K_gt_all = np.load(gt_intrinsics_npy)

    # 3. 默认值
    if K_gt_all is None:
        print("[警告] 未找到内参文件，使用默认 ScanNet 内参。")
        K_gt_all = np.array([[577.87, 0, 319.5], [0, 577.87, 239.5], [0, 0, 1]])

    # 初始化 TSDF Volume
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=VOXEL_SIZE,
        sdf_trunc=TRUNC_MARGIN,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )

    submap_ids = get_submap_ids(vggt_dir)
    print(f"共发现 {len(submap_ids)} 个 submaps。")

    processed_global_frames = set()

    for submap_id in tqdm(submap_ids, desc=f"融合 {scene_name}"):
        
        # --- A: 收集数据用于计算 Scale ---
        pred_centers = []
        gt_centers = []
        valid_indices = [] 

        for i in range(SUBMAP_WINDOW):
            pred_pose_file = os.path.join(vggt_dir, "pose", f"{submap_id:06d}_{i:02d}.npy")
            if not os.path.exists(pred_pose_file): continue

            global_frame_idx = submap_id * STRIDE + i
            if global_frame_idx >= len(full_gt_poses): continue

            pred_pose = np.load(pred_pose_file)
            gt_pose = full_gt_poses[global_frame_idx]

            if np.isinf(gt_pose).any() or np.isnan(gt_pose).any(): continue
            if np.sum(gt_pose[:3, 3]) == 0: continue 

            pred_centers.append(pred_pose[:3, 3])
            gt_centers.append(gt_pose[:3, 3])
            valid_indices.append((i, global_frame_idx))

        # --- B: 计算 Umeyama Scale ---
        if len(pred_centers) < 5: continue

        scale = compute_umeyama_scale(np.array(pred_centers), np.array(gt_centers))
        scale = abs(scale) 

        if scale < 0.01 or scale > 100.0:
            print(f"Submap {submap_id} 尺度异常 ({scale:.4f})，跳过。")
            continue

        # --- C: 执行融合 ---
        for (local_i, global_idx) in valid_indices:
            
            if not FUSE_OVERLAP_FRAMES:
                if global_idx in processed_global_frames:
                    continue
                else:
                    processed_global_frames.add(global_idx)

            depth_path = os.path.join(vggt_dir, "depth", f"{submap_id:06d}_{local_i:02d}.png")
            if not os.path.exists(depth_path): continue

            depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if depth_raw is None: continue

            # 1. 应用 Scale
            depth_meters = (depth_raw.astype(np.float32) / DEPTH_SCALE_FACTOR) * scale
            depth_o3d = o3d.geometry.Image(depth_meters)

            # 2. 准备位姿 (W2C)
            pose_c2w = full_gt_poses[global_idx]
            pose_w2c = np.linalg.inv(pose_c2w)

            # 3. 准备内参 (处理多帧不同内参的情况)
            # 如果 K_gt_all 是 (N, 3, 3) 或 (N, 4, 4)
            if K_gt_all.ndim == 3:
                K = K_gt_all[global_idx]
                if K.shape == (4, 4): K = K[:3, :3] # 确保取 3x3
            else:
                # 全局一致的内参
                K = K_gt_all
                if K.shape == (4, 4): K = K[:3, :3]

            fx, fy = K[0,0], K[1,1]
            cx, cy = K[0,2], K[1,2]
            h, w = depth_raw.shape
            intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)

            # 4. 准备颜色图
            img_path = os.path.join(scene_dir, "images", f"{global_idx}.jpg") 
            if not os.path.exists(img_path):
                 color_img = np.ones((h, w, 3), dtype=np.uint8) * 200
            else:
                 color_img = cv2.imread(img_path)
                 color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
            
            color_o3d = o3d.geometry.Image(color_img)

            # 5. 融合
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_o3d,
                depth_o3d,
                depth_scale=1.0, 
                depth_trunc=MAX_DEPTH,
                convert_rgb_to_intensity=False
            )

            volume.integrate(rgbd, intrinsic, pose_w2c)

    print("正在提取 Mesh...")
    mesh = volume.extract_triangle_mesh()
    
    if MIN_CLUSTER_TRIANGLES > 0:
        mesh = remove_small_clusters(mesh, MIN_CLUSTER_TRIANGLES)

    mesh.compute_vertex_normals()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    mode_str = "overlap" if USE_OVERLAP else "no_overlap"
    out_path = os.path.join(OUTPUT_DIR, f"{scene_name}_{mode_str}_mesh.ply")
    
    o3d.io.write_triangle_mesh(out_path, mesh)
    print(f"完成。结果已保存至: {out_path}")

def main():
    print(f"开始批量处理 {len(SCENE_LIST)} 个场景...")
    for scene in SCENE_LIST:
        try:
            process_single_scene(scene)
        except Exception as e:
            print(f"[严重错误] 处理场景 {scene} 失败: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()