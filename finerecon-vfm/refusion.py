import os
import glob
import numpy as np
import open3d as o3d
from tqdm import tqdm
import argparse

def load_poses(pose_path):
    """
    读取 poses.npy
    假设格式为 (N, 4, 4) 或 (N, 3, 4) 的 c2w 矩阵
    """
    poses = np.load(pose_path)
    # 如果是 (N, 3, 4)，补全最后一行变成 (N, 4, 4)
    if poses.shape[1] == 3:
        bottom = np.zeros((poses.shape[0], 1, 4))
        bottom[:, :, 3] = 1.0
        poses = np.concatenate([poses, bottom], axis=1)
    return poses

def load_intrinsic(intrinsic_path):
    """
    读取内参文件。
    这里假设是一个txt，存了4x4矩阵。
    """
    try:
        K_matrix = np.loadtxt(intrinsic_path)
        # 只需要 3x3 部分用于投影
        intrinsic_3x3 = K_matrix[:3, :3]
        return intrinsic_3x3
    except Exception as e:
        print(f"Error loading intrinsic: {e}")
        return None

def keep_largest_cluster(mesh):
    """
    保留 Mesh 中最大的连通区域，删除小的悬浮碎片。
    """
    print("  -> Filtering: Keeping largest connected component...")
    
    # 1. 连通性分析
    # triangle_clusters: 每个三角形所属的聚类ID
    # cluster_n_triangles: 每个聚类包含的三角形数量
    # cluster_area: 每个聚类的表面积 (这里我们主要用三角形数量，通常也是准的)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error):
        triangle_clusters, cluster_n_triangles, cluster_area = (
            mesh.cluster_connected_triangles())
    
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    
    if len(cluster_n_triangles) == 0:
        print("  -> Warning: No clusters found.")
        return mesh

    # 2. 找到最大的聚类 ID
    largest_cluster_idx = cluster_n_triangles.argmax()
    print(f"  -> Found {len(cluster_n_triangles)} clusters. Largest has {cluster_n_triangles[largest_cluster_idx]} triangles.")
    
    # 3. 移除其他聚类的三角形
    triangles_to_remove = triangle_clusters != largest_cluster_idx
    mesh.remove_triangles_by_mask(triangles_to_remove)
    
    # 4. 清理未引用的顶点 (因为删除了三角形，有些顶点可能孤立了)
    mesh.remove_unreferenced_vertices()
    
    return mesh

def refusion_scene(mesh_path, pose_path, intrinsic_path, output_path, voxel_size=0.04, img_h=1080, img_w=1920, skip_frame=5, device='cuda:0'):
    """
    对单个场景进行 Refusion
    """
    # ---------------------------------------------------------
    # 1. 资源加载
    # ---------------------------------------------------------
    print(f"\n[Processing] {os.path.basename(mesh_path)}")
    
    # 加载 Mesh
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if len(mesh.vertices) == 0:
        print(f"Warning: Empty mesh found at {mesh_path}, skipping.")
        return

    # 加载 Poses
    if not os.path.exists(pose_path):
        print(f"Error: Pose file not found: {pose_path}")
        return
    c2w_poses = load_poses(pose_path) # N x 4 x 4

    # 加载 Intrinsic
    if not os.path.exists(intrinsic_path):
        print(f"Error: Intrinsic file not found: {intrinsic_path}")
        return
    K = load_intrinsic(intrinsic_path)
    
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Open3D Tensor Intrinsic (3x3)
    intrinsic_tensor = o3d.core.Tensor(K, dtype=o3d.core.Dtype.Float64)

    # ---------------------------------------------------------
    # 2. 准备 Raycasting Scene (用于生成深度图)
    # ---------------------------------------------------------
    try:
        t_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(t_mesh)
    except Exception as e:
        print(f"Error initializing RaycastingScene: {e}. Ensure Open3D >= 0.16")
        return

    # ---------------------------------------------------------
    # 3. 初始化 TSDF Volume (用于融合)
    # ---------------------------------------------------------
    sdf_trunc = voxel_size * 6.5 
    
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor
    )

    n_poses = len(c2w_poses)
    
    # ---------------------------------------------------------
    # 4. 循环：投影 -> 获取深度 -> 重新融合
    # ---------------------------------------------------------
    indices = range(0, n_poses, skip_frame)
    
    print(f"  -> Raycasting & Fusing ({len(indices)} frames)...")
    
    for idx in tqdm(indices):
        c2w = c2w_poses[idx]
        
        if np.isinf(c2w).any() or np.isnan(c2w).any():
            continue

        # --- A. 生成光线 (Raycasting) ---
        try:
            w2c = np.linalg.inv(c2w)
        except np.linalg.LinAlgError:
            continue

        extrinsic_tensor = o3d.core.Tensor(w2c, dtype=o3d.core.Dtype.Float64)
        
        rays = scene.create_rays_pinhole(
            intrinsic_matrix=intrinsic_tensor,
            extrinsic_matrix=extrinsic_tensor,
            width_px=img_w,
            height_px=img_h
        )
        
        ans = scene.cast_rays(rays)
        depth_map_tensor = ans['t_hit']
        
        depth_map_np = depth_map_tensor.numpy()
        # 处理无穷大（未击中）的情况
        depth_map_np[np.isinf(depth_map_np)] = 0.0 
        
        # --- B. 融合 (Integration) ---
        # 1. 深度图转 Image
        depth_img = o3d.geometry.Image(depth_map_np.astype(np.float32))
        
        # 2. 构造一个全黑的颜色图 (uint8)
        color_img = o3d.geometry.Image(np.zeros((img_h, img_w, 3), dtype=np.uint8))
        
        # 3. 合成为 RGBDImage
        # depth_scale=1.0 非常重要！因为 raycasting 的结果已经是米单位，不需要缩放
        # depth_trunc=1000.0 设大一点，防止远处的房子被切掉
        rgbd_img = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_img, 
            depth_img, 
            depth_scale=1.0, 
            depth_trunc=1000.0, 
            convert_rgb_to_intensity=False
        )
        
        volume.integrate(
            rgbd_img,
            o3d.camera.PinholeCameraIntrinsic(img_w, img_h, fx, fy, cx, cy),
            w2c
        )

    # ---------------------------------------------------------
    # 5. 提取 Mesh 并保存
    # ---------------------------------------------------------
    print("  -> Extracting clean mesh...")
    clean_mesh = volume.extract_triangle_mesh()
    
    # [新增] 只保留最大连通域
    clean_mesh = keep_largest_cluster(clean_mesh)
    
    o3d.io.write_triangle_mesh(output_path, clean_mesh)
    print(f"  -> Saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Refusion script to clean internal noise from meshes.")
    parser.add_argument("--scenes_dir", type=str, default="/mnt/archive/users/mingyuhang/organized_datasets/preprocessed/TanksAndTemples")
    parser.add_argument("--mesh_dir", type=str, default="lightning_logs/superpoint_8_4_test")
    parser.add_argument("--out_dir", type=str, default="clean_mesh")
    parser.add_argument("--voxel_size", type=float, default=0.04)
    parser.add_argument("--img_w", type=int, default=640)
    parser.add_argument("--img_h", type=int, default=480)
    parser.add_argument("--skip", type=int, default=1)

    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    mesh_files = glob.glob(os.path.join(args.mesh_dir, "*.ply")) + glob.glob(os.path.join(args.mesh_dir, "*.obj"))
    
    print(f"Found {len(mesh_files)} meshes to process.")

    for mesh_f in mesh_files:
        scene_name = os.path.splitext(os.path.basename(mesh_f))[0]
        
        scene_pose_path = os.path.join(args.scenes_dir, scene_name, "poses.npy")
        scene_intr_path = os.path.join(args.scenes_dir, scene_name, "intrinsic.txt")
        
        output_f = os.path.join(args.out_dir, f"{scene_name}_clean.ply")
        
        if not os.path.exists(scene_pose_path):
            # 处理可能的命名不匹配 (e.g. Barn_colmap -> Barn)
            if "_" in scene_name:
                short_name = scene_name.rsplit("_", 1)[0]
                scene_pose_path_alt = os.path.join(args.scenes_dir, short_name, "poses.npy")
                scene_intr_path_alt = os.path.join(args.scenes_dir, short_name, "intrinsic.txt")
                if os.path.exists(scene_pose_path_alt):
                    scene_pose_path = scene_pose_path_alt
                    scene_intr_path = scene_intr_path_alt
                else:
                    print(f"Skipping {scene_name}, poses.npy not found.")
                    continue
            else:
                print(f"Skipping {scene_name}, poses.npy not found.")
                continue

        refusion_scene(
            mesh_path=mesh_f,
            pose_path=scene_pose_path,
            intrinsic_path=scene_intr_path,
            output_path=output_f,
            voxel_size=args.voxel_size,
            img_w=args.img_w,
            img_h=args.img_h,
            skip_frame=args.skip
        )

if __name__ == "__main__":
    main()