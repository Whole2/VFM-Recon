import open3d as o3d
import numpy as np
import copy

def preprocess_point_cloud(pcd, voxel_size):
    """预处理：降采样 + 估计法线 + 计算 FPFH 特征"""
    # 1. 降采样 (为了加速特征计算)
    pcd_down = pcd.voxel_down_sample(voxel_size)
    
    # 2. 估计法线 (搜索半径是体素的2倍)
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    
    # 3. 计算 FPFH 特征 (这是几何指纹，搜索半径是体素的5倍)
    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    """第一阶段：基于特征的全局 RANSAC Sim3 配准"""
    distance_threshold = voxel_size * 1.5
    print(f"   [RANSAC] 正在进行全局特征匹配 (这可能需要几秒钟)...")
    
    # 核心：使用 RANSAC 配合 Sim3 (with_scaling=True)
    # 这不是靠猜位置，而是靠特征指纹匹配
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True, # mutual_filter
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True),
        3, # ransac_n: 采样点数，Sim3 至少需要3个点
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(1000000, 0.999) # 跑 400万次迭代，确保找到最优
    )
    return result

def refine_registration(source, target, init_trans, voxel_size):
    """第二阶段：高精度点到面 ICP 微调 (Rigid SE3)"""
    distance_threshold = voxel_size * 0.4
    print(f"   [ICP] 正在进行高精度 Point-to-Plane 微调...")
    
    # 注意：这时候通常已经尺度对齐了，为了精度，我们固定住尺度，只做刚体微调 (SE3)
    # Point-to-Plane 比 Point-to-Point 精度高得多
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, init_trans,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
    )
    return result

def ultimate_align(rec_mesh_path, gt_pcd_path):
    # -------------------------------------------------------
    # 1. 准备数据
    # -------------------------------------------------------
    print("1. 加载数据...")
    rec_mesh = o3d.io.read_triangle_mesh(rec_mesh_path)
    gt_pcd = o3d.io.read_point_cloud(gt_pcd_path)
    
    # 采样点云
    rec_pcd = rec_mesh.sample_points_poisson_disk(number_of_points=50000)
    
    # -------------------------------------------------------
    # 2. 预估并应用初始尺度 (关键修改!)
    # -------------------------------------------------------
    # 计算两个点云的包围盒对角线长度
    bbox_rec = rec_pcd.get_axis_aligned_bounding_box()
    bbox_gt = gt_pcd.get_axis_aligned_bounding_box()
    
    len_rec = np.linalg.norm(bbox_rec.get_max_bound() - bbox_rec.get_min_bound())
    len_gt = np.linalg.norm(bbox_gt.get_max_bound() - bbox_gt.get_min_bound())
    
    # 粗略计算尺度比例
    init_scale = len_gt / len_rec
    print(f"2. 预估初始尺度: {init_scale:.4f} (GT尺寸/Rec尺寸)")
    
    # 设定体素大小 (基于 GT 尺寸)
    voxel_size = len_gt / 50.0
    print(f"   设定体素大小: {voxel_size:.3f}")

    # 【核心操作】直接先把 Source 缩放到和 Target 差不多大
    # 注意：这里我们只缩放用于配准的临时点云，不缩放原始 mesh，最后统一用矩阵处理
    rec_pcd_scaled = copy.deepcopy(rec_pcd)
    # center=(0,0,0) 表示围绕原点缩放，这样等效于乘以一个对角矩阵
    rec_pcd_scaled.scale(init_scale, center=(0,0,0)) 

    # -------------------------------------------------------
    # 3. 提取特征
    # -------------------------------------------------------
    print("3. 提取特征...")
    # 注意：这里传入的是已经缩放过的 rec_pcd_scaled
    source_down, source_fpfh = preprocess_point_cloud(rec_pcd_scaled, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(gt_pcd, voxel_size)

    # -------------------------------------------------------
    # 4. 全局 RANSAC (改为刚体 Rigid 配准)
    # -------------------------------------------------------
    # 因为我们已经手动缩放了，现在只需要解 旋转 + 平移 (6DoF)，这比 Sim3 稳定得多
    print("4. 执行全局 RANSAC (Rigid) 配准...")
    
    distance_threshold = voxel_size * 1.5
    
    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=False), # 【改为 False】
        3,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(1000000, 0.999)
    )
    
    print(f"   RANSAC Fitness: {result_ransac.fitness:.4f}")
    
    # -------------------------------------------------------
    # 5. 精细微调 (ICP)
    # -------------------------------------------------------
    print("5. ICP 微调...")
    result_icp = refine_registration(source_down, target_down, result_ransac.transformation, voxel_size)
    print(f"   ICP Fitness: {result_icp.fitness:.4f}, RMSE: {result_icp.inlier_rmse:.4f}")

    # -------------------------------------------------------
    # 6. 组合最终变换矩阵
    # -------------------------------------------------------
    # 最终变换 T = T_icp * S_init
    # 构造缩放矩阵 (4x4)
    scale_matrix = np.eye(4) * init_scale
    scale_matrix[3, 3] = 1.0
    
    # 矩阵乘法顺序：先缩放(init_scale)，再刚体变换(result_icp)
    final_transform = result_icp.transformation @ scale_matrix
    
    print(f"   最终计算出的变换矩阵:\n{final_transform}")

    # -------------------------------------------------------
    # 7. 应用并保存
    # -------------------------------------------------------
    rec_mesh.transform(final_transform)
    
    rec_mesh.paint_uniform_color([1, 0.7, 0])
    gt_pcd.paint_uniform_color([0, 0.65, 0.92])
    
    # 验证是否 Scale 正常 (检查矩阵的行列式)
    det = np.linalg.det(final_transform[:3, :3])
    final_scale = np.cbrt(det)
    print(f"   最终复合尺度: {final_scale:.4f}")
    
    if final_scale < 0.001:
        print("[错误] 尺度依然异常，请检查输入点云是否为空或重合！")
    else:
        o3d.visualization.draw_geometries([rec_mesh, gt_pcd], window_name="Correct Alignment")
        o3d.io.write_triangle_mesh("Ignatius_aligned.ply", rec_mesh)
        print("已保存: Ignatius_aligned.ply")

# 换成你的路径
ultimate_align(r"D:\recon\final_version\tnt\vggt_version\Ignatius_clean_cropped.ply", r"scan\Ignatius.ply")
