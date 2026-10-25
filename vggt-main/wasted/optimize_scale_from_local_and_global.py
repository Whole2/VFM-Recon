import numpy as np
from scipy.optimize import least_squares

def optimize_scales(
    num_submaps, 
    initial_estimates_local, # 你的 estimate_scale 算出的 local scales list
    initial_estimate_global, # 你的 estimate_scale 算出的 global scale
    overlap_ratios_neighbors, # 列表: [(i, i+1, median_ratio), ...]
    overlap_ratios_global     # 列表: [(i, global_idx, median_ratio), ...]
):
    """
    Variables: x[0]...x[N-1] are log(scale_local), x[N] is log(scale_global)
    """
    
    # 初始猜测 (全部转为 log 域)
    x0 = np.log(initial_estimates_local + [initial_estimate_global])
    
    def loss_function(x):
        residuals = []
        
        # 1. Prior Term (先验项): 尽量不要偏离 estimate_scale 算出的结果太远
        # 权重可以设低一点，因为我们更相信 overlap 的一致性，但需要这个来定绝对尺度
        w_prior = 1.0 
        current_locals = x[:-1]
        current_global = x[-1]
        
        prior_locals = np.log(initial_estimates_local)
        prior_global = np.log(initial_estimate_global)
        
        residuals.extend(w_prior * (current_locals - prior_locals))
        residuals.append(w_prior * (current_global - prior_global))
        
        # 2. Consistency Term (相邻 Submap 一致性)
        # 权重设高，因为物理上重叠区域深度必须一致
        w_consistency = 10.0 
        for (idx_i, idx_j, r_ij) in overlap_ratios_neighbors:
            # log(S_i) - log(S_j) - log(r_ij) ≈ 0
            # r_ij = Depth_j / Depth_i
            # So S_i * D_i = S_j * D_j => S_i/S_j = D_j/D_i = r_ij
            diff = x[idx_i] - x[idx_j] - np.log(r_ij)
            residuals.append(w_consistency * diff)
            
        # 3. Global Consistency Term (Global vs Local 一致性)
        w_global_link = 10.0
        for (idx_local, _, r_lg) in overlap_ratios_global:
            # log(S_local) - log(S_global) - log(r_lg) ≈ 0
            # r_lg = Depth_global / Depth_local
            diff = x[idx_local] - current_global - np.log(r_lg)
            residuals.append(w_global_link * diff)
            
        return np.array(residuals)

    # 优化
    res = least_squares(loss_function, x0, verbose=1)
    
    # 还原为线性尺度
    optimized_scales = np.exp(res.x)
    
    return optimized_scales[:-1], optimized_scales[-1]

# --- 你的现有流程中需要增加的部分 ---

# 1. 假设 submaps 是一个 list
# 2. 计算 overlap_ratios_neighbors
neighbor_ratios = []
for i in range(len(submaps) - 1):
    # 取 submap[i] 最后一张 和 submap[i+1] 第一张
    # 假设它们是同一帧图像
    d1 = submaps[i].prediction[-1] 
    d2 = submaps[i+1].prediction[0]
    
    # 计算中位数比率 (加上一些保护措施防止除以0)
    valid_mask = (d1 > 1e-3) & (d2 > 1e-3)
    ratio = np.median(d2[valid_mask] / d1[valid_mask])
    neighbor_ratios.append((i, i+1, ratio))

# 3. 计算 overlap_ratios_global
global_ratios = []
for i in range(len(submaps)):
    # 取 submap[i] 第一张 和 Global Map 中对应的帧
    d_local = submaps[i].prediction[0]
    d_global = global_submap.prediction[i] # 假设 global 存的是所有 submap 的第一帧
    
    valid_mask = (d_local > 1e-3) & (d_global > 1e-3)
    ratio = np.median(d_global[valid_mask] / d_local[valid_mask])
    global_ratios.append((i, "global", ratio))

# 4. 运行优化
final_local_scales, final_global_scale = optimize_scales(
    len(submaps), 
    initial_estimates_list, # [1.2, 0.9, 1.5 ...] 来自你的脚本
    initial_global_estimate, # 1.1 来自你的脚本
    neighbor_ratios,
    global_ratios
)

# 5. 用 final_local_scales[i] 去乘以 submap[i] 的所有深度，然后 fusion