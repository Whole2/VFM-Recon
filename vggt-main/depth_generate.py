import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
import glob
import os
import numpy as np
import tqdm
import cv2
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
import math
import json

# ================= Configuration Switches =================

# --- 1. 功能开关 ---
RUN_ORIGINAL_SUBMAPS = True       # 生成切分后的详细 Submaps
RUN_GLOBAL_KEYFRAMES = False       # 生成由每个Submap第一帧组成的 Global Map

# --- 2. 关键帧配置 (新增) ---
# 是否使用 deepvmvs.json 指定关键帧
USE_KEYFRAME_FILE = True
# 关键帧文件路径
KEYFRAME_FILE_PATH = "deepvmvs.json"

# --- 3. 场景选择配置 (新增) ---
# 模式选择: 
#   "LIST"  -> 仅处理 SCENE_LIST 中列出的场景
#   "RANGE" -> 按照字母顺序处理从 SCENE_RANGE_START 到 SCENE_RANGE_END 之间的场景 (包含首尾)
#   "ALL"   -> 处理 BASE_PATH 下的所有文件夹
SCENE_SELECTION_MODE = "RANGE" 

# [模式 LIST] 指定要处理的场景列表
SCENE_LIST = [
    "scene0718_00",
    # "scene0718_01"
]

# [模式 RANGE] 指定范围 (字符串排序)
SCENE_RANGE_START = "scene0707_00"
SCENE_RANGE_END   = "scene0806_00"

# --- 4. Submap 切分参数 ---
# 注意：若开启关键帧模式，这里的Window是指“8个关键帧”，而不是原始视频的8帧
SUBMAP_FRAME_WINDOW = 32          # 每个 Submap 包含的帧数
SUBMAP_OVERLAP = 4                # 相邻 Submap 之间的重叠帧数
assert SUBMAP_OVERLAP < SUBMAP_FRAME_WINDOW, "Overlap must be smaller than Frame Window!"

# --- 5. 基础路径 ---
BASE_PATH = "/mnt/archive/users/mingyuhang/organized_datasets/preprocessed/ScanNet"

# ==========================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

# Initialize the model (Load ONCE globally)
print("Initializing VGGT model...")
model = VGGT()
state_dict = torch.hub.load_state_dict_from_url(
    "https://hf-mirror.com/facebook/VGGT-1B/resolve/main/model.pt"
)
model.load_state_dict(state_dict, strict=False)
model.eval()
model = model.to(device)
print("Model loaded.")

def load_keyframe_indices(json_path):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Keyframe file not found: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def get_target_scenes(base_path, mode, specific_list, range_start, range_end):
    """
    根据配置模式筛选需要处理的场景列表
    """
    all_scenes_in_dir = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])
    
    if mode == "ALL":
        print(f"Mode ALL: Found {len(all_scenes_in_dir)} scenes in directory.")
        return all_scenes_in_dir
    
    elif mode == "LIST":
        # 过滤掉文件夹里不存在的
        valid_scenes = [s for s in specific_list if s in all_scenes_in_dir]
        print(f"Mode LIST: Requested {len(specific_list)}, Found {len(valid_scenes)} valid scenes.")
        return valid_scenes
        
    elif mode == "RANGE":
        # 字符串比较筛选范围
        valid_scenes = []
        for s in all_scenes_in_dir:
            if s >= range_start and s <= range_end:
                valid_scenes.append(s)
        print(f"Mode RANGE: From {range_start} to {range_end}, Found {len(valid_scenes)} scenes.")
        return valid_scenes
    
    else:
        raise ValueError(f"Unknown SCENE_SELECTION_MODE: {mode}")

def process_single_scene(scene_path, keyframe_indices=None):
    scene_name = os.path.basename(scene_path)
    print(f"\n{'='*40}")
    print(f"Processing Scene: {scene_name}")
    if keyframe_indices is not None:
        print(f"Keyframe Mode: ON (Selecting {len(keyframe_indices)} frames from config)")
    else:
        print(f"Keyframe Mode: OFF (Using all frames found)")
    print(f"{'='*40}")

    # Output directories
    # 为了区分，可以在输出文件夹名上加标记，或者保持原样
    suffix = "kf" if keyframe_indices is not None else "full"
    output_dir = os.path.join(scene_path, f"vggt_depth_{SUBMAP_FRAME_WINDOW}_{SUBMAP_OVERLAP}")
    global_output_dir = os.path.join(scene_path, f"vggt_depth_global_{SUBMAP_FRAME_WINDOW}_{SUBMAP_OVERLAP}")

    if RUN_ORIGINAL_SUBMAPS:
        os.makedirs(output_dir, exist_ok=True)
    if RUN_GLOBAL_KEYFRAMES:
        os.makedirs(global_output_dir, exist_ok=True)

    # 1. Glob all images first (to ensure we have the base set index mapping)
    # 假设 images 文件夹下的文件名排序对应 index 0, 1, 2...
    all_images_paths = sorted(glob.glob(os.path.join(scene_path, "images/*.jpg")))
    
    if len(all_images_paths) == 0:
        print(f"WARNING: No images found in {scene_path}/images. Skipping.")
        return

    # 2. Filter paths based on keyframes if needed
    target_image_paths = []
    
    if keyframe_indices is not None:
        # 过滤
        for idx in keyframe_indices:
            if 0 <= idx < len(all_images_paths):
                target_image_paths.append(all_images_paths[idx])
            else:
                print(f"Warning: Keyframe index {idx} out of bounds for {scene_name} (total {len(all_images_paths)}). Ignored.")
        
        if len(target_image_paths) == 0:
            print(f"Error: No valid keyframes found for {scene_name}. Skipping.")
            return
    else:
        target_image_paths = all_images_paths

    # 3. Load Images
    print(f"Loading {len(target_image_paths)} images...")
    try:
        # load_and_preprocess_images 接受路径列表
        images = load_and_preprocess_images(target_image_paths)
    except Exception as e:
        print(f"Error loading images for {scene_path}: {e}")
        return

    # 4. Calculate Submap Parameters on the loaded images (Keyframes or Full)
    stride = SUBMAP_FRAME_WINDOW - SUBMAP_OVERLAP
    
    submap_indices = []
    curr_start = 0
    while curr_start < len(images):
        curr_end = min(curr_start + SUBMAP_FRAME_WINDOW, len(images))
        submap_indices.append((curr_start, curr_end))
        if curr_end == len(images):
            break
        curr_start += stride

    n_submaps = len(submap_indices)
    print(f"Settings -> Window: {SUBMAP_FRAME_WINDOW}, Overlap: {SUBMAP_OVERLAP}, Stride: {stride}")
    print(f"Total processing frames: {len(images)}, Submaps to generate: {n_submaps}")

    # ==============================================================================
    # Feature 1: Submap Processing
    # ==============================================================================
    if RUN_ORIGINAL_SUBMAPS:
        print(f"\n--- [{scene_name}] Starting Submap Processing ---")
        
        for i, (start, end) in tqdm.tqdm(enumerate(submap_indices), total=n_submaps, desc="Processing Submaps"):
            images_batch = images[start:end].to(device)

            if images_batch.shape[0] < 1:
                continue

            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=dtype):
                    predictions = model(images_batch)
            
            # --- Save Depth Maps ---
            depth_map = predictions["depth"]
            depth_map = depth_map.squeeze(0).cpu().numpy() * 5000.0
            
            for j in range(depth_map.shape[0]):
                # Filename: submapID_frameInSubmapID.png
                filename = os.path.join(output_dir, f"{i:06d}_{j:02d}.png")
                cv2.imwrite(filename, depth_map[j].astype(np.uint16))
            
            # 这里省略了 Pose 处理的保存代码，如果需要保存 Pose，请按原逻辑添加

        print("Submap processing done.")

    # ==============================================================================
    # Feature 2: Global Keyframe Processing
    # ==============================================================================
    if RUN_GLOBAL_KEYFRAMES:
        print(f"\n--- [{scene_name}] Starting Global Keyframe Processing ---")
        
        # indices relative to the 'images' tensor (which might be keyframes only)
        # 取每个submap的第一帧
        batch_indices = [start_idx for (start_idx, _) in submap_indices]
        batch_indices = [idx for idx in batch_indices if idx < len(images)]
        
        if not batch_indices:
            print("No keyframes found.")
        else:
            print(f"Collecting {len(batch_indices)} representative frames...")
            
            global_images_batch = images[batch_indices].to(device)
            
            try:
                with torch.no_grad():
                    with torch.cuda.amp.autocast(dtype=dtype):
                        predictions = model(global_images_batch)
                
                depth_map = predictions["depth"] # [1, T, H, W]
                depth_map = depth_map.squeeze(0).cpu().numpy() * 5000.0
                
                print(f"Saving {depth_map.shape[0]} global depth maps to {global_output_dir}...")
                
                for k, relative_idx in enumerate(batch_indices):
                    # Naming: 
                    # 如果使用了 Keyframe 文件，这里的命名最好能对应回原始视频的第几帧，
                    # 还是对应回 Submap 逻辑的第几帧？
                    # 你的原始需求是：“按照原本的逻辑...命名逻辑依然submap序号加submap内序号”
                    # 但 Global Map 通常需要唯一的 ID。
                    
                    # 方案 A: 使用过滤后的列表中的 index (0, 1, 2...) -> 方便看顺序
                    # filename = os.path.join(global_output_dir, f"depth_{relative_idx:06d}.png")
                    
                    # 方案 B: 还原回真实视频的 Frame ID (更具物理意义)
                    # 如果 keyframe_indices 存在，images[relative_idx] 对应的是 original[keyframe_indices[relative_idx]]
                    if keyframe_indices is not None:
                        real_frame_id = keyframe_indices[relative_idx]
                    else:
                        real_frame_id = relative_idx # 原地映射
                        
                    filename = os.path.join(global_output_dir, f"depth_{real_frame_id:06d}.png")
                    cv2.imwrite(filename, depth_map[k].astype(np.uint16))
                    
                print("Global Keyframe processing done.")
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"ERROR: OOM during Global processing for {scene_name}.")
                else:
                    raise e

# --- Main Loop ---
if __name__ == "__main__":
    
    # 1. 准备关键帧数据
    kf_data = {}
    if USE_KEYFRAME_FILE:
        print(f"Loading keyframes from {KEYFRAME_FILE_PATH}...")
        try:
            kf_data = load_keyframe_indices(KEYFRAME_FILE_PATH)
            print(f"Loaded keyframe config for {len(kf_data)} scenes.")
        except Exception as e:
            print(f"Failed to load keyframe file: {e}")
            exit(1)

    # 2. 筛选场景
    try:
        target_scenes = get_target_scenes(
            BASE_PATH, 
            SCENE_SELECTION_MODE, 
            SCENE_LIST, 
            SCENE_RANGE_START, 
            SCENE_RANGE_END
        )
    except Exception as e:
        print(f"Error in scene selection: {e}")
        exit(1)

    if not target_scenes:
        print("No scenes matches the selection criteria.")
        exit(0)

    # 3. 循环处理
    print(f"\nStarting processing for {len(target_scenes)} scenes...")
    
    for scene_name in target_scenes:
        full_path = os.path.join(BASE_PATH, scene_name)
        
        # 获取当前场景的关键帧列表 (如果开启且存在)
        current_scene_kf = None
        if USE_KEYFRAME_FILE:
            if scene_name in kf_data:
                # 确保是按照序号排序的 (sort)
                current_scene_kf = sorted(kf_data[scene_name])
            else:
                print(f"Warning: Scene {scene_name} not found in json file. Falling back to ALL frames (or skip based on policy).")
                # 这里默认策略：如果在json里没找到，就用全帧。如果希望跳过，可以加 continue
                current_scene_kf = None 

        if os.path.exists(full_path):
            process_single_scene(full_path, keyframe_indices=current_scene_kf)
        else:
            print(f"ERROR: Scene path not found on disk: {full_path}")

    print("\nAll requested scenes completed.")