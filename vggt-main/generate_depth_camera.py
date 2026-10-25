import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
import glob
import os
import numpy as np
import tqdm
import cv2
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
import math
import json

# ================= Configuration Switches =================

# --- 1. 功能及保存开关 ---
RUN_ORIGINAL_SUBMAPS = True       # 生成切分后的详细 Submaps
SAVE_DEPTH = True                 # 是否保存深度图 (.png) -> 存储在 /depth
SAVE_POSES = True                 # 是否保存外参 (c2w 4x4 .npy) -> 存储在 /pose
SAVE_INTRINSICS = True            # 是否保存内参 (4x4 .npy) -> 存储在 /intrinsics

# --- 2. 关键帧配置 ---
USE_KEYFRAME_FILE = False
KEYFRAME_FILE_PATH = "deepvmvs.json"

# --- 3. 场景选择配置 ---
SCENE_SELECTION_MODE = "ALL" 

SCENE_LIST = ["scene0718_00"]
SCENE_RANGE_START = "scene0707_00"
SCENE_RANGE_END   = "scene0806_00"

# --- 4. Submap 切分参数 ---
SUBMAP_FRAME_WINDOW = 8          

# [NEW] 重叠逻辑开关
# True:  使用由 SUBMAP_OVERLAP 定义的重叠 (stride = window - overlap)
# False: 禁用重叠，直接按窗口切分 (stride = window)
USE_OVERLAP = False               

SUBMAP_OVERLAP = 4               
if USE_OVERLAP:
    assert SUBMAP_OVERLAP < SUBMAP_FRAME_WINDOW, "Overlap must be smaller than Frame Window!"

# --- 5. 基础路径 ---
BASE_PATH = "/mnt/archive/users/mingyuhang/organized_datasets/preprocessed/TanksAndTemples"

# ==========================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

# Initialize the model
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
    all_scenes_in_dir = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])
    if mode == "ALL":
        return all_scenes_in_dir
    elif mode == "LIST":
        return [s for s in specific_list if s in all_scenes_in_dir]
    elif mode == "RANGE":
        return [s for s in all_scenes_in_dir if range_start <= s <= range_end]
    else:
        raise ValueError(f"Unknown SCENE_SELECTION_MODE: {mode}")

def process_single_scene(scene_path, keyframe_indices=None):
    scene_name = os.path.basename(scene_path)
    print(f"\n{'='*40}")
    print(f"Processing Scene: {scene_name}")
    
    # [NEW] 根据开关确定实际的 overlap 和 stride
    if USE_OVERLAP:
        actual_overlap = SUBMAP_OVERLAP
        stride = SUBMAP_FRAME_WINDOW - actual_overlap
        print(f"Mode: Overlap ENABLED. Window: {SUBMAP_FRAME_WINDOW}, Overlap: {actual_overlap}, Stride: {stride}")
    else:
        actual_overlap = 0
        stride = SUBMAP_FRAME_WINDOW
        print(f"Mode: Overlap DISABLED. Window: {SUBMAP_FRAME_WINDOW}, Overlap: 0, Stride: {stride}")

    # 基础输出目录 (文件名会自动反映实际的 overlap 是多少)
    base_output_dir = os.path.join(scene_path, f"vggt_out_{SUBMAP_FRAME_WINDOW}_{actual_overlap}")
    
    # 定义并创建子文件夹
    depth_dir = os.path.join(base_output_dir, "depth")
    pose_dir = os.path.join(base_output_dir, "pose")
    intrinsic_dir = os.path.join(base_output_dir, "intrinsics")

    if RUN_ORIGINAL_SUBMAPS:
        os.makedirs(base_output_dir, exist_ok=True)
        if SAVE_DEPTH: os.makedirs(depth_dir, exist_ok=True)
        if SAVE_POSES: os.makedirs(pose_dir, exist_ok=True)
        if SAVE_INTRINSICS: os.makedirs(intrinsic_dir, exist_ok=True)

    all_images_paths = sorted(glob.glob(os.path.join(scene_path, "images/*.jpg")))
    if len(all_images_paths) == 0: return

    target_image_paths = []
    if keyframe_indices is not None:
        for idx in keyframe_indices:
            if 0 <= idx < len(all_images_paths):
                target_image_paths.append(all_images_paths[idx])
    else:
        target_image_paths = all_images_paths

    if len(target_image_paths) == 0: return

    print(f"Loading {len(target_image_paths)} images...")
    images = load_and_preprocess_images(target_image_paths)

    # 计算切分索引
    submap_indices = []
    curr_start = 0
    while curr_start < len(images):
        curr_end = min(curr_start + SUBMAP_FRAME_WINDOW, len(images))
        submap_indices.append((curr_start, curr_end))
        if curr_end == len(images): break
        curr_start += stride

    if RUN_ORIGINAL_SUBMAPS:
        print(f"--- [{scene_name}] Starting Submap Processing ({len(submap_indices)} submaps) ---")
        _, _, img_h, img_w = images.shape 

        for i, (start, end) in tqdm.tqdm(enumerate(submap_indices), total=len(submap_indices), desc="Processing Submaps"):
            images_batch = images[start:end].to(device)
            if images_batch.shape[0] < 1: continue

            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=dtype):
                    predictions = model(images_batch)
            
            # --- 1. 深度图保存 ---
            if SAVE_DEPTH:
                depth_maps = predictions["depth"].squeeze(0).cpu().numpy() * 5000.0
                for j in range(depth_maps.shape[0]):
                    cv2.imwrite(os.path.join(depth_dir, f"{i:06d}_{j:02d}.png"), depth_maps[j].astype(np.uint16))
            
            # --- 2. 相机参数保存 ---
            if SAVE_POSES or SAVE_INTRINSICS:
                pose_enc = predictions["pose_enc"]
                extrinsics_w2c, intrinsics = pose_encoding_to_extri_intri(
                    pose_enc, 
                    image_size_hw=(img_h, img_w), 
                    pose_encoding_type="absT_quaR_FoV"
                )
                
                # 处理外参 (c2w 4x4)
                if SAVE_POSES:
                    ex_w2c = extrinsics_w2c.squeeze(0).cpu().numpy()
                    for j in range(ex_w2c.shape[0]):
                        R_w2c = ex_w2c[j, :3, :3]
                        t_w2c = ex_w2c[j, :3, 3:]
                        R_c2w = R_w2c.T
                        t_c2w = -R_c2w @ t_w2c
                        pose_c2w = np.eye(4)
                        pose_c2w[:3, :3] = R_c2w
                        pose_c2w[:3, 3:] = t_c2w
                        np.save(os.path.join(pose_dir, f"{i:06d}_{j:02d}.npy"), pose_c2w)

                # 处理内参 (4x4)
                if SAVE_INTRINSICS:
                    ins_3x3 = intrinsics.squeeze(0).cpu().numpy()
                    for j in range(ins_3x3.shape[0]):
                        K_4x4 = np.eye(4)
                        K_4x4[:3, :3] = ins_3x3[j]
                        np.save(os.path.join(intrinsic_dir, f"{i:06d}_{j:02d}.npy"), K_4x4)

        print(f"Scene {scene_name} completed.")

# --- Main Loop ---
if __name__ == "__main__":
    kf_data = {}
    if USE_KEYFRAME_FILE:
        try:
            kf_data = load_keyframe_indices(KEYFRAME_FILE_PATH)
        except Exception as e:
            print(f"Failed to load keyframe file: {e}"); exit(1)

    target_scenes = get_target_scenes(BASE_PATH, SCENE_SELECTION_MODE, SCENE_LIST, SCENE_RANGE_START, SCENE_RANGE_END)

    for scene_name in target_scenes:
        full_path = os.path.join(BASE_PATH, scene_name)
        current_scene_kf = sorted(kf_data[scene_name]) if (USE_KEYFRAME_FILE and scene_name in kf_data) else None
        if os.path.exists(full_path):
            process_single_scene(full_path, keyframe_indices=current_scene_kf)