#!/usr/bin/env python3
# vggt_predict_and_save_depth.py
import os
import math
from pathlib import Path
import torch
import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image
from torchvision.transforms.functional import to_tensor
import random

ROOT_DIR = "/mnt/archive/datasets/ScanNet/TartanAir"
IMAGE_SUBDIR = "image_left"
OUTPUT_SUBDIR = "vggt_pred_depth_left"
# =====================================================

MODEL_PT = None  # e.g. "/path/to/model.pt" or None
# ------------------------------------------------


# 推理参数（可调）
WINDOW_SIZE = 16        # 每次送入模型的视角数（根据显存调整）
CONF_THRESH = 0.5       # 融合时的置信度阈值（也用来 mask 可视化）
TARGET_SIZE = 518       # VGGT 预处理目标尺寸（默认 518）
MODE = "pad"            # "pad" 保留全部像素；若要中心裁剪可改 "crop"

device = "cuda" if torch.cuda.is_available() else "cpu"
# 选择半精度（如果 GPU 支持 bfloat16 用 bfloat16）
if torch.cuda.is_available():
    try:
        cap = torch.cuda.get_device_capability()[0]
        dtype = torch.bfloat16 if cap >= 8 else torch.float16
    except Exception:
        dtype = torch.float16
else:
    dtype = torch.float32

# ------------------------------
# 加载 VGGT 模型（并保证 buffers 在 device）
# ------------------------------
from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

model = VGGT()
state_dict = torch.hub.load_state_dict_from_url(
    "https://hf-mirror.com/facebook/VGGT-1B/resolve/main/model.pt"
)
model.load_state_dict(state_dict, strict=False)
model.eval()
model = model.to(device)


def load_and_preprocess_with_meta(image_paths, mode="pad", target_size=TARGET_SIZE):
    """
    Args:
        image_paths: list of file paths (jpg/png...)
        mode: "pad" or "crop"
        target_size: int
    Returns:
        imgs_tensor: torch.Tensor [S, 3, target_size, target_size]
        metas: list of dict, 每个 dict 包含:
           {
             "orig_size": (ow, oh),
             "resized_size": (rw, rh),
             "pad": (pad_left, pad_top, pad_right, pad_bottom)
           }
    """
    to_t = to_tensor
    images = []
    metas = []
    shapes = set()
    for p in image_paths:
        pil = Image.open(p)
        # handle alpha: paste on white
        if pil.mode == "RGBA":
            bg = Image.new("RGBA", pil.size, (255, 255, 255, 255))
            pil = Image.alpha_composite(bg, pil)
        pil = pil.convert("RGB")
        ow, oh = pil.size  # width, height

        # compute resize that preserves aspect ratio
        if mode == "pad":
            if ow >= oh:
                rw = target_size
                rh = int(round(oh * (rw / ow) / 14.0) * 14)
            else:
                rh = target_size
                rw = int(round(ow * (rh / oh) / 14.0) * 14)
        else:  # crop
            rw = target_size
            rh = int(round(oh * (rw / ow) / 14.0) * 14)

        # ensure at least 1
        rw = max(1, rw)
        rh = max(1, rh)

        pil_resized = pil.resize((rw, rh), Image.Resampling.BICUBIC)
        t = to_t(pil_resized)  # C,H,W, values [0,1]

        pad_left = pad_top = pad_right = pad_bottom = 0
        if mode == "pad":
            h_padding = target_size - t.shape[1]
            w_padding = target_size - t.shape[2]
            if h_padding < 0 or w_padding < 0:
                t = torch.nn.functional.interpolate(t.unsqueeze(0), size=(target_size, target_size), mode='bilinear', align_corners=False)[0]
                rw, rh = target_size, target_size
            else:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left
                if h_padding > 0 or w_padding > 0:
                    t = torch.nn.functional.pad(t, (pad_left, pad_right, pad_top, pad_bottom), value=1.0)

        images.append(t)
        shapes.add((t.shape[1], t.shape[2]))
        metas.append({
            "orig_size": (ow, oh),
            "resized_size": (rw, rh),
            "pad": (pad_left, pad_top, pad_right, pad_bottom)
        })

    if len(shapes) > 1:
        max_h = max(s[0] for s in shapes)
        max_w = max(s[1] for s in shapes)
        new_images = []
        for img in images:
            h_padding = max_h - img.shape[1]
            w_padding = max_w - img.shape[2]
            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left
                img = torch.nn.functional.pad(img, (pad_left, pad_right, pad_top, pad_bottom), value=1.0)
            new_images.append(img)
        images = new_images

    imgs_tensor = torch.stack(images)  # S,3,H,W
    return imgs_tensor, metas


# ------------------------------
# 一个 window 的推理函数：返回 depth, conf, metas, extrinsics, intrinsics（全部在 CPU 上）
# ------------------------------
def predict_window_depth(image_paths):
    imgs, metas = load_and_preprocess_with_meta(image_paths, mode=MODE, target_size=TARGET_SIZE)
    imgs = imgs.to(device)  # [S,3,H,W]
    # 为 aggregator 提供 batch 维度：[1, S, 3, H, W]
    imgs_batch = imgs.unsqueeze(0)

    with torch.no_grad():

        with torch.cuda.amp.autocast(enabled=(dtype != torch.float32), dtype=(dtype if dtype in (torch.float16, torch.bfloat16) else None)):
            # aggregator -> camera head -> depth head
            aggregated_tokens_list, ps_idx = model.aggregator(imgs_batch)
            # camera encoding -> extrinsic/intrinsic
            pose_enc = model.camera_head(aggregated_tokens_list)[-1]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, imgs_batch.shape[-2:])
            depth_map, depth_conf = model.depth_head(aggregated_tokens_list, imgs_batch, ps_idx)

    # normalize shapes and move to cpu
    # extrinsic: [1, S, 4, 4] -> squeeze batch -> [S,4,4]
    extrinsic = extrinsic.squeeze(0).cpu().numpy()
    intrinsic = intrinsic.squeeze(0).cpu().numpy()

    # depth_map: [1, S, H, W, 1] or [1, S, H, W] depending; handle common cases
    # ensure final depth: [S, H, W]
    depth = depth_map
    if isinstance(depth, torch.Tensor):
        depth = depth.cpu()
    # remove trailing channel dim if present
    if depth.dim() == 5:  # [B,S,H,W,1]
        depth = depth[0, ..., 0]
    elif depth.dim() == 4 and depth.shape[-1] == 1:  # [S,H,W,1]
        depth = depth[..., 0]
    elif depth.dim() == 4 and depth.shape[0] == 1:  # [1,S,H,W]
        depth = depth[0]
    # depth now tensor on CPU shape [S,H,W]
    depth = depth.numpy() if isinstance(depth, torch.Tensor) else depth

    # depth_conf similar
    conf = depth_conf
    if isinstance(conf, torch.Tensor):
        conf = conf.cpu()
    if conf is not None:
        if conf.dim() == 4 and conf.shape[0] == 1:  # [1,S,H,W]
            conf = conf[0]
        conf = conf.numpy()

    return depth, conf, metas, extrinsic, intrinsic


def process_one_scene(image_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    poses_out_dir = os.path.join(output_dir, "poses")
    os.makedirs(poses_out_dir, exist_ok=True)

    img_paths = sorted([
        str(p) for p in Path(image_dir).iterdir()
        if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ])
    N = len(img_paths)
    print(f"Found {N} images in {image_dir}")

    if N == 0:
        print("Skip empty scene.")
        return

    acc_depth = [None] * N
    acc_conf = [None] * N
    # store per-image extrinsic/intrinsic (numpy)
    acc_extrinsic = [None] * N
    acc_intrinsic = [None] * N

    for start in tqdm(range(0, N, WINDOW_SIZE), desc=f"windows ({Path(image_dir).parent.name})"):
        end = min(start + WINDOW_SIZE, N)
        window_paths = img_paths[start:end]

        depth_maps, depth_confs, metas, extrinsics, intrinsics = predict_window_depth(window_paths)

        for i in range(len(window_paths)):
            idx = start + i
            meta = metas[i]
            depth_pred = depth_maps[i]
            conf_pred = depth_confs[i] if depth_confs is not None else np.ones_like(depth_pred)

            pad_left, pad_top, pad_right, pad_bottom = meta["pad"]
            rw, rh = meta["resized_size"]

            Ht, Wt = depth_pred.shape
            x0, y0 = pad_left, pad_top
            x1, y1 = min(Wt, x0 + rw), min(Ht, y0 + rh)

            cropped_depth = depth_pred[y0:y1, x0:x1]
            cropped_conf = conf_pred[y0:y1, x0:x1]

            ow, oh = meta["orig_size"]
            if cropped_depth.size == 0:
                continue

            # resize back to original image size
            depth_orig = cv2.resize(cropped_depth, (ow, oh), interpolation=cv2.INTER_LINEAR)
            conf_orig = cv2.resize(cropped_conf, (ow, oh), interpolation=cv2.INTER_LINEAR)

            if acc_depth[idx] is None:
                acc_depth[idx] = depth_orig * conf_orig
                acc_conf[idx] = conf_orig
            else:
                acc_depth[idx] += depth_orig * conf_orig
                acc_conf[idx] += conf_orig

            # store poses (one per image). extrinsics/intrinsics for the window are numpy arrays
            try:
                extr_np = extrinsics[i]  # shape (4,4)
                intr_np = intrinsics[i]  # shape (3,3)
                acc_extrinsic[idx] = extr_np
                acc_intrinsic[idx] = intr_np
            except Exception as e:
                # safety: if shapes don't match, skip storing
                print(f"[WARN] could not store pose for idx {idx}: {e}")

    print("Saving depth PNGs and poses...")
    for i, path in enumerate(tqdm(img_paths)):
        base = os.path.splitext(os.path.basename(path))[0]

        # save pose if exists
        if acc_extrinsic[i] is not None:
            np.save(os.path.join(poses_out_dir, f"{base}_extrinsic.npy"), acc_extrinsic[i].astype(np.float64))
        if acc_intrinsic[i] is not None:
            np.save(os.path.join(poses_out_dir, f"{base}_intrinsic.npy"), acc_intrinsic[i].astype(np.float64))

        if acc_depth[i] is None:
            continue

        depth_final = acc_depth[i] / (acc_conf[i] + 1e-8)
        depth_final[acc_conf[i] < CONF_THRESH] = 0.0

        valid = depth_final > 0
        if np.any(valid):
            vmin = depth_final[valid].min()
            vmax = np.percentile(depth_final[valid], 99)
            depth_norm = np.clip((depth_final - vmin) / (vmax - vmin + 1e-8), 0, 1)
        else:
            depth_norm = depth_final

        depth_u16 = (depth_norm * 65535.0).astype(np.uint16)
        out_path = os.path.join(output_dir, base + ".png")
        cv2.imwrite(out_path, depth_u16)

    print(f"Scene done: {output_dir}. Poses saved to: {poses_out_dir}")


if __name__ == "__main__":
    # ================== 手动指定要处理的场景 ==================
    SCENE_NAMES = [
        "abandonedfactory",
        "ocean",
        "seasonsforest",
    ]
    # =========================================================

    print(f"Scenes to process: {SCENE_NAMES}")

    for scene_name in SCENE_NAMES:
        scene_dir = Path(ROOT_DIR) / scene_name
        image_dir = scene_dir / IMAGE_SUBDIR
        output_dir = scene_dir / OUTPUT_SUBDIR

        if not image_dir.exists():
            print(f"[Skip] {scene_name}: {image_dir} not found")
            continue

        print(f"\n=== Processing scene: {scene_name} ===")
        process_one_scene(str(image_dir), str(output_dir))

    print("\nAll specified scenes finished.")
