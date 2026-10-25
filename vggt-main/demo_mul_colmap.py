import os
import argparse
import traceback
from pathlib import Path
import torch

from demo_colmap import demo_fn


def parse_batch_args():
    parser = argparse.ArgumentParser(description="Batch VGGT Demo for Multiple Scenes")

    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Root directory containing multiple scene subdirectories"
    )
    parser.add_argument(
        "--scene_range",
        type=str,
        default=None,
        help="Range of scenes to process, e.g. 'scene0707-scene0806'"
    )
    parser.add_argument("--use_ba", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_reproj_error", type=float, default=8.0)
    parser.add_argument("--shared_camera", action="store_true", default=False)
    parser.add_argument("--camera_type", type=str, default="SIMPLE_PINHOLE")
    parser.add_argument("--vis_thresh", type=float, default=0.2)
    parser.add_argument("--query_frame_num", type=int, default=8)
    parser.add_argument("--max_query_pts", type=int, default=4096)
    parser.add_argument("--fine_tracking", action="store_true", default=True)
    parser.add_argument("--conf_thres_value", type=float, default=5.0)

    return parser.parse_args()


def is_scene_folder(path: Path):
    """判断是否为场景文件夹（有 images 子目录）"""
    return path.is_dir() and (path / "color").exists()


def extract_scene_main_number(name: str) -> int:
    """
    从 scene0707_00 这种命名中提取主编号 707
    规则：匹配 sceneXXXX 或数字部分
    """
    import re
    m = re.search(r"scene(\d{4})", name)
    if m:
        return int(m.group(1))
    digits = "".join(ch for ch in name if ch.isdigit())
    return int(digits[:4]) if len(digits) >= 4 else -1


def filter_scene_range(scene_dirs, range_str):
    """根据 scene0707_00-scene0806_00 的范围筛选"""
    if range_str is None:
        return scene_dirs

    try:
        start_name, end_name = range_str.split("-")
    except ValueError:
        raise ValueError("scene_range format error, should be like: scene0707-scene0806")

    start_num = extract_scene_main_number(start_name)
    end_num = extract_scene_main_number(end_name)
    if start_num > end_num:
        start_num, end_num = end_num, start_num

    filtered = []
    for p in scene_dirs:
        num = extract_scene_main_number(p.name)
        if start_num <= num <= end_num:
            filtered.append(p)

    return sorted(filtered, key=lambda x: extract_scene_main_number(x.name))


def main():
    args = parse_batch_args()
    root = Path(args.root_dir)
    if not root.exists():
        raise ValueError(f"Root directory does not exist: {root}")

    # 找到所有场景目录
    scene_dirs = sorted([p for p in root.iterdir()])
    #scene_dirs = sorted([p for p in root.iterdir() if is_scene_folder(p)])
    if len(scene_dirs) == 0:
        raise ValueError(f"No valid scene folders found under {root}")

    # 按范围过滤
    scene_dirs = filter_scene_range(scene_dirs, args.scene_range)
    if len(scene_dirs) == 0:
        raise ValueError(f"No scenes matched range: {args.scene_range}")

    print(f"🔍 Found {len(scene_dirs)} scenes to process:")
    for s in scene_dirs:
        print(f"  - {s.name}")
    print("-" * 60)

    # 批量运行
    for idx, scene_dir in enumerate(scene_dirs, 1):
        print(f"\n[{idx}/{len(scene_dirs)}] Processing scene: {scene_dir.name}")
        print("-" * 40)

        try:
            scene_args = argparse.Namespace(
                scene_dir=str(scene_dir),
                seed=args.seed,
                use_ba=args.use_ba,
                max_reproj_error=args.max_reproj_error,
                shared_camera=args.shared_camera,
                camera_type=args.camera_type,
                vis_thresh=args.vis_thresh,
                query_frame_num=args.query_frame_num,
                max_query_pts=args.max_query_pts,
                fine_tracking=args.fine_tracking,
                conf_thres_value=args.conf_thres_value,
            )

            with torch.no_grad():
                demo_fn(scene_args)

            print(f"✅ Scene {scene_dir.name} finished successfully!")

        except Exception:
            print(f"❌ Failed to process scene: {scene_dir.name}")
            traceback.print_exc()
            continue

    print("\n🎯 All selected scenes processed!")


if __name__ == "__main__":
    main()
