import os
import shutil
from pathlib import Path

def select_keyframes(
    root_dir: str, 
    output_dir: str, 
    start_scene: str, 
    end_scene: str, 
    frame_interval: int = 25
):
    """
    从指定序号范围的场景中，每隔 frame_interval 选择一帧作为关键帧，并保存到 output_dir。
    
    Args:
        root_dir: 大文件夹路径，里面包含多个场景文件夹
        output_dir: 保存关键帧的文件夹
        start_scene: 起始场景编号，例如 "scene0707_00"
        end_scene: 结束场景编号，例如 "scene0806_00"
        frame_interval: 间隔帧数（默认25）
    """
    os.makedirs(output_dir, exist_ok=True)

    # 遍历 root_dir 下所有场景文件夹
    scene_folders = sorted([f for f in os.listdir(root_dir) if f.startswith("scene")])
    
    # 筛选指定范围的场景
    selected_scenes = []
    for scene in scene_folders:
        if start_scene <= scene <= end_scene:
            selected_scenes.append(scene)

    if not selected_scenes:
        raise ValueError(f"No scenes found in range {start_scene} to {end_scene}")

    print(f"Selected {len(selected_scenes)} scenes from {start_scene} to {end_scene}")

    # 遍历每个场景
    for scene in selected_scenes:
        color_dir = os.path.join(root_dir, scene, "color")
        if not os.path.exists(color_dir):
            print(f"Warning: {color_dir} does not exist, skipping")
            continue

        # 获取所有图像并排序
        image_paths = sorted(Path(color_dir).glob("*"))
        print(f"{scene}: {len(image_paths)} frames found")

        # 创建场景输出文件夹
        scene_output_dir = os.path.join(output_dir, scene)
        os.makedirs(scene_output_dir, exist_ok=True)

        # 每隔 frame_interval 拷贝一帧
        keyframe_count = 0
        for idx, img_path in enumerate(image_paths):
            if idx % frame_interval == 0:
                dst_path = os.path.join(scene_output_dir, img_path.name)
                shutil.copy(img_path, dst_path)
                keyframe_count += 1

        print(f"{scene}: {keyframe_count} keyframes saved to {scene_output_dir}")

    print("All scenes processed.")


if __name__ == "__main__":
    # 示例调用
    ROOT_DIR = "/mnt/archive/datasets/ScanNet/scans"
    OUTPUT_DIR = "/mnt/archive/users/mingyuhang/vggt/keyframe"
    START_SCENE = "scene0757_00"
    END_SCENE = "scene0757_00"
    FRAME_INTERVAL = 100

    select_keyframes(ROOT_DIR, OUTPUT_DIR, START_SCENE, END_SCENE, FRAME_INTERVAL)
