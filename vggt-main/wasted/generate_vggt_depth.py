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

path = "/mnt/archive/users/mingyuhang/organized_datasets/preprocessed/scans/Barn"
# output_dir = os.path.join(path, "vggt_depth")
output_dir = os.path.join(path, "vggt_depth")
os.makedirs(output_dir, exist_ok=True)

images = sorted(glob.glob(os.path.join(path, "images/*.jpg")))
# poses = np.load(os.path.join(path, "poses.npy"))
K = np.loadtxt(os.path.join(path, "intrinsic.txt"))

device = "cuda" if torch.cuda.is_available() else "cpu"
# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

# Initialize the model and load the pretrained weights.
# This will automatically download the model weights the first time it's run, which may take a while.
# model = VGGT.from_pretrained("/mnt/homes/mingyuhang-ldap/.cache/torch/hub/checkpoints/model.pt").to(device)
#xmodified
model = VGGT()
state_dict = torch.hub.load_state_dict_from_url(
    "https://hf-mirror.com/facebook/VGGT-1B/resolve/main/model.pt"
)
model.load_state_dict(state_dict, strict=False)
model.eval()
model = model.to(device)

# Load and preprocess example images (replace with your own image paths)
# image_names = ["path/to/imageA.png", "path/to/imageB.png", "path/to/imageC.png"]  
# image_names = [images[i] for i in np.random.permutation(len(images))[:10]]
images = load_and_preprocess_images(images)

frame_window = 16
total_pcd = []
total_color = []

last_pose = None
n_submaps = len(images) // (frame_window - 1) + 1
print(len(images), len(images) // (frame_window - 1), n_submaps)

# for i in tqdm.tqdm(range(0, len(images), frame_window-1)):
all_poses = []
all_intrinsics = []
for i in range(0,n_submaps):
    # if i > 4:
    #     break
    start = i * (frame_window - 1)
    end = min((i+1) * (frame_window - 1) + 1, len(images))
    images_batch = images[start:end].to(device)
    # print(start, end, len(images), len(images_batch))

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            # Predict attributes including cameras, depth maps, and point maps.
            predictions = model(images_batch)
    
    depth_map = predictions["depth"]
    depth_map = depth_map.squeeze(0).cpu().numpy() * 5000.0
    for j in range(depth_map.shape[0]):
        # index = i * (frame_window - 1) + j
        
        # if i > 0 and 
        filename = os.path.join(output_dir, f"{i:06d}_{j:02d}.png")
        cv2.imwrite(filename, depth_map[j].astype(np.uint16))
    
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images_batch.shape[-2:])
    extrinsic = extrinsic.squeeze(0).cpu().numpy()
    intrinsic = intrinsic.squeeze(0).cpu().numpy()
    extrinsic = np.concatenate([extrinsic, np.array([0, 0, 0, 1]).reshape(1, 1, 4).repeat(len(extrinsic), 0)], axis=1)

    all_poses.append(extrinsic)
    all_intrinsics.append(intrinsic)
        
            # extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images_batch.shape[-2:])
            
            # poses = torch.eye(4, dtype=extrinsic.dtype, device=device).unsqueeze(0).repeat(extrinsic.shape[1], 1, 1)
            # poses[:, :3, :4] = extrinsic.squeeze(0)

            # if last_pose is None:
            #     last_pose = poses[-1]
            # else:
            #     # the first pose of the next submap should be the last pose of the previous submap
            #     ref_pose = poses[0]
            #     delta_pose = last_pose @ ref_pose.inverse()
            #     # convert all poses to the first submap's coordinate system
            #     # assuming the first frame is the reference frame
            #     for j in range(len(poses)):
            #         poses[j] = delta_pose @ poses[j]
                
            #     # update the last pose
            #     last_pose = poses[-1]
            
            # point_map = unproject_depth_map_to_point_map(
            #     depth_map.squeeze(0), poses, intrinsic.squeeze(0))
            
            # point_map = point_map.reshape(-1, 3)
            # color_map = images_batch.permute(0, 2, 3, 1).reshape(-1, 3).detach().cpu().numpy()

            # # downsample the point map for visualization
            # if point_map.shape[0] > 1000:
            #     sel = np.random.permutation(point_map.shape[0])[:10000]
            #     point_map = point_map[sel]
            #     color_map = color_map[sel]
                
            # total_pcd.append(point_map)
            # total_color.append(color_map)
all_poses = np.concatenate(all_poses, axis=0)
all_intrinsics = np.concatenate(all_intrinsics, axis=0)
np.save(os.path.join(output_dir, "poses.npy"), all_poses)
np.save(os.path.join(output_dir, "intrinsics.npy"), all_intrinsics)

# # Merge the point clouds and colors into a single point cloud
# total_pcd = np.concatenate(total_pcd, axis=0)
# total_color = np.concatenate(total_color, axis=0)

# # Visualize the point cloud
# import open3d as o3d
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(total_pcd)
# pcd.colors = o3d.utility.Vector3dVector(total_color)
# o3d.io.write_point_cloud("test.ply", pcd)