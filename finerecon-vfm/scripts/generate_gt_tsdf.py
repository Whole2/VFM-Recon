import argparse
import glob
import os

import cv2
import numpy as np
import torch
import tqdm

from tsdf_fusion import TSDFVolumeTorch


parser = argparse.ArgumentParser()
parser.add_argument("--dataset-dir", required=True)
parser.add_argument("--output-dir", required=True)
args = parser.parse_args()

# depth_dirs = sorted(
#     [d for d in glob.glob(os.path.join(args.dataset_dir, "pred_depths/depths", "*")) if os.path.isdir(d)])
# scan_dirs = [d.replace("pred_depths/depths", "scans") for d in depth_dirs]
scan_dirs = sorted(
    [d for d in glob.glob(os.path.join(args.dataset_dir, "scans", "*")) if os.path.isdir(d)])
depth_dirs = [d.replace("scans", "pred_depths/depths") for d in scan_dirs]
# print(f"{len(scan_dirs)}")

scan_dirs = scan_dirs[1300:]
depht_dirs = depth_dirs[1300:]

MAX_DEPTH = 3.5
voxel_size = 0.02
margin = int(np.round(0.04 / voxel_size))
device = "cuda"

os.makedirs(args.output_dir, exist_ok=True)

for scan_dir, depth_dir in tqdm.tqdm(zip(scan_dirs, depth_dirs), total=len(scan_dirs)):
    # if scan_dir.split("/")[-1]!= "scene0532_00":
    #     continue
    scene_name = os.path.basename(scan_dir)
    outfile = os.path.join(args.output_dir, f"{scene_name}.npz")
    if os.path.exists(outfile):
        print(f"skipping existing file: {outfile}")
        continue

    print(f"processing {scene_name}")

    sort_key = lambda f: int(os.path.basename(f).split(".")[0])
    # valid_frames = [int(f.split()[1]) for f in open(os.path.join(scan_dir, "valid_frames.txt")).readlines()]
    # rgb_imgfiles = [os.path.join(scan_dir, "color", f"{f}.jpg") for f in valid_frames]
    rgb_imgfiles = sorted(
        glob.glob(os.path.join(scan_dir, "color/*.jpg")), key=sort_key)
    # depth_imgfiles = [os.path.join(scan_dir, "depth", f"{f}.png") for f in valid_frames]
    depth_imgfiles = sorted(
        glob.glob(os.path.join(scan_dir, "depth/*.png")), key=sort_key)
    assert len(rgb_imgfiles) == len(depth_imgfiles), "RGB and depth image counts do not match, got {} and {}".format(
        len(rgb_imgfiles), len(depth_imgfiles)
    )
    
    pose_files = sorted(glob.glob(os.path.join(scan_dir, "pose/*.txt")), key=sort_key)
    # pose_files = [os.path.join(scan_dir, "pose", f"{f}.txt") for f in valid_frames]

    intr_file = os.path.join(scan_dir, "intrinsic/intrinsic_depth.txt")
    # intr_file = os.path.join(depth_dir, "intrinsic_depth.txt")

    imheight, imwidth = cv2.imread(depth_imgfiles[0], cv2.IMREAD_ANYDEPTH).shape

    K = np.loadtxt(intr_file)[:3, :3]

    u = np.arange(0, imwidth, 10)
    v = np.arange(0, imheight, 10)
    uu, vv = np.meshgrid(u, v)
    uv = np.c_[uu.flatten(), vv.flatten()]
    pix_vecs = (np.linalg.inv(K) @ np.c_[uv, np.ones((len(uv), 1))].T).T

    pts = []
    for i in tqdm.trange(0, len(pose_files), 10, leave=False, desc='computing scene bounds'):
        pose = np.loadtxt(pose_files[i])
        if np.any(np.isinf(pose)):
            # raise ValueError("Invalid pose")
            continue
        depth_img = cv2.imread(depth_imgfiles[i], cv2.IMREAD_ANYDEPTH).astype(np.float32) / 1000.0
        depth_img[depth_img > MAX_DEPTH] = 0
        depth = depth_img[uv[:, 1], uv[:, 0]]
        valid = depth > 0
        xyz_cam = pix_vecs[valid] * depth[valid, None]
        xyz = (pose @ np.c_[xyz_cam, np.ones((len(xyz_cam), 1))].T).T[:, :3]
        pts.append(xyz)

    pts = np.concatenate(pts, axis=0)

    minbound = np.min(pts, axis=0) - 3 * margin * voxel_size
    maxbound = np.max(pts, axis=0) + 3 * margin * voxel_size

    voxel_dim = torch.from_numpy(np.ceil((maxbound - minbound) / voxel_size)).int()
    origin = torch.from_numpy(minbound).float()

    torch.cuda.empty_cache()
    try:
        tsdf_vol = TSDFVolumeTorch(
            voxel_dim.to(device),
            origin.to(device),
            voxel_size,
            margin=margin,
            device=device,
        )
    except Exception as e:
        print(e)
        continue

    for i in tqdm.trange(len(pose_files), leave=False, desc='TSDF fusion'):
        pose = np.loadtxt(pose_files[i])
        if np.any(np.isinf(pose)):
            continue
        depth_img = (
            cv2.imread(depth_imgfiles[i], cv2.IMREAD_ANYDEPTH).astype(np.float32) / 1000
        )
        depth_img[depth_img > MAX_DEPTH] = 0
        tsdf_vol.integrate(
            torch.from_numpy(depth_img),
            torch.from_numpy(K).float(),
            torch.from_numpy(pose).float(),
            1,
        )

    tsdf, weight = tsdf_vol.get_volume()

    tsdf[weight == 0] = torch.nan

    unobserved_col_mask = (
        (weight == 0).all(dim=-1, keepdim=True).repeat(1, 1, tsdf.shape[-1])
    )
    tsdf[unobserved_col_mask] = -1

    maxbound = origin + voxel_size * torch.tensor(tsdf.shape)

    np.savez(
        outfile,
        tsdf=tsdf.cpu().numpy(),
        origin=origin.numpy(),
        voxel_size=voxel_size,
        maxbound=maxbound.numpy(),
    )

    # from skimage.measure import marching_cubes
    # v,f, _, _ = marching_cubes(tsdf.cpu().numpy(), 0, spacing=(voxel_size,)*3, mask=(tsdf!=-1).cpu().numpy())
    # v += origin.numpy()
    # import open3d as o3d
    # mesh = o3d.geometry.TriangleMesh()
    # mesh.vertices = o3d.utility.Vector3dVector(v)
    # mesh.triangles = o3d.utility.Vector3iVector(f)
    # mesh.compute_vertex_normals()
    # o3d.io.write_triangle_mesh(os.path.join(args.output_dir, f"{scene_name}.ply"), mesh)
