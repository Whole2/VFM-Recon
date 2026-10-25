# This file is derived from [Atlas](https://github.com/magicleap/Atlas).
# Originating Author: Zak Murez (zak.murez.com)
# Modified for [NeuralRecon](https://github.com/zju3dv/NeuralRecon) by Yiming Xie.

# Original header:
# Copyright 2020 Magic Leap, Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
os.environ["PYOPENGL_PLATFORM"] = "osmesa"

import numpy as np
import pyrender
import torch
import trimesh

import open3d as o3d
import multiprocessing
import cv2
import tqdm
from rich.progress import Progress
torch.multiprocessing.set_sharing_strategy('file_system')

def pirnt_metrics(fname):
    key_names = ['AbsRel', 'AbsDiff', 'SqRel', 'RMSE', 'LogRMSE', 'r1', 'r2', 'r3', 'complete', 'dist1', 'dist2',
                 'prec', 'recal', 'fscore']

    metrics = json.load(open(fname, 'r'))
    metrics = sorted([(scene, metric) for scene, metric in metrics.items()], key=lambda x: x[0])
    scenes = [m[0] for m in metrics]
    metrics = [m[1] for m in metrics]

    keys = metrics[0].keys()
    metrics1 = {m: [] for m in keys}
    for m in metrics:
        for k in keys:
            metrics1[k].append(m[k])

    for k in key_names:
        if k in metrics1:
            v = np.nanmean(np.array(metrics1[k]))
        else:
            v = np.nan
        print('%10s %0.3f' % (k, v))

def eval_mesh(file_pred, file_trgt, threshold=.05, down_sample=.02):
    """ Compute Mesh metrics between prediction and target.

    Opens the Meshs and runs the metrics

    Args:
        file_pred: file path of prediction
        file_trgt: file path of target
        threshold: distance threshold used to compute precision/recal
        down_sample: use voxel_downsample to uniformly sample mesh points

    Returns:
        Dict of mesh metrics
    """

    pcd_pred = o3d.io.read_point_cloud(file_pred)
    pcd_trgt = o3d.io.read_point_cloud(file_trgt)
    if down_sample:
        pcd_pred = pcd_pred.voxel_down_sample(down_sample)
        pcd_trgt = pcd_trgt.voxel_down_sample(down_sample)
    verts_pred = np.asarray(pcd_pred.points)
    verts_trgt = np.asarray(pcd_trgt.points)

    _, dist1 = nn_correspondance(verts_pred, verts_trgt)
    _, dist2 = nn_correspondance(verts_trgt, verts_pred)
    dist1 = np.array(dist1)
    dist2 = np.array(dist2)

    precision = np.mean((dist2 < threshold).astype('float'))
    recal = np.mean((dist1 < threshold).astype('float'))
    fscore = 2 * precision * recal / (precision + recal)
    metrics = {'dist1': np.mean(dist2),
               'dist2': np.mean(dist1),
               'prec': precision,
               'recal': recal,
               'fscore': fscore,
               }
    return metrics


def nn_correspondance(verts1, verts2):
    """ for each vertex in verts2 find the nearest vertex in verts1

    Args:
        nx3 np.array's

    Returns:
        ([indices], [distances])

    """

    indices = []
    distances = []
    if len(verts1) == 0 or len(verts2) == 0:
        return indices, distances

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(verts1)
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    for vert in verts2:
        _, inds, dist = kdtree.search_knn_vector_3d(vert, 1)
        indices.append(inds[0])
        distances.append(np.sqrt(dist[0]))

    return indices, distances


def eval_depth(depth_pred, depth_trgt):
    """ Computes 2d metrics between two depth maps

    Args:
        depth_pred: mxn np.array containing prediction
        depth_trgt: mxn np.array containing ground truth

    Returns:
        Dict of metrics
    """
    mask1 = depth_pred > 0  # ignore values where prediction is 0 (% complete)
    mask = (depth_trgt < 10) * (depth_trgt > 0) * mask1

    depth_pred = depth_pred[mask]
    depth_trgt = depth_trgt[mask]
    abs_diff = np.abs(depth_pred - depth_trgt)
    abs_rel = abs_diff / depth_trgt
    sq_diff = abs_diff ** 2
    sq_rel = sq_diff / depth_trgt
    sq_log_diff = (np.log(depth_pred) - np.log(depth_trgt)) ** 2
    thresh = np.maximum((depth_trgt / depth_pred), (depth_pred / depth_trgt))
    r1 = (thresh < 1.25).astype('float')
    r2 = (thresh < 1.25 ** 2).astype('float')
    r3 = (thresh < 1.25 ** 3).astype('float')

    metrics = {}
    metrics['AbsRel'] = np.mean(abs_rel)
    metrics['AbsDiff'] = np.mean(abs_diff)
    metrics['SqRel'] = np.mean(sq_rel)
    metrics['RMSE'] = np.sqrt(np.mean(sq_diff))
    metrics['LogRMSE'] = np.sqrt(np.mean(sq_log_diff))
    metrics['r1'] = np.mean(r1)
    metrics['r2'] = np.mean(r2)
    metrics['r3'] = np.mean(r3)
    metrics['complete'] = np.mean(mask1.astype('float'))

    return metrics

class Renderer():
    """OpenGL mesh renderer

    Used to render depthmaps from a mesh for 2d evaluation
    """

    def __init__(self, height=480, width=640):
        print("line171")
        self.renderer = pyrender.OffscreenRenderer(width, height)
        print("line173")
        self.scene = pyrender.Scene()
        print("line175")
        # self.render_flags = pyrender.RenderFlags.SKIP_CULL_FACES

    def __call__(self, height, width, intrinsics, pose, mesh):
        self.renderer.viewport_height = height
        self.renderer.viewport_width = width
        self.scene.clear()
        self.scene.add(mesh)
        cam = pyrender.IntrinsicsCamera(cx=intrinsics[0, 2], cy=intrinsics[1, 2],
                                        fx=intrinsics[0, 0], fy=intrinsics[1, 1])
        self.scene.add(cam, pose=self.fix_pose(pose))
        return self.renderer.render(self.scene)  # , self.render_flags)

    def fix_pose(self, pose):
        # 3D Rotation about the x-axis.
        t = np.pi
        c = np.cos(t)
        s = np.sin(t)
        R = np.array([[1, 0, 0],
                      [0, c, -s],
                      [0, s, c]])
        axis_transform = np.eye(4)
        axis_transform[:3, :3] = R
        return pose @ axis_transform

    def mesh_opengl(self, mesh):
        return pyrender.Mesh.from_trimesh(mesh)

    def delete(self):
        self.renderer.delete()


def process(scene, pred_path, gt_path, width=640, height=480, max_depth=3):
    """ Process a single scene for evaluation

    Args:
        pred_path: path to predicted mesh
        scene: scene name
        gt_path: path to ground truth
        total_scenes_index: index of current scene in total list
        total_scenes_count: total number of scenes
        width: width of depthmap
        height: height of depthmap
        max_depth: maximum depth value to consider

    Returns:
        scene name and metrics
    """
    test_framid = os.listdir(os.path.join(gt_path, scene, 'color'))
    n_imgs = len(test_framid)
    intrinsic_dir = os.path.join(gt_path, scene, 'intrinsic', 'intrinsic_depth.txt')
    cam_intr = np.loadtxt(intrinsic_dir, delimiter=' ')[:3, :3]
    voxel_size = 4
    print("line225")
    # re-fuse to remove hole filling since filled holes are penalized in
    # mesh metrics
    # tsdf_fusion = TSDFFusion(vol_dim, float(voxel_size)/100, origin, color=False)
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=float(voxel_size) / 100,
        sdf_trunc=3 * float(voxel_size) / 100,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    print("line233")
    mesh_file = os.path.join(pred_path, '%s.ply' % scene.replace('/', '-'))
    print("line235")
    mesh = trimesh.load(mesh_file, process=False)
    print("line237")
    # mesh renderer
    renderer = Renderer()
    print("line240")
    mesh_opengl = renderer.mesh_opengl(mesh)
    print("line242")
    for i in range(n_imgs):
        cam_pose = np.loadtxt(os.path.join(gt_path, scene, "pose", f"{i}.txt"), delimiter=' ')

        # Read depth image and camera pose
        depth_im = cv2.imread(os.path.join(gt_path, scene, "depth", f"{i}.png"), -1).astype(np.float32)
        depth_im /= 1000.  # depth is saved in 16-bit PNG in millimeters
        depth_im[depth_im > max_depth] = 0

        # Read RGB image
        # color_image = cv2.cvtColor(cv2.imread(os.path.join(gt_path, scene, "color", str(id) + ".jpg")), cv2.COLOR_BGR2RGB)
        # color_image = cv2.resize(color_image, (depth_im.shape[1], depth_im.shape[0]), interpolation=cv2.INTER_AREA)
        if cam_pose[0][0] == np.inf or cam_pose[0][0] == -np.inf or cam_pose[0][0] == np.nan:
            continue

        _, depth_pred = renderer(height, width, cam_intr, cam_pose, mesh_opengl)

        temp = eval_depth(depth_pred, depth_im)
        if i == 0:
            metrics_depth = temp
        else:
            metrics_depth = {key: value + temp[key] for key, value in metrics_depth.items()}

        # placeholder
        color_im = np.repeat(depth_pred[:, :, np.newaxis] * 255, 3, axis=2).astype(np.uint8)
        depth_pred = o3d.geometry.Image(depth_pred)
        color_im = o3d.geometry.Image(color_im)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color_im, depth_pred, depth_scale=1.0,
                                                                  depth_trunc=5.0,
                                                                  convert_rgb_to_intensity=False)

        volume.integrate(
            rgbd,
            o3d.camera.PinholeCameraIntrinsic(width=width, height=height, fx=cam_intr[0, 0], fy=cam_intr[1, 1],
                                              cx=cam_intr[0, 2],
                                              cy=cam_intr[1, 2]), np.linalg.inv(cam_pose))

    metrics_depth = {key: value / n_imgs for key, value in metrics_depth.items()}

    # save trimed mesh
    file_mesh_trim = os.path.join(pred_path, '%s_trim_single.ply' % scene.replace('/', '-'))
    o3d.io.write_triangle_mesh(file_mesh_trim, volume.extract_triangle_mesh())

    # eval trimed mesh
    file_mesh_trgt = os.path.join(gt_path, scene, scene + '_vh_clean.ply')
    metrics_mesh = eval_mesh(file_mesh_trim, file_mesh_trgt)

    metrics = {**metrics_depth, **metrics_mesh}

    # rslt_file = os.path.join(pred_path, '%s_metrics.json' % scene.replace('/', '-'))
    # json.dump(metrics, open(rslt_file, 'w'))

    return scene, metrics


def process_with_single_worker(scenes_split, w_idx, pred_path, gt_path, max_depth):
    metrics = {}
    print("line296")
    with Progress() as p:
        print("line298")
        task = p.add_task(f"Processing {w_idx}", total=len(scenes_split), visible=False)
        print("line300")
        for i, scene in tqdm.tqdm(enumerate(scenes_split), nrows=w_idx+1, total=len(scenes_split)):
            print("line302")
            scene, temp = process(scene, pred_path, gt_path, width=640, height=480, max_depth=max_depth)
            print("line304")
            if temp is not None:
                metrics[scene] = temp
            p.update(task, advance=1)
    return metrics


def split_list(_list, n):
    assert len(_list) >= n
    ret = [[] for _ in range(n)]
    for idx, item in enumerate(_list):
        ret[idx % n].append(item)
    return ret


def evaluate_scannet_mesh(pred_path, gt_path, max_depth=3.0, n_proc=1):
    split = os.path.join(gt_path, "xtk/splits/scannetv2_test.txt")
    with open(split, 'r') as f:
        scenes = f.readlines()
    scenes = [s.strip() for s in scenes]
    scenes_split = split_list(scenes, n_proc)

    gt_path = os.path.join(gt_path, "scans")
    with multiprocessing.Pool(processes=n_proc) as pool:
        results = pool.starmap(process_with_single_worker, [
            (scenes_split[w_idx], w_idx, pred_path, gt_path, max_depth)
            for w_idx in range(n_proc)
        ])

    metrics = {}
    for r in results:
        metrics.update(r)

    rslt_file = os.path.join(pred_path, 'metrics.json')
    json.dump(metrics, open(rslt_file, 'w'))

    pirnt_metrics(rslt_file)
    return metrics    

if __name__ == "__main__":
    import argparse
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--pred_path', type=str, required=True, help='Path to predicted meshes')
    arg_parser.add_argument('--gt_path', type=str, required=True, help='Path to ScanNet dataset')
    arg_parser.add_argument('--max_depth', type=float, default=3.0, help='Max depth to consider')
    arg_parser.add_argument('--n_proc', type=int, default=8, help='Number of processes to use')
    args = arg_parser.parse_args()
    
    evaluate_scannet_mesh(
        args.pred_path, args.gt_path, max_depth=args.max_depth, n_proc=args.n_proc)
    rslt_file = os.path.join(args.pred_path, 'metrics.json')
    pirnt_metrics(rslt_file)
