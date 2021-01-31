from multiprocessing import Pool

import matplotlib.pyplot as plt
import cv2
import numpy as np
import os.path as osp
import os
import open3d as o3d

import zarr
from tqdm import tqdm, trange

from deepv2d.data_stream.sevenscenes import SevenScenes
from deepv2d.utils.io import load_test_scanids
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--inputdir', default='/home/xieyiming/repo/DeepV2D/7scenes_output')
parser.add_argument('--fast', default=False, action='store_true')
parser.add_argument('--inv_pose', default=False, action='store_true')
args = parser.parse_args()


def reconstruct_one_scene(scene_idx):
    root = '/data/7scenes'
    ss = SevenScenes(root, scene_idx, False)
    depth_dir = args.inputdir

    output_path = f'{depth_dir}_mesh/{ss.scene}/{ss.seq}.ply'

    # paths = list(os.listdir(depth_dir))
    camera_poses = []
    depths = []
    images = []
    for scene_name, imgid1, imgid2 in ss.test_data:
        filepath = ss.file_paths[imgid1]
        rgb, cam = ss.db.load_sample(filepath, 480, 640)
        images.append(rgb)
        depth_pred_path = osp.join(depth_dir, scene_name + "_" + str(imgid1) + "_" + str(imgid2) + '_depth.zarr')
        depth_pred = zarr.load(depth_pred_path)
        assert depth_pred is not None
        depths.append(depth_pred[0] * 1000)
        # pose_pred_path = scene_name + "_" + str(imgid1) + "_" + str(imgid2) + '_pose.zarr'
        # pose_pred = zarr.load(pose_pred_path)
        pose, intrinsics = cam
        camera_poses.append(pose)
        fx, fy, cx, cy = ss.intrinsics

    # print('reconstructing...')
    voxel_length = 0.04
    sdf_trunc = voxel_length * 3
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    bar = trange(len(camera_poses), leave=False)
    for i in bar:
        bar.set_postfix_str(f'Integrate {i}')

        color: np.ndarray = images[i]
        depth: np.ndarray = depths[i]
        color = o3d.geometry.Image(color)
        depth = o3d.geometry.Image(depth)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, depth_trunc=4.0, convert_rgb_to_intensity=False)
        cp = camera_poses[i]
        if args.inv_pose:
            cp = np.linalg.inv(cp)
        volume.integrate(
            rgbd,
            o3d.camera.PinholeCameraIntrinsic(640, 480, fx, fy, cx, cy),
            cp)

    # print("Extract a triangle mesh from the volume and visualize it.")
    mesh: o3d.geometry.TriangleMesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    os.makedirs(osp.dirname(output_path), exist_ok=True)
    o3d.io.write_triangle_mesh(output_path, mesh)


def main():
    scene_idxs = range(18)
    if not args.fast:
        bar = tqdm(scene_idxs)
        for scene_idx in bar:
            bar.set_postfix_str(f'scanid {scene_idx}')
            reconstruct_one_scene(scene_idx)
    else:
        with Pool(8) as p:
            results = list(tqdm(p.imap_unordered(reconstruct_one_scene, scene_idxs), total=len(scene_idxs)))


if __name__ == '__main__':
    main()
