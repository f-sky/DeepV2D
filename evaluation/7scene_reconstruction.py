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

parser.add_argument('--depth_dir', default='/home/xieyiming/repo/DeepV2D/7scenes_output')
parser.add_argument('--fast', default=False, action='store_true')
args = parser.parse_args()


def reconstruct_one_scene(scene_idx):
    root = '/data/7scenes'
    ss = SevenScenes(root, scene_idx, False)
    depth_dir = args.inputdir

    output_path = f'{depth_dir}_mesh/{ss.scene}/{ss.seq}.ply'

    paths = list(os.listdir(scannet_output))
    this_scan_output_paths = list(filter(lambda path: scanid in path and 'depth' in path, paths))
    this_scan_output_paths = sorted(this_scan_output_paths, key=lambda x: int(x.split('_')[2]))
    camera_poses_gt = []
    camera_poses_pred = []
    depths = []
    images = []
    total_pose = np.eye(4)
    for path in this_scan_output_paths:
        imgid1 = int(path.split('_')[2])
        # if imgid1 > 40:
        #     continue
        depth_pred = zarr.load(osp.join(scannet_output, path))
        depths.append(depth_pred[0] * 1000)

        camera_poses_gt.append(np.loadtxt(os.path.join(scandir, 'pose', '%d.txt' % imgid1), delimiter=' '))

        pose = zarr.load(osp.join(scannet_output, path.replace('depth', 'pose')))[1]
        pose = total_pose = pose @ total_pose
        # pose = np.linalg.inv(pose)  # todo:check
        camera_poses_pred.append(pose)

        imgpath = os.path.join(scandir, 'color', '%d.jpg' % imgid1)
        image = cv2.imread(imgpath)
        image = cv2.resize(image, (640, 480))
        images.append(image)

    camera_poses = camera_poses_gt

    depth_intrinsics = os.path.join(scandir, 'intrinsic/intrinsic_depth.txt')
    K = np.loadtxt(depth_intrinsics, delimiter=' ')
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

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
        volume.integrate(
            rgbd,
            o3d.camera.PinholeCameraIntrinsic(640, 480, fx, fy, cx, cy),
            np.linalg.inv(camera_poses[i]))

    # print("Extract a triangle mesh from the volume and visualize it.")
    mesh: o3d.geometry.TriangleMesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()

    o3d.io.write_triangle_mesh(output_path + '.ply', mesh)


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
