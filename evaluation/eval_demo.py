import os.path as osp
import pickle
import sys
import zarr

from deepv2d.data_stream.neufu_demo import DemoDataset
import open3d as o3d

sys.path.append('deepv2d')

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import cv2
import os
import time
import argparse
import glob

from deepv2d import vis
from deepv2d.core import config
from deepv2d.data_stream.scannet import ScanNet
from deepv2d.deepv2d import DeepV2D


def make_predictions(args):
    cfg = config.cfg_from_file(args.cfg)
    deepv2d = DeepV2D(cfg, args.model, use_fcrn=True, mode=args.mode)
    print('making predictions...')
    os.makedirs(args.output_dir, exist_ok=True)
    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.allow_growth = True
    with tf.Session(config=tfconfig) as sess:
        deepv2d.set_session(sess)

        depth_predictions, pose_predictions, data_blobs = [], [], []
        db = DemoDataset(args.data_path, args.scene)
        bar = trange(len(db))
        for i in bar:
            if i < len(db) * args.start:
                continue
            test_data_blob = db[i]

            scanid, imageid_1, imageid_2 = test_data_blob['scanid'], test_data_blob['imageid_1'], test_data_blob[
                'imageid_2']
            bar.set_postfix_str(f'scanid {scanid}')
            # check cache
            depth_cache_path = osp.join(args.output_dir, f'{scanid}_{imageid_1}_{imageid_2}_depth.zarr')
            pose_cache_path = osp.join(args.output_dir, f'{scanid}_{imageid_1}_{imageid_2}_pose.zarr')
            if not args.ignore_cache and osp.exists(depth_cache_path) and osp.exists(pose_cache_path):
                depth_pred = zarr.load(depth_cache_path)
                poses_pred = zarr.load(pose_cache_path)
            else:
                test_blob = db[i]
                images, intrinsics = test_blob['images'], test_blob['intrinsics']
                camposes = test_blob['poses']
                depth_pred, poses_pred = deepv2d(images, intrinsics,
                                                 camposes=camposes, iters=1)
                zarr.save(depth_cache_path, depth_pred)
                zarr.save(pose_cache_path, poses_pred)

            # use keyframe depth for evaluation
            depth_predictions.append(depth_pred[0])

            # BA-Net evaluates pose as the relative transformation between two frames
            delta_pose = poses_pred[1] @ np.linalg.inv(poses_pred[0])
            pose_predictions.append(delta_pose)

            data_blobs.append({'scanid': scanid,
                               'imageid_1': imageid_1,
                               'imageid_2': imageid_2})

    predictions = (depth_predictions, pose_predictions, data_blobs)
    return predictions


def reconstruct_one_scene(args):
    predictions_dir = args.output_dir

    output_path = f'demo_mesh/{args.scene}'
    output_path += 'posegt'

    if osp.exists(output_path + '.ply'):
        return
    scandir = osp.join(args.data_path, args.scene)
    paths = list(os.listdir(predictions_dir))
    this_scan_output_paths = list(filter(lambda path: args.scene in path and 'depth' in path, paths))
    this_scan_output_paths = sorted(this_scan_output_paths, key=lambda x: int(x.split('_')[1]))
    camera_poses_gt = []
    # camera_poses_pred = []
    depths = []
    images = []
    # total_pose = np.eye(4)
    for path in this_scan_output_paths:
        imgid1 = int(path.split('_')[1])
        depth_pred = zarr.load(osp.join(predictions_dir, path))
        depths.append(depth_pred[0] * 1000)
        posegt = np.loadtxt(os.path.join(args.data_path, args.scene, 'poses', "{:0>5d}.txt".format(imgid1)),
                            dtype='f', delimiter=' ')
        camera_poses_gt.append(posegt)

        # pose = zarr.load(osp.join(predictions_dir, path.replace('depth', 'pose')))[1]
        # pose = total_pose = pose @ total_pose
        # camera_poses_pred.append(pose)

        imgpath = os.path.join(scandir, 'images', '%05d.jpg' % imgid1)
        image = cv2.imread(imgpath)
        image = cv2.resize(image, (640, 480))
        images.append(image)

    camera_poses = camera_poses_gt  # actually gt
    ds = DemoDataset(args.data_path, args.scene)
    # K = ds.intrinsics
    # K = np.loadtxt(depth_intrinsics, delimiter=' ')
    fx, fy, cx, cy = ds.intrinsics

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='cfgs/scannet.yaml', help='config file used to train the model')
    parser.add_argument('--model', default='models/scannet.ckpt', help='path to model checkpoint')
    # parser.add_argument('--dataset_dir', help='path to scannet dataset',
    #                     default='/data/scannet/scannet/tvt')

    parser.add_argument('--mode', default='keyframe', help='config file used to train the model')
    parser.add_argument('--fcrn', action="store_true", help='use single image depth initializiation')
    # parser.add_argument('--n_iters', type=int, default=8, help='number of video frames to use for reconstruction')
    parser.add_argument('--viz', action="store_true", help='display depth maps during inference')
    # parser.add_argument('--recons', action="store_true", help='reconstruction task')
    parser.add_argument('--ignore_cache', action="store_true")
    parser.add_argument('--start', type=float, default=0.0)
    parser.add_argument('--data_path', type=str, default='/raid/Datasets/neufu_demo')
    # parser.add_argument('--input_campose', default=False, action='store_true')
    parser.add_argument('--output_dir', default='demo_output', type=str)
    parser.add_argument('--scene', default='2020-11-20T20-17-05', type=str)
    parser.add_argument('--recons_only', default=False, action='store_true')
    # parser.add_argument('--test_split', default='deepv2d', type=str,choices=['deepv2d','official'])
    args = parser.parse_args()
    if not args.recons_only:
        predictions = make_predictions(args)

    reconstruct_one_scene(args)
    # if not args.recons:
    #     evaluate(groundtruth, predictions)


if __name__ == '__main__':
    main()
