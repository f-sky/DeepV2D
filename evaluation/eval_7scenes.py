import os.path as osp
import pickle
import sys
import zarr

from deepv2d.utils.timer import Timer

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
from deepv2d.data_stream.sevenscenes import SevenScenes
from deepv2d.deepv2d import DeepV2D

from evaluation import eval_utils


def write_to_folder(images, intrinsics, test_id):
    dest = os.path.join("scannet/%06d" % test_id)

    if not os.path.isdir(dest):
        os.makedirs(dest)

    for i, img in enumerate(images):
        cv2.imwrite(os.path.join(dest, '%d.png' % i), img)

    np.savetxt(os.path.join(dest, 'intrinsics.txt'), intrinsics)


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
        db = SevenScenes(args.dataset_dir, args.scene_idx)
        bar = trange(len(db.test_data))
        for i in bar:
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
                iters = 1
                depth_pred, poses_pred = deepv2d(images, intrinsics,
                                                 camposes=camposes, iters=iters)
                # timer.toc()
                # print('ave', timer.average_time)
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

            # depth_groundtruth.append(test_blob['depth'])
            # pose_groundtruth.append(test_blob['pose'])

    predictions = (depth_predictions, pose_predictions, data_blobs)
    # groundtruth = (depth_groundtruth, pose_groundtruth)
    return groundtruth, predictions


def evaluate(groundtruth, predictions):
    print('evaluating...')
    pose_results = {}
    depth_results = {}

    depth_groundtruth, pose_groundtruth = groundtruth
    depth_predictions, pose_predictions = predictions[0:2]

    num_test = len(depth_groundtruth)
    for i in trange(num_test):
        # match scales using median
        scalor = eval_utils.compute_scaling_factor(depth_groundtruth[i], depth_predictions[i])
        depth_predictions[i] = scalor * depth_predictions[i]

        depth_metrics = eval_utils.compute_depth_errors(depth_groundtruth[i], depth_predictions[i])
        pose_metrics = eval_utils.compute_pose_errors(pose_groundtruth[i], pose_predictions[i])

        if i == 0:
            for pkey in pose_metrics:
                pose_results[pkey] = []
            for dkey in depth_metrics:
                depth_results[dkey] = []

        for pkey in pose_metrics:
            pose_results[pkey].append(pose_metrics[pkey])

        for dkey in depth_metrics:
            depth_results[dkey].append(depth_metrics[dkey])

    ### aggregate metrics
    for pkey in pose_results:
        pose_results[pkey] = np.mean(pose_results[pkey])

    for dkey in depth_results:
        depth_results[dkey] = np.mean(depth_results[dkey])

    print(("{:>1}, " * len(depth_results)).format(*depth_results.keys()))
    print(("{:10.4f}, " * len(depth_results)).format(*depth_results.values()))

    print(("{:>16}, " * len(pose_results)).format(*pose_results.keys()))
    print(("{:16.4f}, " * len(pose_results)).format(*pose_results.values()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='cfgs/scannet.yaml', help='config file used to train the model')
    parser.add_argument('--model', default='models/scannet.ckpt', help='path to model checkpoint')
    parser.add_argument('--dataset_dir', default='/data/7scenes')
    parser.add_argument('--scene_idx', default=0)

    parser.add_argument('--mode', default='keyframe', help='config file used to train the model')
    parser.add_argument('--fcrn', action="store_true", help='use single image depth initializiation')
    parser.add_argument('--ignore_cache', action="store_true")
    # parser.add_argument('--start', type=float, default=0.0)
    # parser.add_argument('--input_campose', default=True, action='store_true')
    parser.add_argument('--output_dir', default='7scenes_output', type=str)
    args = parser.parse_args()

    groundtruth, predictions = make_predictions(args)
