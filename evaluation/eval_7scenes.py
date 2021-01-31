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


def make_predictions(args):
    cfg = config.cfg_from_file(args.cfg)
    deepv2d = DeepV2D(cfg, args.model, use_fcrn=True, mode=args.mode)
    print('making predictions...')
    os.makedirs(args.output_dir, exist_ok=True)
    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.allow_growth = True
    with tf.Session(config=tfconfig) as sess:
        deepv2d.set_session(sess)

        for scene_idx in range(18):
            db = SevenScenes(args.dataset_dir, scene_idx, args.inv_pose)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='cfgs/scannet.yaml', help='config file used to train the model')
    parser.add_argument('--model', default='models/scannet.ckpt', help='path to model checkpoint')
    parser.add_argument('--dataset_dir', default='/data/7scenes')
    parser.add_argument('--scene_idx', default=0)

    parser.add_argument('--mode', default='keyframe', help='config file used to train the model')
    parser.add_argument('--fcrn', action="store_true", help='use single image depth initializiation')
    parser.add_argument('--ignore_cache', action="store_true")
    parser.add_argument('--output_dir', default='7scenes_output', type=str)
    parser.add_argument('--inv_pose', default=False, action='store_true')
    args = parser.parse_args()

    make_predictions(args)
