import os.path as osp
import numpy as np
import sys

import zarr
from dl_ext.vision_ext.datasets.kitti.load import load_calib
from tqdm import trange

sys.path.append('deepv2d')

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import cv2
import os
import time
import argparse

# from deepv2d import eval_utils
from deepv2d.core import config
from deepv2d.deepv2d import DeepV2D
from deepv2d.data_stream.kitti import KittiRaw


# def process_for_evaluation(depth, scale, crop):
#     """ During training ground truth depths are scaled and cropped, we need to
#         undo this for evaluation """
#     depth = (1.0 / scale) * np.pad(depth, [[crop, 0], [0, 0]], 'mean')
#     return depth

def img_to_rect(u, v, depth_rect, K):
    fu, fv, cu, cv = K
    x = ((u - cu) * depth_rect) / fu
    y = ((v - cv) * depth_rect) / fv
    pts_rect = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1), depth_rect.reshape(-1, 1)), axis=1)
    return pts_rect


def depth_to_rect(depth_map, intrinsic):
    x_range = np.arange(0, depth_map.shape[1])
    y_range = np.arange(0, depth_map.shape[0])
    x_idxs, y_idxs = np.meshgrid(x_range, y_range)
    x_idxs, y_idxs = x_idxs.reshape(-1), y_idxs.reshape(-1)
    depth = depth_map[y_idxs, x_idxs]
    pts_rect = img_to_rect(x_idxs, y_idxs, depth, intrinsic)
    return pts_rect


def make_predictions(args):
    """ Run inference over the test images """

    np.random.seed(1234)
    cfg = config.cfg_from_file(args.cfg)

    db = KittiRaw(args.dataset_dir, split='object')
    scale = db.args['scale']
    crop = db.args['crop']

    os.makedirs(args.output_dir, exist_ok=True)

    deepv2d = DeepV2D(cfg, args.model, use_fcrn=False, mode='keyframe')
    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.allow_growth = True

    with tf.Session(config=tfconfig) as sess:
        deepv2d.set_session(sess)

        predictions = []
        # it = db.test_set_iterator()
        for i in trange(7481):
            # for i, data_blob in enumerate():
            try:
                data_blob = db.test_set_iterator(i)
            except FileNotFoundError:
                # np.empty((0, 4)).tofile(osp.join(args.output_dir, '%06d.bin' % i))
                zarr.save(osp.join(args.output_dir, '%06d.zarr' % i), np.zeros(()))
                continue
            images, intrinsics, poses = data_blob['images'], data_blob['intrinsics'], data_blob['poses']
            if not args.input_pose:
                poses = None
                iters = args.n_iters
            else:
                iters = 1
            depth_predictions, _ = deepv2d(images, intrinsics,
                                           iters=iters, camposes=poses)
            keyframe_depth = depth_predictions[0]
            keyframe_depth = keyframe_depth / scale

            zarr.save(osp.join(args.output_dir, '%06d.zarr' % i), keyframe_depth)

            # keyframe_image = images[0]
            #
            # pts = depth_to_rect(keyframe_depth, intrinsics)
            # calib = load_calib('/home/linghao/Datasets/kitti', 'training', i)
            # lidar = calib.rect_to_lidar(pts)
            # lidar = np.concatenate((lidar, np.ones((lidar.shape[0], 1))), axis=1)
            # lidar.tofile(osp.join(args.output_dir, '%06d.bin' % i))

        return predictions


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='cfgs/kitti.yaml', help='config file used to train the model')
    parser.add_argument('--model', default='models/kitti.ckpt', help='path to model checkpoint')
    parser.add_argument('--dataset_dir', default='/home/linghao/Datasets/kitti/kitti_raw',
                        help='config file used to train the model')
    parser.add_argument('--viz', action="store_true", help='display depth maps during inference')
    parser.add_argument('--n_iters', type=int, default=5, help='number of video frames to use for reconstruction')
    parser.add_argument('--output_dir', type=str, default='kitti_object_depth')
    parser.add_argument('--input_pose', default=True)
    args = parser.parse_args()

    # run inference on the test images
    predictions = make_predictions(args)
