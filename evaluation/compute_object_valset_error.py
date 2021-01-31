from collections import Counter
from multiprocessing import Pool
from warnings import warn

import numpy as np
import os.path as osp

import zarr
from dl_ext.vision_ext.datasets.kitti.load import load_calib, load_velodyne, load_image_info, load_label_2
from tqdm import tqdm

from deepv2d.data_stream.kitti_utils import sub2ind
from evaluation.eval_kitti import evaluate, process_for_evaluation

kr = '/home/linghao/Datasets/kitti/'
split = 'training'
FG_ONLY = True


def load_kins_mask_2(kittiroot, split, imgid):
    assert split == 'training'
    path = osp.join(kittiroot, 'object', split, 'kins_mask_2/%06d.zarr' % imgid)
    if not osp.exists(path):
        warn(path + ' not exists. return zeros')
        labels = load_label_2(kittiroot, split, imgid)
        H, W, _ = load_image_info(kittiroot, split, imgid)
        mask = np.zeros((len(labels), H, W)).astype(np.uint8)
    else:
        mask = zarr.load(path)
    return mask


def f(imgid):
    calib = load_calib(kr, split, imgid)
    h, w, _ = load_image_info(kr, split, imgid)
    depth_pred = zarr.load('kitti_object_depth/%06d.zarr' % imgid)

    velodyne = load_velodyne(kr, split, imgid)[:, :3]
    velo_pts, depthimg = calib.lidar_to_img(velodyne)
    velo_pts_im = np.concatenate((velo_pts, depthimg.reshape((-1, 1))), axis=1)
    velo_pts_im[:, 0] = np.round(velo_pts_im[:, 0]) - 1
    velo_pts_im[:, 1] = np.round(velo_pts_im[:, 1]) - 1
    val_inds = (velo_pts_im[:, 0] >= 0) & (velo_pts_im[:, 1] >= 0)
    val_inds = val_inds & (velo_pts_im[:, 0] < w) & (velo_pts_im[:, 1] < h)
    velo_pts_im = velo_pts_im[val_inds, :]
    # depthimg = depthimg[val_inds]

    depth = np.zeros((h, w))
    depth[velo_pts_im[:, 1].astype(np.int), velo_pts_im[:, 0].astype(np.int)] = velo_pts_im[:, 2]

    # find the duplicate points and choose the closest depth
    inds = sub2ind(depth.shape, velo_pts_im[:, 1], velo_pts_im[:, 0])
    dupe_inds = [item for item, count in Counter(inds).items() if count > 1]
    for dd in dupe_inds:
        pts = np.where(inds == dd)[0]
        x_loc = int(velo_pts_im[pts[0], 0])
        y_loc = int(velo_pts_im[pts[0], 1])
        depth[y_loc, x_loc] = velo_pts_im[pts, 2].min()

    depth[depth < 0] = 0
    if FG_ONLY:
        mask_2 = load_kins_mask_2(kr, split, imgid)
        labels = load_label_2(kr, split, imgid)
        assert len(mask_2) == len(labels)
        for i, label in enumerate(labels):
            if label.cls.name != 'Car':
                mask_2[i].fill(0)
        mask = np.clip(mask_2.sum(0), a_min=0, a_max=1)
        depth = depth * mask
    return depth_pred, depth


def main():
    valset = list(map(int, open('/home/linghao/Datasets/kitti/object/split_set/val_set.txt').read().splitlines()))
    all_preds, all_gts = [], []
    with Pool(8) as p:
        results = list(tqdm(p.imap(f, valset), total=len(valset)))
    for r in tqdm(results):
        all_preds.append(r[0])
        all_gts.append(r[1])
    scale = 1
    crop = 108
    predictions = []
    for keyframe_depth in all_preds:
        pred = process_for_evaluation(keyframe_depth, scale, crop)
        predictions.append(pred.astype(np.float32))
    evaluate(list(zip(range(len(all_gts)), all_gts)), predictions)


if __name__ == '__main__':
    main()
