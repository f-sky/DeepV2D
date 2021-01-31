import numpy as np
import os.path as osp
import os
import pickle

from dl_ext.vision_ext.datasets.kitti.load import load_calib, load_label_2, load_velodyne
from tqdm import tqdm, trange
from deepv2d.utils.vis3d import Vis3D
from PIL import Image


def main():
    KITTIROOT = '/home/linghao/Datasets/kitti'
    pred_dir = '/home/linghao/PycharmProjects/DeepV2D/kitti_object_depth'
    with Vis3D(('x', '-y', '-z'), 'dbg') as vis3d:
        for i in trange(7481):
            if i > 23: break
            vis3d.set_scene_id(i)
            lidar = np.fromfile(osp.join(pred_dir, '%06d.bin' % i)).reshape(-1, 4)[:, :3]
            calib = load_calib(KITTIROOT, 'training', i)
            lidar = calib.lidar_to_rect(lidar)
            vis3d.add_point_cloud(lidar)
            lidar = load_velodyne(KITTIROOT, 'training', i)[:, :3]
            lidar = calib.lidar_to_rect(lidar)
            vis3d.add_point_cloud(lidar, name='gt')
            lidar = np.fromfile('/raid/linghao/project_data/PointRCNN/data/triangulate_pts/%06d.bin' % i).reshape(-1, 4)[:, :3]
            lidar = calib.lidar_to_rect(lidar)
            vis3d.add_point_cloud(lidar, name='tri')

            labels = load_label_2(KITTIROOT, 'training', i)
            for label in labels:
                vis3d.add_label(label)


if __name__ == '__main__':
    main()
