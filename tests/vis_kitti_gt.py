import os.path as osp
import matplotlib.pyplot as plt
import pickle
import sys

sys.path.append('deepv2d')
from deepv2d.data_stream.kitti import KittiRaw

root = '/home/linghao/Datasets/kitti/kitti_raw'
ds = KittiRaw(root)

testfiles = open('data/kitti/test_files_eigen.txt').read().splitlines()
groundtruth = pickle.load(open('data/kitti/kitti_groundtruth.pickle', 'rb'))

drive = testfiles[0].split('/')[1].replace('_sync', '')
img = osp.join(root, testfiles[0])
velo_path = img.replace('image_02', 'velodyne_points').replace('.png', '.bin')
depth = ds.load_depth(velo_path, img, drive)

plt.imshow(groundtruth[0][1], 'jet')
plt.show()
print()
