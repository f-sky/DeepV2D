import sys
sys.path.append('deepv2d')

from deepv2d.data_stream.kitti import KittiRaw

ds = KittiRaw('/home/linghao/Datasets/kitti/kitti_raw', split='object')
ds.test_set_iterator(3)
