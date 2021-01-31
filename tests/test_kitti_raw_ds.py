import sys

sys.path.append('deepv2d')
from deepv2d.data_stream.kitti import KittiRaw

ds = KittiRaw('/home/linghao/Datasets/kitti/kitti_raw')

d=ds[0]
print()
