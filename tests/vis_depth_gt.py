import matplotlib.pyplot as plt
import sys
import tensorflow as tf

# sys.path.insert(0, '.')
# sys.path.insert(0, './deepv2d')
from deepv2d.data_layer import DataLayer
from deepv2d.core import config
cfg = config.cfg_from_file('cfgs/kitti.yaml')
dl = DataLayer('/raid/linghao/project_data/deepv2d/kitti_train.tf_records',
               batch_size=3)
n = dl.next()
v = tf.Session().run(n[3])
plt.imshow(v[0, :, :, 0], 'jet')
plt.show()
print()
