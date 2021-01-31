import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import matplotlib.pyplot as plt
import sys

import tensorflow as tf

from deepv2d.geometry.transformation import VideoSE3Transformation

sys.path.append('deepv2d')
from deepv2d.data_stream.kitti import KittiRaw
from deepv2d.modules.depth import DepthNetwork
from deepv2d.core import config

tfconfig = tf.ConfigProto()
tfconfig.gpu_options.allow_growth = True
# tfconfig = tf.ConfigProto(
#     device_count={'GPU': 0}
# )
with tf.Session(config=tfconfig) as sess:
    cfg = config.cfg_from_file('cfgs/kitti.yaml')
    cfg.STRUCTURE.COST_VOLUME_DEPTH = 64
    db = KittiRaw('/home/linghao/Datasets/kitti/kitti_raw', split='object')
    data_blob = db.test_set_iterator(3)
    images, intrinsics, poses = data_blob['images'], data_blob['intrinsics'], data_blob['poses']
    # images.fill(255.0)
    images = tf.constant(images[None])
    intrinsics = tf.constant(intrinsics[None])
    poses = tf.constant(poses[None])
    Ts = VideoSE3Transformation(matrix=poses)

    depth_net = DepthNetwork(cfg.STRUCTURE, is_training=False)
    depth_predictions = depth_net.forward(Ts, images, intrinsics)

    saver = tf.train.Saver(tf.model_variables())
    sess.run(tf.global_variables_initializer())
    # saver.restore(sess, 'models/kitti.ckpt')
    # np.save('tmp/coords.npy', sess.run(depth_net.coords))
    # np.save('tmp/w00.npy', sess.run(depth_net.w00))
    # np.save('tmp/w01.npy', sess.run(depth_net.w01))
    # np.save('tmp/w10.npy', sess.run(depth_net.w10))
    # np.save('tmp/w11.npy', sess.run(depth_net.w11))
    # np.save('tmp/coords00.npy', sess.run(depth_net.coords00))
    # np.save('tmp/coords01.npy', sess.run(depth_net.coords01))
    # np.save('tmp/coords10.npy', sess.run(depth_net.coords10))
    # np.save('tmp/coords11.npy', sess.run(depth_net.coords11))
    # np.save('tmp/img00.npy', sess.run(depth_net.img00))
    # np.save('tmp/img01.npy', sess.run(depth_net.img01))
    # np.save('tmp/img10.npy', sess.run(depth_net.img10))
    # np.save('tmp/img11.npy', sess.run(depth_net.img11))
    #
    # np.save('tmp/indices.npy', sess.run(depth_net.indices))
    # np.save('tmp/out.npy', sess.run(depth_net.out))
    # np.save('tmp/valid.npy', sess.run(depth_net.valid))
    # np.save('tmp/o.npy', sess.run(depth_net.o))

    # np.save('tmp/vv.npy', sess.run(depth_net.vv))
    # np.save('tmp/sx0.npy', sess.run(depth_net.sx0))
    # np.save('tmp/sx1.npy', sess.run(depth_net.sx1))
    # np.save('tmp/hg1x.npy', sess.run(depth_net.hg1x))
    # np.save('tmp/hg1sx.npy', sess.run(depth_net.hg1sx))
    # np.save('tmp/logits.npy', sess.run(depth_net.logits))

    dpv = sess.run(depth_predictions)
    plt.imshow(dpv[0], 'jet')
    plt.show()

# np.save('tmp/sbn.npy', sess.run(depth_net.sbn))
# # np.save('tmp/ys21.npy', sess.run(depth_net.ys21))
# np.save('tmp/net2.npy', sess.run(depth_net.net2))
# np.save('tmp/net3.npy', sess.run(depth_net.net3))
# np.save('tmp/net4.npy', sess.run(depth_net.net4))
# np.save('tmp/net5.npy', sess.run(depth_net.net5))
# np.save('tmp/net6.npy', sess.run(depth_net.net6))
# np.save('tmp/net7.npy', sess.run(depth_net.net7))
# np.save('tmp/net9.npy', sess.run(depth_net.net9))
# np.save('tmp/net10.npy', sess.run(depth_net.net10))
# np.save('tmp/embd.npy', sess.run(depth_net.embd))
# # np.save('tmp/x1.npy', sess.run(depth_net.x1))
# np.save('tmp/pool1.npy', sess.run(depth_net.pool1))
# np.save('tmp/low1.npy', sess.run(depth_net.low1))
# np.save('tmp/low3.npy', sess.run(depth_net.low3))
# np.save('tmp/up2.npy', sess.run(depth_net.up2))
# np.save('tmp/inputs.npy', sess.run(depth_net.inputsv))
