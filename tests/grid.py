import numpy as np
import tensorflow as tf
import sys

from utils.bilinear_sampler import bilinear_sampler

sys.path.insert(0, '.')
from deepv2d.geometry.transformation import VideoSE3Transformation, intrinsics_vec_to_matrix

np.random.seed(0)

poses = np.load('../MonoTrack/tmp/poses.npy').astype(np.float32)[None]
intrinsics = np.load('../MonoTrack/tmp/intrinsics.npy').astype(np.float32)[None]
fmaps = np.load('../MonoTrack/tmp/fmaps.npy').astype(np.float32)
intrinsics = tf.constant(intrinsics)
intrinsics = intrinsics_vec_to_matrix(intrinsics / 4.0)
depths = tf.lin_space(0.1, 8.0, 36)
dd = 36
batch = 1
num = 5
ht = 192 // 4
wd = 896 // 4

Ts = VideoSE3Transformation(matrix=poses)
ii, jj = tf.meshgrid(tf.range(1), tf.range(0, 5))
ii = tf.reshape(ii, [-1])
jj = tf.reshape(jj, [-1])
Tij = Ts.gather(jj) * Ts.gather(ii).inv()

depths = tf.reshape(depths, [1, 1, dd, 1, 1])
depths = tf.tile(depths, [batch, num, 1, ht, wd])
coords = Tij.transform(depths, intrinsics)

coords = tf.transpose(coords, [0, 1, 3, 4, 2, 5])
volume = bilinear_sampler(fmaps, coords, batch_dims=2)
volume = tf.transpose(volume, [0, 2, 3, 4, 1, 5])
with tf.Session() as sess:
    res = sess.run(depths)
    np.save('tmp/depths.npy', sess.run(depths))
    np.save('tmp/intrinsics.npy', sess.run(intrinsics))
    np.save('tmp/Tij.npy', sess.run(Tij.G))
    # res = sess.run(coords)
    print()
