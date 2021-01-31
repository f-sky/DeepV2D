import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
import matplotlib.pyplot as plt
import sys
import warnings

warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)
import tensorflow as tf
from tensorflow.contrib import slim

np.random.seed(0)
config = tf.ConfigProto(
    device_count={'GPU': 0}
)
weight = np.ones((3, 3))[:, :, None, None]

# weight = np.load('../MonoTrack/tmp/conv0w.npy').transpose(2, 3, 1, 0)
# bias = np.load('../MonoTrack/tmp/conv0b.npy')
with tf.Session(config=config) as sess:
    x = tf.constant(np.ones((1, 96, 96, 1)))
    x = tf.pad(x, tf.constant([[0, 0], [0, 1], [0, 1], [0, 0]]), constant_values=0)
    y = slim.conv2d(x, 1, [3, 3], stride=2, padding='VALID', activation_fn=None,
                    weights_initializer=tf.constant_initializer(weight),
                    biases_initializer=tf.constant_initializer(0), )
    sess.run(tf.global_variables_initializer())
    res = sess.run(y)
    print(res[0, :, :, 0].shape)
# todo: kernelsize=3,stride=2
# 5.21374078
# 手动pad0 3.0595844

# random weight
# 1->1
# SAME 13.21971972
# manual pad + valid 7.41130873

# all 1 weight
# SAME 25
# manual pad + valid 16
