import tensorflow as tf
from tensorflow.contrib import slim

# from deepv2d.modules.networks import hg

x = tf.placeholder(tf.float32, [1, 160, 160, 3])
x = slim.max_pool2d(x, [2, 2], padding='SAME')
print()
