import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

images = np.full((1, 24, 24, 24, 160), 255.0)
images = tf.constant(images)
with slim.arg_scope([slim.conv3d],
                    weights_regularizer=slim.l2_regularizer(0.00005),
                    normalizer_fn=None,
                    activation_fn=None,
                    reuse=False):
    net = slim.conv3d(images, 32, [3, 4, 5], padding='VALID', stride=1)
    print(net)
    print(tf.global_variables())
