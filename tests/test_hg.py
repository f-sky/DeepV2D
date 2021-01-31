import tensorflow as tf

from deepv2d.modules.networks import hg

x = tf.placeholder(tf.float32, [1, 48, 272, 32, 48])
x = hg.hourglass_3d(x, 4, 48)
print(tf.trainable_variables())
