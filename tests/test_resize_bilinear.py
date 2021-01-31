import numpy as np
import tensorflow as tf
import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

x = np.load('../DeepV2D/tmp/logits.npy')
x = tf.constant(x)
y = tf.image.resize_bilinear(x, (192, 1088))
res = tf.Session().run(y)
print()

