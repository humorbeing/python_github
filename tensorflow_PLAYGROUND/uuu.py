import tensorflow_playground
# import torch
import tensorflow_playground
import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
#
with tf.variable_scope('learning_rate'):
    lr = tf.Variable(0.0, trainable=False)