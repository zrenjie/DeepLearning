import tensorflow as tf
import numpy as np
import os
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib import keras

save_dir = 'path/to/mnist'
datasets = input_data.read_data_sets(save_dir, one_hot=True)

X_train = tf.placeholder(tf.float32, shape=[None, 28*28])
y_train = tf.placeholder(tf.float32, shape=[None, 10])

W = tf.Variable(tf.truncated_normal([784,10], mean=0.))
bias = tf.Variable(tf.zeros([10], dtype=tf.float32))

pred = tf.matmul(X_train, W) + bias

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y_train))
train_op = tf.train.AdagradOptimizer(0.5).minimize(loss)

saver = tf.train.Saver(keep_checkpoint_every_n_hours=0.2)
with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     for i in range(10000):
#         x, y = datasets.train.next_batch(64)
#         sess.run([train_op], feed_dict={X_train:x, y_train:y})
#         
#         if i %100 == 0:
#             saver.save(sess, save_dir, global_step=i)
#             print('Current step: ', i)
    
    saver.restore(sess, os.path.join('path/to', 'mnist-9900'))
    x_test, y_test = datasets.test.next_batch(120)
    print(sess.run(pred, feed_dict={X_train:x_test, y_train:y_test}))
        
