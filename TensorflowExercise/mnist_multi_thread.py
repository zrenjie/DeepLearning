import tensorflow as tf
import numpy as np
import os
from tensorflow.examples.tutorials.mnist import input_data

save_dir = 'path/to/mnist'
filename = os.path.join(save_dir, 'train.tfrecords')
# datasets = input_data.read_data_sets(save_dir, reshape=False, dtype=tf.uint8)
# X_train = datasets.train.images
# Y_train = datasets.train.labels
# 
# print(X_train.shape)
# print(Y_train.shape)
# 
# writer = tf.python_io.TFRecordWriter(os.path.join(save_dir, 'train.tfrecords'))
# for i in range(X_train.shape[0]):
#     example = tf.train.Example(features=tf.train.Features(feature={
#         'height':tf.train.Feature(int64_list=tf.train.Int64List(value=[X_train.shape[1]])),
#         'width':tf.train.Feature(int64_list=tf.train.Int64List(value=[X_train.shape[2]])),
#         'depth': tf.train.Feature(int64_list=tf.train.Int64List(value=[X_train.shape[3]])),
#         'label':tf.train.Feature(int64_list=tf.train.Int64List(value=[Y_train[i]])),
#         'image_raw':tf.train.Feature(bytes_list=tf.train.BytesList(value=[X_train[i].tostring()]))}))
#     
#     writer.write(example.SerializeToString())
# writer.close()

# string_input_producer to queue filenames
filename_queue = tf.train.string_input_producer([filename], num_epochs=10)

# tfrecordrecorder to read from tfrecords
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(serialized_example, features={
    'image_raw':tf.FixedLenFeature([], tf.string),
    'label':tf.FixedLenFeature([], tf.int64)})

image = tf.decode_raw(features['image_raw'], tf.uint8)
image.set_shape([784])
image = tf.cast(image, tf.float32) * (1. / 255) - 0.5

label = tf.cast(features['label'], tf.int32)
images_batch, labels_batch = tf.train.shuffle_batch(
    [image, label], batch_size=128,
    capacity=2000,
    min_after_dequeue=1000)

print(image)

sess = tf.Session()

# batch_size to get minibatch samples


if __name__ == '__main__':
    pass
