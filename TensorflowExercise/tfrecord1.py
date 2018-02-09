

import tensorflow as tf
import os
import numpy as np
from tensorflow.examples.tutorials import mnist

save_dir = 'path/to/mnist'
data_splits = ['train', 'validation', 'test']

if __name__ == '__main__':
#     datasets = mnist.input_data.read_data_sets(save_dir,
#                                     dtype=tf.uint8,
#                                     reshape=False,
#                                     validation_size=1000)
#     print(datasets)
#     for d in range(len(data_splits)):
#         writer = tf.python_io.TFRecordWriter(os.path.join(save_dir, data_splits[d] + '.tfrecords'))
#         dataset = datasets[d]
#         print(dataset.images.shape)
#         for i in range(dataset.images.shape[0]):
#             image = dataset.images[i].tostring()
#             example = tf.train.Example(features=tf.train.Features(feature={
#                 'height':tf.train.Feature(int64_list=tf.train.Int64List(value=
#                                                                         [dataset.images.shape[1]])),
#                 'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[
#                     dataset.images.shape[2]])),
#                 'depth': tf.train.Feature(int64_list=tf.train.Int64List(value=[
#                     dataset.images.shape[3]])),
#                 'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[
#                     int(dataset.labels[i])])),
#                 'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[
#                     image]))}))
#             writer.write(example.SerializeToString())
#         writer.close()
        
    filename = os.path.join(save_dir, 'train.tfrecords')
    iterator = tf.python_io.tf_record_iterator(filename)
    serilized_example = next(iterator)
        
    example = tf.train.Example()
    example.ParseFromString(serilized_example)
    height = example.features.feature['height'].int64_list.value
    width = example.features.feature['width'].int64_list.value 
    label = example.features.feature['label'].int64_list.value 
    image_raw = example.features.feature['image_raw'].bytes_list.value 
    print('type of image_raw:', type(image_raw))
    print('height:', height)
    img_flat = np.fromstring(image_raw[0], np.uint8)
    image = np.reshape(img_flat, newshape=(height[0], width[0], -1))
    print('shape of image:', image.shape)
        
        
        
        

    
    
    
    