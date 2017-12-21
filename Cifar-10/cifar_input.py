
"""Routine for decoding the cifar-10 binary file format."""

import os
import tensorflow as tf

from six.moves import xrange

# Process images of this size.
IMAGE_SIZE = 24

NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN=50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL=10000

def read_cifar10(filename_queue):
    """Reads and parses examples from cifar10 data files.
    Args:
        filename_queue: A queue of strings with the filenames to read from.
    
    Returns:
        An object representing a single example, with the following fields:
            height: number of rows in the result
            width: number of columns in the result
            depth: number of color channels in the result
            key: a scalar string tensor describing the filename & record number for this example
            label: an int32 tensor with the label in the range 0~9
            uint8image: a [height, width, depth] uint8 tensor with the image data
    """
    
    class CIFAR10Record(object):
        pass
    
    result = CIFAR10Record()
    
    label_bytes = 1
    result.height = 32
    result.width = 32
    result.depth = 3
    image_bytes = result.height * result.width * result.depth
    record_bytes = label_bytes + image_bytes
    
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key, value = reader.read(filename_queue)
    print('result.key:', result.key)
    
    record_bytes = tf.decode_raw(value, tf.uint8)
    
    result.label = tf.cast(
        tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)
    
    depth_major = tf.reshape(
        tf.strided_slice(record_bytes, [label_bytes],
                         [label_bytes + image_bytes]),
        [result.depth, result.height, result.width])
    
    result.uint8image = tf.transpose(depth_major, [1,2,0])
    
    return result

def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
    """Construct a queued batch of images and labels.
    
    Args:
      image: 3-D tensor of type.float32
      label: 1-D tensor of type.int32
      min_queue_examples: int32, minimum number of samples to retain in the queue that
        provides of batches of examples
      batch_size: number of images per batch
      shuffle: boolean indicating whether to use a shuffling queue
      
    Returns:
      images: images, 4-D tensor of [batch_size, height, width, 3] size
      labels: labels, 1-D tensor of [batch_size] size
    """
    num_preprocess_threads = 16
    if shuffle:
        images, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        images, label_batch = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size)
        
    tf.summary.image('images', images)
    print('Number of images: ', images.shape[0])
    
    return images, tf.reshape(label_batch, [batch_size])
      

def distored_inputs(data_dir, batch_size):
    """Construct distorted input for cifar training using the Reader ops.
    
    Args:
      data_dir: Path to the cifar-10 data directory
      batch_size: number of images per batch
      
    Returns:
      images: Images, 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size
      labels: Labels, 1D tensor of [batch_size] size
    """
    filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
                 for i in xrange(1, 6)]
    
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)
        
    filename_queue = tf.train.string_input_producer(filenames)
    
    read_input = read_cifar10(filename_queue)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)
    
    height = IMAGE_SIZE
    width = IMAGE_SIZE
    
    distorted_image = tf.random_crop(reshaped_image, [height, width, 3])
    distorted_image = tf.image.random_flip_left_right(distorted_image)
    distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)
    distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)
    
    float_image = tf.image.per_image_standardization(distorted_image)
    
    float_image.set_shape([height, width,3])
    read_input.label.set_shape([1])
    
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * 
                             min_fraction_of_examples_in_queue)
    print('Filling queue with %d cifar images before starting to train. '
          'This will take a few minutes.' % min_queue_examples)
    
    return _generate_image_and_label_batch(float_image, read_input.label, min_queue_examples, batch_size, shuffle=True)

def inputs(eval_data, data_dir, batch_size):
    """Construct input for cifar evaluation using the reader ops.
    
    Args:
      eval_data: bool, indicating if one should use the train or eval dataset
      data_dir: path to the cifar-10 data directory
      batch_size: number of images per batch
      
    Returns:
      images: Images, 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE,3] size
      labels: Labels, 1D tensor of [batch_size] size
    """
    if not eval_data:
        filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
                     for i in xrange(1, 6)]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
    else:
        filenames = [os.path.join(data_dir, 'test_batch.bin')]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
        
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)
        
    filename_queue = tf.train.string_input_producer(filenames)
    
    read_input = read_cifar10(filename_queue)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)
    
    height = IMAGE_SIZE
    width = IMAGE_SIZE
    
    resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
                                                           height, width)
    
    float_image = tf.image.per_image_standardization(resized_image)
    
    float_image.set_shape([height, width,3])
    read_input.label.set_shape([1])
    
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_examples_per_epoch * 
                             min_fraction_of_examples_in_queue)
    
    return _generate_image_and_label_batch(float_image, read_input.label, min_queue_examples, batch_size, shuffle=False)
