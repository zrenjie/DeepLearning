"""Builds the cifar-10 network
Summary of available functions:

  #Compute input images and labels for training. 
  inputs, labels = distorted_inputs()
  
  #Compute inference on the model inputs to make a prediction
  predictions = inference(inputs)
  
  #Compute the total loss of the prediction with respect to the labels
  loss = loss(predictions, labels)
  
  #Create a graph to run one step of training with respect to the loss
  train_op = train(loss, global_step)
"""
import argparse
import os
import re 
import sys
import tarfile 
 
from six.moves import urllib
import tensorflow as tf
from source import cifar_input
from source.cifar_input import NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN,\
    NUM_EXAMPLES_PER_EPOCH_FOR_EVAL


parser = argparse.ArgumentParser()

#Basic model parameters
parser.add_argument('--batch_size', type=int, default=128,
                    help='Number of images to process in a batch.')

parser.add_argument('--data_dir', type=str, default='/tmp/cifar10_data',
                    help='Path to the cifar-10 data directory.')

parser.add_argument('--use_fp16', type=bool, default=False,
                    help='Train the model using fp16.')

FLAGS = parser.parse_args()

#Global constants describing the cifar-10 data set
IMAGE_SIZE = cifar_input.IMAGE_SIZE
NUM_CLASSES = cifar_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL


#Constants describing the training process
MOVING_AVERAGE_DECAY = 0.9999
NUM_EPOCHS_PER_DECAY = 350.0
LEARNING_RATE_DECAY_FACTOR = 0.1
INITIAL_LEARNING_RATE = 0.1


TOWER_NAME = 'tower'

DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'


def _activation_summary(x):
    """Helper to create summaries for activations.
    
    Create a summary that provides a histogram of activations
    Create a summary that measures the sparsity of activations.
    
    Args:
        x: Tensor
    Returns:
        nothing
    """
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity',
                      tf.nn.zero_fraction(x))
    

def _variable_on_cpu(name,shape, initializer):
    """Helper to create a variable stored on cpu memory
    
    Args:
        name: name of variable
        shape: list of ints
        initializer: initializer for variable
    
    Returns:
        Variable tensor
    """
    with tf.device('/cpu:0'):
        dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var


def _variable_with_weight_decay(name, shape, stddev, wd):
    """Helper to create an initialized Variable with weight decay.
    
    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.
    
    Args:
        name: name of the variable
        shape: list of ints
        stddev: standard deviation of a truncated Gaussion
        wd: add L2 loss weight decay multiplied by this float. If None, weight decay is not added for
            this Variable.
    
    Returns:
        Variable Tensor
    """
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = _variable_on_cpu(
        name,
        shape,
        tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
    
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    
    return var


def distorted_inputs():
    """Construct distorted input for cifar training using the reader ops.
    
    Returns:
        images:  Images, 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size
        labels: Labels, 1D tensor of [batch_size] size
        
    Raiese:
        ValueError: if no data_dir
    """
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')
    data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
    images, labels = cifar_input.distored_inputs(data_dir=data_dir, batch_size=FLAGS.batch_size)
    
    if FLAGS.use_fp16:
        images = tf.cast(images, tf.float16)
        labels = tf.cast(labels, tf.float16)
        
    return images, labels


def inputs(eval_data):
    """Construct input for cifar evaluation using the reader ops.
    
    Args:
        eval_data: bool, indicating if one should use the train or eval data set.
        
    Returns:
        images: Images. 4D tensor
        labels: Labels. 1D tensor
        
    Raises:
        ValueError: if no data_dir
    """
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')
    
    data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
    images, labels = cifar_input.inputs(eval_data, data_dir, FLAGS.batch_size)
    
    if FLAGS.use_fp16:
        images = tf.cast(images, tf.float16)
        labels = tf.cast(labels, tf.float16)
        
    return images, labels


def inference(images):
    """Build the cifar-10 model.
    
    Args:
        images: Images returned from distorted_inputs() or inputs()
        
    Returns:
        Logits.
    """
    with tf.variable_scope('conv1') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[5,5,3,64], stddev=5e-2, wd=0.0)
        conv = tf.nn.conv2d(images, kernel, [1,1,1,1], padding='SAME')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv1)
        
    pool1 = tf.nn.max_pool(conv1, ksize=[1,3,3,1], strides=[1,2,2,1],
                           padding='SAME', name='pool1')
    
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001/9.0, beta=0.75, name='norm1')
    
    with tf.variable_scope('conv2') as scope:
        kernel = _variable_with_weight_decay('weights', [5,5,64,64], 5e-2, 0.0)
        conv = tf.nn.conv2d(norm1, kernel, [1,1,1,1], padding='SAME')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv,biases)
        conv2 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv2)
        
    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001/9.0, beta=0.75, name='norm2')
    
    pool2 = tf.nn.max_pool(norm2, ksize=[1,3,3,1],
                           strides=[1,2,2,1], padding='SAME', name='pool2')
    
    with tf.variable_scope('local3') as scope:
        reshape = tf.reshape(pool2, [FLAGS.batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = _variable_with_weight_decay('weights', [dim, 384], 0.04, 0.004)
        biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name='scope.name')
        _activation_summary(local3)
        
    with tf.variable_scope('local4') as scope:
        weights = _variable_with_weight_decay('weights', [384,192], 0.04, 0.004)
        biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
        _activation_summary(local4)
        
    with tf.variable_scope('softmax_linear') as scope:
        weights = _variable_with_weight_decay('weights', [192, NUM_CLASSES], 1/192.0, 0.0)
        biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
        _activation_summary(softmax_linear)
        
    return softmax_linear


def loss(logits, labels):
    """Add L2Loss to all the trainable variables
    
    Add Summary for 'loss' and 'loss/avg'
    Args:
        logits: logits from inference()
        labels: Labels from distorted_inputs()
        
    Returns:
        Loss tensor of type float
    """
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
    
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
    """Add summaries for losses in cifar-10 model
    
    Generates moving average for all losses and associated summaries for visualizing 
    the performance of the network.
    
    Args:
        total_loss: Total loss from loss()
    Returns:
        loss_averages_op: op for generating moving averages of losses
    """
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='abg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])
    
    for l in losses + [total_loss]:
        tf.summary.scalar(l.op.name + ' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))
        
    return loss_averages_op


def train(total_loss, global_step):
    """Train cifar-10 model.
    
    Create an optimizer and apply to all trainable variables. Add moving
    average for all trainable variables.
    
    Args:
        total_loss: Total loss from loss()
        global_step: Integer Variable counting the number of training steps processed
    Returns:
        train_op: op for training
    """
    print('global_step:', global_step)
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
    
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE, global_step, decay_steps, 
                                    LEARNING_RATE_DECAY_FACTOR, staircase=True)
    tf.summary.scalar('learning_rate', lr)
    
    loss_averages_op = _add_loss_summaries(total_loss)
    
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(total_loss)
        
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)
        
    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)
            
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    
    with tf.control_dependencies([apply_gradient_op, variable_averages_op]):
        train_op = tf.no_op(name='train')
        
    return train_op


def maybe_download_and_extract():
    """Download and extract the tarball from Alex's website."""
    dest_directory = FLAGS.data_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
        
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>>Downloading %s %.1f%%' % (filename,
                                                           float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes')
    
    extracted_dir_path = os.path.join(dest_directory, 'cifar-10-batches-bin')
    if not os.path.exists(extracted_dir_path):
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)




