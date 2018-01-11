
from six.moves import urllib
import tensorflow as tf
import os
from tempfile import gettempdir
import zipfile
import collections
import numpy as np
import random
import math
from astropy.coordinates.builtin_frames.utils import norm

url = 'http://mattmahoney.net/dc/'

def maybe_download(filename, expected_bytes):
    """Download a file if not present, and make sure it's the right size."""
    local_filename = os.path.join(gettempdir(), filename)
    print(local_filename)
    if not os.path.exists(local_filename):
        local_filename, _ = urllib.request.urlretrieve(url + filename, local_filename)
        
    statinfo = os.stat(local_filename)
    if statinfo.st_size == expected_bytes:
        print("Found and verified ", filename)
    else:
        print(statinfo.st_size)
        raise Exception('Failed to verify ' + local_filename + 
                        '. Can you get to it with a browser?')
        
    return local_filename

filename = maybe_download("text8.zip", 31344016)

def read_data(filename):
    """Extract the first file enclosed in a zip file as a list of words."""
    with zipfile.ZipFile(filename) as f:
        for name in f.namelist():
            print(name)
         
        print(type(f.read(f.namelist()[0])))   
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    
    return data

vocabolary = read_data(filename)
print("Data size: ", len(vocabolary))

vocabolary_size = 50000


def build_dataset(words, n_words):
    """Process raw inputs into a dataset."""
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    
    data = list()
    unk_count = 0
    for word in words:
        index = dictionary.get(word,0)
        if index == 0:
            unk_count += 1
        data.append(index)
    
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary


data, count, dictionary, reverse_dictionary = build_dataset(vocabolary, vocabolary_size)

del vocabolary   # Hint to reduce memory
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])


data_index = 0

def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1
    buffer = collections.deque(maxlen=span)
    if data_index + span > len(data):
        data_index = 0
    buffer.extend(data[data_index:data_index + span])
    data_index += span
    for i in range(batch_size // num_skips):
        context_words = [w for w in range(span) if w != skip_window]
        words_to_use = random.sample(context_words, num_skips)
        for j, context_word in enumerate(words_to_use):
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j] = buffer[context_word]
        if data_index == len(data):
            buffer[:] = data[:span]
            data_index = span
        else:
            buffer.append(data[data_index])
            data_index += 1
        
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels

batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
    print(batch[i], reverse_dictionary[batch[i]],
          '->', labels[i,0], reverse_dictionary[labels[i,0]])
    

batch_size = 128
embedding_size = 128
skip_window = 1
num_skips = 2
num_sampled = 64

valid_size = 16
valid_window = 100
valid_examples = np.random.choice(valid_window, valid_size, replace=False)

graph = tf.Graph()

with graph.as_default():
    
    # Input data
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size,1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
    
    with tf.device('/cpu:0'):
        embeddings = tf.Variable(
            tf.random_uniform([vocabolary_size, embedding_size], -1.0, 1.0))
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)
        
        # Construct the variables for the NCE loss
        nce_weights = tf.Variable(
            tf.truncated_normal([vocabolary_size, embedding_size], stddev = 1.0 / math.sqrt(embedding_size)))
        
        nce_biases = tf.Variable(tf.zeros([vocabolary_size]))
        
    loss = tf.reduce_mean(
        tf.nn.nce_loss(weights=nce_weights,
                       biases = nce_biases,
                       labels=train_labels,
                       inputs=embed,
                       num_sampled=num_sampled,
                       num_classes=vocabolary_size))
    
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
    
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(
        normalized_embeddings, valid_dataset)
    similarity = tf.matmul(
        valid_embeddings, normalized_embeddings, transpose_b=True)
    
    init = tf.global_variables_initializer()
    
num_steps = 100001

        

    
