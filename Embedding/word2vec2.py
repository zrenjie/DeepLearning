""""Multi_thread word2vec mini-batched skip-gram model.
The key ops used are:
 * placeholder for feeding in tensors for each example.
 * embedding_lookup for fetching rows from the embedding matrix.
 * sigmoid_cross_entropy_with_logits to calculate the loss.
 * GradientDescentOptimizer for optimizing the loss.
 * skipgram custom op that does input processing.
"""
import os
import sys
import threading 
import time

from six.moves import xrange 
import numpy as np
import tensorflow as tf
from astropy.convolution.tests.test_convolve_fft import options
from word2vec import labels
from tensorflow.python.ops.nn_impl import _compute_sampled_logits
from builtins import int
from pandas.util.doctools import idx


word2vec = tf.load_op_library(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'word2vec_ops.so'))

flags = tf.app.flags 

flags.DEFINE_string('save_path', None, 'Directory to write the model and '
                    'training summaries')
flags.DEFINE_string('train_data', None, 'Training text file. '
                    'E.g., unzipped file http://mattmahoney.net/dc/text8.zip.')
flags.DEFINE_string(
    'eval_data', None, 'File consisting of analogies of four tokens.'
    'embedding 2 - embedding 1 + embedding 3 should be close to embedding 4.')
flags.DEFINE_integer('embedding_size', 200, 'The embedding dimension size')
flags.DEFINE_integer(
    'epochs_to_train', 15,
    'Number of epochs to train. Each epoch processes the training data once completely.')
flags.DEFINE_float('learning_rate', 0.2, 'Initial learning rate.')
flags.DEFINE_integer('num_neg_samples', 100,
                     'Negative samples per training example.')
flags.DEFINE_integer('batch_size', 16,
                     'Number of training examples processed per step '
                     '(size of a minibatch).')
flags.DEFINE_integer('concurrent_steps', 12,
                     'The number of concurrent training steps.')
flags.DEFINE_integer('window_size', 5, 
                     'The number of words to predict to the left and right'
                     'of the target word.')
flags.DEFINE_integer('min_count', 5,
                     'The minimum number of word occurrences for it to be '
                     'included in the vocabulary.')
flags.DEFINE_float('subsample', 1e-3,
                   'Subsample threshold for word occurence. Words that appear '
                   'with higher frequency will be randomly down-sampled. Set '
                   'to 0 to disable.')
flags.DEFINE_boolean(
    'interactive', False,
    'If true, enters a IPython interactive session to play with the trained '
    'model. E.g., try model.analogy blabla')
flags.DEFINE_integer('Statistics_interval', 5,
                     'Print statistics every n seconds.')
flags.DEFINE_integer('summary_interval', 5,
                     'Save training summary to file every n seconds.')
flags.DEFINE_integer('checkpoint_interval', 600,
                     'Checkpoint the model (i.e. save the parameters) every n '
                     'seconds ' )

FLAGS = flags.FLAGS


class Options(object):
    """Options used by our word2vec model."""
    
    def __init__(self):
        self.emb_dim = FLAGS.embedding_size
        self.train_data = FLAGS.train_data
        self.num_samples = FLAGS.num_neg_samples
        self.learning_rate = FLAGS.learning_rate
        self.epochs_to_train = FLAGS.epochs_to_train
        self.concurrent_steps = FLAGS.concurrent_steps
        self.batch_size = FLAGS.batch_size
        self.window_size = FLAGS.window_size
        self.min_count = FLAGS.min_count
        self.subsample = FLAGS.subsample
        self.statistics_interval = FLAGS.statistics_interval
        self.summary_interval = FLAGS.summary_interval
        self.checkpoint_interval = FLAGS.checkpoint_interval
        self.save_path = FLAGS.save_path
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
            
        self.eval_data = FLAGS.eval_data


class Word2Vec(object):
    """Word2Vec model (Skipgram)."""
    
    def __init__(self, Options, session):
        self._options = options
        self._session = session
        self._word2id = {}
        self._id2word = {}
        self.build_graph()
        self.build_eval_graph()
        self.save_vocab()
        
    
    def read_analogies(self):
        """Reads through the analogy question file.
        
        Returns:
            questions: a [n, 4] numpy array containing the analogy question's word ids.
            question_skipped: questions skipped due to unknown words.
        """
        questions = []
        question_skipped = 0
        with open(self._options.eval_data, 'rb') as analogy_f:
            for line in analogy_f:
                if line.startswith(b":"):
                    continue
                words = line.stripe().lower().split(b" ")
                ids = [self._word2id.get(w.strip()) for w in words]
                if None in ids or len(ids) != 4:
                    question_skipped += 1
                else:
                    questions.append(np.array(ids))
                    
        print('Eval analogy file: ', self._options.eval_data )
        print('Questions: ', len(questions))
        print('Skipped: ', question_skipped)
        self._analogy_questions = np.array(questions, dtype=np.int32)
        
        
    def forwar (self, examples, labels):
        """Build the graph for the forward pass."""
        opts = self._options
        
        init_width = 0.5 / opts.emb_dim
        emb = tf.Variable(
            tf.random_uniform([opts.vocal_size, opts.emb_dim], -init_width, init_width),
            name='emb')
        self._emb = emb
        
        sm_w_t = tf.Variable(
            tf.zeros([opts.vocal_size, opts.emb_dim]),
            name = 'sm_w_t')
        
        sm_b = tf.Variable(
            tf.zeros([opts.vocal_size]), name='sm_b')
        
        self.global_step = tf.Variable(0, name='global_step')
        
        labels_matrix = tf.reshape(tf.cast(labels, dtype=tf.int64), [opts.batch_size, 1])
        
        # Negative sampling
        sampled_ids, _, _ = (tf.nn.fixed_unigram_candidate_sampler(
            true_classes=labels_matrix,
            num_true=1, 
            num_sampled=opts.num_samples, 
            unique=True, 
            range_max=opts.vocal_size, 
            distortion=0.75, 
            unigrams=opts.vocab_counts.tolist()))
        
        # Embeddings for examples: [batch_size, emb_dim]
        example_emb = tf.nn.embedding_lookup(emb, examples)
        
        true_w = tf.nn.embedding_lookup(sm_w_t, labels)
        true_b = tf.nn.embedding_lookup(sm_b, labels)
        
        
        sampled_w = tf.nn.embedding_lookup(sm_w_t, sampled_ids)
        sampled_b = tf.nn.embedding_lookup(sm_b, sampled_ids)
        
        true_logits = tf.reduce_sum(tf.multiply(example_emb, true_w), 1) + true_b
        
        sampled_b_vec = tf.reshape(sampled_b, [opts.num_samples])
        sampled_logits = tf.matmul(example_emb,
                                   sampled_w,
                                   transpose_b=True) + sampled_b_vec
        return true_logits, sampled_logits
    
    def nce_loss(self, true_logits, sampled_logits):
        """Build the graph for the NCE loss."""
        
        # cross-entropy(logits, labels)
        opts = self._options
        true_xent = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(true_logits), logits=true_logits)
        sampled_xent = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.zeros_like(sampled_logits), logits = sampled_logits)
        
        nce_loss_tensor = (tf.reduce_sum(true_xent) + 
                           tf.reduce_sum(sampled_xent)) / opts.batch_size
        return nce_loss_tensor
    
    def optimize(self, loss):
        """Build the graph to optimize the loss function. """
        # Optimizer nodes.
        # Linear learning rate decay.
        opts = self._options
        words_to_train = float(opts.words_per_epoch * opts.epochs_to_train)
        lr = opts.learning_rate * tf.maximum(
            0.0001, 1.0 - tf.cast(self._words, tf.float32) / words_to_train)
        self._lr = lr
        optimizer = tf.train.GradientDescentOptimizer(lr)
        train = optimizer.minimize(loss,
                                   global_step=self.global_step,
                                   gate_gradients=optimizer.GATE_NONE)
        self._train = train
        
    
    def build_eval_graph(self):
        """Build the eval graph."""
        # Eval graph
        
        # Each analogy task is to predict the 4th word (d) given three
        # words:a, b, c. E.g., a=italy, b=rome, c=france, we should predict d= pairs.
        
        # The eval feeds three vectors of word ids for a, b, c, each of
        # which is of size N, where N is the number of analogies we want to 
        # evaluate in one batch
        analogy_a = tf.placeholder(dtype=tf.int32)
        analogy_b = tf.placeholder(dtype=tf.int32)
        analogy_c = tf.placeholder(dtype=tf.int32)
        
        # Normalized word embeddings of shape [vocab_size, emb_dim]
        nemb = tf.nn.l2_normalize(self._emb, 1)
        
        #Each row of a_emb, b_emb, c_emb is a word's embedding vector.
        # They all have the shape [N, emb_dim]
        a_emb = tf.gather(nemb, analogy_a)
        b_emb = tf.gather(nemb, analogy_b)
        c_emb = tf.gather(nemb, analogy_c)
        
        # We expect that d's embedding vectors on the unit hyper-sphere is
        # near: c_emb + (b_emb - a_emb), which has the shape [N, emb_dim].
        target = c_emb + (b_emb - a_emb)
        
        # Compute cosine distance between each pair of target and vocab.
        #dist has shape [N, vocab_size]
        dist = tf.matmul(target, nemb, transpose_b=True)
        
        # For each question (row in dist), find the top 4 words
        _, pred_idx = tf.nn.top_k(dist, 4)
        
        # Nodes for computing neighbors for a given word according to 
        # their cosine distance
        nearby_word = tf.placeholder(dtype=tf.int32)
        nearby_emb = tf.gather(nemb, nearby_word)
        nearby_dist = tf.matmul(nearby_emb, nemb, transpose_b=True)
        nearby_val, nearby_idx = tf.nn.top_k(nearby_dist,
                                             min(1000, self._options.vocab_size))
        
        # Nodes in the construct graph which are used by training and 
        # evaluation to run/feed/fetch.
        self._analogy_a = analogy_a
        self._analogy_b = analogy_b
        self._analogy_c = analogy_c
        self._analogy_pred_idx = pred_idx
        self._nearby_word = nearby_word
        self._nearby_val = nearby_val
        self._nearby_idx = nearby_idx
        
    
    def build_graph(self):
        """Build the graph for the full model."""
        opts = self._options
        
        # The training data. A text file.
        (words, counts, words_per_epoch, self._epoch, self._words, examples,
         labels) = word2vec.skipgram_word2vec(filename=opts.train_data,
                                              batch_size=opts.batch_size,
                                              window_size=opts.window_size,
                                              min_count=opts.min_count,
                                              subsample=opts.subsample)
         
        (opts.vocab_words, opts.vocab_counts,
         opts.words_per_epoch) = self._session.run([words, counts, words_per_epoch])
        opts.vocab_size = len(opts.vocab_words)
        print('Data file: ', opts.train_data)
        print('Vocab size: ', opts.vocab_size - 1, ' + UNK')
        print('Words per epoch: ', opts.words_per_epoch)
        self._examples = examples
        self._labels = labels
        self._id2word = opts.vocab_words
        for i, w in enumerate(self._id2word):
            self._word2id[w] = i
        
        true_logits, sampled_logits = self.forwar(examples, labels)
        loss = self.nce_loss(true_logits, sampled_logits)
        tf.summary.scalar('NCE loss', loss)
        self._loss = loss
        self.optimize(loss)
        
        tf.global_variables_initializer().run()
        self.saver = tf.train.Saver()
        
    def save_vocab(self):
        """Save the vocabulary to a file so the model can be reloaded."""
        opts = self._options
        with open(os.path.join(opts.save_path, 'vocab.txt'), 'w') as f:
            for i in xrange(opts.vocab_size):
                vocab_word = tf.compat.as_text(opts.vocab_words[i]).encode('utf-8')
                f.write("%s %d\n" % (vocab_word, opts.vocab_counts[i]))
                
    
    def _train_thread_body(self):
        initial_epoch, =self._session.run([self._epoch])
        while True:
            _, epoch = self._session.run([self._train, self._epoch])
            if epoch != initial_epoch:
                break
            
    
    def train(self):
        opts = self._options
        
        initial_epoch, initial_words = self._session.run([self._epoch, self._words])
        
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(opts.save_path, self._session.graph)
        workers = []
        for _ in xrange(opts.concurrent_steps):
            t = threading.Thread(target=self._train_thread_body)
            t.start()
            workers.append(t)
        
        last_words, last_time, last_summary_time = initial_words, time.time(), 0
        last_checkpoint_time = 0
        while True:
            time.sleep(opts.statistics_interval)
            (epoch, step, loss, words, lr) = self.session.run(
                [self._epoch, self.global_step, self._loss, self._words, self._lr]
                )
            
            now = time.time()
            last_words, last_time, rate = words, now, (words - last_words) / (now - last_time)
            print("Epoch %4d step %8d: lr = %5.3f loss = %6.2f words/sec = %8.0f\r" %
                  (epoch, step, lr, loss, rate), end="")
            sys.stdout.flush()
            if now - last_summary_time > opts.summary_interval:
                summary_str = self._session.run(summary_op)
                summary_writer.add_summary(summary_str, step)
                last_summary_time = now
            if now - last_checkpoint_time > opts.checkpoint_interval:
                self.saver.save(self._session,
                                os.path.join(opts.save_path, 'model.ckpt'),
                                global_step = step.astype(int))
                last_checkpoint_time = now 
            if epoch != initial_epoch:
                break
        
        for t in workers:
            t.join()
            
        return epoch
    
    def _predict(self, analogy):
        """Predict the top 4 answers for analogy questions."""
        idx, = self._session.run([self._analogy_pred_idx], {
            self._analogy_a: analogy[:, 0],
            self._analogy_b: analogy[:,1],
            self._analogy_c: analogy[:, 2]})
        
        return idx
    
    
    def eval(self):
        # How many questions we get right at precision@1
        correct = 0
        
        try:
            total = self._analogy_questions.shape[0]
        except AttributeError as e:
            raise AttributeError("Need to read analogy questions.")
        
        start = 0
        while start < total:
            limit = start + 2500
            sub = 
        
           
        
