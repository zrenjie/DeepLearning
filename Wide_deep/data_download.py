"""Download and clean the census income dataset."""

import tensorflow as tf
import argparse
import os
import sys

from six.moves import urllib

DATA_URL = "http://archive.ics.uci.edu/ml/machine-learning-databases/adult"
TRAINING_FILE = 'adult.data'
TRAINING_URL = '%s/%s' % (DATA_URL, TRAINING_FILE)
EVAL_FILE = 'adult.test'
EVAL_URL='%s/%s' % (DATA_URL, EVAL_FILE)

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', type=str, default='/tmp/census_data',
                    help='Directory to download census data')

def _download_and_clean_file(filename, url):
    """Download data from url, and makes changes to match the CSV format. """
    temp_file, _ = urllib.request.urlretrieve(url)
    with tf.gfile.Open(temp_file, 'r') as temp_eval_file:
        with tf.gfile.Open(filename, 'w') as eval_file:
            for line in temp_eval_file:
                line = line.strip()
                line = line.replace(', ', ',')
                
                if not line or ',' not in line:
                    continue
                if line[-1] == '.':
                    line = line[:-1]
                
                line += '\n'
                eval_file.write(line)
    tf.gfile.Remove(temp_file)
                    
def main(unused_argv):
    if not tf.gfile.Exists(FLAGS.data_dir):
        tf.gfile.MakeDirs(FLAGS.data_dir)
        
    training_file_path = os.path.join(FLAGS.data_dir, TRAINING_FILE)
    _download_and_clean_file(training_file_path, TRAINING_URL)
    
    eval_file_path = os.path.join(FLAGS.data_dir, EVAL_FILE)
    _download_and_clean_file(eval_file_path, EVAL_URL)
    
    
if __name__ == '__main__':
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(argv=[sys.argv[0]] + unparsed)
