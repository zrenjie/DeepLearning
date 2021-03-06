"""Example code for tensorflow wide & deep tutorial using tf.learn api."""

import argparse
import shutil
import sys

import tensorflow as tf
from boto.gs.lifecycle import AGE
from blaze.compute.core import base

_CSV_COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'gender',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
    'income_bracket'
    ]

_CSV_COLUMN_DEFAULTS = [[0], [''], [0], [''], [0], [''], [''], [''], [''], [''],
                        [0], [0], [0], [''], ['']]

parser = argparse.ArgumentParser()

parser.add_argument('--model_dir', type=str, default='/tmp/census_model',
                    help='Base directory for the model.')

parser.add_argument(
    '--model_type', type=str, default='wide_deep',
    help='Valid model types: {"wide", "deep", "wide_deep"}')

parser.add_argument(
    '--train_epochs', type=int, default=40,
    help='Number of training epochs.')

parser.add_argument(
    '--epochs_per_eval', type=int, default=2,
    help='The number of training epochs to run between evaluations.')

parser.add_argument(
    '--batch_size', type=int, default=40,
    help='Number of examples per batch.')

parser.add_argument(
    '--train_data', type=str, default='/tmp/census_data/adult.data',
    help='Path to the training data.')

parser.add_argument(
    '--test_data', type=str, default='/tmp/census_data/adult.test',
    help='Path to the test data.')

_NUM_EXAMPLES = {
    'train': 32561,
    'validation':16281
    }


def build_model_columns():
    """Build a set of wide and deep feature columns."""
    age = tf.feature_column.numeric_column('age')
    education_num = tf.feature_column.numeric_column('education_num')
    capital_gain = tf.feature_column.numeric_column('capital_gain')
    capital_loss = tf.feature_column.numeric_column('capital_loss')
    hours_per_week = tf.feature_column.numeric_column('hours_per_week')
    
    education = tf.feature_column.categorical_column_with_vocabulary_list('education', [
        'Bachelors', 'HS-grad', 'Masters', '9th', 'Some-college',
        'Assoc-acdm', 'Assoc-voc', '7th-8th', 'Doctorate', 'Prof-school',
        '5th-6th', '10th', '1st-4th', 'Preschool', '12th'])
    
    marital_status = tf.feature_column.categorical_column_with_vocabulary_list('marital_status', [
        'Married-civ-spouse', 'Divorced', 'Married-spouse-absent',
        'Never-married', 'Seperated', 'Married-AF-spouse', 'Widowed'])
    
    relationship = tf.feature_column.categorical_column_with_vocabulary_list('relationship', [
        'Husband', 'Not-in-family', 'Wife', 'Own-child', 'Unmarried',
        'Other-relative'])
    
    workclass = tf.feature_column.categorical_column_with_vocabulary_list('workclass', [
        'Self-emp-not-inc', 'Private', 'State-gov', 'Federal-gov',
        'Local-gov', '?', 'Self-emp_inc', 'Without-pay', 'Never-worked'])
    
    occupation = tf.feature_column.categorical_column_with_hash_bucket('occupation', hash_bucket_size=1000)
    
    age_buckets = tf.feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
    base_columns = [
        education, marital_status, relationship, workclass, occupation,
        age_buckets]
    
    crossed_columns = [
        tf.feature_column.crossed_column(['education', 'occupation'], hash_bucket_size=2000),
        tf.feature_column.crossed_column([age_buckets, 'education', 'occupation'], hash_bucket_size=1000)]
    
    wide_columns = base_columns + crossed_columns
    
    deep_columns = [
        age,
        education_num,
        capital_gain,
        capital_loss,
        hours_per_week,
        tf.feature_column.indicator_column(workclass),
        tf.feature_column.indicator_column(education),
        tf.feature_column.indicator_column(marital_status),
        tf.feature_column.indicator_column(relationship),
        tf.feature_column.embedding_column(occupation, dimension=8)]
    
    return wide_columns, deep_columns


def build_estimator(model_dir, model_type):
    """Build an estimator appropriate for the given model type."""
    wide_columns, deep_columns = build_model_columns()
    hidden_units = [100, 75, 50, 25]
    
    #Create a tf.estimator.RunConfig to ensure the model runs on CPU, which 
    # trains faster than GPU for this model
    run_config = tf.estimator.RunConfig().replace(session_config=tf.ConfigProto(device_count={'GPU':0}))
    
    if model_type == 'wide':
        return tf.estimator.LinearClassifier(
            model_dir=model_dir,
            feature_columns=wide_columns,
            config=run_config)
    elif model_type == 'deep':
        return tf.estimator.DNNClassifier(
            model_dir=model_dir,
            feature_columns=deep_columns,
            hidden_units=hidden_units,
            config=run_config)
    else:
        return tf.estimator.DNNLinearCombinedClassifier(
            model_dir=model_dir,
            linear_feature_columns=wide_columns,
            dnn_feature_columns=deep_columns,
            dnn_hidden_units=hidden_units,
            config=run_config
            )

def input_fn(data_file, num_epochs, shuffle, batch_size):
    """Generate an input function for the estimator."""
    assert tf.gfile.Exists(data_file), (
        '%s not found. Please make sure you have either run data_download.py or '
        'set both arguments --train_data and --test_data.' % data_file)
    
    def parse_csv(value):
        print('Parsing', data_file)
        columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS)
        features = dict(zip(_CSV_COLUMNS, columns))
        labels = features.pop('income_bracket')
        return features, tf.equal(labels, '>50K')
    
    dataset = tf.data.TextLineDataset(data_file)
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size = _NUM_EXAMPLES['train'])
        
    dataset = dataset.map(parse_csv, num_parallel_calls = 5)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    
    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    return features, labels

def main(unused_argv):
    # Clean up the model directory if present
    shutil.rmtree(FLAGS.model_dir, ignore_errors=True)
    model = build_estimator(FLAGS.model_dir, FLAGS.model_type)
    
    # Train and evaluate the model every 'FLAGS.epochs_per_eval' epochs
    for n in range(FLAGS.train_epochs // FLAGS.epochs_per_eval):
        model.train(input_fn=lambda: input_fn(
            FLAGS.train_data, FLAGS.epochs_per_eval, True, FLAGS.batch_size))
        
        results = model.evaluate(input_fn=lambda: input_fn(
            FLAGS.test_data, 1, False, FLAGS.batch_size))
        
        print('Results at epoch', (n + 1) * FLAGS.epochs_per_eval)
        print('-' * 60)
        
        for key in sorted(results):
            print('%s: %s' % (key, results[key]))
            
            
if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
    


