
import numpy as np

FILENAME = 'winequality-white.csv'

# Find the min and max of each column
def dataset_minmax(dataset):
    value_min = np.min(dataset,0)
    value_max = np.max(dataset,0)
    minmax = np.vstack((value_min, value_max))
    return minmax


# Normalize dataset into range 0-1
def normalize_dataset(dataset, minmax):
    for i in range(dataset.shape[1]):
        dataset[:, i] = (dataset[:, i] - minmax[i,0]) / (minmax[i,1] - minmax[i, 0])


def cross_validation_split(dataset, k_folds):
    pass


def rmse_metric(actual, predicted):
    pass


def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    pass



def predict(row, coefficients):
    pass



def coefficients_sgd(train, l_rate, n_epoch):
    pass



def linear_regression_sgd(train, test, l_rate, n_epoch):
    pass



# cross validation split


# predict 


# train

dataset = np.genfromtxt(FILENAME, delimiter=',')
dataset_minmax(dataset)
