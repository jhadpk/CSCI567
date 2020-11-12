from sklearn.datasets import make_classification, make_blobs, make_moons, load_iris
from sklearn.model_selection import train_test_split
# from linear_regression import mapping_data

import json
import numpy as np
import pandas as pd


# Binary classification data
def toy_data_binary():
  data = make_classification(n_samples=500, 
                              n_features=2,
                              n_informative=1, 
                              n_redundant=0, 
                              n_repeated=0, 
                              n_classes=2, 
                              n_clusters_per_class=1, 
                              class_sep=1., 
                              random_state=42)

  X_train, X_test, y_train, y_test = train_test_split(data[0], data[1], train_size=0.7, random_state=42)
  return X_train, X_test, y_train, y_test


def moon_dataset():
  data = make_moons(n_samples=500, shuffle=True, noise=0.2, random_state=42)
  X_train, X_test, y_train, y_test = train_test_split(data[0], data[1], train_size=0.7, random_state=42)
  return X_train, X_test, y_train, y_test


# Multiple classification data

def toy_data_multiclass():
  data = make_blobs(n_samples=2000,
                     n_features=2,
                     random_state=42,
                     centers=[[-0.5, 1],[0, 1.5],[0.5, 1]],
                     cluster_std=0.2);
  X_train, X_test, y_train, y_test = train_test_split(data[0], data[1], train_size=0.7, random_state=42)
  return X_train, X_test, y_train, y_test


# Hand-written digits data
def data_loader_mnist(dataset='mnist_subset.json'):
  # This function reads the MNIST data and separate it into train, val, and test set
  with open(dataset, 'r') as f:
        data_set = json.load(f)
  train_set, valid_set, test_set = data_set['train'], data_set['valid'], data_set['test']

  return np.asarray(train_set[0]), \
          np.asarray(test_set[0]), \
          np.asarray(train_set[1]), \
          np.asarray(test_set[1])



