from sklearn.model_selection import train_test_split
from sklearn import datasets
import numpy as np
import time


def toy_dataset(total_clusters=2, sample_per_cluster=50):
    # TODO: add sample size ease
    np.random.seed(int(time.time()))
    N = total_clusters * sample_per_cluster
    y = np.zeros(N)
    np.random.seed(43)
    x = np.random.standard_normal(size=(N, 2))
    for i in range(total_clusters):
        theta = 2*np.pi*i / total_clusters
        x[i*sample_per_cluster:(i+1)*sample_per_cluster] = x[i*sample_per_cluster:(i+1)*sample_per_cluster] + \
            (total_clusters * np.cos(theta), total_clusters * np.sin(theta))
        y[i*sample_per_cluster:(i+1)*sample_per_cluster] = i

    
    return x, y


def load_digits():
    digits = datasets.load_digits()
    x = digits.data/16
    x = x.reshape([x.shape[0], -1])
    y = digits.target
    
    return train_test_split(x, y, random_state=42, test_size=0.25)




