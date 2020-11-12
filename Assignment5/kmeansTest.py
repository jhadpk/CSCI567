# DO NOT CHANGE THIS FILE

import numpy as np
from data_loader import toy_dataset, load_digits
from kmeans import KMeans, KMeansClassifier, get_k_means_plus_plus_center_indices as k_plus, get_lloyd_k_means as k_vanilla, transform_image
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from utils import Figure
from sklearn.metrics import mean_squared_error

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

################################################################################
# KMeans on 2D toy dataset
# The dataset is generated from N Gaussian distributions equally spaced on N radius circle.
# Here, N=4
# KMeans on this dataset should be able to identify the 4 clusters almost clearly.
################################################################################


def kmeans_toy():
    print("[+] K-Means on Toy Dataset")

    print("[+] K-Means Vanilla")
    kmeans_builder(k_vanilla)
    print()

    print("[+] K-Means Plus Plus")
    kmeans_builder(k_plus)
    print()


def kmeans_builder(centroid_func):
    samples_per_cluster = 50
    n_cluster = 4

    x, y = toy_dataset(n_cluster, samples_per_cluster)
    fig = Figure()
    fig.ax.scatter(x[:, 0], x[:, 1], c=y)
    fig.savefig('plots/toy_dataset_real_labels.png')

    fig.ax.scatter(x[:, 0], x[:, 1])
    fig.savefig('plots/toy_dataset.png')

    k_means = KMeans(n_cluster=n_cluster, max_iter=100, e=1e-8)

    centroids, membership, i = k_means.fit(x, centroid_func)



    assert centroids.shape == (n_cluster, 2), \
        ('centroids for toy dataset should be numpy array of size {} X 2'
            .format(n_cluster))

    assert membership.shape == (samples_per_cluster * n_cluster,), \
        'membership for toy dataset should be a vector of size {}'.format(len(membership))

    assert type(i) == int and i > 0,  \
        'Number of updates for toy datasets should be integer and positive'

    print('[success] : kmeans clustering done on toy dataset')
    print('Toy dataset K means clustering converged in {} steps'.format(i))

    fig = Figure()
    fig.ax.scatter(x[:, 0], x[:, 1], c=membership)
    fig.ax.scatter(centroids[:, 0], centroids[:, 1], c='red')
    fig.savefig('plots/toy_dataset_predicted_labels.png')



################################################################################
# KMeans for image compression
# Here we use k-means for compressing an image
# We load an image 'baboon.tiff',  scale it to [0,1] and compress it.
################################################################################


def kmeans_image_compression():

    print("[+] K-Means Image Compression")
    im = plt.imread('baboon.tiff')
    N, M = im.shape[:2]
    im = im / 255

    # convert to RGB array
    data = im.reshape(N * M, 3)

    k_means = KMeans(n_cluster=16, max_iter=100, e=1e-6)
    centroids, _, i = k_means.fit(data)

    print('[+] RGB centroids computed in {} iteration'.format(i))
    new_im = transform_image(im, centroids)

    assert new_im.shape == im.shape, \
        'Shape of transformed image should be same as image'

    mse = np.sum((im - new_im)**2) / (N * M)
    print('[+] Mean square error per pixel is {}\n'.format(mse))
    plt.imsave('plots/compressed_baboon.png', new_im)




################################################################################
# Kmeans for classification
# Here we use k-means for classifying digits
# We find N clusters in the data and label each cluster with the majority class that belongs to that cluster.
# Test samples are labeled based on which cluster they belong to
################################################################################


def kmeans_classification():
    print("[+] K-Means Classification")

    x_train, x_test, y_train, y_test = load_digits()

    print("[+] K-Means Vanilla")
    kmeans_classification_builder(k_vanilla, x_train, x_test, y_train, y_test)
    print()

    print("[+] K-Means Plus Plus")
    kmeans_classification_builder(k_plus, x_train, x_test, y_train, y_test)

    try:
        linear_classifier = LogisticRegression()
        linear_classifier.fit(x_train, y_train)
        y_hat_test = linear_classifier.predict(x_test)
    except:
	    pass
    print('[*] Accuracy of logistic regression classifier is {}'
          .format(np.mean(y_hat_test == y_test)))

    KNNClassifier = KNeighborsClassifier()
    KNNClassifier.fit(x_train, y_train)
    y_hat_test = KNNClassifier.predict(x_test)
    print('[*] Accuracy of Nearest Neighbour classifier is {}'
          .format(np.mean(y_hat_test == y_test)))


def kmeans_classification_builder(centroid_func, x_train, x_test, y_train, y_test):

    # plot some train data
    N = 25
    l = int(np.ceil(np.sqrt(N)))

    im = np.zeros((10 * l, 10 * l))
    for m in range(l):
        for n in range(l):
            if (m * l + n < N):
                im[10 * m:10 * m + 8, 10 * n:10 * n +
                    8] = x_train[m * l + n].reshape([8, 8])
    plt.imsave('plots/digits.png', im, cmap='Greys')

    n_cluster = 10
    classifier = KMeansClassifier(n_cluster=n_cluster, max_iter=100, e=1e-6)

    classifier.fit(x_train, y_train, centroid_func)
    y_hat_test = classifier.predict(x_test)

    assert y_hat_test.shape == y_test.shape, \
        'y_hat_test and y_test should have same shape'

    print('[*] Prediction accuracy of K-means classifier with {} cluster is {}'.
          format(n_cluster, np.mean(y_hat_test == y_test)))


if __name__ == '__main__':
    kmeans_toy()
    kmeans_image_compression()
    kmeans_classification()
