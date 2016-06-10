import numpy as np
import pandas as pd
from random import random, uniform
from sklearn.datasets.samples_generator import make_blobs
import sys
from sklearn import datasets


def generate_spherical_clusters(number_of_samples, number_of_clusters, n_features=2,  variances=None, filename=""):
    """
    :param number_of_samples:  The total number of points equally divided among clusters.
    :param number_of_clusters: The number of clusters to generate
    :param n_features:         The number of features for each sample.
    :param variances:          The standard deviation of the clusters.
    :param filename:           The file to store the results
    :return:
    """
    if variances is None: variances = [0.5 for _ in xrange(number_of_clusters)]
    if filename == "":
        filename = "./Data/spherical_" + str(number_of_samples) + "_features_" + str(n_features) \
                                  + "_cluster_" + str(number_of_clusters) + ".csv"
    random_state = 170
    X, y = make_blobs(n_samples=number_of_samples, centers=number_of_clusters, n_features=n_features,
                      random_state=random_state, cluster_std=variances)
    features = ["features_" + str(i+1) for i in xrange(n_features)]
    df = pd.DataFrame()
    for i, feature in enumerate(features): df[feature] = X[:, i]
    df["class"] = y
    df.to_csv(filename, index=False)

    return X, y


def _generate_spherical_clusters():
    X, y = generate_spherical_clusters(10000, 4)
    import matplotlib.pyplot as plt
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.show()


def generate_anisotropically_clusters(number_of_samples, number_of_clusters, n_features=2,  variances=None, filename=""):
    """
    :param number_of_samples:  The total number of points equally divided among clusters.
    :param number_of_clusters: The number of clusters to generate
    :param n_features:         The number of features for each sample.
    :param variances:          The standard deviation of the clusters.
    :param filename:           The file to store the results
    :return:
    """

    if variances is None: variances = [0.5 for _ in xrange(number_of_clusters)]
    if filename == "":
        filename = "./Data/anisotropically_" + str(number_of_samples) + "_features_" + str(n_features) \
                   + "_cluster_" + str(number_of_clusters) + ".csv"
    random_state = 170
    X, y = make_blobs(n_samples=number_of_samples, centers=number_of_clusters, n_features=n_features,
                      random_state=random_state, cluster_std=variances)
    transformation = np.array([[random() if i == j else uniform(-1, 1) for j in xrange(n_features)] for i in xrange(n_features)])
    X = np.dot(X, transformation)

    features = ["features_" + str(i + 1) for i in xrange(n_features)]
    df = pd.DataFrame()
    for i, feature in enumerate(features): df[feature] = X[:, i]
    df["class"] = y
    df.to_csv(filename, index=False)

    return X, y


def _generate_anisotropically_clusters():

    X, y = generate_anisotropically_clusters(10000, 4)
    import matplotlib.pyplot as plt
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.show()


def generate_varied_clusters(number_of_samples, number_of_clusters, n_features=2,  variances=None, filename=""):
    """
    :param number_of_samples:  The total number of points equally divided among clusters.
    :param number_of_clusters: The number of clusters to generate
    :param n_features:         The number of features for each sample.
    :param variances:          The standard deviation of the clusters.
    :param filename:           The file to store the results
    :return:
    """
    if variances is None: variances = [uniform(1, 3) for _ in xrange(number_of_clusters)]
    if filename == "":
        filename = "./Data/varied_" + str(number_of_samples) + "_features_" + str(n_features) \
                                  + "_cluster_" + str(number_of_clusters) + ".csv"
    random_state = 170
    X, y = make_blobs(n_samples=number_of_samples, centers=number_of_clusters, n_features=n_features,
                      random_state=random_state, cluster_std=variances)
    features = ["features_" + str(i+1) for i in xrange(n_features)]
    df = pd.DataFrame()
    for i, feature in enumerate(features): df[feature] = X[:, i]
    df["class"] = y
    df.to_csv(filename, index=False)

    return X, y


def _generate_varied_clusters():
    X, y = generate_varied_clusters(10000, 4)
    import matplotlib.pyplot as plt
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.show()


if __name__ == "__main__":
    no_instances = [100, 1000, 10000, 100000, 1000000]
    no_features = [2, 4, 8, 16, 32, 64, 128, 256, 512]
    no_clusters = [2, 4, 6, 8]

    for no_instance in no_instances:
        for no_feature in no_features:
            for no_cluster in no_clusters:
                print "# ",
                sys.stdout.flush()
                generate_spherical_clusters(number_of_samples=no_instance, number_of_clusters=no_cluster, n_features=no_feature)
                generate_anisotropically_clusters(number_of_samples=no_instance, number_of_clusters=no_cluster, n_features=no_feature)
                generate_varied_clusters(number_of_samples=no_instance, number_of_clusters=no_cluster, n_features=no_feature)
            print
        print
