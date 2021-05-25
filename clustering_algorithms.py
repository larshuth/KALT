import numpy as np
import matplotlib.pyplot as plt

import dataset_tranformations

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

import pandas as pd


def plotting_dbscan(dataset, labels, x_var='', y_var=''):
    # Building the label to colour mapping
    colours = {}

    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]

    colours[-1] = 'k'

    # Building the colour vector for each data point
    cvec = [colors[label] for label in labels]

    # For the construction of the legend of the plot
    if not x_var and y_var:
        x_var = labels[0]
        y_var = labels[1]

    r = plt.scatter(dataset[x_var], dataset[y_var], color='r')
    g = plt.scatter(dataset[x_var], dataset[y_var], color='g')
    b = plt.scatter(dataset[x_var], dataset[y_var], color='b')
    k = plt.scatter(dataset[x_var], dataset[y_var], color='k')

    # according to the colour vector defined
    plt.figure(figsize=(9, 9))
    plt.scatter(dataset[x_var], dataset[y_var], c=cvec)

    plt.show()
    return


def density_based_spatial_clustering_of_applications_with_noise(dataset):
    db = DBSCAN(eps=0.3, min_samples=5).fit(dataset)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True

    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    return db, labels


def main(algorithm='dbscan', dataset='happiness and alcohol', pca_bool=True):
    print("pick a god and pray")

    algorithms = {'dbscan': density_based_spatial_clustering_of_applications_with_noise}
    datasets = {'happiness and alcohol': dataset_tranformations.happiness_alcohol_consumption}

    x, x_principal = datasets[dataset](pca_bool=pca_bool)
    db, labels = algorithms[algorithm](x_principal)
    plotting_dbscan(x, labels, 'HappinessScore', 'HDI')
    return 0


if __name__ == "__main__":
    main()
