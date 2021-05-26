import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from itertools import cycle
from sklearn import preprocessing, metrics
from sklearn.cluster import MeanShift, KMeans, estimate_bandwidth
from sklearn.decomposition import PCA


def liver_disorders(pca_bool=True):
    liver_disorders = pd.read_csv('./datasets/liver_disorders.data', sep=',', names=[
        "Mean Corpuscular Volume",
        "Alkaline Phosphotase",
        "Alamine Aminotransferase",
        "Aspartate Aminotransferase",
        "Gamma-Glutamyl Transpeptidase",
        "Number of Half-Pint Equivalents of Alcoholic Beaverages Drunk per Day",
        "Selector"])
    # drop null values
    liver_disorders = liver_disorders.dropna()
    # drop "Selector" column since it was only used to split the data into train/test sets
    liver_disorders = liver_disorders.drop("Selector", axis=1)

    if pca_bool:
        scaler = preprocessing.StandardScaler()
        data_scaled = scaler.fit_transform(liver_disorders)
        #kriege normalized atm nur ein cluster :(
        #data_scaled = preprocessing.normalize(data_scaled)

        components = get_components(data_scaled)
        pca = PCA(n_components = components)
        pca.fit(data_scaled)
        scores_pca = pca.transform(data_scaled)

        return scores_pca
    
    else:
        return liver_disorders.values


def mean_shift(data):
    bandwidth = estimate_bandwidth(data)
    mean_shift = MeanShift(bandwidth=bandwidth, bin_seeding = True)

    mean_shift.fit(data)
    labels = mean_shift.labels_
    cluster_centers = mean_shift.cluster_centers_
    labels_unique = np.unique(labels)
    n_clusters = len(labels_unique)

    return mean_shift, labels, n_clusters


def plotting_mean_shift(mean_shift, labels, n_clusters, data):
    plt.figure(1)
    plt.clf()

    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    for k, col in zip(range(n_clusters), colors):
        my_members = labels == k
        cluster_center = mean_shift.cluster_centers_[k]
        plt.plot(data[my_members, 0], data[my_members, 1], col + '.')
        plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)
    plt.title('Estimated number of clusters: %d' % n_clusters)
    plt.show()


def get_components(data):
    pca = PCA()
    pca.fit(data)
    for i in range(0, len(pca.explained_variance_ratio_.cumsum())-1):
        if pca.explained_variance_ratio_.cumsum()[i] >= 0.8:
            return i+1
    #plt.plot(range(1,len(list(data))+1), pca.explained_variance_ratio_.cumsum(), marker = 'o', linestyle = '--')
    #plt.show()


plotting_mean_shift(mean_shift(liver_disorders())[0], mean_shift(liver_disorders())[1], mean_shift(liver_disorders())[2], liver_disorders())

