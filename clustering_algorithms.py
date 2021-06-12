import numpy as np
import plot_clustering

from sklearn.cluster import (
    DBSCAN,
    MeanShift,
    KMeans,
    AgglomerativeClustering
)
from sklearn.metrics import davies_bouldin_score, jaccard_score, silhouette_score, rand_score, cluster
from dunnindex.dunn_sklearn import dunn


def density_based_spatial_clustering_of_applications_with_noise(
    dataset_x, dbscan_params, dataset_y=None
):
    """
    Performs density-based clustering of applications with noise on datasets transformed as we as a group we agreed
    upon. This code is based upon
    https://www.geeksforgeeks.org/implementing-dbscan-algorithm-using-sklearn/
    @param dataset_x: features of the dataset as an array (required)
    @param dataset_y: labels of the dataset as an array (not required, default = none)
    @param dbscan_params: epsilon neighborhood and cluster neighborhood as required for dbscan
    """
    db = DBSCAN(
        eps=dbscan_params["epsilon_neighborhood"],
        min_samples=dbscan_params["clustering_neighborhood"],
    ).fit(dataset_x)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True

    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)
    return db, labels, n_clusters_


def mean_shift(data, meanshift_params):
    """
    Performs Mean Shift clustering on given dataset.
    Based on:
    https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MeanShift.html
    @param data: processed dataset (e.g. liver disorders dataset)
    @param meanshift_params['bandwidth']: distance of kernel function or size of "window", either automatically estimated or given by user
    @return mean shift instance, index of cluster each data point belongs to, number of clusters
    """

    mean_shift = MeanShift(bandwidth=meanshift_params["bandwidth"])

    mean_shift.fit(data)
    labels = mean_shift.labels_
    cluster_centers = mean_shift.cluster_centers_
    labels_unique = np.unique(labels)
    n_clusters = len(labels_unique)

    return mean_shift, labels, n_clusters


def k_Means(dataset_x, k_means_params):
    """
    Performs k-Means clustering on given dataset.
    Based on:
    https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

    @param dataset_x: features of the dataset as an array (required)
    @param k_means_params: parameters for the algorithm
    @return kmeans instance, index of cluster each data point belongs to, number of clusters
    """
    n_clusters = k_means_params["clusters"]
    kmeans = KMeans(n_clusters=k_means_params["clusters"])
    kmeans.fit(dataset_x)
    labels = kmeans.labels_
    return kmeans, labels, n_clusters


def optimal_cluster_count(dataset_x):
    sil = []
    kmax = 8
    # dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
    for k in range(3, kmax + 1):
        kmeans = KMeans(n_clusters=k).fit(dataset_x)
        labels = kmeans.labels_
        sil.append(silhouette_score(dataset_x, labels, metric="euclidean"))

    return int(np.argmax(sil) + 3)


def ahc_algo(data, ahc_algo_params):
    """
    Fits the model while using allgomorative hierarchical clustering.
    Plots the result eaither by showing a dendogram, a scatter or both.
    @param data: the data to be used for ahc algorithm
    @param show_dendrogram: if you want to show the result through using a dandogram
    @param show_scatter: if you want to show the results though scattering the datapoints
    @param n_clusters: if you want plot the scattered data use n_clusters to show n clusters
    """
    n_clusters = ahc_algo_params["n_clusters"]
    link = ahc_algo_params["link"]

    # for scatter
    cluster = AgglomerativeClustering(n_clusters, affinity="euclidean", linkage=link)
    cluster.fit_predict(data)

    labels = cluster.labels_

    return cluster, labels, n_clusters


def estimate_clusters_ahc(data, link, clusters):
    model = AgglomerativeClustering(distance_threshold=0, n_clusters=None, linkage=link)
    model = model.fit(data)

    plot_clustering.show_estimated_clusters_ahc(model, clusters)


def purity_score(labels_true, labels_pred):
    """
    Purity Metric as described in https://stackoverflow.com/questions/34047540/python-clustering-purity-metric.

    :param y_true:
    :param y_pred:
    :return: metric of purity
    """
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = cluster.contingency_matrix(labels_true, labels_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


def evaluation(datapoints, labels_pred, labeled=False, labels_real=None):
    if labeled:
        # purity
        purity = purity_score(labels_real, labels_pred)
        # Rand
        rand = rand_score(labels_real, labels_pred)
        # Jaccard
        jaccard = jaccard_score(labels_real, labels_pred, average='macro')
        return {'purity': [purity], 'rand': [rand], 'jaccard': [jaccard]}
    else:
        # Davies Bouldin
        davies_bouldin = davies_bouldin_score(datapoints, labels_pred)
        # Silhouette Coefficient
        silhouette = silhouette_score(datapoints, labels_pred)
        # Dunn
        pairwise_distances = np.array(list(np.array(list(np.linalg.norm(i - j) for i in datapoints)) for j in datapoints))
        dunn_ = dunn(labels_pred, pairwise_distances)
        return {'davies': [davies_bouldin], 'silhouette': [silhouette], 'dunn': [dunn_]}

