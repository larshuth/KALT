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
    dataset_x, dbscan_params=None
):
    """
    Performs density-based clustering of applications with noise on datasets transformed as we as a group we agreed
    upon. This code is based upon
    https://www.geeksforgeeks.org/implementing-dbscan-algorithm-using-sklearn/
    @param dataset_x: features of the dataset as an numpy.ndarray (required)
    @param dbscan_params: epsilon neighborhood and cluster neighborhood as required for dbscan (optional)
    @return: DBSCAN.fit instance, , number of clusters in the clustering
    """
    if not dbscan_params:
        dbscan_params = {
            "epsilon_neighborhood": 0.3,
            "clustering_neighborhood": 5
        }
    # creation of DBSCAN instance and clustering the data set
    db = DBSCAN(
        eps=dbscan_params["epsilon_neighborhood"],
        min_samples=dbscan_params["clustering_neighborhood"],
    ).fit(dataset_x)
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)   # number of noise points

    return db, labels, n_clusters


def mean_shift(data, meanshift_params):
    """
    Performs Mean Shift clustering on given dataset.
    Based on:
    https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MeanShift.html

    @param data: processed dataset (e.g. liver disorders dataset)
    @param meanshift_params: parameters used for mean shift, has a single key 'bandwidth': 
        distance of kernel function or size of 'window', either automatically estimated or given by user
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
    kmeans = KMeans(n_clusters=k_means_params["clusters"], random_state = k_means_params["random_state"])
    kmeans.fit(dataset_x)
    labels = kmeans.labels_
    return kmeans, labels, n_clusters


def optimal_cluster_count(dataset_x):
    """
    Uses the silhouette index for computing a cluster count for a certain dataset.
    @param dataset_x: features of the dataset as an array (required)
    @return estimated optimal cluster count
    """
    sil = []
    kmax = 8
    # dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
    for k in range(3, kmax + 1):
        kmeans = KMeans(n_clusters=k).fit(dataset_x)
        labels = kmeans.labels_
        sil.append(silhouette_score(dataset_x, labels, metric="euclidean"))

    return int(np.argmax(sil) + 3)


def hac_algo(data, hac_algo_params):
    """
    Performs Hierarchical Agglomerative Clustering on a given data set.
    Based on:
    https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html
    
    @param data: the data to be used for hac algorithm as an array
    @param hac_algo_params: parameters for the algorithm, namely the number of clusters and the linkage
    @return: HAC model, labels of the model, number of clusters 
    """
    n_clusters = hac_algo_params["n_clusters"]
    link = hac_algo_params["link"]

    # for scatter
    cluster = AgglomerativeClustering(n_clusters, affinity="euclidean", linkage=link)
    cluster.fit_predict(data)

    labels = cluster.labels_

    return cluster, labels, n_clusters


def estimate_clusters_hac(data, link, clusters):
    """
    This function uses the HAC algorithm on a given data set to show the results in a dedrogram.
    The dendrogram will only show the biggest n clusters stored in parameter 'clusters'.
    Based on:
    https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html
    
    @param data: the preprocessed data set as an array
    @param link: the linkage method to be used for HAC algorithm
    @clusters: the number of the biggest clusters to be plotted in a dendrogram
    """
    model = AgglomerativeClustering(distance_threshold=0, n_clusters=None, linkage=link)
    model = model.fit(data)

    plot_clustering.show_estimated_clusters_hac(model, clusters)


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


def work_with_noise(datapoints, labels_pred, labeled=False, labels_real=None):
    """
    DBSCAN labels all noise points with -1, which is interpreted as its own cluster by the evaluation metrics.
    Using this function filters out the points labeled as noise from all givine inputs
    :param datapoints: data set as array of numpy ndarrays
    :param labels_pred: array of labels as assigned by the previously used clustering algorithm
    :param labeled: Boolean value indicating whether there are real labels associated with the dataset
    :param labels_real: array of labels as given in the original data set source
    :return: input but all indices labeled as -1 in labels_pred filtered out
    """
    if -1 in labels_pred:
        not_noise = list(label != -1 for label in labels_pred)
        datapoints_without_noise = list(datapoints[i] for i in range(len(datapoints)) if not_noise[i])
        labels_pred_without_noise = list(labels_pred[i] for i in range(len(labels_pred)) if not_noise[i])
        if labeled:
            labels_real_without_noise = list(labels_real[i] for i in range(len(labels_real)) if not_noise[i])
        else:
            labels_real_without_noise = labels_real

    else:
        datapoints_without_noise = datapoints
        labels_pred_without_noise = labels_pred
        labels_real_without_noise = labels_real
    return datapoints_without_noise, labels_pred_without_noise, labels_real_without_noise


def evaluation(datapoints, labels_pred, labeled=False, labels_real=None, secret_lars_lever=False):
    """
    Evaluation of the input clustering using external (if labeled) or internal validation metrics.
    :param datapoints: data set as array of numpy ndarrays
    :param labels_pred: array of labels as assigned by the previously used clustering algorithm
    :param labeled: Boolean value indicating whether there are real labels associated with the dataset
    :param labels_real: array of labels as given in the original data set source
    :param secret_lars_lever: Bool value indicating a modified approach for the evaluation
    :return: Dictionary of calculated scores
    """
    if secret_lars_lever:
        datapoints, labels_pred, labels_real = work_with_noise(datapoints, labels_pred, labeled, labels_real)

    if labeled:
        # external validation methods (require real labels)
        # purity
        purity = purity_score(labels_real, labels_pred)
        # Rand
        rand = rand_score(labels_real, labels_pred)
        # Jaccard
        jaccard = jaccard_score(labels_real, labels_pred, average='macro')
        return {'purity': [purity], 'rand': [rand], 'jaccard': [jaccard]}
    else:
        # internal validation methods
        # including safety net to avoid crashing when only one cluster is sent in to be evaluated.
        if len(set(labels_pred)) > 1:
            # Davies Bouldin
            davies_bouldin = davies_bouldin_score(datapoints, labels_pred)
            # Silhouette Coefficient
            silhouette = silhouette_score(datapoints, labels_pred)
            # Dunn
            pairwise_distances = np.array(
                list(np.array(list(np.linalg.norm(i - j) for i in datapoints)) for j in datapoints))
            dunn_ = dunn(labels_pred, pairwise_distances)
        else:
            davies_bouldin = max(list(np.linalg.norm(i - j) for i in datapoints for j in datapoints))
            silhouette = 0
            dunn_ = 0
        return {'davies': [davies_bouldin], 'silhouette': [silhouette], 'dunn': [dunn_]}

