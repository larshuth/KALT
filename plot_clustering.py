import matplotlib.pyplot as plt
from matplotlib import cm
import streamlit as st
from itertools import cycle
import numpy as np

from scipy.cluster.hierarchy import dendrogram


def plotting_mean_shift(mean_shift, labels, n_clusters, data):
    """
    Displays 2D-plot of mean shift clustering points.
    Based on:
    https://scikit-learn.org/stable/auto_examples/cluster/plot_mean_shift.html#sphx-glr-auto-examples-cluster-plot-mean-shift-py
    https://matplotlib.org/
    https://stackoverflow.com/questions/925024/how-can-i-remove-the-top-and-right-axis-in-matplotlib


    @param mean_shift: mean shift instance
    @param labels: index of cluster each data point belongs to
    @param n_clusters: number of clusters
    @param data: processed data (e.g. liver_disorders)
    """

    fig = plt.figure(1)

    colors = cycle("bcmrgykbgrcmykbgrcmykbgrcmyk")
    for k, col in zip(range(n_clusters), colors):
        my_members = labels == k
        cluster_center = mean_shift.cluster_centers_[k]
        plt.plot(
            data[my_members, 0],
            data[my_members, 1],
            col + ".",
            markeredgecolor="#fff",
            markeredgewidth=0.7,
            markersize=8,
        )
        plt.plot(
            cluster_center[0],
            cluster_center[1],
            "o",
            markerfacecolor=col,
            markeredgecolor="#fff",
            markersize=10,
        )

    plt.title("Mean Shift - #clusters: %d" % n_clusters)
    ax = plt.gca()
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_facecolor("#eff2f7")

    plt.grid(color="#fff")


def plotting_dbscan(dbscan, labels, n_clusters, data, x_var="", y_var=""):
    """
    Displays 2D-plot of dbscans.
    Based on:
    https://www.geeksforgeeks.org/implementing-dbscan-algorithm-using-sklearn/
    """

    fig = plt.figure(1)

    colors = cycle("bcmrgykbgrcmykbgrcmykbgrcmyk")
    for k, col in zip(range(-1, n_clusters), colors):
        my_members = labels == k
        plt.plot(
            data[my_members, 0],
            data[my_members, 1],
            col + ".",
            markeredgecolor="#fff",
            markeredgewidth=0.7,
            markersize=8,
        )

    plt.title(f"DBSCAN - #clusters: {n_clusters}")
    # according to the colour vector defined
    ax = plt.gca()
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_facecolor("#eff2f7")
    plt.grid(color="#fff")


def plotting_kmeans(kmeans, labels, n_clusters, data):
    """
    Displays 2D-plot of k-Means clustering points.
    @param kmenas: k-Means instance
    @param labels: index of cluster each data point belongs to
    @param n_clusters: number of clusters
    @param data: processed data (e.g. seeds)
    """
    fig = plt.figure(1)

    colors = cycle("bcmrgykbgrcmykbgrcmykbgrcmyk")
    for k, col in zip(range(-1, n_clusters), colors):
        my_members = labels == k
        plt.plot(
            data[my_members, 0],
            data[my_members, 1],
            col + ".",
            markeredgecolor="#fff",
            markeredgewidth=0.7,
            markersize=8,
        )

    plt.title(f"k-Means - #clusters: {n_clusters}")
    # according to the colour vector defined
    ax = plt.gca()
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_facecolor("#eff2f7")
    plt.grid(color="#fff")


def plotting_hac(hac_algo, labels, n_clusters, data):
    """
    Displays 2D-plot of k-Means clustering points.
    Based on:
    https://matplotlib.org/
    https://stackoverflow.com/questions/925024/how-can-i-remove-the-top-and-right-axis-in-matplotlib
    
    @param hac_algo: HAC instance/model
    @param labels: index of cluster each data point belongs to
    @param n_clusters: number of clusters
    @param data: processed data
    """

    colors = cycle("bcmrgykbgrcmykbgrcmykbgrcmyk")
    for k, col in zip(range(-1, n_clusters), colors):
        my_members = labels == k
        plt.plot(
            data[my_members, 0],
            data[my_members, 1],
            col + ".",
            markeredgecolor="#fff",
            markeredgewidth=0.7,
            markersize=8,
        )

    plt.title(f"HAC - #clusters: {n_clusters}")
    # according to the colour vector defined
    ax = plt.gca()
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_facecolor("#eff2f7")
    plt.grid(color="#fff")


def plot_dendrogram(model, **kwargs):
    """
    Creates a linkage matrix and then creates a dendrogram.
    Based on:
    https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html
    
    @param model: HAC instance/model
    @**kwargs: adjustable parameters for the dendrogram
    """
    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    dendrogram(linkage_matrix, **kwargs)


def show_estimated_clusters_hac(model, clusters):
    """
    Plots a dendrogram for the processed data. Shows only the n biggest clusters.
    
    @param model: HAC instance/model
    @param clusters: integer value for n biggest clusters
    """
    # dendrogram
    fig = plt.figure(figsize=(15, 5))
    plt.title("Hierarchical Clustering Dendrogram")
    plot_dendrogram(model, truncate_mode="lastp", p=clusters)
    plt.xlabel("Number of points in node.")
    plt.ylabel("Distances between new clusters.")
    st.pyplot(fig)


def evaluation_plot(results):
    results_t = results.T

    # print out the pure values
    st.dataframe(results)

    metrics_optimum = {
        'purity': 'bigger better', 'rand': 'bigger better', 'jaccard': 'bigger better',
        'davies': 'lower better', 'silhouette': 'bigger better', 'dunn': 'bigger better'
    }

    names = ["DBSCAN", "Mean Shift", "k-Means", "Hierarchical\nAgglomerative\nClustering"]

    fig = plt.figure()

    counter = 1
    for index, metric in results_t.iterrows():
        metric_l = list(metric)
        if metrics_optimum[index] == 'bigger better' and index != 'davies':
            colors = cm.RdYlGn([y / float(max(metric_l)) for y in metric_l])
        else:
            colors = cm.RdYlGn_r([((y - float(min(metric_l))) / float(max(metric_l))) for y in metric_l])

        ax = plt.subplot(310 + counter)
        plt.bar(list(dict(metric)), metric_l, color=colors)
        plt.ylabel('Score')
        ax.set_title(index)
        if counter < 3:
            plt.xticks([], [])
        else:
            plt.xticks(list(dict(metric)), names, rotation='horizontal')
        counter += 1

    st.pyplot(fig)
