import matplotlib.pyplot as plt
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
    plt.clf()

    #farben ändern...
    colors = cycle('bcmrgykbgrcmykbgrcmykbgrcmyk')
    for k, col in zip(range(n_clusters), colors):
        my_members = labels == k
        cluster_center = mean_shift.cluster_centers_[k]
        plt.plot(data[my_members, 0], data[my_members, 1], col + '.', markeredgecolor='#fff', markeredgewidth=0.7, markersize=8)
        plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
             markeredgecolor='#fff', markersize=10)
    
    plt.title('Mean Shift - Estimated number of clusters: %d' % n_clusters)
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_facecolor('#eff2f7')

    plt.grid(color='#fff')
    plt.show()
    st.pyplot(fig)


def plotting_dbscan(dbscan, labels, n_clusters, data, x_var="", y_var=""):
    """
    Ploting of the "Happiness and Alcohol Consumption" dataset, based on
    https://www.geeksforgeeks.org/implementing-dbscan-algorithm-using-sklearn/
    """

    fig = plt.figure(figsize=(9, 9))
    plt.clf()

    # farben ändern...
    colors = cycle('bcmrgykbgrcmykbgrcmykbgrcmyk')
    for k, col in zip(range(-1, n_clusters), colors):
        my_members = labels == k
        plt.plot(data[my_members, 0], data[my_members, 1], col + '.', markeredgecolor='#fff', markeredgewidth=0.7,
                 markersize=8)

    plt.title(f'DBSCAN - Estimated number of clusters: {n_clusters}')
    # according to the colour vector defined
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_facecolor('#eff2f7')
    plt.grid(color='#fff')

    plt.show()

    st.pyplot(fig)
    return


def plotting_kmeans(kmeans, labels, n_clusters, data):
    fig = plt.figure(figsize=(9, 9))
    plt.clf()

    # farben ändern...
    colors = cycle('bcmrgykbgrcmykbgrcmykbgrcmyk')
    for k, col in zip(range(-1, n_clusters), colors):
        my_members = labels == k
        plt.plot(data[my_members, 0], data[my_members, 1], col + '.', markeredgecolor='#fff', markeredgewidth=0.7,
                 markersize=8)
    
    plt.title(f'k-Means with {n_clusters}')
    # according to the colour vector defined
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_facecolor('#eff2f7')
    plt.grid(color='#fff')
    plt.show()
    st.pyplot(fig)


def plotting_ahc(ahc_algo, labels, n_clusters, data):
    fig = plt.figure(figsize=(9, 9))
    plt.clf()

    # farben ändern...
    colors = cycle('bcmrgykbgrcmykbgrcmykbgrcmyk')
    for k, col in zip(range(-1, n_clusters), colors):
        my_members = labels == k
        plt.plot(data[my_members, 0], data[my_members, 1], col + '.', markeredgecolor='#fff', markeredgewidth=0.7,
                 markersize=8)

    plt.title(f'ahc - currently showing clusters: {n_clusters}')
    # according to the colour vector defined
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_facecolor('#eff2f7')
    plt.grid(color='#fff')
    plt.show()
    st.pyplot(fig)


def plot_dendrogram(model, **kwargs):
    """
    creates linkage matrix and then plots the dendrogram
    https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html
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

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    dendrogram(linkage_matrix, **kwargs)


def show_estimated_clusters_ahc(model, clusters):
    # dendrogram
    fig = plt.figure(figsize=(15, 5))
    plt.title('Hierarchical Clustering Dendrogram')
    plot_dendrogram(model, truncate_mode='lastp', p=clusters)
    plt.xlabel("Number of points in node.")
    plt.ylabel("Distances between new clusters.")
    st.pyplot(fig)
