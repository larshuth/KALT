#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd

import seaborn as sns
sns.set()

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
from sklearn.decomposition import PCA


def ahc_algo(data, show_dendrogram=True, show_scatter=True, n_clusters=4):
    """
    Fits the model while using allgomorative hierarchical clustering. 
    Plots the result eaither by showing a dendogram, a scatter or both.
    
    @param data: the data to be used for ahc algorithm
    @show_dendogram: if you want to show the result through using a dandogram
    @show_scatter: if you want to show the results though scattering the datapoints
    @param n_clusters: if you want plot the scattered data use n_clusters to show n clusters
    """
    data_x, data_y = data
    if show_dendrogram and show_scatter:
        # plot both a dendrogram and a scattered data with n clusters
        
        # for dendrogram
        model = AgglomerativeClustering(distance_threshold=0, n_clusters=None, linkage='ward')
        model = model.fit(data_x)
        # for scatter
        cluster = AgglomerativeClustering(n_clusters, affinity='euclidean', linkage='ward')
        cluster.fit_predict(data_x)
        
        # dendrogram
        fig = plt.figure(figsize=(15, 5))
        plt.subplot(1,2,1)
        plt.title('Hierarchical Clustering Dendrogram')
        plot_dendrogram(model, truncate_mode='level', p=3)
        plt.xlabel("Number of points in node.")
    
        # scatter
        plt.subplot(1,2,2)
        plt.scatter(data_x[:,0],data_x[:,1], c=cluster.labels_, cmap='rainbow')
        plt.title("Scattered data with "+ str(n_clusters) + " clusters.")
        
        plt.tight_layout()

        return fig
    elif show_dendrogram and not show_scatter:
        # plot a dendogram
        """
        setting distance_threshold=0 ensures we compute the full tree 
        and getting the individual distances_ for plotting the dendogram
        """
        model = AgglomerativeClustering(distance_threshold=0, n_clusters=None, linkage='ward')
        model = model.fit(data_x)

        # plot a dendrogram for visualization and describe its axes
        plt.title('Hierarchical Clustering Dendrogram')
        # plot the top three levels of the dendrogram
        plot_dendrogram(model, truncate_mode='level', p=3)
        plt.xlabel("Number of points in node.")
        plt.show()
        
    elif show_scatter and not show_dendrogram:
        # plot the scattered data showing n clusters
        cluster = AgglomerativeClustering(n_clusters, affinity='euclidean', linkage='ward')
        cluster.fit_predict(data_x)
        
        plt.title("Scattered data with "+ str(n_clusters) + " clusters.")
        labels = cluster.labels_
        plt.scatter(data_x[:,0],data_x[:,1], c=labels, cmap='rainbow')
        
        # return labels

# plot calculation in form of a dendrogram
# https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html
def plot_dendrogram(model, **kwargs):
    """
    creates linkage matrix and then plots the dendrogram
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


def pca_distribution(data):
    """
    Plots the distribution of your given data to find out how many components would be good use (around 80%)
    @param data: already standardized
    """
    pca = PCA()
    pca.fit(data)
    
    fig = plt.figure(figsize=(10, 8))
    plt.plot(range(1, 19), pca.explained_variance_ratio_.cumsum(), marker='o', linestyle='--')
    plt.title('Explained Variance by Components')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')

    return fig, 18


def ahc_algo_acc(dataset_x, dataset_y=None, n_clusters=4):
    """
    Fits the model while using allgomorative hierarchical clustering.
    Returns accuracy of labeled and unlabeled data.

    @param dataset_x: data without labels
    @param dataset_y: labels
    @ params n_clusters: if you want plot the scattered data use n_clusters to show n clusters
    """
    labeled = True
    try: 
        if dataset_y == None:
            labeled = False
    except:
        pass

    cluster = AgglomerativeClustering(n_clusters, affinity='euclidean', linkage='ward')
    cluster.fit_predict(dataset_x)
        
    labels = cluster.labels_
    
    if labeled:
        return metrics.jaccard_score(dataset_y, labels, average='macro'), "jaccard score"
    else:
        return metrics.davies_bouldin_score(dataset_x, labels), "davies blouldin score"

