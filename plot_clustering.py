import matplotlib.pyplot as plt
import streamlit as st
from itertools import cycle
import numpy as np


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

