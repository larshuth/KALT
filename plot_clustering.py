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


def plotting_dbscan(dbscan, labels, n_clusters, dataset, x_var="", y_var=""):
    """
    Ploting of the "Happiness and Alcohol Consumption" dataset, based on
    https://www.geeksforgeeks.org/implementing-dbscan-algorithm-using-sklearn/
    """

    # Building the label to colour mapping
    colours = {}

    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

    colours[-1] = "k"

    # Building the colour vector for each data point
    cvec = [colors[label] for label in labels]

    # For the construction of the legend of the plot
    if not x_var and y_var:
        x_var = labels[0]
        y_var = labels[1]

    r = plt.scatter(dataset[x_var], dataset[y_var], color="r")
    g = plt.scatter(dataset[x_var], dataset[y_var], color="g")
    b = plt.scatter(dataset[x_var], dataset[y_var], color="b")
    k = plt.scatter(dataset[x_var], dataset[y_var], color="k")

    # according to the colour vector defined
    fig = plt.figure(figsize=(9, 9))
    plt.clf()
    ax = plt.gca()
    ax.set_facecolor('#eff2f7')
    plt.grid(color='#fff')

    plt.scatter(dataset[x_var], dataset[y_var], c=cvec)
    plt.show()

    st.pyplot(fig)
    return

