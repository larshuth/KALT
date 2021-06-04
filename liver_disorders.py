import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import dataset_tranformations as data

from itertools import cycle
from sklearn import preprocessing, metrics
from sklearn.cluster import MeanShift, KMeans, estimate_bandwidth
from sklearn.decomposition import PCA


def liver_disorders(pca_bool=True):
    """
    Returns the liver disorders dataset obained from https://archive.ics.uci.edu/ml/datasets/liver+disorders after possibly having reduced its dimensionality by using PCA.
    Based on: 
    https://365datascience.com/tutorials/python-tutorials/pca-k-means/

    @param pca_bool: boolean value to decide whether sklearn's PCA should be applied or not
    @return: processed liver disorders dataset according to pca_bool
    """

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

    # reduce dimensionality
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


def mean_shift(data, bandwidth):
    """
    Performs Mean Shift clustering on given dataset.
    Based on: 
    https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MeanShift.html

    @param data: processed dataset (e.g. liver disorders dataset)
    @param bandwidth: distance of kernel function or size of "window", either automatically estimated or given by user
    @return mean shift instance, index of cluster each data point belongs to, number of clusters
    """

    #bandwidth = estimate_bandwidth(data)
    mean_shift = MeanShift(bandwidth=bandwidth)

    mean_shift.fit(data)
    labels = mean_shift.labels_
    cluster_centers = mean_shift.cluster_centers_
    labels_unique = np.unique(labels)
    n_clusters = len(labels_unique)

    return mean_shift, labels, n_clusters


def plotting_mean_shift(mean_shift, labels, n_clusters, data):
    """
    Displays 2D-plot of mean shift clustering points. 
    Based on: 
    https://scikit-learn.org/stable/auto_examples/cluster/plot_mean_shift.html#sphx-glr-auto-examples-cluster-plot-mean-shift-py
    https://matplotlib.org/
    https://stackoverflow.com/questions/925024/how-can-i-remove-the-top-and-right-axis-in-matplotlib


    @param mean_shift: mean shift instance
    @labels: index of cluster each data point belongs to
    @n_clusters: number of clusters
    @data: processed data (e.g. liver_disorders)
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
    
    plt.title('Estimated number of clusters: %d' % n_clusters)
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_facecolor('#e6e6e6')

    plt.grid(color='#fff')
    plt.show()
    st.pyplot(fig)


def get_components(data):
    """
    Decides on the number of components to keep during sklearn's PCA by analyzing their variance ratio. The variance ratio should be equal or greater than 80%.
    Based on: 
    https://365datascience.com/tutorials/python-tutorials/pca-k-means/

    @param data: processed dataset (e.g. liver disorders dataset)
    @return: number of components to perserve
    """

    pca = PCA()
    pca.fit(data)
    for i in range(0, len(pca.explained_variance_ratio_.cumsum())-1):
        if pca.explained_variance_ratio_.cumsum()[i] >= 0.8:
            return i+1
        else:
            pass

dataset = data.liver_disorders()[0]
# slider 
# https://docs.streamlit.io/en/stable/api.html
bandwidth = st.slider('Bandwidth', min_value=1.0, max_value=4.0, value=float(round(estimate_bandwidth(dataset), 2)))

# plot clustering
#plotting_mean_shift(mean_shift(liver_disorders(), bandwidth)[0], mean_shift(liver_disorders(), bandwidth)[1], mean_shift(liver_disorders(), bandwidth)[2], liver_disorders())
plotting_mean_shift(mean_shift(dataset, bandwidth)[0], mean_shift(dataset, bandwidth)[1], mean_shift(dataset, bandwidth)[2], dataset)