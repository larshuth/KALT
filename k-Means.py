import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.cluster import KMeans
import seaborn as sns
sns.set()
from sklearn.metrics import davies_bouldin_score, jaccard_score
from seeds import fixseeds


def k_Means(dataset_x, dataset_y = None, clusters = 3, feature_1 = 0, feature_2 = 1):
    """
    Shows a 2D-plot of clustered points. If it's a labeled dataset it also shows the real clusters
    
    @param dataset_x: features of the dataset as an array (required)
    @param dataset_y: labels of the dataset as an array (not required, default = none)
    @param clusters: amount of clusters (not required, default = 3)
    @param feature_1: x_axis of the plot; enter the number of the column (not required, default = 0)
    @param feature_2: y_axis of the plot; enter the number of the column (not required, default = 1)
    """
    labeled = True
    try: 
        if dataset_y == None:
            labeled = False
    except:
        pass
    kmeans = KMeans(n_clusters=clusters, random_state=42)
    kmeans.fit(dataset_x)
    result = kmeans.labels_
    x_axis = dataset_x[:,feature_1]
    y_axis = dataset_x[:,feature_2]
    if labeled:
        fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True)
        plt.scatter(x_axis, y_axis, c = result, ax=ax1)
        plt.scatter(x_axis, y_axis, c = dataset_y, ax=ax2)
        st.pyplot(fig)
    else:
        fig, ax = plt.subplots() 
        ax = plt.scatter(x_axis, y_axis, c = result)
        st.pyplot(fig)


def k_Means_acc(dataset_x, dataset_y = None, clusters = None):
    """
    Returns the accuracy of the K-Means algorithm
    
    @param dataset_x: features of the dataset as an array (required)
    @param dataset_y: labels of the dataset as an array (not required, default = none)
    @param clusters: amount of clusters (not required, default = 3)
    @return: jaccard score if labeled and davies bouldin score if not labeled
    """
    labeled = True
    try: 
        if dataset_y == None:
            labeled = False
    except:
        pass
    kmeans = KMeans(n_clusters=clusters, random_state=42)
    kmeans.fit(dataset_x)
    result = kmeans.labels_
    if labeled:
        return jaccard_score(dataset_y, result, average='macro')
    else:
        return davies_bouldin_score(dataset_x, result)


st.write("""
    # Tolga
    ## Datascience project
    ### K-Means    
    """
    )    
    
features, labels = fixseeds()
clusters = st.slider('Amount of Clusters', min_value=1, max_value=7, value=int)
k_Means(dataset_x = features, dataset_y = None, clusters = clusters, feature_1 = 0, feature_2 = 1)
