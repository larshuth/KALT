#!/usr/bin/env python
# coding: utf-8

# In[175]:


import numpy as np
import pandas as pd

import seaborn as sns
sns.set()

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# In[192]:


def hcvdataset(file_path="hcvdat0.csv", use_pca=True, n_components=9):
    """
    - loads data and uses pca if desired
    - returns np.array
    - keep in mind that this dataset is unlabeled !!!
    """
    
    X = pd.read_csv(file_path, index_col = 0) 
    # one-hot encoding for row Category and Sex
    X_tmp = pd.get_dummies(X[['Category', 'Sex']])
    X = X.drop(columns=['Category', 'Sex']).merge(X_tmp, left_index=True, right_index=True)
    # since we have non-existent values we fill them with 0s
    X = X.fillna(0)
    
    """still unsure, if standardscaler should be performed generally or not"""
    # Standardize features by removing the mean and scaling to unit variance
    # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
    scaler = StandardScaler()
    segmentation_std = scaler.fit_transform(X)
    segmentation_std.shape
    
    if use_pca:
        # use pca
        pca = PCA(n_components)
        pca.fit(segmentation_std)
        pca.transform(segmentation_std)
        scores_pca = pca.transform(segmentation_std)
        return scores_pca
    else:
        return segmentation_std


# In[190]:


# Preview the first 5 lines of the loaded data without pca
hcvdataset(use_pca=False)[:-5]


# In[178]:


# Preview the first 5 lines of the loaded data with pca
hcvdataset()[:5]


# In[179]:


def ahc_algo(data, show_dendrogram=True, show_scatter=True, n_clusters=4):
    """
    - fits the model while using allgomorative hierarchical clustering
    - n_clusters: if you want plot the scattered data use n_clusters to show n clusters
    """
    if show_dendrogram and show_scatter:
        # plot both a dendrogram and a scattered data with n clusters
        
        # for dendrogram
        model = AgglomerativeClustering(distance_threshold=0, n_clusters=None, linkage='ward')
        model = model.fit(data)
        # for scatter
        cluster = AgglomerativeClustering(n_clusters, affinity='euclidean', linkage='ward')
        cluster.fit_predict(data)
        
        # dendrogram
        plt.figure(figsize=(15, 5))
        plt.subplot(1,2,1)
        plt.title('Hierarchical Clustering Dendrogram')
        plot_dendrogram(model, truncate_mode='level', p=3)
        plt.xlabel("Number of points in node.")
    
        # scatter
        plt.subplot(1,2,2)
        plt.scatter(data[:,0],data[:,1], c=cluster.labels_, cmap='rainbow')
        plt.title("Scattered data with "+ str(n_clusters) + " clusters.")
        
        plt.tight_layout()
        
        return cluster.labels_
    elif show_dendrogram and not show_scatter:
        # plot a dendogram
        """
        setting distance_threshold=0 ensures we compute the full tree 
        and getting the individual distances_ for plotting the dendogram
        """
        model = AgglomerativeClustering(distance_threshold=0, n_clusters=None, linkage='ward')
        model = model.fit(data)

        # plot a dendrogram for visualization and describe its axes
        plt.title('Hierarchical Clustering Dendrogram')
        # plot the top three levels of the dendrogram
        plot_dendrogram(model, truncate_mode='level', p=3)
        plt.xlabel("Number of points in node.")
        plt.show()
        
    elif show_scatter and not show_dendrogram:
        # plot the scattered data showing n clusters
        cluster = AgglomerativeClustering(n_clusters, affinity='euclidean', linkage='ward')
        cluster.fit_predict(data)
        
        plt.title("Scattered data with "+ str(n_clusters) + " clusters.")
        labels = cluster.labels_
        plt.scatter(data[:,0],data[:,1], c=labels, cmap='rainbow')
        
        return labels


# In[180]:


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


# In[199]:


def pca_distribution(data):
    """
    - plots the distribution of your given data to find out how many components would be good use (around 80%)
    - data: already standardized
    """
    pca = PCA()
    pca.fit(data)
    
    plt.figure(figsize = (10,8))
    plt.plot(range(1,19), pca.explained_variance_ratio_.cumsum(), marker = 'o', linestyle = '--')
    plt.title('Explained Variance by Components')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')


# In[198]:


pca_distribution(hcvdataset(use_pca=False))


# In[193]:


X = hcvdataset(use_pca=True)
temp = ahc_algo(X)
metrics.davies_bouldin_score(X, temp)


# In[194]:


X = hcvdataset(use_pca=False)
temp = ahc_algo(X, False, n_clusters=4)
metrics.davies_bouldin_score(X, temp)


# In[ ]:




