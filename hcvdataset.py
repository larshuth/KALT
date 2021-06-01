#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def hcvdataset(file_path="hcvdat0.csv", use_pca=True, n_components=9):
    """
    Returns the features, or the features with pca, and labels of the hcv dataset as a tuple of a np.array.
    Please keep in mind that this dataset is unlabeled !!!
    
    @param file_path: please specify where the hcv dataset is located on your device
    @param use_pca: if pca is desired
    @param n_components: if you want to use pca, please select the amount components you want to use
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
        return scores_pca, None
    else:
        return segmentation_std, None