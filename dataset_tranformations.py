import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import preprocessing

import pandas as pd
from sklearn.preprocessing import normalize

import streamlit as st


def happiness_alcohol_consumption(file_path="./datasets/HappinessAlcoholConsumption.csv", pca_bool=True):
    """
    Transforms the "Happiness and Alcohol Consumption" dataset,
    https://www.kaggle.com/marcospessotto/happiness-and-alcohol-consumption
    The reduction of dimensionality is based on the work in the articles
    https://www.geeksforgeeks.org/implementing-dbscan-algorithm-using-sklearn/
    https://365datascience.com/tutorials/python-tutorials/pca-k-means/
    """

    x = pd.read_csv(file_path)
    x = x.drop(["Country", "Region", "Hemisphere"], axis=1)

    # Scaling the data to bring all the attributes to a comparable level
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    # Normalizing the data so that
    # the data approximately follows a Gaussian distribution
    x_normalized = normalize(x_scaled)

    if pca_bool:
        components = get_components(x_scaled)
        pca = PCA(n_components=components)
        x_principal = pca.fit_transform(x_normalized)
        x_principal = pd.DataFrame(x_principal)
        x_principal.columns = ["1", "2", "3"]
    else:
        x_principal = x_normalized

    return x_principal, None


def fixseeds(file_path="./datasets/seeds_dataset.txt", pca_bool=True):
    """
    Returns the features and labels of the seeds dataset

    @param pca_bool: uses pca und features

    @return: features and labels of the seeds dataset
    """
    seeds = pd.read_csv(file_path, header=None, sep='\t')
    seeds[[7]] = seeds[[7]].add(-1)
    x = seeds[[0, 1, 2, 3, 4, 5, 6]]
    y = seeds.loc[:, 7]
    scaler = StandardScaler()
    segmentation_std = scaler.fit_transform(x)
    if pca_bool:
        components = get_components(segmentation_std)
        pca = PCA(n_components=components)
        pca.fit(segmentation_std)
        scores_pca = pca.transform(segmentation_std)
        x = scores_pca
    else:
        x = segmentation_std
    return x, y


def hcvdataset(file_path="./datasets/hcvdat0.csv", pca_bool=True):
    """
    Returns the features, or the features with pca, and labels of the hcv dataset as a tuple of a np.array.
    Please keep in mind that this dataset is unlabeled !!!

    @param file_path: please specify where the hcv dataset is located on your device
    @param use_pca: if pca is desired
    """

    X = pd.read_csv(file_path, index_col=0)
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

    if pca_bool:
        # use pca
        components = get_components(segmentation_std)
        pca = PCA(n_components=components)
        pca.fit(segmentation_std)
        pca.transform(segmentation_std)
        scores_pca = pca.transform(segmentation_std)
        return scores_pca, None
    else:
        return segmentation_std, None


def liver_disorders(file_path="./datasets/liver_disorders.data", pca_bool=True):
    """
    Returns the liver disorders dataset obained from https://archive.ics.uci.edu/ml/datasets/liver+disorders after possibly having reduced its dimensionality by using PCA.
    Based on:
    https://365datascience.com/tutorials/python-tutorials/pca-k-means/

    @param pca_bool: boolean value to decide whether sklearn's PCA should be applied or not
    @return: processed liver disorders dataset according to pca_bool
    """

    liver_disorders = pd.read_csv(file_path, sep=',', names=[
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
        # kriege normalized atm nur ein cluster :(
        # data_scaled = preprocessing.normalize(data_scaled)

        components = get_components(data_scaled)
        pca = PCA(n_components=components)
        pca.fit(data_scaled)
        scores_pca = pca.transform(data_scaled)

        return scores_pca, None

    else:
        return liver_disorders.values, None


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


def plotting_happiness_and_alcohol(dataset, labels, x_var="", y_var=""):
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
