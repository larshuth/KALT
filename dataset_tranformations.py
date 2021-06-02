import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import pandas as pd
from sklearn.preprocessing import normalize

import streamlit as st


def happiness_alcohol_consumption(location="/home/lars/Downloads/", pca_bool=True):
    """
    Transforms the "Happiness and Alcohol Consumption" dataset,
    https://www.kaggle.com/marcospessotto/happiness-and-alcohol-consumption
    The reduction of dimensionality is based on the work in the articles
    https://www.geeksforgeeks.org/implementing-dbscan-algorithm-using-sklearn/
    https://365datascience.com/tutorials/python-tutorials/pca-k-means/
    """

    dataset = location + "HappinessAlcoholConsumption.csv"
    x = pd.read_csv(dataset)
    x = x.drop(["Country", "Region", "Hemisphere"], axis=1)

    # Scaling the data to bring all the attributes to a comparable level
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    # Normalizing the data so that
    # the data approximately follows a Gaussian distribution
    x_normalized = normalize(x_scaled)

    if pca_bool:
        pca = PCA(n_components=3)
        x_principal = pca.fit_transform(x_normalized)
        x_principal = pd.DataFrame(x_principal)
        x_principal.columns = ["P1", "P2", "P3"]
    else:
        x_principal = x_normalized

    return x, x_principal


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
