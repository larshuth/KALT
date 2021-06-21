import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize, LabelEncoder, OneHotEncoder

def happiness_alcohol_consumption(
    file_path="./datasets/HappinessAlcoholConsumption.csv", pca_bool=True
):
    """
    Transforms the "Happiness and Alcohol Consumption" dataset,
    https://www.kaggle.com/marcospessotto/happiness-and-alcohol-consumption
    The reduction of dimensionality is based on the work in the articles
    https://www.geeksforgeeks.org/implementing-dbscan-algorithm-using-sklearn/
    https://365datascience.com/tutorials/python-tutorials/pca-k-means/
    """

    x = pd.read_csv(file_path)
    x.set_index("Country", inplace=True)

    # cleaning typo
    x.loc[x['Hemisphere'] == 'noth', 'Hemisphere'] = 'north'

    # one hot encoding out categorical variables
    categoricals = ['Region', 'Hemisphere']
    for category in categoricals:
        x = pd.concat([x, pd.get_dummies(x[category], prefix=category, drop_first=True)], axis=1)
    x = x.drop(categoricals, axis=1)

    # fixing the problem with , . loading as float in the GDP attribute
    x['GDP_PerCapita'][x['GDP_PerCapita'] < 101] = x['GDP_PerCapita'][x['GDP_PerCapita'] < 101] * 1000

    # Scaling the data to bring all the attributes to a comparable level
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    x_normalized = normalize(x_scaled)

    if pca_bool:
        components = get_components(x_normalized)
        pca = PCA(n_components=components)
        pca.fit_transform(x_normalized)
        x_principal = pca.transform(x_normalized)
    else:
        x_principal = x_scaled

    return x_principal, None


def fixseeds(file_path="./datasets/seeds_dataset.txt", pca_bool=True):
    """
    Returns the features and labels of the seeds dataset

    @param pca_bool: uses pca und features

    @return: features and labels of the seeds dataset
    """
    seeds = pd.read_csv(file_path, header=None, sep="\t")
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
    Returns the HCV dataset obtained from https://archive.ics.uci.edu/ml/datasets/HCV+data . 
    If the variable pca_bool is true, then a dimensionality reduction with PCA was used, else the data just got standerdized.
    Based on:
    https://365datascience.com/tutorials/python-tutorials/pca-k-means/

    @param file_path: string for location of the HCV data set
    @param use_pca: boolean value to decide whether sklearn's PCA should be applied or not
    @return: processed HCV dataset according to pca_bool value, None as no labels exist
    """

    X = pd.read_csv(file_path, index_col=0)
    # one-hot encoding for row Category and Sex
    X_tmp = pd.get_dummies(X[["Category", "Sex"]])
    X = X.drop(columns=["Category", "Sex"]).merge(
        X_tmp, left_index=True, right_index=True
    )
    # since we have non-existent values we fill them with 0s
    X = X.fillna(X.mean())
    X = X.drop(columns=["Sex_f", "Sex_m", "Category_0s=suspect Blood Donor"])

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
    Returns the liver disorders dataset obtained from https://archive.ics.uci.edu/ml/datasets/liver+disorders after possibly having reduced its dimensionality by using PCA.
    Based on:
    https://365datascience.com/tutorials/python-tutorials/pca-k-means/

    @param file_path: path of liver disorders dataset
    @param pca_bool: boolean value to decide whether sklearn's PCA should be applied or not
    @return: processed liver disorders dataset according to pca_bool, None as no labels exist
    """

    liver_disorders = pd.read_csv(
        file_path,
        sep=",",
        names=[
            "Mean Corpuscular Volume",
            "Alkaline Phosphatase",
            "Alanine Aminotransferase",
            "Aspartate Aminotransferase",
            "Gamma-Glutamyl Transpeptidase",
            "Number of Half-Pint Equivalents of Alcoholic Beaverages Drunk per Day",
            "Selector"
        ],
    )

    # drop null values
    liver_disorders = liver_disorders.dropna()
    # drop duplicates
    liver_disorders = liver_disorders.drop_duplicates()
    # drop "Selector" column since it was only used to split the data into train/test sets
    liver_disorders = liver_disorders.drop("Selector", axis=1)

    # reduce dimensionality
    if pca_bool:
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(liver_disorders)

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
    for i in range(0, len(pca.explained_variance_ratio_.cumsum()) - 1):
        if pca.explained_variance_ratio_.cumsum()[i] >= 0.8:
            return i + 1
        else:
            pass
