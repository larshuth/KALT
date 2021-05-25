import numpy as np

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import pandas as pd
from sklearn.preprocessing import normalize


def happiness_alcohol_consumption(location='/home/lars/Downloads/', pca_bool=True):
    dataset = location+'HappinessAlcoholConsumption.csv'
    x = pd.read_csv(dataset)
    x = x.drop(['Country', 'Region', 'Hemisphere'], axis=1)

    # Scaling the data to bring all the attributes to a comparable level
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    if pca_bool:
        # Normalizing the data so that
        # the data approximately follows a Gaussian distribution
        x_normalized = normalize(x_scaled)

        pca = PCA(n_components=3)
        x_principal = pca.fit_transform(x_normalized)
        x_principal = pd.DataFrame(x_principal)
        x_principal.columns = ['P1', 'P2', 'P3']
    else:
        x_principal = x

    return x, x_principal
