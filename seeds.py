import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def fixseeds():
    """
    Returns the features and labels of the seeds dataset
    
    @return: features and labels of the seeds dataset
    """
    seeds = pd.read_csv('datasets/seeds_dataset.txt', header = None, sep='\t')
    seeds[[7]] = seeds[[7]].add(-1)
    x = seeds[[0, 1, 2, 3, 4, 5, 6]]
    y = seeds.loc[:,7]
    return x, y


def pca_for_seeds(x):
    """
    Returns the features of the seeds dataset after using the pca algorithm
    
    @param x: original features of the seeds dataset
    @return: features of the seeds dataset after using the pca algorithm
    """
    scaler = StandardScaler()
    segmentation_std = scaler.fit_transform(x)
    pca = PCA(n_components = 3)
    pca.fit(segmentation_std)
    scores_pca = pca.transform(segmentation_std)
    return scores_pca
