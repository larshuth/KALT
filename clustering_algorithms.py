import numpy as np

import dataset_tranformations

from sklearn.cluster import DBSCAN

import streamlit as st


def density_based_spatial_clustering_of_applications_with_noise(
    dataset_x, dataset_y=None, epsilon_neighborhood=0.3, cluster_neighborhood=5
):
    """
    Performs density-based clustering of applications with noise on datasets transformed as we as a group we agreed
    upon. This code is based upon
    https://www.geeksforgeeks.org/implementing-dbscan-algorithm-using-sklearn/

    @param dataset_x: features of the dataset as an array (required)
    @param dataset_y: labels of the dataset as an array (not required, default = none)
    @param epsilon_neighborhood: max distance to still be considered a neighbor (not required, default = 0.3)
    @param cluster_neighborhood: number of points in a neighborhood to make a cluster (not required, default = 5)
    """

    db = DBSCAN(eps=epsilon_neighborhood, min_samples=cluster_neighborhood).fit(
        dataset_x
    )
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True

    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)
    return db, labels


def main(algorithm="dbscan", dataset="happiness and alcohol", pca_bool=True):
    print("pick a god and pray")

    algorithms = {"dbscan": density_based_spatial_clustering_of_applications_with_noise}
    datasets = {
        "happiness and alcohol": dataset_tranformations.happiness_alcohol_consumption,
        "seeds": dataset_tranformations.fixseeds,
        "HCV Impfungen, Erkrankungen und mehr": dataset_tranformations.hcvdataset,
        "liver disorder": dataset_tranformations.liver_disorders
    }

    st.write(
        """
    # Sata Dience.
    
    Correlation?
    Clustering?
    """
    )

    dataset = st.selectbox("Use PCA for cluster calculation?", ("Yes", "No"))
    pca_string = st.selectbox("Use PCA for cluster calculation?", ("Yes", "No"))
    if pca_string == "Yes":
        pca_bool = True
    else:
        pca_bool = False

    epsilon = st.slider(
        "Epsilon Neighborhood", min_value=0.05, max_value=1.0, value=0.3, step=0.05
    )
    size = st.slider(
        "Min Neighborhood Size", min_value=1.0, max_value=15.0, value=5.0, step=1.0
    )

    x, x_principal = datasets[dataset](location="./datasets/", pca_bool=pca_bool)
    db, labels = algorithms[algorithm](
        x_principal, epsilon_neighborhood=epsilon, cluster_neighborhood=size
    )
    dataset_tranformations.plotting_happiness_and_alcohol(
        x, labels, "HappinessScore", "HDI"
    )
    return 0


if __name__ == "__main__":
    main()
