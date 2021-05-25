import numpy as np

import dataset_tranformations

from sklearn.cluster import DBSCAN


def density_based_spatial_clustering_of_applications_with_noise(dataset):
    """
    Performs density-based clustering of applications with noise on datasets transformed as we as a group we agreed
    upon. This code is based upon
    https://www.geeksforgeeks.org/implementing-dbscan-algorithm-using-sklearn/
    """

    db = DBSCAN(eps=0.3, min_samples=5).fit(dataset)
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
        "happiness and alcohol": dataset_tranformations.happiness_alcohol_consumption
    }

    x, x_principal = datasets[dataset](pca_bool=pca_bool)
    db, labels = algorithms[algorithm](x_principal)
    dataset_tranformations.plotting_happiness_and_alcohol(
        x, labels, "HappinessScore", "HDI"
    )
    return 0


if __name__ == "__main__":
    main()
