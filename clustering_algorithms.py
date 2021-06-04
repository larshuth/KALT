import numpy as np
import streamlit as st
import dataset_tranformations
import plot_clustering

from sklearn.cluster import DBSCAN, MeanShift, KMeans, estimate_bandwidth


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


def mean_shift(data, bandwidth):
    """
    Performs Mean Shift clustering on given dataset.
    Based on: 
    https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MeanShift.html

    @param data: processed dataset (e.g. liver disorders dataset)
    @param bandwidth: distance of kernel function or size of "window", either automatically estimated or given by user
    @return mean shift instance, index of cluster each data point belongs to, number of clusters
    """

    #bandwidth = estimate_bandwidth(data)
    mean_shift = MeanShift(bandwidth=bandwidth)

    mean_shift.fit(data)
    labels = mean_shift.labels_
    cluster_centers = mean_shift.cluster_centers_
    labels_unique = np.unique(labels)
    n_clusters = len(labels_unique)

    return mean_shift, labels, n_clusters


def main(algorithm="dbscan", dataset="happiness and alcohol", pca_bool=True):
    print("pick a god and pray")

    algorithms = {
        "dbscan": density_based_spatial_clustering_of_applications_with_noise,
        "mean shift": mean_shift
    }
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

    dataset_choice = st.selectbox("Which dataset?", tuple(ds for ds in datasets))
    algorithm_choice = st.selectbox("Which algorithm?", tuple(alg for alg in algorithms))


    pca_string = st.selectbox("Use PCA for cluster calculation?", ("Yes", "No"))
    if pca_string == "Yes":
        pca_bool = True
    else:
        pca_bool = False

    # for dbscan
    epsilon = st.slider(
        "Epsilon Neighborhood", min_value=0.05, max_value=1.0, value=0.3, step=0.05
    )
    size = st.slider(
        "Min Neighborhood Size", min_value=1.0, max_value=15.0, value=5.0, step=1.0
    )

    # for mean shift
    bandwidth = st.slider(
       'Bandwidth', min_value=1.0, max_value=4.0, value=float(round(estimate_bandwidth(datasets[dataset_choice](pca_bool=pca_bool)[0]), 2))
    )

    
    if algorithm_choice == "dbscan":
        x, y = datasets[dataset_choice](pca_bool=pca_bool)
        print(x)
        db, labels = algorithms[algorithm](
            x, epsilon_neighborhood=epsilon, cluster_neighborhood=size
        )
        dataset_tranformations.plotting_happiness_and_alcohol(
            x, labels, x.columns[0], x.columns[1]
        )
    elif algorithm_choice == "mean shift":
        mean_shift_data = datasets[dataset_choice](pca_bool=pca_bool)[0]
        plot_clustering.plotting_mean_shift(
            mean_shift(mean_shift_data, bandwidth)[0], 
            mean_shift(mean_shift_data, bandwidth)[1], 
            mean_shift(mean_shift_data, bandwidth)[2], 
            mean_shift_data
        )

    return 0


if __name__ == "__main__":
    main()
