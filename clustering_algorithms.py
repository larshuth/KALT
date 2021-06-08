import numpy as np
import streamlit as st
import dataset_tranformations
import plot_clustering

from sklearn.cluster import DBSCAN, MeanShift, KMeans, estimate_bandwidth


def density_based_spatial_clustering_of_applications_with_noise(
    dataset_x, dbscan_params, dataset_y=None
):
    """
    Performs density-based clustering of applications with noise on datasets transformed as we as a group we agreed
    upon. This code is based upon
    https://www.geeksforgeeks.org/implementing-dbscan-algorithm-using-sklearn/

    @param dataset_x: features of the dataset as an array (required)
    @param dataset_y: labels of the dataset as an array (not required, default = none)
    @param dbscan_params: epsilon neighborhood and cluster neighborhood as required for dbscan
    """
    print('yeehaw')

    db = DBSCAN(eps=dbscan_params['epsilon_neighborhood'], min_samples=dbscan_params['clustering_neighborhood']).fit(
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
    return db, labels, n_clusters_


def mean_shift(data, meanshift_params):
    """
    Performs Mean Shift clustering on given dataset.
    Based on: 
    https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MeanShift.html

    @param data: processed dataset (e.g. liver disorders dataset)
    @param bandwidth: distance of kernel function or size of "window", either automatically estimated or given by user
    @return mean shift instance, index of cluster each data point belongs to, number of clusters
    """

    #bandwidth = estimate_bandwidth(data)
    mean_shift = MeanShift(bandwidth=meanshift_params['bandwidth'])

    mean_shift.fit(data)
    labels = mean_shift.labels_
    cluster_centers = mean_shift.cluster_centers_
    labels_unique = np.unique(labels)
    n_clusters = len(labels_unique)

    return mean_shift, labels, n_clusters


def main(algorithm="dbscan", dataset="happiness and alcohol", pca_bool=True):
    print("pick a god and pray")

    db_scan_string = 'DBSCAN'

    algorithms = {
        db_scan_string: density_based_spatial_clustering_of_applications_with_noise,
        "mean shift": mean_shift
    }
    plotting_algorithms = {
        db_scan_string: plot_clustering.plotting_dbscan,
        "mean shift": plot_clustering.plotting_mean_shift
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
    
    Colleration?
    Culstering?
    """
    )

    dataset_choice = st.selectbox("Which dataset?", tuple(ds for ds in datasets))
    algorithm_choice = st.selectbox("Which algorithm?", tuple(alg for alg in algorithms))

    pca_string = st.selectbox("Use PCA for cluster calculation?", ("Yes", "No"))
    if pca_string == "Yes":
        pca_bool = True
    else:
        pca_bool = False
    
    if algorithm_choice == db_scan_string:
        epsilon = st.slider(
            "Epsilon Neighborhood", min_value=0.05, max_value=1.0, value=0.3, step=0.05
        )
        clustering_neighborhood = st.slider(
            "Min Neighborhood Size", min_value=1.0, max_value=15.0, value=5.0, step=1.0
        )
        algo_parameters = {
            'epsilon_neighborhood': epsilon,
            'clustering_neighborhood': clustering_neighborhood
        }
    elif algorithm_choice == "mean shift":
        # for mean shift
        bandwidth = st.slider(
            'Bandwidth', min_value=1.0, max_value=4.0,
            value=float(round(estimate_bandwidth(datasets[dataset_choice](pca_bool=pca_bool)[0]), 2))
        )
        algo_parameters = {
            'bandwidth': bandwidth,
        }

    x, y = datasets[dataset_choice](pca_bool=pca_bool)

    fitted_data, labels, n_clusters = algorithms[algorithm_choice](
        x, algo_parameters
    )

    plotting_algorithms[algorithm_choice](fitted_data, labels, n_clusters, x)

    return 0


if __name__ == "__main__":
    main()
