import email.contentmanager

import streamlit as st

import dataset_tranformations
import plot_clustering
import clustering_algorithms

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

from sklearn.cluster import (
    estimate_bandwidth,
)


def single_algo(
    db_scan_string,
    algorithms,
    plotting_algorithms,
    datasets,
    dataset_choice,
    dataset_start_epsilons,
    dataset_max_epsilons,
):
    """
    Displays the single algorithm page. Each algorithm can individually be applied on each data set and be plotted. 
    Users can choose the data set, algorithm and parameters specific to their chosen algorithm.

    @param db_scan_string: "DBSCAN"
    @param algorithms: Dictionary containing the clustering algorithms of the clustering_algorithms module
    @param plotting_algorithms: Dictionary containing the plotting functions of each clustering algorithm of the plot_clustering module
    @param datasets: Dictionary containing the data sets of the data_transformation module
    @param dataset_choice: String containing the user-chosen data set
    @param dataset_start_epsilons: Dictionary containing the start epsilon values for each data set for the DBSCAN algorithm
    @param dataset_max_epsilons: Dictionary containing the maximum eplison values for each data set for the DBSCAN algorithm
    @return 
    """

    st.write("### Compare the parameters of a single algorithm.")

    algorithm_choice = st.selectbox(
        "Choose algorithm:", tuple(alg for alg in algorithms)
    )

    pca_string = st.selectbox("Use PCA for cluster calculation?", ("Yes", "No"))
    if pca_string == "Yes":
        pca_bool = True
    else:
        pca_bool = False

    if algorithm_choice == db_scan_string:
        max_epsilon = dataset_max_epsilons[dataset_choice]
        start_epsilon = dataset_start_epsilons[dataset_choice]

        epsilon = st.slider(
            "Epsilon Neighborhood",
            min_value=0.05,
            max_value=max_epsilon,
            value=start_epsilon,
            step=0.05,
        )
        clustering_neighborhood = st.slider(
            "Min Neighborhood Size", min_value=2.0, max_value=15.0, value=5.0, step=1.0
        )
        algo_parameters = {
            "epsilon_neighborhood": epsilon,
            "clustering_neighborhood": clustering_neighborhood,
        }
    elif algorithm_choice == "Mean Shift":
        # for mean shift
        if dataset_choice == "Happiness and alcohol":
            estimated_bandwidth = round(
                estimate_bandwidth(datasets[dataset_choice](pca_bool=True)[0], quantile=0.15),
                2
            )
        else:
            estimated_bandwidth = round(
                estimate_bandwidth(datasets[dataset_choice](pca_bool=pca_bool)[0]),
                2
            )

        bandwidth = st.slider(
            "Bandwidth",
            min_value=0.5,
            max_value=4.0,
            value=float(estimated_bandwidth)
        )

        algo_parameters = {
            "bandwidth": bandwidth,
        }
    elif algorithm_choice == "k-Means":
        # k-Means
        n_clusters = st.slider(
            "Clusters",
            min_value=1,
            max_value=8,
            step=1,
            value=clustering_algorithms.optimal_cluster_count(datasets[dataset_choice](pca_bool=pca_bool)[0]),
        )
        algo_parameters = {"clusters": n_clusters,
                           "random_state": 42
                          }
    elif algorithm_choice == "Hierarchical Agglomerative Clustering":
        # for hierarchical clustering
        print("I selected hac_algo.")

        linkage = st.selectbox(
            "Choose the linkage", ("ward", "average", "complete/maximum", "single")
        )

        df = pd.DataFrame(
            {
                "ward": ["minimizes the variance of the clusters being merged"],
                "average": ["minimizes the variance of the clusters being merged"],
                "complete/maximum": [
                    "linkage uses the maximum distances between all observations of the two sets"
                ],
                "single": [
                    "uses the minimum of the distances between all observations of the two sets"
                ],
            }
        )
        df.index = [""] * len(df)
        st.write(df[linkage])

        if linkage == "complete/maximum":
            linkage = "complete"

        show_cluster = st.slider(
            "Show n clusters", min_value=2, max_value=8, value=4, step=1
        )
        x, y = datasets[dataset_choice](pca_bool=pca_bool)
        clustering_algorithms.estimate_clusters_hac(x, linkage, show_cluster)

        algo_parameters = {"link": linkage, "n_clusters": show_cluster}

    x, y = datasets[dataset_choice](pca_bool=pca_bool)

    fitted_data, labels, n_clusters = algorithms[algorithm_choice](x, algo_parameters)

    fig = plt.figure()
    plotting_algorithms[algorithm_choice](fitted_data, labels, n_clusters, x)
    st.pyplot(fig)

    clustering_algorithms.evaluation(x, labels, False, y)
    return


def all_algo(
    db_scan_string,
    algorithms,
    plotting_algorithms,
    datasets,
    dataset_choice,
    dataset_start_epsilons,
    secret_lars_lever
):
    """
    Displays the all algorithms page. Each algorithm will be applied to each data set with pre-set parameters and be plotted next to each other. 
    Users can choose the data set to apply the algorithms on.

    @param db_scan_string: "DBSCAN"
    @param algorithms: Dictionary containing the clustering algorithms of the clustering_algorithms module
    @param plotting_algorithms: Dictionary containing the plotting functions of each clustering algorithm of the plot_clustering module
    @param datasets: Dictionary containing the data sets of the data_transformation module
    @param dataset_choice: String containing the user-chosen data set
    @param dataset_start_epsilons: Dictionary containing the start epsilon values for each data set for the DBSCAN algorithm
    @param secret_lars_lever: Bool value indicating a modified approach for the evaluation
    @return 
    """

    st.write("### Compare all four algorithms.")
    x, y = datasets[dataset_choice](pca_bool=True)
    fig = plt.figure()

    external_validation = False
    if dataset_choice == 'Seeds':
        evaluation_type = st.selectbox("Type of cluster evaluation:", ("Internal", "External"))
        if evaluation_type == "Internal":
            external_validation = False
        else:
            external_validation = True

    if external_validation:
        results = pd.DataFrame({'purity': list(), 'rand': list(), 'jaccard': list()})
    else:
        results = pd.DataFrame({'davies': list(), 'silhouette': list(), 'dunn': list()})

    for algo, i in zip(algorithms, range(1, len(algorithms) + 1)):
        if algo == "Mean Shift":
            if dataset_choice == "Happiness and alcohol":
                estimated_bandwidth = estimate_bandwidth(datasets[dataset_choice](pca_bool=True)[0], quantile=0.15)
            else:
                estimated_bandwidth = estimate_bandwidth(datasets[dataset_choice](pca_bool=True)[0])

            algo_parameters = {
                "bandwidth": estimated_bandwidth
            }

        elif algo == "Hierarchical Agglomerative Clustering":
            algo_parameters = {"link": "ward", "n_clusters": 4}
        elif algo == "DBSCAN":
            algo_parameters = {
                "epsilon_neighborhood": dataset_start_epsilons[dataset_choice],
                "clustering_neighborhood": 5,
            }
        elif algo == "k-Means":
            algo_parameters = {
                "clusters": clustering_algorithms.optimal_cluster_count(
                    datasets[dataset_choice](pca_bool=True)[0]
                ),
                "random_state": 42
            }
        fitted_data, labels, n_clusters = algorithms[algo](x, algo_parameters)
        plt.subplot(2, 2, i)
        plotting_algorithms[algo](fitted_data, labels, n_clusters, x)

        if algo == "DBSCAN":
            scores = pd.DataFrame(clustering_algorithms.evaluation(x, labels, external_validation, y, secret_lars_lever))
        else:
            scores = pd.DataFrame(clustering_algorithms.evaluation(x, labels, external_validation, y, False))
        results = pd.concat([results, scores], ignore_index=True)
    
    plt.tight_layout()
    st.pyplot(fig)

    # plotting the evaluation of our results
    results = results.rename(
        index={0: db_scan_string, 1: "Mean Shift", 2: "k-Means", 3: "Hierarchical Agglomerative Clustering"})

    plot_clustering.evaluation_plot(results)

    return


def main():
    """
    Main function providing headline, sidebar and properties used for clustering and plotting. If radiobutton "All Algorithms" is selected, 
    the corresponding page will be displayed by calling all_algo(). Similarly, single_algo() will be called by choosing the "Single Algorithm" page.

    @return 0
    """

    # Build a sidebar for the web frontend from which to choose whether to execute all algorithms or a single slgorithm
    sid = st.sidebar
    page = sid.radio("Choose page:", ("All Algorithms", "Single Algorithm"))

    sid.markdown("---")

    sid.header("About")
    t1 = "The web application practically examines the algorithms DBSCAN, Mean Shift, k-Means and Hierarchical Agglomerative Clustering further by giving the option to apply them on different data sets. \
        Therefore, the algorithms are compared and evaluated under the 'All Algorithms' page though can also be examined individually on the 'Single Algorithm' page on which users are eligible to play around \
        with their specific parameters."
    sid.markdown(t1, unsafe_allow_html=True)

    sid.markdown("---")

    sid.header("Creators")
    sid.markdown(
        """This project is created by KALT:
        \n **K**atharina Dahmann
        \n **A**licia Wirth
        \n **L**ars Huth
        \n **T**olga Tel"""
    )

    sid.markdown("---")
    secret_lars_lever = sid.checkbox('', value=False, key=None, help=None)

    db_scan_string = "DBSCAN"

    # our algorithms for clustering, algorithms for plotting, datasets
    algorithms = {
        db_scan_string: clustering_algorithms.density_based_spatial_clustering_of_applications_with_noise,
        "Mean Shift": clustering_algorithms.mean_shift,
        "k-Means": clustering_algorithms.k_Means,
        "Hierarchical Agglomerative Clustering": clustering_algorithms.hac_algo,
    }

    plotting_algorithms = {
        db_scan_string: plot_clustering.plotting_dbscan,
        "Mean Shift": plot_clustering.plotting_mean_shift,
        "k-Means": plot_clustering.plotting_kmeans,
        "Hierarchical Agglomerative Clustering": plot_clustering.plotting_hac,
    }

    datasets = {
        "Happiness and alcohol": dataset_tranformations.happiness_alcohol_consumption,
        "Seeds": dataset_tranformations.fixseeds,
        "HCV dataset": dataset_tranformations.hcvdataset,
        "Liver disorders": dataset_tranformations.liver_disorders,
    }

    # preset parameters for DBSCAN
    dataset_max_epsilons = {
        "Happiness and alcohol": 0.8,
        "Seeds": 0.8,
        "HCV dataset": 5.0,
        "Liver disorders": 2.3,
    }

    dataset_start_epsilons = {
        "Happiness and alcohol": 0.6,
        "Seeds": 0.5,
        "HCV dataset": 2.5,
        "Liver disorders": 0.8,
    }

    st.write(
        """
    # Principles of Data Science.
    Summer 2021 - Prof. Wiese
    
    ## Assignment for comparing clustering algorithm. 
    """
    )

    dataset_choice = st.selectbox("Choose data set:", tuple(ds for ds in datasets))

    # page display the chosen option
    if page == "All Algorithms":
        all_algo(
            db_scan_string,
            algorithms,
            plotting_algorithms,
            datasets,
            dataset_choice,
            dataset_start_epsilons,
            secret_lars_lever
        )
    else:
        single_algo(
            db_scan_string,
            algorithms,
            plotting_algorithms,
            datasets,
            dataset_choice,
            dataset_start_epsilons,
            dataset_max_epsilons,
        )

    return 0


if __name__ == "__main__":
    main()
