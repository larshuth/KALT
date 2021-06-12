import streamlit as st

import dataset_tranformations
import plot_clustering
import clustering_algorithms

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import (
    estimate_bandwidth,
)
from sklearn.metrics import silhouette_score


def single_algo(
    db_scan_string,
    algorithms,
    plotting_algorithms,
    datasets,
    dataset_choice,
    dataset_start_epsilons,
    dataset_max_epsilons,
):

    st.write("### Compare the parameters of a single algorithm.")

    algorithm_choice = st.selectbox(
        "Which algorithm?", tuple(alg for alg in algorithms)
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
            "Min Neighborhood Size", min_value=1.0, max_value=15.0, value=5.0, step=1.0
        )
        algo_parameters = {
            "epsilon_neighborhood": epsilon,
            "clustering_neighborhood": clustering_neighborhood,
        }
    elif algorithm_choice == "Mean Shift":
        # for mean shift
        bandwidth = st.slider(
            "Bandwidth",
            min_value=1.0,
            max_value=4.0,
            value=float(
                round(
                    estimate_bandwidth(datasets[dataset_choice](pca_bool=pca_bool)[0]),
                    2,
                )
            ),
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
        algo_parameters = {"clusters": n_clusters}
    elif algorithm_choice == "Agglomerative Hierarchical Clustering":
        # for hierarchical clustering
        print("I selected ahc_algo.")

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
        clustering_algorithms.estimate_clusters_ahc(x, linkage, show_cluster)

        algo_parameters = {"link": linkage, "n_clusters": show_cluster}

    x, y = datasets[dataset_choice](pca_bool=pca_bool)

    fitted_data, labels, n_clusters = algorithms[algorithm_choice](x, algo_parameters)

    fig = plt.figure()
    plotting_algorithms[algorithm_choice](fitted_data, labels, n_clusters, x)
    st.pyplot(fig)

    clustering_algorithms.evaluation(x, labels, False, y)


def all_algo(
    db_scan_string,
    algorithms,
    plotting_algorithms,
    datasets,
    dataset_choice,
    dataset_start_epsilons,
):

    st.write("### Compare all four algorithms.")
    x, y = datasets[dataset_choice](pca_bool=True)
    fig = plt.figure()

    external_validation = False

    if not external_validation:
        results = pd.DataFrame({'davies': list(), 'silhouette': list(), 'dunn': list()})
    else:
        results = pd.DataFrame({'davies': list(), 'silhouette': list(), 'dunn': list()})

    for algo, i in zip(algorithms, range(1, len(algorithms) + 1)):
        if algo == "Mean Shift":
            algo_parameters = {
                "bandwidth": estimate_bandwidth(
                    datasets[dataset_choice](pca_bool=True)[0]
                )
            }
        elif algo == "Agglomerative Hierarchical Clustering":
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
                )
            }
        fitted_data, labels, n_clusters = algorithms[algo](x, algo_parameters)
        plt.subplot(2, 2, i)
        plotting_algorithms[algo](fitted_data, labels, n_clusters, x)

        scores = clustering_algorithms.evaluation(x, labels, external_validation, y)
        results = results.append(pd.DataFrame(scores), index=algo)

    plt.tight_layout()
    st.pyplot(fig)

    st.dataframe(results)


def main():
    print("pick a god and pray")

    sid = st.sidebar
    page = sid.radio("Choose Comparison", ("All Algorithms", "Single Algorithm"))

    sid.markdown("---")

    sid.header("About")
    t1 = "Here will be described what this page is to be used for."
    sid.markdown(t1, unsafe_allow_html=True)

    sid.markdown("---")

    sid.header("Creators")
    sid.markdown(
        """This is a KALT project. The project members are:
        \n Katharina Dahmann
        \n Alicia Wirth
        \n Lars Huth
        \n Tolga Tel"""
    )

    sid.markdown("---")

    db_scan_string = "DBSCAN"

    algorithms = {
        db_scan_string: clustering_algorithms.density_based_spatial_clustering_of_applications_with_noise,
        "Mean Shift": clustering_algorithms.mean_shift,
        "k-Means": clustering_algorithms.k_Means,
        "Agglomerative Hierarchical Clustering": clustering_algorithms.ahc_algo,
    }
    plotting_algorithms = {
        db_scan_string: plot_clustering.plotting_dbscan,
        "Mean Shift": plot_clustering.plotting_mean_shift,
        "k-Means": plot_clustering.plotting_kmeans,
        "Agglomerative Hierarchical Clustering": plot_clustering.plotting_ahc,
    }

    datasets = {
        "Happiness and alcohol": dataset_tranformations.happiness_alcohol_consumption,
        "Seeds": dataset_tranformations.fixseeds,
        "HCV dataset": dataset_tranformations.hcvdataset,
        "Liver disorders": dataset_tranformations.liver_disorders,
    }

    dataset_max_epsilons = {
        "Happiness and alcohol": 0.8,
        "Seeds": 0.8,
        "HCV dataset": 5.0,
        "Liver disorders": 2.3,
    }

    dataset_start_epsilons = {
        "Happiness and alcohol": 0.4,
        "Seeds": 0.5,
        "HCV dataset": 2.5,
        "Liver disorders": 0.8,
    }

    st.write(
        """
    # Sata Dience.

    Colleration?
    Culstering?
    """
    )

    dataset_choice = st.selectbox("Which dataset?", tuple(ds for ds in datasets))

    # page display
    if page == "All Algorithms":
        all_algo(
            db_scan_string,
            algorithms,
            plotting_algorithms,
            datasets,
            dataset_choice,
            dataset_start_epsilons,
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
