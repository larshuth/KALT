import ahc_algo as ahc
import hcvdataset as hcv

import streamlit as st
import pandas as pd
import numpy as np


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    temp_dis = ahc.pca_distribution(hcv.hcvdataset(use_pca=False)[0])
    st.write(temp_dis[0])
    number = st.slider("Pick a number for component usage for pca", 2, temp_dis[1])
    #if you want to call ahc_alg_acc()
    X = hcv.hcvdataset(use_pca=True, n_components=number)
    temp = ahc.ahc_algo_acc(X[0])
    st.write("""ahc algorithm""")
    st.write("accuracy calculated with", temp[1], ":", temp[0])
    st.write(ahc.ahc_algo(X))

