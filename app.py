import streamlit as st
from constants import INITIAL_OPTIONS
from app_views import preprocessing as pp
from app_views import train_models as tm

st.title("Cell segmentation")
selected_option = st.selectbox("What do you want to do ?", INITIAL_OPTIONS, placeholder="Select an option")
if st.button("Submit"):
    if selected_option == "Preprocess images":
        pp.run_preprocessing_page()
    elif selected_option == "Train a model":
        tm.run_train_models_page()
