import streamlit as st
import os
from src.data_preprocess import normalization as norm
from src.data_preprocess import transformations as transf

def run_preprocessing_page():
    st.title("Preprocessing")
    if st.button("Preprocess"):
        try:
            norm.main()
            st.success("Normalization completed successfully!")
            transf.main()
            st.success("Transformations completed successfully!")
            st.success("Preprocessing completed successfully!")
        except Exception as e:
            st.error(f"An error occurred during preprocessing: {e}")

