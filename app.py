import streamlit as st
import numpy as np
import cv2
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import pickle
from PIL import Image

from src.models.unet.unetr2d import UNETR2D

PRETRAINED_MODELS = ["Pretrained: Best Dice model"]
MODELS = ["UNETR2D", "UNET", "GANS"]
TASKS = ["Choose Model", "Upload Data", "Train Model", "Save Model", "Test Model", "Run Model on New Data"]
SUPPORTED_EXTENSIONS = ["png", "jpg", "jpeg"]

def preprocess_image(image_file):
    image = Image.open(image_file).convert('L')
    image = image.resize((128, 128))
    image = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(image, axis=(0, 1))

st.title("Cell Segmentation Model Manager")

task = st.selectbox("Select a task", TASKS)

if task == "Choose Model":
    model_type = st.selectbox("Select a Model", PRETRAINED_MODELS + MODELS)
    st.session_state["model_type"] = model_type
    st.write(f"Selected Model: {model_type}")

elif task == "Upload Data":
    uploaded_images = st.file_uploader("Upload Image Files", accept_multiple_files=True, type=SUPPORTED_EXTENSIONS)
    uploaded_masks = st.file_uploader("Upload Mask Files", accept_multiple_files=True, type=SUPPORTED_EXTENSIONS)
    uploaded_test_images = st.file_uploader("Upload Test Image Files", accept_multiple_files=True, type=SUPPORTED_EXTENSIONS)
    uploaded_test_masks = st.file_uploader("Upload Test Mask Files", accept_multiple_files=True, type=SUPPORTED_EXTENSIONS)
    if uploaded_images and uploaded_masks:
        images = np.array([preprocess_image(img) for img in uploaded_images])
        masks = np.array([preprocess_image(mask) for mask in uploaded_masks])
        test_images = np.array([preprocess_image(img) for img in uploaded_test_images]) if uploaded_test_images else None
        test_masks = np.array([preprocess_image(mask) for mask in uploaded_test_masks]) if uploaded_test_masks else None
        np.save("images.npy", images) if images is not None else None
        np.save("masks.npy", masks) if masks is not None else None
        np.save("test_images.npy", test_images) if test_images is not None else None
        np.save("test_masks.npy", test_masks) if test_masks is not None else None
        st.success("Data uploaded and saved successfully!")

elif task == "Train Model":
    if os.path.exists("images.npy") and os.path.exists("masks.npy"):
        model_type = st.session_state.get("model_type", None)
        if model_type in PRETRAINED_MODELS:
            st.warning("Pretrained model is not available for this task! Go test the model instead.")
            st.stop()
        images = np.load("images.npy")
        masks = np.load("masks.npy")
        st.warning("TODO: Train the model using the uploaded data!")
    else:
        st.warning("Upload data first!")

elif task == "Save Model":
    if "model" in st.session_state and (st.session_state.get("model_type", None) not in PRETRAINED_MODELS):
        name = st.text_input("Enter Model Name", key="model_name")
        torch.save(st.session_state["model"], f"{name}_trained.pth")
        st.success("Model saved successfully!")
    elif st.session_state.get("model_type", None) in PRETRAINED_MODELS:
        st.warning("Pretrained models cannot be saved!")
        st.stop()
    else:
        st.warning("Train a model first!")

elif task == "Test Model":
    if "model" in st.session_state:
        st.warning("TODO: Test the model using the uploaded data")
    else:
        st.warning("Train a model first!")

elif task == "Run Model on New Data":
    uploaded_test_images = st.file_uploader("Upload New Image Files", accept_multiple_files=True, type=["png", "jpg", "jpeg"])
    if uploaded_test_images and "model" in st.session_state:
        st.warning("TODO: Run the model on the new data!")
    else:
        st.warning("Upload images and train a model first!")
