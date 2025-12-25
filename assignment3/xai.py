from xai_densenet import run_xai as densenet
from xai_efficientnet import run_xai as efficientnet
import os
import torch

if __name__ == "__main__":

    # --- CONFIGURATION ---
    MODEL_PATH = "./model/EfficientNet/best_model.pth"
    XAI_OUTPUT_DIR = "./results/EfficientNet"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create TP/FP/TN/FN directories for Task 4
    for cat in ['TP', 'FP', 'TN', 'FN']:
        os.makedirs(os.path.join(XAI_OUTPUT_DIR, cat), exist_ok=True)

    efficientnet(MODEL_PATH, XAI_OUTPUT_DIR, DEVICE)

    # --- CONFIGURATION ---
    MODEL_PATH = "./model/DenseNet/best_model.pth"
    XAI_OUTPUT_DIR = "./results/DenseNet"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create TP/FP/TN/FN directories for Task 4
    for cat in ['TP', 'FP', 'TN', 'FN']:
        os.makedirs(os.path.join(XAI_OUTPUT_DIR, cat), exist_ok=True)

    densenet(MODEL_PATH, XAI_OUTPUT_DIR, DEVICE)
