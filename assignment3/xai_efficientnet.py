import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import os
import pandas as pd
from tqdm import tqdm
from torchvision import transforms, datasets
from torchcam.methods import GradCAMpp
from model import EfficientNetV2


def disable_inplace(model):
    """Recursively disables inplace operations for gradient stability."""
    for module in model.modules():
        if hasattr(module, 'inplace'):
            module.inplace = False
    return model


def generate_rise_masks(n_masks=1000, grid_size=10, input_size=(224, 224), p1=0.2):
    """Generates random bilinear upsampled masks for RISE sampling."""
    masks = []
    for _ in range(n_masks):
        # Create a coarse random grid and upsample to 224x224
        grid = np.random.rand(grid_size, grid_size) < p1
        grid = grid.astype(np.float32)

        mask = torch.from_numpy(grid).unsqueeze(0).unsqueeze(0)
        mask = F.interpolate(mask, size=input_size, mode='bilinear', align_corners=False)
        masks.append(mask)

    return torch.cat(masks)


def calculate_metrics(model, image_tensor, heatmap, device):
    """Calculates Confidence Drop and RRS metrics"""
    threshold = np.percentile(heatmap, 50)
    mask = torch.tensor(heatmap < threshold).to(device).float()
    mask = mask.view(1, 1, 224, 224)

    with torch.no_grad():
        orig_logit = model(image_tensor).item()
        masked_logit = model(image_tensor * mask).item()

    # (Original - Masked) / |Original| to handle negative logits properly.
    conf_drop = (orig_logit - masked_logit) / (abs(orig_logit) + 1e-8)
    # RRS = Average relevance of top 15% pixels
    rrs = np.mean(heatmap[heatmap >= threshold])
    return conf_drop, rrs


def run_xai(MODEL_PATH, XAI_OUTPUT_DIR, DEVICE):
    # 1. Load Model
    model = EfficientNetV2().to(DEVICE)
    model = disable_inplace(model)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE), strict=False)
    model.eval()

    # 2. Setup Test Data
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_dataset = datasets.ImageFolder('chest_xray/test', transform=transform)

    # Pre-generate RISE masks
    print("Generating 1000 RISE masks...")
    rise_masks = generate_rise_masks().to(DEVICE)

    # 3. Initialize Explainers
    # Targeting the final expansion convolution of EfficientNetV2-S
    target_layer = 'features.8.0'

    try:
        cam_extractor = GradCAMpp(model, target_layer=target_layer)
    except ValueError:
        target_layer = 'features.7'  # Fallback for different library versions
        cam_extractor = GradCAMpp(model, target_layer=target_layer)

    metrics_list = []

    print(f"Running XAI for the whole test set ({len(test_dataset)} images)...")
    for img_id in tqdm(range(len(test_dataset))):
        image_tensor, label_idx = test_dataset[img_id]
        image_tensor = image_tensor.unsqueeze(0).to(DEVICE)

        # Create leaf tensor for attribution
        input_tensor = image_tensor.clone().detach().requires_grad_(True)

        # Get Prediction
        output = model(input_tensor)
        pred_prob = torch.sigmoid(output).item()
        pred_idx = 1 if pred_prob > 0.5 else 0

        # Sort into Confusion Matrix Categories
        if label_idx == 1 and pred_idx == 1:
            cat_name = 'TP'
        elif label_idx == 0 and pred_idx == 1:
            cat_name = 'FP'
        elif label_idx == 0 and pred_idx == 0:
            cat_name = 'TN'
        else:
            cat_name = 'FN'

        true_label = "Normal" if label_idx == 0 else "Pneumonia"
        pred_label = "Normal" if pred_idx == 0 else "Pneumonia"

        # --- GENERATE HEATMAPS ---

        # A. Grad-CAM++ (Explaining index 0)
        model.zero_grad()
        gcam_out = model(input_tensor)
        gcam_raw = cam_extractor(0, gcam_out)[0]

        # Force 4D shape [1, 1, 7, 7] before interpolation
        # If gcam_raw is [1, 7, 7], unsqueeze(0) makes it [1, 1, 7, 7]
        if gcam_raw.ndim == 3:
            gcam_raw = gcam_raw.unsqueeze(0)
        elif gcam_raw.ndim == 2:
            gcam_raw = gcam_raw.unsqueeze(0).unsqueeze(0)

        gcam_upsampled = F.interpolate(gcam_raw, size=(224, 224), mode='bilinear', align_corners=False)
        gcam_heatmap = gcam_upsampled.squeeze().cpu().numpy()

        # Remove hooks immediately after generating the heatmap
        cam_extractor.remove_hooks()

        # B. RISE Attribution Logic
        with torch.no_grad():
            masked_inputs = input_tensor * rise_masks
            preds_logits = []

            for i in range(0, len(rise_masks), 50):
                logits = model(masked_inputs[i:i + 50])
                preds_logits.append(logits)

            all_logits = torch.cat(preds_logits).view(-1, 1, 1, 1)

            # --- THE CONTRASTIVE STEP ---
            # Use logits to avoid the 0.0-1.0 saturation trap
            # Heatmap = Sum(Mask_i * Logit_i) / Sum(Mask_i)
            crise_raw = (rise_masks * all_logits).sum(dim=0).squeeze().cpu().numpy()
            crise_raw /= (rise_masks.sum(dim=0).squeeze().cpu().numpy() + 1e-8)

        # Normalize to [0, 1] range instead of centering at zero
        # This removes the "negative" (blue) evidence and focuses on intensity.
        crise_norm = (crise_raw - crise_raw.min()) / (crise_raw.max() - crise_raw.min() + 1e-8)

        # --- CALCULATE FAITHFULNESS METRICS ---
        g_drop, g_rrs = calculate_metrics(model, input_tensor, gcam_heatmap, DEVICE)
        r_drop, r_rrs = calculate_metrics(model, input_tensor, crise_norm, DEVICE)

        metrics_list.append({
            "Image_ID": img_id, "Category": cat_name, "True": true_label, "Pred": pred_label,
            "GCAM_Drop": round(g_drop, 4), "GCAM_RRS": round(g_rrs, 4),
            "C-RISE_Drop": round(r_drop, 4), "C-RISE_RRS": round(r_rrs, 4)
        })

        # --- PLOTTING (3-Panel Format) ---
        fig = plt.figure(figsize=(12, 5))
        # Denormalize for visualization
        img_vis = (image_tensor[0].permute(1, 2, 0).detach().cpu().numpy() * 0.225) + 0.456

        plt.subplot(1, 3, 1)
        plt.title(f"Original: {true_label}")
        plt.imshow(np.clip(img_vis, 0, 1))
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.title(f"GradCAM++ (Drop: {g_drop:.2f} | RRS: {g_rrs:.2f})")
        plt.imshow(np.clip(img_vis, 0, 1), alpha=1.0)
        plt.imshow(gcam_heatmap, cmap='jet', alpha=0.5)
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.title(f"C-RISE (Drop: {r_drop:.2f} | RRS: {r_rrs:.2f})")
        plt.imshow(np.clip(img_vis, 0, 1))
        plt.imshow(crise_norm, cmap='jet', alpha=0.5)
        plt.axis('off')

        caption_color = 'green' if pred_idx == label_idx else 'red'
        plt.figtext(0.5, 0.05, f"ID: {img_id} | GT: {true_label} | Pred: {pred_label}",
                    ha="center", fontsize=14, fontweight='bold', color=caption_color)

        plt.tight_layout()
        plt.savefig(os.path.join(XAI_OUTPUT_DIR, cat_name, f"ID_{img_id}_{cat_name}.png"))
        plt.close(fig)

        # Re-initialize the extractor for the NEXT image in the loop
        cam_extractor = GradCAMpp(model, target_layer=target_layer)

    # Save quantitative results
    pd.DataFrame(metrics_list).to_csv(os.path.join(XAI_OUTPUT_DIR, "xai_metrics.csv"), index=False)
    print(f"Analysis complete. Results saved in {XAI_OUTPUT_DIR}")


if __name__ == "__main__":
    # --- CONFIGURATION ---
    MODEL_PATH = "./model/EfficientNet/best_model.pth"
    XAI_OUTPUT_DIR = "./results/EfficientNet"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create TP/FP/TN/FN directories for Task 4
    for cat in ['TP', 'FP', 'TN', 'FN']:
        os.makedirs(os.path.join(XAI_OUTPUT_DIR, cat), exist_ok=True)

    run_xai(MODEL_PATH, XAI_OUTPUT_DIR, DEVICE)
