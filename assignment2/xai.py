# %% Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import copy
from utils import CNNModel, get_test_loader, get_device
import os


# Set GPU device
device = get_device()


# %% Load data
TEST_ROOT = "data/brain_mri/testing"


# %% Building the model
model = CNNModel()
model.to(device)


# %% LOAD THE TRAINED WEIGHTS
LOAD_PATH = './model/brain_mri_vgg16_best.pth'
try:
    model.load_state_dict(torch.load(LOAD_PATH))
    print(f"Loaded model weights from {LOAD_PATH}")
except FileNotFoundError:
    print(f"Error: Could not find {LOAD_PATH}.")
    exit()


# %% EVALUATION MODE
model.eval()


# %% Prepare data for evaluation
test_dataset, test_loader = get_test_loader(TEST_ROOT, batch_size=1, shuffle=False)


# ==========================================
# 1. LRP
# ==========================================

def new_layer(layer, g):
    layer = copy.deepcopy(layer)
    try:
        layer.weight = torch.nn.Parameter(g(layer.weight))
    except AttributeError:
        pass
    try:
        layer.bias = torch.nn.Parameter(g(layer.bias))
    except AttributeError:
        pass
    return layer


def dense_to_conv(layers):
    newlayers = []
    for i, layer in enumerate(layers):
        if isinstance(layer, nn.Linear):
            newlayer = None
            if i == 0:
                m, n = 512, layer.weight.shape[0]
                newlayer = nn.Conv2d(m, n, 7)
                newlayer.weight = nn.Parameter(layer.weight.reshape(n, m, 7, 7))
            else:
                m, n = layer.weight.shape[1], layer.weight.shape[0]
                newlayer = nn.Conv2d(m, n, 1)
                newlayer.weight = nn.Parameter(layer.weight.reshape(n, m, 1, 1))
            newlayer.bias = nn.Parameter(layer.bias)
            newlayers += [newlayer]
        else:
            newlayers += [layer]
    return newlayers


def get_linear_layer_indices(model):
    offset = len(model.vgg16._modules['features']) + 1
    indices = []
    for i, layer in enumerate(model.vgg16._modules['classifier']):
        if isinstance(layer, nn.Linear):
            indices.append(i)
    indices = [offset + val for val in indices]
    return indices


def apply_lrp_on_vgg16(model, image):
    image = torch.unsqueeze(image, 0)
    layers = list(model.vgg16._modules['features']) \
             + [model.vgg16._modules['avgpool']] \
             + dense_to_conv(list(model.vgg16._modules['classifier']))
    linear_layer_indices = get_linear_layer_indices(model)
    n_layers = len(layers)
    activations = [image] + [None] * n_layers

    for layer in range(n_layers):
        if layer in linear_layer_indices:
            if layer == 32:
                activations[layer] = activations[layer].reshape((1, 512, 7, 7))
        activation = layers[layer].forward(activations[layer])
        if isinstance(layers[layer], torch.nn.modules.pooling.AdaptiveAvgPool2d):
            activation = torch.flatten(activation, start_dim=1)
        activations[layer + 1] = activation

    output_activation = activations[-1].detach().cpu().numpy()
    max_activation = output_activation.max()
    # one_hot_output = [val if val == max_activation else 0
    #                   for val in output_activation[0]]
    # activations[-1] = torch.FloatTensor([one_hot_output]).to(device)
    one_hot_output = np.where(output_activation == max_activation, output_activation, 0.0)
    activations[-1] = torch.from_numpy(one_hot_output).to(device)

    relevances = [None] * n_layers + [activations[-1]]
    for layer in range(0, n_layers)[::-1]:
        current = layers[layer]
        if isinstance(current, torch.nn.MaxPool2d):
            layers[layer] = torch.nn.AvgPool2d(2)
            current = layers[layer]
        if isinstance(current, torch.nn.Conv2d) or \
                isinstance(current, torch.nn.AvgPool2d) or \
                isinstance(current, torch.nn.Linear):
            activations[layer] = activations[layer].data.requires_grad_(True)
            if layer <= 16:
                rho = lambda p: p + 0.25 * p.clamp(min=0)
                incr = lambda z: z + 1e-9
            if 17 <= layer <= 30:
                rho = lambda p: p
                incr = lambda z: z + 1e-9 + 0.25 * ((z ** 2).mean() ** .5).data
            if layer >= 31:
                rho = lambda p: p
                incr = lambda z: z + 1e-9
            z = incr(new_layer(layers[layer], rho).forward(activations[layer]))
            s = (relevances[layer + 1] / z).data
            (z * s).sum().backward()
            c = activations[layer].grad
            relevances[layer] = (activations[layer] * c).data
        else:
            relevances[layer] = relevances[layer + 1]
    return relevances[0]


# ==========================================
# 2. GRAD-CAM
# ==========================================

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Hook the layers
        self.target_layer.register_backward_hook(self.save_gradient)
        self.target_layer.register_forward_hook(self.save_activation)

    def save_gradient(self, module, grad_input, grad_output):
        # Save gradients for Backward pass
        self.gradients = grad_output[0]

    def save_activation(self, module, input, output):
        # Save activations for Forward pass
        self.activations = output

    def generate(self, input_image, target_class=None):
        # Force input to require gradient.
        # This ensures the backward pass goes all the way through the frozen layers.
        input_image = input_image.clone().detach().requires_grad_(True)

        # 1. Forward Pass
        output = self.model(input_image)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # 2. Backward Pass
        self.model.zero_grad()
        # One-hot encode the target
        one_hot_output = torch.FloatTensor(1, output.size()[-1]).zero_().to(device)
        one_hot_output[0][target_class] = 1

        # Trigger hooks
        output.backward(gradient=one_hot_output)

        # 3. Compute Grad-CAM
        # Global Average Pooling on gradients (Weights)
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        # Weighted sum of activations
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        # ReLU to keep only positive influence
        cam = F.relu(cam)

        # 4. Resize to match input image size (e.g., 256x256)
        cam = F.interpolate(cam, size=input_image.shape[2:], mode='bilinear', align_corners=False)

        # Normalize 0-1
        cam = cam - cam.min()
        cam = cam / cam.max()

        return cam.squeeze().detach().cpu().numpy()


# ==========================================
# 3. GUIDED BACKPROP
# ==========================================

class GuidedBackprop:
    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.update_relus()

    def update_relus(self):
        def relu_backward_hook_function(module, grad_in, grad_out):
            # Clamps negative gradients to 0
            if isinstance(module, nn.ReLU):
                return (torch.clamp(grad_in[0], min=0.0),)

        # Loop through all modules and attach hooks to ReLUs
        for module in self.model.modules():
            if isinstance(module, nn.ReLU):
                self.hooks.append(module.register_backward_hook(relu_backward_hook_function))

    def generate(self, input_image, target_class=None):
        # 1. Enable Gradients on Input
        input_image = input_image.requires_grad_(True)

        # 2. Forward Pass
        output = self.model(input_image)
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # 3. Backward Pass
        self.model.zero_grad()

        # Create one-hot target
        one_hot_output = torch.FloatTensor(1, output.size()[-1]).zero_().to(device)
        one_hot_output[0][target_class] = 1

        output.backward(gradient=one_hot_output)

        # 4. Get Input Gradients
        gradients = input_image.grad.squeeze().detach().cpu().numpy()

        # 5. Clean up: Remove hooks so they don't affect other methods
        for hook in self.hooks:
            hook.remove()

        # 6. Process for Visualization (Sum channels + Abs)
        saliency = np.sum(np.abs(gradients), axis=0)  # Sum across RGB channels
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())  # Normalize

        return saliency


# ==========================================
# 4. XAI
# ==========================================

all_preds = []
all_labels = []

with torch.no_grad():
    for i, (inputs, labels) in enumerate(test_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        # Store predictions and labels
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())


class_names = test_dataset.classes
no_tumor_idx = class_names.index('no_tumor')

arr_preds = np.array(all_preds)
arr_labels = np.array(all_labels)

categories = {
    "1_False_Negatives": np.where((arr_labels != no_tumor_idx) & (arr_preds == no_tumor_idx))[0],
    "2_False_Positives": np.where((arr_labels == no_tumor_idx) & (arr_preds != no_tumor_idx))[0],
    "3_Tumor_Confusion": np.where((arr_labels != arr_preds) & (arr_labels != no_tumor_idx) & (arr_preds != no_tumor_idx))[0],
    "4_True_Positives":  np.where((arr_labels == arr_preds) & (arr_labels != no_tumor_idx))[0]
}


def calculate_confidence_drop(model, image_tensor, heatmap, target_class):
    """
    Calculates how much the model's confidence drops when we remove
    the top 20% most relevant pixels identified by the heatmap.
    """
    # 1. Find threshold for top 20% salient pixels
    flat_heatmap = heatmap.flatten()
    threshold = np.percentile(flat_heatmap, 80)
    mask = torch.from_numpy(heatmap > threshold).to(device)

    # 2. Perturb image (Black out the hot spots)
    # Expand mask to match image channels (C, H, W)
    mask_3d = mask.unsqueeze(0).expand_as(image_tensor)
    perturbed_img = image_tensor.clone()
    perturbed_img[mask_3d] = 0  # Set important pixels to black

    # 3. Re-evaluate
    model.eval()
    with torch.no_grad():
        # Original Score
        outputs_orig = model(image_tensor.unsqueeze(0))
        score_orig = torch.softmax(outputs_orig, dim=1)[0, target_class].item()

        # Perturbed Score
        outputs_pert = model(perturbed_img.unsqueeze(0))
        score_pert = torch.softmax(outputs_pert, dim=1)[0, target_class].item()

    return score_orig, score_pert, (score_orig - score_pert)


def calculate_rrs(heatmap, percentile=90):
    total_relevance = np.sum(heatmap)
    if total_relevance == 0:
        return 0

    # Identify the threshold for the most relevant region (ROI)
    threshold = np.percentile(heatmap, percentile)
    roi_mask = heatmap >= threshold

    # Sum relevance inside the ROI
    roi_relevance = np.sum(heatmap[roi_mask])

    # RRS = Energy in ROI / Total Energy
    rrs_score = roi_relevance / total_relevance
    return round(rrs_score, 4)


def xai(model, test_dataset, cat_folder, cat_name, img_id, metrics_list):
    try:
        image_tensor, label_idx = test_dataset[img_id]
        image_tensor = image_tensor.to(device)
        input_batch = image_tensor.unsqueeze(0)  # Add batch dimension

        # --- Prediction ---
        outputs = model(input_batch)
        pred_idx = outputs.max(1).indices.item()
        pred_label = test_dataset.classes[pred_idx]
        true_label = test_dataset.classes[label_idx]

        # --- METHOD 1: LRP ---
        lrp_relevances = apply_lrp_on_vgg16(model, image_tensor)
        lrp_relevances = lrp_relevances.permute(0, 2, 3, 1).detach().cpu().numpy()[0]
        lrp_heatmap = np.interp(lrp_relevances, (lrp_relevances.min(), lrp_relevances.max()), (0, 1))[:, :, 0]

        # Metric: LRP RRS
        _, _, lrp_drop = calculate_confidence_drop(model, image_tensor, lrp_heatmap, pred_idx)

        # Metric: LRP Confidence Drop
        lrp_rrs = calculate_rrs(lrp_heatmap)

        # --- METHOD 2: Grad-CAM ---
        target_layer = model.vgg16.features[28]
        gradcam_obj = GradCAM(model, target_layer)
        gradcam_heatmap = gradcam_obj.generate(input_batch, target_class=pred_idx)

        # Metric: Grad-CAM Confidence Drop
        _, _, gradcam_drop = calculate_confidence_drop(model, image_tensor, gradcam_heatmap, pred_idx)

        # Metric: Grad-CAM RRS
        gradcam_rrs = calculate_rrs(gradcam_heatmap)

        # --- METHOD 3: Guided Backprop ---
        model = CNNModel().to(device)
        model.load_state_dict(torch.load(LOAD_PATH))
        model.eval()

        gbp_obj = GuidedBackprop(model)
        gbp_heatmap = gbp_obj.generate(input_batch, target_class=pred_idx)

        # Metric: GBP Confidence Drop
        _, _, gbp_drop = calculate_confidence_drop(model, image_tensor, gbp_heatmap, pred_idx)

        # Metric: GBP RRS
        gbp_rrs = calculate_rrs(gbp_heatmap)

        # --- SAVE METRICS ---
        metrics_list.append({
            "Image_ID": img_id,
            "Category": cat_name,
            "True_Label": true_label,
            "Pred_Label": pred_label,
            "LRP_Conf_Drop": round(lrp_drop, 4),
            "LRP_RRS": round(lrp_rrs, 4),
            "GradCAM_Conf_Drop": round(gradcam_drop, 4),
            "GradCAM_RRS": round(gradcam_rrs, 4),
            "GBP_Conf_Drop": round(gbp_drop, 4),
            "GBP_RRS": round(gbp_rrs, 4)
        })

        # --- PLOTTING ---
        fig = plt.figure(figsize=(15, 5))

        # Original
        plt.subplot(1, 4, 1)
        plt.title(f"Original: {true_label}")
        # Permute (C,H,W) -> (H,W,C) for matplotlib
        plt.imshow(image_tensor.permute(1, 2, 0).detach().cpu().numpy())
        plt.axis('off')

        # LRP
        plt.subplot(1, 4, 2)
        plt.title(f"LRP (Drop: {lrp_drop:.2f} | RRS: {lrp_rrs:.2f})")
        plt.imshow(lrp_heatmap, cmap="seismic")
        plt.axis('off')

        # Grad-CAM
        plt.subplot(1, 4, 3)
        plt.title(f"Grad-CAM (Drop: {gradcam_drop:.2f} | RRS: {gradcam_rrs:.2f})")
        plt.imshow(image_tensor.permute(1, 2, 0).detach().cpu().numpy(), alpha=1.0)  # Background
        plt.imshow(gradcam_heatmap, cmap='jet', alpha=0.5)  # Overlay
        plt.axis('off')

        # Guided Backprop
        plt.subplot(1, 4, 4)
        plt.title(f"Guided BP (Drop: {gbp_drop:.2f} | RRS: {gbp_rrs:.2f})")
        plt.imshow(gbp_heatmap, cmap='gray')
        plt.axis('off')

        caption_color = 'green' if pred_idx == label_idx else 'red'
        caption_text = f"ID: {img_id} | Ground Truth: {true_label} | Prediction: {pred_label}"

        plt.figtext(0.5, 0.05, caption_text,
                    ha="center",
                    fontsize=14,
                    fontweight='bold',
                    color=caption_color)

        plt.tight_layout()

        # Save logic
        filename = f"ID_{img_id}_True_{true_label}_Pred_{pred_label}.png"
        save_path = os.path.join(cat_folder, filename)
        plt.savefig(save_path)
        plt.close(fig)

    except Exception as e:
        print(f"Skipping ID {img_id} due to error: {e}")


SAVE_ROOT = "results_batch"
os.makedirs(SAVE_ROOT, exist_ok=True)
print(f"Batch processing started. Saving to: {os.path.abspath(SAVE_ROOT)}")

metrics_data = []

for cat_name, id_list in categories.items():
    cat_folder = os.path.join(SAVE_ROOT, cat_name)
    os.makedirs(cat_folder, exist_ok=True)
    print(f"\nProcessing Category: {cat_name} ({len(id_list)} images)...")
    for img_id in id_list:
        xai(model, test_dataset, cat_folder, cat_name, img_id, metrics_data)

df_metrics = pd.DataFrame(metrics_data)
csv_path = os.path.join(SAVE_ROOT, "xai_metrics_report.csv")
df_metrics.to_csv(csv_path, index=False)
# %%
