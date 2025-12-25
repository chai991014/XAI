import torch
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from preprocess import get_dataloaders
from model import ResNet50, EfficientNetV2, DenseNet


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Data (Test Set)
    _, _, test_loader = get_dataloaders(BASE_PATH, batch_size=BATCH_SIZE)

    # 2. Instantiate Model
    if MODEL_CHOICE == "ResNet":
        model = ResNet50()
    elif MODEL_CHOICE == "EfficientNet":
        model = EfficientNetV2()
    elif MODEL_CHOICE == "DenseNet":
        model = DenseNet()
    else:
        raise ValueError("Invalid MODEL_CHOICE")

    # 3. Load Trained Weights
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model weights not found at {MODEL_PATH}")
        return

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device), strict=False)
    model = model.to(device)
    model.eval()

    all_preds = []
    all_probs = []
    all_labels = []

    print(f"Evaluating {MODEL_CHOICE} on test set...")

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device).float().view(-1, 1)

            # Forward pass with mixed precision compatible autocast
            with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                outputs = model(images)
                probs = torch.sigmoid(outputs)
                preds = torch.round(probs)

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # --- METRICS CALCULATION ---
    # As required by Task 2 and the Rubric
    report = classification_report(all_labels, all_preds, target_names=['Normal', 'Pneumonia'], output_dict=True)

    # 4. Evaluation Table (Task 2)
    metrics_df = pd.DataFrame(report).transpose()
    print("\n--- Evaluation Metrics Table ---")
    print(metrics_df)
    metrics_df.to_csv(os.path.join(SAVE_DIR, f'test_metrics.csv'))

    # --- VISUALIZATIONS (Task 2) ---

    # 5. Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Pneumonia'],
                yticklabels=['Normal', 'Pneumonia'])
    plt.title(f'Confusion Matrix: {MODEL_CHOICE}')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(SAVE_DIR, 'confusion_matrix.png'))
    plt.show()

    # 6. ROC Curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve: {MODEL_CHOICE}')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(SAVE_DIR, 'roc_curve.png'))
    plt.show()

    print(f"\nEvaluation Complete. Visuals saved in {SAVE_DIR}")


if __name__ == "__main__":
    # CONFIGURATION
    # MODEL_CHOICE = "ResNet"
    # MODEL_CHOICE = "DenseNet"
    MODEL_CHOICE = "EfficientNet"
    BASE_PATH = 'chest_xray'
    BATCH_SIZE = 32
    MODEL_PATH = f"./model/{MODEL_CHOICE}/best_model.pth"
    SAVE_DIR = f"./model/{MODEL_CHOICE}"

    evaluate()
