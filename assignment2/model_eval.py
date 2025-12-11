# %% Imports
import torch
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from utils import CNNModel, get_test_loader, get_device


# Set GPU device
device = get_device()

# %% Load data
TEST_ROOT = "data/brain_mri/testing"


# %% Building the model (Must match training structure)
model = CNNModel()
model.to(device)


# %% LOAD THE TRAINED WEIGHTS
LOAD_PATH = './model/brain_mri_vgg16_best.pth'
try:
    model.load_state_dict(torch.load(LOAD_PATH))
    print(f"Loaded model weights from {LOAD_PATH}")
except FileNotFoundError:
    print(f"Error: Could not find {LOAD_PATH}. Please make sure the path is correct.")
    exit()


# %% EVALUATION SETUP
model.eval()
test_dataset, test_loader = get_test_loader(TEST_ROOT, batch_size=32, shuffle=False)


# %% RUN EVALUATION ON FULL DATASET
print("Starting evaluation on the full test set...")
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


# %% METRICS AND REPORT
# Convert class indices to class names
class_names = test_dataset.classes
no_tumor_idx = class_names.index('no_tumor')
arr_preds = np.array(all_preds)
arr_labels = np.array(all_labels)

# --- CASE 1: FALSE NEGATIVES (The Most Dangerous) ---
# Actual = NOT 'no_tumor', Predicted = 'no_tumor'
# i.e., Patient has cancer, AI says healthy.
false_negatives = np.where((arr_labels != no_tumor_idx) & (arr_preds == no_tumor_idx))[0]

# --- CASE 2: FALSE POSITIVES (The False Alarm) ---
# Actual = 'no_tumor', Predicted = NOT 'no_tumor'
# i.e., Patient is healthy, AI says cancer.
false_positives = np.where((arr_labels == no_tumor_idx) & (arr_preds != no_tumor_idx))[0]

# --- CASE 3: WRONG TUMOR TYPE (Confusion) ---
# Actual != Predicted, and neither is 'no_tumor'
tumor_confusion = np.where((arr_labels != arr_preds) &
                           (arr_labels != no_tumor_idx) &
                           (arr_preds != no_tumor_idx))[0]

# --- CASE 4: TRUE POSITIVES (Sanity Check) ---
# Pick a random correct prediction that IS a tumor
true_positives = np.where((arr_labels == arr_preds) & (arr_labels != no_tumor_idx))[0]

print("\n" + "=" * 40)
print("SUGGESTED IMAGE IDS FOR REPORT")
print("=" * 40)
print(f"1. FALSE NEGATIVES (Dangerous Misses) -> Total: {len(false_negatives)}")
print(f"   IDs: {false_negatives[:10]}")
print(f"\n2. FALSE POSITIVES (False Alarms) -> Total: {len(false_positives)}")
print(f"   IDs: {false_positives[:10]}")
print(f"\n3. TUMOR CONFUSION (Wrong Type) -> Total: {len(tumor_confusion)}")
print(f"   IDs: {tumor_confusion[:10]}")
print(f"\n4. TRUE POSITIVES (Correct Tumors) -> Total: {len(true_positives)}")
print(f"   IDs: {true_positives[:10]}")


# MULTICLASS REPORT & MATRIX
print("\n" + "=" * 40)
print("FINAL CLASSIFICATION REPORT")
print("=" * 40)
print(classification_report(all_labels, all_preds, target_names=class_names))

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix (Multi-Class)')
plt.tight_layout()
plt.savefig('confusion_matrix_multiclass.png')
plt.show()

# BINARY REPORT (Tumor vs No Tumor)
binary_labels = [0 if x == no_tumor_idx else 1 for x in all_labels]
binary_preds = [0 if x == no_tumor_idx else 1 for x in all_preds]

cm_binary = confusion_matrix(binary_labels, binary_preds)
tn, fp, fn, tp = cm_binary.ravel()

sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

print("\n" + "=" * 40)
print("BINARY METRICS (Clinical Focus)")
print("=" * 40)
print(f"Sensitivity (Recall): {sensitivity:.4f}")
print("  (Ability to find all tumors. Higher is better for safety.)")
print(f"Specificity:          {specificity:.4f}")
print("  (Ability to identify healthy patients. Higher is better for efficiency.)")
print(f"False Negatives:      {fn}")
print("  (CRITICAL: Patients with tumor sent home as healthy)")

# Plot Binary Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm_binary, annot=True, fmt='d', cmap='Reds',
            xticklabels=['Healthy', 'Tumor'], yticklabels=['Healthy', 'Tumor'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Binary Confusion Matrix (Tumor Detection)')
plt.tight_layout()
plt.savefig('confusion_matrix_binary.png')
plt.show()
