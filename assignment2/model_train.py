# %% Imports
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from utils import CNNModel, get_train_loader, get_test_loader, get_device


# Set GPU device
device = get_device()


# %% Load data
TRAIN_ROOT = "data/brain_mri/training"
TEST_ROOT = "data/brain_mri/testing"


# %% Building the model
model = CNNModel()
model.to(device)


# %% Prepare data for pretrained model and create data loaders
train_dataset, train_loader = get_train_loader(TRAIN_ROOT, batch_size=32, shuffle=True)
test_dataset, test_loader = get_test_loader(TEST_ROOT, batch_size=32, shuffle=True)


# %% Train
cross_entropy_loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001, weight_decay=2e-3)
epochs = 10
train_losses = []
test_losses = []
best_test_loss = float('inf')
best_model_path = './model/brain_mri_vgg16_best.pth'

print(f"\n------Start training process------\n")
# Iterate x epochs over the train data
for epoch in range(epochs):

    # --- TRAINING PHASE ---
    model.train()
    running_train_loss = 0.0

    for i, batch in enumerate(train_loader, 0):
        inputs, labels = batch
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        # Labels are automatically one-hot-encoded
        loss = cross_entropy_loss(outputs, labels)
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item()

    avg_train_loss = running_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # --- TESTING PHASE ---
    model.eval()
    running_test_loss = 0.0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = cross_entropy_loss(outputs, labels)
            running_test_loss += loss.item()

    avg_test_loss = running_test_loss / len(test_loader)
    test_losses.append(avg_test_loss)

    print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f}")

    # --- SAVE BEST MODEL ---
    if avg_test_loss < best_test_loss:
        best_test_loss = avg_test_loss
        torch.save(model.state_dict(), best_model_path)
        print(f"  --> New best model saved! (Epoch: {epoch + 1} | Loss: {best_test_loss:.4f})")

print(f"\n------End training process------\n")


# %% SAVE COMBINED LOSS CURVE
plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs + 1), train_losses, marker='o', label='Training Loss')
plt.plot(range(1, epochs + 1), test_losses, marker='x', linestyle='--', label='Testing Loss')
plt.title('Training vs Testing Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('loss_curve_combined.png')
print("Combined loss curve saved to loss_curve_combined.png")
plt.show()
