import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import os
import time
from tqdm import tqdm
import numpy as np
from preprocess import get_dataloaders
from model import ResNet50, EfficientNetV2, DenseNet
from utils import setup_logger


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, delta=0, path='checkpoint.pth'):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.best_epoch = 0

    def __call__(self, val_loss, model, epoch):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, epoch):
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
        self.best_epoch = epoch


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize Loggers
    log_file_path = setup_logger(SAVE_DIR, MODEL_CHOICE)
    csv_log_path = os.path.join(SAVE_DIR, "metrics_log.csv")
    with open(csv_log_path, "w") as f:
        f.write("Epoch,Train_Loss,Train_Acc,Val_Loss,Val_Acc\n")

    train_loader, val_loader, _ = get_dataloaders(BASE_PATH, batch_size=BATCH_SIZE)

    # Model Selection
    if MODEL_CHOICE == "ResNet":
        model = ResNet50()
    elif MODEL_CHOICE == "EfficientNet":
        model = EfficientNetV2()
    elif MODEL_CHOICE == "DenseNet":
        model = DenseNet()
    else:
        raise ValueError("Invalid MODEL_CHOICE")

    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    scaler = torch.amp.GradScaler('cuda')

    # Initialize Early Stopping
    early_stopping = EarlyStopping(patience=PATIENCE, path=os.path.join(SAVE_DIR, f"best_model.pth"))

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    print(f"Start training for {MODEL_CHOICE}...")

    start_time = time.time()

    for epoch in range(EPOCHS):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for i, (images, labels) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Train Epoch {epoch + 1}", leave=False):
            images, labels = images.to(device), labels.to(device).float().view(-1, 1)
            optimizer.zero_grad()

            with torch.amp.autocast(device_type='cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            preds = torch.round(torch.sigmoid(outputs))
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            # for images, labels in val_loader:
            for i, (images, labels) in tqdm(enumerate(val_loader), total=len(val_loader), desc=f"Val Epoch {epoch + 1}", leave=False):
                images, labels = images.to(device), labels.to(device).float().view(-1, 1)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                preds = torch.round(torch.sigmoid(outputs))
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        epoch_train_loss = train_loss / len(train_loader)
        epoch_val_loss = val_loss / len(val_loader)
        epoch_train_acc = train_correct / train_total
        epoch_val_acc = val_correct / val_total

        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_acc'].append(epoch_val_acc)

        metrics = [epoch + 1, epoch_train_loss, epoch_train_acc, epoch_val_loss, epoch_val_acc]

        with open(csv_log_path, "a") as f:
            f.write(",".join([f"{m:.4f}" for m in metrics]) + "\n")

        print(f"Epoch {epoch + 1}/{EPOCHS} | Train Loss: {metrics[1]:.4f} | Train Acc: {metrics[2]:.4f} | Val Loss: {metrics[3]:.4f} | Val Acc: {metrics[4]:.4f}")

        # Check Early Stopping
        early_stopping(epoch_val_loss, model, epoch + 1)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at Epoch {epoch + 1}")
            break

    total_time = time.time() - start_time

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title(f'{MODEL_CHOICE} Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title(f'{MODEL_CHOICE} Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f'learning_curves.png'))
    plt.show()

    print("\n" + "=" * 60)
    print(f"FINAL SUMMARY: {MODEL_CHOICE}")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Best Epoch Saved: {early_stopping.best_epoch}")
    print(f"Model Path: {early_stopping.path}")
    print(f"CSV Metrics: {csv_log_path}")
    print(f"Text Log: {log_file_path}")
    print("=" * 60)


if __name__ == "__main__":
    # CONFIGURATION
    # MODEL_CHOICE = "ResNet"
    # MODEL_CHOICE = "DenseNet"
    MODEL_CHOICE = "EfficientNet"
    BATCH_SIZE = 32
    LEARNING_RATE = 0.0001
    EPOCHS = 75
    PATIENCE = 10  # For Early Stopping
    BASE_PATH = 'chest_xray'
    SAVE_DIR = f"./model/{MODEL_CHOICE}"
    os.makedirs(SAVE_DIR, exist_ok=True)

    train()
