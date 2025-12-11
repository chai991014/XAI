import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    return device


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)

        for param in self.vgg16.features.parameters():
            param.requires_grad = False

        self.vgg16.classifier = nn.Sequential(
            nn.Linear(25088, 128),  # 1. Drastic reduction: 25,088 inputs -> 128 neurons
            nn.ReLU(),              # 2. Activation function
            nn.Dropout(0.5),        # 3. Drop 50% of neurons (Crucial for overfitting)
            nn.Linear(128, 4)       # 4. Output layer: 128 inputs -> 4 Classes
        )

    def forward(self, x):
        x = self.vgg16(x)
        return x


def get_train_loader(root_path, batch_size=32, shuffle=True):
    train_dataset = torchvision.datasets.ImageFolder(
        root=root_path,
        transform=transforms.Compose([
            transforms.Resize((255, 255)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=30),
            transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.8, 1.2)),
            transforms.ToTensor()
        ])
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )

    return train_dataset, train_loader


def get_test_loader(root_path, batch_size=32, shuffle=False):
    test_dataset = torchvision.datasets.ImageFolder(
        root=root_path,
        transform=transforms.Compose([
            transforms.Resize((255, 255)),
            transforms.ToTensor()
        ])
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=shuffle  # Shuffle False is better for evaluation to keep order consistent
    )

    return test_dataset, test_loader
