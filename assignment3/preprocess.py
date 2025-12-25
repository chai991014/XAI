import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from sklearn.model_selection import train_test_split
import os


def get_dataloaders(base_dir, img_size=224, batch_size=32):
    # 1. Define SOTA Data Augmentation
    # Includes rotations, horizontal flips, and color jitters as requested
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomRotation(10),                          # +/- 10 degrees
        transforms.RandomHorizontalFlip(),                      # Anatomically consistent
        transforms.ColorJitter(brightness=0.1, contrast=0.1),   # +/- 10%
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet Stats
    ])

    # Standard preprocessing for Validation and Test
    val_test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 2. Combine and Split Data
    # Load both 'train' and 'val' folders as one large training pool
    train_path = os.path.join(base_dir, 'train')
    original_val_path = os.path.join(base_dir, 'val')

    # Load base datasets for both Train (augmented) and Val (standard) transforms
    # Use ConcatDataset to merge the small val folder into train
    train_base_aug = datasets.ImageFolder(train_path, transform=train_transform)
    extra_val_aug = datasets.ImageFolder(original_val_path, transform=train_transform)
    combined_aug = torch.utils.data.ConcatDataset([train_base_aug, extra_val_aug])

    train_base_std = datasets.ImageFolder(train_path, transform=val_test_transform)
    extra_val_std = datasets.ImageFolder(original_val_path, transform=val_test_transform)
    combined_std = torch.utils.data.ConcatDataset([train_base_std, extra_val_std])

    # Extract targets for stratification
    targets = train_base_aug.targets + extra_val_aug.targets
    indices = list(range(len(combined_aug)))

    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.2,
        stratify=targets,
        random_state=42
    )

    # Create Subsets from the correctly matched combined datasets
    train_subset = Subset(combined_aug, train_idx)
    val_subset = Subset(combined_std, val_idx)  # Uses val indices on the full combined pool

    # 3. Test Set (remains separate)
    test_dataset = datasets.ImageFolder(os.path.join(base_dir, 'test'), transform=val_test_transform)

    # 4. Class Balancing (WeightedRandomSampler)
    # Get labels for the training subset specifically
    train_targets = torch.tensor([targets[i] for i in train_idx])
    class_count = torch.bincount(train_targets)
    class_weights = 1. / class_count.float()
    sample_weights = class_weights[train_targets]

    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    # 5. DataLoaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    BASE_PATH = 'chest_xray'
    # Use 224 for ResNet/DenseNet; 150 for EfficientNetV2-S
    train_ld, val_ld, test_ld = get_dataloaders(BASE_PATH, img_size=224)
    print(f"Successfully loaded {len(train_ld.dataset)} training images with class balancing.")
