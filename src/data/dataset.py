import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_dataloaders(data_root="celebdf_frames", batch_size=32, val_split=0.2, num_workers=4):
    """
    Automatically loads 'real' and 'fake' folders from celebdf_frames/
    and splits into train/val DataLoaders.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    full_dataset = datasets.ImageFolder(root=data_root, transform=transform)

    val_size = int(val_split * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f"[Info] Loaded dataset from {data_root}")
    print(f"       Total: {len(full_dataset)}, Train: {train_size}, Val: {val_size}")
    print(f"       Classes: {full_dataset.classes}")

    return train_loader, val_loader