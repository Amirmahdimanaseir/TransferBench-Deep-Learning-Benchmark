"""
TransferBench
=============

Dataset Utilities

Supported Datasets

- CIFAR10
- CIFAR100

Generate

- Train Loader
- Validation Loader
"""

from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader


# -----------------------------------------------------
# Transforms
# -----------------------------------------------------

def build_transforms(image_size=224):

    train_transform = transforms.Compose([

        transforms.Resize((image_size, image_size)),

        transforms.RandomHorizontalFlip(),

        transforms.RandomCrop(
            image_size,
            padding=4,
        ),

        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
        ),

        transforms.ToTensor(),

        transforms.Normalize(

            mean=[0.485, 0.456, 0.406],

            std=[0.229, 0.224, 0.225],

        ),

    ])

    test_transform = transforms.Compose([

        transforms.Resize((image_size, image_size)),

        transforms.ToTensor(),

        transforms.Normalize(

            mean=[0.485, 0.456, 0.406],

            std=[0.229, 0.224, 0.225],

        ),

    ])

    return train_transform, test_transform


# -----------------------------------------------------
# CIFAR10
# -----------------------------------------------------

def load_cifar10(
    root="data",
    batch_size=32,
    image_size=224,
):

    train_transform, test_transform = build_transforms(
        image_size
    )

    train_dataset = datasets.CIFAR10(

        root=root,

        train=True,

        download=True,

        transform=train_transform,

    )

    test_dataset = datasets.CIFAR10(

        root=root,

        train=False,

        download=True,

        transform=test_transform,

    )

    train_loader = DataLoader(

        train_dataset,

        batch_size=batch_size,

        shuffle=True,

        num_workers=2,

        pin_memory=True,

    )

    test_loader = DataLoader(

        test_dataset,

        batch_size=batch_size,

        shuffle=False,

        num_workers=2,

        pin_memory=True,

    )

    return (

        train_loader,

        test_loader,

        train_dataset.classes,

    )


# -----------------------------------------------------
# CIFAR100
# -----------------------------------------------------

def load_cifar100(
    root="data",
    batch_size=32,
    image_size=224,
):

    train_transform, test_transform = build_transforms(
        image_size
    )

    train_dataset = datasets.CIFAR100(

        root=root,

        train=True,

        download=True,

        transform=train_transform,

    )

    test_dataset = datasets.CIFAR100(

        root=root,

        train=False,

        download=True,

        transform=test_transform,

    )

    train_loader = DataLoader(

        train_dataset,

        batch_size=batch_size,

        shuffle=True,

        num_workers=2,

        pin_memory=True,

    )

    test_loader = DataLoader(

        test_dataset,

        batch_size=batch_size,

        shuffle=False,

        num_workers=2,

        pin_memory=True,

    )

    return (

        train_loader,

        test_loader,

        train_dataset.classes,

    )


# -----------------------------------------------------
# Factory
# -----------------------------------------------------

def load_dataset(

    dataset_name="cifar10",

    batch_size=32,

    image_size=224,

):

    dataset_name = dataset_name.lower()

    if dataset_name == "cifar10":

        return load_cifar10(

            batch_size=batch_size,

            image_size=image_size,

        )

    elif dataset_name == "cifar100":

        return load_cifar100(

            batch_size=batch_size,

            image_size=image_size,

        )

    else:

        raise ValueError(

            f"Unknown dataset: {dataset_name}"

        )


# -----------------------------------------------------
# Dataset Information
# -----------------------------------------------------

def dataset_summary(
    train_loader,
    test_loader,
    classes,
):

    print("=" * 70)

    print("Dataset Summary")

    print("=" * 70)

    print(f"Classes        : {len(classes)}")

    print(f"Train Batches  : {len(train_loader)}")

    print(f"Test Batches   : {len(test_loader)}")

    print(f"Class Names    : {classes}")

    print("=" * 70)