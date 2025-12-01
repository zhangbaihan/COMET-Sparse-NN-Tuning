import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import scipy.io
import torch.utils.data as data

root = 'data'

def get_data_loaders(dataset_name, batch_size, data_augmentation=False):
    """
    Returns train, validation, and test DataLoader objects along with number of classes for the specified dataset.
    
    Args:
        dataset_name (str): Name of the dataset ('cifar10', 'cifar100', 'svhn', 'tiny_imagenet', 'SARCOS').
        batch_size (int): Batch size for the loaders.
        data_augmentation (bool): Whether to apply data augmentation to training set (only for CIFAR10).
    
    Returns:
        train_loader, val_loader, test_loader, num_classes
    """
    
    if dataset_name == 'cifar10':
        # Define transforms
        if not data_augmentation:
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            print('Applying data augmentation for CIFAR10 training set...')
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.Resize(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # Load full training dataset and split train/validation
        full_train_dataset = datasets.CIFAR10(root=root, train=True, transform=train_transform, download=True)
        train_size = int(len(full_train_dataset) * 0.8)
        val_size = len(full_train_dataset) - train_size
        train_set, val_set = torch.utils.data.random_split(full_train_dataset, [train_size, val_size])

        test_set = datasets.CIFAR10(root=root, train=False, transform=test_transform, download=True)
        num_classes = 10

    elif dataset_name == 'cifar100':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])

        full_train_dataset = datasets.CIFAR100(root=root, train=True, transform=transform, download=True)
        train_size = int(len(full_train_dataset) * 0.8)
        val_size = len(full_train_dataset) - train_size
        train_set, val_set = torch.utils.data.random_split(full_train_dataset, [train_size, val_size])

        test_set = datasets.CIFAR100(root=root, train=False, transform=transform, download=True)
        num_classes = 100

    elif dataset_name == 'svhn':
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.Resize(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
        ])

        test_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
        ])

        full_train_dataset = datasets.SVHN(root=root, split='train', transform=train_transform, download=True)
        train_size = int(len(full_train_dataset) * 0.8)
        val_size = len(full_train_dataset) - train_size
        train_set, val_set = torch.utils.data.random_split(full_train_dataset, [train_size, val_size])

        test_set = datasets.SVHN(root=root, split='test', transform=test_transform, download=True)
        num_classes = 10

    elif dataset_name == 'tiny_imagenet':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

        val_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

        image_datasets = {
            "train": datasets.ImageFolder(f'{root}/tiny-imagenet-200/train', transform=train_transform),
            "val": datasets.ImageFolder(f'{root}/tiny-imagenet-200/val', transform=val_transform),
            "test": datasets.ImageFolder(f'{root}/tiny-imagenet-200/test', transform=test_transform)
        }

        train_loader = DataLoader(image_datasets["train"], batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(image_datasets["val"], batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(image_datasets["test"], batch_size=batch_size, shuffle=False)
        num_classes = 200

        return train_loader, val_loader, test_loader, num_classes

    elif dataset_name == 'SARCOS':
        # Load regression dataset from .mat files
        train_data = scipy.io.loadmat(f'{root}/SARCOS/sarcos_inv.mat')['sarcos_inv']
        test_data = scipy.io.loadmat(f'{root}/SARCOS/sarcos_inv_test.mat')['sarcos_inv_test']

        train_dataset = TensorDataset(
            torch.from_numpy(train_data[:, :21]).float(),
            torch.from_numpy(train_data[:, 22]).float().unsqueeze(1)
        )
        test_dataset = TensorDataset(
            torch.from_numpy(test_data[:, :21]).float(),
            torch.from_numpy(test_data[:, 22]).float().unsqueeze(1)
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        test_loader = None
        num_classes = 1

        return train_loader, val_loader, test_loader, num_classes

    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")

    # For CIFAR and SVHN datasets, create loaders after splits
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, num_classes
