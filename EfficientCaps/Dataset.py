import os
import struct
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import Callable, Optional


class LocalMNIST(Dataset):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable[[np.ndarray], torch.Tensor]] = None,
    ):
        """
        Args:
            root (str): path to the root folder
            train (bool): using training dataset or not
            transform (callable): transform function for the images
        """
        self.root = root
        self.transform = transform

        # specify the path
        if train:
            images_file = "train-images.idx3-ubyte"
            labels_file = "train-labels.idx1-ubyte"
        else:
            images_file = "t10k-images.idx3-ubyte"
            labels_file = "t10k-labels.idx1-ubyte"

        # load the images
        with open(os.path.join(root, images_file), "rb") as f:
            _, num, rows, cols = struct.unpack(">IIII", f.read(16))
            self.data = np.frombuffer(f.read(), dtype=np.uint8).reshape(
                num, rows, cols, 1
            )

        # load the labels
        with open(os.path.join(root, labels_file), "rb") as f:
            _, num = struct.unpack(">II", f.read(8))
            self.labels = np.frombuffer(f.read(), dtype=np.uint8)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        image = self.data[idx]  # shape (28, 28, 1)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float()
            image = (image / 255.0 - 0.5) / 0.5  # [0,1] -> [-1,1]

        label = torch.tensor(label, dtype=torch.long)

        return image, label


def get_mnist_dataloader(cfg):
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=cfg.data_mean if cfg.data_mean else [0.5],
                std=cfg.data_std if cfg.data_std else [0.5],
            ),
        ]
    )

    dataset = LocalMNIST(root=cfg.mnist_path, train=cfg.train, transform=transform)

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=True,
        pin_memory=True,
    )

    return dataloader


class LocalCIFAR10(Dataset):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable[[np.ndarray], torch.Tensor]] = None,
    ):
        """
        Args:
            root (str): path to the root folder
            train (bool): using training dataset or not
            transform (callable): transform function for the images
        """
        self.root = root
        self.transform = transform
        self.data = []
        self.labels = []

        if train:
            files = [f"data_batch_{i}" for i in range(1, 6)]
        else:
            files = ["test_batch"]

        for file in files:
            path = os.path.join(root, file)
            with open(path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                self.data.append(entry["data"])
                self.labels.extend(entry["labels"])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose(0, 2, 3, 1)  # (N, H, W, C)

    def __len__(self):
        return len(self.data)

    def __getitem__(
        self,
        idx: int,
    ):
        image = self.data[idx]  # shape (32, 32, 3)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float()
            image = (image / 255.0 - 0.5) / 0.5  # [0,1] -> [-1,1]

        label = torch.tensor(label, dtype=torch.long)

        return image, label


def get_cifar10_dataloader(cfg):
    dataset = LocalCIFAR10(
        root=cfg.cifar10_path,
        train=True,
        transform=transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=cfg.data_mean if cfg.data_mean else [0.4914, 0.4822, 0.4465],
                    std=cfg.data_std if cfg.data_std else [0.2023, 0.1994, 0.2010],
                ),
            ]
        ),
    )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=True,
        pin_memory=True,
    )

    return dataloader
