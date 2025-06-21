import os
import struct
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
