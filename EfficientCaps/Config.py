from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Mnist_Train_Loader_Cfg:
    mnist_path: str = "path/to/MNIST_ORG"
    train: bool = True
    batch_size: int = 128
    num_workers: int = 4
    data_mean: Optional[List[float]] = None
    data_std: Optional[List[float]] = None


@dataclass
class Mnist_Test_Loader_Cfg:
    mnist_path: str = "path/to/MNIST_ORG"
    train: bool = False
    batch_size: int = 128
    num_workers: int = 4
    data_mean: Optional[List[float]] = None
    data_std: Optional[List[float]] = None


@dataclass
class Mnist_Training_Cfg:
    start_epoch: int = 0
    num_epochs: int = 50
    lr: float = 1e-3
    output_dir: str = "path/to/outputs/"
    save_model_epochs: int = 20
    resume: Optional[str] = None
    beta1: float = 0.9
    beta2: float = 0.999
    step_size: int = 10
    gamma: float = 0.1
    num_classes: int = 10


# config for Cifar10
@dataclass
class Cifar10_Model_Cfg:
    Conv_Cfg: List[List[int]] = field(
        default_factory=lambda: [
            [3, 32, 5, 1],
            [32, 64, 3, 1],
            [64, 64, 3, 1],
            [64, 128, 3, 2],
        ]
    )
    in_channels: int = 128
    kernel_size: int = 11
    num_capsules: int = 16
    capsule_dim: int = 8
    stride: int = 1
    in_capsules: List[int] = field(default_factory=lambda: [16, 8])
    out_capsules: List[int] = field(default_factory=lambda: [10, 16])


@dataclass
class Cifar10_Train_Loader_Cfg:
    cifar10_path: str = "path/to/cifar-10-batches-py"
    train: bool = True
    batch_size: int = 128
    num_workers: int = 4
    data_mean: Optional[List[float]] = None
    data_std: Optional[List[float]] = None


@dataclass
class Cifar10_Test_Loader_Cfg:
    cifar10_path: str = "path/to/cifar-10-batches-py"
    train: bool = False
    batch_size: int = 128
    num_workers: int = 4
    data_mean: Optional[List[float]] = None
    data_std: Optional[List[float]] = None


@dataclass
class Cifar10_Training_Cfg:
    start_epoch: int = 0
    num_epochs: int = 50
    lr: float = 5e-4
    output_dir: str = "path/to/outputs/"
    save_model_epochs: int = 20
    resume: Optional[str] = None
    beta1: float = 0.9
    beta2: float = 0.999
    step_size: int = 10
    gamma: float = 0.98
    num_classes: int = 10
