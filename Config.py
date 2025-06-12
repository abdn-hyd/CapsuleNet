from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Mnist_Capsnet_Cfg:
    # ConvLayer Cfg
    cnn_in_channels: int = 1
    cnn_out_channels: int = 256
    cnn_kernel_size: int = 9
    cnn_stride: int = 1

    # PrimaryCaps Cfg
    pc_capsule_dim: int = 8
    pc_num_routes: int = 32 * 6 * 6
    pc_in_channels: int = 256
    pc_out_channels: int = 32
    pc_kernel_size: int = 9

    # DigitCaps Cfg
    digit_caps: Dict[str, List[int]] = field(
        default_factory=lambda: {
            "num_capsules": [10],
            "num_routes": [32 * 6 * 6],
            "capsule_in_dim": [8],
            "capsule_out_dim": [16],
        }
    )
    num_layers: int = 1


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
