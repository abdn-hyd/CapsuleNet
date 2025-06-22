import torch
import torch.nn as nn
from typing import List


class ConvLayer(nn.Module):
    def __init__(
        self,
        Conv_Cfg: List[List[int]] = [
            [1, 32, 5, 2, 2],
        ],
    ):
        super(ConvLayer, self).__init__()
        modules_list = []
        for cfg in Conv_Cfg:
            modules_list.append(
                nn.Conv2d(
                    in_channels=cfg[0],
                    out_channels=cfg[1],
                    kernel_size=cfg[2],
                    stride=cfg[3],
                    padding=cfg[4],
                )
            )
            modules_list.append(nn.BatchNorm2d(num_features=cfg[1]))
            modules_list.append(nn.ReLU())

        self.conv_layers = nn.Sequential(*modules_list)

    def init_params(
        self,
    ):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    def forward(
        self,
        x: torch.Tensor,
    ):
        out = self.conv_layers(x)
        return out


class PrimaryCaps(nn.Module):
    def __init__(
        self,
        in_channels: int = 32,
        num_types_capsules: int = 32,
        pose: int = 4,
        kernel_size: int = 1,
        stride: int = 1,
    ):
        super(PrimaryCaps, self).__init__()
        """
        Primary layer will ouput a pose matrix and an activation
        Args:
            in_channels: the number of channels of input Tensor
            num_types_capsules: the number of types of capsules
            pose: the size of pose matrix
            kernel_size: kernel size of convolution
            stride: stride of convolution
        """
        self.pose = nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_types_capsules * pose * pose,
            kernel_size=kernel_size,
            stride=1,
            bias=True,
        )
        self.activation = nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_types_capsules,
            kernel_size=kernel_size,
            stride=stride,
            bias=True,
        )
        self.sigmoid = nn.Sigmoid()

    def init_params(self):
        pass

    def forward(
        self,
        x: torch.Tensor,
    ):
        pose = self.pose(x)
        activation = self.sigmoid(self.activation(x))
        out = torch.cat([pose, activation], dim=1)
        out = out.permute(0, 2, 3, 1)
        return out


class EM_Routing(nn.Module):
    def __init__(
        self,
    ):
        super(EM_Routing, self).__init__()
        self.convLayer = ConvLayer(
            Conv_Cfg=[
                [1, 32, 5, 2, 2],
            ]
        )

        # PrimaryCaps
        self.primaryCaps = PrimaryCaps(
            in_channels=32,
            num_types_capsules=32,
            pose=4,
            kernel_size=1,
            stride=1,
        )

    def forward(
        self,
        x: torch.Tensor,
    ):
        out = self.convLayer(x)
        out = self.primaryCaps(out)
        return out


if __name__ == "__main__":
    x = torch.randn(2, 1, 32, 32)
    model = EM_Routing()
    with torch.no_grad():
        out = model(x)
        print(out.shape)
