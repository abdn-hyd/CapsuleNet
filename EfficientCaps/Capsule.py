import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class Squash(nn.Module):
    def __init__(
        self,
        eps: float = 1e-20,
    ):
        """
        A modified squash function which is originated from "Capsule Network on Complex Data"
        """
        super(Squash, self).__init__()
        self.eps = eps

    def forward(
        self,
        x: torch.Tensor,
    ):
        norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        coef = 1 - 1 / (torch.exp(norm) + self.eps)
        unit = x / (norm + self.eps)
        return coef * unit


class ConvLayer(nn.Module):
    def __init__(
        self,
        Conv_Cfg: List[List[int]] = [
            [1, 32, 5, 1],
            [32, 64, 3, 1],
            [64, 64, 3, 1],
            [64, 128, 3, 2],
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
        in_channels: int = 128,
        kernel_size: int = 9,
        num_capsules: int = 16,
        capsule_dim: int = 8,
        stride: int = 1,
    ):
        super(PrimaryCaps, self).__init__()
        self.in_channels = in_channels
        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim
        # kernel_size == H, W, the final output shape be 1 x 1
        self.dw_conv2d = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.num_capsules * self.capsule_dim,
            kernel_size=kernel_size,
            stride=stride,
            groups=in_channels,
        )
        self.squash = Squash()

    def forward(
        self,
        x: torch.Tensor,
    ):
        out = self.dw_conv2d(x)
        out = out.view(-1, self.num_capsules, self.capsule_dim)
        return out


# Self Attention Routing
class RoutingCaps(nn.Module):
    def __init__(
        self,
        in_capsules: List[int] = [16, 8],
        out_capsules: List[int] = [10, 16],
    ):
        """
        Args:
            in_capsules: (Number of Capsules, Capsule Dimension),
            out_capsules: (Number of Capsules, Capsule Dimension),
        """
        super(RoutingCaps, self).__init__()
        self.N0, self.D0 = in_capsules[0], in_capsules[1]
        self.N1, self.D1 = out_capsules[0], out_capsules[1]
        self.squash = Squash()

        # initialize the routing parameters
        self.W = nn.Parameter(torch.randn(self.N1, self.N0, self.D0, self.D1))
        nn.init.kaiming_normal_(self.W)
        self.b = nn.Parameter(torch.zeros(self.N1, self.N0, self.N0))

    def forward(
        self,
        x: torch.Tensor,
    ):
        # sum ji -> j, means project each input capsule to output prediction
        U = torch.einsum("...ij,kijl->...kil", x, self.W)  # shape: B, N1, N0, D1
        U_T = U.permute(0, 1, 3, 2)

        # self attention to produce coupling coefficients
        A = torch.matmul(U, U_T) / torch.sqrt(
            torch.tensor(self.D0).float()
        )  # shape: B, N1, N0, N0
        C = torch.softmax(A, dim=-1) + self.b  # shape: B, N1, N0, N0

        # new capsules
        S = torch.einsum("...kil,...kiz->...kl", U, C)  # shape: B, N1, D1
        S = self.squash(S)
        return S


# fefault config for Mnist
class EfficientCaps(nn.Module):
    def __init__(
        self,
        Conv_Cfg: List[List[int]] = [
            [1, 32, 5, 1],
            [32, 64, 3, 1],
            [64, 64, 3, 1],
            [64, 128, 3, 2],
        ],
        # config of primary capsule layer
        in_channels: int = 128,
        kernel_size: int = 9,
        num_capsules: int = 16,
        capsule_dim: int = 8,
        stride: int = 1,
        #  config of routing capsule layer
        in_capsules: List[int] = [16, 8],
        out_capsules: List[int] = [10, 16],
    ):
        super(EfficientCaps, self).__init__()
        self.convLayer = ConvLayer(
            Conv_Cfg=Conv_Cfg,
        )
        self.primaryCaps = PrimaryCaps(
            in_channels=in_channels,
            kernel_size=kernel_size,
            num_capsules=num_capsules,
            capsule_dim=capsule_dim,
            stride=stride,
        )
        self.routingCaps = RoutingCaps(
            in_capsules=in_capsules,
            out_capsules=out_capsules,
        )

    def forward(
        self,
        x: torch.Tensor,
    ):
        out = self.convLayer(x)
        out = self.primaryCaps(out)
        out = self.routingCaps(out)
        return out

    def MarginalLoss(
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
    ):
        """
        Calculates the Marginal Loss for Capsule Networks.

        This loss function encourages the length of the correct digit's capsule
        to be close to 1 and the length of incorrect digits' capsules to be close to 0.

        Args:
            x (torch.Tensor): The output capsule vectors from the final DigitCap layer.
                              Shape: (batch_size, num_digit_capsules, digit_capsule_dim).
            labels (torch.Tensor): One-hot encoded ground truth labels.
                                   Shape: (batch_size, num_classes).

        Returns:
            torch.Tensor: The calculated marginal loss (scalar).
        """
        batch_size = x.shape[0]
        v_c = torch.sqrt((x**2).sum(dim=2, keepdim=True))
        left = F.relu(0.9 - v_c).view(batch_size, -1)
        right = F.relu(v_c - 0.1).view(batch_size, -1)
        margin_loss = labels * left + 0.5 * (1.0 - labels) * right
        # we sum the loss for the dim = 1 which is the dimension of classes
        margin_loss = margin_loss.sum(dim=1)
        return margin_loss.mean()


if __name__ == "__main__":
    x = torch.randn(2, 1, 28, 28)
    model = EfficientCaps()
    with torch.no_grad():
        out = model(x)
        print(out.shape)
