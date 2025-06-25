import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


# class Squash(nn.Module):
#     def __init__(
#         self,
#         eps: float = 1e-20,
#     ):
#         """
#         A modified squash function which is originated from "Capsule Network on Complex Data"
#         """
#         super(Squash, self).__init__()
#         self.eps = eps
#
#     def forward(
#         self,
#         x: torch.Tensor,
#     ):
#         norm = torch.norm(x, p=2, dim=-1, keepdim=True)
#         coef = 1 - 1 / (torch.exp(norm) + self.eps)
#         unit = x / (norm + self.eps)
#         return coef * unit


class Squash(nn.Module):
    def __init__(
        self,
        eps: float = 1e-5,
    ):
        super(Squash, self).__init__()
        self.eps = eps

    def forward(
        self,
        x: torch.Tensor,
    ):
        """
        squashing function can shunk short tensor to almost zero and long tensor can be slightly below 1
        """
        squared_norm = (x**2).sum(dim=-1, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        out = scale * x / (torch.sqrt(squared_norm) + self.eps)
        return out


class ConvLayer(nn.Module):
    def __init__(
        self,
        Conv_Cfg: List[List[int]],
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
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        capsule_dim: int,
        stride: int,
    ):
        super(PrimaryCaps, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.capsule_dim = capsule_dim

        # kernel_size == H, W, the final output shape be 1 x 1
        self.conv2d = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels * capsule_dim,
            kernel_size=kernel_size,
            stride=stride,
        )
        self.squash = Squash()

    def forward(
        self,
        x: torch.Tensor,
    ):
        out = self.conv2d(x)
        B, C, H, W = out.shape
        out = out.view(B, self.out_channels * H * W, self.capsule_dim)
        out = self.squash(out)
        return out


# Self Attention Routing
class RoutingCaps(nn.Module):
    def __init__(
        self,
        in_capsules: List[int],
        out_capsules: List[int],
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


class PG_Caps(nn.Module):
    def __init__(
        self,
        Conv_Cfgs: List[List[List[int]]] = [
            [
                [3, 64, 3, 1, 1],
                [64, 128, 3, 1, 1],
            ],
            [
                [3, 64, 3, 2, 1],
                [64, 128, 3, 1, 1],
            ],
            [
                [3, 64, 3, 4, 1],
                [64, 128, 3, 1, 1],
            ],
        ],
        PCaps_Cfgs: List[List[int]] = [
            [128, 1, 9, 8, 3],
            [128, 1, 5, 8, 3],
            [128, 4, 3, 8, 3],
        ],
        RCaps_Cfg: List[List[int]] = [[96, 8], [10, 16]],
    ):
        super(PG_Caps, self).__init__()

        self.Primary_list = nn.ModuleList()
        for i in range(len(Conv_Cfgs)):
            convLayer = ConvLayer(
                Conv_Cfg=Conv_Cfgs[i],
            )
            primaryCaps = PrimaryCaps(
                in_channels=PCaps_Cfgs[i][0],
                out_channels=PCaps_Cfgs[i][1],
                kernel_size=PCaps_Cfgs[i][2],
                capsule_dim=PCaps_Cfgs[i][3],
                stride=PCaps_Cfgs[i][4],
            )
            self.Primary_list.append(nn.Sequential(*[convLayer, primaryCaps]))

        self.routingCaps = RoutingCaps(
            in_capsules=RCaps_Cfg[0],
            out_capsules=RCaps_Cfg[1],
        )

    def forward(
        self,
        x: torch.Tensor,
    ):
        outs = []
        for i in range(len(self.Primary_list)):
            outs.append(self.Primary_list[i](x))

        out = torch.cat(outs, dim=1)
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
        B = x.shape[0]
        v_c = torch.sqrt((x**2).sum(dim=2, keepdim=True))
        left = F.relu(0.9 - v_c).view(B, -1)
        right = F.relu(v_c - 0.1).view(B, -1)
        margin_loss = labels * left + 0.5 * (1.0 - labels) * right
        # we sum the loss for the dim = 1 which is the dimension of classes
        margin_loss = margin_loss.sum(dim=1)
        return margin_loss.mean()


if __name__ == "__main__":
    x = torch.randn(2, 3, 32, 32)
    model = PG_Caps()
    with torch.no_grad():
        out = model(x)
        print(out.shape)
