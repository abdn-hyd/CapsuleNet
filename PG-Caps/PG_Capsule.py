import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


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
        self.modules_list = nn.ModuleList()
        for cfg in Conv_Cfg:
            self.modules_list.append(
                nn.Sequential(
                    *[
                        nn.Conv2d(
                            in_channels=cfg[0],
                            out_channels=cfg[1],
                            kernel_size=cfg[2],
                            stride=cfg[3],
                            padding=cfg[4],
                        ),
                        nn.BatchNorm2d(num_features=cfg[1]),
                    ]
                )
            )

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
        out = None
        for i in range(len(self.modules_list)):
            if i == 0:
                out = F.relu(self.modules_list[i](x))
            else:
                ori_out = out
                out = self.modules_list[i](out)
                if ori_out.shape[1] == out.shape[1]:
                    out += ori_out
                out = F.relu((out))
        return out


class PrimaryCaps(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_caps: int,
        caps_dim: int,
        kernel_size: int,
    ):
        super(PrimaryCaps, self).__init__()
        self.in_channels = in_channels
        self.num_caps = num_caps
        self.caps_dim = caps_dim

        self.pc_layer = nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_caps * caps_dim,
            kernel_size=kernel_size,
            stride=kernel_size,
        )
        self.act = nn.LayerNorm(caps_dim)

    def forward(
        self,
        x: torch.Tensor,
    ):
        u = self.pc_layer(x)
        u = u.view(u.shape[0], self.num_caps, self.caps_dim, u.shape[-2], u.shape[-1])
        u = u.permute(0, 1, 3, 4, 2)
        u = u.reshape(u.shape[0], -1, self.caps_dim)
        out = self.act(u)
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


class MarginalLoss(nn.Module):
    def __init__(
        self,
    ):
        super(MarginalLoss, self).__init__()

    def forward(
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


class PG_Caps_Cifar10(nn.Module):
    def __init__(
        self,
        Conv_Cfgs: List[List[List[int]]] = [
            [
                [3, 64, 5, 1, 2],
                [64, 128, 3, 1, 1],
                [128, 128, 3, 1, 1],
                [128, 256, 3, 1, 1],
            ],
            [
                [3, 64, 5, 2, 2],
                [64, 128, 3, 1, 1],
                [128, 128, 3, 1, 1],
                [128, 256, 3, 1, 1],
            ],
            [
                [3, 64, 5, 4, 2],
                [64, 128, 3, 1, 1],
                [128, 128, 3, 1, 1],
                [128, 256, 3, 1, 1],
            ],
        ],
        PCaps_Cfgs: List[List[int]] = [
            [256, 4, 16, 4],
            [256, 4, 16, 4],
            [256, 4, 16, 4],
        ],
        RCaps1_Cfgs: List[List[int]] = [[336, 16], [168, 24]],
        RCaps2_Cfgs: List[List[int]] = [[168, 24], [10, 32]],
    ):
        super(PG_Caps_Cifar10, self).__init__()

        self.Primary_list = nn.ModuleList()
        for i in range(len(Conv_Cfgs)):
            convLayer = ConvLayer(
                Conv_Cfg=Conv_Cfgs[i],
            )
            primaryCaps = PrimaryCaps(
                in_channels=PCaps_Cfgs[i][0],
                num_caps=PCaps_Cfgs[i][1],
                caps_dim=PCaps_Cfgs[i][2],
                kernel_size=PCaps_Cfgs[i][3],
            )
            self.Primary_list.append(nn.Sequential(*[convLayer, primaryCaps]))

        self.routingCaps1 = RoutingCaps(
            in_capsules=RCaps1_Cfgs[0],
            out_capsules=RCaps1_Cfgs[1],
        )
        self.routingCaps2 = RoutingCaps(
            in_capsules=RCaps2_Cfgs[0],
            out_capsules=RCaps2_Cfgs[1],
        )

        self.MarginalLoss = MarginalLoss()

    def forward(
        self,
        x: torch.Tensor,
    ):
        outs = []
        for i in range(len(self.Primary_list)):
            outs.append(self.Primary_list[i](x))
        out = torch.cat(outs, dim=1)
        out = self.routingCaps1(out)
        out = self.routingCaps2(out)
        return out


if __name__ == "__main__":
    x = torch.randn(2, 3, 32, 32)
    model = PG_Caps_Cifar10()
    with torch.no_grad():
        out = model(x)
        print(out.shape)
