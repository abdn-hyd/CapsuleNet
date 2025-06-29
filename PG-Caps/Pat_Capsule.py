from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


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


class EncoderBlock(nn.Module):
    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        attn_mask: torch.Tensor = None,
    ):
        super().__init__()
        self.num_heads = num_heads

        # Attention block
        self.ln_1 = nn.LayerNorm(hidden_dim)
        # batch_first = True, no need to permute the x
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=attention_dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(hidden_dim, mlp_dim)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(mlp_dim, hidden_dim)),
                ]
            )
        )
        self.attn_mask = attn_mask

    def attention_mask(self, x: torch.Tensor):
        if self.attn_mask is not None:
            return self.attn_mask.to(dtype=x.dtype, device=x.device)
        else:
            return self.attn_mask

    def forward(self, input: torch.Tensor):
        torch._assert(
            input.dim() == 3,
            f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}",
        )
        x = self.ln_1(input)
        x, _ = self.attn(x, x, x, need_weights=True, attn_mask=self.attention_mask(x))
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y


class Encoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        attn_mask: torch.Tensor = None,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.attn_mask = attn_mask
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = EncoderBlock(
                num_heads, hidden_dim, mlp_dim, dropout, attention_dropout, attn_mask
            )
        self.layers = nn.Sequential(layers)

    def forward(self, x: torch.Tensor):
        return self.layers(x)


class Patchify(nn.Module):
    def __init__(
        self,
        num_patches: int,
        in_channels: int,
        hidden_dim: int,
        patch_size: int,
        num_layers: int,
        num_heads: int,
        dropout: float = 0.9,
        attention_dropout: float = 0.9,
        attn_mask: torch.Tensor = None,
    ):
        super(Patchify, self).__init__()
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.conv_proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        scale = hidden_dim**-0.5
        self.pos_embedding = nn.Parameter(scale * torch.randn(num_patches, hidden_dim))
        self.ln_1 = nn.LayerNorm(hidden_dim)

        # encoder
        self.encoder = Encoder(
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            mlp_dim=hidden_dim * 4,
            dropout=dropout,
            attention_dropout=attention_dropout,
        )

        self.ln_2 = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
    ):
        n, c, h, w = x.shape
        p = self.patch_size

        n_h = h // p
        n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.conv_proj(x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, n_h * n_w)
        x = x.reshape(n, self.hidden_dim, n_h * n_w)
        x = x.permute(0, 2, 1)

        # add position embedding
        x = x + self.pos_embedding.to(x.dtype)
        x = self.ln_1(x)

        # Encoder
        x = self.encoder(x)
        x = self.ln_2(x)
        return x


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


class PatchCaps(nn.Module):
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
            [
                [3, 64, 5, 8, 2],
                [64, 128, 3, 1, 1],
                [128, 128, 3, 1, 1],
                [128, 256, 3, 1, 1],
            ],
        ],
        Patch_Cfgs: List[List[int]] = [
            [64, 256, 48, 4, 2, 12],
            [16, 256, 48, 4, 2, 12],
            [4, 256, 48, 4, 2, 12],
            [1, 256, 48, 4, 2, 12],
        ],
        RCaps1_Cfgs: List[List[int]] = [
            [64, 48],
            [32, 48],
        ],
        RCaps2_Cfgs: List[List[int]] = [
            [16, 48],
            [8, 48],
        ],
        RCaps3_Cfgs: List[List[int]] = [
            [4, 48],
            [2, 48],
        ],
        RCaps4_Cfgs: List[List[int]] = [
            [1, 48],
            [1, 48],
        ],
        RCaps5_Cfgs: List[List[int]] = [
            [43, 48],
            [10, 48],
        ],
    ):
        super(PatchCaps, self).__init__()
        self.Patch_list = nn.ModuleList()
        for i in range(len(Patch_Cfgs)):
            Modules = []
            Modules.append(ConvLayer(Conv_Cfg=Conv_Cfgs[i]))
            Modules.append(
                Patchify(
                    num_patches=Patch_Cfgs[i][0],
                    in_channels=Patch_Cfgs[i][1],
                    hidden_dim=Patch_Cfgs[i][2],
                    patch_size=Patch_Cfgs[i][3],
                    num_layers=Patch_Cfgs[i][4],
                    num_heads=Patch_Cfgs[i][5],
                )
            )
            self.Patch_list.append(nn.Sequential(*Modules))
        self.RCaps1 = RoutingCaps(
            in_capsules=RCaps1_Cfgs[0],
            out_capsules=RCaps1_Cfgs[1],
        )
        self.RCaps2 = RoutingCaps(
            in_capsules=RCaps2_Cfgs[0],
            out_capsules=RCaps2_Cfgs[1],
        )
        self.RCaps3 = RoutingCaps(
            in_capsules=RCaps3_Cfgs[0],
            out_capsules=RCaps3_Cfgs[1],
        )
        self.RCaps4 = RoutingCaps(
            in_capsules=RCaps4_Cfgs[0],
            out_capsules=RCaps4_Cfgs[1],
        )
        self.RCaps5 = RoutingCaps(
            in_capsules=RCaps5_Cfgs[0],
            out_capsules=RCaps5_Cfgs[1],
        )

        self.MarginalLoss = MarginalLoss()

    def forward(
        self,
        x: torch.Tensor,
    ):
        outs = []
        for i in range(len(self.Patch_list)):
            outs.append(self.Patch_list[i](x))

        # multi-scale routing
        out1 = self.RCaps1(outs[0])
        out2 = self.RCaps2(outs[1])
        out3 = self.RCaps3(outs[2])
        out4 = self.RCaps4(outs[3])
        out5 = torch.cat([out1, out2, out3, out4], dim=1)
        out5 = self.RCaps5(out5)
        return out5


if __name__ == "__main__":
    x = torch.randn(2, 3, 32, 32)
    model = PatchCaps()
    with torch.no_grad():
        out = model(x)
        print(out.shape)
