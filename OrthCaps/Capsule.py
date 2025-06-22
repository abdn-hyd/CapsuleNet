import torch
import torch.nn as nn
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
            [3, 64, 3, 2, 1],
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
        num_capsules: int = 8,
        in_channels: int = 64,
        out_channels: int = 16,
        kernel_size: int = 9,
        stride: int = 1,
        padding: int = 0,
    ):
        super(PrimaryCaps, self).__init__()
        self.num_capsules = num_capsules
        self.capsules = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    groups=in_channels,
                )
                for _ in range(num_capsules)
            ]
        )
        self.squash = Squash()

    def forward(
        self,
        x: torch.Tensor,
    ):
        """
        for a single capsule, we have: b x 64 x 16 x 16 -> b x 16 x 8 x 8
        by stacking them together, then we we have: b x 8 x 16 x 8 x 8
        """
        u = [capsule(x) for capsule in self.capsules]
        u = torch.stack(u, dim=1)  # shape: b x 8 x 16 x 8 x 8
        return self.squash(u)


class PrunedCaps(nn.Module):
    def __init__(
        self,
        theta: float = 0.7,
    ):
        super(PrunedCaps, self).__init__()
        self.theta = theta

    def forward(
        self,
        u: torch.Tensor,
    ):
        B, n, d, H, W = u.shape
        u_flat = u.view(B, n, -1)
        l2_norm = torch.norm(u_flat, p=2, dim=-1)  # shape: B x n
        sorted_indices = torch.argsort(l2_norm, dim=1)
        u_ordered = torch.gather(
            u_flat, 1, sorted_indices.unsqueeze(-1).expand(-1, -1, u_flat.shape[-1])
        )

        mask = torch.ones((B, n, 1), device=u.device, dtype=torch.bool)

        # calculate the cosine similarity matrix
        u_norm = u_ordered / (l2_norm.gather(1, sorted_indices).unsqueeze(-1) + 1e-8)
        cos_sim = torch.matmul(u_norm, u_norm.transpose(1, 2))

        # filter the mask
        for i in range(n):
            # caution: j>i, index j will have a more important mask
            important_mask = torch.arange(n, device=u.device) > i
            current_sim = cos_sim[:, i, important_mask]

            # purne the capsule if the similarity score is larger than theta
            prune_mask = (current_sim > self.theta).any(dim=1)
            mask[:, i] = mask[:, i] & ~prune_mask.unsqueeze(1)

        u_pruned = u_ordered * mask

        mask_bool = mask.squeeze()
        kept_counts = mask_bool.sum(dim=1)
        n_prime = kept_counts.max().item()

        pruned_caps = []
        for b in range(B):
            kept_indices = mask[b].squeeze().nonzero().squeeze(-1)  # [n_kept,]

            batch_caps = u_ordered[b][kept_indices]  # [n_kept, d*W*H]

            if batch_caps.size(0) < n_prime:
                padding = torch.zeros(
                    n_prime - batch_caps.size(0),
                    batch_caps.size(1),
                    device=u.device,
                )
                batch_caps = torch.cat([batch_caps, padding], dim=0)

            pruned_caps.append(batch_caps)

        # reshape: [B, n', d*W*H] -> [B, n', d, W, H]
        u_pruned = torch.stack(pruned_caps).view(B, n_prime, d, W, H)

        return u_pruned


class OrthCaps_S(nn.Module):
    def __init__(
        self,
    ):
        super(OrthCaps_S, self).__init__()
        self.convLayer = ConvLayer(
            Conv_Cfg=[
                [3, 64, 3, 2, 1],
            ],
        )

        # PrimaryCaps
        self.primaryCaps = PrimaryCaps(
            num_capsules=8,
            in_channels=64,
            out_channels=16,
            kernel_size=9,
            stride=1,
            padding=0,
        )

        # PrunedCaps
        self.prunedCaps = PrunedCaps(theta=0.7)

    def forward(
        self,
        x: torch.Tensor,
    ):
        out = self.convLayer(x)
        out = self.primaryCaps(out)
        out = self.prunedCaps(out)
        return out


if __name__ == "__main__":
    x = torch.randn(2, 3, 32, 32)
    model = OrthCaps_S()
    with torch.no_grad():
        out = model(x)
        print(out.shape)
