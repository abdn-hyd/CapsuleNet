import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 64,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
    ):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.relu = nn.ReLU()

    def forward(
        self,
        x: torch.Tensor,
    ):
        return self.relu(self.conv(x))


class PrimaryCaps(nn.Module):
    def __init__(
        self,
        num_capsules: int = 8,
        in_channels: int = 64,
        out_channels: int = 16,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
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
                )
                for _ in range(num_capsules)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
    ):
        """
        for a single capsule, we have: b x 64 x 16 x 16 -> b x 16 x 16 x 16
        by stacking them together, then we we have: b x 8 x 16 x 16 x 16
        """
        u = [capsule(x) for capsule in self.capsules]
        u = torch.stack(u, dim=1)  # shape: b x 8 x 16 x 16 x 16
        return self.squash(u)

    def squash(
        self,
        x: torch.Tensor,
        eps: float = 1e-5,
    ):
        """
        squashing function can shunk short tensor to almost zero and long tensor can be slightly below 1
        """
        squared_norm = (x**2).sum(dim=-1, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        out = scale * x / (torch.sqrt(squared_norm) + eps)
        return out


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
        u_norm = torch.norm(u_flat, p=2, dim=-1)  # shape: B x n
        sorted_u_norm, sorted_indices = torch.sort(u_norm, dim=-1, descending=False)
        u_ordered = torch.gather(
            u_flat, 1, sorted_indices.unsqueeze(-1).expand(-1, -1, u_flat.shape[-1])
        )

        mask = torch.ones((B, n, 1), device=u.device)

        for b_idx in range(B):
            cosine_sim_matrix = F.cosine_similarity(
                u_ordered[b_idx].unsqueeze(1), u_ordered[b_idx].unsqueeze(0), dim=2
            )

            for i in range(n):
                for j in range(i + 1, n):
                    if cosine_sim_matrix[i, j] > self.theta:
                        mask[b_idx, i] = 0

        u_pruned_flat = mask * u_ordered

        output_list = []

        for b_idx in range(B):
            current_mask = mask[b_idx].squeeze()

            filtered_rows = u_pruned_flat[b_idx][current_mask == 1]
            output_list.append(filtered_rows)

        u_pruned = torch.stack(output_list, dim=0).view(B, -1, d, H, W)
        return u_pruned


class OrthCaps_D(nn.Module):
    def __init__(
        self,
    ):
        super(OrthCaps_D, self).__init__()
        self.convLayer = ConvLayer(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=2,
            padding=1,
        )

        # PrimaryCaps
        self.primaryCaps = PrimaryCaps(
            num_capsules=8,
            in_channels=64,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
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
    model = OrthCaps_D()
    with torch.no_grad():
        out = model(x)
        print(out.shape)
