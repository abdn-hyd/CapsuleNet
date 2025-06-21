import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# default config for mnist datatset
class ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 256,
        kernel_size: int = 9,
        stride: int = 1,
    ):
        """
        Args:
            in_channels (int): Number of channels in the input image.
                               For MNIST, this is 1 (grayscale).
            out_channels (int): Number of channels produced by the convolution.
                                This will be the input channels for the PrimaryCaps layer.
            kernel_size (int): Size of the convolutional kernel.
            stride (int): Stride of the convolution.
        """
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
        )
        self.relu = nn.ReLU()

    def forward(
        self,
        x: torch.Tensor,
    ):
        return self.relu(self.conv(x))


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


# default config for mnist datatset
class PrimaryCaps(nn.Module):
    def __init__(
        self,
        capsule_dim: int = 8,
        num_capsules: int = 32 * 6 * 6,
        in_channels: int = 256,
        out_channels: int = 32,
        kernel_size: int = 9,
    ):
        """
        Args:
            capsule_dim (int): the dimension of a single final capsule
            num_capsules (int): the number of capsules
            in_channels (int): refer to the the number of channels of input
            out_channels (int): refer to the number of channels of ouput after a single conv layer
            kernel_size (int): the size of kernel for each different conv
        """
        super(PrimaryCaps, self).__init__()
        self.num_capsules = num_capsules
        self.capsules = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=0,
                )
                for _ in range(capsule_dim)
            ]
        )
        self.sqush = Squash()

    def forward(
        self,
        x: torch.Tensor,
    ):
        """
        for a single capsule, we have: b x 256 x 20 x 20 -> b x 32 x 6 x 6
        by stacking them together, then we we have: b x 8 x 32 x 6 x 6
        """
        u = [capsule(x) for capsule in self.capsules]
        u = torch.stack(u, dim=1)  # shape: b x 8 x 32 x 6 x 6
        u = u.view(x.shape[0], self.num_capsules, -1)  # shape: b x (32 x 6 x 6) x 8
        return self.squash(u)


class DigitCap(nn.Module):
    def __init__(
        self,
        num_routes: int,
        num_capsules: int,
        capsule_in_dim: int,
        capsule_out_dim: int,
        num_iterations: int = 3,
    ):
        """
        Args:
            num_routes (int): the number of capsules of previous layer
            num_capsules (int): the number of capsules of current layer
            capsule_in_dim (int): the capsule dimension of previous layer
            capsule_out_dim (int): the capsule dimension of current layer
            num_iterations (int): the number of iterations of dynamic routing
        """
        super(DigitCap, self).__init__()
        self.num_routes = num_routes
        self.num_capsules = num_capsules
        self.num_iterations = num_iterations
        self.W = nn.Parameter(
            torch.randn(1, num_routes, num_capsules, capsule_out_dim, capsule_in_dim)
        )
        self.squash = Squash()

    def forward(
        self,
        u: torch.Tensor,
    ):
        # shape: b x num_routes x capsule_in_dim -> b x num_routes x num_capsules x capsule_in_dim x 1
        u = torch.stack([u for _ in range(self.num_capsules)], dim=2).unsqueeze(-1)

        # calculating u_hat = W*u
        # shape: batch_size x num_routes x num_capsules x capsule_out_dim x 1
        u_hat = torch.matmul(self.W, u)

        # setting them all to 0, initially
        b_ij = torch.zeros(1, self.num_routes, self.num_capsules, 1)

        # moving b_ij to GPU, if available
        b_ij = b_ij.to(DEVICE)

        # update coupling coefficients and calculate v_j
        # shape: batch_size x 1 x num_capsules x capsule_out_dim x 1
        v_j = self.dynamic_routing(b_ij, u_hat)

        # shape: batch_size x num_capsules x capsule_out_dim
        return v_j.view(v_j.shape[0], v_j.shape[2], -1)

    def dynamic_routing(
        self,
        b_ij: torch.Tensor,
        u_hat: torch.Tensor,
    ):
        """
        This iterative process refines the agreement between lower-level
        and higher-level capsules by updating routing coefficients.

        Args:
            b_ij (torch.Tensor): Raw routing logits, initialized to zeros.
                                 Shape: (num_capsules_current, batch_size, num_routes_previous, 1, capsule_out_dim)
            u_hat (torch.Tensor): Prediction vectors from lower-level capsules.
                                  Shape: (num_capsules_current, batch_size, num_routes_previous, 1, capsule_out_dim)

        Returns:
            torch.Tensor: The final squashed capsule vectors for the current layer.
                          Shape: (num_capsules_current, batch_size, 1, 1, capsule_out_dim)
        """
        # update b_ij, c_ij for number of routing iterations
        for iteration in range(self.num_iterations):
            # softmax calculation of coupling coefficients, c_ij
            # dim = 1 refers to the dimension of num_routes, which we use softmax to calculate the probability of previous capsule layer
            c_ij = F.softmax(b_ij, dim=1).unsqueeze(-1)

            # calculating the total inputs using dot product, s_j = sum(c_ij*u_hat)
            # shape: batch_size x 1 x num_capsules x capsule_out_dim x 1
            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)

            # squashing to get a normalized vector output, v_j
            v_j = self.squash(s_j)

            # if not on the last iteration, calculate agreement and new b_ij
            if iteration < self.num_iterations - 1:
                # agreement, basically we calculate the similarity between the prediction tensor(u_hat) and output tensors(v_j)
                # shape: batch_size x num_routes x num_capsules x 1 x 1
                a_ij = (u_hat.transpose(-2, -1) * v_j).sum(dim=-1, keepdim=True)

                # new b_ij
                # shape: batch_size x num_routes x num_capsules x capsule_out_dim x 1
                b_ij = b_ij + a_ij.squeeze(-1).mean(dim=0, keepdim=True)

        return v_j


class CapsNet(nn.Module):
    def __init__(
        self,
        cfg: dataclass,
    ):
        super(CapsNet, self).__init__()

        # ConvLater setup
        self.ConvLayer = ConvLayer(
            cfg.cnn_in_channels,
            cfg.cnn_out_channels,
            cfg.cnn_kernel_size,
            cfg.cnn_stride,
        )

        # PrimaryCaps setup
        self.PrimaryCap = PrimaryCaps(
            cfg.pc_capsule_dim,
            cfg.pc_num_routes,
            cfg.pc_in_channels,
            cfg.pc_out_channels,
        )

        DigitCap_layers = []
        for i in range(cfg.num_layers):
            layer = DigitCap(
                cfg.digit_caps["num_routes"][i],
                cfg.digit_caps["num_capsules"][i],
                cfg.digit_caps["capsule_in_dim"][i],
                cfg.digit_caps["capsule_out_dim"][i],
            )
            DigitCap_layers.append(layer)
        self.DigitCap = nn.Sequential(*DigitCap_layers)

    def forward(
        self,
        x: torch.Tensor,
    ):
        out = self.ConvLayer(x)
        out = self.PrimaryCap(out)
        out = self.DigitCap(out)

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
