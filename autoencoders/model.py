"""Neural network modules for the autoencoder model."""

from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F


class GSC(nn.Module):
    """
    Grouped Spatial Convolution (GSC) module.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolutional kernel.
        stride (int): Stride value for the convolution.
        padding (int): Padding value for the convolution.
        num_groups (int): Number of groups for GroupNorm.

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        num_groups: int,
    ):
        super().__init__()
        self.layers = nn.Sequential(
            nn.GroupNorm(num_groups, in_channels),
            nn.SiLU(),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the GSC module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.

        """
        return self.layers(x)


class ResidualBlock(nn.Module):
    """
    Residual block module for a neural network.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        num_groups (int): Number of groups for GroupNorm.
    """

    def __init__(self, in_channels: int, out_channels: int, num_groups: int):
        super().__init__()
        self.main_layers = nn.Sequential(
            GSC(in_channels, out_channels, 3, 1, 1, num_groups),
            GSC(out_channels, out_channels, 3, 1, 1, num_groups),
        )
        if in_channels != out_channels:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual_layer = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the residual block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the residual block.
        """
        return self.main_layers(x) + self.residual_layer(x)


class Downsample(nn.Module):
    """
    Downsample module that performs downsampling on the input tensor.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        num_groups (int): Number of groups for GroupNorm.

    Attributes:
        layers (nn.Sequential): Sequential container for the downsampling layers.

    """

    def __init__(self, in_channels: int, out_channels: int, num_groups: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1),
            ResidualBlock(in_channels, out_channels, num_groups),
            ResidualBlock(out_channels, out_channels, num_groups),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Downsample module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after downsampling.

        """
        return self.layers(x)


class Upsample(nn.Module):
    """
    Upsample module that performs upsampling of the input tensor.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        num_groups (int): Number of groups for the GroupNorm.

    Attributes:
        layers (nn.Sequential): Sequential container for the upsampling layers.

    """

    def __init__(self, in_channels: int, out_channels: int, num_groups: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1),
            ResidualBlock(in_channels, out_channels, num_groups),
            ResidualBlock(out_channels, out_channels, num_groups),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Upsample module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.

        """
        return self.layers(x)


class Encoder(nn.Module):
    def __init__(self, channels: Sequence[int], num_groups: int):
        """
        Encoder module for the autoencoder model.

        Args:
            channels (Sequence[int]): A sequence of integers representing the number of
                tensor channels passing through the encoder. for example if input tensor
                has 3 channels, and the encoder has 3 layers, then a valid sequence
                would be: [3, 64, 128, 256].
            num_groups (int): The number of groups to use for GroupNorm.

        """
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels[0], channels[1], 3, padding=1),
            ResidualBlock(channels[1], channels[1], num_groups),
            *[
                Downsample(channels[i], channels[i + 1], num_groups)
                for i in range(1, len(channels) - 1)
            ],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the encoder module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Encoded output tensor.

        """
        return self.layers(x)


class Decoder(nn.Module):
    """
    Decoder module for an autoencoder.

    Args:
        channels (Sequence[int]): A sequence of integers representing the number of
            tensor channels passing through the decoder. For example if input tensor
            has 3 channels, and the decoder has 3 layers, then a valid sequence
            would be: [256, 128, 64, 3].
        num_groups (int): The number of groups to use for group normalization.

    Attributes:
        layers (nn.Sequential): Sequential container for the decoder layers.

    """

    def __init__(self, channels: Sequence[int], num_groups: int):
        super().__init__()
        self.layers = nn.Sequential(
            *[
                Upsample(channels[i], channels[i + 1], num_groups)
                for i in range(len(channels) - 2)
            ],
            ResidualBlock(channels[-2], channels[-2], num_groups),
            nn.Conv2d(channels[-2], channels[-1], 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the decoder.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.

        """
        return self.layers(x)


class SymmetricalAutoencoder(nn.Module):
    """
    A symmetrical autoencoder model.

    Args:
        channels (Sequence[int]): A sequence of integers representing the number of
            tensor channels passing through the autoencoder. For example if input tensor
            has 3 channels, and the encoder and decoder have 3 layers each, then a valid
            sequence would be: [3, 64, 128, 256].
        num_groups (int): The number of groups to divide the channels into.

    Attributes:
        encoder (Encoder): The encoder module of the autoencoder.
        decoder (Decoder): The decoder module of the autoencoder.
    """

    def __init__(self, channels: Sequence[int], num_groups: int):
        super().__init__()
        self.encoder = Encoder(channels, num_groups)
        self.decoder = Decoder(channels[::-1], num_groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the symmetrical autoencoder.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        x = self.encoder(x)
        x = self.decoder(x)
        return x
