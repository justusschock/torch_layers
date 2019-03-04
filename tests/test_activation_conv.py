from torch_layers import ActivationConv2d
import torch


def test_activation_conv():
    assert ActivationConv2d(3, 3, 1, 1, padding=2, activation="relu")(
        torch.rand(1, 3, 12, 12)).min() >= 0

    assert ActivationConv2d(3, 3, 1, 1, padding=2, activation="ReLU")(
        torch.rand(1, 3, 12, 12)).min() >= 0

