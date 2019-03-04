from torch_layers import MaxPool2dSamePadding
import torch


def test_same_padding():
    input_tensor_even = torch.rand(1, 3, 12, 12)
    input_tensor_odd = torch.rand(1, 3, 11, 11)

    assert MaxPool2dSamePadding(3, 1, padding=1)(
        input_tensor_even).shape == input_tensor_even.shape

    assert MaxPool2dSamePadding(3, 1, padding=1)(
        input_tensor_odd).shape == input_tensor_odd.shape

    assert MaxPool2dSamePadding(3, 1, padding='same')(
        input_tensor_even).shape == input_tensor_even.shape

    assert MaxPool2dSamePadding(3, 1, padding='same')(
        input_tensor_odd).shape == input_tensor_odd.shape
