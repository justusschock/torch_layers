from torch_layers import Conv2dWithSamePadding
import torch


def test_same_padding():
    input_tensor_even = torch.rand(1, 3, 12, 12)
    input_tensor_odd = torch.rand(1, 3, 11, 11)

    assert Conv2dWithSamePadding(3, 3, 3, padding=1)(
        input_tensor_even).shape == input_tensor_even.shape

    assert Conv2dWithSamePadding(3, 3, 3, padding=1)(
        input_tensor_odd).shape == input_tensor_odd.shape

    assert Conv2dWithSamePadding(3, 3, 3, padding='same')(
        input_tensor_even).shape == input_tensor_even.shape

    assert Conv2dWithSamePadding(3, 3, 3, padding='same')(
        input_tensor_odd).shape == input_tensor_odd.shape

