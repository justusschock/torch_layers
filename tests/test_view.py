from torch_layers import View

import torch

def test_view():
    input_tensor = torch.rand(10, 3, 2, 1, 20)

    assert View((None, -1))(input_tensor).shape == (10, 3 * 2 * 1 * 20)
    assert View((20, 3, 2, 1, 10))(input_tensor).shape == (20, 3, 2, 1, 10)
    assert View((20, 3, 2, 1, -1))(input_tensor).shape == (20, 3, 2, 1, 10)
