from torch_layers import Upsample

import torch


def test_upsample():
    input_tensor = torch.rand(1, 3, 12, 12)

    assert Upsample((224, 224))(input_tensor).shape == (1, 3, 224, 224)
    try:

        tmp = Upsample((3, 224, 224))(input_tensor)
        tmp = Upsample((5, 224, 224))(input_tensor)

        assert False, "should have raised RuntimeError"

    except RuntimeError:
        assert True
