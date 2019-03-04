import torch
from torch.nn import functional as F


class Upsample(torch.nn.Module):
    """
    Module to perform Upsampling to a given size.
    Should replace the deprecated ``torch.nn.Upsample2d``

    """

    def __init__(self, target_size, mode='bilinear'):
        """[summary]

        Parameters
        ----------
        target_size : int or tuple
            output spatial size.
        mode : str
            algorithm used for upsampling:
            'nearest' | 'linear' | 'bilinear' | 'trilinear' | 'area'. 
            Default: 'nearest'

        """

        super().__init__()
        self.target_size = target_size
        self.mode = mode

    def forward(self, input_tensor):
        """
        Performs the actual upsampling

        Parameters
        ----------
        input_tensor : :class:`torch.Tensor`
            the input tensor to be upsampled

        Returns
        -------
        :class:`torch.Tensor`
            upsampled tensor

        """

        return F.interpolate(input_tensor, self.target_size, mode=self.mode,
                             align_corners=False)
