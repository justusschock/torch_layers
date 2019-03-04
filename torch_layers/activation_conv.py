import torch
from .conv_padding_same import Conv2dWithSamePadding


class ActivationConv(torch.nn.Module):
    """
    A combination of convolution with optional same-padding and arbitrary 
    activation

    See Also
    --------
    :class:`Conv2dWithSamePadding`

    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, activation=None,
                 **activation_kwargs):
        """

        Parameters
        ----------
        in_channels : int
            channels of convolution input
        out_channels : int
            number of filters and output channes
        kernel_size : int or Iterable
            specifies the kernel dimensions. 
            If int: same kernel size is used for all dimensions
        stride : int or Iterable, optional
            specifies the convolution strides (default: 1)
            If int: same stride is used for all dimensions
        padding : int or Iterable, optional
            specifies the input padding (default: 0)
            If int: same padding is used for all dimensions
        dilation : int or str or Iterable, optional
            specifies the convolution dilation (default: 1)
            If int: same dilation is used for all dimensions
            If str: only supported string is 'same', which calculates the 
                necessary padding during forward
        groups : int, optional
            number of convolution groups (default: 1)
        bias : bool, optional
            whether to include a bias or not (default: True)
        activation : str, optional
            the activation to apply; must be a valid name of module or function 
            in ``torch.nn`` or ``torch.nn.functional`` (the default is None, 
            which won't apply any activation)

        """

        super().__init__()

        self._conv = Conv2dWithSamePadding(in_channels=in_channels,
                                           out_channels=out_channels,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           padding=padding,
                                           dilation=dilation,
                                           groups=groups,
                                           bias=bias)

        if activation is not None:
            if hasattr(torch.nn, activation):
                self._activation = getattr(
                    torch.nn, activation)(**activation_kwargs)

                self._activation_kwargs = {}

            elif hasattr(torch.nn.functional, activation):
                self._activation = getattr(torch.nn.functional, activation)
                self._activation_kwargs = activation_kwargs

            else:
                raise ValueError(
                    "No activation with name %s found in torch.nn or \
                    torch.nn.functional" % activation)

    def forward(self, input_tensor):
        """
        convolves the input and applies activation afterwards

        Parameters
        ----------
        input_tensor : :class:`torch.Tensor`
            the input tensor to be convolved

        Returns
        -------
        :class:`torch.Tensor`
            tensor after convolution and activation

        """

        return self._activation(self._conv(input_tensor),
                                **self._activation_kwargs)
