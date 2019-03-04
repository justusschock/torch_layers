import torch
from torch.nn import functional as F
from torch.nn.modules.utils import _pair


class Conv2dWithSamePadding(torch.nn.Conv2d):
    """
    A Convolution which also accepts 'same' as valid padding to calculate the 
    necessary padding during forward.

    Note
    ----
    Possible loss of throughput if ``padding='same'`` due to the additional 
    padding calculation

    See Also
    --------
    :class:`torch.nn.modules.conv.Conv2d`

    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
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
        padding : int or Iterable or str, optional
            specifies the input padding (default: 0)
            If int: same padding is used for all dimensions
            If str: only supported string is 'same', which calculates the 
                necessary padding during forward
        dilation : int or Iterable, optional
            specifies the convolution dilation (default: 1)
            If int: same dilation is used for all dimensions
        groups : int, optional
            number of convolution groups (default: 1)
        bias : bool, optional
            whether to include a bias or not (default: True)

        """

        # initialize with padding=0
        super().__init__(in_channels=in_channels, out_channels=out_channels,
                         kernel_size=kernel_size, padding=0, dilation=dilation,
                         groups=groups, bias=bias)

        self.padding = padding

        if self.padding == "same":
            self._forward_fn = self._convolution_same_padding
        else:
            self.padding = _pair(self.padding)
            self._forward_fn = super().forward

    def forward(self, input_tensor: torch.Tensor):
        """
        Convolves the input

        Parameters
        ----------
        input_tensor : :class: `torch.Tensor`
            the input tensor to be convolved

        Returns
        -------
        :class: `torch.Tensor`
            tensor after convolution

        """

        return self._forward_fn(input_tensor)

    def _convolution_same_padding(self, input_tensor):
        """
        Convolves the input and calculates the necessary padding

        Parameters
        ----------
        input_tensor : :class: `torch.Tensor`
            the input tensor to be convolved

        Returns
        -------
        :class: `torch.Tensor`
            tensor after convolution

        """

        # adapted from https://github.com/pytorch/pytorch/issues/3867
        input_rows = input_tensor.size(2)
        filter_rows = self.weight.size(2)

        out_rows = (
            input_rows + self.stride[0] - 1) // self.stride[0]
        padding_rows = max(0,
                           (out_rows - 1)*self.stride[0] +
                           (filter_rows - 1) *
                           self.dilation[0] + 1 - input_rows)

        rows_odd = (padding_rows % 2 != 0)

        input_cols = input_tensor.size(3)
        filter_cols = self.weight.size(3)

        out_cols = (
            input_cols + self.stride[1] - 1) // self.stride[1]
        padding_cols = max(0, (out_cols - 1) * self.stride[1] +
                           (filter_cols - 1) * self.dilation[1] + 1 -
                           input_cols)
        cols_odd = (padding_cols % 2 != 0)

        if rows_odd or cols_odd:
            input_tensor = F.pad(
                input_tensor, [0, int(cols_odd), 0, int(rows_odd)])

        return F.conv2d(input_tensor, self.weight, self.bias,
                        self.stride,
                        padding=(padding_rows // 2, padding_cols // 2),
                        dilation=self.dilation, groups=self.groups)
