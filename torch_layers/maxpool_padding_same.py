import torch
from torch.nn import functional as F
from torch.nn.modules.utils import _pair


class MaxPool2dSamePadding(torch.nn.MaxPool2d):
    """
    A MaxPooling which also accepts 'same' as valid padding to calculate the 
    necessary padding during forward.

    Note
    ----
    Possible loss of throughput if ``padding='same'`` due to the additional 
    padding calculation

    See Also
    --------
    :class:`torch.nn.modules.pooling.MaxPool2d`

    """

    def __init__(self, kernel_size, stride=None, padding=0, dilation=1,
                 return_indices=False, ceil_mode=False):
        """

        Parameters
        ----------
        kernel_size : int or iterable
            the size of the window to take a max over
        stride : int or iterable
            the stride of the window. Default value is same as ``kernel_size``
        padding : int or iterable
            implicit zero padding to be added on both sides
        dilation : int or iterable
            a parameter that controls the stride of elements in the window
        return_indices : bool
            if ``True``, will return the max indices along with the outputs.
            Useful for :class:`torch.nn.MaxUnpool2d` later
        ceil_mode : bool
            when True, will use `ceil` instead of `floor` to compute the output 
            shape

        """

        super().__init__(kernel_size=_pair(kernel_size), stride=_pair(stride),
                         padding=0, dilation=_pair(dilation),
                         return_indices=return_indices, ceil_mode=ceil_mode)

        self.padding = padding

        if self.padding == "same":
            self._new_forward_fn = self._maxpool_same_padding
        else:
            self.padding = _pair(self.padding)
            self._new_forward_fn = super().forward

    def forward(self, input_tensor):
        """
        Performs the actual pooling
        
        Parameters
        ----------
        input_tensor : :class:`torch.Tensor`
            the input tensor to be pooled
        
        Returns
        -------
        :class:`torch.Tensor`
            pooled tensor

        """

        return self._new_forward_fn(input_tensor)

    def _maxpool_same_padding(self, input_tensor):
        """
        Performs the actual pooling for ``padding='same'`` and calculates the 
        necessary padding
        
        Parameters
        ----------
        input_tensor : :class:`torch.Tensor`
            the input tensor to be pooled
        
        Returns
        -------
        :class:`torch.Tensor`
            pooled tensor
            
        """
        input_rows = input_tensor.size(2)
        filter_rows = self.kernel_size[0]

        out_rows = (
            input_rows + self.stride[0] - 1) // self.stride[0]
        padding_rows = max(0,
                           (out_rows - 1)*self.stride[0] +
                           (filter_rows - 1) *
                           self.dilation[0] + 1 - input_rows)

        rows_odd = (padding_rows % 2 != 0)

        input_cols = input_tensor.size(3)
        filter_cols = self.kernel_size[1]

        out_cols = (
            input_cols + self.stride[1] - 1) // self.stride[1]
        padding_cols = max(0, (out_cols - 1) * self.stride[1] +
                           (filter_cols - 1) * self.dilation[1] + 1 -
                           input_cols)
        cols_odd = (padding_cols % 2 != 0)

        if rows_odd or cols_odd:
            input_tensor = F.pad(
                input_tensor, [0, int(cols_odd), 0, int(rows_odd)])

        return F.max_pool2d(input_tensor, self.kernel_size, self.stride,
                            padding=(padding_rows // 2, padding_cols // 2),
                            dilation=self.dilation,
                            return_indices=self.return_indices,
                            ceil_mode=self.ceil_mode)
