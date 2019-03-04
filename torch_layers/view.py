import torch
from copy import deepcopy


class View(torch.nn.Module):
    """
    A module to change a tensor's view

    """

    def __init__(self, view=(None, -1)):
        """

        Parameters
        ----------
        view : tuple, optional
            the new view of the tensor (the first argument can be None, which 
            will be replaced by the current batchsize; also supports a 
            single -1, which specifies a dimension range, which has to be 
            infered during runtime) 
            Default: (None, -1)

        """

        super().__init__()

        self._view = list(view)

    def forward(self, input_tensor):
        """
        Performs the actual change of view

        Parameters
        ----------
        input_tensor : :class:`torch.Tensor`
            tensor whose view will be changed

        Returns
        -------
        :class:`torch.Tensor`
            tensor with changed view

        """

        view = deepcopy(self._view)
        if view[0] is None:
            view[0] = input_tensor.size(0)

        return input_tensor.view(*view)


class Flatten(View):
    """
    A module to flatten all dimensions despite the batch dimension    
    """

    def __init__(self):
        super().__init__(view=[None, -1])
