__version__ = '0.1.2'

from .activation_conv import ActivationConv as ActivationConv2d
from .conv_padding_same import Conv2dWithSamePadding
from .maxpool_padding_same import MaxPool2dSamePadding
from .upsample import Upsample
from .view import View, Flatten

from functools import partial as __partial

Conv2dReLU = __partial(ActivationConv2d, activation="ReLU")
