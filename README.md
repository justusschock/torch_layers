# Custom PyTorch layers

[![Build Status](https://travis-ci.com/justusschock/torch_layers.svg?branch=master)](https://travis-ci.com/justusschock/torch_layers) [![codecov](https://codecov.io/gh/justusschock/torch_layers/branch/master/graph/badge.svg)](https://codecov.io/gh/justusschock/torch_layers)

This repository implements various layers (either to replace deprecated layers of [`torch`](https://github.com/pytorch/pytorch) or to add useful new features or to encapsulate parts of the functional API into layers).

Currently the following layers are implemented:

* [`Conv2dWithSamePadding`](torch_layers/conv_padding_same) [[docs]](https://justusschock.github.io/torch_layers/_api/_build/torch_layers/conv_padding_same.html)
* [`ActivationConv`](torch_layers/activation_conv) [[docs]](https://justusschock.github.io/torch_layers/_api/_build/torch_layers/activation_conv.html)
* [`MaxPool2dSamePadding`](torch_layers/maxpool_padding_same) [[docs]](https://justusschock.github.io/torch_layers/_api/_build/torch_layers/maxpool_padding_same.html)
* [`Upsample`](torch_layers/upsample) [[docs]](https://justusschock.github.io/torch_layers/_api/_build/torch_layers/upsample.html)
* [`View`](torch_layers/view) [[docs]](https://justusschock.github.io/torch_layers/_api/_build/torch_layers/view.html)
