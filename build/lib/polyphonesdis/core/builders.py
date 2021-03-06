#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Model and loss construction functions."""

from polyphonesdis.core.config import cfg
from polyphonesdis.core.net import SoftCrossEntropyLoss
# from polyphonesdis.models.char_word2vec import CHARW2VNet

from torch.nn import MSELoss
from torch.nn import BCELoss


# Supported models
# _models = {"charw2c": CHARW2VNet}

# Supported loss functions
_loss_funs = {
    "cross_entropy": SoftCrossEntropyLoss,
    "mse_loss": MSELoss,
    "bce_loss": BCELoss}


def get_model():
    """Gets the model class specified in the config."""
    err_str = "Model type '{}' not supported"
    assert cfg.MODEL.TYPE in _models.keys(), err_str.format(cfg.MODEL.TYPE)
    return _models[cfg.MODEL.TYPE]


def get_loss_fun(loss=None):
    """Gets the loss function class specified in the config."""
    if loss is None:
        err_str = "Loss function type '{}' not supported"
        assert cfg.MODEL.LOSS_FUN in _loss_funs.keys(), err_str.format(cfg.TRAIN.LOSS)
        return _loss_funs[cfg.MODEL.LOSS_FUN]
    else:
        err_str = "Loss function type '{}' not supported"
        assert loss in _loss_funs.keys(), err_str.format(cfg.TRAIN.LOSS)
        return _loss_funs[loss]


def build_model():
    """Builds the model."""
    return get_model()()


def build_loss_fun(loss=None):
    """Build the loss function."""
    return get_loss_fun(loss)()


def register_model(name, ctor):
    """Registers a model dynamically."""
    _models[name] = ctor


def register_loss_fun(name, ctor):
    """Registers a loss function dynamically."""
    _loss_funs[name] = ctor
