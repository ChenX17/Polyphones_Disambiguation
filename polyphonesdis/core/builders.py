#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Model and loss construction functions."""

from polyphonesdis.core.config import cfg
from polyphonesdis.core.net import SoftCrossEntropyLoss
from polyphonesdis.models.char_word2vec import CHARW2VNet
from polyphonesdis.models.char_word2vec_pos import CHARW2VPOSNet
from polyphonesdis.models.char_word2vec_cws import CHARW2VCWSNet
from polyphonesdis.models.char_word2vec_flag import CHARW2VFLAGNet
from polyphonesdis.models.char_word2vec_pos_cws import CHARW2VCWSPOSNet

from torch.nn import MSELoss
from torch.nn import BCELoss
from torch.nn import CrossEntropyLoss


# Supported models
_models = {"charw2c": CHARW2VNet, "charw2cpos": CHARW2VPOSNet, "charw2ccws": CHARW2VCWSNet, "charw2cflag":CHARW2VFLAGNet, "charw2cposcws": CHARW2VCWSPOSNet}

# Supported loss functions
_loss_funs = {
    "cross_entropy": CrossEntropyLoss,
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
    return get_model()(cfg)


def build_loss_fun(loss=None, reduction='mean'):
    """Build the loss function."""
    return get_loss_fun(loss)(reduction=reduction)


def register_model(name, ctor):
    """Registers a model dynamically."""
    _models[name] = ctor


def register_loss_fun(name, ctor):
    """Registers a loss function dynamically."""
    _loss_funs[name] = ctor
