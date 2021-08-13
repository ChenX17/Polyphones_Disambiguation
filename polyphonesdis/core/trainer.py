#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tools for training and testing a model."""

import random
from copy import deepcopy
from torch.nn import functional as F

import numpy as np
import polyphonesdis.core.benchmark as benchmark
import polyphonesdis.core.builders as builders
import polyphonesdis.core.checkpoint as cp
import polyphonesdis.core.config as config
import polyphonesdis.core.distributed as dist
import polyphonesdis.core.logging as logging
import polyphonesdis.core.meters as meters
import polyphonesdis.core.net as net
import polyphonesdis.core.optimizer as optim
import polyphonesdis.datasets.loader as data_loader
import torch
from polyphonesdis.core.config import cfg
from polyphonesdis.core.io import pathmgr
from polyphonesdis.core.meters import acc
from tensorboardX import SummaryWriter

# writer = SummaryWriter(cfg.OUT_DIR)
logger = logging.get_logger(__name__)


def setup_env():
    """Sets up environment for training or testing."""
    if dist.is_master_proc():
        # Ensure that the output dir exists
        pathmgr.mkdirs(cfg.OUT_DIR)
        # Save the config
        config.dump_cfg()
        # Set the writer
        # writer = SummaryWriter(cfg.OUT_DIR)

    # Setup logging
    logging.setup_logging()
    # Log torch, cuda, and cudnn versions
    version = [torch.__version__, torch.version.cuda, torch.backends.cudnn.version()]
    logger.info("PyTorch Version: torch={}, cuda={}, cudnn={}".format(*version))
    # Log the config as both human readable and as a json
    logger.info("Config:\n{}".format(cfg)) if cfg.VERBOSE else ()
    logger.info(logging.dump_log_data(cfg, "cfg", None))
    # Fix the RNG seeds (see RNG comment in core/config.py for discussion)
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    random.seed(cfg.RNG_SEED)
    # Configure the CUDNN backend
    torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK


def setup_model():
    """Sets up a model for training or testing and log the results."""
    # Build the model
    model = builders.build_model()
    logger.info("Model:\n{}".format(model)) if cfg.VERBOSE else ()
    # Log model complexity
    # logger.info(logging.dump_log_data(net.complexity(model), "complexity"))
    # Transfer the model to the current GPU device
    err_str = "Cannot use more GPU devices than available"
    assert cfg.NUM_GPUS <= torch.cuda.device_count(), err_str
    cur_device = torch.cuda.current_device()
    model = model.cuda(device=cur_device)
    # Use multi-process data parallel model in the multi-gpu setting
    if cfg.NUM_GPUS > 1:
        # Make model replica operate on the current device
        ddp = torch.nn.parallel.DistributedDataParallel
        model = ddp(module=model, device_ids=[cur_device], output_device=cur_device)
    return model


def comp_loss(preds, labels, seq_lens, loss_fun):
    loss = loss_fun(preds.transpose(1, 2), labels)
    mask = labels > 1
    loss = (loss * mask.float()).sum()/mask.sum().item()
    return loss


def train_epoch(loader, model, ema, loss_fun, optimizer, meter, cur_epoch, writer):
    """Performs one epoch of training."""
    # Shuffle the data
    data_loader.shuffle(loader, cur_epoch)
    # Update the learning rate
    lr = optim.get_epoch_lr(cur_epoch)
    optim.set_lr(optimizer, lr)
    # Enable training mode
    model.train()
    ema.train()
    meter.reset()
    meter.iter_tic()
    total_poly = 0
    total_correct_poly = 0
    total_loss = 0.0
    for cur_iter, (inputs, labels, seq_lens) in enumerate(loader):
        # Transfer the data to the current GPU device
        for key in inputs.keys():
            inputs[key] = inputs[key].cuda()
        labels = labels.cuda()
        preds = model(inputs, labels, seq_lens)

        loss = comp_loss(preds, labels, seq_lens, loss_fun)
        # Perform the backward pass and update the parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size, max_seq_len = inputs['char'].size()
        mask = (torch.arange(max_seq_len).expand(
                batch_size, max_seq_len) < seq_lens.unsqueeze(1)).cuda()
        accuracy, poly, correct_poly = acc(mask, labels, preds)
        total_poly += poly
        total_correct_poly += correct_poly
        total_loss += loss.item()

        loss  = dist.scaled_all_reduce([loss])[0]
        total_acc = float(total_correct_poly)/float(total_poly)
        if (cur_iter+1) % 500 == 0:
            writer.add_scalar('train_acc', total_acc, cur_iter+cur_epoch*len(loader))
            writer.add_scalar('train_loss', total_loss/cur_iter, cur_iter+cur_epoch*len(loader))
        # Copy the stats from GPU to CPU (sync point)
        loss = loss.item()
        meter.iter_toc()
        # Update and log stats
        mb_size = inputs['mask'].size(0) * cfg.NUM_GPUS
        meter.update_stats(loss, lr, accuracy, mb_size)
        meter.log_iter_stats(cur_epoch, cur_iter)
        meter.iter_tic()
    # Log epoch stats
    meter.log_epoch_stats(cur_epoch)


@torch.no_grad()
def test_epoch(loader, model, meter, cur_epoch, loss_fun, writer=None):
    """Evaluates the model on the test set."""
    # Enable eval mode
    model.eval()
    meter.reset()
    meter.iter_tic()
    total_acc = 0.0
    total_poly = 0
    total_correct_poly = 0
    rec = 0.0
    total_loss = 0.0
    for cur_iter, (inputs, labels, seq_lens) in enumerate(loader):
        # Transfer the data to the current GPU device
        for key in inputs.keys():
            inputs[key] = inputs[key].cuda()
        labels = labels.cuda()

        # Compute the predictions
        preds = model(inputs, labels, seq_lens)
        loss = comp_loss(preds, labels, seq_lens, loss_fun)
        # Compute the errors

        batch_size, max_seq_len = inputs['char'].size()
        mask = (torch.arange(max_seq_len).expand(
                batch_size, max_seq_len) < seq_lens.unsqueeze(1)).cuda()

        accuracy, poly, correct_poly = acc(mask, labels, preds)
        meter.update_stats(loss, accuracy, batch_size)
        total_poly += poly
        total_correct_poly += correct_poly
        total_loss += loss.item()
    if writer:
        writer.add_scalar('val_acc', float(total_correct_poly)/float(total_poly), cur_epoch)
        writer.add_scalar('val_loss', total_loss/len(loader), cur_epoch)

    meter.log_epoch_stats(cur_epoch)
    #print('acc: ', float(total_correct_poly)/float(total_poly), 'loss: ', total_loss*batch_size/len(loader))
    return float(total_correct_poly)/float(total_poly)


def train_model():
    """Trains the model."""
    # Setup training/testing environment
    setup_env()
    writer = SummaryWriter(cfg.OUT_DIR)
    # Construct the model, ema, loss_fun, and optimizer
    model = setup_model()
    ema = deepcopy(model)
    loss_fun = builders.build_loss_fun("cross_entropy", "none").cuda()
    optimizer = optim.construct_optimizer(model)
    # Load checkpoint or initial weights
    start_epoch = 0
    if cfg.TRAIN.AUTO_RESUME and cp.has_checkpoint():
        file = cp.get_last_checkpoint()
        epoch = cp.load_checkpoint(file, model, ema, optimizer)[0]
        logger.info("Loaded checkpoint from: {}".format(file))
        start_epoch = epoch + 1
    elif cfg.TRAIN.WEIGHTS:
        cp.load_checkpoint(cfg.TRAIN.WEIGHTS, model, ema)
        logger.info("Loaded initial weights from: {}".format(cfg.TRAIN.WEIGHTS))
    # Create data loaders and meters
    train_loader = data_loader.construct_train_loader()
    test_loader = data_loader.construct_test_loader()
    train_meter = meters.TrainMeter(len(train_loader))
    test_meter = meters.TestMeter(len(test_loader))
    ema_meter = meters.TestMeter(len(test_loader), "test_ema")
    # Perform the training loop
    logger.info("Start epoch: {}".format(start_epoch + 1))
    for cur_epoch in range(start_epoch, cfg.OPTIM.MAX_EPOCH):
        # Train for one epoch
        params = (train_loader, model, ema, loss_fun, optimizer, train_meter)
        train_epoch(*params, cur_epoch, writer)
        '''
        # Compute precise BN stats
        if cfg.BN.USE_PRECISE_STATS:
            net.compute_precise_bn_stats(model, train_loader)
            net.compute_precise_bn_stats(ema, train_loader)
        '''
        # Evaluate the model
        test_acc = test_epoch(test_loader, model, test_meter, cur_epoch, loss_fun, writer)
        # Save a checkpoint
        file = cp.save_checkpoint(model, ema, optimizer, cur_epoch, test_acc)
        logger.info("Wrote checkpoint to: {}".format(file))
    writer.close()


def test_model():
    """Evaluates a trained model."""
    # Setup training/testing environment
    setup_env()
    # Construct the model
    model = setup_model()
    # Load model weights
    cp.load_checkpoint(cfg.TEST.WEIGHTS, model)
    logger.info("Loaded model weights from: {}".format(cfg.TEST.WEIGHTS))
    # Create data loaders and meters
    test_loader = data_loader.construct_test_loader()
    test_meter = meters.TestMeter(len(test_loader))
    # Evaluate the model
    test_epoch(test_loader, model, test_meter, 0, writer=None)


def time_model():
    """Times model."""
    # Setup training/testing environment
    setup_env()
    # Construct the model and loss_fun
    model = setup_model()
    loss_fun = builders.build_loss_fun().cuda()
    # Compute model and loader timings
    benchmark.compute_time_model(model, loss_fun)

def time_model_and_loader():
    """Times model and data loader."""
    # Setup training/testing environment
    setup_env()
    # Construct the model and loss_fun
    model = setup_model()
    loss_fun = builders.build_loss_fun().cuda()
    # Create data loaders
    train_loader = data_loader.construct_train_loader()
    test_loader = data_loader.construct_test_loader()
    # Compute model and loader timings
    benchmark.compute_time_full(model, loss_fun, train_loader, test_loader)
