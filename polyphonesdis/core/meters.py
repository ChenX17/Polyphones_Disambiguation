#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Meters."""

from collections import deque

import numpy as np
from torch.nn.functional import threshold
import polyphonesdis.core.logging as logging
import torch
from polyphonesdis.core.config import cfg
from polyphonesdis.core.timer import Timer


logger = logging.get_logger(__name__)


def time_string(seconds):
    """Converts time in seconds to a fixed-width string format."""
    days, rem = divmod(int(seconds), 24 * 3600)
    hrs, rem = divmod(rem, 3600)
    mins, secs = divmod(rem, 60)
    return "{0:02},{1:02}:{2:02}:{3:02}".format(days, hrs, mins, secs)


def topk_errors(preds, labels, ks):
    """Computes the top-k error for each k."""
    err_str = "Batch dim of predictions and labels must match"
    assert preds.size(0) == labels.size(0), err_str
    # Find the top max_k predictions for each sample
    _top_max_k_vals, top_max_k_inds = torch.topk(
        preds, max(ks), dim=1, largest=True, sorted=True
    )
    # (batch_size, max_k) -> (max_k, batch_size)
    top_max_k_inds = top_max_k_inds.t()
    # (batch_size, ) -> (max_k, batch_size)
    rep_max_k_labels = labels.view(1, -1).expand_as(top_max_k_inds)
    # (i, j) = 1 if top i-th prediction for the j-th sample is correct
    top_max_k_correct = top_max_k_inds.eq(rep_max_k_labels)
    # Compute the number of topk correct predictions for each k
    topks_correct = [top_max_k_correct[:k, :].reshape(-1).float().sum() for k in ks]
    return [(1.0 - x / preds.size(0)) * 100.0 for x in topks_correct]

def accuracy(preds, labels):
    threshold = 0.5
    """Computes the top-k error for each k."""
    err_str = "Batch dim of predictions and labels must match"
    assert preds.size(0) == labels.size(0), err_str
    # Find the top max_k predictions for each sample
    
    preds > threshold == labels.bool()
    accuracy = 0.0
    return accuracy

def acc(mask, y, predicts):
    predicts = torch.argmax(predicts, dim=-1)
    mask_poly = y > 1
    total_correct_poly = ((predicts == y) * mask * mask_poly).sum().item()
    total_poly = mask_poly.sum().item()
    acc = float(total_correct_poly) / float(total_poly)
    return acc, total_poly, total_correct_poly


def gpu_mem_usage():
    """Computes the GPU memory usage for the current device (MB)."""
    mem_usage_bytes = torch.cuda.max_memory_allocated()
    return mem_usage_bytes / 1024 / 1024


class ScalarMeter(object):
    """Measures a scalar value (adapted from Detectron)."""

    def __init__(self, window_size):
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0

    def reset(self):
        self.deque.clear()
        self.total = 0.0
        self.count = 0

    def add_value(self, value):
        self.deque.append(value)
        self.count += 1
        self.total += value

    def get_win_median(self):
        return np.median(self.deque)

    def get_win_avg(self):
        return np.mean(self.deque)

    def get_global_avg(self):
        return self.total / self.count


class TrainMeter(object):
    """Measures training stats."""

    def __init__(self, epoch_iters, phase="train"):
        self.epoch_iters = epoch_iters
        self.max_iter = cfg.OPTIM.MAX_EPOCH * epoch_iters
        self.phase = phase
        self.iter_timer = Timer()
        self.loss = ScalarMeter(cfg.LOG_PERIOD)
        self.loss_total = 0.0
        self.acc = 0.0
        self.acc_total = 0.0
        self.lr = None
        self.num_samples = 0

    def reset(self, timer=False):
        if timer:
            self.iter_timer.reset()
        self.loss.reset()
        self.loss_total = 0.0
        self.acc_total = 0.0
        self.lr = None
        self.acc = 0.0
        self.num_samples = 0

    def iter_tic(self):
        self.iter_timer.tic()

    def iter_toc(self):
        self.iter_timer.toc()

    def update_stats(self, loss, lr, acc, mb_size):
        # Current minibatch stats
        self.loss.add_value(loss)
        self.lr = lr
        # Aggregate stats
        self.loss_total += loss * mb_size
        self.num_samples += mb_size
        self.acc = acc
        self.acc_total += acc

    def get_iter_stats(self, cur_epoch, cur_iter):
        cur_iter_total = cur_epoch * self.epoch_iters + cur_iter + 1
        eta_sec = self.iter_timer.average_time * (self.max_iter - cur_iter_total)
        mem_usage = gpu_mem_usage()
        stats = {
            "epoch": "{}/{}".format(cur_epoch + 1, cfg.OPTIM.MAX_EPOCH),
            "iter": "{}/{}".format(cur_iter + 1, self.epoch_iters),
            "time_avg": self.iter_timer.average_time,
            "time_diff": self.iter_timer.diff,
            "eta": time_string(eta_sec),
            "loss": self.loss.get_win_median(),
            "train_acc": self.acc,
            "lr": self.lr,
            "mem": int(np.ceil(mem_usage)),
        }
        return stats

    def log_iter_stats(self, cur_epoch, cur_iter):
        if (cur_iter + 1) % cfg.LOG_PERIOD == 0:
            stats = self.get_iter_stats(cur_epoch, cur_iter)
            logger.info(logging.dump_log_data(stats, self.phase + "_iter"))

    def get_epoch_stats(self, cur_epoch):
        cur_iter_total = (cur_epoch + 1) * self.epoch_iters
        eta_sec = self.iter_timer.average_time * (self.max_iter - cur_iter_total)
        mem_usage = gpu_mem_usage()
        avg_loss = self.loss_total / self.num_samples
        avg_acc = self.acc_total / self.num_samples
        stats = {
            "epoch": "{}/{}".format(cur_epoch + 1, cfg.OPTIM.MAX_EPOCH),
            "time_avg": self.iter_timer.average_time,
            "time_epoch": self.iter_timer.average_time * self.epoch_iters,
            "eta": time_string(eta_sec),
            "loss": self.loss.get_win_median(),
            "avg_train_acc": avg_acc,
            "lr": self.lr,
            "mem": int(np.ceil(mem_usage)),
        }
        return stats

    def log_epoch_stats(self, cur_epoch):
        stats = self.get_epoch_stats(cur_epoch)
        logger.info(logging.dump_log_data(stats, self.phase + "_epoch"))


class TestMeter(object):
    """Measures testing stats."""

    def __init__(self, epoch_iters, phase="test"):
        self.epoch_iters = epoch_iters
        self.phase = phase
        self.iter_timer = Timer()
        self.num_samples = 0
        self.loss_total = 0.0
        self.acc_total = 0.0
        self.loss = 0.0

    def reset(self, min_errs=False):
        self.iter_timer.reset()
        self.num_samples = 0
        self.loss_total = 0.0
        self.acc_total = 0.0

    def iter_tic(self):
        self.iter_timer.tic()

    def iter_toc(self):
        self.iter_timer.toc()

    def update_stats(self, loss, test_acc, mb_size):
        # Current minibatch stats
        self.loss = loss.item()
        self.loss_total += loss.item() * mb_size
        self.num_samples += mb_size
        self.acc = test_acc
        self.acc_total += test_acc * mb_size

    def get_iter_stats(self, cur_epoch, cur_iter):
        mem_usage = gpu_mem_usage()

        iter_stats = {
            "epoch": "{}/{}".format(cur_epoch + 1, cfg.OPTIM.MAX_EPOCH),
            "iter": "{}/{}".format(cur_iter + 1, self.epoch_iters),
            "time_avg": self.iter_timer.average_time,
            "time_diff": self.iter_timer.diff,
            "mem": int(np.ceil(mem_usage)),
        }
        return iter_stats

    def log_iter_stats(self, cur_epoch, cur_iter):
        if (cur_iter + 1) % cfg.LOG_PERIOD == 0:
            stats = self.get_iter_stats(cur_epoch, cur_iter)
            logger.info(logging.dump_log_data(stats, self.phase + "_iter"))

    def get_epoch_stats(self, cur_epoch):
        mem_usage = gpu_mem_usage()
        avg_loss = self.loss_total / self.num_samples
        avg_acc = self.acc_total / self.num_samples
        stats = {
            "epoch": "{}/{}".format(cur_epoch + 1, cfg.OPTIM.MAX_EPOCH),
            "time_avg": self.iter_timer.average_time,
            "loss": avg_loss,
            "avg_test_acc": avg_acc,
            "time_epoch": self.iter_timer.average_time * self.epoch_iters,
            "mem": int(np.ceil(mem_usage)),
        }
        return stats

    def log_epoch_stats(self, cur_epoch):
        stats = self.get_epoch_stats(cur_epoch)
        logger.info(logging.dump_log_data(stats, self.phase + "_epoch"))

    def acc(self, preds, labels):
        threshold = 0.2
        """Computes the top-k error for each k."""
        err_str = "Batch dim of predictions and labels must match"
        assert preds.size(0) == labels.size(0), err_str
        # Find the top max_k predictions for each sample
        
        preds = torch.sigmoid(preds) > threshold
        # torch.sum(preds == labels.bool())
        accuracy = torch.sum(preds == labels.bool())/labels.size(0)
        recall = torch.sum(preds*labels.bool())/torch.sum(labels.bool())
        return accuracy, recall
