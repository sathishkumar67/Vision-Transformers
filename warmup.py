from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, _LRScheduler
from torch.utils.data import DataLoader, Dataset

import torchvision
import torchvision.transforms as transforms
from torchvision.models import vit_l_16

import lightning as L
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

from datasets import load_dataset
import os
import shutil
import sys
from PIL import Image
from typing import Optional
from dataclasses import dataclass
import argparse as ap
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from huggingface_hub import hf_hub_url, hf_hub_download, login, HfApi


class WarmupCosineLR_Step(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, max_steps, min_lr=0, last_step=-1):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr = min_lr
        super(WarmupCosineLR, self).__init__(optimizer, last_step)

    def get_lr(self):
        if self.last_step < self.warmup_steps:
            # Linear warmup
            return [base_lr * (self.last_step + 1) / self.warmup_steps for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            cos_anneal_epoch = self.last_step - self.warmup_steps
            total_cos_anneal_epochs = self.max_steps - self.warmup_steps
            return [
                self.min_lr + (base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * cos_anneal_epoch / total_cos_anneal_epochs))
                for base_lr in self.base_lrs
            ]



class WarmupCosineLR_Epoch(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, max_epochs, min_lr=0, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        super(WarmupCosineLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            return [base_lr * (self.last_epoch + 1) / self.warmup_epochs for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            cos_anneal_epoch = self.last_epoch - self.warmup_epochs
            total_cos_anneal_epochs = self.max_epochs - self.warmup_epochs
            return [
                self.min_lr + (base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * cos_anneal_epoch / total_cos_anneal_epochs))
                for base_lr in self.base_lrs
            ]

