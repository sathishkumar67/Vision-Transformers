from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

import torchvision
import torchvision.transforms as transforms
from torchvision.models import vit_b_16

import lightning as L
from lightning.pytorch import Trainer

from datasets import load_dataset
from schedulefree.adamw_schedulefree import AdamWScheduleFree

import os
import sys
from PIL import Image
from typing import Optional
from dataclasses import dataclass
import argparse as ap
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from huggingface_hub import hf_hub_url, hf_hub_download, login, HfApi