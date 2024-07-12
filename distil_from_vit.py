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
from torchvision.transforms import v2
from torchvision.models import resnet50, vit_b_16

import lightning as L
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

from datasets import load_dataset
from schedulefree.adamw_schedulefree import AdamWScheduleFree

import os
import shutil
import sys
from PIL import Image
from typing import Optional
from dataclasses import dataclass
import argparse as ap
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from huggingface_hub import hf_hub_url, hf_hub_download, login, HfApi


# define the args
@dataclass
class Args:
    epochs: int = 10
    lr: float = 3e-4
    batch_size: int = 64
    seed: int = 42
    num_workers: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_class: int = 128
    device_count: int = 1
    temperature: int = 2
    weight_decay: float = 1e-5
        
    def __post_init__(self):
        torch.manual_seed(self.seed)
        
# transform images
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])


# distillation model
class Distillation(L.LightningModule):
    def __init__(self, student_model, teacher_model):
        super().__init__()

        # student and teacher model
        self.student_model = student_model
        self.teacher_model = teacher_model

        # adding mixup data augmentation
        self.mixup = v2.MixUp(alpha=1.0, num_classes=args.num_class)

        # storing loss
        self.train_loss = []
        self.val_loss = []

    def training_step(self, batch, batch_idx):
        loss = 0
        self.student_model.train()
        optimizer = self.optimizers()
        optimizer.train()
        optimizer.zero_grad()
        
        batch, label = batch
        # adding mixup data augmentation
        batch, label = self.mixup(batch, label)

        # forward
        student_output = self.student_model(batch)
        
        with torch.inference_mode():
            teacher_output = self.teacher_model(batch)

        loss += F.kl_div(F.log_softmax(student_output / args.temperature, dim=1), F.softmax(teacher_output / args.temperature, dim=1), reduction='sum') * (args.temperature**2) / batch.shape[0]
        self.train_loss.append(loss.item())
        self.log("Train_KL_Loss", loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = AdamWScheduleFree(self.student_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        return optimizer
    
    def validation_step(self, batch, batch_idx):
        self.student_model.eval()
        optimizer = self.optimizers()
        optimizer.eval()
        
        batch, label = batch
        # adding mixup data augmentation
        batch, label = self.mixup(batch, label)
        out = self.student_model(batch)
        loss = F.cross_entropy(out, label)
        self.val_loss.append(loss.item())
        self.log("Val_Loss", loss, prog_bar=True)
        return loss


# set the args
args = Args()

# set the seed
torch.manual_seed(args.seed)


# prepare the dataset
train_dataset = torchvision.datasets.Food101(root='./data', split='train', download=True, transform=transform)
test_dataset = torchvision.datasets.Food101(root='./data', split='test', download=True, transform=transform)

# prepare dataloader
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)


# downloading the teacher's checkpoint
hf_hub_download(repo_id='pt-sk/VIT-B-16_Food-101', filename="vit_b_16_10epoch_Food101.pt", local_dir="/kaggle/working/")


# preparing the teacher model
teacher_model = vit_b_16(pretrained=False)
teacher_model.heads = nn.Linear(768, args.num_class)
teacher_model.load_state_dict(torch.load("/kaggle/working/vit_b_16_10epoch_Food101.pt"))

# preparing the student model
student_model = resnet50(pretrained=False)
student_model.fc = nn.Linear(2048, args.num_class)


# instantiating the distillation model
distillation_model = Distillation(student_model, teacher_model)

# define modelcheckpoint callback
checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="{epoch:02d}",
        save_top_k=-1,  # Save all models
        mode="min",
        monitor="Val_Loss",
        save_weights_only=True,
        every_n_epochs=1
    )


# train the model
trainer = Trainer(max_epochs=args.epochs,
                accelerator=args.device,
                callbacks=[checkpoint_callback])
    
# fit the model
trainer.fit(distillation_model, train_dataloader, test_dataloader)


# accessing the train loss, val loss
train_loss = np.array(distillation_model.train_loss)
val_loss = np.array(distillation_model.val_loss)

# save the loss
np.save("train_loss.npy", train_loss)
np.save("val_loss.npy", val_loss)
