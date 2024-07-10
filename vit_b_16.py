# import packages
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
    batch_size: int = 32
    seed: int = 42
    num_workers: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_class: int = 128
    device_count: int = 1
        
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

class VisionTransformer(L.LightningModule):
    def __init__(self, model, args: Args):
        super().__init__()
        self.model = model
        self.args = args
        self.train_loss = []
        self.val_loss = []
        
    def training_step(self, batch, batch_idx):
        self.model.train()
        optimizer = self.optimizers()
        optimizer.train()
        optimizer.zero_grad()
        
        batch, label = batch
        out = self.model(batch)
        loss = F.cross_entropy(out, label)
        self.train_loss.append(loss.item())
        self.log("Train_Loss", loss, prog_bar=True)

        return loss
    
    def configure_optimizers(self):
        optimizer = AdamWScheduleFree(self.model.parameters(), lr=self.args.lr)
        
        return optimizer
    
    def validation_step(self, batch, batch_idx):
        self.model.eval()
        optimizer = self.optimizers()
        optimizer.eval()
        
        batch, label = batch
        out = self.model(batch)
        loss = F.cross_entropy(out, label)
        self.val_loss.append(loss.item())
        self.log("Val_Loss", loss, prog_bar=True)

        return loss

def main():
    # set the seed
    torch.manual_seed(42)

    # instantiate the args
    args = Args()

    # prepare dataset
    train_dataset = torchvision.datasets.Food101(root='./data', split='train', download=True, transform=transform)
    test_dataset = torchvision.datasets.Food101(root='./data', split='test', download=True, transform=transform)

    # prepare dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    # prepare model
    vision_model = vit_b_16(pretrained=False)
    vision_model.heads = nn.Linear(768, args.num_class)

    # instantiate the model
    vision_transformer = VisionTransformer(vision_model, args)

    # Define ModelCheckpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="{epoch:02d}-{val_loss:.2f}",
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
    trainer.fit(vision_transformer, train_dataloader, test_dataloader)

    # remove the data directory
    shutil.rmtree('./data')

    # accessing the train and val loss
    train_loss = np.array(vision_transformer.train_loss)
    val_loss = np.array(vision_transformer.val_loss)

    # save the train and val loss
    np.save('train_loss.npy', train_loss)
    np.save('val_loss.npy', val_loss)


if __name__ == "__main__":
    main()
