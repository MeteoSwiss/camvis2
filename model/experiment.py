import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from dataset import MultiMagnificationPatches
from model import MultiMagnificationNet

DIR_PATH = os.path.dirname(os.path.realpath(__file__))


class Experiment():
    def __init__(self, cfg):
        """Initialization"""

        # Config, dataset and network
        self.cfg = cfg
        
        self.dataset = MultiMagnificationPatches(
            dataset_path=self.cfg.DATASET_PATH,
            demagnification_factors=self.cfg.FACTORS,
            random_seed=self.cfg.RANDOM_SEED, 
            val_view=self.cfg.VAL_VIEW,
            test_view=None,) # No test set as we are doing cross validation

        # cuDNN setup
        if self.cfg.COMPUTE_MODE == "reproducibility":
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        elif self.cfg.COMPUTE_MODE == "performance":
            torch.backends.cudnn.benchmark = True

        # Fix random seeds
        torch.manual_seed(self.cfg.RANDOM_SEED)
        random.seed(self.cfg.RANDOM_SEED)
        np.random.seed(self.cfg.RANDOM_SEED)

        # Model
        self.net = MultiMagnificationNet(
            num_levels=len(self.cfg.FACTORS), 
            size_hidden=self.cfg.HIDDEN_CHANNELS, 
            share_weights=self.cfg.SHARE_WEIGHTS,
            use_mlp=self.cfg.USE_MLP,
            align_features=self.cfg.ALIGN_FEATURES)
        self.net.to(device=self.cfg.DEVICE)
        
        if self.cfg.TRAIN_LOGS:
            self.writer = SummaryWriter(f"{self.cfg.TRAIN_LOGS_PATH}/self.cfg.RUN_NAME")

        # Model weights setting
        if self.cfg.LOAD_CHECKPOINT:
            path = f"{self.cfg.CHECKPOINTS_PATH}/bestmodel_{self.cfg.RUN_NAME}.pt"
            self.net.load_state_dict(torch.load(path, map_location=self.cfg.DEVICE))
            print(f"Loaded model state dict from file {path}")

        # Optimizer, scheduler, early stopper and criterion
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.cfg.LEARNING_RATE)
        if self.cfg.SELECTION_FUNCTION == "loss":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=10, cooldown=0, 
                threshold=1e-4, min_lr=self.cfg.MIN_LEARNING_RATE/10)
        else:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.1, patience=10, cooldown=0, 
                threshold=1e-4, min_lr=self.cfg.MIN_LEARNING_RATE/10)
        self.criterion = nn.BCEWithLogitsLoss()
        self.scaler = torch.cuda.amp.GradScaler()

    def train(self):
        pass

    def eval(self):
        pass