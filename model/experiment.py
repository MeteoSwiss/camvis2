import os
import random
from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torcheval.metrics.functional import (multiclass_accuracy, multiclass_f1_score)
from tqdm import trange

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

    def seed_worker(self, worker_id):
        """Initializ dataloader workers random seeds (https://pytorch.org/docs/stable/notes/randomness.html)"""
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def save_net_params(self):
        """Model parameters checkpoint"""
        path = f"{self.cfg.CHECKPOINTS_PATH}/bestmodel_{self.cfg.RUN_NAME}.pt"
        torch.save(self.net.state_dict(), path)

    def infer_round_light(self, dataloader):
        """Forward pass and loss only, lighter version of the inference_round"""

        # Set model to train mode, dataloader and metrics
        self.net.eval()
        accum_loss = 0
        step_count = 0

        # Process batches
        with torch.no_grad():   
            for minibatch in dataloader:

                # Predict batch
                input = minibatch[0].to(dtype=torch.float32, device=self.cfg.DEVICE, non_blocking=True)
                target = minibatch[1].to(dtype=torch.float32, device=self.cfg.DEVICE, non_blocking=True)
                pred = self.net(input).squeeze()
                loss = self.criterion(pred, target)

                # Metrics update
                accum_loss += loss.item()
                step_count += 1
        
        return accum_loss/step_count
    
    def infer_round(self, dataloader):
        """Forward pass and global metrics computation"""

        # Set model to eval mode, dataloader and metrics
        self.net.eval()

        all_preds = torch.zeros(len(self.dataset))
        all_targets = torch.zeros(len(self.dataset))
        step_count = 0

        # Process batches
        with torch.no_grad():   
            for minibatch in dataloader:

                # Predict batch
                input = minibatch[0].to(dtype=torch.float32, device=self.cfg.DEVICE, non_blocking=True)
                target = minibatch[1].to(dtype=torch.float32, device=self.cfg.DEVICE, non_blocking=True)
                pred = self.net(input).squeeze()
                # Store results and targets
                all_preds[self.cfg.BATCH_SIZE*step_count:self.cfg.BATCH_SIZE*(step_count+1)] = pred.detach()
                all_targets[self.cfg.BATCH_SIZE*step_count:self.cfg.BATCH_SIZE*(step_count+1)] = target.detach()

                step_count += 1

        # Metrics Computation
        loss = self.criterion(all_preds, all_targets).cpu()
        all_preds = F.sigmoid(all_preds)
        round_preds = all_preds.round().to(dtype=torch.int64)
        round_targets = all_targets.round().to(dtype=torch.int64)
        acc = multiclass_accuracy(round_preds, round_targets, average="macro", num_classes=2).cpu()
        f1 = multiclass_f1_score(round_preds, round_targets, average="macro", num_classes=2).cpu()
        return loss, acc, f1, all_preds.cpu(), all_targets.cpu()
    
    def train_round_mixed_precision(self, dataloader):
        """Forward pass, backard pass, metrics computation and parameters update"""
        
        # Set model to train mode, dataloader and metrics
        self.net.train()
        accum_loss = 0
        step_count = 0

        # Process batches
        for minibatch in dataloader:

            # Reset gradients
            self.optimizer.zero_grad()

            # Predict batch
            input = minibatch[0].to(dtype=torch.float32, device=self.cfg.DEVICE, non_blocking=True)
            target = minibatch[1].to(dtype=torch.float32, device=self.cfg.DEVICE, non_blocking=True)

            # Using AMP to improve processing speed
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                pred = self.net(input).squeeze()

                # Backpropagate loss and update parameters
                loss = self.criterion(pred, target)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

            # Metrics update
            accum_loss += loss.item()
            step_count += 1
    
        return accum_loss/step_count
    
    def train_round(self, dataloader):
        """Forward pass, backard pass, metrics computation and parameters update"""
        
        # Set model to train mode, dataloader and metrics
        self.net.train()
        accum_loss = 0
        step_count = 0

        # Process batches
        for minibatch in dataloader:

            # Reset gradients
            self.optimizer.zero_grad()

            # Predict batch
            input = minibatch[0].to(dtype=torch.float32, device=self.cfg.DEVICE, non_blocking=True)
            target = minibatch[1].to(dtype=torch.float32, device=self.cfg.DEVICE, non_blocking=True)

            pred = self.net(input).squeeze()

            # Backpropagate loss and update parameters
            loss = self.criterion(pred, target)
            loss.backward()
            self.optimizer.step()

            # Metrics update
            accum_loss += loss.item()
            step_count += 1
    
        return accum_loss/step_count

    def train(self):
        """Training loop"""

        # Ensure reproducibility
        g = torch.Generator()
        g.manual_seed(self.cfg.RANDOM_SEED)
        
        # Initialize best model score
        if self.cfg.SELECTION_FUNCTION == "loss":
            bestmodel_score = 1e9
        else:
            bestmodel_score = 0

        # Prepare dataloaders for training
        self.dataset.train()
        train_dataloader = DataLoader(
            self.dataset, 
            shuffle=True, 
            batch_size=self.cfg.BATCH_SIZE, 
            num_workers=self.cfg.NUM_WORKERS, 
            pin_memory=True, 
            prefetch_factor=2, 
            persistent_workers=False, 
            worker_init_fn=self.seed_worker, 
            generator=g, 
        )
        
        if not self.cfg.TRAIN_FAST:
            self.dataset.train(inference_mode=True)
            train_infer_dataloader = DataLoader(
                self.dataset, 
                shuffle=False, 
                batch_size=self.cfg.BATCH_SIZE, 
                num_workers=self.cfg.NUM_WORKERS, 
                pin_memory=True, 
                prefetch_factor=2, 
                persistent_workers=False, 
            )
        
        self.dataset.val()
        val_infer_dataloader = DataLoader(
            self.dataset, 
            shuffle=False, 
            batch_size=self.cfg.BATCH_SIZE, 
            num_workers=self.cfg.NUM_WORKERS, 
            pin_memory=True, 
            prefetch_factor=2, 
            persistent_workers=False, 
        )
        
        pbar = trange(self.cfg.EPOCH_NUM)
        start_time = time()

        for epoch in pbar:

            # Train
            self.dataset.train()
            if self.cfg.AMP:
                train_loss = self.train_round_mixed_precision(train_dataloader)
            else:
                train_loss = self.train_round(train_dataloader)

            # Validate
            self.dataset.val()
            val_loss, val_acc, val_f1, _, _ = self.infer_round(val_infer_dataloader)

            if not self.cfg.TRAIN_FAST:
                self.dataset.train(inference_mode=True)
                train_loss, train_acc, train_f1, _, _ = self.infer_round(train_infer_dataloader)

            # Log progess
            if self.cfg.TRAIN_FAST:
                pbar.set_description(
                    f"metrics at epoch {epoch+1} | "
                    f"train/val loss : {train_loss:.2f}/{val_loss:.2f} | "
                    f"val acc : {val_acc:.2f} | "
                    f"val f1 : {val_f1:.2f}"
                )
            else:
                pbar.set_description(
                    f"train/val metrics at epoch {epoch+1} | "
                    f"loss : {train_loss:.2f}/{val_loss:.2f} | "
                    f"acc : {train_acc:.2f}/{val_acc:.2f} | "
                    f"f1 : {train_f1:.2f}/{val_f1:.2f}"
                )

            # Save model as best model if it has the best validation decision score
            if self.cfg.SAVE_CHECKPOINTS:
                if self.cfg.SELECTION_FUNCTION == "acc":
                    if val_acc > bestmodel_score:
                        self.save_net_params()
                        bestmodel_score = val_acc
                elif self.cfg.SELECTION_FUNCTION == "f1":
                    if val_f1 > bestmodel_score:
                        self.save_net_params()
                        bestmodel_score = val_f1
                elif self.cfg.SELECTION_FUNCTION == "loss":
                    if val_loss < bestmodel_score:
                        self.save_net_params()
                        bestmodel_score = val_loss
            
            # Update learning rate
            if self.cfg.SELECTION_FUNCTION == "acc":
                self.scheduler.step(val_acc)
            elif self.cfg.SELECTION_FUNCTION == "f1":
                self.scheduler.step(val_f1)
            elif self.cfg.SELECTION_FUNCTION == "loss":
                self.scheduler.step(val_loss)

            # Log results
            if self.cfg.TRAIN_LOGS:
                self.writer.add_scalar("run time", time()-start_time, epoch+1)
                self.writer.add_scalar("train loss", train_loss, epoch+1)
                if not self.cfg.TRAIN_FAST:
                    self.writer.add_scalar("train accuracy", train_acc, epoch+1)
                    self.writer.add_scalar("train f1 score", train_f1, epoch+1)
                self.writer.add_scalar("val loss", val_loss, epoch+1)
                self.writer.add_scalar("val accuracy", val_acc, epoch+1)
                self.writer.add_scalar("val f1 score", val_f1, epoch+1)
                self.writer.add_scalar("learning rate", self.optimizer.param_groups[0]['lr'], epoch+1)

            # Check if minimal learning rate has been reached
            if self.optimizer.param_groups[0]['lr'] < self.cfg.MIN_LEARNING_RATE:
                train_dataloader = None
                train_infer_dataloader = None
                val_infer_dataloader = None
                exit()
        
        train_dataloader = None
        train_infer_dataloader = None
        val_infer_dataloader = None

        exit()

    def eval(self):
        pass