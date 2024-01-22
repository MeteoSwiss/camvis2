# Copyright (c) 2024 MeteoSwiss, contributors listed in AUTHORS
#
# Distributed under the terms of the BSD 3-Clause License.
#
# SPDX-License-Identifier: BSD-3-Clauseimport argparse

import os
import random
from time import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torcheval.metrics.functional import (multiclass_accuracy, multiclass_f1_score)
from tqdm import tqdm, trange

from dataset import MultiMagnificationPatches
from model import MultiMagnificationNet

plt.switch_backend('agg')

DIR_PATH = os.path.dirname(os.path.realpath(__file__))


class Experiment():
    def __init__(self, cfg):
        """
        A class representing an experiment for training and evaluating a neural network model.

        Args:
            cfg (object): An object containing various configuration parameters for the experiment.

        Attributes:
            cfg (object): Configuration object containing experiment parameters.
            dataset (MultiMagnificationPatches): Dataset object for training and evaluation.
            net (MultiMagnificationNet): Neural network model for the experiment.
            writer (SummaryWriter): Tensorboard writer for logging training information.
            optimizer (torch.optim.Adam): Optimizer for updating model parameters.
            scheduler (torch.optim.lr_scheduler.ReduceLROnPlateau): Learning rate scheduler.
            criterion (nn.BCEWithLogitsLoss): Loss function for training.
            scaler (torch.cuda.amp.GradScaler): Gradient scaler for mixed-precision training.

        Methods:
            seed_worker(worker_id): Initialize dataloader workers random seeds.
            save_net_params(): Save the model parameters to a checkpoint file.
            infer_round_light(dataloader): Perform a lighter version of the inference round for validation.
            infer_round(dataloader): Perform a full inference round and compute global metrics.
            train_round_mixed_precision(dataloader): Perform forward pass, backward pass, metrics computation, and parameters update with mixed-precision training.
            train_round(dataloader): Perform forward pass, backward pass, metrics computation, and parameters update with regular training.
            bootstrap_infer(dataloader, num_iterations=10, num_samples=False, cf_mat=True): Perform inference with bootstrapping for more interpretable evaluation.
            make_cf_mat(preds, targets): Build a confusion matrix and save it as an image.
            infer_on_images(): Perform inference on images of the current dataset subset and save the results.

        """

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
        
        # Tensorboard logging
        if self.cfg.TRAIN_LOGS:
            self.writer = SummaryWriter(f"{self.cfg.TRAIN_LOGS_PATH}/{self.cfg.MODEL_NAME}")

        # Model weights setting
        if self.cfg.LOAD_CHECKPOINT:
            path = f"{self.cfg.CHECKPOINTS_PATH}/bestmodel_{self.cfg.MODEL_NAME}.pt"
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
        """
        Initialize dataloader workers random seeds based on the given worker identifier.
        Based on https://pytorch.org/docs/stable/notes/randomness.html

        Args:
            worker_id (int): Identifier for the dataloader worker.

        Note:
            This method sets the random seed for NumPy and Python's built-in random module
            to ensure reproducibility in dataloader workers. The seed is computed using the
            initial seed from PyTorch, allowing for reproducibility across different libraries.

        Returns:
            None
        """
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def save_net_params(self):
        """
        Save the current state of the neural network parameters as a checkpoint.

        Returns:
            None

        Note:
            This method saves the state dictionary of the neural network's parameters to a
            specified file path. The saved file can later be used to restore the model to
            its current state.
            The file is saved in PyTorch's native format (`.pt`).
        """
        path = f"{self.cfg.CHECKPOINTS_PATH}/bestmodel_{self.cfg.MODEL_NAME}.pt"
        torch.save(self.net.state_dict(), path)

    def infer_round_light(self, dataloader):
        """
        Forward pass and loss only, lighter version of the inference_round.

        Args:
            dataloader (torch.utils.data.DataLoader): DataLoader providing batches of data.

        Returns:
            float: Average loss over all batches.

        Note:
            This method performs a forward pass and calculates the loss on the provided DataLoader,
            without updating the model's weights. It is a lighter version of the `infer_round` method.
        """

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
        """
        Forward pass and global metrics computation.

        Args:
            dataloader (torch.utils.data.DataLoader): DataLoader providing batches of data.

        Returns:
            tuple: Tuple containing the following elements:
                float: Loss computed over all batches.
                float: Macro-average accuracy.
                float: Macro-average F1 score.
                torch.Tensor: Predictions for all samples.
                torch.Tensor: Ground truth labels for all samples.

        Note:
            This method performs a forward pass on the provided DataLoader, computes global metrics,
            and returns a tuple with loss, accuracy, F1 score, predictions, and ground truth labels.
        """

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
        """
        Forward pass, backward pass, metrics computation, and parameters update using mixed precision.

        Args:
            dataloader (torch.utils.data.DataLoader): DataLoader providing batches of data.

        Returns:
            float: Average loss over all batches.

        Note:
            This method performs a forward pass, backward pass, and updates model parameters using mixed precision.
            It is designed to speed up training by using Automatic Mixed Precision (AMP).
        """
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
        """
        Forward pass, backward pass, metrics computation, and parameters update.

        Args:
            dataloader (torch.utils.data.DataLoader): DataLoader providing batches of data.

        Returns:
            float: Average loss over all batches.

        Note:
            This method performs a forward pass, backward pass, and updates model parameters.
            It is used during the training phase.
        """
        
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
        """
        Train the model using the specified configuration.

        Note:
            This method performs the training loop, including forward and backward passes, metrics computation,
            model saving, learning rate scheduling, and logging.
        """
        
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
        
        pbar = trange(self.cfg.NUM_EPOCHS)
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

    def bootstrap_infer(self, dataloader, num_iterations = 10, num_samples = False, cf_mat=True):
        """
        Perform inference with bootstrapping for a more interpretable evaluation of the performance.

        Args:
            dataloader (torch.utils.data.DataLoader): DataLoader providing batches of data.
            num_iterations (int): Number of bootstrap iterations.
            num_samples (int, optional): Number of samples to draw in each iteration. Default is False.
            cf_mat (bool, optional): Whether to compute and display a confusion matrix. Default is True.

        Returns:
            tuple: Tuple containing the following elements:
                torch.Tensor: Concatenated losses over all iterations.
                torch.Tensor: Concatenated macro-average accuracies over all iterations.
                torch.Tensor: Concatenated macro-average F1 scores over all iterations.

        Note:
            This method performs inference using bootstrapping, sampling with replacement from the predictions and targets
            obtained from a single round of inference. It provides more interpretable evaluation metrics by considering
            variability in performance due to random sampling.
        """

        # Initialize variables to store performances
        concat_loss = torch.zeros(num_iterations)
        concat_acc = torch.zeros(num_iterations)
        concat_f1 = torch.zeros(num_iterations)
        
        # All predictions and targets
        _, _, _, preds, targets = self.infer_round(dataloader)

        if cf_mat:
            self.make_cf_mat(preds, targets)

        # Set random numbers generator and use a large seed number
        generator = torch.Generator()
        generator.manual_seed(self.cfg.RANDOM_SEED**5)


        # If nothing specified, sample the same amount of elements as the set size
        if not num_samples:
            num_samples = len(preds)

        for i in range(num_iterations):
            idx = torch.multinomial(input = torch.ones(len(preds)).float(), num_samples=num_samples, replacement=True, generator=generator)

            concat_loss[i] = self.criterion(preds[idx],targets[idx])
            round_preds = preds[idx].round().to(dtype=torch.int64)
            round_targets = targets[idx].round().to(dtype=torch.int64)
            concat_acc[i] = multiclass_accuracy(round_preds, round_targets, average="macro", num_classes=2)
            concat_f1[i] = multiclass_f1_score(round_preds, round_targets, average="macro", num_classes=2)

        return concat_loss, concat_acc, concat_f1

    def make_cf_mat(self, preds, targets):
        """
        Build the confusion matrix and save it as an image.

        Args:
            preds (torch.Tensor): Predicted values.
            targets (torch.Tensor): True target values.

        Returns:
            numpy.ndarray: Flattened confusion matrix.

        Note:
            This method uses sklearn's confusion_matrix and ConfusionMatrixDisplay for visualization.
            The confusion matrix image is saved in the specified path.
        """
        cf_mat = confusion_matrix(targets.round().int(), preds.round().int(), labels=[0,1])
        disp = ConfusionMatrixDisplay(confusion_matrix=cf_mat, display_labels=["nonvisible:0","visible:1"])
        disp.plot()
        plt.tight_layout()
        plt.savefig(f"{self.cfg.EVAL_MAT_PATH}/{self.cfg.MODEL_NAME}.png")
        plt.close()
        return cf_mat.flatten()

    def eval(self):
        """
        Evaluate the model using inference on images and bootstrap aggregation.

        Note:
            This method performs inference on images, and performs bootstrapping to evaluate the model's performance.
            It logs the evaluation results to a CSV file.
        """
        
        # Infer on images
        self.dataset.val()
        self.infer_on_images()
        
        # Make dataloader
        infer_dataloader = DataLoader(
            self.dataset, 
            shuffle=False, 
            batch_size=self.cfg.BATCH_SIZE, 
            num_workers=self.cfg.NUM_WORKERS, 
            pin_memory=False, 
            prefetch_factor=2, 
            persistent_workers=False
        )
        
        # Bootstrap phase
        loss, acc, f1 = self.bootstrap_infer(infer_dataloader)
        self.infer_dataloader = None
        
        # Results logging
        with open(f"{self.cfg.EVAL_SCORES_PATH}/{self.cfg.EVAL_SCORES_FNAME}.csv", "a") as logs_file:
            for i in range(len(loss)):
                logs_file.write(
                    f"\n{self.cfg.MODEL_NAME},"
                    f"{self.cfg.VAL_FOLD},"
                    f"{self.cfg.VAL_VIEW},"
                    f"{loss[i].item():.4f},"
                    f"{acc[i].item():.4f},"
                    f"{f1[i].item():.4f},"
                    f"{self.cfg.EVAL_GROUP}"
                )
        exit()

    
    def infer_on_images(self):
        """
        Perform inference on the images of the current dataset subset and save the results.

        Note:
            This method visualizes the ground truth and predicted labels on source images and saves them.
            It also creates histograms for the prevailing visibility estimation and saves them.
        """

        # Check if directory exists for this run, if not, make one
        if not os.path.exists(f"{self.cfg.EVAL_IMG_PATH}/{self.cfg.MODEL_NAME}"):
            os.makedirs(f"{self.cfg.EVAL_IMG_PATH}/{self.cfg.MODEL_NAME}")
        if not os.path.exists(f"{self.cfg.EVAL_HIST_PATH}/{self.cfg.MODEL_NAME}"):
            os.makedirs(f"{self.cfg.EVAL_HIST_PATH}/{self.cfg.MODEL_NAME}")

        # Colormap for plotting
        cmap = LinearSegmentedColormap.from_list("", ["fuchsia", "yellow", "lightgreen"])

        # Bins for hist plots
        xbins = np.log10(100000)*np.linspace(0, 100, 51)/100
        ybins = np.linspace(0,1,11)

        # Get split data and source image names
        df = self.dataset.split_metadata.copy()
        names = df["source_image"].unique().tolist()
        df["distance"] = df["depth"]

        # Set model to evaluation mode
        self.net.eval()

        # Process source images
        for image_name in tqdm(names):

            # Get source image
            picture = cv2.imread(f"{self.cfg.RAW_IMG_PATH}/{image_name}.jp2")[:,:,::-1].copy()
                
            # Get corresponding patches and labels
            subdf = df.loc[df["source_image"] == image_name]
            h = [int(x) for x in subdf["h_canvas"].tolist()]
            w = [int(x) for x in subdf["w_canvas"].tolist()]
            label = subdf["label"].tolist()

            # Plot source image with patches and ground truth and save it
            plt.imshow(picture)
            plt.scatter(w, h, c = label, s = 5, cmap = cmap, alpha = 0.8, vmin=0, vmax=1)
            plt.savefig(f"{self.cfg.EVAL_IMG_PATH}/{self.cfg.MODEL_NAME}/{image_name}_true.png")
            plt.close()

            # Gather inputs
            self.dataset.split_metadata = subdf
            dataloader = DataLoader(self.dataset, shuffle=False, batch_size = len(subdf), num_workers=self.cfg.NUM_WORKERS)
            inputs, _, _ = next(iter(dataloader))

            # Put to device and infer
            inputs = inputs.to(device=self.cfg.DEVICE)

            with torch.no_grad():
                scores = self.net(inputs)
                scores = F.sigmoid(scores)
            scores = scores.cpu().squeeze().tolist()
            subdf["pred"] = scores

            # Plot source image with patches and predicted labels and save it
            plt.imshow(picture)
            plt.scatter(w, h, c = scores, s = 5, cmap = cmap, alpha = 0.8, vmin=0, vmax=1)
            plt.savefig(f"{self.cfg.EVAL_IMG_PATH}/{self.cfg.MODEL_NAME}/{image_name}_pred.png")
            plt.close()

            # Make histplot for prevailing visibility estimation
            plt.figure(figsize=(6,4))
            sns.set(style="ticks")
            g = sns.JointGrid(data=subdf, x="distance", y="pred", marginal_ticks=True)
            g.ax_joint.set(xscale="log")
            g.ax_joint.set(ylim=[0,1])
            g.ax_joint.set(xlim=[1,100000])
            g.ax_joint.set(ylabel="Estimated Visibility")
            g.ax_joint.set(xlabel="Distance [m]")
            g.ax_joint.set(xticklabels=["","1","10","100","1'000","10'000","100'000",""])
            g.ax_joint.xaxis.grid(visible=True, which="both")
            g.ax_joint.yaxis.grid(visible=True, which="major")
            cax = g.figure.add_axes([0.15, .15, 0.02, .2])
            g.plot_joint(
                sns.histplot, discrete=(False, False),
                cmap="light:#4A235A", pmax=.8, cbar=True, cbar_ax=cax, bins=(xbins,ybins)
            )
            sns.histplot(data=subdf, x="distance", color="#4A235A", bins=xbins, ax=g.ax_marg_x)
            sns.histplot(data=subdf, y="pred", color="#4A235A", bins=ybins, ax=g.ax_marg_y)
            g.fig.set_figwidth(12)
            g.fig.set_figheight(8)
            plt.tight_layout()
            plt.show()
            plt.savefig(f"{self.cfg.EVAL_HIST_PATH}/{self.cfg.MODEL_NAME}/{image_name}.png")
            g = None
            plt.close()