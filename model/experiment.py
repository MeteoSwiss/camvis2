
import os
import random
from time import time

import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
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

    def bootstrap_infer(self, dataloader, num_iterations = 10, num_samples = False, random_state = False, cf_mat=True):
        """Perform inference with bootstrapping for a more interpretable evaluation of the performance"""

        # Initialize variables to store performances
        concat_loss = torch.zeros(num_iterations)
        concat_acc = torch.zeros(num_iterations)
        concat_f1 = torch.zeros(num_iterations)
        
        # Get initial random state and all predictions and targets
        if not random_state:
            random_state = 42
        _, _, _, preds, targets = self.infer_round(dataloader)

        if cf_mat:
            self.make_cf_mat(preds, targets)
            num_bins = 5
            quantiles = [i/num_bins for i in range(num_bins+1)]
            values = self.dataset.metadata["depth"].quantile(quantiles, interpolation="higher").tolist()
            depths = torch.tensor(self.dataset.split_metadata["depth"].tolist())
            df = pd.DataFrame(columns=["bin","count","rate","tfpn"])
            tfpns = ["TN", "FP", "FN", "TP"]
            bin = 0
            dist_ranges = []
            for i, j in zip(values[:-1], values[1:]):
                bin += 1
                dist_ranges.append(f"{int(i)}-{int(j)}")
                sub_preds = preds[(depths>=i) & (depths<j)]
                sub_targets = targets[(depths>=i) & (depths<j)]
                mat_name = self.cfg.RUN_NAME + f"_{int(i)}-{int(j)}"
                counts = self.make_cf_mat(sub_preds, sub_targets, mat_name)
                rates = counts/counts.sum()
                for tfpn in tfpns:
                    df.loc[len(df)] = [bin, counts[tfpns.index(tfpn)], rates[tfpns.index(tfpn)], tfpn]
            
            sub_df = df.pivot(index="bin",columns="tfpn",values="rate").dropna()
            #sns.set_theme()
            ax = sub_df.plot.area(colormap = cm.get_cmap('Set3', len(sub_df.columns)), grid=True)
            ax.set_xticks([i+1 for i in range(len(dist_ranges))])
            ax.set_xticklabels(dist_ranges)
            plt.title("Evolution of TN, FP, FN and TP rates across bins")
            plt.tight_layout()
            plt.savefig(self.cfg.INFERENCE_PATH + "lineplots/" + self.cfg.RUN_NAME + "_tfpn.png")
            plt.close()

            df["tf"] = df["tfpn"].apply(lambda x : x[0])
            sub_df = df.groupby(["bin","tf"])["rate"].sum().reset_index()
            sub_df = sub_df.pivot(index="bin",columns="tf",values="rate").dropna()
            #sns.set_theme()
            ax = sub_df.plot.area(colormap = cm.get_cmap('Set3', len(sub_df.columns)), grid=True)
            ax.set_xticks([i+1 for i in range(len(dist_ranges))])
            ax.set_xticklabels(dist_ranges)
            plt.title("Evolution of True / False predictions across bins")
            plt.tight_layout()
            plt.savefig(self.cfg.INFERENCE_PATH + "lineplots/" + self.cfg.RUN_NAME + "_tf.png")
            plt.close()

            df["pn"] = df["tfpn"].apply(lambda x : x[1])
            sub_df = df.groupby(["bin","pn"])["rate"].sum().reset_index()
            sub_df = sub_df.pivot(index="bin",columns="pn",values="rate").dropna()
            #sns.set_theme()
            ax = sub_df.plot.area(colormap = cm.get_cmap('Set3', len(sub_df.columns)), grid=True)
            ax.set_xticks([i+1 for i in range(len(dist_ranges))])
            ax.set_xticklabels(dist_ranges)
            plt.title("Evolution of Positive / Negative predictions across bins")
            plt.tight_layout()
            plt.savefig(self.cfg.INFERENCE_PATH + "lineplots/" + self.cfg.RUN_NAME + "_pn.png")
            plt.close()


        # Set random numbers generator and use a large seed number
        generator = torch.Generator()
        generator.manual_seed(random_state**5)


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

    def make_cf_mat(self, preds, targets, name=None):
        """Building of confusion matrix and saving as an image"""
        cf_mat = confusion_matrix(targets.round().int(), preds.round().int(), labels=[0,1])
        disp = ConfusionMatrixDisplay(confusion_matrix=cf_mat, display_labels=["nonvisible:0","visible:1"])
        disp.plot()
        plt.tight_layout()
        if name:
            plt.savefig(self.cfg.INFERENCE_PATH + "cf_mats/" + name + ".png")
        else:
            plt.savefig(self.cfg.INFERENCE_PATH + "cf_mats/" + self.cfg.RUN_NAME + ".png")
        plt.close()
        return cf_mat.flatten()

    def eval(self):
        """Evaluation of inference"""
        
        # Infer on images
        self.dataset.val()
        self.infer_on_images()
        
        # Validate
        self.dataset.val()
        infer_dataloader = DataLoader(self.dataset, shuffle=False, batch_size=self.cfg.BATCH_SIZE, num_workers=self.cfg.NUM_WORKERS, 
                                           pin_memory=False, prefetch_factor=2, persistent_workers=False)
        
        # Bootstrap phase
        loss, acc, f1 = self.bootstrap_infer(infer_dataloader)
        self.infer_dataloader = None
        
        
        with open(self.cfg.INFERENCE_PATH + "scores.csv", "a") as logs_file:
            for i in range(len(loss)):
                logs_file.write(f"\n{self.cfg.RUN_NAME},{self.cfg.VIEW},val,{loss[i].item():.4f},{acc[i].item():.4f},{f1[i].item():.4f}")
        
        
        exit()

    
    def infer_on_images(self):
        """Perform inferenc on the images of the current dataset subset and save them"""

        # Check if directory exists for this run, if not, make one
        if not os.path.exists(self.cfg.INFERENCE_PATH + f"test_images/{self.cfg.RUN_NAME}"):
            os.makedirs(self.cfg.INFERENCE_PATH + f"test_images/{self.cfg.RUN_NAME}")
        if not os.path.exists(self.cfg.INFERENCE_PATH + f"histplots/{self.cfg.RUN_NAME}"):
            os.makedirs(self.cfg.INFERENCE_PATH + f"histplots/{self.cfg.RUN_NAME}")

        # Colormap for plotting
        cmap = LinearSegmentedColormap.from_list("", ["fuchsia", "yellow", "lightgreen"])

        # Bins for hist plots
        #xbins = np.log10(np.exp(((np.log(100000)-np.log(5))*np.linspace(0, 255, 51)/255)+np.log(5)))
        xbins = np.log10(100000)*np.linspace(0, 100, 51)/100
        ybins = np.linspace(0,1,11)

        # Get split data and source image names
        df = self.dataset.split_metadata.copy()
        names = df["source_image"].unique().tolist()
        df["distance"] = df["depth"]

        # Set model to evaluation mode
        self.net.eval()

        # Dataframe to save inference data on all images
        all_inference_data = pd.read_csv(self.cfg.INFERENCE_PATH + f"all_points.csv")

        # Process source images
        for image_name in tqdm(names):

            # Get source image
            picture = cv2.imread(self.cfg.DATASET_PATH.rsplit("/",2)[0] + f"/2024-01-11_all_images/{image_name}.jp2")[:,:,::-1].copy()
                
            # Get corresponding patches and labels
            subdf = df.loc[df["source_image"] == image_name]
            h = [int(x) for x in subdf["h_coord"].tolist()]
            w = [int(x) for x in subdf["w_coord"].tolist()]
            label = subdf["label"].tolist()

            # Plot source image with patches and ground truth and save it
            plt.imshow(picture)
            plt.scatter(w, h, c = label, s = 5, cmap = cmap, alpha = 0.8, vmin=0, vmax=1)
            plt.savefig(self.cfg.INFERENCE_PATH + f"test_images/{self.cfg.RUN_NAME}/{image_name}_true.png")
            plt.close()

            # Gather inputs
            self.dataset.split_metadata = subdf
            dataloader = DataLoader(self.dataset, shuffle=False, batch_size = len(subdf), num_workers=self.cfg.NUM_WORKERS)
            inputs, _, _ = next(iter(dataloader))

            # Put to CUDA and infer
            inputs = inputs.to(device=self.cfg.DEVICE)

            with torch.no_grad():
                scores = self.net(inputs)
                scores = F.sigmoid(scores)
            scores = scores.cpu().squeeze().tolist()
            subdf["pred"] = scores

            # Plot source image with patches and predicted labels and save it
            plt.imshow(picture)
            plt.scatter(w, h, c = scores, s = 5, cmap = cmap, alpha = 0.8, vmin=0, vmax=1)
            plt.savefig(self.cfg.INFERENCE_PATH + f"test_images/{self.cfg.RUN_NAME}/{image_name}_pred.png")
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

            # Add the joint and marginal histogram plots
            g.plot_joint(
                sns.histplot, discrete=(False, False),
                cmap="light:#4A235A", pmax=.8, cbar=True, cbar_ax=cax, bins=(xbins,ybins)#binwidth=(0.1,0.05) #cmap="light:#03012d"
            )
            sns.histplot(data=subdf, x="distance", color="#4A235A", bins=xbins, ax=g.ax_marg_x) #element="step", legend=False
            #g.ax_marg_x.set(ylabel=None)
            sns.histplot(data=subdf, y="pred", color="#4A235A", bins=ybins, ax=g.ax_marg_y)
            #g.plot_marginals(sns.histplot, color="#4A235A", element="step", )

            g.fig.set_figwidth(12)
            g.fig.set_figheight(8)
            
            plt.tight_layout()
            plt.show()
            plt.savefig(self.cfg.INFERENCE_PATH + f"histplots/{self.cfg.RUN_NAME}/{image_name}.png")
            g = None
            plt.close()

            all_inference_data = pd.concat([all_inference_data, subdf])
            
        all_inference_data.to_csv(self.cfg.INFERENCE_PATH + f"all_points.csv", index=False)