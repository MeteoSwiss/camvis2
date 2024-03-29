# Copyright (c) 2024 MeteoSwiss, contributors listed in AUTHORS
#
# Distributed under the terms of the BSD 3-Clause License.
#
# SPDX-License-Identifier: BSD-3-Clauseimport argparse

import argparse
import os
import warnings

import torch
from experiment import Experiment

warnings.filterwarnings("ignore")

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
DEPTHS_PATH = f"{DIR_PATH}/../data/raw/depth_maps"
VIEWS = sorted([view.split(".npy")[0] for view in os.listdir(DEPTHS_PATH) if view.endswith(".npy")])

def parse_args():
    """
    Classification of image patches into visible and non-visible patches.
    This function parses command line arguments for model hyperparameters and experiment configuration.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="Model hyperparameters and experiment config parser")

    # Define command line arguments
    parser.add_argument("--ALIGN_FEATURES", dest="ALIGN_FEATURES",
                       action="store_true", 
                       help="If used, allows to encode levels so that features present on the resized images are physically aligned")
    
    parser.add_argument("--AMP", dest="AMP",
                       action="store_true", 
                       help="If used, enables automatic mixed precision for a faster and lighter computation")
    
    parser.add_argument("--BATCH_SIZE", dest="BATCH_SIZE",
                      help="Mini-batch size to feed to the model, prefferably a power of 2 up to 32",
                      default = 32,
                      type=int)
    
    parser.add_argument("--CHECKPOINTS_PATH", dest="CHECKPOINTS_PATH",
                      help="Path to the checkpoints folder",
                      default = f"{DIR_PATH}/../outputs/checkpoints",
                      type=str)
    
    parser.add_argument("--COMPUTE_MODE", dest="COMPUTE_MODE",
                      help="Choose algorithms w.r.t reproducibility of performance",
                      default = "reproducibility",
                      choices = ["reproducibility", "performance"],
                      type=str)

    parser.add_argument("--DATASET_PATH", dest="DATASET_PATH",
                      help="Path to the dataset",
                      default = f"{DIR_PATH}/../data/processed",
                      type=str)

    parser.add_argument("--DEVICE", dest="DEVICE",
                      help="Device to use for forward and backward passes. Can be cpu or cuda:n for gpu n or just cuda if you have only one",
                      default = "cuda",
                      type=str)
    
    parser.add_argument("--EVAL_GROUP", dest="EVAL_GROUP",
                      help="Custom value to group this run with others in the evaluation logs",
                      type=str, default=False, nargs="?")
    
    parser.add_argument("--EVAL_HIST_PATH", dest="EVAL_HIST_PATH",
                      help="Path to the histplots obtained at evaluation time",
                      default = f"{DIR_PATH}/../outputs/val_histplots",
                      type=str)
    
    parser.add_argument("--EVAL_IMG_PATH", dest="EVAL_IMG_PATH",
                      help="Path to the images with evaluated visibility",
                      default = f"{DIR_PATH}/../outputs/val_images",
                      type=str)
    
    parser.add_argument("--EVAL_MAT_PATH", dest="EVAL_MAT_PATH",
                      help="Path to the confusion matrices obtained at evaluation time",
                      default = f"{DIR_PATH}/../outputs/val_cf_matrices",
                      type=str)
    
    parser.add_argument("--EVAL_SCORES_FNAME", dest="EVAL_SCORES_FNAME",
                      help="Path to the logs of evaluation of patches",
                      default = f"eval_scores",
                      type=str)
    
    parser.add_argument("--EVAL_SCORES_PATH", dest="EVAL_SCORES_PATH",
                      help="Path to the logs of evaluation of patches",
                      default = f"{DIR_PATH}/../outputs/val_scores",
                      type=str)

    parser.add_argument("--FACTORS", dest="FACTORS",
                      help="Demagnification factos to be used for input data",
                      choices=[2, 4, 8, 16, 32],
                      nargs='+',
                      type=int,
                      default=[2, 4, 8, 16, 32])
    
    parser.add_argument("--HIDDEN_CHANNELS", dest="HIDDEN_CHANNELS",
                      help="Number of channels in the model for intermediate representation",
                      default = 128,
                      type=int)
    
    parser.add_argument("--LEARNING_RATE", dest="LEARNING_RATE",
                      help="Initial learning rate to update model weights",
                      default = 1e-4,
                      type=float)

    parser.add_argument("--LOAD_CHECKPOINT", dest="LOAD_CHECKPOINT",
                      help="When used, allows to load the pretrained checkpoint that matches the run name",
                      action="store_true")
    
    parser.add_argument("--MIN_LEARNING_RATE", dest="MIN_LEARNING_RATE",
                      help="Minimal learning rate for the scheduler (script stops when the learning rate is below this limit)",
                      default = 1e-5,
                      type=float)
    
    parser.add_argument("--MODEL_NAME", dest="MODEL_NAME",
                      help="When used, allows to specify the name of the model for the log files and checkpoints",
                      type=str, default=False, nargs="?")

    parser.add_argument("--NUM_EPOCHS", dest="NUM_EPOCHS",
                      help="Number of training epochs, -1 for infinite (scheduler ends the training)",
                      default = 100,
                      type=int)
    
    parser.add_argument("--NUM_WORKERS", dest="NUM_WORKERS",
                      help="Number of workers on CPU for data loading",
                      default = 32,
                      type=int)
    
    parser.add_argument("--RANDOM_SEED", dest="RANDOM_SEED",
                      help="Random seed used through the whole experiment",
                      default = 42,
                      type=int)
    
    parser.add_argument("--RAW_IMG_PATH", dest="RAW_IMG_PATH",
                      help="Path to the raw images",
                      default = f"{DIR_PATH}/../data/raw/images",
                      type=str)

    parser.add_argument("--RUN_MODE", dest="RUN_MODE",
                      choices=["train", "eval"],
                      help="whether to train or evaluate a  model",
                      default = "train",
                      type=str)
    
    parser.add_argument("--SAVE_CHECKPOINTS", dest="SAVE_CHECKPOINTS",
                       action="store_true", help="Whether or not to save trained checkpoints")
    
    parser.add_argument("--SELECTION_FUNCTION", dest="SELECTION_FUNCTION",
                        choices=["loss", "acc", "f1"], 
                        help="The function that is used to evaluate whether to save checkpoints",
                        default="acc",
                        type=str)

    parser.add_argument("--SHARE_WEIGHTS", dest="SHARE_WEIGHTS",
                       action="store_true", 
                       help="If used, allows to use shared weights for the encoding of each magnification level")

    parser.add_argument("--TRAIN_FAST", dest="TRAIN_FAST",
                       action="store_true", 
                       help="If used, sacrifices train-time metrics evaluation on the train set to decrease epoch durations")
    
    parser.add_argument("--TRAIN_LOGS", dest="TRAIN_LOGS",
                      action="store_true", help="Whether or not to use tensorboard logging")
    
    parser.add_argument("--TRAIN_LOGS_PATH", dest="TRAIN_LOGS_PATH",
                      help="Path to the logs",
                      default = f"{DIR_PATH}/../outputs/training_logs",
                      type=str)
    
    parser.add_argument("--USE_MLP", dest="USE_MLP",
                       action="store_true", 
                       help="If used, allows to use an MLP instead of a linear layer at the tail of the model")
    
    parser.add_argument("--VAL_FOLD", dest="VAL_FOLD",
                      help="Index of fold (a.k.a. camear view) to be used for validation",
                      default=1,
                      type=int)

    # Parse the command line arguments
    args = parser.parse_args()
    return args

def do_checks(cfg):
    """
    Perform pre-run checks and configurations based on the provided configuration (cfg).

    Args:
        cfg (argparse.Namespace): Configuration settings obtained from command line arguments.

    Returns:
        argparse.Namespace: Updated configuration after performing checks and configurations.
    """
    # Check if ALIGN_FEATURES is suitable when FACTORS is not a sequence of powers of 2 with a step of +1 in exponent
    if cfg.ALIGN_FEATURES & (not any([[2, 4, 8, 16, 32][i:i+len(cfg.FACTORS)] == cfg.FACTORS for i in range(6-len(cfg.FACTORS))])):
        raise Exception("ALIGN_FEATURES is not suitable since FACTORS is not a sequence of powers of 2 with a step of +1 in exponent")

    # Check if VAL_FOLD is within valid range for k-fold cross-validation
    if cfg.VAL_FOLD < 1 or cfg.VAL_FOLD > len(VIEWS):
        raise Exception(f"Available data only allows for k-fold cross validation with k between 1 and {len(VIEWS)}")
    
    # Set VAL_VIEW based on VAL_FOLD
    cfg.VAL_VIEW = VIEWS[cfg.VAL_FOLD-1]

    # Use CPU if no GPU is available
    if (cfg.DEVICE != "cpu") & (not torch.cuda.is_available()):
        cfg.DEVICE = "cpu"
        print("No GPU found, using CPU")

    # Set default MODEL_NAME if not provided
    if not cfg.MODEL_NAME:
        subdirectories = [d for d in os.listdir(cfg.TRAIN_LOGS_PATH) if os.path.isdir(os.path.join(cfg.TRAIN_LOGS_PATH, d))]
        cfg.MODEL_NAME = f"run_{len(subdirectories)}"

    # Set default EVAL_GROUP if not provided
    if not cfg.EVAL_GROUP:
        cfg.EVAL_GROUP = cfg.EVAL_SCORES_FNAME

    # If running in evaluation mode, create CSV file for evaluation scores if it doesn't exist
    if cfg.RUN_MODE == "eval":
        if not os.path.exists(f"{cfg.EVAL_SCORES_PATH}/{cfg.EVAL_SCORES_FNAME}.csv"):
            columns = "model_name,fold,view,loss,acc,f1,group"
            with open(f"{cfg.EVAL_SCORES_PATH}/{cfg.EVAL_SCORES_FNAME}.csv", "w") as file:
                file.write(columns)

    return cfg

def main():
    """
    Main entry point for the script.
    - Parses command line arguments.
    - Performs pre-run checks and configurations.
    - Initiates and runs the experiment based on the specified mode ("train" or "eval").
    """
    # Configuration
    cfg = parse_args()
    cfg = do_checks(cfg)

    # Run
    if cfg.RUN_MODE == "train":
        experiment = Experiment(cfg)
        experiment.train()
    elif cfg.RUN_MODE == "eval":
        experiment = Experiment(cfg)
        experiment.eval()

if __name__ == "__main__":
    main()