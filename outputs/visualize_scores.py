# Copyright (c) 2024 MeteoSwiss, contributors listed in AUTHORS
#
# Distributed under the terms of the BSD 3-Clause License.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import os

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

def make_plot(df, x, y):
    """
    Create a scatter plot with stripplot from the Seaborn library to visualize performance per fold.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        x (str): The column name for the x-axis.
        y (str): The column name for the y-axis.

    Returns:
        None

    Note:
        This function generates a scatter plot using Seaborn's stripplot to visualize the performance
        per fold based on the specified DataFrame and column names for the x and y axes. It includes
        customizations such as color-coded folds, a legend with unique fold values, and annotations
        for cameras. The plot is displayed with tight layout and a grid along the y-axis.

    Example:
        >>> make_plot(df, x="group", y="acc")
    """
    # Create a colormap
    colormap = cm.get_cmap('Set3', len(df["fold"].unique()))

    # Create a scatter plot
    plt.figure(figsize=(9, 5))

    sns.stripplot(data=df, x=x, y=y, hue="fold", palette=colormap, dodge=False, jitter=0.3)

    # Set labels and title
    plt.xlabel(f"Model group")
    plt.ylabel(f"Performance : {y}")
    plt.title('Validation Performance per Fold')

    # Manually create a legend with all unique "fold" values
    views = sorted(df["view"].unique().tolist())
    handles = []
    labels = []
    for i in range(len(views)):
        handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colormap(i), markersize=10))
        labels.append(f"fold {i+1}, cam : {views[i][:4]}, angle : {views[i][5]}")

    # Show legend
    plt.legend(handles, labels, title='Fold', bbox_to_anchor=(1.05, 1), loc='upper left')

    x_offsets = [0.6, 1.7, 2.7, 3.8, 4.8, 5.9, 6.9]
    # Add text for cameras
    plt.text(
        x_offsets[df["group"].nunique()-1],
        df[y].min(), 
        s="1148 : Chateau-d'oex\n1157 : Guetsch-Andermatt\n1159 : La Dôle\n1198 : Casaccia\n1206 : Torrent", 
        bbox=dict(facecolor='white', alpha=0.5),
    )

    plt.tight_layout()
    plt.grid(axis="y")

def parse_args():
    """
    Parse command-line arguments for visualization of scores.

    Returns:
        argparse.Namespace: Parsed arguments.

    Example:
        >>> args = parse_args()
        >>> print(args.EVAL_SCORES_FNAME)
        eval_scores
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-n", "--EVAL_SCORES_FNAME", dest="EVAL_SCORES_FNAME",
                      help="File name of the logs of patches evaluation",
                      default = f"eval_scores",
                      type=str)
    
    parser.add_argument("-v", "--VIZUALIZE", dest="VIZUALIZE",
                       action="store_true", 
                       help="If used, displays generated graphs")
    
    parser.add_argument("-s", "--SAVE", dest="SAVE",
                      help="If used, saves the generated graphs",
                      action="store_true", )

    args = parser.parse_args()
    return args

def main():
    """
    Main function for visualizing evaluation scores.

    Reads command-line arguments, loads evaluation scores from a CSV file, and creates plots based on specified metrics.

    Example:
        >>> main()
    """
    cfg = parse_args()
    df = pd.read_csv(f"{DIR_PATH}/val_scores/{cfg.EVAL_SCORES_FNAME}.csv")

    
    for metric in ["acc", "f1", "loss"]:
        make_plot(df, x="group", y=metric)
        if cfg.SAVE:
            if not os.path.exists(f"{DIR_PATH}/val_scores/{cfg.EVAL_SCORES_FNAME}"):
                os.makedirs(f"{DIR_PATH}/val_scores/{cfg.EVAL_SCORES_FNAME}")
            plt.savefig(f"{DIR_PATH}/val_scores/{cfg.EVAL_SCORES_FNAME}/{metric}.png")
        if cfg.VIZUALIZE:
            plt.show()
        plt.close()



if __name__ == "__main__":
    main()