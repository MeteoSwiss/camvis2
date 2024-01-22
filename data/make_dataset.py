# Copyright (c) 2024 MeteoSwiss, contributors listed in AUTHORS
#
# Distributed under the terms of the BSD 3-Clause License.
#
# SPDX-License-Identifier: BSD-3-Clauseimport argparse

import os

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Constants
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

PATCH_SIZE = 128
PATCH_NUMBER = 500
IMAGE_SHAPE = (4140, 6198)
DEMAGNIFICATION_FACTORS = (2, 4, 8, 16, 32)

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
IMAGES_PATH = f"{DIR_PATH}/raw/images"
DEPTHS_PATH = f"{DIR_PATH}/raw/depth_maps"
OUTPUTS_PATH = f"{DIR_PATH}/processed"

def check_paths():
    """
    Checks if necessary directories exist and creates them if needed

    Args: 
        None

    Returns:
        None
    """

    for dir in [IMAGES_PATH, DEPTHS_PATH, OUTPUTS_PATH]:
        if not os.path.exists(dir):
            os.makedirs(dir)

def get_image_data(img_name):
    """
    Retrieves an image along with its mask and depth map.

    Args:
        img_name (str): Image name without file extension. The structure of img_name is as follows:
            "{webcam_id}_{camera_angle}_{timestamp}"

            - webcam_id (int): Webcam identifier.
            - camera_angle (int): Camera angle identifier.
            - timestamp (str): Timestamp in the format "YYYY-MM-DD_HHMM".

    Returns:
        tuple: A tuple containing three NumPy arrays:
            - img (np.ndarray): RGB image with three channels (shape: (h, w, 3)).
            - mask (np.ndarray): Image mask (shape: (h, w)).
            - depth (np.ndarray): Depth map for the image view with four channels:
                - Channel 1: Actual depth in meters (shape: (h, w)).
                - Channels 2-4: WGS84 Cartesian XYZ coordinates of the pixel (shape: (h, w, 3)).

    Example:
        >>> img, mask, depth = get_image_data("1148_1_2022-08-06_0800")
    """

    # Image
    img = cv2.imread(f"{IMAGES_PATH}/{img_name}.jp2")
    img = img[:,:,::-1].copy() # BGR to RGB

    # Mask
    visible = cv2.imread(f"{IMAGES_PATH}/{img_name}_0002_visible.png", cv2.IMREAD_UNCHANGED)[:,:,3]
    nonvisible = cv2.imread(f"{IMAGES_PATH}/{img_name}_0001_nonvisible.png", cv2.IMREAD_UNCHANGED)[:,:,3]
    mask = np.ones_like(visible)*2
    mask[visible>1] = 1
    mask[nonvisible>1] = 0

    # Depth map and WGS84 coordinates
    depth = np.load(f"{DEPTHS_PATH}/{img_name[:6]}.npy")

    return img, mask, depth

def sample_patches(mask, depth):
    """
    Samples patches of data from an input image, mask, and depth map.

    Args:
        mask (np.ndarray): Image mask (shape: (h, w)).
        depth (np.ndarray): Depth map for the image view with four channels:
            - Channel 1: Actual depth in meters (shape: (h, w)).
            - Channels 2-4: WGS84 Cartesian XYZ coordinates of the pixel (shape: (h, w, 3)).

    Returns:
        tuple: A tuple containing four NumPy arrays:
            - canvas_coords (np.ndarray): Coordinates of sampled central pixels (shape: (2, PATCH_NUMBER)).
            - labels (np.ndarray): Labels corresponding to the sampled central pixels (shape: (PATCH_NUMBER,)).
            - depths (np.ndarray): Depth values corresponding to the sampled central pixels (shape: (PATCH_NUMBER,)).
            - WGS84_coords (np.ndarray): WGS84 Cartesian XYZ coordinates corresponding to the sampled central pixels (shape: (PATCH_NUMBER, 3)).

    Note:
        This function samples central pixels from the input image, excluding invalid pixels based on the mask and depth map.
        It assigns weights to pixels based on their scarcity in the depth-adjusted map and ensures that central pixels are
        sampled with probabilities proportional to these weights. The sampled coordinates, labels, depths, and WGS84 coordinates
        are returned in separate arrays.

    Example:
        >>> canvas_coords, labels, depths, WGS84_coords = sample_patches(img, mask, depth)
    """

    # Set pixels that are eligible for central pixels
    valid_pixels = np.ones_like(mask)
    valid_pixels[mask == 2] = 0
    valid_pixels[np.isnan(depth[: , :, 0])] = 0
    valid_pixels[:PATCH_SIZE//2, :] = 0
    valid_pixels[-PATCH_SIZE//2:, :] = 0
    valid_pixels[:, :PATCH_SIZE//2] = 0
    valid_pixels[:, -PATCH_SIZE//2:] = 0

    # Scale depth and create special depth value for non valid pixels (max depth value + 1)
    depth_adjusted = (np.log(depth[:, :, 0])/np.log(100000)*255).round()
    depth_adjusted[valid_pixels == 0] = depth_adjusted[valid_pixels != 0].max()+1
    depth_adjusted = depth_adjusted.astype("int")
    
    # Make weights matrix for sampling
    values, counts = np.unique(depth_adjusted, return_counts=True)
    scarcity = np.zeros(values.max()+1)
    scarcity[values] = 1/counts
    weights = scarcity[depth_adjusted]

    # Disable sampling of invalid pixels and rescale weights
    weights[valid_pixels == 0] = 0
    weights /= weights.sum()

    # Sample central pixels
    indices = np.random.choice(np.arange(len(depth_adjusted.flatten())), PATCH_NUMBER, replace=False, p=weights.flatten())
    dummy = np.zeros_like(depth_adjusted)
    dummy.ravel()[indices] = 1
    canvas_coords = np.array(np.where(dummy==1))
    
    # Extract data
    labels = mask[canvas_coords[0], canvas_coords[1]]
    depths = depth[canvas_coords[0], canvas_coords[1], 0]
    WGS84_coords = depth[canvas_coords[0], canvas_coords[1], 1::].T

    return canvas_coords, labels, depths, WGS84_coords

def process_metadata():
    """
    Processes metadata for images, extracting information from image data and generating a DataFrame.

    Returns:
        pd.DataFrame: A DataFrame containing metadata for sampled patches from images, with columns:
            - 'patch_index': Index of the patch in the DataFrame.
            - 'label': Image label corresponding to the sampled central pixel.
            - 'depth': Depth value corresponding to the sampled central pixel.
            - 'h_canvas': Vertical coordinate of the sampled central pixel on the canvas.
            - 'w_canvas': Horizontal coordinate of the sampled central pixel on the canvas.
            - 'source_image': Original image name.
            - 'source_camera': Webcam identifier.
            - 'source_position': Camera angle identifier.
            - 'source_view': Camera location and angle combination.
            - 'source_date': Date of image capture.
            - 'source_time': Time of image capture.
            - 'x_WGS84': WGS84 Cartesian X coordinate of the sampled central pixel.
            - 'y_WGS84': WGS84 Cartesian Y coordinate of the sampled central pixel.
            - 'z_WGS84': WGS84 Cartesian Z coordinate of the sampled central pixel.
            - 'image_index': Index of the image featuring the patch. 
            - 'depth_index': Index of the depth map featuring the patch.
    """

    metadata = pd.DataFrame()
    img_names = sorted([elem[:-4] for elem in os.listdir(IMAGES_PATH) if elem.endswith('.jp2')])
    pbar = tqdm(img_names)

    for img_name in pbar:
        img_metadata = pd.DataFrame()

        # Process image data
        pbar.set_description(f"Processing image {img_name}")
        _, mask, depth = get_image_data(img_name)
        canvas_coords, labels, depths, WGS84_coords = sample_patches(mask, depth)

        # Populate metadata DataFrame
        img_metadata["label"] = labels
        img_metadata["depth"] = depths
        img_metadata["h_canvas"] = canvas_coords[0]
        img_metadata["w_canvas"] = canvas_coords[1]
        img_metadata["source_image"] = img_name
        img_metadata["source_camera"] = img_name[:4]
        img_metadata["source_position"] = img_name[5]
        img_metadata["source_view"] = img_name[:6]
        img_metadata["source_date"] = img_name[7:-5]
        img_metadata["source_time"] = img_name[-4:]
        img_metadata["x_WGS84"] = WGS84_coords[0]
        img_metadata["y_WGS84"] = WGS84_coords[1]
        img_metadata["z_WGS84"] = WGS84_coords[2]

        metadata = img_metadata if metadata.empty else pd.concat([metadata, img_metadata])

    # Set an image index
    temp = metadata["source_image"].unique().tolist()
    metadata["image_index"] = metadata["source_image"].apply(lambda x : temp.index(x))

    # Set a depth map index
    temp = metadata[["source_camera","source_position"]].drop_duplicates().sort_values(by=["source_camera","source_position"])
    temp = temp.reset_index(drop=True).reset_index().rename(columns={"index":"depth_index"})
    metadata = pd.merge(metadata, temp, how="left", left_on=["source_camera","source_position"], right_on=["source_camera","source_position"])

    # Set indices and save
    metadata = metadata.reset_index().rename(columns={"index": "patch_index"})
    metadata = metadata.astype({"image_index":"int", "depth_index":"int"})
    metadata.to_csv(f"{OUTPUTS_PATH}/metadata.csv", index=False)

    return metadata

def process_data(metadata):
    """
    Processes data by resizing depth maps and images based on demagnification factors, 
    and saves the resulting tensors to PyTorch files.

    Args:
        metadata (pd.DataFrame): DataFrame containing metadata for sampled patches from images.

    Returns:
        None

    Note:
        This function iterates over specified demagnification factors, resizes depth maps and images accordingly,
        and saves the resulting tensors to PyTorch files. The progress is tracked using a tqdm progress bar.

    Example:
        >>> process_data(metadata)
    """

    pbar = tqdm(total=(len(DEMAGNIFICATION_FACTORS) * (metadata["image_index"].nunique() + 1)))

    for factor in DEMAGNIFICATION_FACTORS:
        # Calculate resized height and width based on demagnification factor
        h = int(np.round(IMAGE_SHAPE[0]/factor))
        w = int(np.round(IMAGE_SHAPE[1]/factor))

        # Initialize tensors for depth maps and images
        depths = torch.empty(size=(metadata["depth_index"].nunique(), 1, h, w))
        images = torch.empty(size=(metadata["image_index"].nunique(), 3, h, w))

         # Process depth maps
        pbar.set_description(f"Processing depth maps with x{factor} de-magnification")
        pool = metadata[["source_camera", "source_position", "depth_index"]].drop_duplicates()
        for i, row in pool.set_index("depth_index").iterrows():
            depth = np.load(f"{DEPTHS_PATH}/{str(row['source_camera'])}_{str(row['source_position'])}.npy")[:,:,0]
            depth = np.log(depth)
            depth = np.nan_to_num(depth, nan=-1)
            resized = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
            depths[i,:,:,:] = torch.from_numpy(resized.copy()).unsqueeze(dim=0)
        pbar.update(1)

        # Process images
        pool = metadata[["source_image", "image_index"]].drop_duplicates()
        for i, row in pool.set_index("image_index").iterrows():
            pbar.set_description(f"Processing image {row['source_image']} with x{factor} de-magnification")
            image = cv2.imread(f"{IMAGES_PATH}/{row['source_image']}.jp2")
            resized = cv2.resize(image, (w, h), interpolation=cv2.INTER_LANCZOS4)
            images[i,:,:,:] = torch.from_numpy(resized[:,:,::-1].copy()).permute(2,0,1)
            pbar.update(1)

        # Save processed data to PyTorch file
        pbar.set_description(f"Saving data with x{factor} de-magnification")
        torch.save((images, depths),f"{OUTPUTS_PATH}/data_demagnified_x{factor}.pt")

def main():
    """
    Creates the dataset
    """

    check_paths()
    print("Sampling patches ...")
    metadata = process_metadata()
    print("Creating model input data ..")
    process_data(metadata)
    print("Dataset ready")


if __name__ == '__main__':
    main()
