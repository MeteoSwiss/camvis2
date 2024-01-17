import os
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Constants
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)

PATCH_SIZE = 128
DEMAGNIFICATION_FACTORS = (2, 4, 8, 16, 32)

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
DATASET_PATH = f"{DIR_PATH}/../data/processed"

class MultiMagnificationPatches():
    """
    PyTorch dataset class for Multi-Magnification Patches.

    Args:
        val_view (str): View for the validation split.
        test_view (str): View for the test split.

    Attributes:
        state (int): Random seed for resampling positives.
        patch_size (int): Size of the patch.
        factors (list): List of demagnification factors.
        metadata (pd.DataFrame): DataFrame containing metadata information.
        data (dict): Dictionary containing demagnified images and depths.
        normalize (transforms.Normalize): Normalization transform.
        split_metadata (pd.DataFrame): DataFrame containing metadata for the current split.

    Methods:
        resample_positives(): Resample positive samples to balance the dataset.
        train(inference_mode=False): Set the dataset in train split.
        val(): Set the dataset in validation split.
        test(): Set the dataset in test split.
        all(): Set the dataset to include all data.
        __len__(): Return the length of the dataset.
        __getitem__(idx): Get item at the specified index.

    Note:
        This dataset class is designed for handling multi-magnification patch-based data.

    Example:
        >>> multi_magnification_dataset = MultiMagnificationPatches(val_view="val_view", test_view="test_view")
        >>> multi_magnification_dataset.train()
        >>> sample = multi_magnification_dataset[0]
    """

    def __init__(self, val_view=None, test_view=None):
        self.state = RANDOM_SEED
        self.patch_size = PATCH_SIZE
        self.factors = DEMAGNIFICATION_FACTORS

        # Load metadata
        self.metadata = pd.read_csv(f"{DATASET_PATH}/metadata.csv")

        # Load data
        self.data = {}
        self.data["mean"] = torch.empty(0)
        self.data["std"] = torch.empty(0)
        for f in self.factors:
            self.data[f"images_x{f}"], self.data[f"depths_x{f}"] =  torch.load(f"{DATASET_PATH}/data_demagnified_x{f}.pt")
            self.data[f"images_x{f}"] /= 255
            self.data["mean"] = torch.cat([self.data["mean"], self.data[f"images_x{f}"].mean(dim=(0,2,3))])
            self.data["mean"] = torch.cat([self.data["mean"], self.data[f"depths_x{f}"].mean(dim=(0,2,3))])
            self.data["std"] = torch.cat([self.data["std"], self.data[f"images_x{f}"].std(dim=(0,2,3))])
            self.data["std"] = torch.cat([self.data["std"], self.data[f"depths_x{f}"].std(dim=(0,2,3))])
            
        # Normalization
        self.normalize = transforms.Normalize(mean=self.data["mean"], std=self.data["std"])

        # Assign splits to specified views
        self.metadata["split"] = "train"
        if val_view:
            self.metadata.loc[self.metadata["source_view"]==val_view, "split"] = "val"
        if test_view:
            self.metadata.loc[self.metadata["source_view"]==test_view, "split"] = "test"

        # Put in no splitting mode (default)
        self.split_metadata = self.metadata
        
    def resample_positives(self):
        whole_split = self.metadata.loc[self.metadata["split"]=="train"]
        negatives = whole_split.loc[whole_split["label"]==0]
        positives = whole_split.loc[whole_split["label"]==1].sample(len(negatives), random_state=self.state)
        self.split_metadata = pd.concat([positives,negatives]).reset_index(drop=True)
        self.state = self.state + 1
        
    def train(self, inference_mode = False):
        self.split_metadata = self.metadata.loc[self.metadata["split"]=="train"].reset_index(drop=True)
        # Disable resampling when on inference mode
        if not inference_mode:
            self.resample_positives()

    def val(self):
        self.split_metadata = self.metadata.loc[self.metadata["split"]=="val"].reset_index(drop=True)

    def test(self):
        self.split_metadata = self.metadata.loc[self.metadata["split"]=="test"].reset_index(drop=True)

    def all(self):
        self.split_metadata = self.metadata

    def __len__(self):
        return len(self.split_metadata)

    
    def __getitem__(self, idx):
        # Get entry and target
        entry = self.split_metadata.iloc[idx]
        target = entry["label"]
        depth = entry["depth"]

        # Initialize input tensor
        patch = torch.empty((0, self.patch_size, self.patch_size))

        # Build patch
        for f in self.factors:
            # Get coordinates
            h = int(round(entry["h_canvas"]/f))
            w = int(round(entry["w_canvas"]/f))
            h_from = max(h-self.patch_size//2, 0)
            h_to = min(h+self.patch_size//2, self.data[f"images_x{f}"].size(2))
            w_from = max(w-self.patch_size//2, 0)
            w_to = min(w+self.patch_size//2, self.data[f"images_x{f}"].size(3))

            # Get sub patch
            sub_patch = torch.cat(
                [
                    self.data[f"images_x{f}"][entry["image_index"], :, h_from:h_to, w_from:w_to], 
                    self.data[f"depths_x{f}"][entry["depth_index"], :, h_from:h_to, w_from:w_to]
                ]
            )
            padder = (
                -min(w-self.patch_size//2, 0), 
                max((w+self.patch_size//2)-self.data[f"images_x{f}"].size(3), 0), 
                -min(h-self.patch_size//2, 0), 
                max((h+self.patch_size//2)-self.data[f"images_x{f}"].size(2), 0),
            )
            sub_patch = torch.cat(
                [
                    F.pad(sub_patch[:3, :, :], padder, value=0),
                    F.pad(sub_patch[3, :, :], padder, value=-1).unsqueeze(dim=0),
                ],
                dim=0,
            )

            patch = torch.cat([patch, sub_patch], dim=0)
        
        # Normalize input
        patch = self.normalize(patch)

        return patch, target, depth
    
def main():
    """
    Loads data and provides an example of a typical batch of data
    """
    dataset = MultiMagnificationPatches(val_view="1148_1", test_view="1206_4")
    loader = DataLoader(dataset, shuffle=False, batch_size = 8, num_workers=1)
    batch = next(iter(loader))

    print(f"Batch example (batch size of 8) :\n\tpatch : torch.tensor with size {batch[0].size()}\n\ttarget : {batch[1]}\n\tdepth : {batch[2]}")
    
if __name__ == "__main__":
    main()


