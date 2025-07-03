import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

class HeightmapDataset(Dataset):
    def __init__(self, heightmap_paths, pathmask_paths=None, target_size=(256, 256)):
        self.heightmap_paths = heightmap_paths
        self.pathmask_paths = pathmask_paths
        self.target_size = target_size

    def __len__(self):
        return len(self.heightmap_paths)

    def __getitem__(self, idx):
        # Load heightmap
        heightmap = np.load(self.heightmap_paths[idx])
        heightmap = torch.tensor(heightmap, dtype=torch.float32).unsqueeze(0)

        # Resize
        heightmap = F.interpolate(
            heightmap.unsqueeze(0),
            size=self.target_size,
            mode="bilinear",
            align_corners=False
        ).squeeze(0)

        if self.pathmask_paths:
            path_mask = np.load(self.pathmask_paths[idx])
            path_mask = torch.tensor(path_mask, dtype=torch.float32).unsqueeze(0)
            
            # Resize mask as well
            path_mask = F.interpolate(
                path_mask.unsqueeze(0),
                size=self.target_size,
                mode="nearest"
            ).squeeze(0)
        else:
            path_mask = torch.zeros_like(heightmap)   # dummy tensor, shape (1, H, W)

        return heightmap, path_mask