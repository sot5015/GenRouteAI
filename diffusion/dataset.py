import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

class HeightmapDataset(Dataset):
    def __init__(self, heightmap_paths, costmap_paths, target_size=(256, 256), apply_masking=True):
        self.heightmap_paths = heightmap_paths
        self.costmap_paths = costmap_paths
        self.target_size = target_size
        self.apply_masking = apply_masking

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

        # Normalize
        heightmap_min = heightmap.min()
        heightmap_max = heightmap.max()
        heightmap = (heightmap - heightmap_min) / (heightmap_max - heightmap_min + 1e-8)

        # Load costmap
        costmap = np.load(self.costmap_paths[idx])
        costmap = torch.tensor(costmap, dtype=torch.float32).unsqueeze(0)
        costmap = F.interpolate(
            costmap.unsqueeze(0),
            size=self.target_size,
            mode="bilinear",
            align_corners=False
        ).squeeze(0)

        # Optional: Masking
        if self.apply_masking:
            mask = self.generate_random_mask(heightmap.shape[-2:])
            masked_heightmap = heightmap * mask + (1 - mask) * 0.5  # fill missing with mean
        else:
            masked_heightmap = heightmap

        return masked_heightmap, costmap, mask

    def generate_random_mask(self, shape, block_size=32, num_blocks=5):
        """
        Generate a mask with random square holes.
        """
        mask = torch.ones(shape, dtype=torch.float32)
        H, W = shape
        for _ in range(num_blocks):
            top = np.random.randint(0, max(1, H - block_size))
            left = np.random.randint(0, max(1, W - block_size))
            mask[top:top+block_size, left:left+block_size] = 0.0
        return mask