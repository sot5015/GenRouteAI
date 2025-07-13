import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        device = t.device
        emb = torch.exp(
            torch.arange(half_dim, dtype=torch.float32, device=device)
            * -(torch.log(torch.tensor(10000.0, device=device)) / half_dim)
        )
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb

def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )

class UNet(nn.Module):
    def __init__(self, in_channels=2, time_emb_dim=128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )

        self.time_proj1 = nn.Linear(time_emb_dim, 64)
        self.time_proj2 = nn.Linear(time_emb_dim, 128)
        self.time_proj3 = nn.Linear(time_emb_dim, 256)

        self.enc1 = conv_block(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = conv_block(128, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = conv_block(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = conv_block(128, 64)

        # NO tanh here!
        self.out_conv = nn.Conv2d(64, 1, kernel_size=1)

    def center_crop(self, enc_feature, target_feature):
        _, _, H, W = enc_feature.size()
        _, _, target_H, target_W = target_feature.size()
        diff_H = H - target_H
        diff_W = W - target_W

        crop_top = diff_H // 2
        crop_bottom = crop_top + target_H
        crop_left = diff_W // 2
        crop_right = crop_left + target_W

        return enc_feature[:, :, crop_top:crop_bottom, crop_left:crop_right]

    def forward(self, x, t):
        B, _, H, W = x.shape
        t_emb = self.time_mlp(t.float())

        t1 = self.time_proj1(t_emb).view(B, 64, 1, 1)
        t2 = self.time_proj2(t_emb).view(B, 128, 1, 1)
        t3 = self.time_proj3(t_emb).view(B, 256, 1, 1)

        x1 = self.enc1(x) + t1
        x2 = self.enc2(self.pool1(x1)) + t2
        x3 = self.bottleneck(self.pool2(x2)) + t3

        x = self.up2(x3)
        x2_cropped = self.center_crop(x2, x)
        x = self.dec2(torch.cat([x, x2_cropped], dim=1))

        x = self.up1(x)
        x1_cropped = self.center_crop(x1, x)
        x = self.dec1(torch.cat([x, x1_cropped], dim=1))

        return self.out_conv(x)

# Test run
if __name__ == "__main__":
    x = torch.randn(1, 128, 172, 172)
    x2 = torch.randn(1, 128, 173, 173)

    model = UNet()
    x2_cropped = model.center_crop(x2, x)
    print(x2_cropped.shape)  # Should be [1, 128, 172, 172]