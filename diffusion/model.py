import torch
import torch.nn as nn
import torch.nn.functional as F

# Positional embedding for timestep t
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -(torch.log(torch.tensor(10000.0)) / half_dim))
        emb = t[:, None] * emb[None, :]  # shape [B, dim/2]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb  # shape [B, dim]

# Basic convolutional block
def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU()
    )

# U-Net with timestep embeddings
class UNet(nn.Module):
    def __init__(self, in_channels=1, time_emb_dim=128):
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

        self.out_conv = nn.Conv2d(64, in_channels, kernel_size=1)

    def forward(self, x, t):
        B, _, H, W = x.shape
        t_emb = self.time_mlp(t.float())

        t1 = self.time_proj1(t_emb).view(B, 64, 1, 1)
        t2 = self.time_proj2(t_emb).view(B, 128, 1, 1)
        t3 = self.time_proj3(t_emb).view(B, 256, 1, 1)

        # Encoder
        x1 = self.enc1(x)              # [B, 64, H, W]
        x1 = x1 + t1                   # add time after encoding
        x2 = self.pool1(x1)
        x2 = self.enc2(x2)
        x2 = x2 + t2

        # Bottleneck
        x3 = self.pool2(x2)
        x3 = self.bottleneck(x3)
        x3 = x3 + t3

        # Decoder
        x = self.up2(x3)
        x = self.dec2(torch.cat([x, x2], dim=1))

        x = self.up1(x)
        x = self.dec1(torch.cat([x, x1], dim=1))

        return self.out_conv(x)

# Optional test
if __name__ == "__main__":
    model = UNet()
    x = torch.randn(4, 1, 64, 64)
    t = torch.randint(0, 1000, (4,))
    out = model(x, t)
    print("Output shape:", out.shape)  # should be [4, 1, 64, 64]