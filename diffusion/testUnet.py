# Class for (for us) testing if the UNet model in the diffusion package works
from model import UNet
import torch
import matplotlib.pyplot as plt

model = UNet()
x = torch.randn(1, 1, 64, 64)
t = torch.randint(0, 1000, (1,))
out = model(x, t).detach().squeeze()

fig, axs = plt.subplots(1, 2)
axs[0].imshow(x.squeeze(), cmap='gray')
axs[0].set_title("Input")
axs[1].imshow(out, cmap='viridis')
axs[1].set_title("Output")
plt.show()