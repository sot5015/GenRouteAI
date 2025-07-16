Info:
 
This project processes terrain heightmap images to generate costmaps and uses a diffusion model to predict optimal paths through partially known terrain.

Notes:

Organize your data in `data/heightmaps`, `data/elevation`, and `data/costmaps` folders.
Model checkpoints and results are saved in timestamped folders inside `results/`.

`batchConvert.py`:

Converts raw terrain heightmap images into elevation arrays (`.npy` files).
Uses a reference colorbar to map pixel colors to elevation values.

`heightmapToData.py`

Contains the function `convert_heightmap_to_elevation_array`.
Implements KDTree-based color matching for fast pixel-to-elevation conversion.

`bestPathFind.py`

Computes costmaps from elevation arrays using a custom path cost algorithm.
Uses a Dijkstra-like approach to generate a full cost surface.

`dataset.py`

Defines the `HeightmapDataset` class for training.
Loads heightmap and costmap pairs, resizes them, normalizes, and applies random masks.

`betaSchedule.py`

Defines the beta noise schedule for diffusion (`linear_beta_schedule`).
Provides functions to compute `alpha` and `alpha_bar` for the forward diffusion process.

`forward.py`

Contains `forward_diffusion_sample` to add noise to costmaps.
Implements the core forward process for training the diffusion model.

`model.py`

Defines a U-Net with sinusoidal time embeddings.
Predicts the noise added during diffusion, conditioned on timestep `t` and terrain data.

`train2.py`

Trains the U-Net to predict noise.
Combines MSE loss and mean bias penalty for stable training.
Saves model checkpoints and loss curves.

`runPipeline.py`

Loads a trained model.
Predicts a costmap from a masked heightmap.
Runs A\* path planning on the predicted costmap and visualizes results.


Dependencies:

Python 3.x
PyTorch
NumPy
OpenCV
scikit-learn
Matplotlib
tqdm
