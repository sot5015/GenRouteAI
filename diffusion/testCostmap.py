import numpy as np
import matplotlib.pyplot as plt


import numpy as np
import matplotlib.pyplot as plt

costmap = np.load("data/costmaps/hei1_costmap.npy")
mask = costmap < 1e6
plt.imshow(np.where(mask, costmap, np.nan), cmap="viridis")
plt.colorbar(label="Cost to reach")
plt.title("Full Costmap from (0,0)")
plt.show()