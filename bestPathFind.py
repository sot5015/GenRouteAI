import numpy as np
import heapq
from tqdm import tqdm
import matplotlib.pyplot as plt

def min_sum_path_with_trace(npy_path, start):
    grid = np.load(npy_path, allow_pickle=True)
    rows, cols = grid.shape
    x1, y1 = start

    total_pixels = rows * cols
    visited_pixels = set()

    queue = []
    heapq.heappush(queue, (0, x1, y1))

    costs = {(x1, y1): 0}
    came_from = {(x1, y1): None}

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    pbar = tqdm(total=total_pixels, desc="Exploring pixels")

    while queue:
        cost, x, y = heapq.heappop(queue)

        if (x, y) not in visited_pixels:
            visited_pixels.add((x, y))
            pbar.update(1)

        for dx, dy in directions:
            nx, ny = x + dx, y + dy

            if 0 <= nx < rows and 0 <= ny < cols:

                if grid[x, y] < grid[nx, ny]:
                    step_cost = (grid[nx, ny] - grid[x, y]) * 0.1
                elif grid[x, y] == grid[nx, ny]:
                    step_cost = 1
                else:
                    step_cost = (grid[x, y] - grid[nx, ny]) * 0.1

                new_cost = cost + step_cost

                if (nx, ny) not in costs or new_cost < costs[(nx, ny)]:
                    costs[(nx, ny)] = new_cost
                    came_from[(nx, ny)] = (x, y)
                    heapq.heappush(queue, (new_cost, nx, ny))

    pbar.close()

    # Build the full costmap
    costmap = np.full_like(grid, np.inf, dtype=np.float32)
    for (px, py), c in costs.items():
        costmap[px, py] = c

    return costmap

if __name__ == "__main__":
    costmap = min_sum_path_with_trace("data/elevation/hei1.npy", (0, 0))
    
    # Replace inf with large number
    costmap[np.isinf(costmap)] = 1e6

    # Save to file
    np.save("data/costmaps/hei1_costmap.npy", costmap)
    
    # Print some basic stats
    print("Min cost:", np.min(costmap))
    print("Max cost:", np.max(costmap))
    print("Any inf:", np.isinf(costmap).any())
    
    