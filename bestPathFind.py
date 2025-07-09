import numpy as np
import heapq
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

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
                    step_cost = (grid[x, y] - grid[nx, ny]) * 0.02

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
    elevation_dir = "data/elevation"
    costmap_dir = "data/costmaps"
    os.makedirs(costmap_dir, exist_ok=True)

    files = [f for f in os.listdir(elevation_dir) if f.endswith(".npy")]
    print(f"Found {len(files)} elevation maps.")

    for f in files:
        path = os.path.join(elevation_dir, f)
        out_name = f.replace(".npy", "_costmap.npy")
        out_path = os.path.join(costmap_dir, out_name)

        if os.path.exists(out_path):
            print(f"Skipping {f} - already processed.")
            continue

        print(f"\nProcessing: {f}")

        costmap = min_sum_path_with_trace(path, (0, 0))
        costmap[np.isinf(costmap)] = 1e6

        np.save(out_path, costmap)

        print("→ Saved:", out_path)
        print("  Min cost:", np.min(costmap))
        print("  Max cost:", np.max(costmap))
        print("  Any inf:", np.isinf(costmap).any())