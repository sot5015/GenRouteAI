import numpy as np
import matplotlib.pyplot as plt
import heapq
import os

"""
Script to compute optimal paths on a costmap using the A* algorithm:
- Loads a costmap from .npy file
- Finds the minimum-cost path between start and goal points
- Visualizes the path overlayed on the costmap
- Prints path length and total cost
"""

# ----------- A* FUNCTIONS ------------

def heuristic(a, b):
    """Euclidean distance as heuristic"""
    return np.linalg.norm(np.array(a) - np.array(b))

def a_star(costmap, start, goal):
    rows, cols = costmap.shape
    open_set = []
    heapq.heappush(open_set, (0, start))

    came_from = {}
    g_score = {start: 0}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            path = reconstruct_path(came_from, current)
            total_cost = g_score[goal]
            return path, total_cost

        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            neighbor = (current[0]+dx, current[1]+dy)

            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
                step_cost = costmap[neighbor]
                tentative_g = g_score[current] + step_cost

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, neighbor))
                    came_from[neighbor] = current

    return None, float("inf")

def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path

# ----------- MAIN SCRIPT ------------

if __name__ == "__main__":
    # --- Load costmap ---
    costmap_path = "data/costmaps/hei7_costmap.npy"
    if not os.path.exists(costmap_path):
        raise FileNotFoundError(f"Costmap not found: {costmap_path}")

    costmap = np.load(costmap_path)

    # --- Define start and goal ---
    start = (0, 0)
    goal = (250, 250)

    # --- Run A* ---
    path, total_cost = a_star(costmap, start, goal)

    if path is None:
        print("No path found.")
    else:
        print(f"Found path of length {len(path)}")
        print(f"Total cost: {total_cost}")

        # Plot costmap and path
        plt.figure(figsize=(8,6))
        plt.imshow(costmap, cmap="viridis")
        y_coords, x_coords = zip(*path)
        plt.plot(x_coords, y_coords, color="red", linewidth=2)
        plt.title("A* Path over Costmap")
        plt.colorbar(label="Cost")
        plt.show()