import numpy as np

def min_sum_path_with_trace(npy_path, start, end):
    grid = np.load(npy_path, allow_pickle=True)
    rows, cols = grid.shape
    x1, y1 = start
    x2, y2 = end

    queue = [(0, x1, y1)]
    costs = {(x1, y1): 0}
    came_from = {(x1, y1): None}

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while queue:
        # Sort to simulate priority queue behavior
        queue.sort(reverse=True)
        cost, x, y = queue.pop()

        if (x, y) == (x2, y2):
            # reconstruct path
            path = []
            curr = (x2, y2)
            while curr:
                path.append(curr)
                curr = came_from[curr]
            path.reverse()
            return cost, path

        for dx, dy in directions:
            nx, ny = x + dx, y + dy

            if 0 <= nx < rows and 0 <= ny < cols:

                if  grid[x, y] < grid[nx, ny]:
                    new_cost = cost + (grid[nx, ny] - grid[x, y]) * 2
                elif  grid[x, y] == grid[nx, ny]:
                    new_cost = cost +  3
                else:
                    new_cost = cost + ( grid[x, y] - grid[nx, ny] ) * 4

                if (nx, ny) not in costs or new_cost < costs[(nx, ny)]:
                    costs[(nx, ny)] = new_cost
                    came_from[(nx, ny)] = (x, y)
                    queue.append((new_cost, nx, ny))

    return None, []  # no path found

if __name__ == "__main__":

    cost, path = min_sum_path_with_trace("data/elevation/hei1.npy", (0, 0), (300, 300))
    print("Minimum path sum:", cost)
    print("Path:", path)
    
