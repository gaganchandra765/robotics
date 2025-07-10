import heapq
import matplotlib.pyplot as plt
import numpy as np

def a_star(grid, start, goal):
    rows, cols = len(grid), len(grid[0])
    open_set = []
    heapq.heappush(open_set, (0 + heuristic(start, goal), 0, start, [start]))  # (f, g, current, path)
    visited = set()

    while open_set:
        f, g, current, path = heapq.heappop(open_set)
        if current == goal:
            return path
        
        if current in visited:
            continue
        visited.add(current)
        
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = current[0] + dx, current[1] + dy
            if 0 <= nx < rows and 0 <= ny < cols:
                if grid[nx][ny] == -1:
                    continue  # obstacle
                if (nx, ny) in visited:
                    continue
                cost = grid[nx][ny]
                heapq.heappush(open_set, (g + cost + heuristic((nx, ny), goal), g + cost, (nx, ny), path + [(nx, ny)]))

    return None  # no path found

def heuristic(a, b):
    # Manhattan Distance
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def visualize_path(grid, path, start, goal, save_path='path_result.png'):
    grid_disp = np.array(grid)
    fig, ax = plt.subplots(figsize=(8, 8))

    cmap = plt.cm.viridis
    cmap.set_under('black')  # for obstacles
    cax = ax.matshow(np.where(grid_disp==-1, -1e-3, grid_disp), cmap=cmap, vmin=0)

    for (i, j), val in np.ndenumerate(grid_disp):
        if val != -1:
            ax.text(j, i, str(val), va='center', ha='center', fontsize=7, color='white')

    if path:
        path_x, path_y = zip(*path)
        ax.plot(path_y, path_x, color='red', linewidth=2, label='Path')
        ax.scatter(start[1], start[0], color='blue', s=100, label='Start', marker='o')
        ax.scatter(goal[1], goal[0], color='green', s=100, label='Goal', marker='X')
    else:
        ax.text(10, 10, "No Path Found", fontsize=14, color='red', ha='center')

    plt.legend()
    plt.title("Quadruped Path Planning")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)  # <-- SAVE the figure here
    print(f"Path image saved as '{save_path}'")

    plt.show()


# Sample Grid (20x20)
np.random.seed(42)
sample_grid = np.random.randint(1, 10, (20, 20)).tolist()

# Add some obstacles
for _ in range(60):
    x, y = np.random.randint(0, 20), np.random.randint(0, 20)
    sample_grid[x][y] = -1

start = (0, 0)
goal = (19, 19)

# Run A* Search
path = a_star(sample_grid, start, goal)

# Visualize
visualize_path(sample_grid, path, start, goal)

# Print Path
if path:
    print("Optimal Path (with cost):")
    print(path)
    total_cost = sum(sample_grid[x][y] for x, y in path)
    print("Total Cost:", total_cost)
else:
    print("No path could be found from start to goal.")

