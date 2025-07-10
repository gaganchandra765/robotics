import pygame
import numpy as np
import heapq
import time
import imageio

# === A* Search ===
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def a_star(grid, start, goal):
    rows, cols = len(grid), len(grid[0])
    open_set = []
    heapq.heappush(open_set, (0 + heuristic(start, goal), 0, start, [start]))
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
                if grid[nx][ny] == -1 or (nx, ny) in visited:
                    continue
                cost = grid[nx][ny]
                heapq.heappush(open_set, (g + cost + heuristic((nx, ny), goal), g + cost, (nx, ny), path + [(nx, ny)]))
    return None

# === Pygame Visualization Settings ===
CELL_SIZE = 30
MARGIN = 2
WIDTH, HEIGHT = CELL_SIZE * 20, CELL_SIZE * 20
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (220, 50, 50)
BLUE = (50, 100, 255)
GREEN = (50, 220, 50)
YELLOW = (255, 255, 0)

def draw_grid(screen, grid, path, current=None, start=None, goal=None):
    for i in range(20):
        for j in range(20):
            x, y = j * CELL_SIZE, i * CELL_SIZE
            value = grid[i][j]
            rect = pygame.Rect(x, y, CELL_SIZE - MARGIN, CELL_SIZE - MARGIN)

            if value == -1:
                color = BLACK
            elif (i, j) == start:
                color = BLUE
            elif (i, j) == goal:
                color = GREEN
            elif (i, j) == current:
                color = YELLOW
            elif (i, j) in path:
                color = RED
            else:
                intensity = 255 - min(200, value * 20)
                color = (intensity, intensity, intensity)

            pygame.draw.rect(screen, color, rect)

# === Animate and Save as Video ===
def animate_robot(grid, path, start, goal, save_video_path="quadruped_path.mp4"):
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Quadruped Path Animation")
    clock = pygame.time.Clock()

    frames = []

    for index, cell in enumerate(path):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        screen.fill(WHITE)
        draw_grid(screen, grid, path[:index+1], current=cell, start=start, goal=goal)
        pygame.display.flip()

        # Capture frame
        frame = pygame.surfarray.array3d(screen)
        frame = np.transpose(frame, (1, 0, 2))  # (width, height, channels) → (height, width, channels)
        frames.append(frame.copy())

        clock.tick(10)  # FPS

    # Save video
    imageio.mimsave(save_video_path, frames, fps=10)
    print(f"Animation saved as '{save_video_path}'")

    time.sleep(2)
    pygame.quit()

# === Main Execution ===
if __name__ == "__main__":
    np.random.seed(42)
    grid = np.random.randint(1, 10, (20, 20)).tolist()

    # Place 60 random obstacles
    for _ in range(60):
        x, y = np.random.randint(0, 20), np.random.randint(0, 20)
        grid[x][y] = -1

    start = (0, 0)
    goal = (19, 19)

    print("🔍 Finding path using A*...")
    path = a_star(grid, start, goal)

    if path:
        total_cost = sum(grid[x][y] for x, y in path)
        print(" Path found!")
        print("Total Cost:", total_cost)
        print("Path:", path)
        animate_robot(grid, path, start, goal, save_video_path="quadruped_path.mp4")
    else:
        print(" No path found from start to goal.")

