"""
Random‑move baseline + training/demo scaffold
adapted to richer reward dict from snake_game.run_episode.
"""

import heapq
import random
from collections import deque

class SnakeBot:
    DIRECTIONS = ("UP", "DOWN", "LEFT", "RIGHT")
    DIR_OFFSET = {
        "UP": (0, -1),
        "DOWN": (0, 1),
        "LEFT": (-1, 0),
        "RIGHT": (1, 0),
    }

    def next_move(self, state):
        head = state["snake"][0]
        food = state["food"]
        width = state["board_width"]
        height = state["board_height"]
        snake = list(state["snake"])

        growing = (head == food)
        if growing:
            obstacles = set(snake)
        else:
            obstacles = set(snake[:-1])

        # Try path to food
        path_to_food = self.a_star(head, food, obstacles, width, height)

        # Check if path to food leads to enough space after move
        if path_to_food and len(path_to_food) > 1:
            next_cell = path_to_food[1]
            new_snake = [next_cell] + snake[:-1] if not growing else [next_cell] + snake
            space = self.free_space(next_cell, set(new_snake), width, height)
            if space >= len(snake):  # enough space, go for food
                return self.direction_from_to(head, next_cell)

        # Else try chasing tail
        tail = snake[-1]
        path_to_tail = self.a_star(head, tail, obstacles, width, height)
        if path_to_tail and len(path_to_tail) > 1:
            next_cell = path_to_tail[1]
            return self.direction_from_to(head, next_cell)

        # Fallback: any safe move
        safe_moves = [d for d in self.DIRECTIONS if self.is_move_safe(head, d, obstacles, width, height)]
        if safe_moves:
            return random.choice(safe_moves)

        return random.choice(self.DIRECTIONS)

    def direction_from_to(self, start, end):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        for d, (ox, oy) in self.DIR_OFFSET.items():
            if (dx, dy) == (ox, oy):
                return d
        return random.choice(self.DIRECTIONS)

    def free_space(self, start, obstacles, width, height):
        # BFS flood fill counting reachable empty spaces
        queue = deque([start])
        visited = set(obstacles)
        count = 0

        while queue:
            pos = queue.popleft()
            count += 1
            for dx, dy in self.DIR_OFFSET.values():
                nx, ny = pos[0] + dx, pos[1] + dy
                if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append((nx, ny))

        return count

    def is_move_safe(self, head, direction, obstacles, width, height):
        dx, dy = self.DIR_OFFSET[direction]
        nx, ny = head[0] + dx, head[1] + dy
        if nx < 0 or nx >= width or ny < 0 or ny >= height:
            return False
        if (nx, ny) in obstacles:
            return False
        return True

    def a_star(self, start, goal, obstacles, width, height):
        def neighbors(pos):
            for dx, dy in self.DIR_OFFSET.values():
                nx, ny = pos[0] + dx, pos[1] + dy
                if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in obstacles:
                    yield (nx, ny)

        def heuristic(pos):
            return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

        open_set = []
        heapq.heappush(open_set, (heuristic(start), 0, start, [start]))
        visited = set()

        while open_set:
            _, cost, current, path = heapq.heappop(open_set)

            if current == goal:
                return path

            if current in visited:
                continue
            visited.add(current)

            for neighbor in neighbors(current):
                if neighbor in visited:
                    continue
                heapq.heappush(open_set, (cost + 1 + heuristic(neighbor), cost + 1, neighbor, path + [neighbor]))

        return None

# --------------------------- Training loop ----------------------------- #
N_EPISODES = 20            # ← adjust here
BLANK_LINES_BEFORE_DEMO = 17

def train_and_demo():
    import snake_game

    bot      = SnakeBot()
    comps    = []   # composite scores
    raw_food = []   # plain scores

    print(f"\nRunning {N_EPISODES} head‑less training games …")
    for ep in range(1, N_EPISODES + 1):
        res = snake_game.run_episode(bot, render=False)
        comps.append(res["composite"])
        raw_food.append(res["score"])
        print(f"Episode {ep:3d}: "
              f"score={res['score']:2d}  ticks={res['ticks']:4d}  "
              f"comp={res['composite']:4d}",
              flush=True)

    avg_comp = sum(comps) / len(comps)
    avg_food = sum(raw_food) / len(raw_food)
    print(f"\nTraining finished. Avg food = {avg_food:.2f} "
          f"| Avg composite = {avg_comp:.2f}\n")

    # push logs up so the clear‑screen in the demo doesn't hide them
    print("\n" * BLANK_LINES_BEFORE_DEMO)

    print("=== Demo game with rendering ===\n")
    snake_game.run_episode(bot, render=True)

if __name__ == "__main__":
    train_and_demo()
