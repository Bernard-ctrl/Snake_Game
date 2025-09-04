import numpy as np
import random

class SnakeGame:
    def __init__(self, width=10, height=10):
        self.width = width
        self.height = height
        self.reset()

    def reset(self):
        self.snake = [(self.width // 2, self.height // 2)]
        self.direction = (0, -1)  # up
        self.food = self._place_food()
        self.score = 0
        self.done = False
        return self._get_state()

    def _place_food(self):
        while True:
            food = (random.randint(0, self.width - 1), random.randint(0, self.height - 1))
            if food not in self.snake:
                return food

    def _get_state(self):
        # Create a grid representation
        # Channel 0: snake head
        # Channel 1: snake body
        # Channel 2: food
        state = np.zeros((3, self.height, self.width), dtype=np.float32)
        head = self.snake[0]
        state[0, head[1], head[0]] = 1  # head
        for body in self.snake[1:]:
            state[1, body[1], body[0]] = 1  # body
        state[2, self.food[1], self.food[0]] = 1  # food
        return state

    def step(self, action):
        # Actions: 0: up, 1: down, 2: left, 3: right
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
        new_direction = directions[action]

        # Prevent reversing
        if (new_direction[0] * -1, new_direction[1] * -1) == self.direction:
            new_direction = self.direction

        self.direction = new_direction

        head = self.snake[0]
        new_head = (head[0] + self.direction[0], head[1] + self.direction[1])

        # Calculate distance to food before move
        old_distance = abs(head[0] - self.food[0]) + abs(head[1] - self.food[1])

        # Check wall collision
        if (new_head[0] < 0 or new_head[0] >= self.width or
            new_head[1] < 0 or new_head[1] >= self.height):
            self.done = True
            return self._get_state(), -20, self.done

        # Check self collision
        if new_head in self.snake:
            self.done = True
            return self._get_state(), -20, self.done

        self.snake.insert(0, new_head)

        # Calculate distance to food after move
        new_distance = abs(new_head[0] - self.food[0]) + abs(new_head[1] - self.food[1])

        reward = 0
        if new_head == self.food:
            self.score += 1
            reward = 50  # Increased reward for eating food
            self.food = self._place_food()
        else:
            self.snake.pop()
            # Small negative reward for each step to encourage efficiency
            reward = -0.1
            
            # Bonus for getting closer to food
            if new_distance < old_distance:
                reward += 0.5
            else:
                reward -= 0.5

        return self._get_state(), reward, self.done

    def render(self):
        grid = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        for x, y in self.snake:
            grid[y][x] = 'O'
        grid[self.snake[0][1]][self.snake[0][0]] = 'H'
        grid[self.food[1]][self.food[0]] = 'F'
        for row in grid:
            print(''.join(row))
        print(f'Score: {self.score}')

if __name__ == "__main__":
    game = SnakeGame()
    state = game.reset()
    print("Initial state shape:", state.shape)
    game.render()
