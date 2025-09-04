import torch
import pygame
import os
from snake_game import SnakeGame
from ai_model import DQN

# Colors - Improved aesthetic
BLACK = (20, 20, 30)  # Dark blue-black background
DARK_BLUE = (25, 25, 45)
WHITE = (220, 220, 255)  # Soft white
GREEN = (50, 200, 100)  # Bright green for snake body
BRIGHT_GREEN = (100, 255, 150)  # Even brighter for head
RED = (255, 100, 100)  # Bright red for food
YELLOW = (255, 255, 100)  # Yellow for score text
PURPLE = (150, 100, 255)  # Purple accent

# Grid settings - Larger display
CELL_SIZE = 35  # Larger cells for bigger visual
GRID_WIDTH = 10   # Keep logical grid same for model compatibility
GRID_HEIGHT = 10

def draw_grid(screen, game):
    for y in range(GRID_HEIGHT):
        for x in range(GRID_WIDTH):
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, WHITE, rect, 1)

def draw_snake(screen, game):
    # Draw body segments with gradient effect
    for i, segment in enumerate(game.snake[1:], 1):
        rect = pygame.Rect(segment[0] * CELL_SIZE, segment[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        # Gradient effect - darker towards tail
        intensity = max(100, 200 - i * 10)
        color = (50, intensity, 100)
        pygame.draw.rect(screen, color, rect)
        # Add subtle border
        pygame.draw.rect(screen, DARK_BLUE, rect, 1)
    
    # Draw head with brighter color
    if game.snake:
        head = game.snake[0]
        rect = pygame.Rect(head[0] * CELL_SIZE, head[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, BRIGHT_GREEN, rect)
        # Add eye-like effect
        eye_size = max(2, CELL_SIZE // 6)
        pygame.draw.circle(screen, BLACK, (head[0] * CELL_SIZE + CELL_SIZE//3, head[1] * CELL_SIZE + CELL_SIZE//3), eye_size)
        pygame.draw.circle(screen, BLACK, (head[0] * CELL_SIZE + 2*CELL_SIZE//3, head[1] * CELL_SIZE + CELL_SIZE//3), eye_size)

def draw_food(screen, game):
    rect = pygame.Rect(game.food[0] * CELL_SIZE, game.food[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
    # Draw main food circle
    center = (game.food[0] * CELL_SIZE + CELL_SIZE//2, game.food[1] * CELL_SIZE + CELL_SIZE//2)
    pygame.draw.circle(screen, RED, center, CELL_SIZE//2 - 2)
    # Add highlight
    pygame.draw.circle(screen, YELLOW, (center[0] - CELL_SIZE//6, center[1] - CELL_SIZE//6), CELL_SIZE//4)

def play_game_gui(model_path='snake_ai_best.pth', episodes=1, delay=50):
    """
    Play the Snake game with a trained AI model.
    
    Args:
        model_path: Path to model file (defaults to best model)
        episodes: Number of episodes to play (default: 1)
        delay: Delay between moves in milliseconds (default: 50 for fast gameplay)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    pygame.init()
    screen = pygame.display.set_mode((GRID_WIDTH * CELL_SIZE, GRID_HEIGHT * CELL_SIZE + 60))
    pygame.display.set_caption("Snake AI - Immortal Mode")
    font = pygame.font.Font(None, 28)
    small_font = pygame.font.Font(None, 20)
    clock = pygame.time.Clock()

    game = SnakeGame()
    model = DQN().to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"Loaded model: {model_path}")
    except FileNotFoundError:
        print(f"Model file '{model_path}' not found. Please train the model first.")
        return

    episode = 0
    total_food_eaten = 0
    running = True
    
    while running and episode < episodes:
        episode += 1
        state = game.reset()
        episode_food = 0
        total_reward = 0
        steps = 0
        episode_running = True

        while episode_running and running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    episode_running = False

            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                q_values = model(state_tensor)
                action = q_values.argmax().item()

            next_state, reward, done = game.step(action)
            state = next_state
            total_reward += reward
            steps += 1

            # Track food eaten
            if reward > 10:  # Food eating reward
                episode_food += 1
                total_food_eaten += 1

            # Draw everything
            screen.fill(BLACK)
            draw_snake(screen, game)
            draw_food(screen, game)
            
            # Draw enhanced stats
            model_name = os.path.splitext(os.path.basename(model_path))[0]
            episode_text = f"{model_name} | Episode: {episode}"
            
            score_text = font.render(episode_text, True, YELLOW)
            screen.blit(score_text, (10, GRID_HEIGHT * CELL_SIZE + 5))
            
            # Show current episode stats
            episode_stats = f"Length: {len(game.snake)} | Food: {episode_food} | Total Food: {total_food_eaten}"
            stats_text = small_font.render(episode_stats, True, WHITE)
            screen.blit(stats_text, (10, GRID_HEIGHT * CELL_SIZE + 30))
            
            pygame.display.flip()
            pygame.time.wait(delay)
            clock.tick(60)

            # For "immortal" mode - don't end episode on death
            # Just reset the game state but continue playing
            if done:
                print(f"Snake died! Score: {game.score}, Food Eaten: {episode_food}, Steps: {steps}")
                # Reset for next attempt but stay in same episode
                state = game.reset()
                episode_food = 0
                steps = 0
                pygame.time.wait(1000)  # Brief pause after death

        print(f"Episode {episode} finished. Total Food: {total_food_eaten}, Total Reward: {total_reward:.1f}")

    pygame.quit()

if __name__ == "__main__":
    play_game_gui()
