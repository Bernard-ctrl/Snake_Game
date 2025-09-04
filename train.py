import torch
from snake_game import SnakeGame
from ai_model import DQNAgent
import matplotlib.pyplot as plt
import numpy as np

def train_agent(episodes=5000, max_steps=2000, batch_size=64, target_update=20):  # Increased episodes, steps, batch size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    game = SnakeGame()
    agent = DQNAgent(device)

    scores = []
    losses = []
    best_score = 0

    for episode in range(episodes):
        state = game.reset()
        total_reward = 0
        episode_loss = 0
        steps = 0

        while not game.done and steps < max_steps:
            action = agent.select_action(state)
            next_state, reward, done = game.step(action)
            agent.replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            steps += 1

            loss = agent.train_step(batch_size)
            if loss is not None:
                episode_loss += loss

        agent.update_epsilon()

        if episode % target_update == 0:
            agent.update_target_net()

        scores.append(game.score)
        losses.append(episode_loss / steps if steps > 0 else 0)

        # Save best model
        if game.score > best_score:
            best_score = game.score
            torch.save(agent.policy_net.state_dict(), 'snake_ai_best.pth')
            print(f"New best score: {best_score} at episode {episode}")

        if episode % 100 == 0:
            avg_score = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
            print(f"Episode {episode}, Score: {game.score}, Avg Score (last 100): {avg_score:.2f}, Epsilon: {agent.epsilon:.3f}, Avg Loss: {losses[-1]:.4f}")

    # Save the final model
    torch.save(agent.policy_net.state_dict(), 'snake_ai_final.pth')
    print("Final model saved as snake_ai_final.pth")
    print(f"Best score achieved: {best_score}")

    # Plot results
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(scores)
    plt.title('Scores over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Score')

    plt.subplot(1, 2, 2)
    plt.plot(losses)
    plt.title('Average Loss over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Loss')

    plt.savefig('training_results.png')
    plt.show()

if __name__ == "__main__":
    train_agent()
