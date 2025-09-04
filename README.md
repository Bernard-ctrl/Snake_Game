# Snake Game AI

This project implements an improved AI model to play the Snake game using Deep Q-Network (DQN) with PyTorch and CUDA acceleration.

## Recent Improvements

- **Enhanced Reward System**: Distance-based rewards, survival bonuses, and penalties for inefficient moves
- **Better Neural Network**: Deeper CNN with batch normalization and dropout for better learning
- **Improved Training**: Longer training (5000 episodes), larger replay buffer, slower exploration decay
- **Best Model Saving**: Automatically saves the best performing model during training

## GUI Features

- **Larger Map**: 20x15 grid (doubled size) for more exciting gameplay
- **Fast Snake**: 50ms delay between moves for rapid, exciting action
- **Clean Design**: Removed grid lines for modern, clean aesthetic
- **Enhanced Visuals**: 
  - Gradient snake body with eyes on head
  - Glowing circular food with highlights
  - Dark blue-black background
  - Colorful UI elements
- **Detailed Stats**: Shows snake length, food eaten per episode, and total food eaten
- **Auto-Model Selection**: Automatically picks the best trained model

## Files

- `snake_game.py`: The Snake game environment with improved rewards
- `ai_model.py`: Enhanced DQN model with better architecture
- `train.py`: Improved training script with longer training and model saving
- `play.py`: GUI script to watch the trained AI play

## Requirements

- Python 3.10+
- PyTorch with CUDA support
- Pygame (for GUI)
- NumPy
- Matplotlib

## Usage

1. Train the AI (this will take longer but produce better results):
   ```
   python train.py
   ```

2. Play with the best trained AI (GUI - auto-selects best model):
   ```
   python play.py
   ```
   
   The script automatically selects the best available model in this order:
   - `snake_ai_best.pth` (highest scoring model from training)
   - `snake_ai_final.pth` (final trained model)
   - `snake_ai.pth` (fallback model)
   
   You can also specify episodes:
   ```bash
   python play.py episodes=10  # Run 10 episodes
   python play.py  # Run continuously until you close the window
   python play.py delay=25   # Super fast
   python play.py delay=100  # Normal speed
   ```