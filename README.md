# Pong AI

A modern implementation of the classic Pong game with advanced AI training capabilities and neural network visualization.

## Features

### Game Modes
- **Single Player vs AI**: Play against AI opponents with different difficulty levels
- **Two Player**: Local multiplayer mode for head-to-head gameplay
- **AI vs AI**: Watch AI opponents compete against each other

### AI Training System
- **Neural Network Training**: Train AI models using reinforcement learning
- **Multiple Difficulty Presets**: Beginner, intermediate, and advanced training modes
- **Model Management**: Save, load, and compare different AI models
- **Real-time Performance Metrics**: Track training progress and AI performance
- **Neural Network Visualization**: Visual representation of the AI's decision-making process

### Visual Features
- **Neon Theme**: Modern cyberpunk-inspired visual design with glowing effects
- **Particle Effects**: Dynamic particle systems for enhanced visual feedback
- **Smooth Animations**: Fluid movement and transitions
- **Real-time Statistics**: Performance metrics and game analytics

### Power-ups and Enhancements
- **Power-up System**: Special abilities that modify gameplay
- **Multi-ball Mode**: Multiple balls for increased challenge
- **Chaotic Ball**: Unpredictable ball behavior patterns
- **Combo System**: Reward streaks and consecutive hits

## Installation

### Requirements
- Python 3.7 or higher
- pip package manager

### Setup
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd ppai
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Starting the Game
Run the main launcher:
```bash
python game_launcher.py
```

### Menu Navigation
- **Arrow Keys**: Navigate menu options
- **Enter**: Select menu item
- **Escape**: Return to previous menu or pause game

### Game Controls
- **Player 1**: W (up) / S (down)
- **Player 2**: Up Arrow / Down Arrow
- **Pause**: Space or Escape during gameplay

### Training AI Models
1. Select "AI Training" from the main menu
2. Choose training difficulty and parameters
3. Monitor progress through real-time metrics
4. Save trained models for later use

## Technical Details

### Architecture
- **Game Engine**: Pygame for graphics and input handling
- **AI Framework**: Custom neural network implementation with NumPy
- **Threading**: Multi-threaded rendering and AI processing for smooth performance
- **Modular Design**: Separate components for game logic, AI, visualization, and UI

### AI Implementation
- **Neural Network**: Feed-forward networks with customizable architecture
- **Training Algorithm**: Reinforcement learning with performance-based rewards
- **State Representation**: Game state encoded as input features for the neural network
- **Parallel Processing**: Multiple AI instances for efficient training

### File Structure
- `pong_game.py`: Core game engine and mechanics
- `ai_controller.py`: AI decision-making logic
- `neural_network.py`: Neural network implementation
- `menu_manager.py`: User interface and menu system
- `game_launcher.py`: Main application entry point
- `ai_training_manager.py`: Training orchestration and management
- `ai_models/`: Saved AI models and training states

### Difficulty Presets
The game includes multiple AI difficulty levels with pre-configured parameters for different skill levels and training scenarios.
