# Q-Learning Robot Navigation Project

## Overview
This project implements a Q-Learning-based approach for robot navigation in a grid environment. The goal is to navigate the robot from a starting point to an objective while avoiding obstacles and collecting strawberries for additional rewards. The project also includes a user-friendly GUI to facilitate interaction and visualization.

## Features
1. **Q-Learning Algorithm**
   - Dynamic learning of optimal paths using reinforcement learning.
   - Rewards for strawberries, penalties for obstacles and revisited states.

2. **Grid Environment**
   - Configurable grid size.
   - Customizable placement of obstacles and strawberries.

3. **Visualization**
   - Heatmaps for optimal Q-values.
   - Graphical representation of the robot's navigation path.

4. **Graphical User Interface (GUI)**
   - A PyQt5-based GUI for interactive setup and visualization.
   - Input fields for grid size, source, objective, obstacles, and strawberries.
   - Buttons to create environments and start Q-learning training.

## GUI Features
The GUI, implemented using PyQt5, includes:

- Input fields for environment size, source, objective, obstacles, and strawberries.
- Parameter tuning for alpha, epsilon, gamma, and number of episodes.
- Buttons to:
  - Create a grid environment.
  - Start training the robot using Q-learning.
- Real-time visualization of the environment and the optimal path.

## Code Structure
- **Q-Learning Implementation:**
  - `q_learning(episodes, alpha, gamma, epsilon, q_table, strawberries)` handles the training process.
  - `find_optimal_path(q_table, source, objective)` finds the optimal path after training.

- **Visualization Tools:**
  - `visualize_path`: Displays the robot’s path.
  - `visualize_q_table`: Shows Q-values for each state-action pair.
  - `visualize_Heatmap_optimal_path`: Displays a heatmap of maximum Q-values.

- **GUI Implementation:**
  - `RobotTrainingApp`: Main PyQt5 application class.
  - `create_environment`: Sets up the grid environment based on user input.
  - `start_training`: Starts the Q-learning process and visualizes the results.

## Getting Started

### Prerequisites
- Python 3.x
- NumPy
- Matplotlib
- PyQt5

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/ourahma/Q_Learning.git
   ```
2. Install dependencies:
   ```bash
   pip install numpy matplotlib PyQt5
   ```

## Usage
1. Set the grid size, source, objective, obstacles, and strawberries in the GUI.
2. Adjust training parameters (alpha, epsilon, gamma, episodes).
3. Click "Create Environment" to visualize the grid.
4. Click "Start Training" to run the Q-learning algorithm and see the optimal path.

## Screenshots
- Creating the environment:

![Creating env ](/images/creating_env.png)

- Visualizing the environment after training: 

![Creating env ](/images/finish_training.png)


## Future Enhancements
- Add support for dynamic obstacle placement during training.
- Implement additional reinforcement learning algorithms.
- Enhance visualization with real-time path updates.

## Contacts
For any questions or suggestions, feel free to reach out:
- **Email:** marouaourahma@gmail.com
- **LinkedIn:** [My LinkedIn Profile](www.linkedin.com/in/maroua-ourahma-293426235)

