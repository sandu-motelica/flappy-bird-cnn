# Flappy Bird AI Agent

This project implements a deep reinforcement learning agent to play the game Flappy Bird using a Convolutional Neural Network (CNN) and Q-Learning. The agent is trained on pixel data from the game environment.

---

## Table of Contents
1. [Project Architecture](#project-architecture)
2. [Neural Network Model](#neural-network-model)
3. [Hyperparameters](#hyperparameters)
4. [Q-Learning Algorithm](#q-learning-algorithm)
5. [Experimentation and Results](#experimentations-and-results)

---

## Project Architecture

The architecture of the project is as follows:

- **Environment**: The agent interacts with the `FlappyBird-v0` environment using `gymnasium`.
- **State Representation**: The agent observes the game as pixel data, which is processed and converted into grayscale tensors for input to the neural network.
- **Replay Memory**: Stores transitions `(state, action, reward, next_state, terminal)` for training stability and to avoid correlated updates.
- **Action Selection**: Uses an epsilon-greedy policy to balance exploration and exploitation.
- **Training Loop**: 
  - Updates the Q-values using the Bellman equation.
  - Minimizes the Mean Squared Error (MSE) loss between predicted Q-values and target Q-values.

---

## Neural Network Model

The agent uses a **Convolutional Neural Network (CNN)** to approximate the Q-function. 

### Architecture Overview:
- **Input Layer**: Takes 4 stacked grayscale frames of the game environment to capture temporal information.
- **Convolutional Layers**:
  - Three layers of convolution with ReLU activation extract spatial features from the game environment.
  - Filters: 32, 64, and 64 in successive layers.
  - Kernel Sizes: 8x8, 4x4, and 3x3.
  - Strides: 4, 2, and 1, respectively.
- **Fully Connected Layers**:
  - The output of the convolutional layers is flattened and passed through a dense layer with 512 units.
  - The final layer produces Q-values for the two possible actions (`flap` or `no-flap`).
- **Activation Functions**: ReLU is used in all layers except the output.

This architecture processes raw pixel data and outputs Q-values for each action, enabling the agent to decide its next move effectively.

---

## Hyperparameters

| Hyperparameter           | Value              |
|--------------------------|--------------------|
| Learning Rate            | 1e-6              |
| Initial Epsilon          | 1.0               |
| Final Epsilon            | 0.0001            |
| Epsilon Decay Iterations | 10,000,000        |
| Replay Memory Size       | 10,000            |
| Minibatch Size           | 32                |
| Discount Factor (Gamma)  | 0.99              |

---

## Q-Learning Algorithm

1. **Action Selection**:
   - The agent selects actions using an epsilon-greedy policy to balance exploration and exploitation.
   
2. **Q-Value Updates**:
   - The Bellman equation is used to compute the target Q-values:
     - For terminal states: \( y_j = r_j \)
     - For non-terminal states: \( y_j = r_j + \gamma \cdot \max(Q(s_{j+1}, a)) \)

3. **Loss Function**:
   - The Mean Squared Error (MSE) is used to compute the difference between predicted and target Q-values.

---

## Experimentation and Results

### Experiments

1. **Epsilon Decay**:
   - Adjusted the epsilon decay rate to balance exploration and exploitation.

2. **Replay Memory Size**:
   - Tested memory size 1000 to improve training stability.

3. **Minibatch Size**:
   - Experimented with minibatches

### Adjustments

To enhance the training process and improve the agent's performance, several modifications were made to the game environment and preprocessing steps:

1. **Environment Adjustments**:
   - The score display was removed from the screen to eliminate distractions and irrelevant features for the agent.
   - The background was simplified to ensure the focus remained on essential elements of the game.

2. **Image Preprocessing**:
   - The input image was cropped to exclude the bottom part, which was deemed irrelevant to the agent's decision-making process.
   - The cropped image was resized to 84x84 pixels and converted to grayscale to reduce computational complexity and emphasize important visual features.

3. **Frame Count**:
   - The maximum possible number of frames (\( 999...999 \)) was set to ensure the agent had sufficient data for each training iteration and evaluation.

### Results

The Flappy Bird model's performance across 100 episodes showed considerable variability, with scores ranging from a low of **7** to a peak of **87**. Early episodes displayed inconsistent results, but the model demonstrated the potential for higher performance in later episodes, achieving notable scores like **78**, **87**, and **63**. This suggests some degree of learning or adaptation over time. However, the distribution of scores remains uneven, with frequent low scores interspersed with sporadic high ones.

---
