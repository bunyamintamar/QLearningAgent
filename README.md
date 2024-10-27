# Q-Learning Agent with Environment Simulation

This project demonstrates a Q-Learning algorithm implementation in a simulated environment, where an agent learns optimal actions to reach a target state while avoiding dangerous states. The environment is structured as a 3x3 grid where each cell is a unique state and each action leads the agent to a different state based on pre-defined transitions.

## Table of Contents
- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Classes Overview](#classes-overview)
- [Q-Learning Algorithm](#q-learning-algorithm)
- [Usage](#usage)
- [Example Simulation](#example-simulation)
- [Compilation](#compilation)
- [Author](#author)
- [License](#license)

## Introduction
Reinforcement learning (RL) is a branch of machine learning where an agent learns by interacting with an environment and receiving feedback in the form of rewards. In this project, we simulate an agent navigating through a grid-based environment to maximize rewards by reaching a target state while avoiding penalized states.

The agent uses the Q-Learning algorithm to iteratively learn optimal actions for each state. The environment provides transitions, while the agent tracks action history, updates Q-values, and adjusts its policy based on the rewards received.

## Project Structure
- `qlearningagent.h`: Header file defining the `Environment` and `QLearningAgent` classes.
- `qlearningagent.cpp`: Source file implementing the functionality of `Environment` and `QLearningAgent`.
- `main.cpp`: Example file to demonstrate how to set up and use the environment and agent.
- `README.md`: Documentation (this file).

## Classes Overview

### Environment
The `Environment` class defines the grid layout and state-action transitions. The environment handles:
- State transitions when the agent performs an action.
- Storing available actions for each state.
- Printing transition information for debugging.

### QLearningAgent
The `QLearningAgent` class encapsulates the Q-Learning algorithm and manages:
- Q-value updates for each state-action pair.
- Exploration-exploitation trade-off using an epsilon-greedy strategy.
- Action history tracking and reward-based updates.

### Grid Layout
The environment is a 3x3 grid with the following layout:

|0|1|2| |3|4|5| |6|7|8|

- Start state: 0
- Target state: 5 (reward: +100)
- Dangerous state: 4 (penalty: -100)

## Q-Learning Algorithm
Q-Learning is an RL algorithm that seeks to learn the best action to take given a particular state by maximizing the cumulative reward. This is achieved by updating Q-values as follows:

\[
Q(s, a) \leftarrow Q(s, a) + \alpha \left( r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right)
\]

Where:
- \( \alpha \): Learning rate
- \( \gamma \): Discount factor
- \( r \): Reward received from performing action \( a \) in state \( s \)
- \( s' \): New state resulting from action \( a \)

### Parameters
- **Learning Rate (`alpha`)**: Determines how much new information overrides old information.
- **Discount Factor (`gamma`)**: The extent to which future rewards are considered.
- **Exploration Rate (`epsilon`)**: Probability of choosing a random action instead of the greedy choice.

## Usage

1. Define the environment transitions using `Environment::setTransition`.
2. Initialize `QLearningAgent` with parameters for learning, discount, and exploration.
3. Run training sessions where the agent navigates from a start state (0) to a target state (5).
4. Update Q-values and action history based on rewards received.
5. Print the Q-Table after training to examine learned policies.

## Example Simulation

In `main.cpp`, an example setup demonstrates the Q-Learning agent:

1. **Environment Setup**: Transitions are set between states based on possible actions:
    ```cpp
    environment.setTransition(0, DOWN, 3);
    environment.setTransition(0, RIGHT, 1);
    // Additional transitions...
    ```

2. **Agent Initialization**:
    ```cpp
    QLearningAgent agent(0.1, 0.7, 0.1, &environment);
    ```

3. **Training Loop**: The agent runs through a series of epochs to learn from experiences.
    ```cpp
    for (int epoch = 0; epoch < 100; epoch++) {
        int state = 0;
        agent.startSession();
        // Agent navigates until reaching target or dangerous state...
        agent.stopSession(reward);
    }
    ```

4. **Q-Table Display**: After training, print the Q-Table to see learned action values.
    ```cpp
    agent.printQTable();
    ```

## Compilation

To compile and run the example, use:
```bash
g++ -std=c++11 main.cpp qlearningagent.cpp -o QLearningExample
./QLearningExample
```

## Author
Bünyamin TAMAR - Software Engineer

## License
This project is open-source under the MIT License
