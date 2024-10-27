/**
 * @file main.cpp
 * @brief Example usage of QLearningAgent and Environment classes.
 *
 * In this example, an environment is created with a grid-like layout,
 * where each cell represents a unique state
 * and each action (UP, DOWN, RIGHT, LEFT) results in a transition
 * to a different state.
 *
 * The agent begins at state 0, aiming to reach a target state (state 5),
 * while avoiding a dangerous state (state 4). The environment rewards
 * the agent with +100 upon reaching the target state and penalizes with -100
 * if it encounters the dangerous state.
 *
 * Key points of the example:
 * - Environment defines a 3x3 grid layout where each state
 * and its transitions are set.
 * - The QLearningAgent learns through trials, storing action histories
 * and updating
 *   Q-values based on rewards received.
 * - After running the simulation for multiple epochs, the agent prints
 * its Q-Table, showing the learned optimal actions for each state.
 *
 * Grid layout:
 * |0|1|2|
 * |3|4|5|
 * |6|7|8|
 *
 * - Start state: 0
 * - Target state: 5
 * - Dangerous state: 4
 *
 * @note This example demonstrates a basic reinforcement learning setup.
 *
 * @date 2024
 * @version 1.0
 * @author Bünyamin TAMAR
 */

#include <iostream>
#include <algorithm>
#include <vector>
#include "qlearningagent.h"
#include <cstdlib>
#include <ctime>

#define UP 0
#define DOWN 1
#define RIGHT 2
#define LEFT 3

/*
 * |0|1|2|
 * |3|4|5|
 * |6|7|8|
 *
 * start state: 0
 * target state: 5
 * dangerous state: 4
 */

int main()
{
    Environment environment;

    environment.setTransition(0, DOWN, 3);
    environment.setTransition(0, RIGHT, 1);

    environment.setTransition(1, LEFT, 0);
    environment.setTransition(1, RIGHT, 2);
    environment.setTransition(1, DOWN, 4);

    environment.setTransition(2, LEFT, 1);
    environment.setTransition(2, DOWN, 5);

    environment.setTransition(3, UP, 0);
    environment.setTransition(3, RIGHT, 4);
    environment.setTransition(3, DOWN, 6);

    environment.setTransition(4, LEFT, 3);
    environment.setTransition(4, UP, 1);
    environment.setTransition(4, RIGHT, 5);
    environment.setTransition(4, DOWN, 7);

    environment.setTransition(5, UP, 2);
    environment.setTransition(5, LEFT, 4);
    environment.setTransition(5, DOWN, 8);

    environment.setTransition(6, UP, 3);
    environment.setTransition(6, RIGHT, 7);

    environment.setTransition(7, LEFT, 6);
    environment.setTransition(7, UP, 4);
    environment.setTransition(7, RIGHT, 8);

    environment.setTransition(8, LEFT, 7);
    environment.setTransition(8, UP, 5);

    QLearningAgent agent(0.1, 0.7, 0.1, &environment);

    for (int epoch = 0; epoch < 100; epoch++) {
        int state = 0;
        std::pair<int, int> nextTransition;
        agent.startSession();

        for (int step = 0; step < 100000; step++) {
            nextTransition = agent.chooseTransition(state);
            int action = nextTransition.first;
            agent.addActionHistory(state, action);
            state = nextTransition.second;

            if (state != 5 && state != 4)
                continue;

            if (state == 5)
                agent.stopSession(100);
            else if (state == 4)
                agent.stopSession(-100);

            break;
        }
    }

    agent.printQTable();
    return 0;
}
