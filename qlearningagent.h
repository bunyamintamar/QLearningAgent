/**
 * @file qlearningagent.h
 * @brief Defines classes for a Q-Learning based learning agent and environment.
 *
 * This header file contains the primary components for a learning agent
 * (QLearningAgent) that uses a Q-Learning algorithm, and an environment
 * (Environment) class that defines state-action transitions. The environment
 * provides transitions between states and actions, while the learning agent
 * uses this setup to perform reward-based learning.
 *
 * - Environment class: Manages transitions between states and actions.
 * - QLearningAgent class: Implements the Q-Learning algorithm
 * to learn optimal actions.
 *
 * @note This code is developed to simulate the training
 * and decision-making processes of an AI agent using the Q-Learning algorithm.
 *
 * @date 2024
 * @version 1.0
 * @author Bünyamin TAMAR
 *
 */

#pragma once

#include <iostream>
#include <vector>
#include <map>

/*----------------------------------------------------------------------------*/

class Environment
{
public:
    Environment();

    void setTransition(int currentState, int action, int nextState);
    std::vector<int> availableActions(int state);
    std::pair<int, int> randomTransition(int state); // { action, nextState }
    int nextState(int state, int action) const;
    void printTransitions() const;

private:
    std::map<int, std::map<int, int>> m_transitions;
};

/*----------------------------------------------------------------------------*/

class QLearningAgent
{
public:
    QLearningAgent(double alpha, double gamma, double epsilon, Environment *env);

    void startSession();
    void stopSession(int reward);
    std::pair<int, int> chooseTransition(int state); // { action, nextState }

    void addActionHistory(int state, int action);
    void printActionHistory();
    void printQTable() const;

private:
    std::pair<int, int> bestTransition(int state);
    double maxQ(int state);
    void updateQTable(int reward);

    std::map<int, std::map<int, int>> m_qTable;
    std::vector<std::pair<int, int>> m_actionHistory;

    double m_alpha;   // Learning rate
    double m_gamma;   // Discounting rate of the next reward
    double m_epsilon; // Discovery rate
    Environment *m_environment;
};
