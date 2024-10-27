#include "qlearningagent.h"
#include <limits>
#include <stdexcept>
#include <algorithm>
#include <cstdlib>
#include <ctime>

/*----------------------------------------------------------------------------*/

Environment::Environment()
{
    srand(static_cast<unsigned int>(time(0)));
}

void Environment::setTransition(int currentState, int action, int nextState)
{
    m_transitions[currentState][action] = nextState;
}

std::vector<int> Environment::availableActions(int state)
{
    std::vector<int> actions;
    auto stateIt = m_transitions.find(state);

    if (stateIt == m_transitions.end())
        return actions;

    actions.resize(stateIt->second.size());
    std::transform(stateIt->second.begin(),
                   stateIt->second.end(), actions.begin(),
                   [](const std::pair<int, int>& actionPair) {
                       return actionPair.first; });

    return actions;
}

std::pair<int, int> Environment::randomTransition(int state)
{
    std::vector<int> actions = availableActions(state);

    if (actions.empty())
        throw std::runtime_error("No available actions for the given state");

    int randomAction = actions[rand() % actions.size()];
    int newState = nextState(state, randomAction);
    return std::make_pair(randomAction, newState);
}

int Environment::nextState(int state, int action) const
{
    auto stateIt = m_transitions.find(state);

    if (stateIt == m_transitions.end()) {
        std::cerr << "Invalid state!" << std::endl;
        return -1;
    }

    auto actionIt = stateIt->second.find(action);

    if (actionIt == stateIt->second.end()) {
        std::cerr << "Invalid action!" << std::endl;
        return -1;
    }

    return actionIt->second;
}

void Environment::printTransitions() const
{
    for (const auto& statePair : m_transitions) {
        int currentState = statePair.first;
        std::cout << "State " << currentState << " transitions: ";

        for (const auto& actionPair : statePair.second) {
            int action = actionPair.first;
            int nxtState = actionPair.second;
            std::cout << "Action " << action
                      << " -> State " << nxtState << ", ";
        }

        std::cout << std::endl;
    }
}

/*----------------------------------------------------------------------------*/

QLearningAgent::QLearningAgent(double alpha,
                               double gamma,
                               double epsilon,
                               Environment *env)
    : m_alpha(alpha)
    , m_gamma(gamma)
    , m_epsilon(epsilon)
    , m_environment(env) {}

void QLearningAgent::startSession()
{
    m_actionHistory.clear();
}

void QLearningAgent::stopSession(int reward)
{
    updateQTable(reward);
}

std::pair<int, int> QLearningAgent::chooseTransition(int state)
{
    if (m_qTable.empty()
        || m_qTable[state].empty()
        || static_cast<double>(rand()) / RAND_MAX < m_epsilon)
        return m_environment->randomTransition(state);

    return bestTransition(state);
}

void QLearningAgent::addActionHistory(int state, int action)
{
    std::pair<int, int> item = std::make_pair(state, action);
    m_actionHistory.push_back(item);
}

void QLearningAgent::printActionHistory()
{
    std::cout << "Action History:" << std::endl;

    for (const auto& item : m_actionHistory)
        std::cout << "State: " << item.first
                  << ", Action: " << item.second << std::endl;
}

void QLearningAgent::printQTable() const
{
    std::cout << "Q-Table:" << std::endl;

    for (const auto& statePair : m_qTable) {
        int state = statePair.first;
        const auto& actionMap = statePair.second;

        std::cout << "State " << state << ": ";

        for (const auto& actionPair : actionMap) {
            int action = actionPair.first;
            double qValue = actionPair.second;
            std::cout << "Action " << action
                      << " -> Q-Value: " << qValue << ", ";
        }

        std::cout << std::endl;
    }
}

std::pair<int, int> QLearningAgent::bestTransition(int state)
{
    if (m_qTable.empty())
        throw std::runtime_error("Q-Table is empty!");

    std::map<int, int> actionMap = m_qTable[state];

    if (actionMap.empty())
        throw std::runtime_error("No record for the state");

    int action = std::max_element(actionMap.begin(), actionMap.end(),
                                  [](const std::pair<int, int>& a,
                                     const std::pair<int, int>& b) {
                                      return a.second < b.second;
                                  })->first;

    int nextState = m_environment->nextState(state, action);
    return std::make_pair(action, nextState);
}

double QLearningAgent::maxQ(int state)
{
    if (state == -1)
        return 0.0;

    auto nextStateActions = m_qTable[state];

    if (nextStateActions.empty())
        return 0.0;

    return std::max_element(nextStateActions.begin(), nextStateActions.end(),
                            [](const std::pair<int, double>& a,
                               const std::pair<int, double>& b) {
                                return a.second < b.second;
                            })->second;
}

void QLearningAgent::updateQTable(int reward)
{
    if (m_actionHistory.empty()) {
        std::cerr << "Action history is empty. No updates to Q-Table."
                  << std::endl;
        return;
    }

    double totalReward = reward;

    for (size_t i = m_actionHistory.size(); i > 0; --i) {
        int state = m_actionHistory[i - 1].first;
        int action = m_actionHistory[i - 1].second;
        int nextState = m_environment->nextState(state, action);
        double currentQValue = m_qTable[state][action];

        currentQValue += m_alpha
                         * (totalReward
                            + m_gamma * maxQ(nextState)
                            - currentQValue);
        m_qTable[state][action] = currentQValue;

        totalReward = currentQValue;
    }
}
