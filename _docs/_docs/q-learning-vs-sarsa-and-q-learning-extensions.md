---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.7
  kernelspec:
    display_name: Python 3
    name: python3
---

<!-- #region id="j0UA7LFAOO8i" -->
# Q-Learning vs SARSA and Q-Learning extensions
<!-- #endregion -->

<!-- #region id="8mE45pCqOoab" -->
Temporal-difference (TD) learning is a combination of these two approaches. It learns directly from experience by sampling, but also bootstraps. This represents a breakthrough in capability that allows agents to learn optimal strategies in any environment. Prior to this point learning was so slow it made problems intractable or you needed a full model of the environment.

In 1989, an implementation of TD learning called Q-learning made it easier for mathematicians to prove convergence. This algorithm is credited with kick-starting the excitement in RL.

SARSA was developed shortly after Q-learning to provide a more general solution to TD learning. The main difference from Q-learning is the lack of argmax in the delta. Instead it calculates the expected return by averaging over all runs (in an online manner).
<!-- #endregion -->

<!-- #region id="g35rHYTLOQcQ" -->
Two fundamental RL algorithms, both remarkably useful, even today. One of the primary reasons for their popularity is that they are simple, because by default they only work with discrete state and action spaces. Of course it is possible to improve them to work with continuous state/action spaces, but consider discretizing to keep things rediculously simple.
<!-- #endregion -->

<!-- #region id="FMcnfqT6PCRD" -->
## Setup
<!-- #endregion -->

```python id="Z4vNwpVuOWFH"
!pip install pygame==1.9.6 pandas==1.0.5 matplotlib==3.2.1
!pip install --upgrade git+git://github.com/david-abel/simple_rl.git@77c0d6b910efbe8bdd5f4f87337c5bc4aed0d79c > /dev/null
```

```python id="e9yVBzYjOd8U"
import numpy as np
import pandas as pd
import math
import json
import os
from collections import defaultdict

# Other imports.
from simple_rl.agents import Agent, QLearningAgent, RandomAgent
from simple_rl.agents import DoubleQAgent, DelayedQAgent
from simple_rl.tasks import GridWorldMDP
from simple_rl.run_experiments import run_single_agent_on_mdp

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, AutoMinorLocator
matplotlib.use("agg", force=True)
```

```python id="QkY1tZFJPAUB"
%matplotlib inline
```

<!-- #region id="q00CeCv3OrNW" -->
## SARSA agent
<!-- #endregion -->

```python id="vLqH8pRdO9MR"
class SARSAAgent(QLearningAgent):
    def __init__(self, actions, goal_reward, name="SARSA",
                 alpha=0.1, gamma=0.99, epsilon=0.1, explore="uniform", anneal=False):
        self.goal_reward = goal_reward
        QLearningAgent.__init__(
            self,
            actions=list(actions),
            name=name,
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
            explore=explore,
            anneal=anneal)

    def policy(self, state):
        return self.get_max_q_action(state)

    def act(self, state, reward, learning=True):
        '''
        This is mostly the same as the base QLearningAgent class. Except that
        the update procedure now generates the action.
        '''
        if learning:
            action = self.update(
                self.prev_state, self.prev_action, reward, state)
        else:
            if self.explore == "softmax":
                # Softmax exploration
                action = self.soft_max_policy(state)
            else:
                # Uniform exploration
                action = self.epsilon_greedy_q_policy(state)

        self.prev_state = state
        self.prev_action = action
        self.step_number += 1

        # Anneal params.
        if learning and self.anneal:
            self._anneal()

        return action

    def update(self, state, action, reward, next_state):
        '''
        Args:
            state (State)
            action (str)
            reward (float)
            next_state (State)
        Summary:
            Updates the internal Q Function according to the Bellman Equation
            using a SARSA update
        '''

        if self.explore == "softmax":
            # Softmax exploration
            next_action = self.soft_max_policy(next_state)
        else:
            # Uniform exploration
            next_action = self.epsilon_greedy_q_policy(next_state)

        # Update the Q Function.
        prev_q_val = self.get_q_value(state, action)
        next_q_val = self.get_q_value(next_state, next_action)
        self.q_func[state][action] = prev_q_val + self.alpha * \
            (reward + self.gamma * next_q_val - prev_q_val)
        return next_action
```

<!-- #region id="AexwAJhfPGSj" -->
## Experiment
<!-- #endregion -->

<!-- #region id="Qa3rNmBZPRgc" -->
Basically I’m training an agent for a maximum of 100 steps, for 500 episodes, averaging over 100 repeats.

Feel free to tinker with the settimgs.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="Kc-4plgNPenw" executionInfo={"status": "ok", "timestamp": 1634457312570, "user_tz": -330, "elapsed": 87866, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="8d7293ec-e33b-4f13-c924-4e82d7f03b71"
np.random.seed(42)
instances = 100
n_episodes = 500
alpha = 0.1
epsilon = 0.1

# Setup MDP, Agents.
mdp = GridWorldMDP(
    width=10, height=4, init_loc=(1, 1), goal_locs=[(10, 1)],
    lava_locs=[(x, 1) for x in range(2, 10)], is_lava_terminal=True, gamma=1.0, walls=[], slip_prob=0.0, step_cost=1.0, lava_cost=100.0)

print("Q-Learning")
rewards = np.zeros((n_episodes, instances))
for instance in range(instances):
    ql_agent = QLearningAgent(
        mdp.get_actions(),
        epsilon=epsilon,
        alpha=alpha)
    # mdp.visualize_learning(ql_agent, delay=0.0001)
    print("  Instance " + str(instance) + " of " + str(instances) + ".")
    terminal, num_steps, reward = run_single_agent_on_mdp(
        ql_agent, mdp, episodes=n_episodes, steps=100)
    rewards[:, instance] = reward
df = pd.DataFrame(rewards.mean(axis=1))
df.to_json("q_learning_cliff_rewards.json")

print("SARSA")
rewards = np.zeros((n_episodes, instances))
for instance in range(instances):
    sarsa_agent = SARSAAgent(
        mdp.get_actions(),
        goal_reward=0,
        epsilon=epsilon,
        alpha=alpha)
    # mdp.visualize_learning(sarsa_agent, delay=0.0001)
    print("  Instance " + str(instance) + " of " + str(instances) + ".")
    terminal, num_steps, reward = run_single_agent_on_mdp(
        sarsa_agent, mdp, episodes=n_episodes, steps=100)
    rewards[:, instance] = reward
df = pd.DataFrame(rewards.mean(axis=1))
df.to_json("sarsa_cliff_rewards.json")
```

<!-- #region id="yuCwZbb6PgMl" -->
## Results
Now you can plot the results for each of the agents.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 279} id="eYGv2mJAPoEM" executionInfo={"status": "ok", "timestamp": 1634457346178, "user_tz": -330, "elapsed": 666, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="aee834e2-23aa-449f-b8d9-8743160571ff"
data_files = [("Q-Learning", "q_learning_cliff_rewards.json"),
              ("SARSA", "sarsa_cliff_rewards.json")]

fig, ax = plt.subplots()
for j, (name, data_file) in enumerate(data_files):
    df = pd.read_json(data_file)
    x = range(len(df))
    y = df.sort_index().values
    ax.plot(x,
            y,
            linestyle='solid',
            label=name)
ax.set_xlabel('Episode')
ax.set_ylabel('Averaged Sum of Rewards')
ax.legend(loc='lower right')
plt.show()
```

<!-- #region id="Aggkb7mZTJLS" -->
## Delayed Q-learning vs. Double Q-learning vs. Q-Learning
Delayed Q-learning and double Q-learning are two extensions to Q-learning that are used throughout RL, so it’s worth considering them in a simple form. Delayed Q-learning simply delays any estimate until there is a statistically significant sample of observations. Slowing update with an exponentially weighted moving average is a similar strategy. Double Q-learning includes two Q-tables, in essence two value estimates, to reduce bias.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="5_XpUIXDTKlg" executionInfo={"status": "ok", "timestamp": 1634458252477, "user_tz": -330, "elapsed": 26158, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="fa7aa67c-2675-460d-ccb5-7a3336fc4718"
np.random.seed(42)
instances = 10
n_episodes = 1000
alpha = 0.1
epsilon = 0.1

# Setup MDP, Agents.
mdp = GridWorldMDP(
    width=10, height=4, init_loc=(1, 1), goal_locs=[(10, 1)],
    lava_locs=[(x, 1) for x in range(2, 10)], is_lava_terminal=True, gamma=1.0, walls=[], slip_prob=0.0, step_cost=1.0, lava_cost=100.0)

print("Q-Learning")
rewards = np.zeros((n_episodes, instances))
for instance in range(instances):
    ql_agent = QLearningAgent(
        mdp.get_actions(),
        epsilon=epsilon,
        alpha=alpha)
    print("  Instance " + str(instance) + " of " + str(instances) + ".")
    terminal, num_steps, reward = run_single_agent_on_mdp(
        ql_agent, mdp, episodes=n_episodes, steps=100)
    rewards[:, instance] = reward
df = pd.DataFrame(rewards.mean(axis=1))
df.to_json("q_learning_cliff_rewards.json")

print("Double-Q-Learning")
rewards = np.zeros((n_episodes, instances))
for instance in range(instances):
    ql_agent = DoubleQAgent(
        mdp.get_actions(),
        epsilon=epsilon,
        alpha=alpha)
    # mdp.visualize_learning(ql_agent, delay=0.0001)
    print("  Instance " + str(instance) + " of " + str(instances) + ".")
    terminal, num_steps, reward = run_single_agent_on_mdp(
        ql_agent, mdp, episodes=n_episodes, steps=100)
    rewards[:, instance] = reward
df = pd.DataFrame(rewards.mean(axis=1))
df.to_json("double_q_learning_cliff_rewards.json")

print("Delayed-Q-Learning")
rewards = np.zeros((n_episodes, instances))
for instance in range(instances):
    ql_agent = DelayedQAgent(
        mdp.get_actions(),
        epsilon1=alpha)
    # mdp.visualize_learning(ql_agent, delay=0.0001)
    print("  Instance " + str(instance) + " of " + str(instances) + ".")
    terminal, num_steps, reward = run_single_agent_on_mdp(
        ql_agent, mdp, episodes=n_episodes, steps=100)
    rewards[:, instance] = reward
df = pd.DataFrame(rewards.mean(axis=1))
df.to_json("delayed_q_learning_cliff_rewards.json")
```

<!-- #region id="t9p0IGhXTUu3" -->
Below is the code to visualise the training of the three agents. As usual, there are a maximum of 100 steps, over 200 episodes, averaged over 10 repeats. Feel free to tinker with those settings.

The results show the differences between the three algorithms. In general, double Q-learning tends to be more stable than Q-learning. And delayed Q-learning is more robust against outliers, but can be problematic in environments with larger state/action spaces. I.e. you might have to wait for a long time to get the required number of samples for a particular state-action pair, which will delay further exploration.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 279} id="u0yVCvbaTYhQ" executionInfo={"status": "ok", "timestamp": 1634458287918, "user_tz": -330, "elapsed": 16, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="8a41c7ce-4c33-41d6-b59f-365778b6de3d"
data_files = [("Q-Learning", "q_learning_cliff_rewards.json"),
              ("Double Q-Learning", "double_q_learning_cliff_rewards.json"),
              ("Delayed Q-Learning", "delayed_q_learning_cliff_rewards.json")]
fig, ax = plt.subplots()
for j, (name, data_file) in enumerate(data_files):
    df = pd.read_json(data_file)
    x = range(len(df))
    y = df.sort_index().values
    ax.plot(x,
            y,
            linestyle='solid',
            label=name)
ax.set_xlabel('Episode')
ax.set_ylabel('Averaged Sum of Rewards')
ax.legend(loc='lower right')
plt.show()
```
