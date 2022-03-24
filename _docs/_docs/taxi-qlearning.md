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

<!-- #region id="-B_w9TyQwsh1" colab_type="text" -->
# Solving the Taxi problem with Q-learning
<!-- #endregion -->

```python id="-I43sfZUChSq" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 187} outputId="147af06a-6909-4e84-9221-c2f9b6b46cb9" executionInfo={"status": "ok", "timestamp": 1591549526118, "user_tz": -330, "elapsed": 1081, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import torch
import gym

env = gym.make('Taxi-v3')

n_state = env.observation_space.n
print(n_state)

n_action = env.action_space.n
print(n_action)

env.reset()

env.render()
```

```python id="f5wPKMuLtlKb" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 153} outputId="11e05c4f-d17b-440b-a452-af782ebdbf44" executionInfo={"status": "ok", "timestamp": 1591549595738, "user_tz": -330, "elapsed": 1334, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
env.step(1)
env.step(4) #pickup

env.render()
```

```python id="QDa718cqv5Py" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 153} outputId="a14417bc-1a19-406f-d558-6286aa2f4e8d" executionInfo={"status": "ok", "timestamp": 1591549641097, "user_tz": -330, "elapsed": 1159, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
env.step(0)
env.step(0)
env.step(0)
env.step(0)
env.step(3)
env.step(5) #drop

env.render()
```

```python id="m8w_sVv6tMZ0" colab_type="code" colab={}
def gen_epsilon_greedy_policy(n_action, epsilon):
    def policy_function(state, Q):
        probs = torch.ones(n_action) * epsilon / n_action
        best_action = torch.argmax(Q[state]).item()
        probs[best_action] += 1.0 - epsilon
        action = torch.multinomial(probs, 1).item()
        return action
    return policy_function


from collections import defaultdict


def q_learning(env, gamma, n_episode, alpha):
    """
    Obtain the optimal policy with off-policy Q-learning method
    @param env: OpenAI Gym environment
    @param gamma: discount factor
    @param n_episode: number of episodes
    @return: the optimal Q-function, and the optimal policy
    """
    n_action = env.action_space.n
    Q = defaultdict(lambda: torch.zeros(n_action))
    for episode in range(n_episode):
        state = env.reset()
        is_done = False
        while not is_done:
            action = epsilon_greedy_policy(state, Q)
            next_state, reward, is_done, info = env.step(action)
            td_delta = reward + gamma * torch.max(Q[next_state]) - Q[state][action]
            Q[state][action] += alpha * td_delta
            length_episode[episode] += 1
            total_reward_episode[episode] += reward
            if is_done:
                break
            state = next_state
    policy = {}
    for state, actions in Q.items():
        policy[state] = torch.argmax(actions).item()
    return Q, policy

gamma = 1

n_episode = 500

alpha = 0.4

epsilon = 0.1

epsilon_greedy_policy = gen_epsilon_greedy_policy(env.action_space.n, epsilon)

length_episode = [0] * n_episode
total_reward_episode = [0] * n_episode

optimal_Q, optimal_policy = q_learning(env, gamma, n_episode, alpha)
```

```python id="Dgq01sIptMXo" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 573} outputId="94aa89f6-061b-42ad-fe85-4603f1d9662b" executionInfo={"status": "ok", "timestamp": 1591549695021, "user_tz": -330, "elapsed": 1511, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import matplotlib.pyplot as plt
plt.plot(length_episode)
plt.title('Episode length over time')
plt.xlabel('Episode')
plt.ylabel('Length')
plt.show()


plt.plot(total_reward_episode)
plt.title('Episode reward over time')
plt.xlabel('Episode')
plt.ylabel('Total reward')
plt.show()
```

```python id="L73gPO6bwQ3S" colab_type="code" colab={}

```
