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

<!-- #region id="KVIjdOWtJ0ZS" -->
# FrozenLake using Q-Learning
<!-- #endregion -->

<!-- #region id="-zYDVzsaOAVV" -->
## Imports
<!-- #endregion -->

```python id="qjoq2JhWMoQI"
import numpy as np
import gym
import random
```

```python id="q2Wrkp1HMoq1"
env = gym.make("FrozenLake-v0")
```

<!-- #region id="5RH3D53tMuG_" -->
## Create the Q-table and initialize it 🗄️
Now, we'll create our Q-table, to know how much rows (states) and columns (actions) we need, we need to calculate the action_size and the state_size
OpenAI Gym provides us a way to do that: env.action_space.n and env.observation_space.n
<!-- #endregion -->

```python id="XXD9WEOzMy5r"
action_size = env.action_space.n
state_size = env.observation_space.n
```

```python colab={"base_uri": "https://localhost:8080/"} id="AW-Ti3swM1Hw" executionInfo={"status": "ok", "timestamp": 1634909516972, "user_tz": -330, "elapsed": 554, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="8cd108e3-eafc-4888-bb98-0aa4371b68a3"
# Create our Q table with state_size rows and action_size columns (64x4)
qtable = np.zeros((state_size, action_size))
print(qtable)
```

<!-- #region id="4wrFaLS1M29a" -->
## Create the hyperparameters ⚙️
Here, we'll specify the hyperparameters
<!-- #endregion -->

```python id="ShVBH2toM7mA"
total_episodes = 20000       # Total episodes
learning_rate = 0.7          # Learning rate
max_steps = 99               # Max steps per episode
gamma = 0.95                 # Discounting rate

# Exploration parameters
epsilon = 1.0                 # Exploration rate
max_epsilon = 1.0             # Exploration probability at start
min_epsilon = 0.01            # Minimum exploration probability 
decay_rate = 0.005            # Exponential decay rate for exploration prob
```

<!-- #region id="j1m7ADqRM722" -->
## The Q learning algorithm 🧠
Now we implement the Q learning algorithm:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="WN0WDTDrNEgA" executionInfo={"status": "ok", "timestamp": 1634909598080, "user_tz": -330, "elapsed": 17917, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="7b02086b-f7c3-4e07-c9ea-7cada07aecd5"
# List of rewards
rewards = []

# 2 For life or until learning is stopped
for episode in range(total_episodes):
    # Reset the environment
    state = env.reset()
    step = 0
    done = False
    total_rewards = 0
    
    for step in range(max_steps):
        # 3. Choose an action a in the current world state (s)
        ## First we randomize a number
        exp_exp_tradeoff = random.uniform(0, 1)
        
        ## If this number > greater than epsilon --> exploitation (taking the biggest Q value for this state)
        if exp_exp_tradeoff > epsilon:
            action = np.argmax(qtable[state,:])
            #print(exp_exp_tradeoff, "action", action)

        # Else doing a random choice --> exploration
        else:
            action = env.action_space.sample()
            #print("action random", action)
            
        
        # Take the action (a) and observe the outcome state(s') and reward (r)
        new_state, reward, done, info = env.step(action)

        # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
        # qtable[new_state,:] : all the actions we can take from new state
        qtable[state, action] = qtable[state, action] + learning_rate * (reward + gamma * np.max(qtable[new_state, :]) - qtable[state, action])
        
        total_rewards += reward
        
        # Our new state is state
        state = new_state
        
        # If done (if we're dead) : finish episode
        if done == True: 
            break
        
    # Reduce epsilon (because we need less and less exploration)
    epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode) 
    rewards.append(total_rewards)
    

print ("Score over time: " +  str(sum(rewards)/total_episodes))
print(qtable)
```

<!-- #region id="xy4odqxoNGhE" -->
## Use our Q-table to play FrozenLake ! 👾
After 10 000 episodes, our Q-table can be used as a "cheatsheet" to play FrozenLake"

By running this cell you can see our agent playing FrozenLake.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="kBi_qh77NJ6a" executionInfo={"status": "ok", "timestamp": 1634909627130, "user_tz": -330, "elapsed": 633, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="77a9234c-cab1-484b-9a40-465d046c0c08"
env.reset()

for episode in range(5):
    state = env.reset()
    step = 0
    done = False
    print("****************************************************")
    print("EPISODE ", episode)

    for step in range(max_steps):
        
        # Take the action (index) that have the maximum expected future reward given that state
        action = np.argmax(qtable[state,:])
        
        new_state, reward, done, info = env.step(action)
        
        if done:
            # Here, we decide to only print the last state (to see if our agent is on the goal or fall into an hole)
            env.render()
            if new_state == 15:
                print("We reached our Goal 🏆")
            else:
                print("We fell into a hole ☠️")
            
            # We print the number of step it took.
            print("Number of steps", step)
            
            break
        state = new_state
env.close()
```

<!-- #region id="ZR-yGpEnOVUz" -->
## PyTorch version
<!-- #endregion -->

```python id="krU58JoumsLF"
import gym
import collections
from torch.utils.tensorboard import SummaryWriter
```

```python id="vtzB-mVNmwZC"
ENV_NAME = "FrozenLake-v0"
GAMMA = 0.9
ALPHA = 0.2
TEST_EPISODES = 20
```

```python id="PkSo69Vvmu5t"
class Agent:
    def __init__(self):
        self.env = gym.make(ENV_NAME)
        self.state = self.env.reset()
        self.values = collections.defaultdict(float)

    def sample_env(self):
        action = self.env.action_space.sample()
        old_state = self.state
        new_state, reward, is_done, _ = self.env.step(action)
        self.state = self.env.reset() if is_done else new_state
        return old_state, action, reward, new_state

    def best_value_and_action(self, state):
        best_value, best_action = None, None
        for action in range(self.env.action_space.n):
            action_value = self.values[(state, action)]
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_value, best_action

    def value_update(self, s, a, r, next_s):
        best_v, _ = self.best_value_and_action(next_s)
        new_v = r + GAMMA * best_v
        old_v = self.values[(s, a)]
        self.values[(s, a)] = old_v * (1-ALPHA) + new_v * ALPHA

    def play_episode(self, env):
        total_reward = 0.0
        state = env.reset()
        while True:
            _, action = self.best_value_and_action(state)
            new_state, reward, is_done, _ = env.step(action)
            total_reward += reward
            if is_done:
                break
            state = new_state
        return total_reward
```

```python colab={"base_uri": "https://localhost:8080/"} id="2-1z_6bGmtJX" executionInfo={"status": "ok", "timestamp": 1634480185460, "user_tz": -330, "elapsed": 53454, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="3d8d2c25-7ec0-4adb-e1e8-f5a617b234c8"
if __name__ == "__main__":
    test_env = gym.make(ENV_NAME)
    agent = Agent()
    writer = SummaryWriter(comment="-q-learning")

    iter_no = 0
    best_reward = 0.0
    while True:
        iter_no += 1
        s, a, r, next_s = agent.sample_env()
        agent.value_update(s, a, r, next_s)

        reward = 0.0
        for _ in range(TEST_EPISODES):
            reward += agent.play_episode(test_env)
        reward /= TEST_EPISODES
        writer.add_scalar("reward", reward, iter_no)
        if reward > best_reward:
            print("Best reward updated %.3f -> %.3f" % (
                best_reward, reward))
            best_reward = reward
        if reward > 0.80:
            print("Solved in %d iterations!" % iter_no)
            break
    writer.close()
```

```python id="8BX2Wq0mm2AI"
%load_ext tensorboard
%tensorboard --logdir runs
```

<!-- #region id="UrnvYHbuOWlX" -->
<p><center><img src='_images/T587798_1.png'></center></p>
<!-- #endregion -->
