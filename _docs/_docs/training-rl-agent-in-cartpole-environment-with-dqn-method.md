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

<!-- #region id="tyMytwWZFUhp" -->
# Training RL Agent in CartPole Environment with DQN method
<!-- #endregion -->

```python id="ysNR-nNZEHYs" executionInfo={"status": "ok", "timestamp": 1638447252506, "user_tz": -330, "elapsed": 3832, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
import argparse
from datetime import datetime
import os
import random
from collections import deque

import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
```

```python id="doMRsX9OEImC" executionInfo={"status": "ok", "timestamp": 1638447252507, "user_tz": -330, "elapsed": 4, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
tf.keras.backend.set_floatx("float64")
```

```python colab={"base_uri": "https://localhost:8080/"} id="dCLXisl6ELQ0" executionInfo={"status": "ok", "timestamp": 1638447281630, "user_tz": -330, "elapsed": 438, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="3a8c0d7e-b074-4241-9d48-ff6480fe7d11"
parser = argparse.ArgumentParser()
parser.add_argument("--env", default="CartPole-v0")
parser.add_argument("--lr", type=float, default=0.005)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--gamma", type=float, default=0.95)
parser.add_argument("--eps", type=float, default=1.0)
parser.add_argument("--eps_decay", type=float, default=0.995)
parser.add_argument("--eps_min", type=float, default=0.01)
parser.add_argument("--logdir", default="logs")

args = parser.parse_args([])
logdir = os.path.join(
    args.logdir, parser.prog, args.env, datetime.now().strftime("%Y%m%d-%H%M%S")
)
print(f"Saving training logs to:{logdir}")
writer = tf.summary.create_file_writer(logdir)
```

```python id="wl9MSHtJEXFT" executionInfo={"status": "ok", "timestamp": 1638447289912, "user_tz": -330, "elapsed": 424, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def store(self, state, action, reward, next_state, done):
        self.buffer.append([state, action, reward, next_state, done])

    def sample(self):
        sample = random.sample(self.buffer, args.batch_size)
        states, actions, rewards, next_states, done = map(np.asarray, zip(*sample))
        states = np.array(states).reshape(args.batch_size, -1)
        next_states = np.array(next_states).reshape(args.batch_size, -1)
        return states, actions, rewards, next_states, done

    def size(self):
        return len(self.buffer)
```

```python id="xFoB1mz4EZHC" executionInfo={"status": "ok", "timestamp": 1638447297678, "user_tz": -330, "elapsed": 425, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
class DQN:
    def __init__(self, state_dim, aciton_dim):
        self.state_dim = state_dim
        self.action_dim = aciton_dim
        self.epsilon = args.eps

        self.model = self.nn_model()

    def nn_model(self):
        model = tf.keras.Sequential(
            [
                Input((self.state_dim,)),
                Dense(32, activation="relu"),
                Dense(16, activation="relu"),
                Dense(self.action_dim),
            ]
        )
        model.compile(loss="mse", optimizer=Adam(args.lr))
        return model

    def predict(self, state):
        return self.model.predict(state)

    def get_action(self, state):
        state = np.reshape(state, [1, self.state_dim])
        self.epsilon *= args.eps_decay
        self.epsilon = max(self.epsilon, args.eps_min)
        q_value = self.predict(state)[0]
        if np.random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        return np.argmax(q_value)

    def train(self, states, targets):
        self.model.fit(states, targets, epochs=1)
```

```python id="c3BmqKjDEatV" executionInfo={"status": "ok", "timestamp": 1638447305244, "user_tz": -330, "elapsed": 698, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
class Agent:
    def __init__(self, env):
        self.env = env
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n

        self.model = DQN(self.state_dim, self.action_dim)
        self.target_model = DQN(self.state_dim, self.action_dim)
        self.update_target()

        self.buffer = ReplayBuffer()

    def update_target(self):
        weights = self.model.model.get_weights()
        self.target_model.model.set_weights(weights)

    def replay_experience(self):
        for _ in range(10):
            states, actions, rewards, next_states, done = self.buffer.sample()
            targets = self.model.predict(states)
            next_q_values = self.target_model.predict(next_states)[
                range(args.batch_size),
                np.argmax(self.model.predict(next_states), axis=1),
            ]
            targets[range(args.batch_size), actions] = (
                rewards + (1 - done) * next_q_values * args.gamma
            )
            self.model.train(states, targets)

    def train(self, max_episodes=1000):
        with writer.as_default():
            for ep in range(max_episodes):
                done, episode_reward = False, 0
                observation = self.env.reset()
                while not done:
                    action = self.model.get_action(observation)
                    next_observation, reward, done, _ = self.env.step(action)
                    self.buffer.store(
                        observation, action, reward, next_observation, done
                    )
                    episode_reward += reward
                    observation = next_observation

                if self.buffer.size() >= args.batch_size:
                    self.replay_experience()
                self.update_target()
                print(f"Episode#{ep} Reward:{episode_reward}")
                tf.summary.scalar("episode_reward", episode_reward, step=ep)
```

```python colab={"base_uri": "https://localhost:8080/"} id="sw6rePMLEcxu" executionInfo={"status": "ok", "timestamp": 1638447363342, "user_tz": -330, "elapsed": 30987, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="1ca7d8b7-83c3-4f3d-c981-143fb94ab5cd"
if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    agent = Agent(env)
    agent.train(max_episodes=10)  # Increase max_episodes value
```

<!-- #region id="ixk34zYIEgB_" -->
---
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="VDwjGZWIE5il" executionInfo={"status": "ok", "timestamp": 1638447434341, "user_tz": -330, "elapsed": 3370, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="99504006-22e9-447b-aab4-aa6d3423da9c"
!pip install -q watermark
%reload_ext watermark
%watermark -a "Sparsh A." -m -iv -u -t -d
```

<!-- #region id="haKsOAX2E1XI" -->
---
<!-- #endregion -->

<!-- #region id="4s5DJf8WE2as" -->
**END**
<!-- #endregion -->
