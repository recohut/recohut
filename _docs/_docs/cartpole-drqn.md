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
# Training RL Agent in CartPole Environment with DRQN method
<!-- #endregion -->

```python id="ysNR-nNZEHYs"
import tensorflow as tf
from datetime import datetime
import os
from tensorflow.keras.layers import Input, Dense, LSTM
from tensorflow.keras.optimizers import Adam
import gym
import argparse
import numpy as np
from collections import deque
import random
```

```python id="doMRsX9OEImC"
tf.keras.backend.set_floatx("float64")
```

```python colab={"base_uri": "https://localhost:8080/"} id="dCLXisl6ELQ0" executionInfo={"status": "ok", "timestamp": 1638448200655, "user_tz": -330, "elapsed": 443, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="449e6724-08b8-4a67-c8c1-c47524f14ad5"
parser = argparse.ArgumentParser()
parser.add_argument("--env", default="CartPole-v0")
parser.add_argument("--lr", type=float, default=0.005)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--time_steps", type=int, default=4)
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

```python id="xFoB1mz4EZHC"
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def store(self, state, action, reward, next_state, done):
        self.buffer.append([state, action, reward, next_state, done])

    def sample(self):
        sample = random.sample(self.buffer, args.batch_size)
        states, actions, rewards, next_states, done = map(np.asarray, zip(*sample))
        states = np.array(states).reshape(args.batch_size, args.time_steps, -1)
        next_states = np.array(next_states).reshape(
            args.batch_size, args.time_steps, -1
        )
        return states, actions, rewards, next_states, done

    def size(self):
        return len(self.buffer)
```

```python id="c3BmqKjDEatV"
class DRQN:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epsilon = args.eps

        self.opt = Adam(args.lr)
        self.compute_loss = tf.keras.losses.MeanSquaredError()
        self.model = self.nn_model()

    def nn_model(self):
        return tf.keras.Sequential(
            [
                Input((args.time_steps, self.state_dim)),
                LSTM(32, activation="tanh"),
                Dense(16, activation="relu"),
                Dense(self.action_dim),
            ]
        )

    def predict(self, state):
        return self.model.predict(state)

    def get_action(self, state):
        state = np.reshape(state, [1, args.time_steps, self.state_dim])
        self.epsilon *= args.eps_decay
        self.epsilon = max(self.epsilon, args.eps_min)
        q_value = self.predict(state)[0]
        if np.random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        return np.argmax(q_value)

    def train(self, states, targets):
        targets = tf.stop_gradient(targets)
        with tf.GradientTape() as tape:
            logits = self.model(states, training=True)
            assert targets.shape == logits.shape
            loss = self.compute_loss(targets, logits)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
```

```python id="PQZ6EpazGSub"
class Agent:
    def __init__(self, env):
        self.env = env
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n

        self.states = np.zeros([args.time_steps, self.state_dim])

        self.model = DRQN(self.state_dim, self.action_dim)
        self.target_model = DRQN(self.state_dim, self.action_dim)
        self.update_target()

        self.buffer = ReplayBuffer()

    def update_target(self):
        weights = self.model.model.get_weights()
        self.target_model.model.set_weights(weights)

    def replay_experience(self):
        for _ in range(10):
            states, actions, rewards, next_states, done = self.buffer.sample()
            targets = self.model.predict(states)
            next_q_values = self.target_model.predict(next_states).max(axis=1)
            targets[range(args.batch_size), actions] = (
                rewards + (1 - done) * next_q_values * args.gamma
            )
            self.model.train(states, targets)

    def update_states(self, next_state):
        self.states = np.roll(self.states, -1, axis=0)
        self.states[-1] = next_state

    def train(self, max_episodes=1000):
        with writer.as_default():
            for ep in range(max_episodes):
                done, episode_reward = False, 0
                self.states = np.zeros([args.time_steps, self.state_dim])
                self.update_states(self.env.reset())
                while not done:
                    action = self.model.get_action(self.states)
                    next_state, reward, done, _ = self.env.step(action)
                    prev_states = self.states
                    self.update_states(next_state)
                    self.buffer.store(
                        prev_states, action, reward * 0.01, self.states, done
                    )
                    episode_reward += reward

                if self.buffer.size() >= args.batch_size:
                    self.replay_experience()
                self.update_target()
                print(f"Episode#{ep} Reward:{episode_reward}")
                tf.summary.scalar("episode_reward", episode_reward, step=ep)
```

```python colab={"base_uri": "https://localhost:8080/"} id="sw6rePMLEcxu" executionInfo={"status": "ok", "timestamp": 1638448246003, "user_tz": -330, "elapsed": 18059, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="ba33b4f7-851b-4359-f321-f61ccb4ab275"
if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    agent = Agent(env)
    agent.train(max_episodes=10)  # Increase max_episodes value
```

<!-- #region id="ixk34zYIEgB_" -->
---
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="VDwjGZWIE5il" executionInfo={"status": "ok", "timestamp": 1638448258528, "user_tz": -330, "elapsed": 3221, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="9331af1c-fceb-4a39-a7d5-d2afe76d9303"
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
