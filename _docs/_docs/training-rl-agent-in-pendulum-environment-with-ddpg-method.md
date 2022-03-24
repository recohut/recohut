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
# Training RL Agent in Pendulum Environment with DDPG method
<!-- #endregion -->

```python id="ysNR-nNZEHYs" executionInfo={"status": "ok", "timestamp": 1638449957580, "user_tz": -330, "elapsed": 3143, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
import argparse
import os
import random
from collections import deque
from datetime import datetime
import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Lambda, concatenate
```

```python id="doMRsX9OEImC" executionInfo={"status": "ok", "timestamp": 1638449957582, "user_tz": -330, "elapsed": 5, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
tf.keras.backend.set_floatx("float64")
```

```python colab={"base_uri": "https://localhost:8080/"} id="dCLXisl6ELQ0" executionInfo={"status": "ok", "timestamp": 1638449966826, "user_tz": -330, "elapsed": 733, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="81c04f44-9b2a-4ab4-b330-571ce9fc82b7"
parser = argparse.ArgumentParser()
parser.add_argument("--env", default="Pendulum-v0")
parser.add_argument("--actor_lr", type=float, default=0.0005)
parser.add_argument("--critic_lr", type=float, default=0.001)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--tau", type=float, default=0.05)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--train_start", type=int, default=2000)
parser.add_argument("--logdir", default="logs")

args = parser.parse_args([])
logdir = os.path.join(
    args.logdir, parser.prog, args.env, datetime.now().strftime("%Y%m%d-%H%M%S")
)
print(f"Saving training logs to:{logdir}")
writer = tf.summary.create_file_writer(logdir)
```

```python id="Y56QUuw5K2EH" executionInfo={"status": "ok", "timestamp": 1638449968294, "user_tz": -330, "elapsed": 6, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
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

```python id="p9IrVspCMzrU" executionInfo={"status": "ok", "timestamp": 1638449975628, "user_tz": -330, "elapsed": 624, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
class Actor:
    def __init__(self, state_dim, action_dim, action_bound):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.model = self.nn_model()
        self.opt = tf.keras.optimizers.Adam(args.actor_lr)

    def nn_model(self):
        return tf.keras.Sequential(
            [
                Input((self.state_dim,)),
                Dense(32, activation="relu"),
                Dense(32, activation="relu"),
                Dense(self.action_dim, activation="tanh"),
                Lambda(lambda x: x * self.action_bound),
            ]
        )

    def train(self, states, q_grads):
        with tf.GradientTape() as tape:
            grads = tape.gradient(
                self.model(states), self.model.trainable_variables, -q_grads
            )
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))

    def predict(self, state):
        return self.model.predict(state)

    def get_action(self, state):
        state = np.reshape(state, [1, self.state_dim])
        return self.model.predict(state)[0]
```

```python id="uSFhh4DuM0v4" executionInfo={"status": "ok", "timestamp": 1638449984893, "user_tz": -330, "elapsed": 810, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
class Critic:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model = self.nn_model()
        self.opt = tf.keras.optimizers.Adam(args.critic_lr)

    def nn_model(self):
        state_input = Input((self.state_dim,))
        s1 = Dense(64, activation="relu")(state_input)
        s2 = Dense(32, activation="relu")(s1)
        action_input = Input((self.action_dim,))
        a1 = Dense(32, activation="relu")(action_input)
        c1 = concatenate([s2, a1], axis=-1)
        c2 = Dense(16, activation="relu")(c1)
        output = Dense(1, activation="linear")(c2)
        return tf.keras.Model([state_input, action_input], output)

    def predict(self, inputs):
        return self.model.predict(inputs)

    def q_gradients(self, states, actions):
        actions = tf.convert_to_tensor(actions)
        with tf.GradientTape() as tape:
            tape.watch(actions)
            q_values = self.model([states, actions])
            q_values = tf.squeeze(q_values)
        return tape.gradient(q_values, actions)

    def compute_loss(self, v_pred, td_targets):
        mse = tf.keras.losses.MeanSquaredError()
        return mse(td_targets, v_pred)

    def train(self, states, actions, td_targets):
        with tf.GradientTape() as tape:
            v_pred = self.model([states, actions], training=True)
            assert v_pred.shape == td_targets.shape
            loss = self.compute_loss(v_pred, tf.stop_gradient(td_targets))
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss
```

```python id="1JQRAq0yM2gq" executionInfo={"status": "ok", "timestamp": 1638449995479, "user_tz": -330, "elapsed": 970, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
class Agent:
    def __init__(self, env):
        self.env = env
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.action_bound = self.env.action_space.high[0]

        self.buffer = ReplayBuffer()

        self.actor = Actor(self.state_dim, self.action_dim, self.action_bound)
        self.critic = Critic(self.state_dim, self.action_dim)

        self.target_actor = Actor(self.state_dim, self.action_dim, self.action_bound)
        self.target_critic = Critic(self.state_dim, self.action_dim)

        actor_weights = self.actor.model.get_weights()
        critic_weights = self.critic.model.get_weights()
        self.target_actor.model.set_weights(actor_weights)
        self.target_critic.model.set_weights(critic_weights)

    def update_target(self):
        actor_weights = self.actor.model.get_weights()
        t_actor_weights = self.target_actor.model.get_weights()
        critic_weights = self.critic.model.get_weights()
        t_critic_weights = self.target_critic.model.get_weights()

        for i in range(len(actor_weights)):
            t_actor_weights[i] = (
                args.tau * actor_weights[i] + (1 - args.tau) * t_actor_weights[i]
            )

        for i in range(len(critic_weights)):
            t_critic_weights[i] = (
                args.tau * critic_weights[i] + (1 - args.tau) * t_critic_weights[i]
            )

        self.target_actor.model.set_weights(t_actor_weights)
        self.target_critic.model.set_weights(t_critic_weights)

    def get_td_target(self, rewards, q_values, dones):
        targets = np.asarray(q_values)
        for i in range(q_values.shape[0]):
            if dones[i]:
                targets[i] = rewards[i]
            else:
                targets[i] = rewards[i] + args.gamma * q_values[i]
        return targets

    def add_ou_noise(self, x, rho=0.15, mu=0, dt=1e-1, sigma=0.2, dim=1):
        return (
            x + rho * (mu - x) * dt + sigma * np.sqrt(dt) * np.random.normal(size=dim)
        )

    def replay_experience(self):
        for _ in range(10):
            states, actions, rewards, next_states, dones = self.buffer.sample()
            target_q_values = self.target_critic.predict(
                [next_states, self.target_actor.predict(next_states)]
            )
            td_targets = self.get_td_target(rewards, target_q_values, dones)

            self.critic.train(states, actions, td_targets)

            s_actions = self.actor.predict(states)
            s_grads = self.critic.q_gradients(states, s_actions)
            grads = np.array(s_grads).reshape((-1, self.action_dim))
            self.actor.train(states, grads)
            self.update_target()

    def train(self, max_episodes=1000):
        with writer.as_default():
            for ep in range(max_episodes):
                episode_reward, done = 0, False

                state = self.env.reset()
                bg_noise = np.zeros(self.action_dim)
                while not done:
                    # self.env.render()
                    action = self.actor.get_action(state)
                    noise = self.add_ou_noise(bg_noise, dim=self.action_dim)
                    action = np.clip(
                        action + noise, -self.action_bound, self.action_bound
                    )

                    next_state, reward, done, _ = self.env.step(action)
                    self.buffer.store(state, action, (reward + 8) / 8, next_state, done)
                    bg_noise = noise
                    episode_reward += reward
                    state = next_state
                if (
                    self.buffer.size() >= args.batch_size
                    and self.buffer.size() >= args.train_start
                ):
                    self.replay_experience()
                print(f"Episode#{ep} Reward:{episode_reward}")
                tf.summary.scalar("episode_reward", episode_reward, step=ep)
```

```python colab={"base_uri": "https://localhost:8080/"} id="zhaejVDlOvM4" executionInfo={"status": "ok", "timestamp": 1638450024967, "user_tz": -330, "elapsed": 21710, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="b7658fe0-bebc-43a1-9d77-b48d3bd9be00"
if __name__ == "__main__":
    env_name = "Pendulum-v0"
    env = gym.make(env_name)
    agent = Agent(env)
    agent.train(max_episodes=2)  # Increase max_episodes value
```

<!-- #region id="ixk34zYIEgB_" -->
---
<!-- #endregion -->

```python id="VDwjGZWIE5il" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1638450138115, "user_tz": -330, "elapsed": 3381, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="e1fa19da-2ad4-4a0a-ad90-c80cb5ba9d10"
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
