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

<!-- #region id="LhETGbFeyexq" -->
# Building an RL Agent to manage social media accounts on the web
<!-- #endregion -->

```python id="fsn3Y3yObopB"
!wget -q --show-progress https://github.com/RecoHut-Projects/drl-recsys/raw/S990517/tools/webgym.zip
!unzip webgym.zip
```

```python id="Ct-TsEZqajVL"
!pip install selenium
!apt-get update # to update ubuntu to correctly run apt install
!apt install chromium-chromedriver
!cp /usr/lib/chromium-browser/chromedriver /usr/bin
```

```python id="I5zXnSjmeOJl"
import sys
sys.path.insert(0,'/usr/lib/chromium-browser/chromedriver')
```

```python id="6aCFmU0TjUIA"
import webgym
```

```python id="_ODJjGqoa5ml"
import argparse
import os
import copy
import random
from collections import deque
from datetime import datetime

import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    Lambda,
    MaxPool2D,
    concatenate,
)
```

```python id="2SsdztX_pc6j"
%load_ext tensorboard
```

```python id="lxcyEZCVdYN-"
tf.keras.backend.set_floatx("float64")
```

<!-- #region id="tnLHxGIj8iYF" -->
## Social Media Like Reply Agent
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="B8YIBwEudcT9" executionInfo={"status": "ok", "timestamp": 1638512367533, "user_tz": -330, "elapsed": 483, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="e0c35986-51af-4fea-e8b0-fb4ac4b07841"
parser = argparse.ArgumentParser(prog="TFRL-SocialMedia-Like-Reply-Agent")
parser.add_argument("--env", default="MiniWoBSocialMediaReplyVisualEnv-v0")
parser.add_argument("--update-freq", type=int, default=16)
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--actor-lr", type=float, default=1e-4)
parser.add_argument("--critic-lr", type=float, default=1e-4)
parser.add_argument("--clip-ratio", type=float, default=0.1)
parser.add_argument("--gae-lambda", type=float, default=0.95)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--logdir", default="logs")

args = parser.parse_args([])
logdir = os.path.join(
    args.logdir, parser.prog, args.env, datetime.now().strftime("%Y%m%d-%H%M%S")
)
print(f"Saving training logs to:{logdir}")
writer = tf.summary.create_file_writer(logdir)
```

```python colab={"base_uri": "https://localhost:8080/"} id="kz9QGt6Wdd6c" executionInfo={"status": "ok", "timestamp": 1638512499116, "user_tz": -330, "elapsed": 51844, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="218c6b60-3de3-4d59-b54b-179656f7226b"
class Actor:
    def __init__(self, state_dim, action_dim, action_bound, std_bound):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = np.array(action_bound)
        self.std_bound = std_bound
        self.weight_initializer = tf.keras.initializers.he_normal()
        self.eps = 1e-5
        self.model = self.nn_model()
        self.model.summary()  # Print a summary of the Actor model
        self.opt = tf.keras.optimizers.Nadam(args.actor_lr)

    def nn_model(self):
        obs_input = Input(self.state_dim, name="im_obs")
        conv1 = Conv2D(
            filters=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            input_shape=self.state_dim,
            data_format="channels_last",
            activation="relu",
        )(obs_input)
        pool1 = MaxPool2D(pool_size=(3, 3), strides=1)(conv1)
        conv2 = Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="valid",
            activation="relu",
        )(pool1)
        pool2 = MaxPool2D(pool_size=(3, 3), strides=1)(conv2)
        conv3 = Conv2D(
            filters=16,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="valid",
            activation="relu",
        )(pool2)
        pool3 = MaxPool2D(pool_size=(3, 3), strides=1)(conv3)
        conv4 = Conv2D(
            filters=8,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="valid",
            activation="relu",
        )(pool3)
        pool4 = MaxPool2D(pool_size=(3, 3), strides=1)(conv4)
        flat = Flatten()(pool4)
        dense1 = Dense(
            16, activation="relu", kernel_initializer=self.weight_initializer
        )(flat)
        dropout1 = Dropout(0.3)(dense1)
        dense2 = Dense(
            8, activation="relu", kernel_initializer=self.weight_initializer
        )(dropout1)
        dropout2 = Dropout(0.3)(dense2)
        # action_dim[0] = 2
        output_val = Dense(
            self.action_dim[0],
            activation="relu",
            kernel_initializer=self.weight_initializer,
        )(dropout2)
        # Scale & clip x[i] to be in range [0, action_bound[i]]
        action_bound = copy.deepcopy(self.action_bound)
        mu_output = Lambda(
            lambda x: tf.clip_by_value(x * action_bound, 1e-9, action_bound),
            name="mu_output",
        )(output_val)
        std_output_1 = Dense(
            self.action_dim[0],
            activation="softplus",
            kernel_initializer=self.weight_initializer,
        )(dropout2)
        std_output = Lambda(
            lambda x: tf.clip_by_value(
                x * action_bound, 1e-9, action_bound / 2, name="std_output"
            )
        )(std_output_1)
        return tf.keras.models.Model(
            inputs=obs_input, outputs=[mu_output, std_output], name="Actor"
        )

    def get_action(self, state):
        # Convert [Image] to np.array(np.adarray)
        state_np = np.array([np.array(s) for s in state])
        if len(state_np.shape) == 3:
            # Convert (w, h, c) to (1, w, h, c)
            state_np = np.expand_dims(state_np, 0)
        mu, std = self.model.predict(state_np)
        action = np.random.normal(mu[0], std[0] + self.eps, size=self.action_dim).astype(
            "int"
        )
        # Clip action to be between 0 and max obs screen size
        action = np.clip(action, 0, self.action_bound)
        # 1 Action per instance of env; Env expects: (num_instances, actions)
        action = (action,)
        log_policy = self.log_pdf(mu, std, action)
        return log_policy, action

    def log_pdf(self, mu, std, action):
        std = tf.clip_by_value(std, self.std_bound[0], self.std_bound[1])
        var = std ** 2
        log_policy_pdf = -0.5 * (action - mu) ** 2 / var - 0.5 * tf.math.log(
            var * 2 * np.pi
        )
        return tf.reduce_sum(log_policy_pdf, 1, keepdims=True)

    def compute_loss(self, log_old_policy, log_new_policy, actions, gaes):
        # Avoid INF in exp by setting 80 as the upper bound since,
        # tf.exp(x) for x>88 yeilds NaN (float32)
        ratio = tf.exp(
            tf.minimum(log_new_policy - tf.stop_gradient(log_old_policy), 80)
        )
        gaes = tf.stop_gradient(gaes)
        clipped_ratio = tf.clip_by_value(
            ratio, 1.0 - args.clip_ratio, 1.0 + args.clip_ratio
        )
        surrogate = -tf.minimum(ratio * gaes, clipped_ratio * gaes)
        return tf.reduce_mean(surrogate)

    def train(self, log_old_policy, states, actions, gaes):
        with tf.GradientTape() as tape:
            mu, std = self.model(states, training=True)
            log_new_policy = self.log_pdf(mu, std, actions)
            loss = self.compute_loss(log_old_policy, log_new_policy, actions, gaes)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

    def save(self, model_dir: str, version: int = 1):
        actor_model_save_dir = os.path.join(
            model_dir, "actor", str(version), "model.savedmodel"
        )
        self.model.save(actor_model_save_dir, save_format="tf")
        print(f"Actor model saved at:{actor_model_save_dir}")


class Critic:
    def __init__(self, state_dim):
        self.state_dim = state_dim
        self.weight_initializer = tf.keras.initializers.he_normal()
        self.model = self.nn_model()
        self.model.summary()  # Print a summary of the Critic model
        self.opt = tf.keras.optimizers.Nadam(args.critic_lr)

    def nn_model(self):
        obs_input = Input(self.state_dim)
        conv1 = Conv2D(
            filters=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            input_shape=self.state_dim,
            data_format="channels_last",
            activation="relu",
        )(obs_input)
        pool1 = MaxPool2D(pool_size=(3, 3), strides=2)(conv1)
        conv2 = Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="valid",
            activation="relu",
        )(pool1)
        pool2 = MaxPool2D(pool_size=(3, 3), strides=2)(conv2)
        conv3 = Conv2D(
            filters=16,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="valid",
            activation="relu",
        )(pool2)
        pool3 = MaxPool2D(pool_size=(3, 3), strides=1)(conv3)
        conv4 = Conv2D(
            filters=8,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="valid",
            activation="relu",
        )(pool3)
        pool4 = MaxPool2D(pool_size=(3, 3), strides=1)(conv4)
        flat = Flatten()(pool4)
        dense1 = Dense(
            16, activation="relu", kernel_initializer=self.weight_initializer
        )(flat)
        dropout1 = Dropout(0.3)(dense1)
        dense2 = Dense(
            8, activation="relu", kernel_initializer=self.weight_initializer
        )(dropout1)
        dropout2 = Dropout(0.3)(dense2)
        value = Dense(
            1, activation="linear", kernel_initializer=self.weight_initializer
        )(dropout2)

        return tf.keras.models.Model(inputs=obs_input, outputs=value, name="Critic")

    def compute_loss(self, v_pred, td_targets):
        mse = tf.keras.losses.MeanSquaredError()
        return mse(td_targets, v_pred)

    def train(self, states, td_targets):
        with tf.GradientTape() as tape:
            v_pred = self.model(states, training=True)
            # assert v_pred.shape == td_targets.shape
            loss = self.compute_loss(v_pred, tf.stop_gradient(td_targets))
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

    def save(self, model_dir: str, version: int = 1):
        critic_model_save_dir = os.path.join(
            model_dir, "critic", str(version), "model.savedmodel"
        )
        self.model.save(critic_model_save_dir, save_format="tf")
        print(f"Critic model saved at:{critic_model_save_dir}")


class PPOAgent:
    def __init__(self, env):
        self.env = env
        self.state_dim = self.env.observation_space.shape
        self.action_dim = self.env.action_space.shape
        # Set action_bounds to be within the actual task-window/browser-view of the agent
        self.action_bound = [self.env.task_width, self.env.task_height]
        self.std_bound = [1e-2, 1.0]

        self.actor = Actor(
            self.state_dim, self.action_dim, self.action_bound, self.std_bound
        )
        self.critic = Critic(self.state_dim)

    def gae_target(self, rewards, v_values, next_v_value, done):
        n_step_targets = np.zeros_like(rewards)
        gae = np.zeros_like(rewards)
        gae_cumulative = 0
        forward_val = 0

        if not done:
            forward_val = next_v_value

        for k in reversed(range(0, len(rewards))):
            delta = rewards[k] + args.gamma * forward_val - v_values[k]
            gae_cumulative = args.gamma * args.gae_lambda * gae_cumulative + delta
            gae[k] = gae_cumulative
            forward_val = v_values[k]
            n_step_targets[k] = gae[k] + v_values[k]
        return gae, n_step_targets

    def train(self, max_episodes=1000):
        with writer.as_default():
            for ep in range(max_episodes):
                state_batch = []
                action_batch = []
                reward_batch = []
                old_policy_batch = []

                episode_reward, done = 0, False

                state = self.env.reset()
                prev_state = state
                step_num = 0

                while not done:
                    # self.env.render()
                    log_old_policy, action = self.actor.get_action(state)

                    next_state, reward, dones, _ = self.env.step(action)
                    step_num += 1
                    print(
                        f"ep#:{ep} step#:{step_num} step_rew:{reward} action:{action} dones:{dones}"
                    )
                    done = np.all(dones)
                    if done:
                        next_state = prev_state
                    else:
                        prev_state = next_state
                    state = np.array([np.array(s) for s in state])
                    next_state = np.array([np.array(s) for s in next_state])
                    reward = np.reshape(reward, [1, 1])
                    log_old_policy = np.reshape(log_old_policy, [1, 1])

                    state_batch.append(state)
                    action_batch.append(action)
                    reward_batch.append((reward + 8) / 8)
                    old_policy_batch.append(log_old_policy)

                    if len(state_batch) >= args.update_freq or done:
                        states = np.array([state.squeeze() for state in state_batch])
                        # Convert ([x, y],) to [x, y]
                        actions = np.array([action[0] for action in action_batch])
                        rewards = np.array(
                            [reward.squeeze() for reward in reward_batch]
                        )
                        old_policies = np.array(
                            [old_pi.squeeze() for old_pi in old_policy_batch]
                        )

                        v_values = self.critic.model.predict(states)
                        next_v_value = self.critic.model.predict(next_state)

                        gaes, td_targets = self.gae_target(
                            rewards, v_values, next_v_value, done
                        )
                        actor_losses, critic_losses = [], []
                        for epoch in range(args.epochs):
                            actor_loss = self.actor.train(
                                old_policies, states, actions, gaes
                            )
                            actor_losses.append(actor_loss)
                            critic_loss = self.critic.train(states, td_targets)
                            critic_losses.append(critic_loss)
                        # Plot mean actor & critic losses on every update
                        tf.summary.scalar("actor_loss", np.mean(actor_losses), step=ep)
                        tf.summary.scalar(
                            "critic_loss", np.mean(critic_losses), step=ep
                        )

                        state_batch = []
                        action_batch = []
                        reward_batch = []
                        old_policy_batch = []

                    episode_reward += reward[0][0]
                    state = next_state[0]

                print(f"Episode#{ep} Reward:{episode_reward} Actions:{action_batch}")
                tf.summary.scalar("episode_reward", episode_reward, step=ep)

    def save(self, model_dir: str, version: int = 1):
        self.actor.save(model_dir, version)
        self.critic.save(model_dir, version)


if __name__ == "__main__":
    env_name = args.env
    env = gym.make(env_name)
    cta_agent = PPOAgent(env)
    cta_agent.train(max_episodes=2)
    # Model saving
    model_dir = "trained_models"
    agent_name = f"PPO_{env_name}-v0"
    agent_version = 1
    agent_model_path = os.path.join(model_dir, agent_name)
    cta_agent.save(agent_model_path, agent_version)
```

```python id="Z2Ln02kN-kyR"
%tensorboard --logdir /content/logs/TFRL-SocialMedia-Like-Reply-Agent/MiniWoBSocialMediaReplyVisualEnv-v0
```

<!-- #region id="REy4PvyI-xjo" -->
<!-- #endregion -->

<!-- #region id="LbSHkK-191Rz" -->
## Social Media Mute Agent
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="IVfl2f449c_M" executionInfo={"status": "ok", "timestamp": 1638512604578, "user_tz": -330, "elapsed": 464, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="871f329d-2829-4675-8d99-7968a82f7c2d"
parser = argparse.ArgumentParser(prog="TFRL-SocialMedia-Mute-User-Agent")
parser.add_argument("--env", default="MiniWoBSocialMediaMuteUserVisualEnv-v0")
parser.add_argument("--update-freq", type=int, default=16)
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--actor-lr", type=float, default=1e-4)
parser.add_argument("--critic-lr", type=float, default=1e-4)
parser.add_argument("--clip-ratio", type=float, default=0.1)
parser.add_argument("--gae-lambda", type=float, default=0.95)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--logdir", default="logs")

args = parser.parse_args([])
logdir = os.path.join(
    args.logdir, parser.prog, args.env, datetime.now().strftime("%Y%m%d-%H%M%S")
)
print(f"Saving training logs to:{logdir}")
writer = tf.summary.create_file_writer(logdir)
```

```python colab={"base_uri": "https://localhost:8080/"} id="UJEVcH6F9c8e" executionInfo={"status": "ok", "timestamp": 1638512681452, "user_tz": -330, "elapsed": 41629, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="484098c3-4f51-498f-b730-56e2cf7d8dd8"
class Actor:
    def __init__(self, state_dim, action_dim, action_bound, std_bound):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = np.array(action_bound)
        self.std_bound = std_bound
        self.weight_initializer = tf.keras.initializers.he_normal()
        self.eps = 1e-5
        self.model = self.nn_model()
        self.model.summary()  # Print a summary of the Actor model
        self.opt = tf.keras.optimizers.Nadam(args.actor_lr)

    def nn_model(self):
        obs_input = Input(self.state_dim, name="im_obs")
        conv1 = Conv2D(
            filters=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            input_shape=self.state_dim,
            data_format="channels_last",
            activation="relu",
        )(obs_input)
        pool1 = MaxPool2D(pool_size=(3, 3), strides=1)(conv1)
        conv2 = Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="valid",
            activation="relu",
        )(pool1)
        pool2 = MaxPool2D(pool_size=(3, 3), strides=1)(conv2)
        conv3 = Conv2D(
            filters=16,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="valid",
            activation="relu",
        )(pool2)
        pool3 = MaxPool2D(pool_size=(3, 3), strides=1)(conv3)
        conv4 = Conv2D(
            filters=8,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="valid",
            activation="relu",
        )(pool3)
        pool4 = MaxPool2D(pool_size=(3, 3), strides=1)(conv4)
        flat = Flatten()(pool4)
        dense1 = Dense(
            16, activation="relu", kernel_initializer=self.weight_initializer
        )(flat)
        dropout1 = Dropout(0.3)(dense1)
        dense2 = Dense(
            8, activation="relu", kernel_initializer=self.weight_initializer
        )(dropout1)
        dropout2 = Dropout(0.3)(dense2)
        # action_dim[0] = 2
        output_val = Dense(
            self.action_dim[0],
            activation="relu",
            kernel_initializer=self.weight_initializer,
        )(dropout2)
        # Scale & clip x[i] to be in range [0, action_bound[i]]
        action_bound = copy.deepcopy(self.action_bound)
        mu_output = Lambda(
            lambda x: tf.clip_by_value(x * action_bound, 1e-9, action_bound),
            name="mu_output",
        )(output_val)
        std_output_1 = Dense(
            self.action_dim[0],
            activation="softplus",
            kernel_initializer=self.weight_initializer,
        )(dropout2)
        std_output = Lambda(
            lambda x: tf.clip_by_value(
                x * action_bound, 1e-9, action_bound / 2, name="std_output"
            )
        )(std_output_1)
        return tf.keras.models.Model(
            inputs=obs_input, outputs=[mu_output, std_output], name="Actor"
        )

    def get_action(self, state):
        # Convert [Image] to np.array(np.adarray)
        state_np = np.array([np.array(s) for s in state])
        if len(state_np.shape) == 3:
            # Convert (w, h, c) to (1, w, h, c)
            state_np = np.expand_dims(state_np, 0)
        mu, std = self.model.predict(state_np)
        action = np.random.normal(mu[0], std[0] + self.eps, size=self.action_dim).astype(
            "int"
        )
        # Clip action to be between 0 and max obs screen size
        action = np.clip(action, 0, self.action_bound)
        # 1 Action per instance of env; Env expects: (num_instances, actions)
        action = (action,)
        log_policy = self.log_pdf(mu, std, action)
        return log_policy, action

    def log_pdf(self, mu, std, action):
        std = tf.clip_by_value(std, self.std_bound[0], self.std_bound[1])
        var = std ** 2
        log_policy_pdf = -0.5 * (action - mu) ** 2 / var - 0.5 * tf.math.log(
            var * 2 * np.pi
        )
        return tf.reduce_sum(log_policy_pdf, 1, keepdims=True)

    def compute_loss(self, log_old_policy, log_new_policy, actions, gaes):
        # Avoid INF in exp by setting 80 as the upper bound since,
        # tf.exp(x) for x>88 yeilds NaN (float32)
        ratio = tf.exp(
            tf.minimum(log_new_policy - tf.stop_gradient(log_old_policy), 80)
        )
        gaes = tf.stop_gradient(gaes)
        clipped_ratio = tf.clip_by_value(
            ratio, 1.0 - args.clip_ratio, 1.0 + args.clip_ratio
        )
        surrogate = -tf.minimum(ratio * gaes, clipped_ratio * gaes)
        return tf.reduce_mean(surrogate)

    def train(self, log_old_policy, states, actions, gaes):
        with tf.GradientTape() as tape:
            mu, std = self.model(states, training=True)
            log_new_policy = self.log_pdf(mu, std, actions)
            loss = self.compute_loss(log_old_policy, log_new_policy, actions, gaes)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

    def save(self, model_dir: str, version: int = 1):
        actor_model_save_dir = os.path.join(
            model_dir, "actor", str(version), "model.savedmodel"
        )
        self.model.save(actor_model_save_dir, save_format="tf")
        print(f"Actor model saved at:{actor_model_save_dir}")


class Critic:
    def __init__(self, state_dim):
        self.state_dim = state_dim
        self.weight_initializer = tf.keras.initializers.he_normal()
        self.model = self.nn_model()
        self.model.summary()  # Print a summary of the Critic model
        self.opt = tf.keras.optimizers.Nadam(args.critic_lr)

    def nn_model(self):
        obs_input = Input(self.state_dim)
        conv1 = Conv2D(
            filters=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            input_shape=self.state_dim,
            data_format="channels_last",
            activation="relu",
        )(obs_input)
        pool1 = MaxPool2D(pool_size=(3, 3), strides=2)(conv1)
        conv2 = Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="valid",
            activation="relu",
        )(pool1)
        pool2 = MaxPool2D(pool_size=(3, 3), strides=2)(conv2)
        conv3 = Conv2D(
            filters=16,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="valid",
            activation="relu",
        )(pool2)
        pool3 = MaxPool2D(pool_size=(3, 3), strides=1)(conv3)
        conv4 = Conv2D(
            filters=8,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="valid",
            activation="relu",
        )(pool3)
        pool4 = MaxPool2D(pool_size=(3, 3), strides=1)(conv4)
        flat = Flatten()(pool4)
        dense1 = Dense(
            16, activation="relu", kernel_initializer=self.weight_initializer
        )(flat)
        dropout1 = Dropout(0.3)(dense1)
        dense2 = Dense(
            8, activation="relu", kernel_initializer=self.weight_initializer
        )(dropout1)
        dropout2 = Dropout(0.3)(dense2)
        value = Dense(
            1, activation="linear", kernel_initializer=self.weight_initializer
        )(dropout2)

        return tf.keras.models.Model(inputs=obs_input, outputs=value, name="Critic")

    def compute_loss(self, v_pred, td_targets):
        mse = tf.keras.losses.MeanSquaredError()
        return mse(td_targets, v_pred)

    def train(self, states, td_targets):
        with tf.GradientTape() as tape:
            v_pred = self.model(states, training=True)
            # assert v_pred.shape == td_targets.shape
            loss = self.compute_loss(v_pred, tf.stop_gradient(td_targets))
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

    def save(self, model_dir: str, version: int = 1):
        critic_model_save_dir = os.path.join(
            model_dir, "critic", str(version), "model.savedmodel"
        )
        self.model.save(critic_model_save_dir, save_format="tf")
        print(f"Critic model saved at:{critic_model_save_dir}")


class PPOAgent:
    def __init__(self, env):
        self.env = env
        self.state_dim = self.env.observation_space.shape
        self.action_dim = self.env.action_space.shape
        # Set action_bounds to be within the actual task-window/browser-view of the agent
        self.action_bound = [self.env.task_width, self.env.task_height]
        self.std_bound = [1e-2, 1.0]

        self.actor = Actor(
            self.state_dim, self.action_dim, self.action_bound, self.std_bound
        )
        self.critic = Critic(self.state_dim)

    def gae_target(self, rewards, v_values, next_v_value, done):
        n_step_targets = np.zeros_like(rewards)
        gae = np.zeros_like(rewards)
        gae_cumulative = 0
        forward_val = 0

        if not done:
            forward_val = next_v_value

        for k in reversed(range(0, len(rewards))):
            delta = rewards[k] + args.gamma * forward_val - v_values[k]
            gae_cumulative = args.gamma * args.gae_lambda * gae_cumulative + delta
            gae[k] = gae_cumulative
            forward_val = v_values[k]
            n_step_targets[k] = gae[k] + v_values[k]
        return gae, n_step_targets

    def train(self, max_episodes=1000):
        with writer.as_default():
            for ep in range(max_episodes):
                state_batch = []
                action_batch = []
                reward_batch = []
                old_policy_batch = []

                episode_reward, done = 0, False

                state = self.env.reset()
                prev_state = state
                step_num = 0

                while not done:
                    # self.env.render()
                    log_old_policy, action = self.actor.get_action(state)

                    next_state, reward, dones, _ = self.env.step(action)
                    step_num += 1
                    print(
                        f"ep#:{ep} step#:{step_num} step_rew:{reward} action:{action} dones:{dones}"
                    )
                    done = np.all(dones)
                    if done:
                        next_state = prev_state
                    else:
                        prev_state = next_state
                    state = np.array([np.array(s) for s in state])
                    next_state = np.array([np.array(s) for s in next_state])
                    reward = np.reshape(reward, [1, 1])
                    log_old_policy = np.reshape(log_old_policy, [1, 1])

                    state_batch.append(state)
                    action_batch.append(action)
                    reward_batch.append((reward + 8) / 8)
                    old_policy_batch.append(log_old_policy)

                    if len(state_batch) >= args.update_freq or done:
                        states = np.array([state.squeeze() for state in state_batch])
                        # Convert ([x, y],) to [x, y]
                        actions = np.array([action[0] for action in action_batch])
                        rewards = np.array(
                            [reward.squeeze() for reward in reward_batch]
                        )
                        old_policies = np.array(
                            [old_pi.squeeze() for old_pi in old_policy_batch]
                        )

                        v_values = self.critic.model.predict(states)
                        next_v_value = self.critic.model.predict(next_state)

                        gaes, td_targets = self.gae_target(
                            rewards, v_values, next_v_value, done
                        )
                        actor_losses, critic_losses = [], []
                        for epoch in range(args.epochs):
                            actor_loss = self.actor.train(
                                old_policies, states, actions, gaes
                            )
                            actor_losses.append(actor_loss)
                            critic_loss = self.critic.train(states, td_targets)
                            critic_losses.append(critic_loss)
                        # Plot mean actor & critic losses on every update
                        tf.summary.scalar("actor_loss", np.mean(actor_losses), step=ep)
                        tf.summary.scalar(
                            "critic_loss", np.mean(critic_losses), step=ep
                        )

                        state_batch = []
                        action_batch = []
                        reward_batch = []
                        old_policy_batch = []

                    episode_reward += reward[0][0]
                    state = next_state[0]

                print(f"Episode#{ep} Reward:{episode_reward} Actions:{action_batch}")
                tf.summary.scalar("episode_reward", episode_reward, step=ep)

    def save(self, model_dir: str, version: int = 1):
        self.actor.save(model_dir, version)
        self.critic.save(model_dir, version)


if __name__ == "__main__":
    env_name = args.env
    env = gym.make(env_name)
    cta_agent = PPOAgent(env)
    cta_agent.train(max_episodes=2)
    # Model saving
    model_dir = "trained_models"
    agent_name = f"PPO_{env_name}"
    agent_version = 1
    agent_model_path = os.path.join(model_dir, agent_name)
    cta_agent.save(agent_model_path, agent_version)
```

```python id="D_2THOMX-2E7"
%tensorboard --logdir /content/logs/TFRL-SocialMedia-Mute-User-Agent/MiniWoBSocialMediaMuteUserVisualEnv-v0
```

<!-- #region id="tNF2n2F5_Hsy" -->
<!-- #endregion -->

<!-- #region id="VlOVmEIW95ae" -->
## Social Media Mute Agent DDPG
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="c42WjLsm-AZL" executionInfo={"status": "ok", "timestamp": 1638512751595, "user_tz": -330, "elapsed": 495, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="04a53a2c-5992-4d07-e720-3af8650c63cf"
parser = argparse.ArgumentParser(
    prog="TFRL-SocialMedia-Mute-User-DDPGAgent"
)
parser.add_argument("--env", default="MiniWoBSocialMediaMuteUserVisualEnv-v0")
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

```python colab={"base_uri": "https://localhost:8080/"} id="mBdf2ZsM-AWs" executionInfo={"status": "ok", "timestamp": 1638512832169, "user_tz": -330, "elapsed": 33164, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="e35c00bb-270e-481c-e073-fbad2c65b974"
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


class Actor:
    def __init__(self, state_dim, action_dim, action_bound):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.weight_initializer = tf.keras.initializers.he_normal()
        self.eps = 1e-5
        self.model = self.nn_model()
        self.opt = tf.keras.optimizers.Adam(args.actor_lr)

    def nn_model(self):
        obs_input = Input(self.state_dim)
        conv1 = Conv2D(
            filters=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            input_shape=self.state_dim,
            data_format="channels_last",
            activation="relu",
        )(obs_input)
        pool1 = MaxPool2D(pool_size=(3, 3), strides=1)(conv1)
        conv2 = Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="valid",
            activation="relu",
        )(pool1)
        pool2 = MaxPool2D(pool_size=(3, 3), strides=1)(conv2)
        conv3 = Conv2D(
            filters=16,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="valid",
            activation="relu",
        )(pool2)
        pool3 = MaxPool2D(pool_size=(3, 3), strides=1)(conv3)
        conv4 = Conv2D(
            filters=8,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="valid",
            activation="relu",
        )(pool3)
        pool4 = MaxPool2D(pool_size=(3, 3), strides=1)(conv4)
        flat = Flatten()(pool4)
        dense1 = Dense(
            16, activation="relu", kernel_initializer=self.weight_initializer
        )(flat)
        dropout1 = Dropout(0.3)(dense1)
        dense2 = Dense(
            8, activation="relu", kernel_initializer=self.weight_initializer
        )(dropout1)
        dropout2 = Dropout(0.3)(dense2)
        # action_dim[0] = 2
        output_val = Dense(
            self.action_dim[0],
            activation="relu",
            kernel_initializer=self.weight_initializer,
        )(dropout2)
        # Scale & clip x[i] to be in range [0, action_bound[i]]
        mu_output = Lambda(
            lambda x: tf.clip_by_value(x * self.action_bound, 1e-9, self.action_bound)
        )(output_val)
        return tf.keras.models.Model(inputs=obs_input, outputs=mu_output, name="Actor")

    def train(self, states, q_grads):
        with tf.GradientTape() as tape:
            grads = tape.gradient(
                self.model(states), self.model.trainable_variables, -q_grads
            )
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))

    def predict(self, state):
        return self.model.predict(state)

    def get_action(self, state):
        # Convert [Image] to np.array(np.adarray)
        state_np = np.array([np.array(s) for s in state])
        if len(state_np.shape) == 3:
            # Convert (w, h, c) to (1, w, h, c)
            state_np = np.expand_dims(state_np, 0)
        action = self.model.predict(state_np)
        # Clip action to be between 0 and max obs screen size
        action = np.clip(action, 0, self.action_bound)
        # 1 Action per instance of env; Env expects: (num_instances, actions)
        return action


class Critic:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.weight_initializer = tf.keras.initializers.he_normal()
        self.model = self.nn_model()
        self.opt = tf.keras.optimizers.Adam(args.critic_lr)

    def nn_model(self):
        obs_input = Input(self.state_dim)
        conv1 = Conv2D(
            filters=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            input_shape=self.state_dim,
            data_format="channels_last",
            activation="relu",
        )(obs_input)
        pool1 = MaxPool2D(pool_size=(3, 3), strides=2)(conv1)
        conv2 = Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="valid",
            activation="relu",
        )(pool1)
        pool2 = MaxPool2D(pool_size=(3, 3), strides=2)(conv2)
        conv3 = Conv2D(
            filters=16,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="valid",
            activation="relu",
        )(pool2)
        pool3 = MaxPool2D(pool_size=(3, 3), strides=1)(conv3)
        conv4 = Conv2D(
            filters=8,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="valid",
            activation="relu",
        )(pool3)
        pool4 = MaxPool2D(pool_size=(3, 3), strides=1)(conv4)
        flat = Flatten()(pool4)
        dense1 = Dense(
            16, activation="relu", kernel_initializer=self.weight_initializer
        )(flat)
        dropout1 = Dropout(0.3)(dense1)
        dense2 = Dense(
            8, activation="relu", kernel_initializer=self.weight_initializer
        )(dropout1)
        dropout2 = Dropout(0.3)(dense2)
        value = Dense(
            1, activation="linear", kernel_initializer=self.weight_initializer
        )(dropout2)

        return tf.keras.models.Model(inputs=obs_input, outputs=value, name="Critic")

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


class DDPGAgent:
    def __init__(self, env):
        self.env = env
        self.state_dim = self.env.observation_space.shape
        self.action_dim = self.env.action_space.shape
        self.action_bound = self.env.action_space.high

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
                targets[i] = args.gamma * q_values[i]
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
                step_num, episode_reward, done = 0, 0, False

                state = self.env.reset()
                prev_state = state
                bg_noise = np.random.randint(
                    self.env.action_space.low,
                    self.env.action_space.high,
                    self.env.action_space.shape,
                )
                while not done:
                    # self.env.render()
                    action = self.actor.get_action(state)
                    noise = self.add_ou_noise(bg_noise, dim=self.action_dim)
                    action = np.clip(action + noise, 0, self.action_bound).astype("int")

                    next_state, reward, dones, _ = self.env.step(action)
                    done = np.all(dones)
                    if done:
                        next_state = prev_state
                    else:
                        prev_state = next_state

                    for (s, a, r, s_n, d) in zip(
                        next_state, action, reward, next_state, dones
                    ):
                        self.buffer.store(s, a, (r + 8) / 8, s_n, d)
                        episode_reward += r

                    step_num += 1  # 1 across num_instances
                    print(
                        f"ep#:{ep} step#:{step_num} step_rew:{reward} action:{action} dones:{dones}"
                    )

                    bg_noise = noise
                    state = next_state
                if (
                    self.buffer.size() >= args.batch_size
                    and self.buffer.size() >= args.train_start
                ):
                    self.replay_experience()
                print(f"Episode#{ep} Reward:{episode_reward}")
                tf.summary.scalar("episode_reward", episode_reward, step=ep)


if __name__ == "__main__":
    env_name = "MiniWoBSocialMediaMuteUserVisualEnv-v0"
    env = gym.make(env_name)
    agent = DDPGAgent(env)
    agent.train(max_episodes=2)
```

```python id="rx8UzI52dmn4"
%tensorboard --logdir /content/logs/TFRL-SocialMedia-Mute-User-DDPGAgent/MiniWoBSocialMediaMuteUserVisualEnv-v0
```

<!-- #region id="E2ITqp9G_hAN" -->
<!-- #endregion -->

<!-- #region id="y_ucjqQ9tuZk" -->
---
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="bW0aBsS5tuZs" executionInfo={"status": "ok", "timestamp": 1638513146554, "user_tz": -330, "elapsed": 4127, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="80ec06f9-f3be-4809-bd59-1544c47260fe"
!pip install -q watermark
%reload_ext watermark
%watermark -a "Sparsh A." -m -iv -u -t -d -p selenium
```

<!-- #region id="ilEAuT87tuZt" -->
---
<!-- #endregion -->

<!-- #region id="QXts9r8TtuZv" -->
**END**
<!-- #endregion -->
