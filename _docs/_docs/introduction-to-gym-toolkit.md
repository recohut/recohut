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

<!-- #region id="quyEfkZKOv4K" -->
# Introduction to Gym toolkit
<!-- #endregion -->

<!-- #region id="pW8uhiWfFHXA" -->
## Gym Environments

The centerpiece of Gym is the environment, which defines the "game" in which your reinforcement algorithm will compete.  An environment does not need to be a game; however, it describes the following game-like features:
* **action space**: What actions can we take on the environment, at each step/episode, to alter the environment.
* **observation space**: What is the current state of the portion of the environment that we can observe. Usually, we can see the entire environment.

Before we begin to look at Gym, it is essential to understand some of the terminology used by this library.

* **Agent** - The machine learning program or model that controls the actions.
Step - One round of issuing actions that affect the observation space.
* **Episode** - A collection of steps that terminates when the agent fails to meet the environment's objective, or the episode reaches the maximum number of allowed steps.
* **Render** - Gym can render one frame for display after each episode.
* **Reward** - A positive reinforcement that can occur at the end of each episode, after the agent acts.
* **Nondeterministic** - For some environments, randomness is a factor in deciding what effects actions have on reward and changes to the observation space.
<!-- #endregion -->

```python id="-6ZCHFDcE88U"
import gym

def query_environment(name):
  env = gym.make(name)
  spec = gym.spec(name)
  print(f"Action Space: {env.action_space}")
  print(f"Observation Space: {env.observation_space}")
  print(f"Max Episode Steps: {spec.max_episode_steps}")
  print(f"Nondeterministic: {spec.nondeterministic}")
  print(f"Reward Range: {env.reward_range}")
  print(f"Reward Threshold: {spec.reward_threshold}")
```

```python colab={"base_uri": "https://localhost:8080/"} id="vywBhM-1E62J" executionInfo={"status": "ok", "timestamp": 1634639010392, "user_tz": -330, "elapsed": 18, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="cb438bd0-8251-403c-8ed4-e6376ace4ed3"
query_environment("CartPole-v1")
```

<!-- #region id="0PJNRCLOE9Ys" -->
The CartPole-v1 environment challenges the agent to move a cart while keeping a pole balanced. The environment has an observation space of 4 continuous numbers:

* Cart Position
* Cart Velocity
* Pole Angle
* Pole Velocity At Tip

To achieve this goal, the agent can take the following actions:

* Push cart to the left
* Push cart to the right

There is also a continuous variant of the mountain car.  This version does not simply have the motor on or off.  For the continuous car the action space is a single floating point number that specifies how much forward or backward force is being applied.
<!-- #endregion -->

<!-- #region id="uSPhz7b7Ozb3" -->
### Simple
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="MIIn2Yl6H-VI" executionInfo={"status": "ok", "timestamp": 1634473799356, "user_tz": -330, "elapsed": 14, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="0092a417-7e35-4361-d56c-f7053c5932d1"
import random
from typing import List


class Environment:
    def __init__(self):
        self.steps_left = 10

    def get_observation(self) -> List[float]:
        return [0.0, 0.0, 0.0]

    def get_actions(self) -> List[int]:
        return [0, 1]

    def is_done(self) -> bool:
        return self.steps_left == 0

    def action(self, action: int) -> float:
        if self.is_done():
            raise Exception("Game is over")
        self.steps_left -= 1
        return random.random()


class Agent:
    def __init__(self):
        self.total_reward = 0.0

    def step(self, env: Environment):
        current_obs = env.get_observation()
        actions = env.get_actions()
        reward = env.action(random.choice(actions))
        self.total_reward += reward


if __name__ == "__main__":
    env = Environment()
    agent = Agent()

    while not env.is_done():
        agent.step(env)

    print("Total reward got: %.4f" % agent.total_reward)
```

<!-- #region id="k_tY2WPsT1c-" -->
### Frozenlake
<!-- #endregion -->

```python id="EeiluUpy54yA"
import gym
```

```python id="LA_fVk6255_n"
env = gym.make("FrozenLake-v0")
```

```python id="lF1ICJzX6D0l" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1605276035934, "user_tz": -330, "elapsed": 1204, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="1edbe302-fd72-4991-8ff2-7cd752ca6eb1"
env.render()
```

```python id="dtAEu_Gn6GZs" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1605276072781, "user_tz": -330, "elapsed": 19497, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="33b98be2-3871-4f92-b3cf-b0da723c2934"
print(env.observation_space)
```

```python id="q-9DEnOp6K7f" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1605276124058, "user_tz": -330, "elapsed": 2596, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="ff778f10-8f99-40b3-fc94-411c2ef66467"
print(env.action_space)
```

<!-- #region id="cMtxi6Kk8Frp" -->
| Number | Action |
| ------ | ------ |
| 0 | Left |
| 1 | Down |
| 2 | Right |
| 3 | Up |
<!-- #endregion -->

<!-- #region id="Uq2b1f5T77Fr" -->
We can obtain the transition probability and the reward function by just typing env.P[state][action]. So, to obtain the transition probability of moving from state S to the other states by performing the action right, we can type env.P[S][right]. But we cannot just type state S and action right directly since they are encoded as numbers. We learned that state S is encoded as 0 and the action right is encoded as 2, so, to obtain the transition probability of state S by performing the action right, we type env.P[0][2]
<!-- #endregion -->

```python id="21WRCUki6bkS" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1605276520994, "user_tz": -330, "elapsed": 1215, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="090db85d-44fe-4b14-ab35-3947402359da"
print(env.P[0][2])
```

<!-- #region id="LrW8_f0e8QHN" -->
Our output is in the form of [(transition probability, next state, reward, Is terminal state?)]
<!-- #endregion -->

```python id="vM0sU4ZF-pE6"
state = env.reset()
```

```python id="uwzVw4IH-mtV" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1605277218076, "user_tz": -330, "elapsed": 1512, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="568f34cc-0d10-4458-a4ce-0d58d4f19990"
env.step(1)
```

```python id="xafLZTl1-tV-"
(next_state, reward, done, info) = env.step(1)
```

<!-- #region id="lxEKBTBf-4F7" -->
- **next_state** represents the next state.
- **reward** represents the obtained reward.
- **done** implies whether our episode has ended. That is, if the next state is a terminal state, then our episode will end, so done will be marked as True else it will be marked as False.
- **info** — Apart from the transition probability, in some cases, we also obtain other information saved as info, which is used for debugging purposes.
<!-- #endregion -->

```python id="tQ3-ZKqN780l"
random_action = env.action_space.sample()
```

```python id="-1kCj-Ww-YJn"
next_state, reward, done, info = env.step(random_action)
```

<!-- #region id="CLpIVnCa92RO" -->
**Generating an episode**
The episode is the agent environment interaction starting from the initial state to the terminal state. The agent interacts with the environment by performing some action in each state. An episode ends if the agent reaches the terminal state. So, in the Frozen Lake environment, the episode will end if the agent reaches the terminal state, which is either the hole state (H) or goal state (G).
<!-- #endregion -->

```python id="tZJsHfU9_QFI" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1605277535095, "user_tz": -330, "elapsed": 1377, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="0e7acc31-d61d-4408-d5b4-a94dc52d23ee"
import gym

env = gym.make("FrozenLake-v0")
state = env.reset()
print('Time Step 0 :')
env.render()
num_timesteps = 20

for t in range(num_timesteps):
  random_action = env.action_space.sample()
  new_state, reward, done, info = env.step(random_action)
  print ('Time Step {} :'.format(t+1))
  env.render()
  if done:
    break
```

<!-- #region id="hoa9ldOGAX6x" -->
Instead of generating one episode, we can also generate a series of episodes by taking some random action in each state
<!-- #endregion -->

```python id="3JgXPnMY_0XX"
import gym
env = gym.make("FrozenLake-v0")
num_episodes = 10
num_timesteps = 20 
for i in range(num_episodes):
    
    state = env.reset()
    print('Time Step 0 :')
    env.render()
    
    for t in range(num_timesteps):
        random_action = env.action_space.sample()
        new_state, reward, done, info = env.step(random_action)
        print ('Time Step {} :'.format(t+1))
        env.render()
        if done:
            break
```

<!-- #region id="h6mvI5dVBQnH" -->
### Cartpole
<!-- #endregion -->

```python id="U7-cNwhyBhhO"
env = gym.make("CartPole-v0")
```

```python id="GFGXLX4GBecG" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1605278280837, "user_tz": -330, "elapsed": 1889, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="93ba1033-bdcc-42f8-ab51-d51b0950cbda"
print(env.observation_space)
```

<!-- #region id="wkwE3siLCuhW" -->
Note that all of these values are continuous, that is:

- The value of the cart position ranges from -4.8 to 4.8.
- The value of the cart velocity ranges from -Inf to Inf (  to  ).
- The value of the pole angle ranges from -0.418 radians to 0.418 radians.
- The value of the pole velocity at the tip ranges from -Inf to Inf.
<!-- #endregion -->

```python id="63pYvqKjCqRi" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1605278365834, "user_tz": -330, "elapsed": 2004, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="3b491354-0eb3-4978-ceea-cdb5600b3831"
print(env.reset())
```

```python id="lDWOJAJkC_A7" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1605278393363, "user_tz": -330, "elapsed": 1463, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="5f2e7745-960b-40dc-8108-66b13008f262"
print(env.observation_space.high)
```

<!-- #region id="c03Pn4jeDIE7" -->
It implies that:

1. The maximum value of the cart position is 4.8.
2. We learned that the maximum value of the cart velocity is +Inf, and we know that infinity is not really a number, so it is represented using the largest positive real value 3.4028235e+38.
3. The maximum value of the pole angle is 0.418 radians.
4. The maximum value of the pole velocity at the tip is +Inf, so it is represented using the largest positive real value 3.4028235e+38.
<!-- #endregion -->

```python id="ho7n-F9VDF3k" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1605278470142, "user_tz": -330, "elapsed": 1457, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="aba8f1da-d706-4a97-8780-2652c0dc8785"
print(env.observation_space.low)
```

<!-- #region id="O6R0sFsNDabE" -->
It states that:

1. The minimum value of the cart position is -4.8.
2. We learned that the minimum value of the cart velocity is -Inf, and we know that infinity is not really a number, so it is represented using the largest negative real value -3.4028235e+38.
3. The minimum value of the pole angle is -0.418 radians.
4. The minimum value of the pole velocity at the tip is -Inf, so it is represented using the largest negative real value -3.4028235e+38.
<!-- #endregion -->

```python id="2VaPg1BCDYmk" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1605278495066, "user_tz": -330, "elapsed": 1365, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="4b8b1e9f-2492-46b5-c30a-43d90ad17409"
print(env.action_space)
```

<!-- #region id="4qorqBHp_nyt" -->
| Number | Action |
| ------ | ------ |
| 0 | Push cart to the left |
| 1 | Push cart to the right |
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="Ho7sieqjO3nS" executionInfo={"status": "ok", "timestamp": 1634473842164, "user_tz": -330, "elapsed": 2342, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="90a473ad-1e87-4a21-ee26-42606cb4b0c3"
import gym


if __name__ == "__main__":
    env = gym.make("CartPole-v0")

    total_reward = 0.0
    total_steps = 0
    obs = env.reset()

    while True:
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        total_steps += 1
        if done:
            break

    print("Episode done in %d steps, total reward %.2f" % (
        total_steps, total_reward))
```

<!-- #region id="lgcMRfz_PiJl" -->
## Wrappers

Very frequently, you will want to extend the environment's functionality in some generic way. For example, imagine an environment gives you some observations, but you want to accumulate them in some buffer and provide to the agent the N last observations. This is a common scenario for dynamic computer games, when one single frame is just not enough to get the full information about the game state. Another example is when you want to be able to crop or preprocess an image's pixels to make it more convenient for the agent to digest, or if you want to normalize reward scores somehow. There are many such situations that have the same structure – you want to "wrap" the existing environment and add some extra logic for doing something. Gym provides a convenient framework for these situations – the Wrapper class.
<!-- #endregion -->

<!-- #region id="RCoATNEwO4YG" -->
**Random action wrapper**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="A-dSd9SfPBVE" executionInfo={"status": "ok", "timestamp": 1634473882660, "user_tz": -330, "elapsed": 476, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="0a56a5ce-de30-4c30-872f-5f2b7592760c"
import gym
from typing import TypeVar
import random

Action = TypeVar('Action')


class RandomActionWrapper(gym.ActionWrapper):
    def __init__(self, env, epsilon=0.1):
        super(RandomActionWrapper, self).__init__(env)
        self.epsilon = epsilon

    def action(self, action: Action) -> Action:
        if random.random() < self.epsilon:
            print("Random!")
            return self.env.action_space.sample()
        return action


if __name__ == "__main__":
    env = RandomActionWrapper(gym.make("CartPole-v0"))

    obs = env.reset()
    total_reward = 0.0

    while True:
        obs, reward, done, _ = env.step(0)
        total_reward += reward
        if done:
            break

    print("Reward got: %.2f" % total_reward)
```

<!-- #region id="0YVaFx3SPL7U" -->
## Atari GAN
<!-- #endregion -->

```python id="zLbSDkjdQ1jJ"
! wget http://www.atarimania.com/roms/Roms.rar
! mkdir /content/ROM/
! unrar e /content/Roms.rar /content/ROM/
! python -m atari_py.import_roms /content/ROM/
```

<!-- #region id="Iilm1SFKRzJe" -->
### Normal
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="3Zk9YQmEQUPM" outputId="aeb3538d-5093-4e49-ca52-3e2646233b16"
import random
import argparse
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import torchvision.utils as vutils

import gym
import gym.spaces

import numpy as np

log = gym.logger
log.set_level(gym.logger.INFO)

LATENT_VECTOR_SIZE = 100
DISCR_FILTERS = 64
GENER_FILTERS = 64
BATCH_SIZE = 16

# dimension input image will be rescaled
IMAGE_SIZE = 64

LEARNING_RATE = 0.0001
REPORT_EVERY_ITER = 100
SAVE_IMAGE_EVERY_ITER = 1000


class InputWrapper(gym.ObservationWrapper):
    """
    Preprocessing of input numpy array:
    1. resize image into predefined size
    2. move color channel axis to a first place
    """
    def __init__(self, *args):
        super(InputWrapper, self).__init__(*args)
        assert isinstance(self.observation_space, gym.spaces.Box)
        old_space = self.observation_space
        self.observation_space = gym.spaces.Box(
            self.observation(old_space.low),
            self.observation(old_space.high),
            dtype=np.float32)

    def observation(self, observation):
        # resize image
        new_obs = cv2.resize(
            observation, (IMAGE_SIZE, IMAGE_SIZE))
        # transform (210, 160, 3) -> (3, 210, 160)
        new_obs = np.moveaxis(new_obs, 2, 0)
        return new_obs.astype(np.float32)


class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        # this pipe converges image into the single number
        self.conv_pipe = nn.Sequential(
            nn.Conv2d(in_channels=input_shape[0], out_channels=DISCR_FILTERS,
                      kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=DISCR_FILTERS, out_channels=DISCR_FILTERS*2,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(DISCR_FILTERS*2),
            nn.ReLU(),
            nn.Conv2d(in_channels=DISCR_FILTERS * 2, out_channels=DISCR_FILTERS * 4,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(DISCR_FILTERS * 4),
            nn.ReLU(),
            nn.Conv2d(in_channels=DISCR_FILTERS * 4, out_channels=DISCR_FILTERS * 8,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(DISCR_FILTERS * 8),
            nn.ReLU(),
            nn.Conv2d(in_channels=DISCR_FILTERS * 8, out_channels=1,
                      kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        conv_out = self.conv_pipe(x)
        return conv_out.view(-1, 1).squeeze(dim=1)


class Generator(nn.Module):
    def __init__(self, output_shape):
        super(Generator, self).__init__()
        # pipe deconvolves input vector into (3, 64, 64) image
        self.pipe = nn.Sequential(
            nn.ConvTranspose2d(in_channels=LATENT_VECTOR_SIZE, out_channels=GENER_FILTERS * 8,
                               kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(GENER_FILTERS * 8),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=GENER_FILTERS * 8, out_channels=GENER_FILTERS * 4,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(GENER_FILTERS * 4),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=GENER_FILTERS * 4, out_channels=GENER_FILTERS * 2,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(GENER_FILTERS * 2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=GENER_FILTERS * 2, out_channels=GENER_FILTERS,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(GENER_FILTERS),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=GENER_FILTERS, out_channels=output_shape[0],
                               kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.pipe(x)


def iterate_batches(envs, batch_size=BATCH_SIZE):
    batch = [e.reset() for e in envs]
    env_gen = iter(lambda: random.choice(envs), None)

    while True:
        e = next(env_gen)
        obs, reward, is_done, _ = e.step(e.action_space.sample())
        if np.mean(obs) > 0.01:
            batch.append(obs)
        if len(batch) == batch_size:
            # Normalising input between -1 to 1
            batch_np = np.array(batch, dtype=np.float32) * 2.0 / 255.0 - 1.0
            yield torch.tensor(batch_np)
            batch.clear()
        if is_done:
            e.reset()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cuda", default=False, action='store_true',
        help="Enable cuda computation")
    args = parser.parse_args(args={})

    device = torch.device("cuda" if args.cuda else "cpu")
    envs = [
        InputWrapper(gym.make(name))
        for name in ('Breakout-v0', 'AirRaid-v0', 'Pong-v0')
    ]
    input_shape = envs[0].observation_space.shape

    net_discr = Discriminator(input_shape=input_shape).to(device)
    net_gener = Generator(output_shape=input_shape).to(device)

    objective = nn.BCELoss()
    gen_optimizer = optim.Adam(
        params=net_gener.parameters(), lr=LEARNING_RATE,
        betas=(0.5, 0.999))
    dis_optimizer = optim.Adam(
        params=net_discr.parameters(), lr=LEARNING_RATE,
        betas=(0.5, 0.999))
    writer = SummaryWriter()

    gen_losses = []
    dis_losses = []
    iter_no = 0

    true_labels_v = torch.ones(BATCH_SIZE, device=device)
    fake_labels_v = torch.zeros(BATCH_SIZE, device=device)

    for batch_v in iterate_batches(envs):
        # fake samples, input is 4D: batch, filters, x, y
        gen_input_v = torch.FloatTensor(
            BATCH_SIZE, LATENT_VECTOR_SIZE, 1, 1)
        gen_input_v.normal_(0, 1)
        gen_input_v = gen_input_v.to(device)
        batch_v = batch_v.to(device)
        gen_output_v = net_gener(gen_input_v)

        # train discriminator
        dis_optimizer.zero_grad()
        dis_output_true_v = net_discr(batch_v)
        dis_output_fake_v = net_discr(gen_output_v.detach())
        dis_loss = objective(dis_output_true_v, true_labels_v) + \
                   objective(dis_output_fake_v, fake_labels_v)
        dis_loss.backward()
        dis_optimizer.step()
        dis_losses.append(dis_loss.item())

        # train generator
        gen_optimizer.zero_grad()
        dis_output_v = net_discr(gen_output_v)
        gen_loss_v = objective(dis_output_v, true_labels_v)
        gen_loss_v.backward()
        gen_optimizer.step()
        gen_losses.append(gen_loss_v.item())

        iter_no += 1
        if iter_no % REPORT_EVERY_ITER == 0:
            log.info("Iter %d: gen_loss=%.3e, dis_loss=%.3e",
                     iter_no, np.mean(gen_losses),
                     np.mean(dis_losses))
            writer.add_scalar(
                "gen_loss", np.mean(gen_losses), iter_no)
            writer.add_scalar(
                "dis_loss", np.mean(dis_losses), iter_no)
            gen_losses = []
            dis_losses = []
        if iter_no % SAVE_IMAGE_EVERY_ITER == 0:
            writer.add_image("fake", vutils.make_grid(
                gen_output_v.data[:64], normalize=True), iter_no)
            writer.add_image("real", vutils.make_grid(
                batch_v.data[:64], normalize=True), iter_no)
```

<!-- #region id="OgEYt26iQWDL" -->
### Ignite
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="EY1DrTxOSO_2" outputId="f9fc6922-9d4b-49ab-9979-f0315d2dc64a"
import random
import argparse
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
from ignite.engine import Engine, Events
from ignite.metrics import RunningAverage
from ignite.contrib.handlers import tensorboard_logger as tb_logger

import torchvision.utils as vutils

import gym
import gym.spaces

import numpy as np

log = gym.logger
log.set_level(gym.logger.INFO)

LATENT_VECTOR_SIZE = 100
DISCR_FILTERS = 64
GENER_FILTERS = 64
BATCH_SIZE = 16

# dimension input image will be rescaled
IMAGE_SIZE = 64

LEARNING_RATE = 0.0001
REPORT_EVERY_ITER = 100
SAVE_IMAGE_EVERY_ITER = 1000


class InputWrapper(gym.ObservationWrapper):
    """
    Preprocessing of input numpy array:
    1. resize image into predefined size
    2. move color channel axis to a first place
    """
    def __init__(self, *args):
        super(InputWrapper, self).__init__(*args)
        assert isinstance(self.observation_space, gym.spaces.Box)
        old_space = self.observation_space
        self.observation_space = gym.spaces.Box(self.observation(old_space.low), self.observation(old_space.high),
                                                dtype=np.float32)

    def observation(self, observation):
        # resize image
        new_obs = cv2.resize(observation, (IMAGE_SIZE, IMAGE_SIZE))
        # transform (210, 160, 3) -> (3, 210, 160)
        new_obs = np.moveaxis(new_obs, 2, 0)
        return new_obs.astype(np.float32)


class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        # this pipe converges image into the single number
        self.conv_pipe = nn.Sequential(
            nn.Conv2d(in_channels=input_shape[0], out_channels=DISCR_FILTERS,
                      kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=DISCR_FILTERS, out_channels=DISCR_FILTERS*2,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(DISCR_FILTERS*2),
            nn.ReLU(),
            nn.Conv2d(in_channels=DISCR_FILTERS * 2, out_channels=DISCR_FILTERS * 4,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(DISCR_FILTERS * 4),
            nn.ReLU(),
            nn.Conv2d(in_channels=DISCR_FILTERS * 4, out_channels=DISCR_FILTERS * 8,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(DISCR_FILTERS * 8),
            nn.ReLU(),
            nn.Conv2d(in_channels=DISCR_FILTERS * 8, out_channels=1,
                      kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        conv_out = self.conv_pipe(x)
        return conv_out.view(-1, 1).squeeze(dim=1)


class Generator(nn.Module):
    def __init__(self, output_shape):
        super(Generator, self).__init__()
        # pipe deconvolves input vector into (3, 64, 64) image
        self.pipe = nn.Sequential(
            nn.ConvTranspose2d(in_channels=LATENT_VECTOR_SIZE, out_channels=GENER_FILTERS * 8,
                               kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(GENER_FILTERS * 8),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=GENER_FILTERS * 8, out_channels=GENER_FILTERS * 4,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(GENER_FILTERS * 4),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=GENER_FILTERS * 4, out_channels=GENER_FILTERS * 2,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(GENER_FILTERS * 2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=GENER_FILTERS * 2, out_channels=GENER_FILTERS,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(GENER_FILTERS),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=GENER_FILTERS, out_channels=output_shape[0],
                               kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.pipe(x)


def iterate_batches(envs, batch_size=BATCH_SIZE):
    batch = [e.reset() for e in envs]
    env_gen = iter(lambda: random.choice(envs), None)

    while True:
        e = next(env_gen)
        obs, reward, is_done, _ = e.step(e.action_space.sample())
        if np.mean(obs) > 0.01:
            batch.append(obs)
        if len(batch) == batch_size:
            # Normalising input between -1 to 1
            batch_np = np.array(batch, dtype=np.float32) * 2.0 / 255.0 - 1.0
            yield torch.tensor(batch_np)
            batch.clear()
        if is_done:
            e.reset()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action='store_true', help="Enable cuda computation")
    args = parser.parse_args(args={})

    device = torch.device("cuda" if args.cuda else "cpu")
    # envs = [InputWrapper(gym.make(name)) for name in ('Breakout-v0', 'AirRaid-v0', 'Pong-v0')]
    envs = [InputWrapper(gym.make(name)) for name in ['Breakout-v0']]
    input_shape = envs[0].observation_space.shape

    net_discr = Discriminator(input_shape=input_shape).to(device)
    net_gener = Generator(output_shape=input_shape).to(device)

    objective = nn.BCELoss()
    gen_optimizer = optim.Adam(params=net_gener.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    dis_optimizer = optim.Adam(params=net_discr.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

    true_labels_v = torch.ones(BATCH_SIZE, device=device)
    fake_labels_v = torch.zeros(BATCH_SIZE, device=device)

    def process_batch(trainer, batch):
        gen_input_v = torch.FloatTensor(
            BATCH_SIZE, LATENT_VECTOR_SIZE, 1, 1)
        gen_input_v.normal_(0, 1)
        gen_input_v = gen_input_v.to(device)
        batch_v = batch.to(device)
        gen_output_v = net_gener(gen_input_v)

        # train discriminator
        dis_optimizer.zero_grad()
        dis_output_true_v = net_discr(batch_v)
        dis_output_fake_v = net_discr(gen_output_v.detach())
        dis_loss = objective(dis_output_true_v, true_labels_v) + \
                   objective(dis_output_fake_v, fake_labels_v)
        dis_loss.backward()
        dis_optimizer.step()

        # train generator
        gen_optimizer.zero_grad()
        dis_output_v = net_discr(gen_output_v)
        gen_loss = objective(dis_output_v, true_labels_v)
        gen_loss.backward()
        gen_optimizer.step()

        if trainer.state.iteration % SAVE_IMAGE_EVERY_ITER == 0:
            fake_img = vutils.make_grid(
                gen_output_v.data[:64], normalize=True)
            trainer.tb.writer.add_image(
                "fake", fake_img, trainer.state.iteration)
            real_img = vutils.make_grid(
                batch_v.data[:64], normalize=True)
            trainer.tb.writer.add_image(
                "real", real_img, trainer.state.iteration)
            trainer.tb.writer.flush()
        return dis_loss.item(), gen_loss.item()

    engine = Engine(process_batch)
    tb = tb_logger.TensorboardLogger(log_dir=None)
    engine.tb = tb
    RunningAverage(output_transform=lambda out: out[1]).\
        attach(engine, "avg_loss_gen")
    RunningAverage(output_transform=lambda out: out[0]).\
        attach(engine, "avg_loss_dis")

    handler = tb_logger.OutputHandler(tag="train",
        metric_names=['avg_loss_gen', 'avg_loss_dis'])
    tb.attach(engine, log_handler=handler,
              event_name=Events.ITERATION_COMPLETED)

    @engine.on(Events.ITERATION_COMPLETED)
    def log_losses(trainer):
        if trainer.state.iteration % REPORT_EVERY_ITER == 0:
            log.info("%d: gen_loss=%f, dis_loss=%f",
                     trainer.state.iteration,
                     trainer.state.metrics['avg_loss_gen'],
                     trainer.state.metrics['avg_loss_dis'])

    engine.run(data=iterate_batches(envs))
```

<!-- #region id="cl5xHInAFOKq" -->
## Render environments in Colab
<!-- #endregion -->

<!-- #region id="SIKFUdcuG8n_" -->
### Alternative 1

It is possible to visualize the game your agent is playing, even on CoLab.  This section provides information on how to generate a video in CoLab that shows you an episode of the game your agent is playing. This video process is based on suggestions found [here](https://colab.research.google.com/drive/1flu31ulJlgiRL1dnN2ir8wGh9p7Zij2t).

Begin by installing **pyvirtualdisplay** and **python-opengl**.
<!-- #endregion -->

```python id="XtIfwpApFs_T"
!pip install gym pyvirtualdisplay > /dev/null 2>&1
!apt-get install -y xvfb python-opengl ffmpeg > /dev/null 2>&1

!apt-get update > /dev/null 2>&1
!apt-get install cmake > /dev/null 2>&1
!pip install --upgrade setuptools 2>&1
!pip install ez_setup > /dev/null 2>&1
!pip install gym[atari] > /dev/null 2>&1

!wget http://www.atarimania.com/roms/Roms.rar
!mkdir /content/ROM/
!unrar e /content/Roms.rar /content/ROM/
!python -m atari_py.import_roms /content/ROM/
```

```python id="oKts3eIlFT-a"
import gym
from gym.wrappers import Monitor
import glob
import io
import base64
from IPython.display import HTML
from pyvirtualdisplay import Display
from IPython import display as ipythondisplay

display = Display(visible=0, size=(1400, 900))
display.start()

"""
Utility functions to enable video recording of gym environment 
and displaying it.
To enable video, just do "env = wrap_env(env)""
"""

def show_video():
  mp4list = glob.glob('video/*.mp4')
  if len(mp4list) > 0:
    mp4 = mp4list[0]
    video = io.open(mp4, 'r+b').read()
    encoded = base64.b64encode(video)
    ipythondisplay.display(HTML(data='''<video alt="test" autoplay 
                loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded.decode('ascii'))))
  else: 
    print("Could not find video")
    

def wrap_env(env):
  env = Monitor(env, './video', force=True)
  return env
```

```python colab={"base_uri": "https://localhost:8080/", "height": 421} id="mCWle-h7FeR3" executionInfo={"status": "ok", "timestamp": 1634639353978, "user_tz": -330, "elapsed": 11880, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="af194f6b-e5b8-44b9-e486-953b778d9fa2"
#env = wrap_env(gym.make("MountainCar-v0"))
env = wrap_env(gym.make("Atlantis-v0"))

observation = env.reset()

while True:
    env.render()
    #your agent goes here
    action = env.action_space.sample() 
    observation, reward, done, info = env.step(action)
    if done: 
      break;
            
env.close()
show_video()
```

<!-- #region id="zBpzrhKBFud1" -->
### Alternative 2
<!-- #endregion -->

```python id="rxNNWY6vG_-X"
!apt-get install -y xvfb python-opengl ffmpeg > /dev/null 2>&1
!pip install colabgymrender
```

```python colab={"base_uri": "https://localhost:8080/", "height": 231} id="VswegGehGuRB" executionInfo={"status": "ok", "timestamp": 1634639514595, "user_tz": -330, "elapsed": 1423, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="c8484da9-8118-4375-c654-59f054495c3b"
import gym
from colabgymrender.recorder import Recorder

env = gym.make("Breakout-v0")
directory = './video'
env = Recorder(env, directory)

observation = env.reset()
terminal = False
while not terminal:
  action = env.action_space.sample()
  observation, reward, terminal, info = env.step(action)

env.play()
```
