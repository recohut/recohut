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
    language: python
    name: python3
---

<!-- #region id="k5saFw-rMWWi" -->
**Chapter 18 – Reinforcement Learning**
<!-- #endregion -->

<!-- #region id="1OkSYnMdOAOk" -->
# Reinforcement Learning fundamentals in action
> Chapter 18 of the hands-on-ml book

- toc: true
- badges: true
- comments: true
- categories: [Reinforcement, RL, ReinforcementLearning]
- author: "<a href='https://github.com/ageron/handson-ml2'>Aurélien Geron</a>"
- image:
<!-- #endregion -->

<!-- #region id="pd12pbIIMWWv" -->
## Setup
First, let's import a few common modules, ensure MatplotLib plots figures inline and prepare a function to save the figures. We also check that Python 3.5 or later is installed (although Python 2.x may work, it is deprecated so we strongly recommend you use Python 3 instead), as well as Scikit-Learn ≥0.20 and TensorFlow ≥2.0.
<!-- #endregion -->

```python id="QY_JO-KJMWWy"
# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)

# Is this notebook running on Colab or Kaggle?
IS_COLAB = "google.colab" in sys.modules
IS_KAGGLE = "kaggle_secrets" in sys.modules

if IS_COLAB or IS_KAGGLE:
    !apt update && apt install -y libpq-dev libsdl2-dev swig xorg-dev xvfb
    !pip install -q -U tf-agents pyvirtualdisplay gym[atari,box2d]

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"

# TensorFlow ≥2.0 is required
import tensorflow as tf
from tensorflow import keras
assert tf.__version__ >= "2.0"

if not tf.config.list_physical_devices('GPU'):
    print("No GPU was detected. CNNs can be very slow without a GPU.")
    if IS_COLAB:
        print("Go to Runtime > Change runtime and select a GPU hardware accelerator.")
    if IS_KAGGLE:
        print("Go to Settings > Accelerator and select GPU.")

# Common imports
import numpy as np
import os

# to make this notebook's output stable across runs
np.random.seed(42)
tf.random.set_seed(42)

# To plot pretty figures
%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# To get smooth animations
import matplotlib.animation as animation
mpl.rc('animation', html='jshtml')

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "rl"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
```

<!-- #region id="4a9jsuu2MWW2" -->
## Introduction to OpenAI gym
<!-- #endregion -->

<!-- #region id="qmhqvD98MWW3" -->
In this notebook we will be using [OpenAI gym](https://gym.openai.com/), a great toolkit for developing and comparing Reinforcement Learning algorithms. It provides many environments for your learning *agents* to interact with. Let's start by importing `gym`:
<!-- #endregion -->

```python id="ydcnQKlJMWW4"
import gym
```

<!-- #region id="PZybtPP6MWW5" -->
Let's list all the available environments:
<!-- #endregion -->

```python id="Oeew_Em-MWW7" outputId="958fad64-21cb-4f44-850c-f8d2f22d9e5f"
gym.envs.registry.all()
```

<!-- #region id="tKprW6mpMWW_" -->
The Cart-Pole is a very simple environment composed of a cart that can move left or right, and pole placed vertically on top of it. The agent must move the cart left or right to keep the pole upright.
<!-- #endregion -->

```python id="yYhMp-HTMWXA"
env = gym.make('CartPole-v1')
```

<!-- #region id="VgmiEEpdMWXB" -->
Let's initialize the environment by calling is `reset()` method. This returns an observation:
<!-- #endregion -->

```python id="qpXs4XTRMWXB"
env.seed(42)
obs = env.reset()
```

<!-- #region id="pvA3aP_YMWXC" -->
Observations vary depending on the environment. In this case it is a 1D NumPy array composed of 4 floats: they represent the cart's horizontal position, its velocity, the angle of the pole (0 = vertical), and the angular velocity.
<!-- #endregion -->

```python id="k7o0JW4nMWXD" outputId="26d209d2-29dd-44db-c484-968ba9b1b4ae"
obs
```

<!-- #region id="_MbuAScSMWXE" -->
An environment can be visualized by calling its `render()` method, and you can pick the rendering mode (the rendering options depend on the environment).
<!-- #endregion -->

<!-- #region id="vdIozAt7MWXF" -->
**Warning**: some environments (including the Cart-Pole) require access to your display, which opens up a separate window, even if you specify `mode="rgb_array"`. In general you can safely ignore that window. However, if Jupyter is running on a headless server (ie. without a screen) it will raise an exception. One way to avoid this is to install a fake X server like [Xvfb](http://en.wikipedia.org/wiki/Xvfb). On Debian or Ubuntu:

```bash
$ apt update
$ apt install -y xvfb
```

You can then start Jupyter using the `xvfb-run` command:

```bash
$ xvfb-run -s "-screen 0 1400x900x24" jupyter notebook
```

Alternatively, you can install the [pyvirtualdisplay](https://github.com/ponty/pyvirtualdisplay) Python library which wraps Xvfb:

```bash
python3 -m pip install -U pyvirtualdisplay
```

And run the following code:
<!-- #endregion -->

```python id="ylC9f6SsMWXG"
try:
    import pyvirtualdisplay
    display = pyvirtualdisplay.Display(visible=0, size=(1400, 900)).start()
except ImportError:
    pass
```

```python id="TIILjaO8MWXH" outputId="1ac16cf2-baf4-4370-dd4c-267f4e08e848"
env.render()
```

<!-- #region id="UYcZDtoCMWXI" -->
In this example we will set `mode="rgb_array"` to get an image of the environment as a NumPy array:
<!-- #endregion -->

```python id="H0rBrHoAMWXI" outputId="d361aac7-7c61-4958-cad3-a48c44bd4f3d"
img = env.render(mode="rgb_array")
img.shape
```

```python id="_l8_WViGMWXN"
def plot_environment(env, figsize=(5,4)):
    plt.figure(figsize=figsize)
    img = env.render(mode="rgb_array")
    plt.imshow(img)
    plt.axis("off")
    return img
```

```python id="5aZcvCMLMWXO" outputId="4c091c88-bf08-4719-fa47-c9efee1b59f4"
plot_environment(env)
plt.show()
```

<!-- #region id="CvU1RQfoMWXQ" -->
Let's see how to interact with an environment. Your agent will need to select an action from an "action space" (the set of possible actions). Let's see what this environment's action space looks like:
<!-- #endregion -->

```python id="UMifBFbeMWXR" outputId="cf38b61d-d7d8-4b19-ee20-f3cbfe6be839"
env.action_space
```

<!-- #region id="4042tBq0MWXS" -->
Yep, just two possible actions: accelerate towards the left or towards the right.
<!-- #endregion -->

<!-- #region id="m7erO2OGMWXT" -->
Since the pole is leaning toward the right (`obs[2] > 0`), let's accelerate the cart toward the right:
<!-- #endregion -->

```python id="EanJnwhDMWXU" outputId="dfa270c5-37c6-4844-eda9-c7ff72a098fd"
action = 1  # accelerate right
obs, reward, done, info = env.step(action)
obs
```

<!-- #region id="zQGRrntBMWXU" -->
Notice that the cart is now moving toward the right (`obs[1] > 0`). The pole is still tilted toward the right (`obs[2] > 0`), but its angular velocity is now negative (`obs[3] < 0`), so it will likely be tilted toward the left after the next step.
<!-- #endregion -->

```python id="lGUFEmKZMWXV" outputId="655f1fc0-6fc7-4cc3-af3f-8707f23b188d"
plot_environment(env)
save_fig("cart_pole_plot")
```

<!-- #region id="hOFB_EsrMWXW" -->
Looks like it's doing what we're telling it to do!
<!-- #endregion -->

<!-- #region id="bsdqPqqnMWXX" -->
The environment also tells the agent how much reward it got during the last step:
<!-- #endregion -->

```python id="hAKLkh-PMWXX" outputId="393fa762-544b-4435-8404-2aa19b1ea7e2"
reward
```

<!-- #region id="zoCG-3k_MWXY" -->
When the game is over, the environment returns `done=True`:
<!-- #endregion -->

```python id="US3cFcQuMWXZ" outputId="f564a78f-4a4b-459f-c5d7-5843c76ba89b"
done
```

<!-- #region id="3g1WXF2kMWXa" -->
Finally, `info` is an environment-specific dictionary that can provide some extra information that you may find useful for debugging or for training. For example, in some games it may indicate how many lives the agent has.
<!-- #endregion -->

```python id="i96nzjY6MWXb" outputId="a0360307-34d8-44fd-e85b-b25e4466cea4"
info
```

<!-- #region id="grpXSNAbMWXd" -->
The sequence of steps between the moment the environment is reset until it is done is called an "episode". At the end of an episode (i.e., when `step()` returns `done=True`), you should reset the environment before you continue to use it.
<!-- #endregion -->

```python id="Kq8XUPwrMWXe"
if done:
    obs = env.reset()
```

<!-- #region id="jPLfhvTuMWXe" -->
Now how can we make the poll remain upright? We will need to define a _policy_ for that. This is the strategy that the agent will use to select an action at each step. It can use all the past actions and observations to decide what to do.
<!-- #endregion -->

<!-- #region id="US6BVjKWMWXf" -->
## A simple hard-coded policy
<!-- #endregion -->

<!-- #region id="0xJIo3tkMWXf" -->
Let's hard code a simple strategy: if the pole is tilting to the left, then push the cart to the left, and _vice versa_. Let's see if that works:
<!-- #endregion -->

```python id="kBVCTXdNMWXg"
env.seed(42)

def basic_policy(obs):
    angle = obs[2]
    return 0 if angle < 0 else 1

totals = []
for episode in range(500):
    episode_rewards = 0
    obs = env.reset()
    for step in range(200):
        action = basic_policy(obs)
        obs, reward, done, info = env.step(action)
        episode_rewards += reward
        if done:
            break
    totals.append(episode_rewards)
```

```python id="berwMp5mMWXh" outputId="c3754bec-d4e3-4406-ef1c-efcd4f36aed7"
np.mean(totals), np.std(totals), np.min(totals), np.max(totals)
```

<!-- #region id="k7jW70ipMWXi" -->
Well, as expected, this strategy is a bit too basic: the best it did was to keep the poll up for only 68 steps. This environment is considered solved when the agent keeps the poll up for 200 steps.
<!-- #endregion -->

<!-- #region id="Pwzn0TVQMWXj" -->
Let's visualize one episode:
<!-- #endregion -->

```python id="qx9rqVdpMWXj"
env.seed(42)

frames = []

obs = env.reset()
for step in range(200):
    img = env.render(mode="rgb_array")
    frames.append(img)
    action = basic_policy(obs)

    obs, reward, done, info = env.step(action)
    if done:
        break
```

<!-- #region id="JJuo3iWsMWXk" -->
Now show the animation:
<!-- #endregion -->

```python id="0KMHT45_MWXk"
def update_scene(num, frames, patch):
    patch.set_data(frames[num])
    return patch,

def plot_animation(frames, repeat=False, interval=40):
    fig = plt.figure()
    patch = plt.imshow(frames[0])
    plt.axis('off')
    anim = animation.FuncAnimation(
        fig, update_scene, fargs=(frames, patch),
        frames=len(frames), repeat=repeat, interval=interval)
    plt.close()
    return anim
```

```python id="q8KZSw-PMWXk"
plot_animation(frames)
```

<!-- #region id="2MKpS-72MWXl" -->
Clearly the system is unstable and after just a few wobbles, the pole ends up too tilted: game over. We will need to be smarter than that!
<!-- #endregion -->

<!-- #region id="vDUWP6ajMWXl" -->
## Neural Network Policies
<!-- #endregion -->

<!-- #region id="8S-zVVIrMWXo" -->
Let's create a neural network that will take observations as inputs, and output the probabilities of actions to take for each observation. To choose an action, the network will estimate a probability for each action, then we will select an action randomly according to the estimated probabilities. In the case of the Cart-Pole environment, there are just two possible actions (left or right), so we only need one output neuron: it will output the probability `p` of the action 0 (left), and of course the probability of action 1 (right) will be `1 - p`.
<!-- #endregion -->

```python id="p1wgi3poMWXp"
keras.backend.clear_session()
tf.random.set_seed(42)
np.random.seed(42)

n_inputs = 4 # == env.observation_space.shape[0]

model = keras.models.Sequential([
    keras.layers.Dense(5, activation="elu", input_shape=[n_inputs]),
    keras.layers.Dense(1, activation="sigmoid"),
])
```

<!-- #region id="xtXdLq5aMWXq" -->
In this particular environment, the past actions and observations can safely be ignored, since each observation contains the environment's full state. If there were some hidden state then you may need to consider past actions and observations in order to try to infer the hidden state of the environment. For example, if the environment only revealed the position of the cart but not its velocity, you would have to consider not only the current observation but also the previous observation in order to estimate the current velocity. Another example is if the observations are noisy: you may want to use the past few observations to estimate the most likely current state. Our problem is thus as simple as can be: the current observation is noise-free and contains the environment's full state.
<!-- #endregion -->

<!-- #region id="8YyDR4GZMWXs" -->
You may wonder why we plan to pick a random action based on the probability given by the policy network, rather than just picking the action with the highest probability. This approach lets the agent find the right balance between _exploring_ new actions and _exploiting_ the actions that are known to work well. Here's an analogy: suppose you go to a restaurant for the first time, and all the dishes look equally appealing so you randomly pick one. If it turns out to be good, you can increase the probability to order it next time, but you shouldn't increase that probability to 100%, or else you will never try out the other dishes, some of which may be even better than the one you tried.
<!-- #endregion -->

<!-- #region id="Cd369r1zMWXt" -->
Let's write a small function that will run the model to play one episode, and return the frames so we can display an animation:
<!-- #endregion -->

```python id="Kn2CgT2ZMWXu"
def render_policy_net(model, n_max_steps=200, seed=42):
    frames = []
    env = gym.make("CartPole-v1")
    env.seed(seed)
    np.random.seed(seed)
    obs = env.reset()
    for step in range(n_max_steps):
        frames.append(env.render(mode="rgb_array"))
        left_proba = model.predict(obs.reshape(1, -1))
        action = int(np.random.rand() > left_proba)
        obs, reward, done, info = env.step(action)
        if done:
            break
    env.close()
    return frames
```

<!-- #region id="FpnwtcD8MWXu" -->
Now let's look at how well this randomly initialized policy network performs:
<!-- #endregion -->

```python id="_TV8_725MWXv"
frames = render_policy_net(model)
plot_animation(frames)
```

<!-- #region id="WfHkU40AMWXv" -->
Yeah... pretty bad. The neural network will have to learn to do better. First let's see if it is capable of learning the basic policy we used earlier: go left if the pole is tilting left, and go right if it is tilting right.
<!-- #endregion -->

<!-- #region id="_rlRywZqMWXw" -->
We can make the same net play in 50 different environments in parallel (this will give us a diverse training batch at each step), and train for 5000 iterations. We also reset environments when they are done. We train the model using a custom training loop so we can easily use the predictions at each training step to advance the environments.
<!-- #endregion -->

```python id="Dk0UxQCSMWXw" outputId="15b2e168-0778-498d-9b42-0850f939739e"
n_environments = 50
n_iterations = 5000

envs = [gym.make("CartPole-v1") for _ in range(n_environments)]
for index, env in enumerate(envs):
    env.seed(index)
np.random.seed(42)
observations = [env.reset() for env in envs]
optimizer = keras.optimizers.RMSprop()
loss_fn = keras.losses.binary_crossentropy

for iteration in range(n_iterations):
    # if angle < 0, we want proba(left) = 1., or else proba(left) = 0.
    target_probas = np.array([([1.] if obs[2] < 0 else [0.])
                              for obs in observations])
    with tf.GradientTape() as tape:
        left_probas = model(np.array(observations))
        loss = tf.reduce_mean(loss_fn(target_probas, left_probas))
    print("\rIteration: {}, Loss: {:.3f}".format(iteration, loss.numpy()), end="")
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    actions = (np.random.rand(n_environments, 1) > left_probas.numpy()).astype(np.int32)
    for env_index, env in enumerate(envs):
        obs, reward, done, info = env.step(actions[env_index][0])
        observations[env_index] = obs if not done else env.reset()

for env in envs:
    env.close()
```

```python id="tfdxS-SjMWXy"
frames = render_policy_net(model)
plot_animation(frames)
```

<!-- #region id="v2J5-17yMWXz" -->
Looks like it learned the policy correctly. Now let's see if it can learn a better policy on its own. One that does not wobble as much.
<!-- #endregion -->

<!-- #region id="TzYnP9QBMWXz" -->
## Policy Gradients
<!-- #endregion -->

<!-- #region id="yp5Ed4BPMWX0" -->
To train this neural network we will need to define the target probabilities `y`. If an action is good we should increase its probability, and conversely if it is bad we should reduce it. But how do we know whether an action is good or bad? The problem is that most actions have delayed effects, so when you win or lose points in an episode, it is not clear which actions contributed to this result: was it just the last action? Or the last 10? Or just one action 50 steps earlier? This is called the _credit assignment problem_.

The _Policy Gradients_ algorithm tackles this problem by first playing multiple episodes, then making the actions in good episodes slightly more likely, while actions in bad episodes are made slightly less likely. First we play, then we go back and think about what we did.
<!-- #endregion -->

<!-- #region id="gaTjgEr9MWX1" -->
Let's start by creating a function to play a single step using the model. We will also pretend for now that whatever action it takes is the right one, so we can compute the loss and its gradients (we will just save these gradients for now, and modify them later depending on how good or bad the action turned out to be):
<!-- #endregion -->

```python id="HRRZy3sbMWX1"
def play_one_step(env, obs, model, loss_fn):
    with tf.GradientTape() as tape:
        left_proba = model(obs[np.newaxis])
        action = (tf.random.uniform([1, 1]) > left_proba)
        y_target = tf.constant([[1.]]) - tf.cast(action, tf.float32)
        loss = tf.reduce_mean(loss_fn(y_target, left_proba))
    grads = tape.gradient(loss, model.trainable_variables)
    obs, reward, done, info = env.step(int(action[0, 0].numpy()))
    return obs, reward, done, grads
```

<!-- #region id="XOgro9uiMWX3" -->
If `left_proba` is high, then `action` will most likely be `False` (since a random number uniformally sampled between 0 and 1 will probably not be greater than `left_proba`). And `False` means 0 when you cast it to a number, so `y_target` would be equal to 1 - 0 = 1. In other words, we set the target to 1, meaning we pretend that the probability of going left should have been 100% (so we took the right action).
<!-- #endregion -->

<!-- #region id="300oqwdHMWX4" -->
Now let's create another function that will rely on the `play_one_step()` function to play multiple episodes, returning all the rewards and gradients, for each episode and each step:
<!-- #endregion -->

```python id="fHpuVWomMWX4"
def play_multiple_episodes(env, n_episodes, n_max_steps, model, loss_fn):
    all_rewards = []
    all_grads = []
    for episode in range(n_episodes):
        current_rewards = []
        current_grads = []
        obs = env.reset()
        for step in range(n_max_steps):
            obs, reward, done, grads = play_one_step(env, obs, model, loss_fn)
            current_rewards.append(reward)
            current_grads.append(grads)
            if done:
                break
        all_rewards.append(current_rewards)
        all_grads.append(current_grads)
    return all_rewards, all_grads
```

<!-- #region id="Ipjw1TlIMWX5" -->
The Policy Gradients algorithm uses the model to play the episode several times (e.g., 10 times), then it goes back and looks at all the rewards, discounts them and normalizes them. So let's create couple functions for that: the first will compute discounted rewards; the second will normalize the discounted rewards across many episodes.
<!-- #endregion -->

```python id="BGI5MEbTMWX6"
def discount_rewards(rewards, discount_rate):
    discounted = np.array(rewards)
    for step in range(len(rewards) - 2, -1, -1):
        discounted[step] += discounted[step + 1] * discount_rate
    return discounted

def discount_and_normalize_rewards(all_rewards, discount_rate):
    all_discounted_rewards = [discount_rewards(rewards, discount_rate)
                              for rewards in all_rewards]
    flat_rewards = np.concatenate(all_discounted_rewards)
    reward_mean = flat_rewards.mean()
    reward_std = flat_rewards.std()
    return [(discounted_rewards - reward_mean) / reward_std
            for discounted_rewards in all_discounted_rewards]
```

<!-- #region id="xpL-iUZsMWX7" -->
Say there were 3 actions, and after each action there was a reward: first 10, then 0, then -50. If we use a discount factor of 80%, then the 3rd action will get -50 (full credit for the last reward), but the 2nd action will only get -40 (80% credit for the last reward), and the 1st action will get 80% of -40 (-32) plus full credit for the first reward (+10), which leads to a discounted reward of -22:
<!-- #endregion -->

```python id="qvM2ylO8MWX8" outputId="c4e56b3c-61f6-41bd-d440-44535699091b"
discount_rewards([10, 0, -50], discount_rate=0.8)
```

<!-- #region id="SAPgMGyaMWX9" -->
To normalize all discounted rewards across all episodes, we compute the mean and standard deviation of all the discounted rewards, and we subtract the mean from each discounted reward, and divide by the standard deviation:
<!-- #endregion -->

```python id="rGtLuWmTMWX-" outputId="c8c75aa5-d8b7-419d-ccf6-cb83dcf0eea3"
discount_and_normalize_rewards([[10, 0, -50], [10, 20]], discount_rate=0.8)
```

```python id="3jawO_kUMWX_"
n_iterations = 150
n_episodes_per_update = 10
n_max_steps = 200
discount_rate = 0.95
```

```python id="V1QL4nNPMWX_"
optimizer = keras.optimizers.Adam(lr=0.01)
loss_fn = keras.losses.binary_crossentropy
```

```python id="JzhOPm8VMWYA"
keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

model = keras.models.Sequential([
    keras.layers.Dense(5, activation="elu", input_shape=[4]),
    keras.layers.Dense(1, activation="sigmoid"),
])
```

```python id="UFyKdwK1MWYB" outputId="c11ce703-2240-46e7-d0d1-09741370a8b6"
env = gym.make("CartPole-v1")
env.seed(42);

for iteration in range(n_iterations):
    all_rewards, all_grads = play_multiple_episodes(
        env, n_episodes_per_update, n_max_steps, model, loss_fn)
    total_rewards = sum(map(sum, all_rewards))                     # Not shown in the book
    print("\rIteration: {}, mean rewards: {:.1f}".format(          # Not shown
        iteration, total_rewards / n_episodes_per_update), end="") # Not shown
    all_final_rewards = discount_and_normalize_rewards(all_rewards,
                                                       discount_rate)
    all_mean_grads = []
    for var_index in range(len(model.trainable_variables)):
        mean_grads = tf.reduce_mean(
            [final_reward * all_grads[episode_index][step][var_index]
             for episode_index, final_rewards in enumerate(all_final_rewards)
                 for step, final_reward in enumerate(final_rewards)], axis=0)
        all_mean_grads.append(mean_grads)
    optimizer.apply_gradients(zip(all_mean_grads, model.trainable_variables))

env.close()
```

```python id="9dudUn3FMWYD"
frames = render_policy_net(model)
plot_animation(frames)
```

<!-- #region id="d_qYvOq7MWYE" -->
## Markov Chains
<!-- #endregion -->

```python id="eE6puB7TMWYE" outputId="5c024980-1269-465b-9a23-70eade06e07a"
np.random.seed(42)

transition_probabilities = [ # shape=[s, s']
        [0.7, 0.2, 0.0, 0.1],  # from s0 to s0, s1, s2, s3
        [0.0, 0.0, 0.9, 0.1],  # from s1 to ...
        [0.0, 1.0, 0.0, 0.0],  # from s2 to ...
        [0.0, 0.0, 0.0, 1.0]]  # from s3 to ...

n_max_steps = 50

def print_sequence():
    current_state = 0
    print("States:", end=" ")
    for step in range(n_max_steps):
        print(current_state, end=" ")
        if current_state == 3:
            break
        current_state = np.random.choice(range(4), p=transition_probabilities[current_state])
    else:
        print("...", end="")
    print()

for _ in range(10):
    print_sequence()
```

<!-- #region id="Kweq7SXtMWYF" -->
## Markov Decision Process
<!-- #endregion -->

<!-- #region id="waXsw7HJMWYG" -->
Let's define some transition probabilities, rewards and possible actions. For example, in state s0, if action a0 is chosen then with proba 0.7 we will go to state s0 with reward +10, with probability 0.3 we will go to state s1 with no reward, and with never go to state s2 (so the transition probabilities are `[0.7, 0.3, 0.0]`, and the rewards are `[+10, 0, 0]`):
<!-- #endregion -->

```python id="lPuwBsl6MWYG"
transition_probabilities = [ # shape=[s, a, s']
        [[0.7, 0.3, 0.0], [1.0, 0.0, 0.0], [0.8, 0.2, 0.0]],
        [[0.0, 1.0, 0.0], None, [0.0, 0.0, 1.0]],
        [None, [0.8, 0.1, 0.1], None]]
rewards = [ # shape=[s, a, s']
        [[+10, 0, 0], [0, 0, 0], [0, 0, 0]],
        [[0, 0, 0], [0, 0, 0], [0, 0, -50]],
        [[0, 0, 0], [+40, 0, 0], [0, 0, 0]]]
possible_actions = [[0, 1, 2], [0, 2], [1]]
```

<!-- #region id="MIm3uVVgMWYH" -->
## Q-Value Iteration
<!-- #endregion -->

```python id="mWvasSMAMWYI"
Q_values = np.full((3, 3), -np.inf) # -np.inf for impossible actions
for state, actions in enumerate(possible_actions):
    Q_values[state, actions] = 0.0  # for all possible actions
```

```python id="NkjiVsYzMWYI"
gamma = 0.90  # the discount factor

history1 = [] # Not shown in the book (for the figure below)
for iteration in range(50):
    Q_prev = Q_values.copy()
    history1.append(Q_prev) # Not shown
    for s in range(3):
        for a in possible_actions[s]:
            Q_values[s, a] = np.sum([
                    transition_probabilities[s][a][sp]
                    * (rewards[s][a][sp] + gamma * np.max(Q_prev[sp]))
                for sp in range(3)])

history1 = np.array(history1) # Not shown
```

```python id="vMen0CEmMWYJ" outputId="ee127175-a3f7-428c-c6d4-5071bd2b137d"
Q_values
```

```python id="ZqNPaAviMWYK" outputId="c79a6f52-bd92-443f-c486-a5d273411d9a"
np.argmax(Q_values, axis=1)
```

<!-- #region id="zVZ3mZ_WMWYL" -->
The optimal policy for this MDP, when using a discount factor of 0.90, is to choose action a0 when in state s0, and choose action a0 when in state s1, and finally choose action a1 (the only possible action) when in state s2.
<!-- #endregion -->

<!-- #region id="VowkQNPrMWYL" -->
Let's try again with a discount factor of 0.95:
<!-- #endregion -->

```python id="FV0Q2ikKMWYL"
Q_values = np.full((3, 3), -np.inf) # -np.inf for impossible actions
for state, actions in enumerate(possible_actions):
    Q_values[state, actions] = 0.0  # for all possible actions
```

```python id="ZOkhuMuhMWYM"
gamma = 0.95  # the discount factor

for iteration in range(50):
    Q_prev = Q_values.copy()
    for s in range(3):
        for a in possible_actions[s]:
            Q_values[s, a] = np.sum([
                    transition_probabilities[s][a][sp]
                    * (rewards[s][a][sp] + gamma * np.max(Q_prev[sp]))
                for sp in range(3)])
```

```python id="wOe-hv0TMWYN" outputId="b4c44d8b-b7f8-4846-8b36-3e04da352178"
Q_values
```

```python id="dpV7a6sqMWYO" outputId="22af3fc3-4076-479f-aceb-0d108b63ae8d"
np.argmax(Q_values, axis=1)
```

<!-- #region id="BgefuSKKMWYP" -->
Now the policy has changed! In state s1, we now prefer to go through the fire (choose action a2). This is because the discount factor is larger so the agent values the future more, and it is therefore ready to pay an immediate penalty in order to get more future rewards.
<!-- #endregion -->

<!-- #region id="sLD467n5MWYQ" -->
## Q-Learning
<!-- #endregion -->

<!-- #region id="Myoj7eu3MWYQ" -->
Q-Learning works by watching an agent play (e.g., randomly) and gradually improving its estimates of the Q-Values. Once it has accurate Q-Value estimates (or close enough), then the optimal policy consists in choosing the action that has the highest Q-Value (i.e., the greedy policy).
<!-- #endregion -->

<!-- #region id="78stUKuuMWYR" -->
We will need to simulate an agent moving around in the environment, so let's define a function to perform some action and get the new state and a reward:
<!-- #endregion -->

```python id="CaJRbfGGMWYR"
def step(state, action):
    probas = transition_probabilities[state][action]
    next_state = np.random.choice([0, 1, 2], p=probas)
    reward = rewards[state][action][next_state]
    return next_state, reward
```

<!-- #region id="1Knc3cccMWYV" -->
We also need an exploration policy, which can be any policy, as long as it visits every possible state many times. We will just use a random policy, since the state space is very small:
<!-- #endregion -->

```python id="whDnvHbrMWYW"
def exploration_policy(state):
    return np.random.choice(possible_actions[state])
```

<!-- #region id="VKNQHd1NMWYW" -->
Now let's initialize the Q-Values like earlier, and run the Q-Learning algorithm:
<!-- #endregion -->

```python id="TkWy3pVZMWYX"
np.random.seed(42)

Q_values = np.full((3, 3), -np.inf)
for state, actions in enumerate(possible_actions):
    Q_values[state][actions] = 0

alpha0 = 0.05 # initial learning rate
decay = 0.005 # learning rate decay
gamma = 0.90 # discount factor
state = 0 # initial state
history2 = [] # Not shown in the book

for iteration in range(10000):
    history2.append(Q_values.copy()) # Not shown
    action = exploration_policy(state)
    next_state, reward = step(state, action)
    next_value = np.max(Q_values[next_state]) # greedy policy at the next step
    alpha = alpha0 / (1 + iteration * decay)
    Q_values[state, action] *= 1 - alpha
    Q_values[state, action] += alpha * (reward + gamma * next_value)
    state = next_state

history2 = np.array(history2) # Not shown
```

```python id="ySqm6su1MWYY" outputId="537e95cd-932b-46c7-9da9-a880c22d151a"
Q_values
```

```python id="CposPS2pMWYZ" outputId="9bdebb66-04be-4397-daaa-148f273ebe51"
np.argmax(Q_values, axis=1) # optimal action for each state
```

```python id="UeI-MFGnMWYZ" outputId="699d8696-5fc1-42c6-fba6-55d33dbb5c53"
true_Q_value = history1[-1, 0, 0]

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
axes[0].set_ylabel("Q-Value$(s_0, a_0)$", fontsize=14)
axes[0].set_title("Q-Value Iteration", fontsize=14)
axes[1].set_title("Q-Learning", fontsize=14)
for ax, width, history in zip(axes, (50, 10000), (history1, history2)):
    ax.plot([0, width], [true_Q_value, true_Q_value], "k--")
    ax.plot(np.arange(width), history[:, 0, 0], "b-", linewidth=2)
    ax.set_xlabel("Iterations", fontsize=14)
    ax.axis([0, width, 0, 24])

save_fig("q_value_plot")
```

<!-- #region id="W1Ymfl3TMWYb" -->
## Deep Q-Network
<!-- #endregion -->

<!-- #region id="dAYtAvGSMWYb" -->
Let's build the DQN. Given a state, it will estimate, for each possible action, the sum of discounted future rewards it can expect after it plays that action (but before it sees its outcome):
<!-- #endregion -->

```python id="H1uodcjeMWYc"
keras.backend.clear_session()
tf.random.set_seed(42)
np.random.seed(42)

env = gym.make("CartPole-v1")
input_shape = [4] # == env.observation_space.shape
n_outputs = 2 # == env.action_space.n

model = keras.models.Sequential([
    keras.layers.Dense(32, activation="elu", input_shape=input_shape),
    keras.layers.Dense(32, activation="elu"),
    keras.layers.Dense(n_outputs)
])
```

<!-- #region id="hMU0wAo_MWYd" -->
To select an action using this DQN, we just pick the action with the largest predicted Q-value. However, to ensure that the agent explores the environment, we choose a random action with probability `epsilon`.
<!-- #endregion -->

```python id="x5Sa5rpxMWYe"
def epsilon_greedy_policy(state, epsilon=0):
    if np.random.rand() < epsilon:
        return np.random.randint(n_outputs)
    else:
        Q_values = model.predict(state[np.newaxis])
        return np.argmax(Q_values[0])
```

<!-- #region id="iDj9SJHQMWYf" -->
We will also need a replay memory. It will contain the agent's experiences, in the form of tuples: `(obs, action, reward, next_obs, done)`. We can use the `deque` class for that (but make sure to check out DeepMind's excellent [Reverb library](https://github.com/deepmind/reverb) for a much more robust implementation of experience replay):
<!-- #endregion -->

```python id="t8D3DJnuMWYf"
from collections import deque

replay_memory = deque(maxlen=2000)
```

<!-- #region id="ISnv-500MWYg" -->
And let's create a function to sample experiences from the replay memory. It will return 5 NumPy arrays: `[obs, actions, rewards, next_obs, dones]`.
<!-- #endregion -->

```python id="cKkdv-WCMWYg"
def sample_experiences(batch_size):
    indices = np.random.randint(len(replay_memory), size=batch_size)
    batch = [replay_memory[index] for index in indices]
    states, actions, rewards, next_states, dones = [
        np.array([experience[field_index] for experience in batch])
        for field_index in range(5)]
    return states, actions, rewards, next_states, dones
```

<!-- #region id="oJnIf2OQMWYh" -->
Now we can create a function that will use the DQN to play one step, and record its experience in the replay memory:
<!-- #endregion -->

```python id="uN9RPRDuMWYi"
def play_one_step(env, state, epsilon):
    action = epsilon_greedy_policy(state, epsilon)
    next_state, reward, done, info = env.step(action)
    replay_memory.append((state, action, reward, next_state, done))
    return next_state, reward, done, info
```

<!-- #region id="-Wv0SIarMWYj" -->
Lastly, let's create a function that will sample some experiences from the replay memory and perform a training step:

**Notes**:
* The first 3 releases of the 2nd edition were missing the `reshape()` operation which converts `target_Q_values` to a column vector (this is required by the `loss_fn()`).
* The book uses a learning rate of 1e-3, but in the code below I use 1e-2, as it significantly improves training. I also tuned the learning rates of the DQN variants below.
<!-- #endregion -->

```python id="hMn8KKBVMWYk"
batch_size = 32
discount_rate = 0.95
optimizer = keras.optimizers.Adam(lr=1e-2)
loss_fn = keras.losses.mean_squared_error

def training_step(batch_size):
    experiences = sample_experiences(batch_size)
    states, actions, rewards, next_states, dones = experiences
    next_Q_values = model.predict(next_states)
    max_next_Q_values = np.max(next_Q_values, axis=1)
    target_Q_values = (rewards +
                       (1 - dones) * discount_rate * max_next_Q_values)
    target_Q_values = target_Q_values.reshape(-1, 1)
    mask = tf.one_hot(actions, n_outputs)
    with tf.GradientTape() as tape:
        all_Q_values = model(states)
        Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
        loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
```

<!-- #region id="lBgWd-HrMWYk" -->
And now, let's train the model!
<!-- #endregion -->

```python id="k3g_AXkIMWYl"
env.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

rewards = [] 
best_score = 0
```

```python id="iAaF43MoMWYm" outputId="efbcef05-3d69-41fb-bff2-a57f7d1e2c4d"
for episode in range(600):
    obs = env.reset()    
    for step in range(200):
        epsilon = max(1 - episode / 500, 0.01)
        obs, reward, done, info = play_one_step(env, obs, epsilon)
        if done:
            break
    rewards.append(step) # Not shown in the book
    if step >= best_score: # Not shown
        best_weights = model.get_weights() # Not shown
        best_score = step # Not shown
    print("\rEpisode: {}, Steps: {}, eps: {:.3f}".format(episode, step + 1, epsilon), end="") # Not shown
    if episode > 50:
        training_step(batch_size)

model.set_weights(best_weights)
```

```python id="RmUAJ3XhMWYn" outputId="faf0f756-221f-49f1-b0db-88381d842059"
plt.figure(figsize=(8, 4))
plt.plot(rewards)
plt.xlabel("Episode", fontsize=14)
plt.ylabel("Sum of rewards", fontsize=14)
save_fig("dqn_rewards_plot")
plt.show()
```

```python id="9lADSKTYMWYn"
env.seed(42)
state = env.reset()

frames = []

for step in range(200):
    action = epsilon_greedy_policy(state)
    state, reward, done, info = env.step(action)
    if done:
        break
    img = env.render(mode="rgb_array")
    frames.append(img)
    
plot_animation(frames)
```

<!-- #region id="bF0SOJW0MWYo" -->
Not bad at all!
<!-- #endregion -->

<!-- #region id="4oHRUnLTMWYp" -->
## Double DQN
<!-- #endregion -->

```python id="OGILQvliMWYp"
keras.backend.clear_session()
tf.random.set_seed(42)
np.random.seed(42)

model = keras.models.Sequential([
    keras.layers.Dense(32, activation="elu", input_shape=[4]),
    keras.layers.Dense(32, activation="elu"),
    keras.layers.Dense(n_outputs)
])

target = keras.models.clone_model(model)
target.set_weights(model.get_weights())
```

```python id="7kMy9DhaMWYq"
batch_size = 32
discount_rate = 0.95
optimizer = keras.optimizers.Adam(lr=6e-3)
loss_fn = keras.losses.Huber()

def training_step(batch_size):
    experiences = sample_experiences(batch_size)
    states, actions, rewards, next_states, dones = experiences
    next_Q_values = model.predict(next_states)
    best_next_actions = np.argmax(next_Q_values, axis=1)
    next_mask = tf.one_hot(best_next_actions, n_outputs).numpy()
    next_best_Q_values = (target.predict(next_states) * next_mask).sum(axis=1)
    target_Q_values = (rewards + 
                       (1 - dones) * discount_rate * next_best_Q_values)
    target_Q_values = target_Q_values.reshape(-1, 1)
    mask = tf.one_hot(actions, n_outputs)
    with tf.GradientTape() as tape:
        all_Q_values = model(states)
        Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
        loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
```

```python id="kKutCf8KMWYr"
replay_memory = deque(maxlen=2000)
```

```python id="wow7DwdrMWYs" outputId="d712fc2d-0de6-4c1a-b59a-098ecea1a604"
env.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

rewards = []
best_score = 0

for episode in range(600):
    obs = env.reset()    
    for step in range(200):
        epsilon = max(1 - episode / 500, 0.01)
        obs, reward, done, info = play_one_step(env, obs, epsilon)
        if done:
            break
    rewards.append(step)
    if step >= best_score:
        best_weights = model.get_weights()
        best_score = step
    print("\rEpisode: {}, Steps: {}, eps: {:.3f}".format(episode, step + 1, epsilon), end="")
    if episode >= 50:
        training_step(batch_size)
        if episode % 50 == 0:
            target.set_weights(model.get_weights())
    # Alternatively, you can do soft updates at each step:
    #if episode >= 50:
        #target_weights = target.get_weights()
        #online_weights = model.get_weights()
        #for index in range(len(target_weights)):
        #    target_weights[index] = 0.99 * target_weights[index] + 0.01 * online_weights[index]
        #target.set_weights(target_weights)

model.set_weights(best_weights)
```

```python id="jYtIf5CFMWYs" outputId="3261ed7b-09c5-45c6-b627-a6e8b35bb090"
plt.figure(figsize=(8, 4))
plt.plot(rewards)
plt.xlabel("Episode", fontsize=14)
plt.ylabel("Sum of rewards", fontsize=14)
save_fig("double_dqn_rewards_plot")
plt.show()
```

```python id="0KOcKyp_MWYt"
env.seed(43)
state = env.reset()

frames = []

for step in range(200):
    action = epsilon_greedy_policy(state)
    state, reward, done, info = env.step(action)
    if done:
        break
    img = env.render(mode="rgb_array")
    frames.append(img)
   
plot_animation(frames)
```

<!-- #region id="eHX8oE_PMWYt" -->
## Dueling Double DQN
<!-- #endregion -->

```python id="C0hl0CeMMWYu"
keras.backend.clear_session()
tf.random.set_seed(42)
np.random.seed(42)

K = keras.backend
input_states = keras.layers.Input(shape=[4])
hidden1 = keras.layers.Dense(32, activation="elu")(input_states)
hidden2 = keras.layers.Dense(32, activation="elu")(hidden1)
state_values = keras.layers.Dense(1)(hidden2)
raw_advantages = keras.layers.Dense(n_outputs)(hidden2)
advantages = raw_advantages - K.max(raw_advantages, axis=1, keepdims=True)
Q_values = state_values + advantages
model = keras.models.Model(inputs=[input_states], outputs=[Q_values])

target = keras.models.clone_model(model)
target.set_weights(model.get_weights())
```

```python id="oEhMeID8MWYv"
batch_size = 32
discount_rate = 0.95
optimizer = keras.optimizers.Adam(lr=7.5e-3)
loss_fn = keras.losses.Huber()

def training_step(batch_size):
    experiences = sample_experiences(batch_size)
    states, actions, rewards, next_states, dones = experiences
    next_Q_values = model.predict(next_states)
    best_next_actions = np.argmax(next_Q_values, axis=1)
    next_mask = tf.one_hot(best_next_actions, n_outputs).numpy()
    next_best_Q_values = (target.predict(next_states) * next_mask).sum(axis=1)
    target_Q_values = (rewards + 
                       (1 - dones) * discount_rate * next_best_Q_values)
    target_Q_values = target_Q_values.reshape(-1, 1)
    mask = tf.one_hot(actions, n_outputs)
    with tf.GradientTape() as tape:
        all_Q_values = model(states)
        Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
        loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
```

```python id="plGGnHXuMWYv"
replay_memory = deque(maxlen=2000)
```

```python id="fyPKD_I0MWYv" outputId="9fa17b2e-6a86-4030-da93-9d60562b8fad"
env.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

rewards = []
best_score = 0

for episode in range(600):
    obs = env.reset()    
    for step in range(200):
        epsilon = max(1 - episode / 500, 0.01)
        obs, reward, done, info = play_one_step(env, obs, epsilon)
        if done:
            break
    rewards.append(step)
    if step >= best_score:
        best_weights = model.get_weights()
        best_score = step
    print("\rEpisode: {}, Steps: {}, eps: {:.3f}".format(episode, step + 1, epsilon), end="")
    if episode >= 50:
        training_step(batch_size)
        if episode % 50 == 0:
            target.set_weights(model.get_weights())

model.set_weights(best_weights)
```

```python id="AOWFV8g5MWYw" outputId="e603db27-3cec-4f81-8ed9-1e524c5489b6"
plt.plot(rewards)
plt.xlabel("Episode")
plt.ylabel("Sum of rewards")
plt.show()
```

```python id="AspVZhkzMWY2"
env.seed(42)
state = env.reset()

frames = []

for step in range(200):
    action = epsilon_greedy_policy(state)
    state, reward, done, info = env.step(action)
    if done:
        break
    img = env.render(mode="rgb_array")
    frames.append(img)
    
plot_animation(frames)
```

<!-- #region id="0eq93Ux-MWY2" -->
This looks like a pretty robust agent!
<!-- #endregion -->

```python id="9So32V-qMWY3"
env.close()
```

<!-- #region id="LQ7s652wMWY3" -->
## Using TF-Agents to Beat Breakout
<!-- #endregion -->

<!-- #region id="54ixeqXSMWY4" -->
Let's use TF-Agents to create an agent that will learn to play Breakout. We will use the Deep Q-Learning algorithm, so you can easily compare the components with the previous implementation, but TF-Agents implements many other (and more sophisticated) algorithms!
<!-- #endregion -->

<!-- #region id="0__41r0EMWY4" -->
### TF-Agents Environments
<!-- #endregion -->

```python id="HWjnGZPqMWY4"
tf.random.set_seed(42)
np.random.seed(42)
```

```python id="XQRBFV9rMWY5" outputId="b3033d56-9548-4945-9a1f-cadcc8569dd3"
from tf_agents.environments import suite_gym

env = suite_gym.load("Breakout-v4")
env
```

```python id="I4kWe0_JMWY6" outputId="86f83219-ca00-4a81-eeae-5b8d398ff614"
env.gym
```

```python id="ZlAouxo3MWY7" outputId="0e8443f8-b90d-416b-f874-e75cd8f148ca"
env.seed(42)
env.reset()
```

```python id="UwpY-xZcMWY8" outputId="214d62d3-80dc-4766-ffa5-1b314ce402ee"
env.step(1) # Fire
```

```python id="nzy8idjgMWY8" outputId="8f6b2cbe-6198-4e8b-b77a-96ff5eecbb8b"
img = env.render(mode="rgb_array")

plt.figure(figsize=(6, 8))
plt.imshow(img)
plt.axis("off")
save_fig("breakout_plot")
plt.show()
```

```python id="nVBtZr-6MWY9" outputId="50129a5e-75b8-467f-f6da-6487221e9432"
env.current_time_step()
```

<!-- #region id="wd9Aj0BuMWY_" -->
### Environment Specifications
<!-- #endregion -->

```python id="KI1tanAuMWZA" outputId="479534f7-0955-4195-ee2c-843172641421"
env.observation_spec()
```

```python id="BLJNOcBIMWZB" outputId="38cf1bdb-03de-4c14-f9df-5ee1464f0301"
env.action_spec()
```

```python id="g375K3ifMWZB" outputId="7d32a06a-7076-4f72-b77d-e22dba289737"
env.time_step_spec()
```

<!-- #region id="JC5uQEY6MWZC" -->
### Environment Wrappers
<!-- #endregion -->

<!-- #region id="idtH6TKpMWZD" -->
You can wrap a TF-Agents environments in a TF-Agents wrapper:
<!-- #endregion -->

```python id="wgkMdIeyMWZD" outputId="a2bc4516-6e06-4c87-b394-1e72218e3480"
from tf_agents.environments.wrappers import ActionRepeat

repeating_env = ActionRepeat(env, times=4)
repeating_env
```

```python id="zCunxhZsMWZD" outputId="1e546793-1e33-4412-b0f9-5988c2ce7ecb"
repeating_env.unwrapped
```

<!-- #region id="prBr_iPWMWZE" -->
Here is the list of available wrappers:
<!-- #endregion -->

```python id="gg0UqPfDMWZE" outputId="077236bb-d277-4cdd-cd8d-7c5d08872742"
import tf_agents.environments.wrappers

for name in dir(tf_agents.environments.wrappers):
    obj = getattr(tf_agents.environments.wrappers, name)
    if hasattr(obj, "__base__") and issubclass(obj, tf_agents.environments.wrappers.PyEnvironmentBaseWrapper):
        print("{:27s} {}".format(name, obj.__doc__.split("\n")[0]))
```

<!-- #region id="y9EyJlaAMWZE" -->
The `suite_gym.load()` function can create an env and wrap it for you, both with TF-Agents environment wrappers and Gym environment wrappers (the latter are applied first).
<!-- #endregion -->

```python id="DoTh1h2NMWZF"
from functools import partial
from gym.wrappers import TimeLimit

limited_repeating_env = suite_gym.load(
    "Breakout-v4",
    gym_env_wrappers=[partial(TimeLimit, max_episode_steps=10000)],
    env_wrappers=[partial(ActionRepeat, times=4)],
)
```

```python id="ex8e1yW3MWZF" outputId="4b10523a-890c-40a0-9b10-baa8ad36bb9b"
limited_repeating_env
```

```python id="5dahdh7TMWZF" outputId="a9847927-763f-4fc8-8b7c-a60173f5ff6c"
limited_repeating_env.unwrapped
```

<!-- #region id="R6WzLhqxMWZG" -->
Create an Atari Breakout environment, and wrap it to apply the default Atari preprocessing steps:
<!-- #endregion -->

<!-- #region id="HWLhesL9MWZG" -->
**Warning**: Breakout requires the player to press the FIRE button at the start of the game and after each life lost. The agent may take a very long time learning this because at first it seems that pressing FIRE just means losing faster. To speed up training considerably, we create and use a subclass of the `AtariPreprocessing` wrapper class called `AtariPreprocessingWithAutoFire` which presses FIRE (i.e., plays action 1) automatically at the start of the game and after each life lost. This is different from the book which uses the regular `AtariPreprocessing` wrapper.
<!-- #endregion -->

```python id="ugF0mrFHMWZG"
from tf_agents.environments import suite_atari
from tf_agents.environments.atari_preprocessing import AtariPreprocessing
from tf_agents.environments.atari_wrappers import FrameStack4

max_episode_steps = 27000 # <=> 108k ALE frames since 1 step = 4 frames
environment_name = "BreakoutNoFrameskip-v4"

class AtariPreprocessingWithAutoFire(AtariPreprocessing):
    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        super().step(1) # FIRE to start
        return obs
    def step(self, action):
        lives_before_action = self.ale.lives()
        obs, rewards, done, info = super().step(action)
        if self.ale.lives() < lives_before_action and not done:
            super().step(1) # FIRE to start after life lost
        return obs, rewards, done, info

env = suite_atari.load(
    environment_name,
    max_episode_steps=max_episode_steps,
    gym_env_wrappers=[AtariPreprocessingWithAutoFire, FrameStack4])
```

```python id="DNqY_tt3MWZH" outputId="3ad21580-6664-48ad-f7bd-4113ce25a2d9"
env
```

<!-- #region id="oWVksdBuMWZH" -->
Play a few steps just to see what happens:
<!-- #endregion -->

```python id="ObFW3SUVMWZI"
env.seed(42)
env.reset()
for _ in range(4):
    time_step = env.step(3) # LEFT
```

```python id="L1yzhZFNMWZI"
def plot_observation(obs):
    # Since there are only 3 color channels, you cannot display 4 frames
    # with one primary color per frame. So this code computes the delta between
    # the current frame and the mean of the other frames, and it adds this delta
    # to the red and blue channels to get a pink color for the current frame.
    obs = obs.astype(np.float32)
    img = obs[..., :3]
    current_frame_delta = np.maximum(obs[..., 3] - obs[..., :3].mean(axis=-1), 0.)
    img[..., 0] += current_frame_delta
    img[..., 2] += current_frame_delta
    img = np.clip(img / 150, 0, 1)
    plt.imshow(img)
    plt.axis("off")
```

```python id="RYw8kRwRMWZI" outputId="8dd2fe10-77c3-4fc1-f0a3-e2c5411302c6"
plt.figure(figsize=(6, 6))
plot_observation(time_step.observation)
save_fig("preprocessed_breakout_plot")
plt.show()
```

<!-- #region id="AkP2OkaVMWZJ" -->
Convert the Python environment to a TF environment:
<!-- #endregion -->

```python id="L8JB1I8mMWZJ"
from tf_agents.environments.tf_py_environment import TFPyEnvironment

tf_env = TFPyEnvironment(env)
```

<!-- #region id="jH8X6YXvMWZJ" -->
### Creating the DQN
<!-- #endregion -->

<!-- #region id="3r1Oua1yMWZK" -->
Create a small class to normalize the observations. Images are stored using bytes from 0 to 255 to use less RAM, but we want to pass floats from 0.0 to 1.0 to the neural network:
<!-- #endregion -->

<!-- #region id="Kf57w308MWZL" -->
Create the Q-Network:
<!-- #endregion -->

```python id="EwZXivlQMWZL"
from tf_agents.networks.q_network import QNetwork

preprocessing_layer = keras.layers.Lambda(
                          lambda obs: tf.cast(obs, np.float32) / 255.)
conv_layer_params=[(32, (8, 8), 4), (64, (4, 4), 2), (64, (3, 3), 1)]
fc_layer_params=[512]

q_net = QNetwork(
    tf_env.observation_spec(),
    tf_env.action_spec(),
    preprocessing_layers=preprocessing_layer,
    conv_layer_params=conv_layer_params,
    fc_layer_params=fc_layer_params)
```

<!-- #region id="psfhTp25MWZQ" -->
Create the DQN Agent:
<!-- #endregion -->

```python id="bWimhBW8MWZR"
from tf_agents.agents.dqn.dqn_agent import DqnAgent

train_step = tf.Variable(0)
update_period = 4 # run a training step every 4 collect steps
optimizer = keras.optimizers.RMSprop(lr=2.5e-4, rho=0.95, momentum=0.0,
                                     epsilon=0.00001, centered=True)
epsilon_fn = keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=1.0, # initial ε
    decay_steps=250000 // update_period, # <=> 1,000,000 ALE frames
    end_learning_rate=0.01) # final ε
agent = DqnAgent(tf_env.time_step_spec(),
                 tf_env.action_spec(),
                 q_network=q_net,
                 optimizer=optimizer,
                 target_update_period=2000, # <=> 32,000 ALE frames
                 td_errors_loss_fn=keras.losses.Huber(reduction="none"),
                 gamma=0.99, # discount factor
                 train_step_counter=train_step,
                 epsilon_greedy=lambda: epsilon_fn(train_step))
agent.initialize()
```

<!-- #region id="jR_moVJoMWZS" -->
Create the replay buffer (this will use a lot of RAM, so please reduce the buffer size if you get an out-of-memory error):
<!-- #endregion -->

<!-- #region id="VuKRujqqMWZT" -->
**Warning**: we use a replay buffer of size 100,000 instead of 1,000,000 (as used in the book) since many people were getting OOM (Out-Of-Memory) errors.
<!-- #endregion -->

```python id="I1Yip0qUMWZU"
from tf_agents.replay_buffers import tf_uniform_replay_buffer

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=tf_env.batch_size,
    max_length=100000) # reduce if OOM error

replay_buffer_observer = replay_buffer.add_batch
```

<!-- #region id="TFpyAYlWMWZV" -->
Create a simple custom observer that counts and displays the number of times it is called (except when it is passed a trajectory that represents the boundary between two episodes, as this does not count as a step):
<!-- #endregion -->

```python id="9ibaw-GdMWZV"
class ShowProgress:
    def __init__(self, total):
        self.counter = 0
        self.total = total
    def __call__(self, trajectory):
        if not trajectory.is_boundary():
            self.counter += 1
        if self.counter % 100 == 0:
            print("\r{}/{}".format(self.counter, self.total), end="")
```

<!-- #region id="iFlRuIpNMWZW" -->
Let's add some training metrics:
<!-- #endregion -->

```python id="rfa0FdK_MWZW"
from tf_agents.metrics import tf_metrics

train_metrics = [
    tf_metrics.NumberOfEpisodes(),
    tf_metrics.EnvironmentSteps(),
    tf_metrics.AverageReturnMetric(),
    tf_metrics.AverageEpisodeLengthMetric(),
]
```

```python id="rOlvPE99MWZX" outputId="9e5de291-5240-4942-9272-33e8607a13c4"
train_metrics[0].result()
```

```python id="VYk5_HcUMWZY" outputId="b3a86e17-08f4-4408-96c3-20f34662ac51"
from tf_agents.eval.metric_utils import log_metrics
import logging
logging.getLogger().setLevel(logging.INFO)
log_metrics(train_metrics)
```

<!-- #region id="dBHfz0aJMWZY" -->
Create the collect driver:
<!-- #endregion -->

```python id="0rxukUfqMWZZ"
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver

collect_driver = DynamicStepDriver(
    tf_env,
    agent.collect_policy,
    observers=[replay_buffer_observer] + train_metrics,
    num_steps=update_period) # collect 4 steps for each training iteration
```

<!-- #region id="1fqEFudBMWZa" -->
Collect the initial experiences, before training:
<!-- #endregion -->

```python id="svNcnhBLMWZb" outputId="ca1d8f56-a53a-4f7c-a155-bd7686f9ef62"
from tf_agents.policies.random_tf_policy import RandomTFPolicy

initial_collect_policy = RandomTFPolicy(tf_env.time_step_spec(),
                                        tf_env.action_spec())
init_driver = DynamicStepDriver(
    tf_env,
    initial_collect_policy,
    observers=[replay_buffer.add_batch, ShowProgress(20000)],
    num_steps=20000) # <=> 80,000 ALE frames
final_time_step, final_policy_state = init_driver.run()
```

<!-- #region id="fVUQVdVgMWZb" -->
Let's sample 2 sub-episodes, with 3 time steps each and display them:
<!-- #endregion -->

<!-- #region id="jNuwfAzQMWZb" -->
**Note**: `replay_buffer.get_next()` is deprecated. We must use `replay_buffer.as_dataset(..., single_deterministic_pass=False)` instead.
<!-- #endregion -->

```python id="X2HtJDsOMWZc"
tf.random.set_seed(9) # chosen to show an example of trajectory at the end of an episode

#trajectories, buffer_info = replay_buffer.get_next( # get_next() is deprecated
#    sample_batch_size=2, num_steps=3)

trajectories, buffer_info = next(iter(replay_buffer.as_dataset(
    sample_batch_size=2,
    num_steps=3,
    single_deterministic_pass=False)))
```

```python id="1FoNe4a_MWZc" outputId="3f5721d5-e8ea-42a6-d0ce-311abfd707ac"
trajectories._fields
```

```python id="2Ymth1VQMWZc" outputId="959abb33-d4a7-4972-a850-43f85c2ea418"
trajectories.observation.shape
```

```python id="5gHw1vL5MWZd" outputId="9e1720e7-de5d-44a8-de9a-f8698049c6d0"
from tf_agents.trajectories.trajectory import to_transition

time_steps, action_steps, next_time_steps = to_transition(trajectories)
time_steps.observation.shape
```

```python id="dU6r4uXaMWZd" outputId="b6bb242b-243b-4a2c-d192-23e3fd3655c9"
trajectories.step_type.numpy()
```

```python id="DgElX-F4MWZd" outputId="9f21125d-ccd4-4df4-a89d-84f1884fd0b0"
plt.figure(figsize=(10, 6.8))
for row in range(2):
    for col in range(3):
        plt.subplot(2, 3, row * 3 + col + 1)
        plot_observation(trajectories.observation[row, col].numpy())
plt.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0, wspace=0.02)
save_fig("sub_episodes_plot")
plt.show()
```

<!-- #region id="FQuATIEMMWZe" -->
Now let's create the dataset:
<!-- #endregion -->

```python id="qp2tH1VVMWZe"
dataset = replay_buffer.as_dataset(
    sample_batch_size=64,
    num_steps=2,
    num_parallel_calls=3).prefetch(3)
```

<!-- #region id="6KKfUkiXMWZf" -->
Convert the main functions to TF Functions for better performance:
<!-- #endregion -->

```python id="BOGNG66RMWZj"
from tf_agents.utils.common import function

collect_driver.run = function(collect_driver.run)
agent.train = function(agent.train)
```

<!-- #region id="1u87i6-GMWZk" -->
And now we are ready to run the main loop!
<!-- #endregion -->

```python id="z2EH-1DpMWZl"
def train_agent(n_iterations):
    time_step = None
    policy_state = agent.collect_policy.get_initial_state(tf_env.batch_size)
    iterator = iter(dataset)
    for iteration in range(n_iterations):
        time_step, policy_state = collect_driver.run(time_step, policy_state)
        trajectories, buffer_info = next(iterator)
        train_loss = agent.train(trajectories)
        print("\r{} loss:{:.5f}".format(
            iteration, train_loss.loss.numpy()), end="")
        if iteration % 1000 == 0:
            log_metrics(train_metrics)
```

<!-- #region id="P78FiHmIMWZm" -->
Run the next cell to train the agent for 50,000 steps. Then look at its behavior by running the following cell. You can run these two cells as many times as you wish. The agent will keep improving! It will likely take over 200,000 iterations for the agent to become reasonably good.
<!-- #endregion -->

```python id="dOA70ozDMWZm" outputId="3e45c3e2-0730-421c-dd62-3e201471989f"
train_agent(n_iterations=50000)
```

```python id="deGYIZxRMWZo"
frames = []
def save_frames(trajectory):
    global frames
    frames.append(tf_env.pyenv.envs[0].render(mode="rgb_array"))

watch_driver = DynamicStepDriver(
    tf_env,
    agent.policy,
    observers=[save_frames, ShowProgress(1000)],
    num_steps=1000)
final_time_step, final_policy_state = watch_driver.run()

plot_animation(frames)
```

<!-- #region id="2JLyjzdPMWZp" -->
If you want to save an animated GIF to show off your agent to your friends, here's one way to do it:
<!-- #endregion -->

```python id="UHyM5n5OMWZq"
import PIL

image_path = os.path.join("images", "rl", "breakout.gif")
frame_images = [PIL.Image.fromarray(frame) for frame in frames[:150]]
frame_images[0].save(image_path, format='GIF',
                     append_images=frame_images[1:],
                     save_all=True,
                     duration=30,
                     loop=0)
```

```html id="sKIWZsRuMWZr" outputId="30be9f8c-0af6-40c8-8b5d-c90938d80e1b"
<img src="images/rl/breakout.gif" />
```

<!-- #region id="16sodmmrMWZt" -->
## Extra material
<!-- #endregion -->

<!-- #region id="TofmazI2MWZu" -->
### Deque vs Rotating List
<!-- #endregion -->

<!-- #region id="Pvtip1YTMWZu" -->
The `deque` class offers fast append, but fairly slow random access (for large replay memories):
<!-- #endregion -->

```python id="q69a10MKMWZu" outputId="8a5b05f6-f0a8-4884-cb30-f5d1595ee9cf"
from collections import deque
np.random.seed(42)

mem = deque(maxlen=1000000)
for i in range(1000000):
    mem.append(i)
[mem[i] for i in np.random.randint(1000000, size=5)]
```

```python id="mNc8Qjk-MWZv" outputId="be0b93b1-323a-4b7c-aca6-d1ab0e4fa3be"
%timeit mem.append(1)
```

```python id="sBv2ssOXMWZw" outputId="b8e0efaf-c5f5-42d8-80ea-56086d7a37f6"
%timeit [mem[i] for i in np.random.randint(1000000, size=5)]
```

<!-- #region id="e5v1_hPaMWZx" -->
Alternatively, you could use a rotating list like this `ReplayMemory` class. This would make random access faster for large replay memories:
<!-- #endregion -->

```python id="fIhRe0w1MWZ2"
class ReplayMemory:
    def __init__(self, max_size):
        self.buffer = np.empty(max_size, dtype=np.object)
        self.max_size = max_size
        self.index = 0
        self.size = 0

    def append(self, obj):
        self.buffer[self.index] = obj
        self.size = min(self.size + 1, self.max_size)
        self.index = (self.index + 1) % self.max_size

    def sample(self, batch_size):
        indices = np.random.randint(self.size, size=batch_size)
        return self.buffer[indices]
```

```python id="oqJb2o4sMWZ3" outputId="f4898ce6-e5f4-4202-d781-173a2cbd17ab"
mem = ReplayMemory(max_size=1000000)
for i in range(1000000):
    mem.append(i)
mem.sample(5)
```

```python id="WgPb8A05MWZ4" outputId="bb719d79-88bb-439e-de0c-55d85a7ba2ca"
%timeit mem.append(1)
```

```python id="AfnCeIR1MWZ4" outputId="c4f22a55-165e-41b6-fb22-c88cb46515e0"
%timeit mem.sample(5)
```

<!-- #region id="8fJY6PRXMWZ5" -->
### Creating a Custom TF-Agents Environment
<!-- #endregion -->

<!-- #region id="k5gbx3esMWZ6" -->
To create a custom TF-Agent environment, you just need to write a class that inherits from the `PyEnvironment` class and implements a few methods. For example, the following minimal environment represents a simple 4x4 grid. The agent starts in one corner (0,0) and must move to the opposite corner (3,3). The episode is done if the agent reaches the goal (it gets a +10 reward) or if the agent goes out of bounds (-1 reward). The actions are up (0), down (1), left (2) and right (3).
<!-- #endregion -->

```python id="xHGOqsL4MWZ6"
class MyEnvironment(tf_agents.environments.py_environment.PyEnvironment):
    def __init__(self, discount=1.0):
        super().__init__()
        self._action_spec = tf_agents.specs.BoundedArraySpec(
            shape=(), dtype=np.int32, name="action", minimum=0, maximum=3)
        self._observation_spec = tf_agents.specs.BoundedArraySpec(
            shape=(4, 4), dtype=np.int32, name="observation", minimum=0, maximum=1)
        self.discount = discount

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._state = np.zeros(2, dtype=np.int32)
        obs = np.zeros((4, 4), dtype=np.int32)
        obs[self._state[0], self._state[1]] = 1
        return tf_agents.trajectories.time_step.restart(obs)

    def _step(self, action):
        self._state += [(-1, 0), (+1, 0), (0, -1), (0, +1)][action]
        reward = 0
        obs = np.zeros((4, 4), dtype=np.int32)
        done = (self._state.min() < 0 or self._state.max() > 3)
        if not done:
            obs[self._state[0], self._state[1]] = 1
        if done or np.all(self._state == np.array([3, 3])):
            reward = -1 if done else +10
            return tf_agents.trajectories.time_step.termination(obs, reward)
        else:
            return tf_agents.trajectories.time_step.transition(obs, reward,
                                                               self.discount)
```

<!-- #region id="-4lxfeqnMWZ7" -->
The action and observation specs will generally be instances of the `ArraySpec` or `BoundedArraySpec` classes from the `tf_agents.specs` package (check out the other specs in this package as well). Optionally, you can also define a `render()` method, a `close()` method to free resources, as well as a `time_step_spec()` method if you don't want the `reward` and `discount` to be 32-bit float scalars. Note that the base class takes care of keeping track of the current time step, which is why we must implement `_reset()` and `_step()` rather than `reset()` and `step()`.

<!-- #endregion -->

```python id="JNWgV_JRMWZ7" outputId="9bb9ba57-7c6b-48d4-e2d2-107ea484d3ab"
my_env = MyEnvironment()
time_step = my_env.reset()
time_step
```

```python id="DwaAnKgHMWZ7" outputId="7a805a6d-cf58-402e-e949-2e18c237e2c1"
time_step = my_env.step(1)
time_step
```
