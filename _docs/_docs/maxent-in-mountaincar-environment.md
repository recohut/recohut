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

<!-- #region id="6pDUf9bZc7JG" -->
# MaxEnt in Mountaincar Environment
<!-- #endregion -->

<!-- #region id="pV5flWpeVM6M" -->
## Setup
<!-- #endregion -->

<!-- #region id="VdBAv8vWVM36" -->
### Installations
<!-- #endregion -->

```python id="y1GyS_RCTdvg"
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

```python id="CfrWqvJJS0I8" executionInfo={"status": "ok", "timestamp": 1636605713248, "user_tz": -330, "elapsed": 3531, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
!pip install -q gym
!pip install -q pylab-sdk
!pip install -q readchar
```

<!-- #region id="7ZVxXGqdVLeL" -->
### Imports
<!-- #endregion -->

```python id="gXzsjx1rS4m5" executionInfo={"status": "ok", "timestamp": 1636606311056, "user_tz": -330, "elapsed": 635, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
import gym
import matplotlib.pyplot as plt
import readchar
import numpy as np
```

<!-- #region id="G1lcBr_bVR6g" -->
### Gym render
<!-- #endregion -->

```python id="8FiXt334VV3l"
from gym.wrappers import Monitor
import glob
import io
import base64
from IPython.display import HTML
from pyvirtualdisplay import Display
from IPython import display as ipythondisplay
```

```python id="rsZZr8XaTdvh" executionInfo={"status": "ok", "timestamp": 1636605860339, "user_tz": -330, "elapsed": 1450, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
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

<!-- #region id="XH0uQ0dGVJ-8" -->
### Params
<!-- #endregion -->

```python id="S7Ut5RnVTZwG" executionInfo={"status": "ok", "timestamp": 1636605860340, "user_tz": -330, "elapsed": 5, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
# MACROS
Push_Left = 0
No_Push = 1
Push_Right = 2
```

```python id="kYjjkkgWTYxO" executionInfo={"status": "ok", "timestamp": 1636605861030, "user_tz": -330, "elapsed": 4, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
# Key mapping
arrow_keys = {
    '\x1b[D': Push_Left,
    '\x1b[B': No_Push,
    '\x1b[C': Push_Right}
```

```python id="MTzgzJ-SUOt8" executionInfo={"status": "ok", "timestamp": 1636605950406, "user_tz": -330, "elapsed": 642, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
n_states = 400 # position - 20, velocity - 20
n_actions = 3
one_feature = 20 # number of state per one feature
q_table = np.zeros((n_states, n_actions)) # (400, 3)
feature_matrix = np.eye((n_states)) # (400, 400)

gamma = 0.99
q_learning_rate = 0.03
theta_learning_rate = 0.05

np.random.seed(1)
```

<!-- #region id="Dr1OTNgjVHV4" -->
## Expert Demo
<!-- #endregion -->

```python id="r7hM2cStTyjf"
# env = wrap_env(gym.make("MountainCar-v0"))

# trajectories = []
# episode_step = 0

# for episode in range(20): # n_trajectories : 20
#     trajectory = []
#     step = 0

#     env.reset()
#     print("episode_step", episode_step)

#     while True: 
#         env.render()
#         print("step", step)

#         key = readchar.readkey()
#         if key not in arrow_keys.keys():
#             break

#         action = arrow_keys[key]
#         state, reward, done, _ = env.step(action)

#         if state[0] >= env.env.goal_position and step > 129: # trajectory_length : 130
#             break

#         trajectory.append((state[0], state[1], action))
#         step += 1

#     trajectory_numpy = np.array(trajectory, float)
#     print("trajectory_numpy.shape", trajectory_numpy.shape)
#     episode_step += 1
#     trajectories.append(trajectory)

# np_trajectories = np.array(trajectories, float)
# print("np_trajectories.shape", np_trajectories.shape)

# np.save("expert_trajectories", arr=np_trajectories)
```

```python colab={"base_uri": "https://localhost:8080/"} id="YFF2azikVZi9" executionInfo={"status": "ok", "timestamp": 1636606274491, "user_tz": -330, "elapsed": 1179, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="4ce3af7c-ae1e-4cef-c688-283189b7abf5"
!wget -q --show-progress https://github.com/reinforcement-learning-kr/lets-do-irl/raw/master/mountaincar/maxent/expert_demo/expert_demo.npy
```

<!-- #region id="UzzcDLaaWB0m" -->
## IRL MaxEnt Training
<!-- #endregion -->

```python id="HLB4YfxVSxEp" executionInfo={"status": "ok", "timestamp": 1636605861031, "user_tz": -330, "elapsed": 4, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
def get_reward(feature_matrix, theta, n_states, state_idx):
    irl_rewards = feature_matrix.dot(theta).reshape((n_states,))
    return irl_rewards[state_idx]

def expert_feature_expectations(feature_matrix, demonstrations):
    feature_expectations = np.zeros(feature_matrix.shape[0])
    
    for demonstration in demonstrations:
        for state_idx, _, _ in demonstration:
            feature_expectations += feature_matrix[int(state_idx)]

    feature_expectations /= demonstrations.shape[0]
    return feature_expectations

def maxent_irl(expert, learner, theta, learning_rate):
    gradient = expert - learner
    theta += learning_rate * gradient

    # Clip theta
    for j in range(len(theta)):
        if theta[j] > 0:
            theta[j] = 0
```

```python id="9ZV3WiJgUV-r" executionInfo={"status": "ok", "timestamp": 1636606284251, "user_tz": -330, "elapsed": 456, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
def idx_demo(env, one_feature):
    env_low = env.observation_space.low     
    env_high = env.observation_space.high   
    env_distance = (env_high - env_low) / one_feature  

    raw_demo = np.load(file="expert_demo.npy")
    demonstrations = np.zeros((len(raw_demo), len(raw_demo[0]), 3))

    for x in range(len(raw_demo)):
        for y in range(len(raw_demo[0])):
            position_idx = int((raw_demo[x][y][0] - env_low[0]) / env_distance[0])
            velocity_idx = int((raw_demo[x][y][1] - env_low[1]) / env_distance[1])
            state_idx = position_idx + velocity_idx * one_feature

            demonstrations[x][y][0] = state_idx
            demonstrations[x][y][1] = raw_demo[x][y][2] 
            
    return demonstrations

def idx_state(env, state):
    env_low = env.observation_space.low
    env_high = env.observation_space.high 
    env_distance = (env_high - env_low) / one_feature 
    position_idx = int((state[0] - env_low[0]) / env_distance[0])
    velocity_idx = int((state[1] - env_low[1]) / env_distance[1])
    state_idx = position_idx + velocity_idx * one_feature
    return state_idx

def update_q_table(state, action, reward, next_state):
    q_1 = q_table[state][action]
    q_2 = reward + gamma * max(q_table[next_state])
    q_table[state][action] += q_learning_rate * (q_2 - q_1)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 776} id="mC_jI_fqTyEY" executionInfo={"status": "ok", "timestamp": 1636607561058, "user_tz": -330, "elapsed": 1189423, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="f644ca3d-1fd6-4f1d-c532-08e5101ab168"
env = wrap_env(gym.make("MountainCar-v0"))

demonstrations = idx_demo(env, one_feature)

expert = expert_feature_expectations(feature_matrix, demonstrations)
learner_feature_expectations = np.zeros(n_states)

theta = -(np.random.uniform(size=(n_states,)))

episodes, scores = [], []

for episode in range(30000):
    state = env.reset()
    score = 0

    if (episode != 0 and episode == 10000) or (episode > 10000 and episode % 5000 == 0):
        learner = learner_feature_expectations / episode
        maxent_irl(expert, learner, theta, theta_learning_rate)
            
    while True:
        state_idx = idx_state(env, state)
        action = np.argmax(q_table[state_idx])
        next_state, reward, done, _ = env.step(action)
        
        irl_reward = get_reward(feature_matrix, theta, n_states, state_idx)
        next_state_idx = idx_state(env, next_state)
        update_q_table(state_idx, action, irl_reward, next_state_idx)
        
        learner_feature_expectations += feature_matrix[int(state_idx)]

        score += reward
        state = next_state
        
        if done:
            scores.append(score)
            episodes.append(episode)
            break

    if episode % 1000 == 0:
        score_avg = np.mean(scores)
        print('{} episode score is {:.2f}'.format(episode, score_avg))
        plt.plot(episodes, scores, 'b')
        np.save("maxent_q_table", arr=q_table)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 421} id="GhJTXT-OcYTr" executionInfo={"status": "ok", "timestamp": 1636608086496, "user_tz": -330, "elapsed": 814, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="53877214-b636-43f5-f642-930fcf54dd3c"
show_video()
```

<!-- #region id="bCWHIAAgTyg7" -->
## Test
<!-- #endregion -->

```python id="kkNGrXpMTye0" executionInfo={"status": "ok", "timestamp": 1636608162802, "user_tz": -330, "elapsed": 498, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
q_table = np.load(file="maxent_q_table.npy") # (400, 3)
one_feature = 20 # number of state per one feature
```

```python id="2waxDvthWjXi" executionInfo={"status": "ok", "timestamp": 1636608166093, "user_tz": -330, "elapsed": 456, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
def idx_to_state(env, state):
    """ Convert pos and vel about mounting car environment to the integer value"""
    env_low = env.observation_space.low
    env_high = env.observation_space.high 
    env_distance = (env_high - env_low) / one_feature 
    position_idx = int((state[0] - env_low[0]) / env_distance[0])
    velocity_idx = int((state[1] - env_low[1]) / env_distance[1])
    state_idx = position_idx + velocity_idx * one_feature
    return state_idx
```

```python colab={"base_uri": "https://localhost:8080/"} id="OMy7a8KwWkjr" executionInfo={"status": "ok", "timestamp": 1636608182257, "user_tz": -330, "elapsed": 9749, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="54dd859e-d2ce-4a28-f34c-788a974b0b84"
env = wrap_env(gym.make("MountainCar-v0"))

episodes, scores = [], []

for episode in range(10):
    state = env.reset()
    score = 0

    while True:
        env.render()
        state_idx = idx_to_state(env, state)
        action = np.argmax(q_table[state_idx])
        next_state, reward, done, _ = env.step(action)
        
        score += reward
        state = next_state
        
        if done:
            scores.append(score)
            episodes.append(episode)
            # pylab.plot(episodes, scores, 'b')
            # pylab.savefig("maxent_test.png")
            break

    if episode % 1 == 0:
        print('{} episode score is {:.2f}'.format(episode, score))
```

```python colab={"base_uri": "https://localhost:8080/", "height": 421} id="dajTkpC-Tdvi" executionInfo={"status": "ok", "timestamp": 1636608185805, "user_tz": -330, "elapsed": 781, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="e8063e00-b61a-4440-aec2-4e87d519927c"
show_video()
```
