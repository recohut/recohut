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

<!-- #region id="Ga2R-fVJQ4p6" -->
# Q-Learning on Lunar Lander and Frozen Lake
<!-- #endregion -->

<!-- #region id="2uAdo5YZUN1g" -->
## Frozen Lake
<!-- #endregion -->

```python id="4nFezzYzUNy-"
import gym
import numpy as np
import time
from IPython.display import clear_output
```

```python colab={"base_uri": "https://localhost:8080/"} id="MHpSlRsyUNvV" executionInfo={"status": "ok", "timestamp": 1634743875574, "user_tz": -330, "elapsed": 505, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="0cab3ff2-df5c-4709-ca0d-b02d079e53b9"
env = gym.make('FrozenLake-v0')
env.render()
```

```python colab={"base_uri": "https://localhost:8080/"} id="dgxb02sDUiXb" executionInfo={"status": "ok", "timestamp": 1634743876296, "user_tz": -330, "elapsed": 6, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="5c0c2458-498d-428c-8508-fcafbea3ac75"
numActions = env.action_space.n
numStates = env.observation_space.n

print(numActions,numStates)
```

```python colab={"base_uri": "https://localhost:8080/"} id="avQxrEjNUlmu" executionInfo={"status": "ok", "timestamp": 1634743878519, "user_tz": -330, "elapsed": 8, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="6b3ee2c8-5567-4744-d0b2-6fc0b27050f2"
Q = np.zeros((numStates,numActions))

print(Q)
```

```python id="ppe2zBTUUq4N"
nE = 10000
mpE = 80
alpha = 0.01
gamma = 0.99
epsilon = 1
edr = 0.0001
```

```python id="A0vUS0q1Uq2M"
for e in range(nE):
    state = env.reset()
    done = False
    for step in range(mpE):
        if np.random.rand() > epsilon:
            action = np.argmax(Q[state,:])
        else:
            action = env.action_space.sample()
        ns,reward,done,info = env.step(action)
        Q[state,action] = (1-alpha)*Q[state,action]+alpha*(
        reward+gamma*np.max(Q[ns,:]))
        state = ns
        if done == True:
            break
    epsilon = epsilon*np.exp(-edr*e)
```

```python colab={"base_uri": "https://localhost:8080/"} id="M0R2dYFKUqx-" executionInfo={"status": "ok", "timestamp": 1634743906461, "user_tz": -330, "elapsed": 10, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="22323285-2d31-44bf-ea59-67eb513b258e"
Q
```

```python colab={"base_uri": "https://localhost:8080/"} id="H3mM8oMKUqu3" executionInfo={"status": "ok", "timestamp": 1634743959042, "user_tz": -330, "elapsed": 28438, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="da285b28-bdbd-4e2e-cff2-cf672f328bc6"
for e in range(4):
    state = env.reset()
    done = False
    print("Episode::",e)
    time.sleep(0.1)
    for step in range(mpE):
        clear_output(wait=True)
        env.render()
        time.sleep(0.3)
        action = np.argmax(Q[state,:])
        ns,reward,done,info = env.step(action)
        if done:
            clear_output(wait=True)
            env.render()
            if reward == 1:
                print("Goal")
                time.sleep(3)
            else:
                print("Hole")
                time.sleep(3)
                clear_output(wait=True)
            break
        state = ns
env.close()
```

<!-- #region id="3Evf6amNULMb" -->
## Lunar Lander
<!-- #endregion -->

```python id="y5j35OgFQ36B"
!pip install box2d-py
```

```python colab={"base_uri": "https://localhost:8080/"} id="6FlXsfAStNyj" executionInfo={"status": "ok", "timestamp": 1634742990996, "user_tz": -330, "elapsed": 30056, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="111f7075-3939-410f-a8ae-418788bdcc6f"
import gym

env = gym.make('LunarLander-v2')

# Nop, fire left engine, main engine, right engine
ACTIONS = env.action_space.n

# Landing pad is always at coordinates (0,0). Coordinates are the first two numbers in state vector.
# Reward for moving from the top of the screen to landing pad and zero speed is about 100..140 points.
# If lander moves away from landing pad it loses reward back. Episode finishes if the lander crashes or
# comes to rest, receiving additional -100 or +100 points. Each leg ground contact is +10. Firing main
# engine is -0.3 points each frame. Solved is 200 points.

import numpy as np
import random

def discretize_state(state):
    dstate = list(state[:5])
    dstate[0] = int(0.5*(state[0]+0.7)*10/2.0) # pos x
    dstate[1] = int(0.5*(state[1]+0.5)*10/2.0) # pos y
    dstate[2] = int(0.5*(state[2]+1.5)*10/3.0) # vel x
    dstate[3] = int(0.5*(state[3]+2)*10/3.0) # vel y
    dstate[4] = int(0.5*(state[4]+3.14159)*10/(2*3.14159)) # angle
    if dstate[0] >= 5: dstate[0] = 4
    if dstate[1] >= 5: dstate[1] = 4
    if dstate[2] >= 5: dstate[2] = 4
    if dstate[3] >= 5: dstate[3] = 4
    if dstate[4] >= 5: dstate[4] = 4
    if dstate[0] < 0: dstate[0] = 0
    if dstate[1] < 0: dstate[1] = 0
    if dstate[2] < 0: dstate[2] = 0
    if dstate[3] < 0: dstate[3] = 0
    if dstate[4] < 0: dstate[4] = 0
    return tuple(dstate)

def run(num_episodes, alpha, gamma, explore_mult):
    max_rewards = []
    last_reward = []
    qtable = np.subtract(np.zeros((5, 5, 5, 5, 5, ACTIONS)), 100) # start all rewards at -100
    explore_rate = 1.0
    for episode in range(num_episodes):
        s = env.reset()
        state = discretize_state(s)
        
        for step in range(10000):

            # select action
            if random.random() < explore_rate:
                action = random.choice(range(ACTIONS))
            else:
                action = np.argmax(qtable[state])

            (new_s, reward, done, _) = env.step(action)
            new_state = discretize_state(new_s)

            # update Q
            best_future_q = np.amax(qtable[new_state]) # returns best possible reward from next state
            prior_val = qtable[state + (action,)]
            qtable[state + (action,)] = (1.0-alpha)*prior_val + alpha*(reward + gamma * best_future_q)
            state = new_state
            
            if done or step == 9999:
                last_reward.append(reward)
                break
        
        if explore_rate > 0.01:
            explore_rate *= explore_mult    
        max_rewards.append(np.amax(qtable))
        
    return (max_rewards, last_reward[-50:], qtable) # return rewards from last 50 episodes


num_episodes = 100
for alpha in [0.05, 0.10, 0.15]:
    for gamma in [0.85, 0.90, 0.95]:
        (max_rewards, last_reward, _) = run(num_episodes=num_episodes, alpha=alpha, gamma=gamma, explore_mult=0.995)
        print("alpha = %.2f, gamma = %.2f, mean last 50 outcomes = %.2f, q max: %.2f, q mean: %.2f" % (alpha, gamma, np.mean(last_reward), np.max(max_rewards), np.mean(max_rewards)))

(max_rewards, last_reward, qtable) = run(num_episodes=200, alpha=0.1, gamma=0.95, explore_mult=0.995)
print("mean last 50 outcomes = %.2f, q max: %.2f, q mean: %.2f" % (np.mean(last_reward), np.max(max_rewards), np.mean(max_rewards)))
np.save('qtable.npy', qtable)
```

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

```python id="0NFnUojZQzxH"
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

```python colab={"base_uri": "https://localhost:8080/", "height": 421} id="2aapiC-KQpyv" executionInfo={"status": "ok", "timestamp": 1634743400926, "user_tz": -330, "elapsed": 56012, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="acf93825-6553-440f-f01b-28a82d0b7c55"
# Use best qtable to play the game (no learning anymore)
import gym
import numpy as np

env = wrap_env(gym.make('LunarLander-v2'))
qtable = np.load('qtable.npy')

for i in range(100):
    s = env.reset()
    state = discretize_state(s)
    for step in range(10000):
        env.render()

        # select action
        action = np.argmax(qtable[state])

        (new_s, reward, done, _) = env.step(action)
        new_state = discretize_state(new_s)

        if done or step == 9999:
            break

        state = new_state
            
env.close()
show_video()
```
