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

<!-- #region id="Q47M1niZgJON" -->
# Batch-Constrained Deep Q-Learning
<!-- #endregion -->

<!-- #region id="YT1iD-K6gXqq" -->
Current off-policy deep reinforcement learning algorithms fail to address extrapolation error by selecting actions with respect to a learned value estimate, without consideration of the accuracy of the estimate. As a result, certain outof-distribution actions can be erroneously extrapolated to higher values. However, the value of an off-policy agent can be accurately evaluated in regions where data is available. 

Batch-Constrained deep Q-learning (BCQ), uses a state-conditioned generative model to produce only previously seen actions. This generative model is combined with a Q-network, to select the highest valued action which is similar to the data in the batch. Unlike any previous continuous control deep reinforcement learning algorithms, BCQ is able to learn successfully without interacting with the environment by considering extrapolation error.

BCQ is based on a simple idea: to avoid extrapolation error a policy should induce a similar state-action visitation to the batch. We denote policies which satisfy this notion as batch-constrained. To optimize off-policy learning for a given batch, batch-constrained policies are trained to select actions with respect to three objectives:

1. Minimize the distance of selected actions to the data in the batch.
2. Lead to states where familiar data can be observed.
3. Maximize the value function.
<!-- #endregion -->

<!-- #region id="TWKqz5cofdJg" -->
## Setup
<!-- #endregion -->

```python id="VpFbnCh_uOd5"
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

<!-- #region id="UJa3NdKyvorg" -->
Restart the runtime. Required.
<!-- #endregion -->

```python id="QvAdeGPduQ0a"
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

```python id="Zk4AGo6tp70a"
import cv2
import gym
import numpy as np
import torch
import importlib
import json
import os

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
```

<!-- #region id="ePikuiYJq8tq" -->
## Params
<!-- #endregion -->

```python id="mHf8KoElq97f"
class Args:

    # env = "PongNoFrameskip-v0" # OpenAI gym environment name
    env = "CartPole-v0" # OpenAI gym environment name
    seed = 0 # Sets Gym, PyTorch and Numpy seeds
    buffer_name = "Default" # Prepends name to filename
    max_timesteps = 1e4 # Max time steps to run environment or train for
    BCQ_threshold = 0.3 # Threshold hyper-parameter for BCQ
    low_noise_p = 0.2 # Probability of a low noise episode when generating buffer
    rand_action_p = 0.2 # Probability of taking a random action when generating buffer, during non-low noise episode

    # Atari Specific
    atari_preprocessing = {
        "frame_skip": 4,
        "frame_size": 84,
        "state_history": 4,
        "done_on_life_loss": False,
        "reward_clipping": True,
        "max_episode_timesteps": 27e3
    }
    
    atari_parameters = {
		# Exploration
		"start_timesteps": 2e4,
		"initial_eps": 1,
		"end_eps": 1e-2,
		"eps_decay_period": 25e4,
		# Evaluation
		"eval_freq": 5e4,
		"eval_eps": 1e-3,
		# Learning
		"discount": 0.99,
		"buffer_size": 1e6,
		"batch_size": 32,
		"optimizer": "Adam",
		"optimizer_parameters": {
			"lr": 0.0000625,
			"eps": 0.00015
		},
		"train_freq": 4,
		"polyak_target_update": False,
		"target_update_freq": 8e3,
		"tau": 1
	}
    
    regular_parameters = {
		# Exploration
		"start_timesteps": 1e3,
		"initial_eps": 0.1,
		"end_eps": 0.1,
		"eps_decay_period": 1,
		# Evaluation
		"eval_freq": 5e3,
		"eval_eps": 0,
		# Learning
		"discount": 0.99,
		"buffer_size": 1e6,
		"batch_size": 64,
		"optimizer": "Adam",
		"optimizer_parameters": {
			"lr": 3e-4
		},
		"train_freq": 1,
		"polyak_target_update": True,
		"target_update_freq": 1,
		"tau": 0.005
	}


args = Args()
```

```python id="qiYgVCp7rPGb"
if not os.path.exists("./results"):
    os.makedirs("./results")

if not os.path.exists("./models"):
    os.makedirs("./models")

if not os.path.exists("./buffers"):
    os.makedirs("./buffers")
```

```python id="6_UXNzIlt4Ca"
# Set seeds
torch.manual_seed(args.seed)
np.random.seed(args.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

<!-- #region id="gEnEkznUp_1c" -->
## Replay buffer
<!-- #endregion -->

```python id="WiuL9NeSqBWt"
def ReplayBuffer(state_dim, is_atari, atari_preprocessing, batch_size, buffer_size, device):
	if is_atari: 
		return AtariBuffer(state_dim, atari_preprocessing, batch_size, buffer_size, device)
	else: 
		return StandardBuffer(state_dim, batch_size, buffer_size, device)
  

class AtariBuffer(object):
	def __init__(self, state_dim, atari_preprocessing, batch_size, buffer_size, device):
		self.batch_size = batch_size
		self.max_size = int(buffer_size)
		self.device = device

		self.state_history = atari_preprocessing["state_history"]

		self.ptr = 0
		self.crt_size = 0

		self.state = np.zeros((
			self.max_size + 1,
			atari_preprocessing["frame_size"],
			atari_preprocessing["frame_size"]
		), dtype=np.uint8)

		self.action = np.zeros((self.max_size, 1), dtype=np.int64)
		self.reward = np.zeros((self.max_size, 1))
		
		# not_done only consider "done" if episode terminates due to failure condition
		# if episode terminates due to timelimit, the transition is not added to the buffer
		self.not_done = np.zeros((self.max_size, 1))
		self.first_timestep = np.zeros(self.max_size, dtype=np.uint8)


	def add(self, state, action, next_state, reward, done, env_done, first_timestep):
		# If dones don't match, env has reset due to timelimit
		# and we don't add the transition to the buffer
		if done != env_done:
			return

		self.state[self.ptr] = state[0]
		self.action[self.ptr] = action
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done
		self.first_timestep[self.ptr] = first_timestep

		self.ptr = (self.ptr + 1) % self.max_size
		self.crt_size = min(self.crt_size + 1, self.max_size)


	def sample(self):
		ind = np.random.randint(0, self.crt_size, size=self.batch_size)

		# Note + is concatenate here
		state = np.zeros(((self.batch_size, self.state_history) + self.state.shape[1:]), dtype=np.uint8)
		next_state = np.array(state)

		state_not_done = 1.
		next_not_done = 1.
		for i in range(self.state_history):

			# Wrap around if the buffer is filled
			if self.crt_size == self.max_size:
				j = (ind - i) % self.max_size
				k = (ind - i + 1) % self.max_size
			else:
				j = ind - i
				k = (ind - i + 1).clip(min=0)
				# If j == -1, then we set state_not_done to 0.
				state_not_done *= (j + 1).clip(min=0, max=1).reshape(-1, 1, 1) #np.where(j < 0, state_not_done * 0, state_not_done)
				j = j.clip(min=0)

			# State should be all 0s if the episode terminated previously
			state[:, i] = self.state[j] * state_not_done
			next_state[:, i] = self.state[k] * next_not_done

			# If this was the first timestep, make everything previous = 0
			next_not_done *= state_not_done
			state_not_done *= (1. - self.first_timestep[j]).reshape(-1, 1, 1)

		return (
			torch.ByteTensor(state).to(self.device).float(),
			torch.LongTensor(self.action[ind]).to(self.device),
			torch.ByteTensor(next_state).to(self.device).float(),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)


	def save(self, save_folder, chunk=int(1e5)):
		np.save(f"{save_folder}_action.npy", self.action[:self.crt_size])
		np.save(f"{save_folder}_reward.npy", self.reward[:self.crt_size])
		np.save(f"{save_folder}_not_done.npy", self.not_done[:self.crt_size])
		np.save(f"{save_folder}_first_timestep.npy", self.first_timestep[:self.crt_size])
		np.save(f"{save_folder}_replay_info.npy", [self.ptr, chunk])

		crt = 0
		end = min(chunk, self.crt_size + 1)
		while crt < self.crt_size + 1:
			np.save(f"{save_folder}_state_{end}.npy", self.state[crt:end])
			crt = end
			end = min(end + chunk, self.crt_size + 1)


	def load(self, save_folder, size=-1):
		reward_buffer = np.load(f"{save_folder}_reward.npy")
		size = min(int(size), self.max_size) if size > 0 else self.max_size
		self.crt_size = min(reward_buffer.shape[0], size)
		
		# Adjust crt_size if we're using a custom size
		size = min(int(size), self.max_size) if size > 0 else self.max_size
		self.crt_size = min(reward_buffer.shape[0], size)

		self.action[:self.crt_size] = np.load(f"{save_folder}_action.npy")[:self.crt_size]
		self.reward[:self.crt_size] = reward_buffer[:self.crt_size]
		self.not_done[:self.crt_size] = np.load(f"{save_folder}_not_done.npy")[:self.crt_size]
		self.first_timestep[:self.crt_size] = np.load(f"{save_folder}_first_timestep.npy")[:self.crt_size]

		self.ptr, chunk = np.load(f"{save_folder}_replay_info.npy")

		crt = 0
		end = min(chunk, self.crt_size + 1)
		while crt < self.crt_size + 1:
			self.state[crt:end] = np.load(f"{save_folder}_state_{end}.npy")
			crt = end
			end = min(end + chunk, self.crt_size + 1)


# Generic replay buffer for standard gym tasks
class StandardBuffer(object):
	def __init__(self, state_dim, batch_size, buffer_size, device):
		self.batch_size = batch_size
		self.max_size = int(buffer_size)
		self.device = device

		self.ptr = 0
		self.crt_size = 0

		self.state = np.zeros((self.max_size, state_dim))
		self.action = np.zeros((self.max_size, 1))
		self.next_state = np.array(self.state)
		self.reward = np.zeros((self.max_size, 1))
		self.not_done = np.zeros((self.max_size, 1))


	def add(self, state, action, next_state, reward, done, episode_done, episode_start):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.crt_size = min(self.crt_size + 1, self.max_size)


	def sample(self):
		ind = np.random.randint(0, self.crt_size, size=self.batch_size)
		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.LongTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)


	def save(self, save_folder):
		np.save(f"{save_folder}_state.npy", self.state[:self.crt_size])
		np.save(f"{save_folder}_action.npy", self.action[:self.crt_size])
		np.save(f"{save_folder}_next_state.npy", self.next_state[:self.crt_size])
		np.save(f"{save_folder}_reward.npy", self.reward[:self.crt_size])
		np.save(f"{save_folder}_not_done.npy", self.not_done[:self.crt_size])
		np.save(f"{save_folder}_ptr.npy", self.ptr)


	def load(self, save_folder, size=-1):
		reward_buffer = np.load(f"{save_folder}_reward.npy")
		
		# Adjust crt_size if we're using a custom size
		size = min(int(size), self.max_size) if size > 0 else self.max_size
		self.crt_size = min(reward_buffer.shape[0], size)

		self.state[:self.crt_size] = np.load(f"{save_folder}_state.npy")[:self.crt_size]
		self.action[:self.crt_size] = np.load(f"{save_folder}_action.npy")[:self.crt_size]
		self.next_state[:self.crt_size] = np.load(f"{save_folder}_next_state.npy")[:self.crt_size]
		self.reward[:self.crt_size] = reward_buffer[:self.crt_size]
		self.not_done[:self.crt_size] = np.load(f"{save_folder}_not_done.npy")[:self.crt_size]

		print(f"Replay Buffer loaded with {self.crt_size} elements.")
```

<!-- #region id="QMaKbJpuqPMO" -->
## Atari preprocessing
<!-- #endregion -->

```python id="nJybxr6AqPJr"
# Atari Preprocessing
# Code is based on https://github.com/openai/gym/blob/master/gym/wrappers/atari_preprocessing.py
class AtariPreprocessing(object):
	def __init__(
		self,
		env,
		frame_skip=4,
		frame_size=84,
		state_history=4,
		done_on_life_loss=False,
		reward_clipping=True, # Clips to a range of -1,1
		max_episode_timesteps=27000
	):
		self.env = env.env
		self.done_on_life_loss = done_on_life_loss
		self.frame_skip = frame_skip
		self.frame_size = frame_size
		self.reward_clipping = reward_clipping
		self._max_episode_steps = max_episode_timesteps
		self.observation_space = np.zeros((frame_size, frame_size))
		self.action_space = self.env.action_space

		self.lives = 0
		self.episode_length = 0

		# Tracks previous 2 frames
		self.frame_buffer = np.zeros(
			(2,
			self.env.observation_space.shape[0],
			self.env.observation_space.shape[1]),
			dtype=np.uint8
		)
		# Tracks previous 4 states
		self.state_buffer = np.zeros((state_history, frame_size, frame_size), dtype=np.uint8)


	def reset(self):
		self.env.reset()
		self.lives = self.env.ale.lives()
		self.episode_length = 0
		self.env.ale.getScreenGrayscale(self.frame_buffer[0])
		self.frame_buffer[1] = 0

		self.state_buffer[0] = self.adjust_frame()
		self.state_buffer[1:] = 0
		return self.state_buffer


	# Takes single action is repeated for frame_skip frames (usually 4)
	# Reward is accumulated over those frames
	def step(self, action):
		total_reward = 0.
		self.episode_length += 1

		for frame in range(self.frame_skip):
			_, reward, done, _ = self.env.step(action)
			total_reward += reward

			if self.done_on_life_loss:
				crt_lives = self.env.ale.lives()
				done = True if crt_lives < self.lives else done
				self.lives = crt_lives

			if done: 
				break

			# Second last and last frame
			f = frame + 2 - self.frame_skip 
			if f >= 0:
				self.env.ale.getScreenGrayscale(self.frame_buffer[f])

		self.state_buffer[1:] = self.state_buffer[:-1]
		self.state_buffer[0] = self.adjust_frame()

		done_float = float(done)
		if self.episode_length >= self._max_episode_steps:
			done = True

		return self.state_buffer, total_reward, done, [np.clip(total_reward, -1, 1), done_float]


	def adjust_frame(self):
		# Take maximum over last two frames
		np.maximum(
			self.frame_buffer[0],
			self.frame_buffer[1],
			out=self.frame_buffer[0]
		)

		# Resize
		image = cv2.resize(
			self.frame_buffer[0],
			(self.frame_size, self.frame_size),
			interpolation=cv2.INTER_AREA
		)
		return np.array(image, dtype=np.uint8)


	def seed(self, seed):
		self.env.seed(seed)
```

<!-- #region id="G9e6SK5TqMVH" -->
## Create Environment
<!-- #endregion -->

```python id="vGg1Z3pSqMRT"
# Create environment, add wrapper if necessary and create env_properties
def make_env(env_name, atari_preprocessing):
	env = wrap_env(gym.make(env_name))
	
	is_atari = gym.envs.registry.spec(env_name).entry_point == 'gym.envs.atari:AtariEnv'
	env = AtariPreprocessing(env, **atari_preprocessing) if is_atari else env

	state_dim = (
		atari_preprocessing["state_history"], 
		atari_preprocessing["frame_size"], 
		atari_preprocessing["frame_size"]
	) if is_atari else env.observation_space.shape[0]

	return (
		env,
		is_atari,
		state_dim,
		env.action_space.n
	)
```

<!-- #region id="ueZKn76hqoBB" -->
## DQN
<!-- #endregion -->

```python id="3ewLgMzlteCx"
# Make env and determine properties
env, is_atari, state_dim, num_actions = make_env(args.env, args.atari_preprocessing)
parameters = args.atari_parameters if is_atari else args.regular_parameters


# Set seeds
env.seed(args.seed)
env.action_space.seed(args.seed)


# Initialize buffer
replay_buffer = ReplayBuffer(state_dim, is_atari, args.atari_preprocessing, parameters["batch_size"], parameters["buffer_size"], device)
```

```python id="K4KSK5hoqn-j"
# Used for Atari
class Conv_Q(nn.Module):
	def __init__(self, frames, num_actions):
		super(Conv_Q, self).__init__()
		self.c1 = nn.Conv2d(frames, 32, kernel_size=8, stride=4)
		self.c2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
		self.c3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
		self.l1 = nn.Linear(3136, 512)
		self.l2 = nn.Linear(512, num_actions)


	def forward(self, state):
		q = F.relu(self.c1(state))
		q = F.relu(self.c2(q))
		q = F.relu(self.c3(q))
		q = F.relu(self.l1(q.reshape(-1, 3136)))
		return self.l2(q)
```

```python id="wX6vCgwNqoqr"
# Used for Box2D / Toy problems
class FC_Q(nn.Module):
	def __init__(self, state_dim, num_actions):
		super(FC_Q, self).__init__()
		self.l1 = nn.Linear(state_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, num_actions)


	def forward(self, state):
		q = F.relu(self.l1(state))
		q = F.relu(self.l2(q))
		return self.l3(q)
```

```python id="e5p4OXgiq5VA"
class DQN(object):
	def __init__(
		self, 
		is_atari,
		num_actions,
		state_dim,
		device,
		discount=0.99,
		optimizer="Adam",
		optimizer_parameters={},
		polyak_target_update=False,
		target_update_frequency=8e3,
		tau=0.005,
		initial_eps = 1,
		end_eps = 0.001,
		eps_decay_period = 25e4,
		eval_eps=0.001,
	):
	
		self.device = device

		# Determine network type
		self.Q = Conv_Q(state_dim[0], num_actions).to(self.device) if is_atari else FC_Q(state_dim, num_actions).to(self.device)
		self.Q_target = copy.deepcopy(self.Q)
		self.Q_optimizer = getattr(torch.optim, optimizer)(self.Q.parameters(), **optimizer_parameters)

		self.discount = discount

		# Target update rule
		self.maybe_update_target = self.polyak_target_update if polyak_target_update else self.copy_target_update
		self.target_update_frequency = target_update_frequency
		self.tau = tau

		# Decay for eps
		self.initial_eps = initial_eps
		self.end_eps = end_eps
		self.slope = (self.end_eps - self.initial_eps) / eps_decay_period

		# Evaluation hyper-parameters
		self.state_shape = (-1,) + state_dim if is_atari else (-1, state_dim)
		self.eval_eps = eval_eps
		self.num_actions = num_actions

		# Number of training iterations
		self.iterations = 0


	def select_action(self, state, eval=False):
		eps = self.eval_eps if eval \
			else max(self.slope * self.iterations + self.initial_eps, self.end_eps)

		# Select action according to policy with probability (1-eps)
		# otherwise, select random action
		if np.random.uniform(0,1) > eps:
			with torch.no_grad():
				state = torch.FloatTensor(state).reshape(self.state_shape).to(self.device)
				return int(self.Q(state).argmax(1))
		else:
			return np.random.randint(self.num_actions)


	def train(self, replay_buffer):
		# Sample replay buffer
		state, action, next_state, reward, done = replay_buffer.sample()

		# Compute the target Q value
		with torch.no_grad():
			target_Q = reward + done * self.discount * self.Q_target(next_state).max(1, keepdim=True)[0]

		# Get current Q estimate
		current_Q = self.Q(state).gather(1, action)

		# Compute Q loss
		Q_loss = F.smooth_l1_loss(current_Q, target_Q)

		# Optimize the Q
		self.Q_optimizer.zero_grad()
		Q_loss.backward()
		self.Q_optimizer.step()

		# Update target network by polyak or full copy every X iterations.
		self.iterations += 1
		self.maybe_update_target()


	def polyak_target_update(self):
		for param, target_param in zip(self.Q.parameters(), self.Q_target.parameters()):
		   target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


	def copy_target_update(self):
		if self.iterations % self.target_update_frequency == 0:
			 self.Q_target.load_state_dict(self.Q.state_dict())


	def save(self, filename):
		torch.save(self.Q.state_dict(), filename + "_Q")
		torch.save(self.Q_optimizer.state_dict(), filename + "_optimizer")


	def load(self, filename):
		self.Q.load_state_dict(torch.load(filename + "_Q"))
		self.Q_target = copy.deepcopy(self.Q)
		self.Q_optimizer.load_state_dict(torch.load(filename + "_optimizer"))
```

```python id="yKxl3XyN0JJ6"
# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10):
	eval_env, _, _, _ = make_env(env_name, args.atari_preprocessing)
	eval_env.seed(seed + 100)

	avg_reward = 0.
	for _ in range(eval_episodes):
		state, done = eval_env.reset(), False
		while not done:
			action = policy.select_action(np.array(state), eval=True)
			state, reward, done, _ = eval_env.step(action)
			avg_reward += reward

	avg_reward /= eval_episodes

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
	print("---------------------------------------")
	return avg_reward
```

```python colab={"base_uri": "https://localhost:8080/"} id="Q4jTDjzCwOFP" executionInfo={"status": "ok", "timestamp": 1634987412783, "user_tz": -330, "elapsed": 49215, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="406cd817-4185-4535-acc6-19577a5a941d"
# For saving files
setting = f"{args.env}_{args.seed}"
buffer_name = f"{args.buffer_name}_{setting}"

# Initialize and load policy
policy = DQN(
    is_atari,
    num_actions,
    state_dim,
    device,
    parameters["discount"],
    parameters["optimizer"],
    parameters["optimizer_parameters"],
    parameters["polyak_target_update"],
    parameters["target_update_freq"],
    parameters["tau"],
    parameters["initial_eps"],
    parameters["end_eps"],
    parameters["eps_decay_period"],
    parameters["eval_eps"],
)

evaluations = []

state, done = env.reset(), False
episode_start = True
episode_reward = 0
episode_timesteps = 0
episode_num = 0
low_noise_ep = np.random.uniform(0,1) < args.low_noise_p
max_episode_steps = gym.make(args.env)._max_episode_steps

# Interact with the environment for max_timesteps
for t in range(int(args.max_timesteps)):

    episode_timesteps += 1

    if t < parameters["start_timesteps"]:
        action = env.action_space.sample()
    else:
        action = policy.select_action(np.array(state))

    # Perform action and log results
    next_state, reward, done, info = env.step(action)
    episode_reward += reward

    # Only consider "done" if episode terminates due to failure condition
    done_float = float(done) if episode_timesteps < max_episode_steps else 0

    # For atari, info[0] = clipped reward, info[1] = done_float
    if is_atari:
        reward = info[0]
        done_float = info[1]
        
    # Store data in replay buffer
    replay_buffer.add(state, action, next_state, reward, done_float, done, episode_start)
    state = copy.copy(next_state)
    episode_start = False

    # Train agent after collecting sufficient data
    if t >= parameters["start_timesteps"] and (t+1) % parameters["train_freq"] == 0:
        policy.train(replay_buffer)

    if done:
        # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
        print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
        # Reset environment
        state, done = env.reset(), False
        episode_start = True
        episode_reward = 0
        episode_timesteps = 0
        episode_num += 1
        low_noise_ep = np.random.uniform(0,1) < args.low_noise_p

    # Evaluate episode
    if (t + 1) % parameters["eval_freq"] == 0:
        evaluations.append(eval_policy(policy, args.env, args.seed))
        np.save(f"./results/behavioral_{setting}", evaluations)
        policy.save(f"./models/behavioral_{setting}")

# Save final policy
policy.save(f"./models/behavioral_{setting}")
```

<!-- #region id="46sQwBLx6m1M" -->
## Generate Buffer
<!-- #endregion -->

```python id="5pnM3-jF7sa9"
# Make env and determine properties
env, is_atari, state_dim, num_actions = make_env(args.env, args.atari_preprocessing)
parameters = args.atari_parameters if is_atari else args.regular_parameters


# Set seeds
env.seed(args.seed)
env.action_space.seed(args.seed)


# Initialize buffer
replay_buffer = ReplayBuffer(state_dim, is_atari, args.atari_preprocessing, parameters["batch_size"], parameters["buffer_size"], device)
```

```python colab={"base_uri": "https://localhost:8080/"} id="nTtLegdr6zyd" executionInfo={"status": "ok", "timestamp": 1634988961512, "user_tz": -330, "elapsed": 15980, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="991a5e62-6c8f-4243-9442-9b43d024171b"
setting = f"{args.env}_{args.seed}"
buffer_name = f"{args.buffer_name}_{setting}"

# Initialize and load policy
policy = DQN(
    is_atari,
    num_actions,
    state_dim,
    device,
    parameters["discount"],
    parameters["optimizer"],
    parameters["optimizer_parameters"],
    parameters["polyak_target_update"],
    parameters["target_update_freq"],
    parameters["tau"],
    parameters["initial_eps"],
    parameters["end_eps"],
    parameters["eps_decay_period"],
    parameters["eval_eps"],
)

policy.load(f"./models/behavioral_{setting}")

evaluations = []

state, done = env.reset(), False
episode_start = True
episode_reward = 0
episode_timesteps = 0
episode_num = 0
low_noise_ep = np.random.uniform(0,1) < args.low_noise_p
max_episode_steps = gym.make(args.env)._max_episode_steps

# Interact with the environment for max_timesteps
for t in range(int(args.max_timesteps)):

    episode_timesteps += 1

    # If generating the buffer, episode is low noise with p=low_noise_p.
    # If policy is low noise, we take random actions with p=eval_eps.
    # If the policy is high noise, we take random actions with p=rand_action_p.
    if not low_noise_ep and np.random.uniform(0,1) < args.rand_action_p - parameters["eval_eps"]:
        action = env.action_space.sample()
    else:
        action = policy.select_action(np.array(state), eval=True)

    # Perform action and log results
    next_state, reward, done, info = env.step(action)
    episode_reward += reward

    # Only consider "done" if episode terminates due to failure condition
    done_float = float(done) if episode_timesteps < max_episode_steps else 0

    # For atari, info[0] = clipped reward, info[1] = done_float
    if is_atari:
        reward = info[0]
        done_float = info[1]
        
    # Store data in replay buffer
    replay_buffer.add(state, action, next_state, reward, done_float, done, episode_start)
    state = copy.copy(next_state)
    episode_start = False

    if done:
        # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
        print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
        # Reset environment
        state, done = env.reset(), False
        episode_start = True
        episode_reward = 0
        episode_timesteps = 0
        episode_num += 1
        low_noise_ep = np.random.uniform(0,1) < args.low_noise_p

# Save final buffer and performance
evaluations.append(eval_policy(policy, args.env, args.seed))
np.save(f"./results/buffer_performance_{setting}", evaluations)
replay_buffer.save(f"./buffers/{buffer_name}")
```

<!-- #region id="BrK18C73qMOU" -->
## Discrete BCQ
<!-- #endregion -->

```python id="dBq599JWqciE"
# Used for Atari
class Conv_Q(nn.Module):
	def __init__(self, frames, num_actions):
		super(Conv_Q, self).__init__()
		self.c1 = nn.Conv2d(frames, 32, kernel_size=8, stride=4)
		self.c2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
		self.c3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

		self.q1 = nn.Linear(3136, 512)
		self.q2 = nn.Linear(512, num_actions)

		self.i1 = nn.Linear(3136, 512)
		self.i2 = nn.Linear(512, num_actions)


	def forward(self, state):
		c = F.relu(self.c1(state))
		c = F.relu(self.c2(c))
		c = F.relu(self.c3(c))

		q = F.relu(self.q1(c.reshape(-1, 3136)))
		i = F.relu(self.i1(c.reshape(-1, 3136)))
		i = self.i2(i)
		return self.q2(q), F.log_softmax(i, dim=1), i
```

```python id="8Rb4Ok01qgvh"
# Used for Box2D / Toy problems
class FC_Q(nn.Module):
	def __init__(self, state_dim, num_actions):
		super(FC_Q, self).__init__()
		self.q1 = nn.Linear(state_dim, 256)
		self.q2 = nn.Linear(256, 256)
		self.q3 = nn.Linear(256, num_actions)

		self.i1 = nn.Linear(state_dim, 256)
		self.i2 = nn.Linear(256, 256)
		self.i3 = nn.Linear(256, num_actions)		


	def forward(self, state):
		q = F.relu(self.q1(state))
		q = F.relu(self.q2(q))

		i = F.relu(self.i1(state))
		i = F.relu(self.i2(i))
		i = self.i3(i)
		return self.q3(q), F.log_softmax(i, dim=1), i
```

```python id="UHEdtWB0qi21"
class discrete_BCQ(object):
	def __init__(
		self, 
		is_atari,
		num_actions,
		state_dim,
		device,
		BCQ_threshold=0.3,
		discount=0.99,
		optimizer="Adam",
		optimizer_parameters={},
		polyak_target_update=False,
		target_update_frequency=8e3,
		tau=0.005,
		initial_eps = 1,
		end_eps = 0.001,
		eps_decay_period = 25e4,
		eval_eps=0.001,
	):
	
		self.device = device

		# Determine network type
		self.Q = Conv_Q(state_dim[0], num_actions).to(self.device) if is_atari else FC_Q(state_dim, num_actions).to(self.device)
		self.Q_target = copy.deepcopy(self.Q)
		self.Q_optimizer = getattr(torch.optim, optimizer)(self.Q.parameters(), **optimizer_parameters)

		self.discount = discount

		# Target update rule
		self.maybe_update_target = self.polyak_target_update if polyak_target_update else self.copy_target_update
		self.target_update_frequency = target_update_frequency
		self.tau = tau

		# Decay for eps
		self.initial_eps = initial_eps
		self.end_eps = end_eps
		self.slope = (self.end_eps - self.initial_eps) / eps_decay_period

		# Evaluation hyper-parameters
		self.state_shape = (-1,) + state_dim if is_atari else (-1, state_dim)
		self.eval_eps = eval_eps
		self.num_actions = num_actions

		# Threshold for "unlikely" actions
		self.threshold = BCQ_threshold

		# Number of training iterations
		self.iterations = 0


	def select_action(self, state, eval=False):
		# Select action according to policy with probability (1-eps)
		# otherwise, select random action
		if np.random.uniform(0,1) > self.eval_eps:
			with torch.no_grad():
				state = torch.FloatTensor(state).reshape(self.state_shape).to(self.device)
				q, imt, i = self.Q(state)
				imt = imt.exp()
				imt = (imt/imt.max(1, keepdim=True)[0] > self.threshold).float()
				# Use large negative number to mask actions from argmax
				return int((imt * q + (1. - imt) * -1e8).argmax(1))
		else:
			return np.random.randint(self.num_actions)


	def train(self, replay_buffer):
		# Sample replay buffer
		state, action, next_state, reward, done = replay_buffer.sample()

		# Compute the target Q value
		with torch.no_grad():
			q, imt, i = self.Q(next_state)
			imt = imt.exp()
			imt = (imt/imt.max(1, keepdim=True)[0] > self.threshold).float()

			# Use large negative number to mask actions from argmax
			next_action = (imt * q + (1 - imt) * -1e8).argmax(1, keepdim=True)

			q, imt, i = self.Q_target(next_state)
			target_Q = reward + done * self.discount * q.gather(1, next_action).reshape(-1, 1)

		# Get current Q estimate
		current_Q, imt, i = self.Q(state)
		current_Q = current_Q.gather(1, action)

		# Compute Q loss
		q_loss = F.smooth_l1_loss(current_Q, target_Q)
		i_loss = F.nll_loss(imt, action.reshape(-1))

		Q_loss = q_loss + i_loss + 1e-2 * i.pow(2).mean()

		# Optimize the Q
		self.Q_optimizer.zero_grad()
		Q_loss.backward()
		self.Q_optimizer.step()

		# Update target network by polyak or full copy every X iterations.
		self.iterations += 1
		self.maybe_update_target()


	def polyak_target_update(self):
		for param, target_param in zip(self.Q.parameters(), self.Q_target.parameters()):
		   target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


	def copy_target_update(self):
		if self.iterations % self.target_update_frequency == 0:
			 self.Q_target.load_state_dict(self.Q.state_dict())
```

```python id="pvX6nG1h8FaT"
# Make env and determine properties
env, is_atari, state_dim, num_actions = make_env(args.env, args.atari_preprocessing)
parameters = args.atari_parameters if is_atari else args.regular_parameters


# Set seeds
env.seed(args.seed)
env.action_space.seed(args.seed)


# Initialize buffer
replay_buffer = ReplayBuffer(state_dim, is_atari, args.atari_preprocessing, parameters["batch_size"], parameters["buffer_size"], device)
```

```python colab={"base_uri": "https://localhost:8080/"} id="6Gm1z1JW8Bxk" executionInfo={"status": "ok", "timestamp": 1634989116620, "user_tz": -330, "elapsed": 66537, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="090c84ac-a153-4680-a87e-bfe9fed9f24e"
# For saving files
setting = f"{args.env}_{args.seed}"
buffer_name = f"{args.buffer_name}_{setting}"

# Initialize and load policy
policy = discrete_BCQ(
    is_atari,
    num_actions,
    state_dim,
    device,
    args.BCQ_threshold,
    parameters["discount"],
    parameters["optimizer"],
    parameters["optimizer_parameters"],
    parameters["polyak_target_update"],
    parameters["target_update_freq"],
    parameters["tau"],
    parameters["initial_eps"],
    parameters["end_eps"],
    parameters["eps_decay_period"],
    parameters["eval_eps"]
)

# Load replay buffer	
replay_buffer.load(f"./buffers/{buffer_name}")

evaluations = []
episode_num = 0
done = True 
training_iters = 0

while training_iters < args.max_timesteps: 
    
    for _ in range(int(parameters["eval_freq"])):
        policy.train(replay_buffer)

    evaluations.append(eval_policy(policy, args.env, args.seed))
    np.save(f"./results/BCQ_{setting}", evaluations)

    training_iters += int(parameters["eval_freq"])
    print(f"Training iterations: {training_iters}")
```

```python id="X6UL_5H08QVC"
!apt-get -qq install tree
```

```python colab={"base_uri": "https://localhost:8080/"} id="bLap25tR8jon" executionInfo={"status": "ok", "timestamp": 1634989150242, "user_tz": -330, "elapsed": 461, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="9a8eff7b-b1dd-46c4-9069-e020307f3716"
!tree --du -h -C .
```

```python colab={"base_uri": "https://localhost:8080/", "height": 421} id="l4Cx77m-8mJE" executionInfo={"status": "ok", "timestamp": 1634989208793, "user_tz": -330, "elapsed": 661, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="ae4aedc8-82cd-48a8-83ac-61ff2820e765"
show_video()
```
