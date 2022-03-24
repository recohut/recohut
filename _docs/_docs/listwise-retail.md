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

<!-- #region id="9FbVBgGDaUV5" -->
# List-wise Product Recommendations using RL methods on Retail dataset
<!-- #endregion -->

```python id="pQhWTElKuY6Z"
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.autograd
from torch.autograd import Variable
import torch.optim as optim

import gym
from gym import spaces

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import random
from collections import deque
import os
```

<!-- #region id="ecbUGmt8uw7i" -->
## Data
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="zkRdSXGGuw5K" executionInfo={"status": "ok", "timestamp": 1639481833996, "user_tz": -330, "elapsed": 4872, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="c19370d0-6516-49a2-84fa-169bb62c4d95"
!gdown --id 1h5DEIT-JYeR5e8D8BK6dny5zYCwth1rl
```

```python colab={"base_uri": "https://localhost:8080/"} id="Ev3QK2Zgu2O5" executionInfo={"status": "ok", "timestamp": 1639481844856, "user_tz": -330, "elapsed": 1594, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="e06b6cf4-718f-42c9-b32b-c177aefae9de"
!unzip dataset.zip
```

```python id="jJA8g_x9v9GK"
PARENT_PATH = 'weight'
ACTOR_PATH = 'weight/actor'
ACTOR_TARGET_PATH = 'weight/actor_target'
CRITIC_PATH = 'weight/critic'
CRITIC_TARGET_PATH = 'weight/critic_target'
```

<!-- #region id="XPEY3UuYu5pf" -->
## Model
<!-- #endregion -->

```python id="t2rvSOLbu8LM"
class Critic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size, action_sequence_length):
        super(Critic, self).__init__()
        self.encode_state = nn.LSTM(state_size,action_size,batch_first = True)
        hidden_stack = [nn.Linear((action_sequence_length + 1)*action_size, hidden_size),
                             nn.ReLU(),]
        for i in range(3):
            hidden_stack.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU()])
        self.hidden_layer = nn.Sequential(*hidden_stack)
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, state, action):
        """
        Params state and actions are torch tensors
        """
        if not isinstance(state,torch.Tensor):
            state = torch.tensor(state)
        if not isinstance(action,torch.Tensor):
            action = torch.tensor(action)
        if (len(state.shape)==2) and (len(action.shape)==2):
            action = action.unsqueeze(0)
            state = state.unsqueeze(0)
        _,(encoded_state,__) = self.encode_state(state)
        encoded_state = encoded_state.squeeze(0)
        action = action.flatten(1)
        x = torch.cat([encoded_state,action],-1)
        x = self.hidden_layer(x)
        x = self.output_layer(x)
        if (len(state.shape)==2) and (len(action.shape)==2):
            x = x.squeeze(0)
        return x
```

```python id="nUxrr7dcvAOu"
class Actor(nn.Module):
    def __init__(self, input_size,input_sequence_length, output_sequence_length, output_size):
        super(Actor, self).__init__()
        self.weight_matrix = torch.nn.Parameter(torch.ones((1,input_sequence_length), requires_grad=True))
        self.Linear = nn.Linear(input_size, output_size)
        self.Activation = nn.Softmax(dim=-1)
        self.output_shape = (output_sequence_length,output_size)
    def forward(self, state):
        """
        Param state is a torch tensor
        """
        state = torch.FloatTensor(state)
        size = len(state.shape)
        if size==2:
            state = state.unsqueeze(0)
        state = self.weight_matrix.matmul(state)
        state = state.squeeze(1)
        action = []
#        x = self.Linear(state)
        action.append(self.Activation(state))
        for i in range(self.output_shape[0]-1):
            indices = action[i].argmax(-1).unsqueeze(-1)
            action_i = action[i].scatter(-1,indices,0)
            action_i = action_i / action_i.sum(-1).unsqueeze(-1)
            action.append(action_i)
        action = torch.cat(action,-1).reshape((-1,self.output_shape[0],self.output_shape[1]))
        if size==2:
            action = action.squeeze(0)
        return action
```

```python id="nrOnx1GcvNmo"
class OUNoise(object):
    def __init__(self, action_space, mu=0.0, theta=0.1, max_sigma=0.5, min_sigma=0.0, decay_period=500):
        self.mu           = mu
        self.theta        = theta
        self.sigma        = max_sigma
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        self.action_dim   = action_space.shape
        self.low          = action_space.low
        self.high         = action_space.high
        self.reset()
        
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
        
    def evolve_state(self):
        x  = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim[0],self.action_dim[1])
        self.state = x + dx
        return self.state
    
    def get_action(self, action, t=0): 
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        action = np.clip(action + ou_state, self.low, self.high)
        action = torch.from_numpy(action)
        action = torch.nn.Softmax(dim=-1)(action).detach().numpy()
        return action
```

```python id="AGQGxVTUvTFN"
class ActionSpace(gym.Space):
    def __init__(self, n_reco, n_item):
        self.shape = (n_reco, n_item)
        self.dtype = np.int64
        self.low = 0
        self.high = 1
        super(ActionSpace, self).__init__(self.shape,self.dtype)
    def sample(self):
        sample = torch.zeros(self.shape,torch.int64)
        indices = torch.randint(0,n_item,(n_reco,1))
        sampe = sample.scatter_(1,indices,1)
        return sampe.numpy()
```

```python id="IqT3xUflvoFc"
class StateSpace(gym.Space):
    def __init__(self, max_state, n_item):
        self.shape = (max_state, n_item)
        self.dtype = np.int64
        super(StateSpace, self).__init__(self.shape,self.dtype)
```

<!-- #region id="Q5d-EPhuvjkv" -->
## Memory buffer
<!-- #endregion -->

```python id="dz_Lj1THvlKj"
class Memory:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)
    
    def push(self, state, action, reward, next_state, done):
        experience = (state, action, np.array([reward]), next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)
        
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def __len__(self):
        return len(self.buffer)
```

<!-- #region id="wq50vLsxwD10" -->
### Agent
<!-- #endregion -->

```python id="mW0xdq3CvytJ"
class DDPGagent:
    def __init__(self, env, hidden_size=576, 
                 actor_learning_rate=1e-4, 
                 critic_learning_rate=1e-3, 
                 gamma=0.99, tau=1e-2, 
                 max_memory_size=50000):
        # Params
        self.size_states = env.observation_space.shape
        self.size_actions = env.action_space.shape
        self.gamma = gamma
        self.tau = tau

        # Networks
        self.actor = Actor(self.size_states[1],self.size_actions[0], hidden_size, self.size_actions[1])
        self.actor_target = Actor(self.size_states[1],self.size_actions[0], hidden_size, self.size_actions[1])
        self.critic = Critic(self.size_states[1] ,self.size_actions[1] , hidden_size, self.size_actions[0])
        self.critic_target = Critic(self.size_states[1] ,self.size_actions[1] , hidden_size, self.size_actions[0])

        self.load_()
        
        # Training
        self.memory = Memory(max_memory_size)        
        self.critic_criterion  = nn.MSELoss()
        self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)
        
        for p in self.actor_target.parameters():
            p.requires_grad = False
        for p in self.critic_target.parameters():
            p.requires_grad = False
    
    def from_probability_distribution_to_action(self,action):
        if not isinstance(action,torch.Tensor):
            action = torch.FloatTensor(action)
        indices = torch.max(action,-1).indices.unsqueeze(-1)
        action = action.zero_().scatter_(-1,indices,1).numpy()
        return action
    
    def get_action(self, state):
        if not isinstance(state,torch.Tensor):
            state = torch.FloatTensor(state)
        with torch.no_grad():
            action = self.actor.forward(state)
        action = action.detach().numpy()
        return action
    
    def update(self, batch_size):
        states, actions, rewards, next_states, _ = self.memory.sample(batch_size)
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
    
        # Critic loss        
        Qvals = self.critic.forward(states, actions)
        next_actions = self.actor_target.forward(next_states)
        next_actions = self.from_probability_distribution_to_action(next_actions)
        next_Q = self.critic_target.forward(next_states, next_actions)
        Qprime = rewards + self.gamma * next_Q
        critic_loss = self.critic_criterion(Qvals, Qprime)

        # Actor loss
        policy_loss = -self.critic.forward(states, self.actor.forward(states)).mean()
        
        # update networks
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward() 
        self.critic_optimizer.step()

        # update target networks
        with torch.no_grad():
            for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
       
            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
    def save_(self):
        if not os.path.exists(PARENT_PATH):
            os.mkdir(PARENT_PATH)
        torch.save(self.actor.state_dict(), ACTOR_PATH)
        torch.save(self.actor_target.state_dict(), ACTOR_TARGET_PATH)
        torch.save(self.critic.state_dict(), CRITIC_PATH)
        torch.save(self.critic_target.state_dict(), CRITIC_TARGET_PATH)
    def load_(self):
        try:
            self.actor.load_state_dict(torch.load(ACTOR_PATH))
            self.actor_target.load_state_dict(torch.load(ACTOR_TARGET_PATH))
            self.critic.load_state_dict(torch.load(CRITIC_PATH))
            self.critic_target.load_state_dict(torch.load(CRITIC_TARGET_PATH))
        except Exception:
            for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                target_param.data.copy_(param.data)

            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(param.data)
        print(self.actor.eval(), self.critic.eval())
```

<!-- #region id="oCC7LoW0vlfZ" -->
## Environment
<!-- #endregion -->

```python id="HLQkXPdXvycs"
class MyEnv(gym.Env):

    def __init__(self,
         history_data: pd.DataFrame,
         item_data: pd.DataFrame,
         user_data: pd.DataFrame,
         dim_action: int = 3,
         max_lag: int = 20,
  ): 
        
        super(MyEnv, self).__init__()
        self.history_data = history_data
        self.item_data = item_data
        self.user_data = user_data
        self.dim_action = dim_action
        self.max_lag = max_lag
        self.list_item = item_data.ID.tolist()
        self.n_item = len(self.list_item)
        self.encode = OneHotEncoder(handle_unknown='ignore')
        self.encode.fit(np.array(self.list_item).reshape(-1,1))
        self.action_space = ActionSpace(self.dim_action, self.n_item)
        self.observation_space = StateSpace(self.max_lag, self.n_item)
        self.idx_current = 0
        
    def step(self, action):
        action = np.array(action)
        _current_itemID = self.history_data.iloc[self.idx_current].ItemID
        _current_AcountID = self.history_data.iloc[self.idx_current].AccountID
        _temp = self.history_data.iloc[:self.idx_current + 1]
        current_frame = _temp[_temp.AccountID == _current_AcountID]
        if (len(current_frame) < self.max_lag):
            first_state = obs = np.zeros((self.max_lag - len(current_frame),self.n_item))
            str_obs = current_frame.ItemID.to_numpy().reshape(-1,1)
            last_state = self.encode.transform(str_obs).toarray()
            obs = np.concatenate([first_state, last_state],0)
        else:
            str_obs = current_frame[-self.max_lag:].ItemID.to_numpy().reshape(-1,1)
            obs = self.encode.transform(str_obs).toarray()
        
        _encode_current_itemID = self.encode.transform([[_current_itemID]]).toarray().reshape(-1)
        reward = 0
        for i in range(self.dim_action):
            if (action[i]==_encode_current_itemID).all():
                reward = self.dim_action - i
                break
        if (np.sum(action,1) > 1).any():
            reward = reward - 10
        done = False
        return obs, reward, done, {}
    def get_observation(self, reset = False):
        if reset:
            self.idx_current = np.random.randint(len(self.history_data))
        else:
            if (self.idx_current+1) == len(self.history_data):
                self.idx_current = 0
            else:
                self.idx_current = self.idx_current + 1
        _current_AcountID = self.history_data.iloc[self.idx_current].AccountID
        _temp = self.history_data.iloc[:self.idx_current]
        recent_past_frame = _temp[_temp.AccountID == _current_AcountID]
        
        first_state = obs = np.zeros((len(recent_past_frame),self.n_item))
        if (len(recent_past_frame) < self.max_lag):
            first_state = obs = np.zeros(( self.max_lag - len(recent_past_frame),self.n_item))
            str_obs = recent_past_frame.ItemID.to_numpy().reshape(-1,1)
            if len(str_obs) !=0:
                last_state = self.encode.transform(str_obs).toarray()
                obs = np.concatenate([first_state, last_state],0)
        else:
            str_obs = recent_past_frame[-self.max_lag:].ItemID.to_numpy().reshape(-1,1)
            obs = self.encode.transform(str_obs).toarray()
        return obs
    
    def render(self, mode='human', close=False):
        # Render the environment to the screen
        raise Exception()
```

<!-- #region id="ih6WzjaFwUSw" -->
## Training
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="vtMSwuuLwMCC" executionInfo={"status": "ok", "timestamp": 1639482215218, "user_tz": -330, "elapsed": 2892, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="9f6c74a4-35d0-4e57-fb43-c04c85da043a"
rating = pd.read_csv('dataset/Data_Acc_Item.csv')
item = pd.read_csv('dataset/Item_inf.csv',index_col = 'Unnamed: 0')
user = pd.read_csv('dataset/train_acc_inf.csv')

env = MyEnv(rating,item,user)
agent = DDPGagent(env)
noise = OUNoise(env.action_space)
batch_size = 100
rewards = []
avg_rewards = []
```

```python colab={"base_uri": "https://localhost:8080/", "height": 502} id="giGBoeJfwTFJ" executionInfo={"status": "error", "timestamp": 1639482223753, "user_tz": -330, "elapsed": 872, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="e39f8e6d-ddae-442a-9045-da164d0a89f4"
for episode in range(20):
    state = env.get_observation(reset = True)
    noise.reset()
    episode_reward = 0
    
    for step in range(500):
        action = agent.get_action(state)
        action = noise.get_action(action, step)
        action = agent.from_probability_distribution_to_action(action)
        new_state, reward, done, _ = env.step(action) 
        agent.memory.push(state, action, reward, new_state, done)
        
        if len(agent.memory) > batch_size:
            agent.update(batch_size)        
        
        state = env.get_observation()
        episode_reward += reward
        print('step {} in episode {} : reward is {}'.format(step, episode, reward))

    rewards.append(episode_reward)
    avg_rewards.append(np.mean(rewards[-10:]))
```

```python id="AeNUU95lwWWJ"
plt.plot(rewards)
plt.plot(avg_rewards)
plt.plot()
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()
```
