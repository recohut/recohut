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

```python id="ntMjzC8bOKqY"
!git clone https://github.com/backgom2357/Recommender_system_via_deep_RL.git
%cd Recommender_system_via_deep_RL
```

```python id="j1zfOQb08nff"
!wget -q --show-progress http://files.grouplens.org/datasets/movielens/ml-1m.zip
!unzip ml-1m.zip
```

```python id="ngfgl9m8OSQ2"
!apt-get -qq install tree
```

```python colab={"base_uri": "https://localhost:8080/"} id="gux9nvVROZd7" executionInfo={"status": "ok", "timestamp": 1634893152987, "user_tz": -330, "elapsed": 484, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="635b3ef7-2f23-458c-f522-4db97ad19eb7"
!tree --du -h .
```

<!-- #region id="MUf_5x895cOH" -->
### Imports
<!-- #endregion -->

```python id="rUbmetdC5hlx" executionInfo={"status": "ok", "timestamp": 1634893655938, "user_tz": -330, "elapsed": 2796, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
from collections import deque
import random

import pandas as pd
import numpy as np
import itertools
import logging, os

import pandas as pd
import numpy as np
import tensorflow as tf
import itertools
import matplotlib.pyplot as plt
import time
import os

import tensorflow as tf

import matplotlib.pyplot as plt
import plotly.express as px

import time
import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from tensorflow.keras.layers import InputLayer, Embedding, Dot, Reshape, Dense
from tensorflow.keras.models import Model

import warnings
warnings.filterwarnings('ignore')

ROOT_DIR = os.getcwd()
DATA_DIR = os.path.join(ROOT_DIR, 'ml-1m/')
STATE_SIZE = 10

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
```

<!-- #region id="Czu9bcvBCUc5" -->
## Preprocessing
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="_rN2sgigB2Ts" executionInfo={"status": "ok", "timestamp": 1634806022404, "user_tz": -330, "elapsed": 3404, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="ab46eb3f-f198-43c1-a635-dd380c4c201f"
#Loading datasets
ratings_list = [i.strip().split("::") for i in open(os.path.join(DATA_DIR,'ratings.dat'), 'r').readlines()]
users_list = [i.strip().split("::") for i in open(os.path.join(DATA_DIR,'users.dat'), 'r').readlines()]
movies_list = [i.strip().split("::") for i in open(os.path.join(DATA_DIR,'movies.dat'),encoding='latin-1').readlines()]
ratings_df = pd.DataFrame(ratings_list, columns = ['UserID', 'MovieID', 'Rating', 'Timestamp'], dtype = np.uint32)
movies_df = pd.DataFrame(movies_list, columns = ['MovieID', 'Title', 'Genres'])
movies_df['MovieID'] = movies_df['MovieID'].apply(pd.to_numeric)

movies_id_to_movies = {movie[0]: movie[1:] for movie in movies_list}

print(len(set(ratings_df["UserID"])) == max([int(i) for i in set(ratings_df["UserID"])]))
print(max([int(i) for i in set(ratings_df["UserID"])]))
```

```python id="zA7LWOGTCBmm" executionInfo={"status": "ok", "timestamp": 1634893799297, "user_tz": -330, "elapsed": 8898, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
ratings_df = ratings_df.applymap(int)
users_dict = {user : [] for user in set(ratings_df["UserID"])}
ratings_df = ratings_df.sort_values(by='Timestamp', ascending=True)
ratings_df_gen = ratings_df.iterrows()
users_dict_for_history_len = {user : [] for user in set(ratings_df["UserID"])}

for data in ratings_df_gen:
    users_dict[data[1]['UserID']].append((data[1]['MovieID'], data[1]['Rating']))
    if data[1]['Rating'] >= 4:
        users_dict_for_history_len[data[1]['UserID']].append((data[1]['MovieID'], data[1]['Rating']))

users_history_lens = [len(users_dict_for_history_len[u]) for u in set(ratings_df["UserID"])]

np.save("user_dict.npy", users_dict)
np.save("users_histroy_len.npy", users_history_lens)
```

<!-- #region id="iHSwKv037rLw" -->
### Tree
<!-- #endregion -->

```python id="1dKx_4kF7spP" executionInfo={"status": "ok", "timestamp": 1634893658734, "user_tz": -330, "elapsed": 7, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
class SumTree:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.tree = np.zeros((buffer_size * 2 - 1))
        self.index = buffer_size - 1

    def update_tree(self, index):
        while True:
            index = (index - 1) // 2
            left = (index * 2) + 1
            right = (index * 2) + 2
            self.tree[index] = self.tree[left] + self.tree[right]
            if index == 0:
                break

    def add_data(self, priority):
        if self.index == self.buffer_size * 2 - 1:
            self.index = self.buffer_size - 1

        self.tree[self.index] = priority
        self.update_tree(self.index)
        self.index += 1

    def search(self, num):
        current = 0
        while True:
            left = (current * 2) + 1
            right = (current * 2) + 2

            if num <= self.tree[left]:
                current = left
            else:
                num -= self.tree[left]
                current = right
            
            if current >= self.buffer_size - 1:
                break

        return self.tree[current], current, current - self.buffer_size + 1

    def update_prioirty(self, priority, index):
        self.tree[index] = priority
        self.update_tree(index)

    def sum_all_prioirty(self):
        return float(self.tree[0])


class MinTree:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.tree = np.ones((buffer_size * 2 - 1))
        self.index = buffer_size - 1

    def update_tree(self, index):
        while True:
            index = (index - 1) // 2
            left = (index * 2) + 1
            right = (index * 2) + 2
            if self.tree[left] > self.tree[right]:
                self.tree[index] = self.tree[right]
            else:
                self.tree[index] = self.tree[left]
            if index == 0:
                break

    def add_data(self, priority):
        if self.index == self.buffer_size * 2 - 1:
            self.index = self.buffer_size - 1

        self.tree[self.index] = priority
        self.update_tree(self.index)
        self.index += 1

    def update_prioirty(self, priority, index):
        self.tree[index] = priority
        self.update_tree(index)

    def min_prioirty(self):
        return float(self.tree[0])
```

<!-- #region id="mP-6ocWh7f5J" -->
### Replay buffer
<!-- #endregion -->

```python id="UHWHnCvv7f2a" executionInfo={"status": "ok", "timestamp": 1634893661262, "user_tz": -330, "elapsed": 7, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
class PriorityExperienceReplay(object):

    '''
    apply PER
    '''

    def __init__(self, buffer_size, embedding_dim):
        self.buffer_size = buffer_size
        self.crt_idx = 0
        self.is_full = False
        
        '''
            state : (300,), 
            next_state : (300,) 변할 수 잇음, 
            actions : (100,), 
            rewards : (1,), 
            dones : (1,)
        '''
        self.states = np.zeros((buffer_size, 3*embedding_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, embedding_dim), dtype=np.float32)
        self.rewards = np.zeros((buffer_size), dtype=np.float32)
        self.next_states = np.zeros((buffer_size, 3*embedding_dim), dtype=np.float32)
        self.dones = np.zeros(buffer_size, np.bool)

        self.sum_tree = SumTree(buffer_size)
        self.min_tree = MinTree(buffer_size)

        self.max_prioirty = 1.0
        self.alpha = 0.6
        self.beta = 0.4
        self.beta_constant = 0.00001

    def append(self, state, action, reward, next_state, done):
        self.states[self.crt_idx] = state
        self.actions[self.crt_idx] = action
        self.rewards[self.crt_idx] = reward
        self.next_states[self.crt_idx] = next_state
        self.dones[self.crt_idx] = done

        self.sum_tree.add_data(self.max_prioirty ** self.alpha)
        self.min_tree.add_data(self.max_prioirty ** self.alpha)
        
        self.crt_idx = (self.crt_idx + 1) % self.buffer_size
        if self.crt_idx == 0:
            self.is_full = True

    def sample(self, batch_size):
        rd_idx = []
        weight_batch = []
        index_batch = []
        sum_priority = self.sum_tree.sum_all_prioirty()
        
        N = self.buffer_size if self.is_full else self.crt_idx
        min_priority = self.min_tree.min_prioirty() / sum_priority
        max_weight = (N * min_priority) ** (-self.beta)

        segment_size = sum_priority/batch_size
        for j in range(batch_size):
            min_seg = segment_size * j
            max_seg = segment_size * (j + 1)

            random_num = random.uniform(min_seg, max_seg)
            priority, tree_index, buffer_index = self.sum_tree.search(random_num)
            rd_idx.append(buffer_index)

            p_j = priority / sum_priority
            w_j = (p_j * N) ** (-self.beta) / max_weight
            weight_batch.append(w_j)
            index_batch.append(tree_index)
        self.beta = min(1.0, self.beta + self.beta_constant)

        batch_states = self.states[rd_idx]
        batch_actions = self.actions[rd_idx]
        batch_rewards = self.rewards[rd_idx]
        batch_next_states = self.next_states[rd_idx]
        batch_dones = self.dones[rd_idx]

        return batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones, np.array(weight_batch), index_batch

    def update_priority(self, priority, index):
        self.sum_tree.update_prioirty(priority ** self.alpha, index)
        self.min_tree.update_prioirty(priority ** self.alpha, index)
        self.update_max_priority(priority ** self.alpha)

    def update_max_priority(self, priority):
        self.max_prioirty = max(self.max_prioirty, priority)
```

<!-- #region id="heHg41DN7fy9" -->
### Replay memory
<!-- #endregion -->

```python id="0wxthxqm7yHA" executionInfo={"status": "ok", "timestamp": 1634893664386, "user_tz": -330, "elapsed": 538, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
class ReplayMemory(object):

    '''
    apply PER, later
    '''

    def __init__(self, replay_memory_size, embedding_dim):
        self.rm_size = replay_memory_size
        self.crt_idx = 0
        
        '''
            state : (300,), 
            next_state : (300,) 변할 수 잇음, 
            actions : (100,), 
            rewards : (1,), 
            dones : (1,)
        '''

        self.states = np.zeros((replay_memory_size, 3*embedding_dim), dtype=np.float32)
        self.actions = np.zeros((replay_memory_size, embedding_dim), dtype=np.float32)
        self.rewards = np.zeros((replay_memory_size), dtype=np.float32)
        self.rewards[replay_memory_size-1] = 777
        self.next_states = np.zeros((replay_memory_size, 3*embedding_dim), dtype=np.float32)
        self.dones = np.zeros(replay_memory_size, np.bool)

    def is_full(self):
        return self.rewards[-1] != 777

    def append(self, state, action, reward, next_state, done):
        self.states[self.crt_idx] = state
        self.actions[self.crt_idx] = action
        self.rewards[self.crt_idx] = reward
        self.next_states[self.crt_idx] = next_state
        self.dones[self.crt_idx] = done

        self.crt_idx = (self.crt_idx + 1) % self.rm_size

    def sample(self, batch_size):
        rd_idx = np.random.choice((1 - self.is_full())*self.crt_idx + self.is_full()*self.rm_size-1, batch_size)
        batch_states = self.states[rd_idx]
        batch_actions = self.actions[rd_idx]
        batch_rewards = self.rewards[rd_idx]
        batch_next_states = self.next_states[rd_idx]
        batch_dones = self.dones[rd_idx]

        return batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones
```

<!-- #region id="rJaA7IWX50TG" -->
### Embedding
<!-- #endregion -->

```python id="09oqGUMN510H" executionInfo={"status": "ok", "timestamp": 1634893667652, "user_tz": -330, "elapsed": 8, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
class MovieGenreEmbedding(tf.keras.Model):
    def __init__(self, len_movies, len_genres, embedding_dim):
        super(MovieGenreEmbedding, self).__init__()
        self.m_g_input = tf.keras.layers.InputLayer(name='input_layer', input_shape=(2,))
        # embedding
        self.m_embedding = tf.keras.layers.Embedding(name='movie_embedding', input_dim=len_movies, output_dim=embedding_dim)
        self.g_embedding = tf.keras.layers.Embedding(name='genre_embedding', input_dim=len_genres, output_dim=embedding_dim)
        # dot product
        self.m_g_merge = tf.keras.layers.Dot(name='movie_genre_dot', normalize=True, axes=1)
        # output
        self.m_g_fc = tf.keras.layers.Dense(1, activation='sigmoid')
        
    def call(self, x):
        x = self.m_g_input(x)
        memb = self.m_embedding(x[0])
        gemb = self.g_embedding(x[1])
        m_g = self.m_g_merge([memb, gemb])
        return self.m_g_fc(m_g)

# class UserMovieEmbedding(tf.keras.Model):
#     def __init__(self, len_users, embedding_dim):
#         super(UserMovieEmbedding, self).__init__()
#         self.m_u_input = tf.keras.layers.InputLayer(name='input_layer', input_shape=(2,))
#         # embedding
#         self.u_embedding = tf.keras.layers.Embedding(name='user_embedding', input_dim=len_users, output_dim=embedding_dim)
#         # dot product
#         self.m_u_merge = tf.keras.layers.Dot(name='movie_user_dot', normalize=False, axes=1)
#         # output
#         self.m_u_fc = tf.keras.layers.Dense(1, activation='sigmoid')
        
#     def call(self, x):
#         x = self.m_u_input(x)
#         uemb = self.u_embedding(x[0])
#         m_u = self.m_u_merge([x[1], uemb])
#         return self.m_u_fc(m_u)


class UserMovieEmbedding(tf.keras.Model):
    def __init__(self, len_users, len_movies, embedding_dim):
        super(UserMovieEmbedding, self).__init__()
        self.m_u_input = tf.keras.layers.InputLayer(name='input_layer', input_shape=(2,))
        # embedding
        self.u_embedding = tf.keras.layers.Embedding(name='user_embedding', input_dim=len_users, output_dim=embedding_dim)
        self.m_embedding = tf.keras.layers.Embedding(name='movie_embedding', input_dim=len_movies, output_dim=embedding_dim)
        # dot product
        self.m_u_merge = tf.keras.layers.Dot(name='movie_user_dot', normalize=False, axes=1)
        # output
        self.m_u_fc = tf.keras.layers.Dense(1, activation='sigmoid')
        
    def call(self, x):
        x = self.m_u_input(x)
        uemb = self.u_embedding(x[0])
        memb = self.m_embedding(x[1])
        m_u = self.m_u_merge([memb, uemb])
        return self.m_u_fc(m_u)

# class UserMovieEmbedding(tf.keras.Model):
#     def __init__(self, len_users, len_movies, embedding_dim):
#         super(UserMovieEmbedding, self).__init__()
#         self.m_u_input = tf.keras.layers.InputLayer(name='input_layer', input_shape=(2,))
#         # embedding
#         self.u_embedding = tf.keras.layers.Embedding(name='user_embedding', input_dim=len_users, output_dim=embedding_dim)
#         self.m_embedding = tf.keras.layers.Embedding(name='movie_embedding', input_dim=len_movies, output_dim=embedding_dim)
#         # dot product
#         self.m_u_concat = tf.keras.layers.Concatenate(name='movie_user_concat', axis=1)
#         # output
#         self.m_u_fc = tf.keras.layers.Dense(1, activation='sigmoid')
        
#     def call(self, x):
#         x = self.m_u_input(x)
#         uemb = self.u_embedding(x[0])
#         memb = self.m_embedding(x[1])
#         m_u = self.m_u_concat([memb, uemb])
#         return self.m_u_fc(m_u)
```

<!-- #region id="hNXw90Fm7YMt" -->
### Environments
<!-- #endregion -->

```python id="dTJUXGCS7ZK_" executionInfo={"status": "ok", "timestamp": 1634893670252, "user_tz": -330, "elapsed": 4, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
class OfflineEnv(object):
    
    def __init__(self, users_dict, users_history_lens, movies_id_to_movies, state_size, fix_user_id=None):

        self.users_dict = users_dict
        self.users_history_lens = users_history_lens
        self.items_id_to_name = movies_id_to_movies
        
        self.state_size = state_size
        self.available_users = self._generate_available_users()

        self.fix_user_id = fix_user_id

        self.user = fix_user_id if fix_user_id else np.random.choice(self.available_users)
        self.user_items = {data[0]:data[1] for data in self.users_dict[self.user]}
        self.items = [data[0] for data in self.users_dict[self.user][:self.state_size]]
        self.done = False
        self.recommended_items = set(self.items)
        self.done_count = 3000
        
    def _generate_available_users(self):
        available_users = []
        for i, length in zip(self.users_dict.keys(), self.users_history_lens):
            if length > self.state_size:
                available_users.append(i)
        return available_users
    
    def reset(self):
        self.user = self.fix_user_id if self.fix_user_id else np.random.choice(self.available_users)
        self.user_items = {data[0]:data[1] for data in self.users_dict[self.user]}
        self.items = [data[0] for data in self.users_dict[self.user][:self.state_size]]
        self.done = False
        self.recommended_items = set(self.items)
        return self.user, self.items, self.done
        
    def step(self, action, top_k=False):

        reward = -0.5
        
        if top_k:
            correctly_recommended = []
            rewards = []
            for act in action:
                if act in self.user_items.keys() and act not in self.recommended_items:
                    correctly_recommended.append(act)
                    rewards.append((self.user_items[act] - 3)/2)
                else:
                    rewards.append(-0.5)
                self.recommended_items.add(act)
            if max(rewards) > 0:
                self.items = self.items[len(correctly_recommended):] + correctly_recommended
            reward = rewards

        else:
            if action in self.user_items.keys() and action not in self.recommended_items:
                reward = self.user_items[action] -3  # reward
            if reward > 0:
                self.items = self.items[1:] + [action]
            self.recommended_items.add(action)

        if len(self.recommended_items) > self.done_count or len(self.recommended_items) >= self.users_history_lens[self.user-1]:
            self.done = True
            
        return self.items, reward, self.done, self.recommended_items

    def get_items_names(self, items_ids):
        items_names = []
        for id in items_ids:
            try:
                items_names.append(self.items_id_to_name[str(id)])
            except:
                items_names.append(list(['Not in list']))
        return items_names
```

<!-- #region id="_Doz1pRW73Ao" -->
### State representation
<!-- #endregion -->

```python id="PVaHNAAE74uM" executionInfo={"status": "ok", "timestamp": 1634893673743, "user_tz": -330, "elapsed": 865, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
class DRRAveStateRepresentation(tf.keras.Model):
    def __init__(self, embedding_dim):
        super(DRRAveStateRepresentation, self).__init__()
        self.embedding_dim = embedding_dim
        self.wav = tf.keras.layers.Conv1D(1, 1, 1)
        self.concat = tf.keras.layers.Concatenate()
        self.flatten = tf.keras.layers.Flatten()
        
    def call(self, x):
        items_eb = tf.transpose(x[1], perm=(0,2,1))/self.embedding_dim
        wav = self.wav(items_eb)
        wav = tf.transpose(wav, perm=(0,2,1))
        wav = tf.squeeze(wav, axis=1)
        user_wav = tf.keras.layers.multiply([x[0], wav])
        concat = self.concat([x[0], user_wav, wav])
        return self.flatten(concat)
```

<!-- #region id="-05RUU3T5jBI" -->
## Model
<!-- #endregion -->

<!-- #region id="qqbhZWiM5m2W" -->
### Actor
<!-- #endregion -->

```python id="2-UDQp7T5lgx" executionInfo={"status": "ok", "timestamp": 1634893675206, "user_tz": -330, "elapsed": 4, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
class ActorNetwork(tf.keras.Model):
    def __init__(self, embedding_dim, hidden_dim):
        super(ActorNetwork, self).__init__()
        self.inputs = tf.keras.layers.InputLayer(name='input_layer', input_shape=(3*embedding_dim,))
        self.fc = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation='relu'),
            tf.keras.layers.Dense(hidden_dim, activation='relu'),
            tf.keras.layers.Dense(embedding_dim, activation='tanh')
        ])
        
    def call(self, x):
        x = self.inputs(x)
        return self.fc(x)

class Actor(object):
    
    def __init__(self, embedding_dim, hidden_dim, learning_rate, state_size, tau):
        
        self.embedding_dim = embedding_dim
        self.state_size = state_size
        
        # 엑터 네트워크 actor network / 타겟 네트워크 target network
        self.network = ActorNetwork(embedding_dim, hidden_dim)
        self.target_network = ActorNetwork(embedding_dim, hidden_dim)
        # 옵티마이저 optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        # 소프트 타겟 네트워크 업데이트 하이퍼파라미터 soft target network update hyperparameter
        self.tau = tau
    
    def build_networks(self):
        # 네트워크들 빌딩 / Build networks
        self.network(np.zeros((1, 3*self.embedding_dim)))
        self.target_network(np.zeros((1, 3*self.embedding_dim)))
    
    def update_target_network(self):
        # 소프트 타겟 네트워크 업데이트 soft target network update
        c_theta, t_theta = self.network.get_weights(), self.target_network.get_weights()
        for i in range(len(c_theta)):
            t_theta[i] = self.tau * c_theta[i] + (1 - self.tau) * t_theta[i]
        self.target_network.set_weights(t_theta)
        
    def train(self, states, dq_das):
        with tf.GradientTape() as g:
            outputs = self.network(states)
            # loss = outputs*dq_das
        dj_dtheta = g.gradient(outputs, self.network.trainable_weights, -dq_das)
        grads = zip(dj_dtheta, self.network.trainable_weights)
        self.optimizer.apply_gradients(grads)
        
    def save_weights(self, path):
        self.network.save_weights(path)
        
    def load_weights(self, path):
        self.network.load_weights(path)
```

<!-- #region id="gZS48eRq5tXb" -->
### Critic
<!-- #endregion -->

```python id="KbGCQ25y5pRI" executionInfo={"status": "ok", "timestamp": 1634893678654, "user_tz": -330, "elapsed": 6, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
class CriticNetwork(tf.keras.Model):
    def __init__(self, embedding_dim,hidden_dim):
        super(CriticNetwork, self).__init__()
        self.inputs = tf.keras.layers.InputLayer(input_shape=(embedding_dim, 3*embedding_dim))
        self.fc1 = tf.keras.layers.Dense(embedding_dim, activation = 'relu')
        self.concat = tf.keras.layers.Concatenate()
        self.fc2 = tf.keras.layers.Dense(hidden_dim, activation = 'relu')
        self.fc3 = tf.keras.layers.Dense(hidden_dim, activation = 'relu')
        self.out = tf.keras.layers.Dense(1, activation = 'linear')
        
    def call(self, x):
        s = self.fc1(x[1])
        s = self.concat([x[0],s])
        s = self.fc2(s)
        s = self.fc3(s)
        return self.out(s)

class Critic(object):
    
    def __init__(self, hidden_dim, learning_rate, embedding_dim, tau):
        
        self.embedding_dim = embedding_dim

        # 크리틱 네트워크 critic network / 타겟 네트워크 target network
        self.network = CriticNetwork(embedding_dim, hidden_dim)
        self.target_network = CriticNetwork(embedding_dim, hidden_dim)
        # 옵티마이저 optimizerq
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        # MSE
        self.loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

        # 소프트 타겟 네트워크 업데이트 하이퍼파라미터 soft target network update hyperparameter
        self.tau = tau

    def build_networks(self):
        self.network([np.zeros((1,self.embedding_dim)), np.zeros((1,3*self.embedding_dim))])
        self.target_network([np.zeros((1,self.embedding_dim)), np.zeros((1,3*self.embedding_dim))])
        self.network.compile(self.optimizer, self.loss)

    def update_target_network(self):
        c_omega = self.network.get_weights()
        t_omega = self.target_network.get_weights()
        for i in range(len(c_omega)):
            t_omega[i] = self.tau * c_omega[i] + (1 - self.tau) * t_omega[i]
        self.target_network.set_weights(t_omega)
        
    def dq_da(self, inputs):
        actions = inputs[0]
        states = inputs[1]
        with tf.GradientTape() as g:
            actions = tf.convert_to_tensor(actions)
            g.watch(actions)
            outputs = self.network([actions, states])
        q_grads = g.gradient(outputs, actions)
        return q_grads

    def train(self, inputs, td_targets, weight_batch):
        weight_batch = tf.convert_to_tensor(weight_batch, dtype=tf.float32)
        with tf.GradientTape() as g:
            outputs = self.network(inputs)
            loss = self.loss(td_targets, outputs)
            weighted_loss = tf.reduce_mean(loss*weight_batch)
        dl_domega = g.gradient(weighted_loss, self.network.trainable_weights)
        grads = zip(dl_domega, self.network.trainable_weights)
        self.optimizer.apply_gradients(grads)
        return weighted_loss


    def train_on_batch(self, inputs, td_targets, weight_batch):
        loss = self.network.train_on_batch(inputs, td_targets, sample_weight=weight_batch)
        return loss
            
    def save_weights(self, path):
        self.network.save_weights(path)
        
    def load_weights(self, path):
        self.network.load_weights(path)
```

<!-- #region id="2icy-3fu5uTW" -->
## Recommender
<!-- #endregion -->

```python id="XdkHhrVI8SFQ"
!mkdir -p save_weights
```

```python id="HASNEhqU8H0G" executionInfo={"status": "ok", "timestamp": 1634893696310, "user_tz": -330, "elapsed": 1446, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
# from replay_buffer import PriorityExperienceReplay
import tensorflow as tf
import numpy as np
from tensorflow.python.ops.gen_math_ops import Exp

# from actor import Actor
# from critic import Critic
# from replay_memory import ReplayMemory
# from embedding import MovieGenreEmbedding, UserMovieEmbedding
# from state_representation import DRRAveStateRepresentation

import matplotlib.pyplot as plt

# import wandb

class DRRAgent:
    
    def __init__(self, env, users_num, items_num, state_size, is_test=False, use_wandb=False):
        
        self.env = env

        self.users_num = users_num
        self.items_num = items_num
        
        self.embedding_dim = 100
        self.actor_hidden_dim = 128
        self.actor_learning_rate = 0.001
        self.critic_hidden_dim = 128
        self.critic_learning_rate = 0.001
        self.discount_factor = 0.9
        self.tau = 0.001

        self.replay_memory_size = 1000000
        self.batch_size = 32
        
        self.actor = Actor(self.embedding_dim, self.actor_hidden_dim, self.actor_learning_rate, state_size, self.tau)
        self.critic = Critic(self.critic_hidden_dim, self.critic_learning_rate, self.embedding_dim, self.tau)
        
        # self.m_embedding_network = MovieGenreEmbedding(items_num, 19, self.embedding_dim)
        # self.m_embedding_network([np.zeros((1,)),np.zeros((1,))])
        # self.m_embedding_network.load_weights('save_weights/m_g_model_weights.h5')

        self.embedding_network = UserMovieEmbedding(users_num, items_num, self.embedding_dim)
        self.embedding_network([np.zeros((1,)),np.zeros((1,))])
        # self.embedding_network = UserMovieEmbedding(users_num, self.embedding_dim)
        # self.embedding_network([np.zeros((1)),np.zeros((1,100))])
        self.embedding_network.load_weights('save_weights/user_movie_embedding_case4.h5')

        self.srm_ave = DRRAveStateRepresentation(self.embedding_dim)
        self.srm_ave([np.zeros((1, 100,)),np.zeros((1,state_size, 100))])

        # PER
        self.buffer = PriorityExperienceReplay(self.replay_memory_size, self.embedding_dim)
        self.epsilon_for_priority = 1e-6

        # ε-탐욕 탐색 하이퍼파라미터 ε-greedy exploration hyperparameter
        self.epsilon = 1.
        self.epsilon_decay = (self.epsilon - 0.1)/500000
        self.std = 1.5

        self.is_test = is_test

        # wandb
        self.use_wandb = use_wandb
        if use_wandb:
            wandb.init(project="drr", 
            entity="diominor",
            config={'users_num':users_num,
            'items_num' : items_num,
            'state_size' : state_size,
            'embedding_dim' : self.embedding_dim,
            'actor_hidden_dim' : self.actor_hidden_dim,
            'actor_learning_rate' : self.actor_learning_rate,
            'critic_hidden_dim' : self.critic_hidden_dim,
            'critic_learning_rate' : self.critic_learning_rate,
            'discount_factor' : self.discount_factor,
            'tau' : self.tau,
            'replay_memory_size' : self.replay_memory_size,
            'batch_size' : self.batch_size,
            'std_for_exploration': self.std})

    def calculate_td_target(self, rewards, q_values, dones):
        y_t = np.copy(q_values)
        for i in range(q_values.shape[0]):
            y_t[i] = rewards[i] + (1 - dones[i])*(self.discount_factor * q_values[i])
        return y_t

    def recommend_item(self, action, recommended_items, top_k=False, items_ids=None):
        if items_ids == None:
            items_ids = np.array(list(set(i for i in range(self.items_num)) - recommended_items))

        items_ebs = self.embedding_network.get_layer('movie_embedding')(items_ids)
        # items_ebs = self.m_embedding_network.get_layer('movie_embedding')(items_ids)
        action = tf.transpose(action, perm=(1,0))
        if top_k:
            item_indice = np.argsort(tf.transpose(tf.keras.backend.dot(items_ebs, action), perm=(1,0)))[0][-top_k:]
            return items_ids[item_indice]
        else:
            item_idx = np.argmax(tf.keras.backend.dot(items_ebs, action))
            return items_ids[item_idx]
        
    def train(self, max_episode_num, top_k=False, load_model=False):
        # 타겟 네트워크들 초기화
        self.actor.update_target_network()
        self.critic.update_target_network()

        if load_model:
            self.load_model("save_weights/actor_50000.h5", "save_weights/critic_50000.h5")
            print('Completely load weights!')

        episodic_precision_history = []

        for episode in range(max_episode_num):
            # episodic reward 리셋
            episode_reward = 0
            correct_count = 0
            steps = 0
            q_loss = 0
            mean_action = 0
            # Environment 리셋
            user_id, items_ids, done = self.env.reset()
            # print(f'user_id : {user_id}, rated_items_length:{len(self.env.user_items)}')
            # print('items : ', self.env.get_items_names(items_ids))
            while not done:
                
                # Observe current state & Find action
                ## Embedding 해주기
                user_eb = self.embedding_network.get_layer('user_embedding')(np.array(user_id))
                items_eb = self.embedding_network.get_layer('movie_embedding')(np.array(items_ids))
                # items_eb = self.m_embedding_network.get_layer('movie_embedding')(np.array(items_ids))
                ## SRM으로 state 출력
                state = self.srm_ave([np.expand_dims(user_eb, axis=0), np.expand_dims(items_eb, axis=0)])

                ## Action(ranking score) 출력
                action = self.actor.network(state)

                ## ε-greedy exploration
                if self.epsilon > np.random.uniform() and not self.is_test:
                    self.epsilon -= self.epsilon_decay
                    action += np.random.normal(0,self.std,size=action.shape)

                ## Item 추천
                recommended_item = self.recommend_item(action, self.env.recommended_items, top_k=top_k)
                
                # Calculate reward & observe new state (in env)
                ## Step
                next_items_ids, reward, done, _ = self.env.step(recommended_item, top_k=top_k)
                if top_k:
                    reward = np.sum(reward)

                # get next_state
                next_items_eb = self.embedding_network.get_layer('movie_embedding')(np.array(next_items_ids))
                # next_items_eb = self.m_embedding_network.get_layer('movie_embedding')(np.array(next_items_ids))
                next_state = self.srm_ave([np.expand_dims(user_eb, axis=0), np.expand_dims(next_items_eb, axis=0)])

                # buffer에 저장
                self.buffer.append(state, action, reward, next_state, done)
                
                if self.buffer.crt_idx > 1 or self.buffer.is_full:
                    # Sample a minibatch
                    batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones, weight_batch, index_batch = self.buffer.sample(self.batch_size)

                    # Set TD targets
                    target_next_action= self.actor.target_network(batch_next_states)
                    qs = self.critic.network([target_next_action, batch_next_states])
                    target_qs = self.critic.target_network([target_next_action, batch_next_states])
                    min_qs = tf.raw_ops.Min(input=tf.concat([target_qs, qs], axis=1), axis=1, keep_dims=True) # Double Q method
                    td_targets = self.calculate_td_target(batch_rewards, min_qs, batch_dones)
        
                    # Update priority
                    for (p, i) in zip(td_targets, index_batch):
                        self.buffer.update_priority(abs(p[0]) + self.epsilon_for_priority, i)

                    # print(weight_batch.shape)
                    # print(td_targets.shape)
                    # raise Exception
                    # Update critic network
                    q_loss += self.critic.train([batch_actions, batch_states], td_targets, weight_batch)
                    
                    # Update actor network
                    s_grads = self.critic.dq_da([batch_actions, batch_states])
                    self.actor.train(batch_states, s_grads)
                    self.actor.update_target_network()
                    self.critic.update_target_network()

                items_ids = next_items_ids
                episode_reward += reward
                mean_action += np.sum(action[0])/(len(action[0]))
                steps += 1

                if reward > 0:
                    correct_count += 1
                
                print(f'recommended items : {len(self.env.recommended_items)},  epsilon : {self.epsilon:0.3f}, reward : {reward:+}', end='\r')

                if done:
                    print()
                    precision = int(correct_count/steps * 100)
                    print(f'{episode}/{max_episode_num}, precision : {precision:2}%, total_reward:{episode_reward}, q_loss : {q_loss/steps}, mean_action : {mean_action/steps}')
                    if self.use_wandb:
                        wandb.log({'precision':precision, 'total_reward':episode_reward, 'epsilone': self.epsilon, 'q_loss' : q_loss/steps, 'mean_action' : mean_action/steps})
                    episodic_precision_history.append(precision)
             
            if (episode+1)%50 == 0:
                plt.plot(episodic_precision_history)
                plt.savefig(f'images/training_precision_%_top_5.png')

            if (episode+1)%1000 == 0:
                self.save_model(f'save_weights/actor_{episode+1}_fixed.h5',
                                f'save_weights/critic_{episode+1}_fixed.h5')

    def save_model(self, actor_path, critic_path):
        self.actor.save_weights(actor_path)
        self.critic.save_weights(critic_path)
        
    def load_model(self, actor_path, critic_path):
        self.actor.load_weights(actor_path)
        self.critic.load_weights(critic_path)
```

<!-- #region id="CmDzqkpk8ZLy" -->
## Training
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 1000} id="ASRF1pBUA9hf" executionInfo={"status": "error", "timestamp": 1634893830572, "user_tz": -330, "elapsed": 9805, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="d4526418-26cb-4f18-f004-fae1285238c8"
import pandas as pd
import numpy as np
import tensorflow as tf
import itertools
import matplotlib.pyplot as plt
import time
import os

MAX_EPISODE_NUM = 8000

# os.environ["CUDA_VISIBLE_DEVICES"]="1"

if __name__ == "__main__":

    print('Data loading...')

    #Loading datasets
    ratings_list = [i.strip().split("::") for i in open(os.path.join(DATA_DIR,'ratings.dat'), 'r').readlines()]
    users_list = [i.strip().split("::") for i in open(os.path.join(DATA_DIR,'users.dat'), 'r').readlines()]
    movies_list = [i.strip().split("::") for i in open(os.path.join(DATA_DIR,'movies.dat'),encoding='latin-1').readlines()]
    ratings_df = pd.DataFrame(ratings_list, columns = ['UserID', 'MovieID', 'Rating', 'Timestamp'], dtype = np.uint32)
    movies_df = pd.DataFrame(movies_list, columns = ['MovieID', 'Title', 'Genres'])
    movies_df['MovieID'] = movies_df['MovieID'].apply(pd.to_numeric)

    print("Data loading complete!")
    print("Data preprocessing...")

    # 영화 id를 영화 제목으로
    movies_id_to_movies = {movie[0]: movie[1:] for movie in movies_list}
    ratings_df = ratings_df.applymap(int)

    # 유저별로 본 영화들 순서대로 정리
    users_dict = np.load('user_dict.npy', allow_pickle=True)

    # 각 유저별 영화 히스토리 길이
    users_history_lens = np.load('users_histroy_len.npy')

    users_num = max(ratings_df["UserID"])+1
    items_num = max(ratings_df["MovieID"])+1

    # Training setting
    train_users_num = int(users_num * 0.8)
    train_items_num = items_num
    train_users_dict = {k:users_dict.item().get(k) for k in range(1, train_users_num+1)}
    train_users_history_lens = users_history_lens[:train_users_num]
    
    print('DONE!')
    time.sleep(2)

    env = OfflineEnv(train_users_dict, train_users_history_lens, movies_id_to_movies, STATE_SIZE)
    recommender = DRRAgent(env, users_num, items_num, STATE_SIZE, use_wandb=False)
    recommender.actor.build_networks()
    recommender.critic.build_networks()
    recommender.train(MAX_EPISODE_NUM, load_model=False)
```

<!-- #region id="rgv0oXzeBE80" -->
## Evaluation
<!-- #endregion -->

```python id="1ZsADhYBPHhK"
!pip install -q wandb
```

```python id="0xHmoWGLPC4O" executionInfo={"status": "ok", "timestamp": 1634893350496, "user_tz": -330, "elapsed": 465, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
#Dependencies
import pandas as pd
import numpy as np
import tensorflow as tf
import itertools
import matplotlib.pyplot as plt
import time

# from envs import OfflineEnv
# from recommender import DRRAgent

import os

ROOT_DIR = os.getcwd()
DATA_DIR = os.path.join(ROOT_DIR, 'ml-1m/')
STATE_SIZE = 10
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="BFkzwXrSPERC" executionInfo={"status": "ok", "timestamp": 1634893401578, "user_tz": -330, "elapsed": 3363, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="0d2dabe4-679a-4f81-82f9-69ba5611f500"
#Loading datasets
ratings_list = [i.strip().split("::") for i in open(os.path.join(DATA_DIR,'ratings.dat'), 'r').readlines()]
users_list = [i.strip().split("::") for i in open(os.path.join(DATA_DIR,'users.dat'), 'r').readlines()]
movies_list = [i.strip().split("::") for i in open(os.path.join(DATA_DIR,'movies.dat'),encoding='latin-1').readlines()]
ratings_df = pd.DataFrame(ratings_list, columns = ['UserID', 'MovieID', 'Rating', 'Timestamp'], dtype = np.uint32)
movies_df = pd.DataFrame(movies_list, columns = ['MovieID', 'Title', 'Genres'])
movies_df['MovieID'] = movies_df['MovieID'].apply(pd.to_numeric)
movies_id_to_movies = {movie[0]: movie[1:] for movie in movies_list}

movies_df.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="qNNVg5zuPS__" executionInfo={"status": "ok", "timestamp": 1634893434144, "user_tz": -330, "elapsed": 2577, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="9a9b6adb-6ff3-46c7-e56f-07fe5899b1bc"
ratings_df = ratings_df.applymap(int)
users_dict = {user : [] for user in set(ratings_df["UserID"])}
ratings_df = ratings_df.sort_values(by='Timestamp', ascending=True)

ratings_df.head(5)
```

```python id="U_C3qq2_PgAB" executionInfo={"status": "ok", "timestamp": 1634893600540, "user_tz": -330, "elapsed": 92522, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
ratings_df_gen = ratings_df.iterrows()
users_dict_for_history_len = {user : [] for user in set(ratings_df["UserID"])}

for data in ratings_df_gen:
    users_dict[data[1]['UserID']].append((data[1]['MovieID'], data[1]['Rating']))
    if data[1]['Rating'] >= 4:
        users_dict_for_history_len[data[1]['UserID']].append((data[1]['MovieID'], data[1]['Rating']))

users_history_lens = [len(users_dict_for_history_len[u]) for u in set(ratings_df["UserID"])]

users_num = max(ratings_df["UserID"])+1
items_num = max(ratings_df["MovieID"])+1

train_users_num = int(users_num * 0.8)
train_items_num = items_num
train_users_dict = {k:users_dict[k] for k in range(1, train_users_num+1)}
train_users_history_lens = users_history_lens[:train_users_num]

eval_users_num = int(users_num * 0.2)
eval_items_num = items_num
eval_users_dict = {k:users_dict[k] for k in range(users_num-eval_users_num, users_num)}
eval_users_history_lens = users_history_lens[-eval_users_num:]
```

```python id="qoQLlDvGPyis" executionInfo={"status": "ok", "timestamp": 1634893600546, "user_tz": -330, "elapsed": 29, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
def evaluate(recommender, env, check_movies = False, top_k=False):

    # episodic reward 리셋
    episode_reward = 0
    steps = 0
    mean_precision = 0
    mean_ndcg = 0
    # Environment 리셋
    user_id, items_ids, done = env.reset()
    
    if check_movies:
        print(f'user_id : {user_id}, rated_items_length:{len(env.user_items)}')
        print('items : \n', np.array(env.get_items_names(items_ids)))

    while not done:

        # Observe current state & Find action
        ## Embedding 해주기
        user_eb = recommender.embedding_network.get_layer('user_embedding')(np.array(user_id))
        items_eb = recommender.embedding_network.get_layer('movie_embedding')(np.array(items_ids))
        ## SRM으로 state 출력
        state = recommender.srm_ave([np.expand_dims(user_eb, axis=0), np.expand_dims(items_eb, axis=0)])
        ## Action(ranking score) 출력
        action = recommender.actor.network(state)
        ## Item 추천
        recommended_item = recommender.recommend_item(action, env.recommended_items, top_k=top_k)
        if check_movies:
            print(f'recommended items ids : {recommended_item}')
            print(f'recommened items : \n {np.array(env.get_items_names(recommended_item), dtype=object)}')
        # Calculate reward & observe new state (in env)
        ## Step
        next_items_ids, reward, done, _ = env.step(recommended_item, top_k=top_k)
        if top_k:
            correct_list = [1 if r > 0 else 0 for r in reward]
            # ndcg
            dcg, idcg = calculate_ndcg(correct_list, [1 for _ in range(len(reward))])
            mean_ndcg += dcg/idcg
            
            #precision
            correct_num = top_k-correct_list.count(0)
            mean_precision += correct_num/top_k
            
        reward = np.sum(reward)
        items_ids = next_items_ids
        episode_reward += reward
        steps += 1
        
        if check_movies:
            print(f'precision : {correct_num/top_k}, dcg : {dcg:0.3f}, idcg : {idcg:0.3f}, ndcg : {dcg/idcg:0.3f}, reward : {reward}')
            print()
        break
    
    if check_movies:
        print(f'precision : {mean_precision/steps}, ngcg : {mean_ndcg/steps}, episode_reward : {episode_reward}')
        print()
    
    return mean_precision/steps, mean_ndcg/steps

def calculate_ndcg(rel, irel):
    dcg = 0
    idcg = 0
    rel = [1 if r>0 else 0 for r in rel]
    for i, (r, ir) in enumerate(zip(rel, irel)):
        dcg += (r)/np.log2(i+2)
        idcg += (ir)/np.log2(i+2)
    return dcg, idcg
```

```python colab={"base_uri": "https://localhost:8080/", "height": 493} id="Z2fj96xLP4be" executionInfo={"status": "error", "timestamp": 1634893722396, "user_tz": -330, "elapsed": 2777, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="d368aa86-821a-4b7c-c729-ebf090278897"
tf.keras.backend.set_floatx('float64')
sum_precision = 0
sum_ndcg = 0
TOP_K = 10

for user_id in eval_users_dict.keys():
    env = OfflineEnv(eval_users_dict, users_history_lens, movies_id_to_movies, STATE_SIZE, fix_user_id=user_id)
    recommender = DRRAgent(env, users_num, items_num, STATE_SIZE)
    recommender.actor.build_networks()
    recommender.critic.build_networks()
    recommender.load_model('save_weights/actor_10000.h5', 
                           'save_weights/critic_10000.h5')
    precision, ndcg = evaluate(recommender, env, top_k=TOP_K)
    sum_precision += precision
    sum_ndcg += ndcg
    
print(f'precision@{TOP_K} : {sum_precision/len(eval_users_dict)}, ndcg@{TOP_K} : {sum_ndcg/len(eval_users_dict)}')
```

```python id="Yyg5ciHvP7Y1" executionInfo={"status": "aborted", "timestamp": 1634893603360, "user_tz": -330, "elapsed": 18, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
sum_precision = 0
sum_ndcg = 0
TOP_K = 10

for user_id in eval_users_dict.keys():
    env = OfflineEnv(eval_users_dict, users_history_lens, movies_id_to_movies, STATE_SIZE, fix_user_id=user_id)
    recommender = DRRAgent(env, users_num, items_num, STATE_SIZE)
    recommender.actor.build_networks()
    recommender.critic.build_networks()
    recommender.load_model('save_weights/actor_8000.h5', 
                           'save_weights/critic_8000.h5')
    precision, ndcg = evaluate(recommender, env, TOP_K=10)
    sum_precision += precision
    sum_ndcg += ndcg
    
print(f'precision@{TOP_K} : {sum_precision/len(eval_users_dict)}, ndcg@{TOP_K} : {sum_ndcg/len(eval_users_dict)}')
```

```python id="hMKrnTrtQCKi"

```
