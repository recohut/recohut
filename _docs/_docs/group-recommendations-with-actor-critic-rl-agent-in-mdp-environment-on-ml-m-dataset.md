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

<!-- #region id="Rgy-wkK33glM" -->
# Group Recommendations with Actor-critic RL Agent in MDP Environment on ML-1m Dataset
<!-- #endregion -->

<!-- #region id="lF29_ys12dy1" -->
<img src='https://github.com/RecoHut-Stanzas/S758139/raw/main/images/group_recommender_actorcritic_1.svg'>
<!-- #endregion -->

<!-- #region id="ILGqY7FA2o8P" -->
## **Step 1 - Setup the environment**
<!-- #endregion -->

<!-- #region id="9CEauQyaqpzo" -->
### **1.1 Install libraries**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="mI-Hbwwv25Z1" outputId="4d175389-bb58-41be-d985-9a856650a183" executionInfo={"status": "ok", "timestamp": 1639924278295, "user_tz": -330, "elapsed": 8915, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
!pip install -q -U git+https://github.com/RecoHut-Projects/recohut.git -b v0.0.3
```

<!-- #region id="vRf7z5fC24vu" -->
### **1.2 Download datasets**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="sQI_J4hQ28hk" outputId="aa2671a6-cd78-4765-c8e2-d65da2b9b301" executionInfo={"status": "ok", "timestamp": 1639924282758, "user_tz": -330, "elapsed": 596, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
!wget -q --show-progress https://files.grouplens.org/datasets/movielens/ml-1m.zip
```

<!-- #region id="JQ6u2WK73Awv" -->
### **1.3 Import libraries**
<!-- #endregion -->

```python id="9r8-VYWyS0rK" executionInfo={"status": "ok", "timestamp": 1639925612894, "user_tz": -330, "elapsed": 483, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
from typing import Tuple, List, Dict

import os
import pandas as pd
from collections import deque, defaultdict
import shutil
import zipfile

import torch
import numpy as np
from scipy.sparse import coo_matrix
```

```python id="-0mNDMpqqx80" executionInfo={"status": "ok", "timestamp": 1639924413693, "user_tz": -330, "elapsed": 6, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
# Utils
from recohut.transforms.user_grouping import GroupGenerator
from recohut.layers.ou_noise import OUNoise

# Models
from recohut.models.actor_critic import Actor, Critic
from recohut.models.embedding import GroupEmbedding

# RL
from recohut.rl.memory import ReplayMemory
from recohut.rl.agents.ddpg import DDPGAgent
from recohut.rl.envs.recsys import Env
```

<!-- #region id="5lye2WyJ5OGv" -->
### **1.4 Set params**
<!-- #endregion -->

```python id="T3ILQ6MYTaZK" executionInfo={"status": "ok", "timestamp": 1639925968893, "user_tz": -330, "elapsed": 6, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
class Config(object):
    """
    Configurations
    """

    def __init__(self):
        # Data
        self.data_folder_path = './data/silver'
        self.item_path = os.path.join(self.data_folder_path, 'movies.dat')
        self.user_path = os.path.join(self.data_folder_path, 'users.dat')
        self.group_path = os.path.join(self.data_folder_path, 'groupMember.dat')
        self.saves_folder_path = os.path.join('saves')

        # Recommendation system
        self.history_length = 5
        self.top_K_list = [5, 10, 20]
        self.rewards = [0, 1]

        # Reinforcement learning
        self.embedding_size = 32
        self.state_size = self.history_length + 1
        self.action_size = 1
        self.embedded_state_size = self.state_size * self.embedding_size
        self.embedded_action_size = self.action_size * self.embedding_size

        # Numbers
        self.item_num = None
        self.user_num = None
        self.group_num = None
        self.total_group_num = None

        # Environment
        self.env_n_components = self.embedding_size
        self.env_tol = 1e-4
        self.env_max_iter = 1000
        self.env_alpha = 0.001

        # Actor-Critic network
        self.actor_hidden_sizes = (128, 64)
        self.critic_hidden_sizes = (32, 16)

        # DDPG algorithm
        self.tau = 1e-3
        self.gamma = 0.9

        # Optimizer
        self.batch_size = 64
        self.buffer_size = 100000
        self.num_episodes = 10 # recommended = 1000
        self.num_steps = 5 # recommended = 100
        self.embedding_weight_decay = 1e-6
        self.actor_weight_decay = 1e-6
        self.critic_weight_decay = 1e-6
        self.embedding_learning_rate = 1e-4
        self.actor_learning_rate = 1e-4
        self.critic_learning_rate = 1e-4
        self.eval_per_iter = 10

        # OU noise
        self.ou_mu = 0.0
        self.ou_theta = 0.15
        self.ou_sigma = 0.2
        self.ou_epsilon = 1.0

        # GPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
```

<!-- #region id="gYytf-uj5Z1t" -->
## **Step 2 - Data preparation**
<!-- #endregion -->

```python id="3zm0w7K8H_eq" executionInfo={"status": "ok", "timestamp": 1639925062899, "user_tz": -330, "elapsed": 6, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
data_path = './ml-1m'
output_path = './data/silver'
```

```python colab={"base_uri": "https://localhost:8080/", "height": 192} id="z1PPWBeyJpAb" executionInfo={"status": "ok", "timestamp": 1639925479246, "user_tz": -330, "elapsed": 399021, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="ae8deec5-e370-48d5-e8fc-f276ebbd9a27"
ratings = pd.read_csv(os.path.join(data_path,'ratings.dat'), sep='::', engine='python', header=None)

group_generator = GroupGenerator(
                    user_ids=np.arange(ratings[0].max()+1),
                    item_ids=np.arange(ratings[1].max()+1),
                    ratings=ratings,
                    output_path=output_path,
                    rating_threshold=4,
                    num_groups=1000,
                    group_sizes=[2, 3, 4, 5],
                    min_num_ratings=20,
                    train_ratio=0.7,
                    val_ratio=0.1,
                    negative_sample_size=100,
                    verbose=True)

shutil.copyfile(src=os.path.join(data_path, 'movies.dat'), dst=os.path.join(output_path, 'movies.dat'))
shutil.copyfile(src=os.path.join(data_path, 'users.dat'), dst=os.path.join(output_path, 'users.dat'))
```

```python colab={"base_uri": "https://localhost:8080/"} id="vdObJCf2LT4M" executionInfo={"status": "ok", "timestamp": 1639925507964, "user_tz": -330, "elapsed": 9, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="f0f74425-0dbb-458a-aa99-5359c938534e"
os.listdir(output_path)
```

```python id="TebaeBH1Lgv5" executionInfo={"status": "ok", "timestamp": 1639925619974, "user_tz": -330, "elapsed": 1235, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
class DataLoader(object):
    """
    Data Loader
    """

    def __init__(self, config: Config):
        """
        Initialize DataLoader
        :param config: configurations
        """
        self.config = config
        self.history_length = config.history_length
        self.item_num = self.get_item_num()
        self.user_num = self.get_user_num()
        self.group_num, self.total_group_num, self.group2members_dict, self.user2group_dict = self.get_groups()

        if not os.path.exists(self.config.saves_folder_path):
            os.mkdir(self.config.saves_folder_path)

    def get_item_num(self) -> int:
        """
        Get number of items
        :return: number of items
        """
        df_item = pd.read_csv(self.config.item_path, sep='::', index_col=0, engine='python')
        self.config.item_num = df_item.index.max()
        return self.config.item_num

    def get_user_num(self) -> int:
        """
        Get number of users
        :return: number of users
        """
        df_user = pd.read_csv(self.config.user_path, sep='::', index_col=0, engine='python')
        self.config.user_num = df_user.index.max()
        return self.config.user_num

    def get_groups(self):
        """
        Get number of groups and group members
        :return: group_num, total_group_num, group2members_dict, user2group_dict
        """
        df_group = pd.read_csv(self.config.group_path, sep=' ', header=None, index_col=None,
                               names=['GroupID', 'Members'])
        df_group['Members'] = df_group['Members']. \
            apply(lambda group_members: tuple(map(int, group_members.split(','))))
        group_num = df_group['GroupID'].max()

        users = set()
        for members in df_group['Members']:
            users.update(members)
        users = sorted(users)
        total_group_num = group_num + len(users)

        df_user_group = pd.DataFrame()
        df_user_group['GroupID'] = list(range(group_num + 1, total_group_num + 1))
        df_user_group['Members'] = [(user,) for user in users]
        df_group = df_group.append(df_user_group, ignore_index=True)
        group2members_dict = {row['GroupID']: row['Members'] for _, row in df_group.iterrows()}
        user2group_dict = {user: group_num + user_index + 1 for user_index, user in enumerate(users)}

        self.config.group_num = group_num
        self.config.total_group_num = total_group_num
        return group_num, total_group_num, group2members_dict, user2group_dict

    def load_rating_data(self, mode: str, dataset_name: str, is_appended=True) -> pd.DataFrame():
        """
        Load rating data
        :param mode: in ['user', 'group']
        :param dataset_name: name of the dataset in ['train', 'val', 'test']
        :param is_appended: True to append all datasets before this dataset
        :return: df_rating
        """
        assert (mode in ['user', 'group']) and (dataset_name in ['train', 'val', 'test'])
        rating_path = os.path.join(self.config.data_folder_path, mode + 'Rating' + dataset_name.capitalize() + '.dat')
        df_rating_append = pd.read_csv(rating_path, sep=' ', header=None, index_col=None,
                                       names=['GroupID', 'MovieID', 'Rating', 'Timestamp'])
        print('Read data:', rating_path)

        if is_appended:
            if dataset_name == 'train':
                df_rating = df_rating_append
            elif dataset_name == 'val':
                df_rating = self.load_rating_data(mode=mode, dataset_name='train')
                df_rating = df_rating.append(df_rating_append, ignore_index=True)
            else:
                df_rating = self.load_rating_data(mode=mode, dataset_name='val')
                df_rating = df_rating.append(df_rating_append, ignore_index=True)
        else:
            df_rating = df_rating_append

        return df_rating

    def _load_rating_matrix(self, df_rating: pd.DataFrame()):
        """
        Load rating matrix
        :param df_rating: rating data
        :return: rating_matrix
        """
        group_ids = df_rating['GroupID']
        item_ids = df_rating['MovieID']
        ratings = df_rating['Rating']
        rating_matrix = coo_matrix((ratings, (group_ids, item_ids)),
                                   shape=(self.total_group_num + 1, self.config.item_num + 1)).tocsr()
        return rating_matrix

    def load_rating_matrix(self, dataset_name: str):
        """
        Load group rating matrix
        :param dataset_name: name of the dataset in ['train', 'val', 'test']
        :return: rating_matrix
        """
        assert dataset_name in ['train', 'val', 'test']

        df_user_rating = self.user2group(self.load_rating_data(mode='user', dataset_name=dataset_name))
        df_group_rating = self.load_rating_data(mode='group', dataset_name=dataset_name)
        df_group_rating = df_group_rating.append(df_user_rating, ignore_index=True)
        rating_matrix = self._load_rating_matrix(df_group_rating)

        return rating_matrix

    def user2group(self, df_user_rating):
        """
        Change user ids to group ids
        :param df_user_rating: user rating
        :return: df_user_rating
        """
        df_user_rating['GroupID'] = df_user_rating['GroupID'].apply(lambda user_id: self.user2group_dict[user_id])
        return df_user_rating

    def _load_eval_data(self, df_data_train: pd.DataFrame(), df_data_eval: pd.DataFrame(),
                        negative_samples_dict: Dict[tuple, list]) -> pd.DataFrame():
        """
        Write evaluation data
        :param df_data_train: train data
        :param df_data_eval: evaluation data
        :param negative_samples_dict: one dictionary mapping (group_id, item_id) to negative samples
        :return: data for evaluation
        """
        df_eval = pd.DataFrame()
        last_state_dict = defaultdict(list)
        groups = []
        histories = []
        actions = []
        negative_samples = []

        for group_id, rating_group in df_data_train.groupby(['GroupID']):
            rating_group.sort_values(by=['Timestamp'], ascending=True, ignore_index=True, inplace=True)
            state = rating_group[rating_group['Rating'] == 1]['MovieID'].values.tolist()
            last_state_dict[group_id] = state[-self.config.history_length:]

        for group_id, rating_group in df_data_eval.groupby(['GroupID']):
            rating_group.sort_values(by=['Timestamp'], ascending=True, ignore_index=True, inplace=True)
            action = rating_group[rating_group['Rating'] == 1]['MovieID'].values.tolist()
            state = deque(maxlen=self.history_length)
            state.extend(last_state_dict[group_id])
            for item_id in action:
                if len(state) == self.config.history_length:
                    groups.append(group_id)
                    histories.append(list(state))
                    actions.append(item_id)
                    negative_samples.append(negative_samples_dict[(group_id, item_id)])
                state.append(item_id)

        df_eval['group'] = groups
        df_eval['history'] = histories
        df_eval['action'] = actions
        df_eval['negative samples'] = negative_samples

        return df_eval

    def load_negative_samples(self, mode: str, dataset_name: str):
        """
        Load negative samples
        :param mode: in ['user', 'group']
        :param dataset_name: name of the dataset in ['val', 'test']
        :return: negative_samples_dict
        """
        assert (mode in ['user', 'group']) and (dataset_name in ['val', 'test'])
        negative_samples_path = os.path.join(self.config.data_folder_path, mode + 'Rating'
                                             + dataset_name.capitalize() + 'Negative.dat')
        negative_samples_dict = {}

        with open(negative_samples_path, 'r') as negative_samples_file:
            for line in negative_samples_file.readlines():
                negative_samples = line.split()
                ids = negative_samples[0][1:-1].split(',')
                group_id = int(ids[0])
                if mode == 'user':
                    group_id = self.user2group_dict[group_id]
                item_id = int(ids[1])
                negative_samples = list(map(int, negative_samples[1:]))
                negative_samples_dict[(group_id, item_id)] = negative_samples

        return negative_samples_dict

    def load_eval_data(self, mode: str, dataset_name: str, reload=False):
        """
        Load evaluation data
        :param mode: in ['user', 'group']
        :param dataset_name: in ['val', 'test']
        :param reload: True to reload the dataset file
        :return: data for evaluation
        """
        assert (mode in ['user', 'group']) and (dataset_name in ['val', 'test'])
        exp_eval_path = os.path.join(self.config.saves_folder_path, 'eval_' + mode + '_' + dataset_name + '_'
                                     + str(self.config.history_length) + '.pkl')

        if reload or not os.path.exists(exp_eval_path):
            if dataset_name == 'val':
                df_rating_train = self.load_rating_data(mode=mode, dataset_name='train')
            else:
                df_rating_train = self.load_rating_data(mode=mode, dataset_name='val')
            df_rating_eval = self.load_rating_data(mode=mode, dataset_name=dataset_name, is_appended=False)

            if mode == 'user':
                df_rating_train = self.user2group(df_rating_train)
                df_rating_eval = self.user2group(df_rating_eval)

            negative_samples_dict = self.load_negative_samples(mode=mode, dataset_name=dataset_name)
            df_eval = self._load_eval_data(df_rating_train, df_rating_eval, negative_samples_dict)
            df_eval.to_pickle(exp_eval_path)
            print('Save data:', exp_eval_path)
        else:
            df_eval = pd.read_pickle(exp_eval_path)
            print('Load data:', exp_eval_path)

        return df_eval
```

<!-- #region id="ygNPCx5o5yAv" -->
## **Step 3 - Training & Evaluation**
<!-- #endregion -->

```python id="eD8tNUa5MBH3" executionInfo={"status": "ok", "timestamp": 1639925693008, "user_tz": -330, "elapsed": 744, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
class Evaluator(object):
    """
    Evaluator
    """

    def __init__(self, config: Config):
        """
        Initialize Evaluator
        :param config: configurations
        """
        self.config = config

    def evaluate(self, agent: DDPGAgent, df_eval: pd.DataFrame(), mode: str, top_K=5):
        """
        Evaluate the agent
        :param agent: agent
        :param df_eval: evaluation data
        :param mode: in ['user', 'group']
        :param top_K: length of the recommendation list
        :return: avg_recall_score, avg_ndcg_score
        """
        recall_scores = []
        ndcg_scores = []

        for _, row in df_eval.iterrows():
            group = row['group']
            history = row['history']
            item_true = row['action']
            item_candidates = row['negative samples'] + [item_true]
            np.random.shuffle(item_candidates)

            state = [group] + history
            items_pred = agent.get_action(state=state, item_candidates=item_candidates, top_K=top_K)

            recall_score = 0
            ndcg_score = 0

            for k, item in enumerate(items_pred):
                if item == item_true:
                    recall_score = 1
                    ndcg_score = np.log2(2) / np.log2(k + 2)
                    break

            recall_scores.append(recall_score)
            ndcg_scores.append(ndcg_score)

        avg_recall_score = float(np.mean(recall_scores))
        avg_ndcg_score = float(np.mean(ndcg_scores))
        print('%s: Recall@%d = %.4f, NDCG@%d = %.4f' % (mode.capitalize(), top_K, avg_recall_score,
                                                        top_K, avg_ndcg_score))
        return avg_recall_score, avg_ndcg_score
```

```python id="nGHg0vTKMRyA" executionInfo={"status": "ok", "timestamp": 1639925766520, "user_tz": -330, "elapsed": 780, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
def train(config: Config, env: Env, agent: DDPGAgent, evaluator: Evaluator,
          df_eval_user: pd.DataFrame(), df_eval_group: pd.DataFrame()):
    """
    Train the agent with the environment
    :param config: configurations
    :param env: environment
    :param agent: agent
    :param evaluator: evaluator
    :param df_eval_user: user evaluation data
    :param df_eval_group: group evaluation data
    :return:
    """
    rewards = []
    for episode in range(config.num_episodes):
        state = env.reset()
        agent.noise.reset()
        episode_reward = 0

        for step in range(config.num_steps):
            action = agent.get_action(state)
            new_state, reward, _, _ = env.step(action)
            agent.replay_memory.push((state, action, reward, new_state))
            state = new_state
            episode_reward += reward

            if len(agent.replay_memory) >= config.batch_size:
                agent.update()

        rewards.append(episode_reward / config.num_steps)
        print('Episode = %d, average reward = %.4f' % (episode, episode_reward / config.num_steps))
        if (episode + 1) % config.eval_per_iter == 0:
            for top_K in config.top_K_list:
                evaluator.evaluate(agent=agent, df_eval=df_eval_user, mode='user', top_K=top_K)
            for top_K in config.top_K_list:
                evaluator.evaluate(agent=agent, df_eval=df_eval_group, mode='group', top_K=top_K)
```

```python colab={"base_uri": "https://localhost:8080/"} id="4v19AhNLMWf2" executionInfo={"status": "ok", "timestamp": 1639926216622, "user_tz": -330, "elapsed": 68010, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="e6b92bad-53bb-4b2f-bd1e-b85a399f55c4"
config = Config()
dataloader = DataLoader(config)
rating_matrix_train = dataloader.load_rating_matrix(dataset_name='val')
df_eval_user_test = dataloader.load_eval_data(mode='user', dataset_name='test')
df_eval_group_test = dataloader.load_eval_data(mode='group', dataset_name='test')
env = Env(config=config, rating_matrix=rating_matrix_train, dataset_name='val')
noise = OUNoise(embedded_action_size=config.embedded_action_size, ou_mu=config.ou_mu,
        ou_theta=config.ou_theta, ou_sigma=config.ou_sigma, ou_epsilon=config.ou_epsilon)
agent = DDPGAgent(config=config, noise=noise, group2members_dict=dataloader.group2members_dict, verbose=True)
evaluator = Evaluator(config=config)
train(config=config, env=env, agent=agent, evaluator=evaluator,
        df_eval_user=df_eval_user_test, df_eval_group=df_eval_group_test)
```

<!-- #region id="wycnKnps68l3" -->
## **Closure**
<!-- #endregion -->

<!-- #region id="BMk2vGOh6-cp" -->
For more details, you can refer to https://github.com/RecoHut-Stanzas/S758139.
<!-- #endregion -->

<!-- #region id="f-_pvwaI7bNX" -->
<a href="https://github.com/RecoHut-Stanzas/S758139/blob/main/reports/S758139_Report.ipynb" alt="S758139_Report"> <img src="https://img.shields.io/static/v1?label=report&message=active&color=green" /></a> <a href="https://github.com/RecoHut-Stanzas/S758139" alt="S758139"> <img src="https://img.shields.io/static/v1?label=code&message=github&color=blue" /></a>
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="LgrJODmhbVxS" outputId="5b92a234-34dc-4252-bf8c-c8c0adabb158" executionInfo={"status": "ok", "timestamp": 1639926294385, "user_tz": -330, "elapsed": 4373, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
!pip install -q watermark
%reload_ext watermark
%watermark -a "Sparsh A." -m -iv -u -t -d
```

<!-- #region id="hKleh6bfbVxT" -->
---
<!-- #endregion -->

<!-- #region id="9MbAhyUhbVxU" -->
**END**
<!-- #endregion -->
