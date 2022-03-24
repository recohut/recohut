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

<!-- #region id="_KBYAkCJL_TJ" -->
# Deep Reinforcement Learning based Group Recommender System

`ActorCriticNetwork`, `DDPG`, `GroupRecommendation`, `MovieLens1M`, `PyTorch`, `ReplayMemory`
<!-- #endregion -->

<!-- #region id="qdBrR-7dC3S_" -->
## Introduction
<!-- #endregion -->

<!-- #region id="3GzyMaFcDdrv" -->
- **Problem:** Group recommendation problem is challenging because recommending some item that satisfies all the members of the group is rare. So it often involves some compromises that the model has to make in order to maximize the overall satisfaction of the group.
- **Hypothesis:** RL agent can learn the required behavior that could the maximize the group's overall satisfaction.
- **Benefits:** Meaningful to those people who want to get recommendations for their groups, such as entertainments with families and travels with friends. This model consider the influences of each group member by one self-attention mechanism.
- **Solution:** A recommender agent is trained with actor-critic network and is optimized with DDPG algorithm, where the experience replay and target networks are used. Matrix factorization based simulator is built to simulate the MDP environment. It is an extended version of LIRD model for group recommendations. The group recommendation is viewed as a classification task. When one item is recommended to a group, if the group chooses the item, this case is marked as a positive sample. Otherwise, it will be a negative sample.
- **Dataset:** MovieLens-1m
- **Preprocessing:** Randomly generate groups with 2-5 users. Then, for each group, if every member gives 4-5 stars to one movie, we assume that this movie is adopted by this group with rating 1. If all members give ratings to one movie, but not all in 4-5 stars, we consider the group gives rating 0 to this movie. For other cases, the group movie ratings are missed. Finally, to ensure each group has enough interactions with items, we require each group has at least 20 ratings. Also, for each rating, 100 rating-missed items are randomly sampled. Both user and group rating data are split into training, validation, and testing datasets with the ratio of 70%, 10%, and 20% respectively by the temporal order.
- **Metrics:** Recall, nDCG
- **Cluster:** PyTorch 1.10 cpu
<!-- #endregion -->

<!-- #region id="0m6N7IjYC3PK" -->
<img src='https://github.com/sparsh-ai/stanza/raw/S758139/images/group_recommender_actorcritic_1.svg'>
<!-- #endregion -->

<!-- #region id="LzmYO55mMHkU" -->
> Note: For theoretical understanding, refer to this [this](https://recohut.notion.site/Deep-Reinforcement-Learning-based-Group-Recommender-System-6399cf01102b485897578d1bccbe3467) report.
<!-- #endregion -->

<!-- #region id="K7RfxQkfNMBp" -->
## Setup
<!-- #endregion -->

```python id="UrwfSRelM24I"
from typing import Tuple, List, Dict
import os
import random
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
from collections import deque, defaultdict
import shutil
import zipfile
import scipy.sparse as sp
from collections import Counter

import torch
import torch.nn.functional as functional
from torch import optim, nn

import gym
from sklearn.decomposition import NMF
```

```python id="XA0gzRH0NXV6"
class Config(object):
    """
    Configurations
    """

    def __init__(self):
        # Data
        self.data_folder_path = os.path.join('data', 'MovieLens-Rand')
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

<!-- #region id="2cXG75oPNwhd" -->
## Utils
<!-- #endregion -->

```python id="E41yGTR-Nxae"
class OUNoise(object):
    """
    Ornstein-Uhlenbeck Noise
    """

    def __init__(self, config: Config):
        """
        Initialize OUNoise
        :param config: configurations
        """
        self.embedded_action_size = config.embedded_action_size
        self.ou_mu = config.ou_mu
        self.ou_theta = config.ou_theta
        self.ou_sigma = config.ou_sigma
        self.ou_epsilon = config.ou_epsilon
        self.ou_state = None
        self.reset()

    def reset(self):
        """
        Reset the OU process state
        """
        self.ou_state = torch.ones(self.embedded_action_size) * self.ou_mu

    def evolve_state(self):
        """
        Evolve the OU process state
        """
        self.ou_state += self.ou_theta * (self.ou_mu - self.ou_state) \
            + self.ou_sigma * torch.randn(self.embedded_action_size)

    def get_ou_noise(self):
        """
        Get the OU noise for one action
        :return OU noise
        """
        self.evolve_state()
        return self.ou_state.copy()


class ReplayMemory(object):
    """
    Replay Memory
    """

    def __init__(self, buffer_size: int):
        """
        Initialize ReplayMemory
        :param buffer_size: size of the buffer
        """
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)

    def __len__(self):
        return len(self.buffer)

    def push(self, experience: tuple):
        """
        Push one experience into the buffer
        :param experience: (state, action, reward, new_state)
        """
        self.buffer.append(experience)

    def sample(self, batch_size: int):
        """
        Sample one batch from the buffer
        :param batch_size: number of experiences in the batch
        :return: batch
        """
        batch = random.sample(self.buffer, batch_size)
        return batch
```

<!-- #region id="np-rYYhl3FLW" -->
## Dataset
<!-- #endregion -->

<!-- #region id="0HgBIqbqF81t" -->
<img src='https://github.com/sparsh-ai/stanza/raw/S758139/images/group_recommender_actorcritic_2.svg'>
<!-- #endregion -->

```python id="asSkk-313FIZ"
!wget -q --show-progress https://files.grouplens.org/datasets/movielens/ml-1m.zip
```

```python id="zSP785vg3FFP"
class GroupGenerator(object):
    """
    Group Data Generator
    """
    def __init__(self, data_path, output_path, rating_threshold, num_groups,
                 group_sizes, min_num_ratings, train_ratio, val_ratio,
                 negative_sample_size, verbose=False):
        self.rating_threshold = rating_threshold
        self.negative_sample_size = negative_sample_size
        users_path = os.path.join(data_path, 'users.dat')
        items_path = os.path.join(data_path, 'movies.dat')
        ratings_path = os.path.join(data_path, 'ratings.dat')

        users = self.load_users_file(users_path)
        items = self.load_items_file(items_path)
        rating_mat, timestamp_mat = \
            self.load_ratings_file(ratings_path, max(users), max(items))

        groups, group_ratings, groups_rated_items_dict, groups_rated_items_set = \
            self.generate_group_ratings(users, rating_mat, timestamp_mat,
                                        num_groups=num_groups,
                                        group_sizes=group_sizes,
                                        min_num_ratings=min_num_ratings)
        members, group_ratings_train, group_ratings_val, group_ratings_test, \
            group_negative_items_val, group_negative_items_test, \
            user_ratings_train, user_ratings_val, user_ratings_test, \
            user_negative_items_val, user_negative_items_test = \
            self.split_ratings(group_ratings, rating_mat, timestamp_mat,
                               groups, groups_rated_items_dict, groups_rated_items_set,
                               train_ratio=train_ratio, val_ratio=val_ratio)

        groups_path = os.path.join(output_path, 'groupMember.dat')
        group_ratings_train_path = os.path.join(output_path, 'groupRatingTrain.dat')
        group_ratings_val_path = os.path.join(output_path, 'groupRatingVal.dat')
        group_ratings_test_path = os.path.join(output_path, 'groupRatingTest.dat')
        group_negative_items_val_path = os.path.join(output_path, 'groupRatingValNegative.dat')
        group_negative_items_test_path = os.path.join(output_path, 'groupRatingTestNegative.dat')
        user_ratings_train_path = os.path.join(output_path, 'userRatingTrain.dat')
        user_ratings_val_path = os.path.join(output_path, 'userRatingVal.dat')
        user_ratings_test_path = os.path.join(output_path, 'userRatingTest.dat')
        user_negative_items_val_path = os.path.join(output_path, 'userRatingValNegative.dat')
        user_negative_items_test_path = os.path.join(output_path, 'userRatingTestNegative.dat')

        self.save_groups(groups_path, groups)
        self.save_ratings(group_ratings_train, group_ratings_train_path)
        self.save_ratings(group_ratings_val, group_ratings_val_path)
        self.save_ratings(group_ratings_test, group_ratings_test_path)
        self.save_negative_samples(group_negative_items_val, group_negative_items_val_path)
        self.save_negative_samples(group_negative_items_test, group_negative_items_test_path)
        self.save_ratings(user_ratings_train, user_ratings_train_path)
        self.save_ratings(user_ratings_val, user_ratings_val_path)
        self.save_ratings(user_ratings_test, user_ratings_test_path)
        self.save_negative_samples(user_negative_items_val, user_negative_items_val_path)
        self.save_negative_samples(user_negative_items_test, user_negative_items_test_path)
        shutil.copyfile(src=os.path.join(data_path, 'movies.dat'), dst=os.path.join(output_path, 'movies.dat'))
        shutil.copyfile(src=os.path.join(data_path, 'users.dat'), dst=os.path.join(output_path, 'users.dat'))

        if verbose:
            num_group_ratings = len(group_ratings)
            num_user_ratings = len(user_ratings_train) + len(user_ratings_val) + len(user_ratings_test)
            num_rated_items = len(groups_rated_items_set)

            print('Save data: ' + output_path)
            print('# Users: ' + str(len(members)))
            print('# Items: ' + str(num_rated_items))
            print('# Groups: ' + str(len(groups)))
            print('# U-I ratings: ' + str(num_user_ratings))
            print('# G-I ratings: ' + str(num_group_ratings))
            print('Avg. # ratings / user: {:.2f}'.format(num_user_ratings / len(members)))
            print('Avg. # ratings / group: {:.2f}'.format(num_group_ratings / len(groups)))
            print('Avg. group size: {:.2f}'.format(np.mean(list(map(len, groups)))))

    def load_users_file(self, users_path):
        users = []

        with open(users_path, 'r') as file:
            for line in file.readlines():
                users.append(int(line.split('::')[0]))

        return users

    def load_items_file(self, items_path):
        items = []

        with open(items_path, 'r', encoding='iso-8859-1') as file:
            for line in file.readlines():
                items.append(int(line.split('::')[0]))

        return items

    def load_ratings_file(self, ratings_path, max_num_users, max_num_items):
        rating_mat = sp.dok_matrix((max_num_users + 1, max_num_items + 1),
                                   dtype=np.int)
        timestamp_mat = rating_mat.copy()

        with open(ratings_path, 'r') as file:
            for line in file.readlines():
                arr = line.replace('\n', '').split('::')
                user, item, rating, timestamp = \
                    int(arr[0]), int(arr[1]), int(arr[2]), int(arr[3])
                rating_mat[user, item] = rating
                timestamp_mat[user, item] = timestamp

        return rating_mat, timestamp_mat

    def generate_group_ratings(self, users, rating_mat, timestamp_mat,
                               num_groups, group_sizes, min_num_ratings):
        np.random.seed(0)
        groups = set()
        groups_ratings = []
        groups_rated_items_dict = {}
        groups_rated_items_set = set()

        while len(groups) < num_groups:
            group_id = len(groups) + 1

            while True:
                group = tuple(np.sort(
                    np.random.choice(users, np.random.choice(group_sizes),
                                     replace=False)))
                if group not in groups:
                    break

            pos_group_rating_counter = Counter()
            neg_group_rating_counter = Counter()
            group_rating_list = []
            group_rated_items = set()

            for member in group:
                _, items = rating_mat[member, :].nonzero()
                pos_items = [item for item in items
                             if rating_mat[member, item] >= self.rating_threshold]
                neg_items = [item for item in items
                             if rating_mat[member, item] < self.rating_threshold]
                pos_group_rating_counter.update(pos_items)
                neg_group_rating_counter.update(neg_items)

            for item, num_ratings in pos_group_rating_counter.items():
                if num_ratings == len(group):
                    timestamp = max([timestamp_mat[member, item]
                                     for member in group])
                    group_rated_items.add(item)
                    group_rating_list.append((group_id, item, 1, timestamp))

            for item, num_ratings in neg_group_rating_counter.items():
                if (num_ratings == len(group)) \
                        or (num_ratings + pos_group_rating_counter[item] == len(group)):
                    timestamp = max([timestamp_mat[member, item]
                                     for member in group])
                    group_rated_items.add(item)
                    group_rating_list.append((group_id, item, 0, timestamp))

            if len(group_rating_list) >= min_num_ratings:
                groups.add(group)
                groups_rated_items_dict[group_id] = group_rated_items
                groups_rated_items_set.update(group_rated_items)
                for group_rating in group_rating_list:
                    groups_ratings.append(group_rating)

        return list(groups), groups_ratings, groups_rated_items_dict, groups_rated_items_set

    def split_ratings(self, group_ratings, rating_mat, timestamp_mat,
                      groups, groups_rated_items_dict, groups_rated_items_set, train_ratio, val_ratio):
        num_group_ratings = len(group_ratings)
        num_train = int(num_group_ratings * train_ratio)
        num_test = int(num_group_ratings * (1 - train_ratio - val_ratio))

        group_ratings = \
            sorted(group_ratings, key=lambda group_rating: group_rating[-1])
        group_ratings_train = group_ratings[:num_train]
        group_ratings_val = group_ratings[num_train:-num_test]
        group_ratings_test = group_ratings[-num_test:]

        timestamp_split_train = group_ratings_train[-1][-1]
        timestamp_split_val = group_ratings_val[-1][-1]

        user_ratings_train = []
        user_ratings_val = []
        user_ratings_test = []

        members = set()
        users_rated_items_dict = {}

        for group in groups:
            for member in group:
                if member in members:
                    continue
                members.add(member)
                user_rated_items = set()
                _, items = rating_mat[member, :].nonzero()
                for item in items:
                    if item not in groups_rated_items_set:
                        continue
                    user_rated_items.add(item)
                    if rating_mat[member, item] >= self.rating_threshold:
                        rating_tuple = (member, item, 1,
                                        timestamp_mat[member, item])
                    else:
                        rating_tuple = (member, item, 0,
                                        timestamp_mat[member, item])
                    if timestamp_mat[member, item] <= timestamp_split_train:
                        user_ratings_train.append(rating_tuple)
                    elif timestamp_split_train < timestamp_mat[member, item] <= timestamp_split_val:
                        user_ratings_val.append(rating_tuple)
                    else:
                        user_ratings_test.append(rating_tuple)

                users_rated_items_dict[member] = user_rated_items

        np.random.seed(0)

        user_negative_items_val = self.get_negative_samples(
            user_ratings_val, groups_rated_items_set, users_rated_items_dict)
        user_negative_items_test = self.get_negative_samples(
            user_ratings_test, groups_rated_items_set, users_rated_items_dict)
        group_negative_items_val = self.get_negative_samples(
            group_ratings_val, groups_rated_items_set, groups_rated_items_dict)
        group_negative_items_test = self.get_negative_samples(
            group_ratings_test, groups_rated_items_set, groups_rated_items_dict)

        return members, group_ratings_train, group_ratings_val, group_ratings_test, \
            group_negative_items_val, group_negative_items_test, \
            user_ratings_train, user_ratings_val, user_ratings_test, \
            user_negative_items_val, user_negative_items_test

    def get_negative_samples(self, ratings, groups_rated_items_set, rated_items_dict):
        negative_items_list = []
        for sample in ratings:
            sample_id, item, _, _ = sample
            missed_items = groups_rated_items_set - rated_items_dict[sample_id]
            negative_items = \
                np.random.choice(list(missed_items), self.negative_sample_size,
                                 replace=(len(missed_items) < self.negative_sample_size))
            negative_items_list.append((sample_id, item, negative_items))
        return negative_items_list

    def save_groups(self, groups_path, groups):
        with open(groups_path, 'w') as file:
            for i, group in enumerate(groups):
                file.write(str(i + 1) + ' '
                           + ','.join(map(str, list(group))) + '\n')

    def save_ratings(self, ratings, ratings_path):
        with open(ratings_path, 'w') as file:
            for rating in ratings:
                file.write(' '.join(map(str, list(rating))) + '\n')

    def save_negative_samples(self, negative_items, negative_items_path):
        with open(negative_items_path, 'w') as file:
            for samples in negative_items:
                user, item, negative_items = samples
                file.write('({},{}) '.format(user, item)
                           + ' '.join(map(str, list(negative_items))) + '\n')
```

```python id="BL05MbaZ3Kfk"
print('Takes approx. 5 mins...')
```

```python colab={"base_uri": "https://localhost:8080/"} id="_sprYbwKFPUU" outputId="bed68e1c-961c-4011-f139-4b88d2764565"
data_folder_path = '.'
data_path = os.path.join(data_folder_path, 'ml-1m')
data_zip_path = os.path.join(data_folder_path, 'ml-1m.zip')
output_path = os.path.join(data_folder_path, 'MovieLens-Rand')

if not os.path.exists(data_path):
    with zipfile.ZipFile(data_zip_path, 'r') as data_zip:
        data_zip.extractall(data_folder_path)
        print('Unzip file: ' + data_zip_path)

if not os.path.exists(output_path):
    os.mkdir(output_path)

group_generator = GroupGenerator(data_path, output_path,
                                    rating_threshold=4,
                                    num_groups=1000,
                                    group_sizes=[2, 3, 4, 5],
                                    min_num_ratings=20,
                                    train_ratio=0.7,
                                    val_ratio=0.1,
                                    negative_sample_size=100,
                                    verbose=True)
```

<!-- #region id="x8ltpqJ3Ovfx" -->
## Dataloader
<!-- #endregion -->

<!-- #region id="3AM4UMwoHEgk" -->
<img src='https://github.com/sparsh-ai/stanza/raw/S758139/images/group_recommender_actorcritic_3.svg'>
<!-- #endregion -->

```python id="QupINL9IOvda"
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

<!-- #region id="C6AZOj-gNNRW" -->
## Model
<!-- #endregion -->

<!-- #region id="n9KZMxY9HiiE" -->
<img src='https://github.com/sparsh-ai/stanza/raw/S758139/images/group_recommender_actorcritic_4.svg'>
<!-- #endregion -->

```python id="2qjXiULoNEfd"
class Actor(nn.Module):
    """
    Actor Network
    """

    def __init__(self, embedded_state_size: int, action_weight_size: int, hidden_sizes: Tuple[int]):
        """
        Initialize Actor
        :param embedded_state_size: embedded state size
        :param action_weight_size: embedded action size
        :param hidden_sizes: hidden sizes
        """
        super(Actor, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(embedded_state_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], action_weight_size),
        )

    def forward(self, embedded_state):
        """
        Forward
        :param embedded_state: embedded state
        :return: action weight
        """
        return self.net(embedded_state)


class Critic(nn.Module):
    """
    Critic Network
    """

    def __init__(self, embedded_state_size: int, embedded_action_size: int, hidden_sizes: Tuple[int]):
        """
        Initialize Critic
        :param embedded_state_size: embedded state size
        :param embedded_action_size: embedded action size
        :param hidden_sizes: hidden sizes
        """
        super(Critic, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(embedded_state_size + embedded_action_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], 1)
        )

    def forward(self, embedded_state, embedded_action):
        """
        Forward
        :param embedded_state: embedded state
        :param embedded_action: embedded action
        :return: Q value
        """
        return self.net(torch.cat([embedded_state, embedded_action], dim=-1))


class Embedding(nn.Module):
    """
    Embedding Network
    """

    def __init__(self, embedding_size: int, user_num: int, item_num: int):
        """
        Initialize Embedding
        :param embedding_size: embedding size
        :param user_num: number of users
        :param item_num: number of items
        """
        super(Embedding, self).__init__()
        self.user_embedding = nn.Embedding(user_num + 1, embedding_size)
        self.item_embedding = nn.Embedding(item_num + 1, embedding_size)
        self.user_attention = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, 1)
        )
        self.user_softmax = nn.Softmax(dim=-1)

    def forward(self, group_members, history):
        """
        Forward
        :param group_members: group members
        :param history: browsing history of items
        :return: embedded state
        """
        embedded_group_members = self.user_embedding(group_members)
        group_member_attentions = self.user_softmax(self.user_attention(embedded_group_members))
        embedded_group = torch.squeeze(torch.inner(group_member_attentions.T, embedded_group_members.T))
        embedded_history = torch.flatten(self.item_embedding(history), start_dim=-2)
        embedded_state = torch.cat([embedded_group, embedded_history], dim=-1)
        return embedded_state
```

<!-- #region id="HMPTwCIqNPqm" -->
## Agent
<!-- #endregion -->

<!-- #region id="8GaGRkM9H_RX" -->
<img src='https://github.com/sparsh-ai/stanza/raw/S758139/images/group_recommender_actorcritic_5.svg'>
<!-- #endregion -->

```python id="X28pRvrlOQiT"
class DDPGAgent(object):
    """
    DDPG (Deep Deterministic Policy Gradient) Agent
    """

    def __init__(self, config: Config, noise: OUNoise, group2members_dict: dict, verbose=False):
        """
        Initialize DDPGAgent
        :param config: configurations
        :param group2members_dict: group members data
        :param verbose: True to print networks
        """
        self.config = config
        self.noise = noise
        self.group2members_dict = group2members_dict
        self.tau = config.tau
        self.gamma = config.gamma
        self.device = config.device

        self.embedding = Embedding(embedding_size=config.embedding_size,
                                         user_num=config.user_num,
                                         item_num=config.item_num).to(config.device)
        self.actor = Actor(embedded_state_size=config.embedded_state_size,
                                 action_weight_size=config.embedded_action_size,
                                 hidden_sizes=config.actor_hidden_sizes).to(config.device)
        self.actor_target = Actor(embedded_state_size=config.embedded_state_size,
                                        action_weight_size=config.embedded_action_size,
                                        hidden_sizes=config.actor_hidden_sizes).to(config.device)
        self.critic = Critic(embedded_state_size=config.embedded_state_size,
                                   embedded_action_size=config.embedded_action_size,
                                   hidden_sizes=config.critic_hidden_sizes).to(config.device)
        self.critic_target = Critic(embedded_state_size=config.embedded_state_size,
                                          embedded_action_size=config.embedded_action_size,
                                          hidden_sizes=config.critic_hidden_sizes).to(config.device)

        if verbose:
            print(self.embedding)
            print(self.actor)
            print(self.critic)

        self.copy_network(self.actor, self.actor_target)
        self.copy_network(self.critic, self.critic_target)

        self.replay_memory = ReplayMemory(buffer_size=config.buffer_size)
        self.critic_criterion = nn.MSELoss()
        self.embedding_optimizer = optim.Adam(self.embedding.parameters(), lr=config.embedding_learning_rate,
                                              weight_decay=config.embedding_weight_decay)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.actor_learning_rate,
                                          weight_decay=config.actor_weight_decay)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.critic_learning_rate,
                                           weight_decay=config.critic_weight_decay)

    def copy_network(self, network: nn.Module, network_target: nn.Module):
        """
        Copy one network to its target network
        :param network: the original network to be copied
        :param network_target: the target network
        """
        for parameters, target_parameters in zip(network.parameters(), network_target.parameters()):
            target_parameters.data.copy_(parameters.data)

    def sync_network(self, network: nn.Module, network_target: nn.Module):
        """
        Synchronize one network to its target network
        :param network: the original network to be synchronized
        :param network_target: the target network
        :return:
        """
        for parameters, target_parameters in zip(network.parameters(), network_target.parameters()):
            target_parameters.data.copy_(parameters.data * self.tau + target_parameters.data * (1 - self.tau))

    def get_action(self, state: list, item_candidates: list = None, top_K: int = 1, with_noise=False):
        """
        Get one action
        :param state: one environment state
        :param item_candidates: item candidates
        :param top_K: top K items
        :param with_noise: True to with noise
        :return: action
        """
        with torch.no_grad():
            states = [state]
            embedded_states = self.embed_states(states)
            action_weights = self.actor(embedded_states)
            action_weight = torch.squeeze(action_weights)
            if with_noise:
                action_weight += self.noise.get_ou_noise()

            if item_candidates is None:
                item_embedding_weight = self.embedding.item_embedding.weight.clone()
            else:
                item_candidates = np.array(item_candidates)
                item_candidates_tensor = torch.tensor(item_candidates, dtype=torch.int).to(self.device)
                item_embedding_weight = self.embedding.item_embedding(item_candidates_tensor)

            scores = torch.inner(action_weight, item_embedding_weight).detach().cpu().numpy()
            sorted_score_indices = np.argsort(scores)[:top_K]

            if item_candidates is None:
                action = sorted_score_indices
            else:
                action = item_candidates[sorted_score_indices]
            action = np.squeeze(action)
            if top_K == 1:
                action = action.item()
        return action

    def get_embedded_actions(self, embedded_states: torch.Tensor, target=False):
        """
        Get embedded actions
        :param embedded_states: embedded states
        :param target: True for target network
        :return: embedded_actions (, actions)
        """
        if not target:
            action_weights = self.actor(embedded_states)
        else:
            action_weights = self.actor_target(embedded_states)

        item_embedding_weight = self.embedding.item_embedding.weight.clone()
        scores = torch.inner(action_weights, item_embedding_weight)
        embedded_actions = torch.inner(functional.gumbel_softmax(scores, hard=True), item_embedding_weight.t())
        return embedded_actions

    def embed_state(self, state: list):
        """
        Embed one state
        :param state: state
        :return: embedded_state
        """
        group_id = state[0]
        group_members = torch.tensor(self.group2members_dict[group_id], dtype=torch.int).to(self.device)
        history = torch.tensor(state[1:], dtype=torch.int).to(self.device)
        embedded_state = self.embedding(group_members, history)
        return embedded_state

    def embed_states(self, states: List[list]):
        """
        Embed states
        :param states: states
        :return: embedded_states
        """
        embedded_states = torch.stack([self.embed_state(state) for state in states], dim=0)
        return embedded_states

    def embed_actions(self, actions: list):
        """
        Embed actions
        :param actions: actions
        :return: embedded_actions
        """
        actions = torch.tensor(actions, dtype=torch.int).to(self.device)
        embedded_actions = self.embedding.item_embedding(actions)
        return embedded_actions

    def update(self):
        """
        Update the networks
        :return: actor loss and critic loss
        """
        batch = self.replay_memory.sample(self.config.batch_size)
        states, actions, rewards, next_states = list(zip(*batch))

        self.embedding_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        embedded_states = self.embed_states(states)
        embedded_actions = self.embed_actions(actions)
        rewards = torch.unsqueeze(torch.tensor(rewards, dtype=torch.int).to(self.device), dim=-1)
        embedded_next_states = self.embed_states(next_states)
        q_values = self.critic(embedded_states, embedded_actions)

        with torch.no_grad():
            embedded_next_actions = self.get_embedded_actions(embedded_next_states, target=True)
            next_q_values = self.critic_target(embedded_next_states, embedded_next_actions)
            q_values_target = rewards + self.gamma * next_q_values

        critic_loss = self.critic_criterion(q_values, q_values_target)
        critic_loss.backward()
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()
        embedded_states = self.embed_states(states)
        actor_loss = -self.critic(embedded_states, self.get_embedded_actions(embedded_states)).mean()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.embedding_optimizer.step()

        self.sync_network(self.actor, self.actor_target)
        self.sync_network(self.critic, self.critic_target)

        return actor_loss.detach().cpu().numpy(), critic_loss.detach().cpu().numpy()
```

<!-- #region id="fYiozoLOORHK" -->
## Environment
<!-- #endregion -->

<!-- #region id="qpR7KporIW3w" -->
<img src='https://github.com/sparsh-ai/stanza/raw/S758139/images/group_recommender_actorcritic_6.svg'>
<!-- #endregion -->

```python id="U4pCAbDEPCvW"
class Env(gym.Env):
    """
    Environment for the recommender system
    https://github.com/openai/gym/blob/master/gym/core.py
    """
    metadata = {'render.modes': ['human']}
    reward_range = (0, 1)

    def __init__(self, config: Config, rating_matrix: csr_matrix, dataset_name: str):
        """
        Initialize Env
        :param config: configurations
        :param rating_matrix: rating matrix
        :param dataset_name: dataset name
        """
        assert dataset_name in ['train', 'val', 'test']
        self.config = config
        self.action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(config.action_size,))
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(config.state_size,))

        self.rating_matrix = rating_matrix
        rating_matrix_coo = rating_matrix.tocoo()
        rating_matrix_rows = rating_matrix_coo.row
        rating_matrix_columns = rating_matrix_coo.col
        self.rating_matrix_index_set = set(zip(*(rating_matrix_rows, rating_matrix_columns)))
        self.env_name = 'env_' + dataset_name + '_' + str(self.config.env_n_components) + '.npy'
        self.env_path = os.path.join(config.saves_folder_path, self.env_name)

        self.rating_matrix_pred = None
        self.load_env()

        self.state = None
        self.reset()

    def load_env(self):
        """
        Load environment
        """
        if not os.path.exists(self.env_path):
            env_model = NMF(n_components=self.config.env_n_components, init='random', tol=self.config.env_tol,
                            max_iter=self.config.env_max_iter, alpha=self.config.env_alpha, verbose=True,
                            random_state=0)
            print('-' * 50)
            print('Train environment:')
            W = env_model.fit_transform(X=self.rating_matrix)
            H = env_model.components_
            self.rating_matrix_pred = W @ H
            print('-' * 50)
            np.save(self.env_path, self.rating_matrix_pred)
            print('Save environment:', self.env_path)
        else:
            self.rating_matrix_pred = np.load(self.env_path)
            print('Load environment:', self.env_path)

    def reset(self):
        """
        Reset the environment
        :return: state
        """
        while True:
            group_id = np.random.choice(range(1, self.config.total_group_num + 1))
            nonzero_row, nonzero_col = self.rating_matrix[group_id, :].nonzero()
            if len(nonzero_col) >= self.config.history_length:
                break
        history = np.random.choice(nonzero_col, size=self.config.history_length, replace=False).tolist()
        self.state = [group_id] + history
        return self.state

    def step(self, action: int):
        """
        Take one action to the environment
        :param action: action
        :return: new_state, reward, done, info
        """
        group_id = self.state[0]
        history = self.state[1:]

        if (group_id, action) in self.rating_matrix_index_set:
            reward = self.rating_matrix[group_id, action]
        else:
            reward_probability = self.rating_matrix_pred[group_id, action]
            reward = np.random.choice(self.config.rewards, p=[1 - reward_probability, reward_probability])

        if reward > 0:
            history = history[1:] + [action]

        new_state = [group_id] + history
        self.state = new_state
        done = False
        info = {}

        return new_state, reward, done, info

    def render(self, mode='human'):
        """
        Render the environment
        :param mode: mode
        """
        pass
```

<!-- #region id="8l1VcWD5PN3_" -->
## Evaluator
<!-- #endregion -->

<!-- #region id="Kb63DZkrI0Af" -->
<img src='https://github.com/sparsh-ai/stanza/raw/S758139/images/group_recommender_actorcritic_7.svg'>
<!-- #endregion -->

```python id="B2N_bt1RPU2M"
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

<!-- #region id="3Qjxy1kbPVMU" -->
## Trainer
<!-- #endregion -->

<!-- #region id="cGyC_wGgJNV6" -->
<img src='https://github.com/sparsh-ai/stanza/raw/S758139/images/group_recommender_actorcritic_8.svg'>
<!-- #endregion -->

```python id="8QicC6mQPYFq"
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

<!-- #region id="6t-k7hTLPbQb" -->
## Main
<!-- #endregion -->

```python colab={"background_save": true} id="31kzTQclPkTJ"
config = Config()
dataloader = DataLoader(config)
rating_matrix_train = dataloader.load_rating_matrix(dataset_name='val')
df_eval_user_test = dataloader.load_eval_data(mode='user', dataset_name='test')
df_eval_group_test = dataloader.load_eval_data(mode='group', dataset_name='test')
env = Env(config=config, rating_matrix=rating_matrix_train, dataset_name='val')
noise = OUNoise(config=config)
agent = DDPGAgent(config=config, noise=noise, group2members_dict=dataloader.group2members_dict, verbose=True)
evaluator = Evaluator(config=config)
train(config=config, env=env, agent=agent, evaluator=evaluator,
        df_eval_user=df_eval_user_test, df_eval_group=df_eval_group_test)
```

<!-- #region id="s9UE_CnUOTl6" -->
```
User: Recall@5 = 0.1304, NDCG@5 = 0.0796
User: Recall@10 = 0.2349, NDCG@10 = 0.1132
User: Recall@20 = 0.3470, NDCG@20 = 0.1416
Group: Recall@5 = 0.1856, NDCG@5 = 0.1149
Group: Recall@10 = 0.3153, NDCG@10 = 0.1568
Group: Recall@20 = 0.4448, NDCG@20 = 0.1901
Episode = 220, average reward = 0.0200
Episode = 221, average reward = 0.0000
Episode = 222, average reward = 0.0000
Episode = 223, average reward = 0.0100
Episode = 224, average reward = 0.0000
Episode = 225, average reward = 0.0000
Episode = 226, average reward = 0.0000
Episode = 227, average reward = 0.0000
Episode = 228, average reward = 0.1300
Episode = 229, average reward = 0.0000
User: Recall@5 = 0.1328, NDCG@5 = 0.0810
User: Recall@10 = 0.2405, NDCG@10 = 0.1156
User: Recall@20 = 0.3595, NDCG@20 = 0.1458
Group: Recall@5 = 0.1876, NDCG@5 = 0.1170
Group: Recall@10 = 0.3249, NDCG@10 = 0.1613
Group: Recall@20 = 0.4619, NDCG@20 = 0.1964
Episode = 230, average reward = 0.3000
Episode = 231, average reward = 0.0000
Episode = 232, average reward = 0.0200
```
<!-- #endregion -->

<!-- #region id="Nl1YCO20Zwtq" -->
---
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="z7cZKFdhZwtv" outputId="010839a5-e2f8-4bc4-8ea1-b78aa47335cf"
!pip install -q watermark
%reload_ext watermark
%watermark -a "Sparsh A." -m -iv -u -t -d
```

<!-- #region id="fkohmJNNMrHK" -->
## References
1. [DRGR: Deep Reinforcement Learning based Group Recommender System](https://arxiv.org/abs/2106.06900v1)
2. [Deep Reinforcement Learning based Group Recommender System](https://recohut.notion.site/Deep-Reinforcement-Learning-based-Group-Recommender-System-6399cf01102b485897578d1bccbe3467)
3. Source code:
 - https://github.com/zefang-liu/group-recommender
 - https://github.com/sparsh-ai/stanza/tree/S758139 


<!-- #endregion -->

<!-- #region id="G6maUiF9Zwtw" -->
---
<!-- #endregion -->

<!-- #region id="HOsXA5uEZwtw" -->
**END**
<!-- #endregion -->
