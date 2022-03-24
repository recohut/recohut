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

```python id="MHNr1zXp4diG" executionInfo={"status": "ok", "timestamp": 1628327509555, "user_tz": -330, "elapsed": 1818, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
from collections import deque

from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import pandas as pd
import numpy as np
import scipy
import scipy.special
import gym
import gym.spaces as spaces
```

```python id="zbZbbhdc4mbG" executionInfo={"status": "ok", "timestamp": 1628327541243, "user_tz": -330, "elapsed": 773, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
class SlateSpace(spaces.MultiDiscrete):
    def __init__(self, nvec):
        assert np.unique(nvec).size == 1, 'each slate position should allow all available items to display.'
        assert len(nvec) <= nvec[0], f'slate size ({len(nvec)}) should be no larger than the number of items ({nvec[0]}).'
        super().__init__(nvec)

    def sample(self):
        # since a slate is a permutation over items with a cut-off
        # we implemented by using numpy for efficiency, avoid for-loop
        return self.np_random.permutation(self.nvec[0])[:len(self.nvec)].astype(self.dtype)

    def sample_batch(self, batch_size):
        # for-loop will be very SLOW!
        # NOTE: we use numpy's `permutation` and `apply_along_axis` to be very efficient!
        n_item = self.nvec[0]
        slate_size = len(self.nvec)

        arr = np.arange(n_item)[None, :]
        arr = np.tile(arr, (batch_size, 1))
        arr = np.apply_along_axis(func1d=self.np_random.permutation, axis=1, arr=arr)
        arr = arr[:, :slate_size]
        return arr

    def contains(self, x):
        is_contained = super().contains(x)
        is_unique = (np.unique(x).size == len(x))
        return is_unique and is_contained

    def __repr__(self):
        return f'SlateSpace({self.nvec})'

    def __eq__(self, other):
        return isinstance(other, SlateSpace) and np.all(self.nvec == other.nvec)
```

```python id="BUzRa3uo4bIv" executionInfo={"status": "ok", "timestamp": 1628327564775, "user_tz": -330, "elapsed": 1326, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
class Env(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 3
    }
    reward_range = (-float('inf'), float('inf'))

    def __init__(
        self, user_ids, item_category, item_popularity,
        hist_seq_len, slate_size,
        user_state_model_callback, reward_model_callback,
    ):
        self.user_ids = user_ids
        assert len(item_category) == len(item_popularity)
        item_category = [str(i) for i in item_category]  # enforce str, otherwise visualization won't work well
        self.item_category = item_category
        item_popularity = np.asarray(item_popularity)
        self.scaled_item_popularity = item_popularity/max(item_popularity)
        self.hist_seq_len = hist_seq_len
        self.slate_size = slate_size
        self.user_state_model_callback = user_state_model_callback
        self.reward_model_callback = reward_model_callback

        self.nan_item_id = -1

        self.user_id = None  # enforce calling `env.reset()`
        self.hist_seq = deque([self.nan_item_id]*hist_seq_len, maxlen=hist_seq_len)  # FIFO que for user's historical interactions
        assert len(self.hist_seq) == hist_seq_len

        obs_dim = len(user_state_model_callback(user_ids[0], self.hist_seq))
        self.observation_space = spaces.Box(
            low=-float('inf'), 
            high=float('inf'), 
            shape=(obs_dim,), 
            dtype=np.float32
        )
        # NOTE: do NOT use `gym.spaces.MultiDiscrete`: it does NOT support unique sampling for slate
        # i.e. a sampled action may contain multiple redundant item in the slate!
        self.action_space = SlateSpace((len(item_category),)*slate_size)

        # some loggings for visualization
        self.user_logs = []
        self.rs_logs = []
        self.timestep = 0

        self.viewer = None
        self.fig, self.axes = None, None
        self.seed()

    def seed(self, seed=None):
        self.rng = np.random.default_rng(seed=seed)
        return self.rng.bit_generator._seed_seq.entropy  # in case `seed=None`, system generated seed will be returned

    def step(self, action):
        assert action in self.action_space
        assert np.unique(action).size == len(action), 'repeated items in slate are not allowed!'
        # append a skip-item at the end of the slate to allow user to skip the slate
        # pre-trained reward model will give a learned reward for skipping
        action = [*action, self.nan_item_id]
        action_item_reward = self.reward_model_callback(self.user_id, self.hist_seq, action)
        assert action_item_reward.ndim == 1 and len(action_item_reward) == len(action)

        # TODO: customize user choice model as input to the environment constructor
        # for the moment, only sampling in proportion to predicted rewards
        choice_dist = scipy.special.softmax(action_item_reward)
        idx = self.rng.choice(len(action), size=None, p=choice_dist)
        clicked_item_id = action[idx]
        is_click = (clicked_item_id != self.nan_item_id)

        # update user state transition
        # NOTE: when user skips, `hist_seq` will not change. 
        # For RL agent training (e.g. DQN), it's important to have exploration!
        # Otherwise, agent might get stuck with suboptimal behavior by repeated observation
        # Also, replay buffer may be dominated by such transitions with identical observations
        if is_click:  # user clicked an item in the slate
            self.hist_seq.append(clicked_item_id)

        self.timestep += 1

        # track interactions for visualization
        self.user_logs.append({
            'timestep': self.timestep,
            'clicked_item_id': clicked_item_id,  # NOTE: include skip activity
            'choice_dist': choice_dist.tolist()
        })
        self.rs_logs.append({
            'timestep': self.timestep,
            'slate': action  # NOTE: include skip pseudo-item
        })

        obs = self._get_obs()
        # Alternative: reward = action_item_reward.min() - 1.*action_item_reward.std()
        reward = action_item_reward[idx]
        if reward <= action_item_reward[-1]:
            reward = 0.
        done = False
        info = {
            'is_click': is_click,
            'clicked_item_id': clicked_item_id,
            'action_item_reward': action_item_reward.tolist(),
            'choice_dist': choice_dist.tolist()
        }
        return obs, reward, done, info

    def _get_obs(self):
        user_state = self.user_state_model_callback(self.user_id, self.hist_seq)  # -> [user_state, ]
        assert user_state in self.observation_space
        return user_state

    def reset(self, **kwargs):
        if kwargs.get('user_id', None) is not None:
            user_id = kwargs['user_id']
            assert user_id in self.user_ids
            self.user_id = user_id
        else:
            self.user_id = self.rng.choice(self.user_ids, size=None)
        self.hist_seq = deque([self.nan_item_id]*self.hist_seq_len, maxlen=self.hist_seq_len)
        assert len(self.hist_seq) == self.hist_seq_len

        # some loggings for visualization
        self.user_logs = []
        self.rs_logs = []
        self.timestep = 0

        return self._get_obs()

    def _get_img(self):
        # clear all previous images
        [ax.cla() for ax in self.axes.flatten()]

        # we require strict ordering of the category type in the plot
        # so we use `pd.Categorical` below in `sns.lineplot` to enforce consistent ordering
        categories = np.unique(self.item_category).tolist()
        categories = ['@skip', *categories]
        # enforce str for each category, otherwise `pd.Categorical` breaks with NaN
        categories = [str(c) for c in categories]

        cat_dist_all = pd.Categorical(self.item_category, categories=categories, ordered=True).value_counts()
        cat_dist_all /= cat_dist_all.sum()  # `normalize` keyword NOT existed for `pd.Categorical`
        def _barplot_cat_dist_all(cat_dist_all, categories, ax):
            sns.barplot(x=cat_dist_all.index, y=cat_dist_all.values, order=categories, alpha=.3, ax=ax)
            for patch in ax.patches:  # draw dashed edge on top for each true_category, better visual
                x = [patch.get_x(), patch.get_x() + patch.get_width()]
                y = [patch.get_height()]*2
                ax.plot(x, y, ls='--', lw=1.5, c=patch.get_edgecolor(), alpha=1.)

        df_user_logs = pd.DataFrame(self.user_logs).sort_values(by='timestep', ascending=True)
        df_rs_logs = pd.DataFrame(self.rs_logs).sort_values(by='timestep', ascending=True)

        user_click_cat = df_user_logs['clicked_item_id'].apply(
            lambda item_id: str(self.item_category[item_id]) if item_id != self.nan_item_id else '@skip'
        )
        user_click_cat = pd.Categorical(user_click_cat, categories=categories, ordered=True)

        # figure [0, 0]: Overall User Choices
        cat_dist_user = user_click_cat.value_counts()
        cat_dist_user /= cat_dist_user.sum()  # `normalize` keyword NOT existed for `pd.Categorical`
        _barplot_cat_dist_all(cat_dist_all, categories, ax=self.axes[0, 0])
        g = sns.barplot(x=cat_dist_user.index, y=cat_dist_user.values, order=categories, alpha=.8, ax=self.axes[0, 0])
        g.set(title='Overall User Choices', ylim=(0., 1.), xlabel='Category', ylabel='Percent')

        # figure [1, 0]: Overall Recommendations
        cat_dist_rs = df_rs_logs.explode('slate')
        cat_dist_rs = cat_dist_rs[cat_dist_rs['slate'] != self.nan_item_id]  # remove skip pseudo-item in slate for visualization
        cat_dist_rs = cat_dist_rs['slate'].apply(
            lambda item_id: str(self.item_category[item_id])
        )
        cat_dist_rs = pd.Categorical(cat_dist_rs, categories=categories, ordered=True).value_counts()
        cat_dist_rs /= cat_dist_rs.sum()  # `normalize` keyword NOT existed for `pd.Categorical`
        _barplot_cat_dist_all(cat_dist_all, categories, ax=self.axes[1, 0])
        g = sns.barplot(x=cat_dist_rs.index, y=cat_dist_rs.values, order=categories, alpha=.8, ax=self.axes[1, 0])
        g.set(title='Overall Recommendations', ylim=(0., 1.), xlabel='Category', ylabel='Percent')

        # figure [0, 1]: Sequential User Choices
        g = sns.lineplot(
            x=range(1, self.timestep+1), y=user_click_cat, 
            marker='o', markersize=8, linestyle='--', alpha=.8,
            ax=self.axes[0, 1]
        )
        g.set(  # gym animation wrapper `Monitor` requires both `yticks` and `yticklabels`
            title='Sequential User Choices', yticks=range(len(categories)), yticklabels=categories,
            xlabel='Timestep', ylabel='Category'
        )
        if self.spec is not None:
            g.set_xlim(1, self.spec.max_episode_steps)

        # figure [1, 1]: Intra-Slate Diversity (Shannon)
        rs_diversity = df_rs_logs['slate'].apply(lambda slate: list(filter(lambda x: x != self.nan_item_id, slate)))
        rs_diversity = rs_diversity.apply(
            lambda slate: [str(self.item_category[item_id]) for item_id in slate]
        )
        _categories_wo_skip = list(filter(lambda c: c != '@skip', categories))
        rs_diversity = rs_diversity.apply(lambda slate: pd.Categorical(slate, categories=_categories_wo_skip, ordered=True))
        rs_diversity = rs_diversity.apply(lambda slate: slate.value_counts().values)
        rs_diversity = rs_diversity.apply(lambda slate: slate/slate.sum())
        rs_diversity = rs_diversity.apply(lambda slate: scipy.stats.entropy(slate, base=len(slate)))
        g = sns.lineplot(
            x=range(1, self.timestep+1), y=rs_diversity,
            marker='o', markersize=8, linestyle='--', alpha=.8,
            ax=self.axes[1, 1]
        )
        g.set(
            title='Intra-Slate Diversity (Shannon)',
            xlabel='Timestep', ylabel='Shannon Entropy',
            ylim=(0., 1.)
        )
        if self.spec is not None:
            g.set_xlim(1, self.spec.max_episode_steps)

        # figure [0, 2]: User Choice Distribution
        # make sure the skip pesudo-item is located in the final position
        assert df_rs_logs['slate'].tail(1).item()[-1] == self.nan_item_id
        choice_dist = df_user_logs['choice_dist'].tail(1).item()
        slate_position = list(range(1, self.slate_size+1+1))  # add one more: for skip pseudo-item
        slate_position = [str(i) for i in slate_position]
        slate_position[-1] = '@skip'
        df = pd.DataFrame({'slate_pos': slate_position, 'click_prob': choice_dist})
        g = sns.barplot(
            x='slate_pos', y='click_prob', 
            order=slate_position, alpha=.8, color='b', data=df,
            ax=self.axes[0, 2]
        )
        g.set(title='User Choice Distribution', xlabel='Slate Position', ylabel='Click Probability')

        # figure [1, 2]: Expected Popularity Complement (EPC)
        # EPC: measures the ability to recommend long-tail items in top positions
        # formula: Eq. (7) in https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.1089.1342&rep=rep1&type=pdf
        slate_epc = df_rs_logs['slate'].apply(lambda slate: list(filter(lambda x: x != self.nan_item_id, slate)))
        _rank_discount = np.log2(np.arange(1, self.slate_size+1) + 1)
        slate_epc = slate_epc.apply(
            lambda slate: np.asarray([1. - self.scaled_item_popularity[item_id] for item_id in slate])/_rank_discount
        )
        slate_epc = slate_epc.apply(
            lambda slate: np.sum(slate)/np.sum(1./_rank_discount)
        )
        g = sns.lineplot(
            x=range(1, self.timestep+1), y=slate_epc,
            marker='o', markersize=8, linestyle='--', alpha=.8,
            ax=self.axes[1, 2]
        )
        g.set(
            title='Expected Popularity Complement (EPC)',
            xlabel='Timestep', ylabel='EPC',
            ylim=(0., 1.)
        )
        if self.spec is not None:
            g.set_xlim(1, self.spec.max_episode_steps)

        self.fig.suptitle(f'User ID: {self.user_id}, Time step: {self.timestep}', y=1.0, size='x-large')
        self.fig.tight_layout()

        self.fig.canvas.draw()
        img = Image.frombytes('RGB', self.fig.canvas.get_width_height(), self.fig.canvas.tostring_rgb())
        img = np.asarray(img)
        return img

    def render(self, mode='human', **kwargs):
        if self.fig is None and self.axes is None:
            self.fig, self.axes = plt.subplots(2, 3, figsize=(3*2*6, 2*2*4))
            sns.set()
        if self.timestep == 0:  # gym Monitor may call `render` at very first step, so return empty image
            self.fig.canvas.draw()
            img = Image.frombytes('RGB', self.fig.canvas.get_width_height(), self.fig.canvas.tostring_rgb())
            img = np.asarray(img)
        else:
            img = self._get_img()
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control.rendering import SimpleImageViewer
            if self.viewer is None:
                maxwidth = kwargs.get('maxwidth', int(4*500))
                self.viewer = SimpleImageViewer(maxwidth=maxwidth)                
            self.viewer.imshow(img)
            return self.viewer.isopen

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
            plt.close('all')  # close all with matplotlib, free memory
            self.fig = None
            self.axes = None
```

```python id="bfx3V6IF4sxh" executionInfo={"status": "ok", "timestamp": 1628327979429, "user_tz": -330, "elapsed": 483, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
# This describes a list of available user IDs for the simulation.
# Normally, a user ID is an integer.
# user_ids = [0, 1, 2]

# user ID will be taken as an input to user_state_model_callback to generate observations of the user state.

# This describes the categories of a list of available items.
# The data type should be a list of strings.
# The indices of the list is assumed to correspond to item IDs.
# item_category = ['sci-fi', 'romance', 'sci-fi']

# The category information is mainly used for visualization via env.render().

# This describe the popularity measure of a list of available items.
# The data type should be a list (or 1-dim array) of integers.
# The indices of the list is assumed to correspond to item IDs.
# item_popularity = [5, 3, 1]

# The popularity information is used for calculating Expected Popularity Complement (EPC) in the visualization.

# This is an integer describing the number of most recently clicked items by the user to encode as the current state of the user.
# hist_seq = [-1, 2, 0]

# The item ID -1 indicates an empty event. In this case, the user clicked two items in the past, first item ID 2 followed by a second item ID 0.
# The internal FIFO queue hist_seq will be taken as an input to both user_state_model_callback and reward_model_callback to generate observations of the user state.

# This is an integer describing the size of the slate (display list of recommended items).
# slate_size = 2

# It induces a combinatorial action space for the RL agent.

# This is a Python callback function taking user_id and hist_seq as inputs to generate an observation of current user state.
# user_state_model_callback

# Note that it is generic.
# Either pre-defined heuristic computations or pre-trained neural network models using user/item embeddings can be wrapped as a callback function.

# This is a Python callback function taking user_id, hist_seq and action as inputs to generate a reward value for each item in the slate. (i.e. action)
# reward_model_callback

# Note that it is generic.
# Either pre-defined heuristic computations or pre-trained neural network models using user/item embeddings can be wrapped as a callback function.
```

```python id="7IrHc0js6STN" executionInfo={"status": "ok", "timestamp": 1628328689207, "user_tz": -330, "elapsed": 442, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
# First, let us sample random embeddings for one user and five items:
user_features = np.random.randn(1, 10)
item_features = np.random.randn(5, 10)

# Now let us define the category and popularity score for each item:
item_category = ['sci-fi', 'romance', 'sci-fi', 'action', 'sci-fi']
item_popularity = [5, 3, 1, 2, 3]

# Then, we define callback functions for user state and reward values:
def user_state_model_callback(user_id, hist_seq):
    return user_features[user_id]

def reward_model_callback(user_id, hist_seq, action):
    return np.inner(user_features[user_id], item_features[action])

# Finally, we are ready to create a simulation environment with OpenAI Gym API:
env_kws = dict(
    user_ids=[0],
    item_category=item_category,
    item_popularity=item_popularity,
    hist_seq_len=3,
    slate_size=2,
    user_state_model_callback=user_state_model_callback,
    reward_model_callback=reward_model_callback
)
env = Env(**env_kws)

# we created the environment with slate size of two items and historical interactions of the recent 3 steps.
# The horizon is 50 time steps.
```

```python colab={"base_uri": "https://localhost:8080/"} id="1C3Xmvpd-hre" executionInfo={"status": "ok", "timestamp": 1628329110772, "user_tz": -330, "elapsed": 13679, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="73382baa-2392-49f7-d758-f0c2e5592d58"
!apt-get install -y xvfb python-opengl ffmpeg > /dev/null 2>&1
!pip install -U colabgymrender
```

```python colab={"base_uri": "https://localhost:8080/"} id="OhjNN-J1-d8s" outputId="b6234b45-4dff-4be3-ebba-f1ecc197cde4"
from colabgymrender.recorder import Recorder

directory = './video'
env = Recorder(env, directory)

observation = env.reset()
terminal = False
while not terminal:
  action = env.action_space.sample()
  observation, reward, terminal, info = env.step(action)

env.play()
```
