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

```python id="EypIbWutrmGL" executionInfo={"status": "ok", "timestamp": 1628693243072, "user_tz": -330, "elapsed": 20, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import os
project_name = "reco-tut-mcp"; branch = "main"; account = "sparsh-ai"
project_path = os.path.join('/content', project_name)
```

```python id="86MgMsi_GD70" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628693245487, "user_tz": -330, "elapsed": 2431, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="1ca05732-5c7d-4d3f-84b2-195b3743f75c"
if not os.path.exists(project_path):
    !cp /content/drive/MyDrive/mykeys.py /content
    import mykeys
    !rm /content/mykeys.py
    path = "/content/" + project_name; 
    !mkdir "{path}"
    %cd "{path}"
    import sys; sys.path.append(path)
    !git config --global user.email "recotut@recohut.com"
    !git config --global user.name  "reco-tut"
    !git init
    !git remote add origin https://"{mykeys.git_token}":x-oauth-basic@github.com/"{account}"/"{project_name}".git
    !git pull origin "{branch}"
    !git checkout main
else:
    %cd "{project_path}"
```

```python colab={"base_uri": "https://localhost:8080/"} id="26O1ZyzvsGVK" executionInfo={"status": "ok", "timestamp": 1628695539340, "user_tz": -330, "elapsed": 639, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="4d94214b-de63-494e-aafa-cea02b6f4719"
!git status
```

```python colab={"base_uri": "https://localhost:8080/"} id="JvukJ591sHVZ" executionInfo={"status": "ok", "timestamp": 1628695772495, "user_tz": -330, "elapsed": 833, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="ae8b1ad2-0207-483d-89cb-e7d44a70594b"
!git add . && git commit -m 'commit' && git push origin main
```

```python id="1SNR9qJYtMaA" executionInfo={"status": "ok", "timestamp": 1628693692797, "user_tz": -330, "elapsed": 434, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import sys
sys.path.insert(0,'./code')
```

<!-- #region id="ow5N4xQ6rp-X" -->
---
<!-- #endregion -->

```python id="h529-OAFtZvL" executionInfo={"status": "ok", "timestamp": 1628695267689, "user_tz": -330, "elapsed": 3518, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import argparse
import json
import logging
import numpy as np
import pandas as pd
import time

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style(style='darkgrid')

from environment import ContextualEnvironment
from policies import *
```

```python id="xAKrsCgothQo" executionInfo={"status": "ok", "timestamp": 1628693954447, "user_tz": -330, "elapsed": 670, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
# List of implemented policies
def set_policies(policies_name, user_segment, user_features, n_playlists):
    # Please see section 3.3 of RecSys paper for a description of policies
    POLICIES_SETTINGS = {
        'random' : RandomPolicy(n_playlists),
        'etc-seg-explore' : ExploreThenCommitSegmentPolicy(user_segment, n_playlists, min_n = 100, cascade_model = True),
        'etc-seg-exploit' : ExploreThenCommitSegmentPolicy(user_segment, n_playlists, min_n = 20, cascade_model = True),
        'epsilon-greedy-explore' : EpsilonGreedySegmentPolicy(user_segment, n_playlists, epsilon = 0.1, cascade_model = True),
        'epsilon-greedy-exploit' : EpsilonGreedySegmentPolicy(user_segment, n_playlists, epsilon = 0.01, cascade_model = True),
        'kl-ucb-seg' : KLUCBSegmentPolicy(user_segment, n_playlists, cascade_model = True),
        'ts-seg-naive' : TSSegmentPolicy(user_segment, n_playlists, alpha_zero = 1, beta_zero = 1, cascade_model = True),
        'ts-seg-pessimistic' : TSSegmentPolicy(user_segment, n_playlists, alpha_zero = 1, beta_zero = 99, cascade_model = True),
        'ts-lin-naive' : LinearTSPolicy(user_features, n_playlists, bias = 0.0, cascade_model = True),
        'ts-lin-pessimistic' : LinearTSPolicy(user_features, n_playlists, bias = -5.0, cascade_model = True),
        # Versions of epsilon-greedy-explore and ts-seg-pessimistic WITHOUT cascade model
        'epsilon-greedy-explore-no-cascade' : EpsilonGreedySegmentPolicy(user_segment, n_playlists, epsilon = 0.1, cascade_model = False),
        'ts-seg-pessimistic-no-cascade' : TSSegmentPolicy(user_segment, n_playlists, alpha_zero = 1, beta_zero = 99, cascade_model = False)
    }

    return [POLICIES_SETTINGS[name] for name in policies_name]
```

```python id="hEBbA-e5zZGG" executionInfo={"status": "ok", "timestamp": 1628695407933, "user_tz": -330, "elapsed": 609, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
# Plots the evolution of expected cumulative regrets curves,
# for all tested policies and over all rounds

def plot_results(result_path, figsize=(10,10)):
    with open(result_path, 'r') as fp:
        cumulative_regrets = json.load(fp)

    plt.figure(figsize=figsize)
    for k,v in cumulative_regrets.items():
        sns.lineplot(data = np.array(v), label=k)
    plt.xlabel("Round")
    plt.ylabel("Cumulative Regret")
    plt.show()
```

```python id="2c07xUu1vNrV" executionInfo={"status": "ok", "timestamp": 1628694177216, "user_tz": -330, "elapsed": 537, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 377} id="qNkVh_nath6e" executionInfo={"status": "ok", "timestamp": 1628694245158, "user_tz": -330, "elapsed": 650, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="666ce188-d71e-43ef-d2aa-44200b3e8896"
users_df = pd.read_parquet('./data/bronze/users_small.parquet.gzip')
users_df
```

```python colab={"base_uri": "https://localhost:8080/", "height": 439} id="nDbKhqkpt7YM" executionInfo={"status": "ok", "timestamp": 1628694246408, "user_tz": -330, "elapsed": 10, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="eb527ecc-13a8-48d7-af64-48a98bb8c9bb"
playlists_df = pd.read_pickle('./data/bronze/playlists.pickle.gzip', compression='gzip')
playlists_df
```

<!-- #region id="KBxrt5K_xgeA" -->
default arguments
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="iFGxRtqfubru" executionInfo={"status": "ok", "timestamp": 1628695024315, "user_tz": -330, "elapsed": 1315, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="8e80af80-cd0b-4fdb-fc51-35898ac19396"
parser = argparse.ArgumentParser()
parser.add_argument("--output_path", type = str, default = "./outputs/results_2021_08_11.json", required = False,
                    help = "Path to json file to save regret values")
parser.add_argument("--policies", type = str, default = "random,ts-seg-naive", required = False,
                    help = "Bandit algorithms to evaluate, separated by commas")
parser.add_argument("--n_recos", type = int, default = 12, required = False,
                    help = "Number of slots L in the carousel i.e. number of recommendations to provide")
parser.add_argument("--l_init", type = int, default = 3, required = False,
                    help = "Number of slots L_init initially visible in the carousel")
parser.add_argument("--n_users_per_round", type = int, default = 20000, required = False,
                    help = "Number of users randomly selected (with replacement) per round")
parser.add_argument("--n_rounds", type = int, default = 100, required = False,
                    help = "Number of simulated rounds")
parser.add_argument("--print_every", type = int, default = 10, required = False,
                    help = "Print cumulative regrets every 'print_every' round")

args = parser.parse_args(args={})

args
```

```python id="tT-xp09zvDkw" executionInfo={"status": "ok", "timestamp": 1628694307525, "user_tz": -330, "elapsed": 1047, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
if args.l_init > args.n_recos:
    raise ValueError('l_init is larger than n_recos')
```

```python id="Gh53N9-VvKIu" executionInfo={"status": "ok", "timestamp": 1628694286820, "user_tz": -330, "elapsed": 425, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
n_users = len(users_df)
n_playlists = len(playlists_df)
n_recos = args.n_recos
print_every = args.print_every

user_features = np.array(users_df.drop(["segment"], axis = 1))
user_features = np.concatenate([user_features, np.ones((n_users,1))], axis = 1)
playlist_features = np.array(playlists_df)

user_segment = np.array(users_df.segment)
```

<!-- #region id="f9-ExnPFxk0_" -->
Evaluation of all policies on small users (useful for quick testing)
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="sKWj37w_yFwY" executionInfo={"status": "ok", "timestamp": 1628695017601, "user_tz": -330, "elapsed": 13, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="5e58588e-2101-400f-dde9-152257b285ee"
args
```

```python id="Pa2z2A_WyHgR" executionInfo={"status": "ok", "timestamp": 1628695209373, "user_tz": -330, "elapsed": 6, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
args.policies = 'random,etc-seg-explore,etc-seg-exploit,epsilon-greedy-explore,epsilon-greedy-exploit,kl-ucb-seg,ts-seg-naive,ts-seg-pessimistic,ts-lin-naive,ts-lin-pessimistic'
args.n_users_per_round = 9
args.output_path = './outputs/results_small_210811_01.json'
```

```python colab={"base_uri": "https://localhost:8080/"} id="8mc3mA1UxpuW" executionInfo={"status": "ok", "timestamp": 1628695086127, "user_tz": -330, "elapsed": 600, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="6d72d126-0fe7-4c33-b1f3-70c5c0f61788"
logger.info("SETTING UP POLICIES")
logger.info("Policies to evaluate: %s \n \n" % (args.policies))

policies_name = args.policies.split(",")
policies = set_policies(policies_name, user_segment, user_features, n_playlists)
n_policies = len(policies)
n_users_per_round = args.n_users_per_round
n_rounds = args.n_rounds
overall_rewards = np.zeros((n_policies, n_rounds))
overall_optimal_reward = np.zeros(n_rounds)
```

```python colab={"base_uri": "https://localhost:8080/"} id="LiQS2yE8v7rI" executionInfo={"status": "ok", "timestamp": 1628695090654, "user_tz": -330, "elapsed": 643, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="fb109e8b-d81e-41bc-d4df-439e2cb151b7"
logger.info("SETTING UP SIMULATION ENVIRONMENT")
logger.info("for %d users, %d playlists, %d recommendations per carousel \n \n" % (n_users, n_playlists, n_recos))

cont_env = ContextualEnvironment(user_features, playlist_features, user_segment, n_recos)
```

```python colab={"base_uri": "https://localhost:8080/"} id="CL6WOxLBwN1q" executionInfo={"status": "ok", "timestamp": 1628695129117, "user_tz": -330, "elapsed": 27379, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="a5bbc55c-424b-4864-af6f-3accf249aac0"
# Simulations for Top-n_recos carousel-based playlist recommendations

logger.info("STARTING SIMULATIONS")
logger.info("for %d rounds, with %d users per round (randomly drawn with replacement)\n \n" % (n_rounds, n_users_per_round))
start_time = time.time()
for i in range(n_rounds):
    # Select batch of n_users_per_round users
    user_ids = np.random.choice(range(n_users), n_users_per_round)
    overall_optimal_reward[i] = np.take(cont_env.th_rewards, user_ids).sum()
    # Iterate over all policies
    for j in range(n_policies):
        # Compute n_recos recommendations
        recos = policies[j].recommend_to_users_batch(user_ids, args.n_recos, args.l_init)
        # Compute rewards
        rewards = cont_env.simulate_batch_users_reward(batch_user_ids= user_ids, batch_recos=recos)
        # Update policy based on rewards
        policies[j].update_policy(user_ids, recos, rewards, args.l_init)
        overall_rewards[j,i] = rewards.sum()
    # Print info
    if i == 0 or (i+1) % print_every == 0 or i+1 == n_rounds:
        logger.info("Round: %d/%d. Elapsed time: %f sec." % (i+1, n_rounds, time.time() - start_time))
        logger.info("Cumulative regrets: \n%s \n" % "\n".join(["	%s : %s" % (policies_name[j], str(np.sum(overall_optimal_reward - overall_rewards[j]))) for j in range(n_policies)]))
```

```python id="Y8Zw_6uTGD71" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628695216250, "user_tz": -330, "elapsed": 837, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="c6fe097d-2c14-4656-d316-e82fd9f66fe3"
# Save results

logger.info("Saving cumulative regrets in %s" % args.output_path)
cumulative_regrets = {policies_name[j] : list(np.cumsum(overall_optimal_reward - overall_rewards[j])) for j in range(n_policies)}

with open(args.output_path, 'w') as fp:
    json.dump(cumulative_regrets, fp)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 334} id="C6yOzUDzr9Qt" executionInfo={"status": "ok", "timestamp": 1628695443808, "user_tz": -330, "elapsed": 3104, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="952d1827-f100-4726-e36f-8a6ddf95c519"
# Plot results

plot_results(args.output_path, figsize=(10,5))
```
