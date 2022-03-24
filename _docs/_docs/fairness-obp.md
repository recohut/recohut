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

<!-- #region id="j_xZZNpDuI9a" -->
# Fairness Evaluation using OBP Library on ML-100k
<!-- #endregion -->

<!-- #region id="CxiWmRiFzT2X" -->
## Setup
<!-- #endregion -->

<!-- #region id="fQ64dOFO0fJe" -->
### Git
<!-- #endregion -->

```python id="2eRcpGL6XfDs"
!git clone -b T014724 https://github.com/sparsh-ai/drl-recsys.git
```

```python colab={"base_uri": "https://localhost:8080/"} id="6zqTPkYVvutg" executionInfo={"status": "ok", "timestamp": 1636786136051, "user_tz": -330, "elapsed": 7, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="85459c32-3c70-4ab8-e73b-06319540b0f5"
%cd drl-recsys
```

<!-- #region id="BXJY8c9d4Xi5" -->
### Installations
<!-- #endregion -->

```python id="DctyNOSdx-7h"
!pip install obp
!pip install wandb
!pip install luigi
```

<!-- #region id="GB_yDppW3_Yt" -->
### Imports
<!-- #endregion -->

```python id="yzkISSHMDccH"
%reload_ext autoreload
%autoreload 2
```

```python id="vrEmNkAAsQlM"
import numpy as np
from tqdm.notebook import tqdm
import sys
import os
import json 
import logging
import pickle
import datetime
import yaml
import pandas as pd
from os import path as osp
from pathlib import Path

import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go

import torch
import luigi

import obp
from obp.policy.policy_type import PolicyType
from obp.utils import convert_to_action_dist
from obp.ope import (
    OffPolicyEvaluation, 
    RegressionModel,
    InverseProbabilityWeighting as IPS,
    SelfNormalizedInverseProbabilityWeighting as SNIPS,
    DirectMethod as DM,
    DoublyRobust as DR,
    DoublyRobustWithShrinkage as DRos,
)

from src.data.dataset import DatasetGeneration
from src.train_model import MovieLens

from src.data.obp_dataset import MovieLensDataset
from src.model.simulator import run_bandit_simulation
from src.model.bandit import EpsilonGreedy, LinUCB, WFairLinUCB, FairLinUCB
from src.environment.ml_env import OfflineEnv, OfflineFairEnv
from src.model.recommender import DRRAgent, FairRecAgent
from src.model.pmf import PMF
from src.recsys_fair_metrics.recsys_fair import RecsysFair
from src.recsys_fair_metrics.util.util import parallel_literal_eval
```

<!-- #region id="NyxCtlrJ3_Ta" -->
### Params
<!-- #endregion -->

```python id="MXBwnUCD3_RD"
class Args:
    data_path = '/content/drl-recsys/data'
    model_path = '/content/drl-recsys/model'
    embedding_network_weights_path = osp.join(model_path,'pmf/emb_50_ratio_0.800000_bs_1000_e_258_wd_0.100000_lr_0.000100_trained_pmf.pt')

    n_groups = 10
    fairness_weight = {k: 1.0 for k in range(1, n_groups + 1)}
    fairness_constraints = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    ENV = dict(drr=OfflineEnv, fairrec=OfflineFairEnv)
    AGENT = dict(drr=DRRAgent, fairrec=FairRecAgent)

    algorithm = "drr"

    train_ids = [
        "movie_lens_100k_2021-10-24_01-42-57", # long training
        "movie_lens_100k_fair_2021-10-24_01-41-02" # long training
    ]
    train_version = "movie_lens_100k"
    train_id = train_ids[1]

    output_path = osp.join(model_path,'{}/{}'.format(train_version, train_id))
    Path(output_path).mkdir(parents=True, exist_ok=True)

    users_num = 943
    items_num = 1682

    state_size = 5
    srm_size = 3
    dim_context = 150

    embedding_dim = 50
    actor_hidden_dim = 512
    actor_learning_rate = 0.0001
    critic_hidden_dim = 512
    critic_learning_rate = 0.001
    discount_factor = 0.9
    tau = 0.01
    learning_starts = 1000
    replay_memory_size = 1000000
    batch_size = 64
    emb_model = "user_movie"
    embedding_network_weights = embedding_network_weights_path

    top_k = [5, 10]
    done_count = 10
```

```python id="SQ0xyi_cT7Ms"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

<!-- #region id="Q40X4lHf4JHw" -->
### Logger
<!-- #endregion -->

```python id="cibwpV5L4JFb"
logging.basicConfig(stream=sys.stdout,
                    level = logging.DEBUG,
                    format='%(asctime)s [%(levelname)s] : %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')

logger = logging.getLogger('Logger')
```

<!-- #region id="2M0-cN2ZzWE-" -->
## Modules
<!-- #endregion -->

<!-- #region id="NZe9e1CW4MWH" -->
### Dataset
<!-- #endregion -->

<!-- #region id="ZhEALNsZJBZY" -->
The dataset module inherits from obp.dataset.base.BaseRealBanditDatset (docs) and should implement three methods:

- load_raw_data(): Load an on-disk representation of the dataset into the module. Used during initialization.
- pre_process(): Perform any preprocessing needed to transform the raw data representation into a final representation.
obtain_batch_bandit_feedback(): Return a dictionary containing (at - least) keys: ["action","position","reward","pscore","context","n_rounds"]

It is also helpful if the dataset module exposes a property len_list, which is how many items the bandit shows the user at a time. Often the answer is 1, though in the case of OBD it's 3.
<!-- #endregion -->

```python id="hPJDSXRp4MTs"
def load_movielens_dataset():
    dataset = MovieLensDataset(
        data_path=args.data_path,
        embedding_network_weights_path=args.embedding_network_weights_path,
        embedding_dim=50,
        users_num=943,
        items_num=1682,
        state_size=5,
        )
    args.dataset = dataset
```

```python id="zlsTaXukJnbM"
def load_dataset_dict():
    # dataset_path = osp.join(args.data_path,"movie_lens_100k_output_path.json")
    # with open(dataset_path) as json_file:
    #     _dataset_path = json.load(json_file)
    _dataset_path = {
        'eval_users_dict': osp.join(args.data_path,'ml-100k','eval_users_dict.pkl'),
        'eval_users_dict_positive_items': osp.join(args.data_path,'ml-100k','eval_users_dict_positive_items.pkl'),
        'eval_users_history_lens': osp.join(args.data_path,'ml-100k','eval_users_history_lens.pkl'),
        'movies_df': osp.join(args.data_path,'ml-100k','movies.csv'),
        'movies_groups': osp.join(args.data_path,'ml-100k','movies_groups.pkl'),
        'ratings_df': osp.join(args.data_path,'ml-100k','ratings.csv'),
        'train_users_dict': osp.join(args.data_path,'ml-100k','train_users_dict.pkl'),
        'train_users_history_lens': osp.join(args.data_path,'ml-100k','train_users_history_lens.pkl'),
        'users_df': osp.join(args.data_path,'ml-100k','users.csv'),
        'users_history_lens': osp.join(args.data_path,'ml-100k','users_history_lens.pkl'),
    }
    dataset = {}
    with open(os.path.join("..", _dataset_path["eval_users_dict"]), "rb") as pkl_file:
        dataset["eval_users_dict"] = pickle.load(pkl_file)
    with open(os.path.join("..", _dataset_path["eval_users_dict_positive_items"]), "rb") as pkl_file:
        dataset["eval_users_dict_positive_items"] = pickle.load(pkl_file)
    with open(os.path.join("..", _dataset_path["eval_users_history_lens"]), "rb") as pkl_file:
        dataset["eval_users_history_lens"] = pickle.load(pkl_file)
    with open(os.path.join("..", _dataset_path["users_history_lens"]), "rb") as pkl_file:
        dataset["users_history_lens"] = pickle.load(pkl_file)
    with open(os.path.join("..", _dataset_path["movies_groups"]), "rb") as pkl_file:
        dataset["movies_groups"] = pickle.load(pkl_file)
    args.dataset = dataset

    args.obp_dataset = MovieLensDataset(
        data_path=args.data_path,
        embedding_network_weights_path=args.embedding_network_weights_path,
        embedding_dim=50,
        users_num=943,
        items_num=1682,
        state_size=5,
        filter_ids=list(dataset["eval_users_dict"].keys())
    )
```

```python id="eEhzULvb4MPE"
def load_bandit_feedback():
    bandit_feedback = args.dataset.obtain_batch_bandit_feedback()
    args.bandit_feedback = bandit_feedback
    print("feedback dict:")
    for key, value in bandit_feedback.items():
        print(f"  {key}: {type(value)}")
    exp_rand_reward = round(bandit_feedback["reward"].mean(),4)
    args.exp_rand_reward = exp_rand_reward
    print(f"Expected reward for uniform random actions: {exp_rand_reward}")
```

```python id="45LiySTM4MM4"
def load_movie_groups():
    with open(osp.join(args.data_path,"ml-100k","movies_groups.pkl"), "rb") as pkl_file:
        movies_groups = pickle.load(pkl_file)
    args.movies_groups = movies_groups
```

<!-- #region id="8h-C7XvYI7yw" -->
### Bandits
<!-- #endregion -->

<!-- #region id="__Z02GYLCWRQ" -->
OPE attempts to estimate the performance of online bandit algorithms using the logged bandit feedback and ReplayMethod(RM).
<!-- #endregion -->

```python id="Ih-FDdIz6xK1"
def create_bandit():
    if args.policy_name=='EpsilonGreedy':
        args.policy = EpsilonGreedy(
            n_actions=args.dataset.n_actions,
            epsilon=0.1,
            n_group=args.n_groups,
            item_group=args.movies_groups,
            fairness_weight=args.fairness_weight,
            )
    if args.policy_name=='WFairLinUCB':
        args.policy = WFairLinUCB(
            dim=args.dataset.dim_context,
            n_actions=args.dataset.n_actions,
            epsilon=0.1,
            n_group=args.n_groups,
            item_group=args.movies_groups,
            fairness_weight=args.fairness_weight,
            batch_size=1,
        )
```

<!-- #region id="_m7Ec6BRP77n" -->
### Reward models
<!-- #endregion -->

```python id="mgw8-mIzP9b8"
def actor_critic_checkpoints():
    actor_checkpoint = sorted(
        [
            int((f.split("_")[1]).split(".")[0])
            for f in os.listdir(args.output_path)
            if f.startswith("actor_")
        ]
    )[-1]
    critic_checkpoint = sorted(
        [
            int((f.split("_")[1]).split(".")[0])
            for f in os.listdir(args.output_path)
            if f.startswith("critic_")
        ]
    )[-1]

    args.actor_checkpoint = actor_checkpoint
    args.critic_checkpoint = critic_checkpoint
    print(actor_checkpoint, critic_checkpoint)
```

```python id="yKXZIFfhUES-"
def pmf_model():
    args.reward_model = PMF(args.users_num, args.items_num, args.embedding_dim)
    args.reward_model.load_state_dict(
        torch.load(args.embedding_network_weights_path, map_location=torch.device(device))
    )
    args.user_embeddings = args.reward_model.user_embeddings.weight.data
    args.item_embeddings = args.reward_model.item_embeddings.weight.data
```

<!-- #region id="eA1yn82BUaR5" -->
### Metrics
<!-- #endregion -->

```python id="CyDe4JcoUaO9"
def calculate_ndcg(rel, irel):
    dcg = 0
    idcg = 0
    rel = [1 if r > 0 else 0 for r in rel]
    for i, (r, ir) in enumerate(zip(rel, irel)):
        dcg += (r) / np.log2(i + 2)
        idcg += (ir) / np.log2(i + 2)

    return dcg, idcg
```

<!-- #region id="e2Fqz5rII_dV" -->
### Simulation
<!-- #endregion -->

```python id="OS9IeIbB6xG6"
def bandit_simulation():
    action_dist, aligned_cvr, cvr, propfair, ufg, group_count = run_bandit_simulation(
        bandit_feedback=args.bandit_feedback,
        policy=args.policy,
        epochs=5,
        )
    args.sim_output = (action_dist, aligned_cvr, cvr, propfair, ufg, group_count)
```

```python id="wQjqqCM4QYZT"
def run_offline_evaluator():

    for K in args.top_k:
        _precision = []
        _ndcg = []
        for i in range(10):
            sum_precision = 0
            sum_ndcg = 0
            sum_propfair = 0

            env = args.ENV[args.algorithm](
                users_dict=args.dataset["eval_users_dict"],
                users_history_lens=args.dataset["eval_users_history_lens"],
                n_groups=args.n_groups,
                movies_groups=args.dataset["movies_groups"],
                state_size=args.state_size,
                done_count=args.done_count,
                fairness_constraints=args.fairness_constraints,
            )
            available_users = env.available_users

            recommender = args.AGENT[args.algorithm](
                env=env,
                users_num=args.users_num,
                items_num=args.items_num,
                genres_num=0,
                movies_genres_id={}, 
                srm_size=args.srm_size,
                state_size=args.state_size,
                train_version=args.train_version,
                is_test=True,
                embedding_dim=args.embedding_dim,
                actor_hidden_dim=args.actor_hidden_dim,
                actor_learning_rate=args.actor_learning_rate,
                critic_hidden_dim=args.critic_hidden_dim,
                critic_learning_rate=args.critic_learning_rate,
                discount_factor=args.discount_factor,
                tau=args.tau,
                replay_memory_size=args.replay_memory_size,
                batch_size=args.batch_size,
                model_path=args.output_path,
                emb_model=args.emb_model,
                embedding_network_weights_path=args.embedding_network_weights_path,
                n_groups=args.n_groups,
                fairness_constraints=args.fairness_constraints,
            )

            recommender.load_model(
                os.path.join(args.output_path, "actor_{}.h5".format(args.actor_checkpoint)),
                os.path.join(
                    args.output_path, "critic_{}.h5".format(args.critic_checkpoint)
                ),
            )
            for user_id in tqdm(available_users):
                eval_env = args.ENV[args.algorithm](
                    users_dict=args.dataset["eval_users_dict"],
                    users_history_lens=args.dataset["eval_users_history_lens"],
                    n_groups=args.n_groups,
                    movies_groups=args.dataset["movies_groups"],
                    state_size=args.state_size,
                    done_count=args.done_count,
                    fairness_constraints=args.fairness_constraints,
                    fix_user_id=user_id
                )

                recommender.env = eval_env
                available_items = set(eval_env.user_items.keys())

                precision, ndcg, propfair = recommender.evaluate(
                    eval_env, top_k=K, available_items=available_items
                )

                sum_precision += precision
                sum_ndcg += ndcg
                sum_propfair += propfair

                del eval_env

            _precision.append(sum_precision / len(args.dataset["eval_users_dict"]))
            _ndcg.append(sum_ndcg / len(args.dataset["eval_users_dict"]))

        print("Precision ", K, round(np.mean(_precision), 4), np.std(_precision))
        print("NDCG ", K, round(np.mean(_ndcg), 4), np.std(_ndcg))
```

```python id="JA7-x9UCUhja"
def run_offline_pmf_evaluator():
    for K in args.top_k:
        _precision = []
        _ndcg = []
        for i in range(10):
            sum_precision = 0
            sum_ndcg = 0
            sum_propfair = 0

            env = OfflineEnv(
                users_dict=args.dataset["eval_users_dict"],
                users_history_lens=args.dataset["eval_users_history_lens"],
                n_groups=args.n_groups,
                movies_groups=args.dataset["movies_groups"],
                state_size=args.state_size,
                done_count=args.done_count,
                fairness_constraints=args.fairness_constraints,
            )
            available_users = env.available_users

            with open(args.output_path, "rb") as pkl_file:
                recommender = pickle.load(pkl_file)

            recommender.len_list = K

            for user_id in tqdm(available_users):
                eval_env = OfflineEnv(
                    users_dict=args.dataset["eval_users_dict"],
                    users_history_lens=args.dataset["eval_users_history_lens"],
                    n_groups=args.n_groups,
                    movies_groups=args.dataset["movies_groups"],
                    state_size=args.state_size,
                    done_count=args.done_count,
                    fairness_constraints=args.fairness_constraints,
                    fix_user_id=user_id
                )

                available_items = set(eval_env.user_items.keys())

                # episodic reward
                episode_reward = 0
                steps = 0

                mean_precision = 0
                mean_ndcg = 0

                # Environment
                user_id, items_ids, done = env.reset()

                while not done:
                    # select a list of actions
                    if recommender.policy_type == PolicyType.CONTEXT_FREE:
                        selected_actions = recommender.select_action(list(available_items))
                    elif recommender.policy_type == PolicyType.CONTEXTUAL:
                        # observe current state & Find action
                        user_eb = args.user_embeddings[user_id]
                        items_eb = args.item_embeddings[items_ids]
                        item_ave = torch.mean(items_eb, 0)
                        context = torch.cat((user_eb, user_eb * item_ave, item_ave), 0).cpu().numpy()
                        context = context.reshape(1, 50)
                        selected_actions = recommender.select_action(context, list(available_items))
                    
                    ## Item
                    recommended_item = selected_actions

                    # Calculate reward and observe new state (in env)
                    ## Step
                    next_items_ids, reward, done, _ = env.step(recommended_item, top_k=K)
                    if top_k:
                        correct_list = [1 if r > 0 else 0 for r in reward]
                        # ndcg
                        dcg, idcg = calculate_ndcg(
                            correct_list, [1 for _ in range(len(reward))]
                        )
                        mean_ndcg += dcg / idcg

                        # precision
                        correct_num = K - correct_list.count(0)
                        mean_precision += correct_num / K
                    else:
                        mean_precision += 1 if reward > 0 else 0

                    reward = np.sum(reward)
                    items_ids = next_items_ids
                    episode_reward += reward
                    steps += 1
                    available_items = (
                        available_items - set(recommended_item) if available_items else None
                    )

                sum_precision += mean_precision / steps
                sum_ndcg += mean_ndcg / steps

                del eval_env

            _precision.append(sum_precision / len(args.dataset["eval_users_dict"]))
            _ndcg.append(sum_ndcg / len(args.dataset["eval_users_dict"]))

        print("Precision ", K, round(np.mean(_precision), 4), np.std(_precision))
        print("NDCG ", K, round(np.mean(_ndcg), 4), np.std(_ndcg))
```

<!-- #region id="TeEeka5bI9eV" -->
### Plotting
<!-- #endregion -->

```python id="fewJTQUW6xEt"
def plot_simulation_output():
    fig = go.Figure([
        go.Scatter(
            x=[i + 1 for i in range(len(args.sim_output[1]))],
            y=args.sim_output[1],
            name="CVR"
        ),
        go.Scatter(
            x=[i + 1 for i in range(len(args.sim_output[1]))],
            y=[args.exp_rand_reward for i in range(len(args.sim_output[1]))],
            name="Mean Reward"
        )
    ])
    fig.update_layout(title="EGreedy")
    fig.update_yaxes(range=[0, 1])
    fig.show()
```

<!-- #region id="P0nc1kyjzX9T" -->
## Jobs
<!-- #endregion -->

```python id="2y8mdDjds6dr" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1636787044767, "user_tz": -330, "elapsed": 16107, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="b9b37292-e42a-434c-a3ac-037608e8f69d"
logger.info('JOB START: LOAD_DATASET')
args = Args()
load_movielens_dataset()
load_bandit_feedback()
load_movie_groups()
logger.info('JOB END: LOAD_DATASET')
```

```python id="3ig3tPpB2Fx-" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1636787321955, "user_tz": -330, "elapsed": 1614, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="06dabcfb-c098-4efe-968a-f9e1a74fea84"
logger.info('JOB START: EPSILON_GREEDY_SIMULATION')
args.policy_name = 'EpsilonGreedy'
create_bandit()
bandit_simulation()
plot_simulation_output()
logger.info('JOB END: EPSILON_GREEDY_SIMULATION')
```

```python id="QUQgKlu4EPWJ"
logger.info('JOB START: WFAIR_LINUCB_SIMULATION')
args.policy_name = 'WFairLinUCB'
create_bandit()
bandit_simulation()
plot_simulation_output()
logger.info('JOB END: WFAIR_LINUCB_SIMULATION')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 1000} id="egS79T79jvVf" executionInfo={"status": "error", "timestamp": 1636796920065, "user_tz": -330, "elapsed": 707145, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="a5f7544d-aea9-4372-be85-39fff9b479a6"
#hide-output
logger.info('JOB START: MODEL_TRAINING_LUIGI_TASK')
luigi.build([DRLTrain()], workers=2, local_scheduler=True)
logger.info('JOB END: MODEL_TRAINING_LUIGI_TASK')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 367} id="JKdiHnxOGI7s" executionInfo={"status": "error", "timestamp": 1636791788820, "user_tz": -330, "elapsed": 454, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="9add69a1-f0f2-4067-f55b-3b4928a80847"
#hide-output
logger.info('JOB START: DRR_OFFLINE_EVALUATION')
args = Args()
args.algorithm = 'drr'
load_dataset_dict()
actor_critic_checkpoints()
run_offline_evaluator()
logger.info('JOB END: DRR_OFFLINE_EVALUATION')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 367} id="8_Uh-x5eSA4n" executionInfo={"status": "error", "timestamp": 1636791800454, "user_tz": -330, "elapsed": 666, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="bb1093fa-feb9-47e3-f3fb-8b3084c51590"
#hide-output
logger.info('JOB START: DRR_OFFLINE_EVALUATION')
args = Args()
args.train_ids = [
    "egreedy_0.1_2021-10-29_23-50-32.pkl",
    "linear_ucb_0.1_2021-11-04_15-01-07.pkl",
    "wfair_linear_ucb_0.1_2021-11-04_15-01-15.pkl"
]
args.train_version = "bandits"
args.train_id = args.train_ids[2]
load_dataset_dict()
run_offline_pmf_evaluator()
logger.info('JOB END: DRR_OFFLINE_EVALUATION')
```

```python id="Z-7Yh5Vziqnu"
TRAINER = dict(
    movie_lens_100k=MovieLens,
    movie_lens_100k_fair=MovieLens,
)


class DRLTrain(luigi.Task):
    use_wandb: bool = luigi.BoolParameter()
    load_model: bool = luigi.BoolParameter()
    evaluate: bool = luigi.BoolParameter()
    train_version: str = luigi.Parameter(default="movie_lens_100k")
    dataset_version: str = luigi.Parameter(default="movie_lens_100k")
    train_id: str = luigi.Parameter(default="")

    def __init__(self, *args, **kwargs):
        super(DRLTrain, self).__init__(*args, **kwargs)

        if len(self.train_id) > 0:
            self.output_path = os.path.join(
                args.model_path, self.train_version, self.train_id
            )
        else:
            dtime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.output_path = os.path.join(
                args.model_path,
                self.train_version,
                str(self.train_version + "_{}".format(dtime)),
            )
            os.makedirs(self.output_path, exist_ok=True)
            os.makedirs(os.path.join(self.output_path, "images"), exist_ok=True)

    def run(self):
        print("---------- Generate Dataset")
        dataset = yield DatasetGeneration(self.dataset_version)

        print("---------- Train Model")
        train = yield TRAINER[self.train_version](
            **self.train_config["model_train"],
            users_num=self.train_config["users_num"],
            items_num=self.train_config["items_num"],
            embedding_dim=self.train_config["embedding_dim"],
            emb_model=self.train_config["emb_model"],
            output_path=self.output_path,
            train_version=self.train_version,
            use_wandb=self.use_wandb,
            load_model=self.load_model,
            dataset_path=dataset.path,
            evaluate=self.evaluate,
        )

    @property
    def train_config(self):
        path = os.path.abspath(
            os.path.join("model", "{}.yaml".format(self.train_version))
        )

        with open(path) as f:
            train_config = yaml.load(f, Loader=yaml.FullLoader)

        return train_config
```

```python id="MCUPDe7hVQTi"
with open(args.output_path, "rb") as pkl_file:
    bandit = pickle.load(pkl_file)

bandit.len_list = 10

selected_actions_list = list()
estimated_rewards = list() 
for index, row in tqdm(obp_dataset.data.iterrows(), total=obp_dataset.data.shape[0]):

    action_ = row["movie_id"]
    reward_ = 0 if row["rating"] < 4 else 1
    user_eb = user_embeddings[row["user_id"]]
    items_eb = item_embeddings[row["item_id_history"]]
    item_ave = torch.mean(items_eb, 0)
    context_ = torch.cat((user_eb, user_eb * item_ave, item_ave), 0).cpu().numpy()

    # select a list of actions
    if bandit.policy_type == PolicyType.CONTEXT_FREE:
        selected_actions = bandit.select_action()
    elif bandit.policy_type == PolicyType.CONTEXTUAL:
        selected_actions = bandit.select_action(
            context_.reshape(1, dim_context)
        )
    action_match_ = action_ == selected_actions[0]
    # update parameters of a bandit policy
    # only when selected actions&positions are equal to logged actions&positions
    if action_match_:
        if bandit.policy_type == PolicyType.CONTEXT_FREE:
            bandit.update_params(action=action_, reward=reward_)
        elif bandit.policy_type == PolicyType.CONTEXTUAL:
            bandit.update_params(
                action=action_,
                reward=reward_,
                context=context_.reshape(1, dim_context),
            )

    

    selected_actions_list.append(selected_actions)
100%|██████████| 16983/16983 [26:23<00:00, 10.73it/s]
_df = obp_dataset.data.copy()
_df["sorted_actions"] = selected_actions_list
_item_metadata = pd.DataFrame(dataset["movies_groups"].items(), columns=["movie_id", "group"])
_df.to_csv("./df.csv", index=False)
_item_metadata.to_csv("./item.csv", index=False)
import numpy as np
def converter(instr):
    return np.fromstring(instr[1:-1],sep=' ')

_df=pd.read_csv("./df.csv",converters={'sorted_actions':converter})
_item_metadata = pd.read_csv("./item.csv")
user_column = "user_id"
item_column = "movie_id"
reclist_column = "sorted_actions"

recsys_fair = RecsysFair(
    df = _df, 
    supp_metadata = _item_metadata,
    user_column = user_column, 
    item_column = item_column, 
    reclist_column = reclist_column, 
)

fair_column = "group"
ex = recsys_fair.exposure(fair_column, 10)
100%|██████████| 16983/16983 [00:00<00:00, 4578471.84it/s]
fig = ex.show(kind='per_group_norm', column=fair_column)
fig.show()
#fig.write_image("exposure_per_group.png")
fig = ex.show(kind='per_rank_pos', column=fair_column)
fig.write_image("exposure_per_rank.png")
```
