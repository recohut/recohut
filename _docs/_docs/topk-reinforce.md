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

<!-- #region id="Skc9_mUcfxTx" -->
# Top-K Off-Policy Correction for a REINFORCE Recommender System
<!-- #endregion -->

<!-- #region id="cXgEkn6wbXSb" -->
## CLI run
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="yGjiMX5nVGr2" executionInfo={"status": "ok", "timestamp": 1634811527051, "user_tz": -330, "elapsed": 68164, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="365fde77-8a2a-4d0a-f5fe-c6b59d2da8e1"
!gdown --id 1erBjYEOa7IuOIGpI8pGPn1WNBAC4Rv0-
!git clone https://github.com/massquantity/DBRL.git
!unzip /content/ECommAI_EUIR_round2_train_20190821.zip
!mv ECommAI_EUIR_round2_train_20190816/*.csv DBRL/dbrl/resources
%cd DBRL/dbrl
```

```python colab={"base_uri": "https://localhost:8080/"} id="-SiMsatqVp6a" executionInfo={"status": "ok", "timestamp": 1634811700903, "user_tz": -330, "elapsed": 173863, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="041071ae-c7c0-4437-f5f7-5585f3291bbd"
!python run_prepare_data.py
```

```python colab={"base_uri": "https://localhost:8080/"} id="2Bhb2NHZV7pH" executionInfo={"status": "ok", "timestamp": 1634812382531, "user_tz": -330, "elapsed": 681644, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="4a80ba3f-46ed-4a07-cf05-9b2d118bfab9"
!python run_pretrain_embeddings.py --lr 0.001 --n_epochs 4
```

```python colab={"base_uri": "https://localhost:8080/"} id="825CG5pQWous" executionInfo={"status": "ok", "timestamp": 1634823180654, "user_tz": -330, "elapsed": 10050064, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="a3f0e9d5-0e8a-4bac-a9af-bd6fd1af29df"
!python run_reinforce.py --n_epochs 1 --lr 1e-5
```

```python colab={"base_uri": "https://localhost:8080/"} id="3_d3DNySXFhc" executionInfo={"status": "ok", "timestamp": 1634823421135, "user_tz": -330, "elapsed": 7393, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="5347b11c-e1ce-44dd-84d4-2bd7005f724f"
!apt-get -qq install tree
!tree --du -h .
```

<!-- #region id="kxN_Fb0fbVCX" -->
## Code analysis
<!-- #endregion -->

<!-- #region id="pRGngazAbWdJ" -->
### Data preparation
<!-- #endregion -->

```python id="zecpI005bbOk"
import os
import sys
sys.path.append(os.pardir)
import warnings
warnings.filterwarnings("ignore")
import argparse
import time
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="run_prepare_data")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(args={})


def bucket_age(age):
    if age < 30:
        return 1
    elif age < 40:
        return 2
    elif age < 50:
        return 3
    else:
        return 4


if __name__ == "__main__":
    args = parse_args()
    print(vars(args))
    np.random.seed(args.seed)
    start_time = time.perf_counter()

    # 1. loading the data into memory

    user_feat = pd.read_csv("resources/user.csv", header=None,
                            names=["user", "sex", "age", "pur_power"])
    item_feat = pd.read_csv("resources/item.csv", header=None,
                            names=["item", "category", "shop", "brand"])
    behavior = pd.read_csv("resources/user_behavior.csv", header=None,
                           names=["user", "item", "behavior", "time"])
    
    # 2. sorting values chronologically and dropping duplicate records

    behavior = behavior.sort_values(by="time").reset_index(drop=True)
    behavior = behavior.drop_duplicates(subset=["user", "item", "behavior"])

    # 3. Choosing 60K random users with short journey and 20K with long journey
    user_counts = behavior.groupby("user")[["user"]].count().rename(
        columns={"user": "count_user"}
    ).sort_values("count_user", ascending=False)

    short_users = np.array(
        user_counts[
            (user_counts.count_user > 5) & (user_counts.count_user <= 50)
        ].index
    )
    long_users = np.array(
        user_counts[
            (user_counts.count_user > 50) & (user_counts.count_user <= 200)
        ].index
    )
    short_chosen_users = np.random.choice(short_users, 60000, replace=False)
    long_chosen_users = np.random.choice(long_users, 20000, replace=False)
    chosen_users = np.concatenate([short_chosen_users, long_chosen_users])

    behavior = behavior[behavior.user.isin(chosen_users)]
    print(f"n_users: {behavior.user.nunique()}, "
          f"n_items: {behavior.item.nunique()}, "
          f"behavior length: {len(behavior)}")

    # 4. merge with all features, bucketizing the age and saving the processed data
    behavior = behavior.merge(user_feat, on="user")
    behavior = behavior.merge(item_feat, on="item")
    behavior["age"] = behavior["age"].apply(bucket_age)
    behavior = behavior.sort_values(by="time").reset_index(drop=True)
    behavior.to_csv("resources/tianchi.csv", header=None, index=False)
    print(f"prepare data done!, "
          f"time elapsed: {(time.perf_counter() - start_time):.2f}")
```

<!-- #region id="n1C4mcuIdRaW" -->
### Embeddings
<!-- #endregion -->

```python id="lH2RLv5ddSK-"
import os
import sys
sys.path.append(os.pardir)
import warnings
warnings.filterwarnings("ignore")
import argparse
from pprint import pprint
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from dbrl.data import process_feat_data, FeatDataset
from dbrl.models import DSSM
from dbrl.utils import sample_items_random, init_param_dssm, generate_embeddings
from dbrl.trainer import pretrain_model
from dbrl.serialization import save_npy, save_json


def parse_args():
    parser = argparse.ArgumentParser(description="run_pretrain_embeddings")
    parser.add_argument("--data", type=str, default="tianchi.csv")
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--embed_size", type=int, default=32)
    parser.add_argument("--loss", type=str, default="cosine",
                        help="cosine or bce loss")
    parser.add_argument("--neg_item", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("A list all args: \n======================")
    pprint(vars(args))
    print()

    # 1. Setting arguments/params

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    PATH = os.path.join("resources", args.data)
    EMBEDDING_PATH = "resources/"
    static_feat = ["sex", "age", "pur_power"]
    dynamic_feat = ["category", "shop", "brand"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    lr = args.lr
    item_embed_size = args.embed_size
    feat_embed_size = args.embed_size
    hidden_size = (256, 128)
    criterion = (
        nn.CosineEmbeddingLoss()
        if args.loss == "cosine"
        else nn.BCEWithLogitsLoss()
    )
    criterion_type = (
        "cosine"
        if "cosine" in criterion.__class__.__name__.lower()
        else "bce"
    )
    neg_label = -1. if criterion_type == "cosine" else 0.
    neg_item = args.neg_item

    # 2. Preprocessing

    columns = ["user", "item", "label", "time", "sex", "age", "pur_power",
               "category", "shop", "brand"]

    (
        n_users,
        n_items,
        train_user_consumed,
        eval_user_consumed,
        train_data,
        eval_data,
        user_map,
        item_map,
        feat_map
    ) = process_feat_data(
        PATH, columns, test_size=0.2, time_col="time",
        static_feat=static_feat, dynamic_feat=dynamic_feat
    )
    print(f"n_users: {n_users}, n_items: {n_items}, "
          f"train_shape: {train_data.shape}, eval_shape: {eval_data.shape}")
    
    # 3. Random negative sampling

    train_user, train_item, train_label = sample_items_random(
        train_data, n_items, train_user_consumed, neg_label, neg_item
    )
    eval_user, eval_item, eval_label = sample_items_random(
        eval_data, n_items, eval_user_consumed, neg_label, neg_item
    )

    # 4. Putting data into torch dataset format and dataloader

    train_dataset = FeatDataset(
        train_user,
        train_item,
        train_label,
        feat_map,
        static_feat,
        dynamic_feat
    )
    eval_dataset = FeatDataset(
        eval_user,
        eval_item,
        eval_label,
        feat_map,
        static_feat,
        dynamic_feat
    )
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=0)
    eval_loader = DataLoader(dataset=eval_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=0)

    # 5. DSSM embedding model training

    model = DSSM(
        item_embed_size,
        feat_embed_size,
        n_users,
        n_items,
        hidden_size,
        feat_map,
        static_feat,
        dynamic_feat,
        use_bn=True
    ).to(device)
    init_param_dssm(model)
    optimizer = Adam(model.parameters(), lr=lr)  # weight_decay

    pretrain_model(model, train_loader, eval_loader, n_epochs, criterion,
                   criterion_type, optimizer, device)
    
    # 6. Generate and save embeddings
    
    user_embeddings, item_embeddings = generate_embeddings(
        model, n_users, n_items, feat_map, static_feat, dynamic_feat, device
    )
    print(f"user_embeds shape: {user_embeddings.shape},"
          f" item_embeds shape: {item_embeddings.shape}")

    save_npy(user_embeddings, item_embeddings, EMBEDDING_PATH)
    save_json(
        user_map, item_map, user_embeddings, item_embeddings, EMBEDDING_PATH
    )
    print("pretrain embeddings done!")
```

<!-- #region id="oNVi23z3etT_" -->
## REINFORCE model
<!-- #endregion -->

```python id="s7upYlxygH08"
import torch
import torch.nn as nn
from torch.distributions import Categorical


class Reinforce(nn.Module):
    def __init__(
            self,
            policy,
            policy_optim,
            beta,
            beta_optim,
            hidden_size,
            gamma=0.99,
            k=10,
            weight_clip=2.0,
            offpolicy_correction=True,
            topk=True,
            adaptive_softmax=True,
            cutoffs=None,
            device=torch.device("cpu"),
    ):
        super(Reinforce, self).__init__()
        self.policy = policy
        self.policy_optim = policy_optim
        self.beta = beta
        self.beta_optim = beta_optim
        self.beta_criterion = nn.CrossEntropyLoss()
        self.gamma = gamma
        self.k = k
        self.weight_clip = weight_clip
        self.offpolicy_correction = offpolicy_correction
        self.topk = topk
        self.adaptive_softmax = adaptive_softmax
        if adaptive_softmax:
            assert cutoffs is not None, (
                "must provide cutoffs when using adaptive_softmax"
            )
            self.softmax_loss = nn.AdaptiveLogSoftmaxWithLoss(
                in_features=hidden_size,
                n_classes=policy.item_embeds.weight.size(0),
                cutoffs=cutoffs,
                div_value=4.
            ).to(device)
        self.device = device

    def update(self, data):
        (
            policy_loss,
            beta_loss,
            action,
            importance_weight,
            lambda_k
        ) = self._compute_loss(data)

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        self.beta_optim.zero_grad()
        beta_loss.backward()
        self.beta_optim.step()

        info = {'policy_loss': policy_loss.cpu().detach().item(),
                'beta_loss': beta_loss.cpu().detach().item(),
                'importance_weight': importance_weight.cpu().mean().item(),
                'lambda_k': lambda_k.cpu().mean().item(),
                'action': action}
        return info

    def _compute_weight(self, policy_logp, beta_logp):
        if self.offpolicy_correction:
            importance_weight = torch.exp(policy_logp - beta_logp).detach()
            wc = torch.tensor([self.weight_clip]).to(self.device)
            importance_weight = torch.min(importance_weight, wc)
        #    importance_weight = torch.clamp(
        #        importance_weight, self.weight_clip[0], self.weight_clip[1]
        #    )
        else:
            importance_weight = torch.tensor([1.]).float().to(self.device)
        return importance_weight

    def _compute_lambda_k(self, policy_logp):
        lam = (
            self.k * ((1. - policy_logp.exp()).pow(self.k - 1)).detach()
            if self.topk
            else torch.tensor([1.]).float().to(self.device)
        )
        return lam

    def _compute_loss(self, data):
        if self.adaptive_softmax:
            state, action = self.policy(data)
            policy_out = self.softmax_loss(action, data["action"])
            policy_logp = policy_out.output

            beta_action = self.beta(state.detach())
            beta_out = self.softmax_loss(beta_action, data["action"])
            beta_logp = beta_out.output
        else:
            state, all_logp, action = self.policy.get_log_probs(data)
            policy_logp = all_logp[:, data["action"]]

            b_logp, beta_logits = self.beta.get_log_probs(state.detach())
            beta_logp = (b_logp[:, data["action"]]).detach()

        importance_weight = self._compute_weight(policy_logp, beta_logp)
        lambda_k = self._compute_lambda_k(policy_logp)

        policy_loss = -(
                importance_weight * lambda_k * data["return"] * policy_logp
        ).mean()

        if self.adaptive_softmax:
            if "beta_label" in data:
                b_state = self.policy.get_beta_state(data)
                b_action = self.beta(b_state.detach())
                b_out = self.softmax_loss(b_action, data["beta_label"])
                beta_loss = b_out.loss
            else:
                beta_loss = beta_out.loss
        else:
            if "beta_label" in data:
                b_state = self.policy.get_beta_state(data)
                _, b_logits = self.beta.get_log_probs(b_state.detach())
                beta_loss = self.beta_criterion(b_logits, data["beta_label"])
            else:
                beta_loss = self.beta_criterion(beta_logits, data["action"])
        return policy_loss, beta_loss, action, importance_weight, lambda_k

    def compute_loss(self, data):
        (
            policy_loss,
            beta_loss,
            action,
            importance_weight,
            lambda_k
        ) = self._compute_loss(data)

        info = {'policy_loss': policy_loss.cpu().detach().item(),
                'beta_loss': beta_loss.cpu().detach().item(),
                'importance_weight': importance_weight.cpu().mean().item(),
                'lambda_k': lambda_k.cpu().mean().item(),
                'action': action}
        return info

    def get_log_probs(self, data=None, action=None):
        with torch.no_grad():
            if self.adaptive_softmax:
                if action is None:
                    _, action = self.policy.forward(data)
                log_probs = self.softmax_loss.log_prob(action)
            else:
            #    _, log_probs = self.policy.get_log_probs(data)
                if action is None:
                    _, action = self.policy.forward(data)
                log_probs = self.policy.softmax_fc(action)
        return log_probs

    def forward(self, state):
        policy_logits = self.policy.get_action(state)
        policy_dist = Categorical(logits=policy_logits)
        _, rec_idxs = torch.topk(policy_dist.probs, 10, dim=1)
        return rec_idxs
```

<!-- #region id="E7fBLQGCf7gE" -->
## Trainer
<!-- #endregion -->

```python id="L0AxMTlueutE"
import os
import sys
sys.path.append(os.pardir)
import warnings
warnings.filterwarnings("ignore")
import argparse
from pprint import pprint
import numpy as np
import torch
from torch.optim import Adam
from dbrl.data import process_data, build_dataloader
from dbrl.models import Reinforce
from dbrl.network import PolicyPi, Beta
from dbrl.trainer import train_model
from dbrl.utils import count_vars, init_param


def parse_args():
    parser = argparse.ArgumentParser(description="run_reinforce")
    parser.add_argument("--data", type=str, default="tianchi.csv")
    parser.add_argument("--user_embeds", type=str,
                        default="tianchi_user_embeddings.npy")
    parser.add_argument("--item_embeds", type=str,
                        default="tianchi_item_embeddings.npy")
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--hist_num", type=int, default=10,
                        help="num of history items to consider")
    parser.add_argument("--n_rec", type=int, default=10,
                        help="num of items to recommend")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--sess_mode", type=str, default="interval",
                        help="Specify when to end a session")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("A list all args: \n======================")
    pprint(vars(args))
    print()

    # 1. Loading user and item embeddings

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PATH = os.path.join("resources", args.data)
    with open(os.path.join("resources", args.user_embeds), "rb") as f:
        user_embeddings = np.load(f)
    with open(os.path.join("resources", args.item_embeds), "rb") as f:
        item_embeddings = np.load(f)
    item_embeddings[-1] = 0.   # last item is used for padding

    # 2. Setting model arguments/params

    n_epochs = args.n_epochs
    hist_num = args.hist_num
    batch_size = eval_batch_size = args.batch_size
    embed_size = item_embeddings.shape[1]
    hidden_size = args.hidden_size
    input_dim = embed_size * (hist_num + 1)
    action_dim = len(item_embeddings)
    policy_lr = args.lr
    beta_lr = args.lr
    weight_decay = args.weight_decay
    gamma = args.gamma
    n_rec = args.n_rec
    pad_val = len(item_embeddings) - 1
    sess_mode = args.sess_mode
    debug = True
    one_hour = int(60 * 60)
    reward_map = {"pv": 1., "cart": 2., "fav": 2., "buy": 3.}
    columns = ["user", "item", "label", "time", "sex", "age", "pur_power",
               "category", "shop", "brand"]

    cutoffs = [
        len(item_embeddings) // 20,
        len(item_embeddings) // 10,
        len(item_embeddings) // 3
    ]

    # 3. Building the data loader

    (
        n_users,
        n_items,
        train_user_consumed,
        test_user_consumed,
        train_sess_end,
        test_sess_end,
        train_rewards,
        test_rewards
    ) = process_data(PATH, columns, 0.2, time_col="time", sess_mode=sess_mode,
                     interval=one_hour, reward_shape=reward_map)

    train_loader, eval_loader = build_dataloader(
        n_users,
        n_items,
        hist_num,
        train_user_consumed,
        test_user_consumed,
        batch_size,
        sess_mode=sess_mode,
        train_sess_end=train_sess_end,
        test_sess_end=test_sess_end,
        n_workers=0,
        compute_return=True,
        neg_sample=False,
        train_rewards=train_rewards,
        test_rewards=test_rewards,
        reward_shape=reward_map
    )

    # 4. Building the model

    policy = PolicyPi(
        input_dim, action_dim, hidden_size, user_embeddings,
        item_embeddings, None, pad_val, 1, device
    ).to(device)
    beta = Beta(input_dim, action_dim, hidden_size).to(device)
    init_param(policy, beta)

    policy_optim = Adam(policy.parameters(), policy_lr, weight_decay=weight_decay)
    beta_optim = Adam(beta.parameters(), beta_lr, weight_decay=weight_decay)

    model = Reinforce(
        policy,
        policy_optim,
        beta,
        beta_optim,
        hidden_size,
        gamma,
        k=10,
        weight_clip=2.0,
        offpolicy_correction=True,
        topk=True,
        adaptive_softmax=False,
        cutoffs=cutoffs,
        device=device,
    )

    var_counts = tuple(count_vars(module) for module in [policy, beta])
    print(f'Number of parameters: policy: {var_counts[0]}, '
          f' beta: {var_counts[1]}')
    
    # 5. Training the model

    train_model(
        model,
        n_epochs,
        n_rec,
        n_users,
        train_user_consumed,
        test_user_consumed,
        hist_num,
        train_loader,
        eval_loader,
        item_embeddings,
        eval_batch_size,
        pad_val,
        device,
        debug=debug,
        eval_interval=10
    )

    torch.save(policy.state_dict(), "resources/model_reinforce.pt")
    print("train and save done!")
```
