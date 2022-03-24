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

<!-- #region id="yKsArBS7bUEz" -->
# Optimal Off-Policy Evaluation from Multiple Logging Policies
<!-- #endregion -->

<!-- #region id="Af03hWaQVe1m" -->
## Imports
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="gtLCLFvuVnyY" executionInfo={"status": "ok", "timestamp": 1633519327877, "user_tz": -330, "elapsed": 20, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="4736b922-98ca-44a0-87a4-c2f36a43fc26"
%tensorflow_version 1.x
```

```python id="GeM0qrS3VezY" executionInfo={"status": "ok", "timestamp": 1633519555309, "user_tz": -330, "elapsed": 539, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
from typing import Dict, List
import argparse
import time
import pickle
import warnings
from pathlib import Path

import yaml
import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning

import tensorflow as tf
from tensorflow.python.framework import ops

warnings.filterwarnings(action="ignore", category=ConvergenceWarning)
```

<!-- #region id="vE6yqaJoWIdd" -->
## Configurations
<!-- #endregion -->

```python id="RgxG3QP2WIa2" executionInfo={"status": "ok", "timestamp": 1633519451927, "user_tz": -330, "elapsed": 670, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
!mkdir -p conf
```

```python colab={"base_uri": "https://localhost:8080/"} id="bEnQtqYQWL6Q" executionInfo={"status": "ok", "timestamp": 1633519498360, "user_tz": -330, "elapsed": 820, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="bab28d5e-ca21-4008-8a90-babe9dcf255c"
%%writefile conf/policy_params.yaml
evaluation: 1.0
behavior1: 0.95
behavior2: 0.05
```

```python colab={"base_uri": "https://localhost:8080/"} id="SfCBk0ZIWL33" executionInfo={"status": "ok", "timestamp": 1633519498953, "user_tz": -330, "elapsed": 7, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="cd100dd3-a426-4763-9980-353d8d96e23c"
%%writefile conf/q_func_hyperparams.yaml
eta: 0.01
std: 0.01
lam: 0.001
batch_size: 256
epochs: 200
```

<!-- #region id="hFk96QhgVew-" -->
## Dataset
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="2LuwCjZNYYxp" executionInfo={"status": "ok", "timestamp": 1633520205872, "user_tz": -330, "elapsed": 3205, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="83aaba30-a4eb-4857-b8b7-a95b542f824e"
!mkdir -p data/optdigits
!wget -q --show-progress -O data/optdigits/optdigits.tra https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra
!wget -q --show-progress -O data/optdigits/optdigits.tes https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes
```

```python id="HsZC26TzVVDl" executionInfo={"status": "ok", "timestamp": 1633519685697, "user_tz": -330, "elapsed": 555, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
def load_datasets(
    data: str, ratio: float, test_size: float = 0.5, random_state: int = 12345
):
    """Load and preprocess raw multiclass classification data."""
    data_path = Path(f"data/{data}")
    le = LabelEncoder()
    if data == "optdigits":
        data_ = np.r_[
            np.loadtxt(data_path / f"{data}.tra", delimiter=","),
            np.loadtxt(data_path / f"{data}.tes", delimiter=","),
        ]
    elif data == "pendigits":
        data_ = np.r_[
            np.loadtxt(data_path / f"{data}.tra", delimiter=","),
            np.loadtxt(data_path / f"{data}.tes", delimiter=","),
        ]
    elif data == "sat":
        data_ = np.r_[
            np.loadtxt(data_path / f"{data}.trn", delimiter=" "),
            np.loadtxt(data_path / f"{data}.tst", delimiter=" "),
        ]
        data_[:, -1] = np.where(data_[:, -1] == 7, 5, data_[:, -1] - 1)
    elif data == "letter":
        data_ = np.genfromtxt(
            data_path / "letter-recognition.data", delimiter=",", dtype="str"
        )
        data_ = np.c_[data_[:, 1:], le.fit_transform(data_[:, 0])].astype(float)

    np.random.shuffle(data_)
    data_tr, data_ev = train_test_split(
        data_, test_size=test_size, random_state=random_state
    )
    n_train, n_eval = data_tr.shape[0], data_ev.shape[0]
    n_dim = np.int(data_tr.shape[1] / 2)
    y_tr, y_ev = data_tr[:, -1].astype(int), data_ev[:, -1].astype(int)
    n_class = np.unique(y_tr).shape[0]
    y_full_ev = np.zeros((n_eval, n_class))
    y_full_ev[np.arange(n_eval), y_ev] = 1
    X_tr, X_ev = data_tr[:, :-1], data_ev[:, :-1]
    X_tr1, X_tr2 = data_tr[:, :n_dim], data_tr[:, n_dim:]
    X_ev1, X_ev2 = data_ev[:, :n_dim], data_ev[:, n_dim:]

    # multiple logger index generation
    ratio1 = ratio / (1 + ratio)
    n_eval1 = np.int(n_eval * ratio1)
    idx1 = np.ones(n_eval, dtype=bool)
    idx1[n_eval1:] = False

    return dict(
        n_train=n_train,
        n_eval=n_eval,
        n_dim=n_dim,
        n_class=n_class,
        n_behavior_policies=2,
        X_tr=X_tr,
        X_tr1=X_tr1,
        X_tr2=X_tr2,
        X_ev=X_ev,
        X_ev1=X_ev1,
        X_ev2=X_ev2,
        y_tr=y_tr,
        y_ev=y_ev,
        y_full_ev=y_full_ev,
        idx1=idx1,
        ratio1=(n_eval1 / n_eval),
    )


def generate_bandit_feedback(data_dict: Dict, pi_b1: np.ndarray, pi_b2: np.ndarray):
    """Generate logged bandit feedback data."""
    n_eval = data_dict["n_eval"]
    idx1, ratio1 = data_dict["idx1"], data_dict["ratio1"]
    idx1_expanded = np.expand_dims(idx1, 1)
    pi_b = pi_b1 * idx1_expanded + pi_b2 * (1 - idx1_expanded)
    pi_b_star = pi_b1 * ratio1 + pi_b2 * (1.0 - ratio1)
    action_set = np.arange(data_dict["n_class"])
    actions = np.zeros(data_dict["n_eval"], dtype=int)
    for i, pvals in enumerate(pi_b):
        actions[i] = np.random.choice(action_set, p=pvals)
    rewards = data_dict["y_full_ev"][np.arange(n_eval), actions]
    return dict(
        n_eval=data_dict["n_eval"],
        n_class=data_dict["n_class"],
        X_ev=data_dict["X_ev"],
        pi_b=pi_b,
        pi_b_star=pi_b_star,
        actions=actions,
        idx1=idx1,
        rewards=rewards,
    )
```

<!-- #region id="5eQuiovsV0RU" -->
## Policies
<!-- #endregion -->

```python id="3K1uMnSoV0OM" executionInfo={"status": "ok", "timestamp": 1633519499605, "user_tz": -330, "elapsed": 6, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
def train_policies(data_dict: Dict, random_state: int = 0) -> List[np.ndarray]:
    """Train evaluation and behavior policies."""
    with open("./conf/policy_params.yaml", "rb") as f:
        policy_params = yaml.safe_load(f)

    policy_list = list()
    for pol in policy_params.keys():
        # make label predictions
        X_tr, y_tr = data_dict[f"X_tr"], data_dict[f"y_tr"]
        clf = LogisticRegression(
            random_state=random_state,
            solver="lbfgs",
            multi_class="multinomial",
        ).fit(X=X_tr, y=y_tr)
        preds = clf.predict(X=data_dict[f"X_ev"]).astype(int)
        # transform predictions into distribution over actions
        alpha = policy_params[pol]
        pi = np.zeros((data_dict["n_eval"], data_dict["n_class"]))
        pi[:, :] = (1.0 - alpha) / data_dict["n_class"]
        pi[np.arange(data_dict["n_eval"]), preds] = (
            alpha + (1.0 - alpha) / data_dict["n_class"]
        )
        policy_list.append(pi)
    return policy_list
```

<!-- #region id="-MdsSR2HVvZK" -->
## Estimators
<!-- #endregion -->

```python id="HSBmxB3PVvXD" executionInfo={"status": "ok", "timestamp": 1633519501438, "user_tz": -330, "elapsed": 1286, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
def calc_ground_truth(y_true: np.ndarray, pi: np.ndarray) -> float:
    """Calculate the ground-truth policy value of an eval policy"""
    return pi[np.arange(y_true.shape[0]), y_true].mean()


def calc_ipw(
    rewards: np.ndarray,
    actions: np.ndarray,
    pi_b: np.ndarray,
    pi_e: np.ndarray,
) -> float:
    n_data = actions.shape[0]
    iw = pi_e[np.arange(n_data), actions] / pi_b[np.arange(n_data), actions]
    return (rewards * iw).mean()


def calc_var(
    rewards: np.ndarray,
    actions: np.ndarray,
    pi_b: np.ndarray,
    pi_e: np.ndarray,
    estimated_q_func: np.ndarray,
):
    n_data = actions.shape[0]
    v = np.average(estimated_q_func, weights=pi_e, axis=1)
    shifted_rewards = rewards - estimated_q_func[np.arange(n_data), actions]
    iw = pi_e[np.arange(n_data), actions] / pi_b[np.arange(n_data), actions]
    return np.var(shifted_rewards * iw + v)


def calc_weighted(
    rewards: np.ndarray,
    actions: np.ndarray,
    idx1: np.ndarray,
    pi_b: np.ndarray,
    pi_e: np.ndarray,
    estimated_q_func: np.ndarray = None,
    n_fold: int = 2,
) -> float:
    estimated_rewards_list = list()
    if estimated_q_func is None:
        estimated_q_func = np.zeros((actions.shape[0], np.int(actions.max() + 1)))
    kf = KFold(n_splits=n_fold, shuffle=True, random_state=12345)
    for train_idx, test_idx in kf.split(rewards):
        rewards_tr, rewards_ev = rewards[train_idx], rewards[test_idx]
        actions_tr, actions_ev = actions[train_idx], actions[test_idx]
        idx1_tr, idx1_ev = idx1[train_idx], idx1[test_idx]
        pi_b_tr, pi_b_ev = pi_b[train_idx], pi_b[test_idx]
        pi_e_tr, pi_e_ev = pi_e[train_idx], pi_e[test_idx]
        estimated_q_func_tr = estimated_q_func[train_idx]
        estimated_q_func_ev = estimated_q_func[test_idx]
        # estimate lambda with one of the fold
        n_data1, n_data2 = idx1_tr.sum(), (~idx1_tr).sum()
        var1 = calc_var(
            rewards=rewards_tr[idx1_tr],
            actions=actions_tr[idx1_tr],
            pi_b=pi_b_tr[idx1_tr],
            pi_e=pi_e_tr[idx1_tr],
            estimated_q_func=estimated_q_func_tr[idx1_tr],
        )
        var2 = calc_var(
            rewards=rewards_tr[~idx1_tr],
            actions=actions_tr[~idx1_tr],
            pi_b=pi_b_tr[~idx1_tr],
            pi_e=pi_e_tr[~idx1_tr],
            estimated_q_func=estimated_q_func_tr[~idx1_tr],
        )
        denominator = (n_data1 / var1) + (n_data2 / var2)
        lam1 = (n_data1 / var1) / denominator
        lam2 = (n_data2 / var2) / denominator
        # estimate the policy value with the other fold
        iw1 = (
            pi_e_ev[idx1_ev, actions_ev[idx1_ev]]
            / pi_b_ev[idx1_ev, actions_ev[idx1_ev]]
        )
        iw2 = (
            pi_e_ev[~idx1_ev, actions_ev[~idx1_ev]]
            / pi_b_ev[~idx1_ev, actions_ev[~idx1_ev]]
        )
        v1 = np.average(estimated_q_func_ev[idx1_ev], weights=pi_e_ev[idx1_ev], axis=1)
        v2 = np.average(
            estimated_q_func_ev[~idx1_ev], weights=pi_e_ev[~idx1_ev], axis=1
        )
        shifted_rewards1 = (
            rewards_ev[idx1_ev] - estimated_q_func_ev[idx1_ev, actions_ev[idx1_ev]]
        )
        shifted_rewards2 = (
            rewards_ev[~idx1_ev] - estimated_q_func_ev[~idx1_ev, actions_ev[~idx1_ev]]
        )
        estimated_rewards = lam1 * (iw1 * shifted_rewards1 + v1).mean()
        estimated_rewards += lam2 * (iw2 * shifted_rewards2 + v2).mean()
        estimated_rewards_list.append(estimated_rewards)
    return np.mean(estimated_rewards_list)


def calc_dr(
    rewards: np.ndarray,
    actions: np.ndarray,
    estimated_q_func: np.ndarray,
    pi_b: np.ndarray,
    pi_e: np.ndarray,
) -> float:
    n_data = actions.shape[0]
    v = np.average(estimated_q_func, weights=pi_e, axis=1)
    iw = pi_e[np.arange(n_data), actions] / pi_b[np.arange(n_data), actions]
    shifted_rewards = rewards - estimated_q_func[np.arange(n_data), actions]
    return (iw * shifted_rewards + v).mean()


def estimate_q_func(
    bandit_feedback,
    pi_e: np.ndarray,
    fitting_method: str = "naive",
    k_fold: int = 2,
) -> np.ndarray:
    # hyperparam
    with open("./conf/q_func_hyperparams.yaml", "rb") as f:
        q_func_hyperparams = yaml.safe_load(f)

    X = bandit_feedback["X_ev"]
    y = bandit_feedback["rewards"]
    pi_b_star = bandit_feedback["pi_b_star"]
    idx1 = bandit_feedback["idx1"].astype(int)
    a = pd.get_dummies(bandit_feedback["actions"]).values
    skf = StratifiedKFold(n_splits=k_fold)
    skf.get_n_splits(X, y)
    estimated_q_func = np.zeros((bandit_feedback["n_eval"], bandit_feedback["n_class"]))
    for train_idx, test_idx in skf.split(X, y):
        X_tr, X_ev = X[train_idx], X[test_idx]
        y_tr, a_tr = y[train_idx], a[train_idx].astype(float)
        pi_e_tr = pi_e[train_idx]
        pi_b_star_tr = pi_b_star[train_idx]
        idx1_tr = idx1[train_idx]
        ops.reset_default_graph()
        clf = QFuncEstimator(
            num_features=X_tr.shape[1],
            num_classes=bandit_feedback["n_class"],
            fitting_method=fitting_method,
            eta=q_func_hyperparams["eta"],
            std=q_func_hyperparams["std"],
            lam=q_func_hyperparams["lam"],
            batch_size=q_func_hyperparams["batch_size"],
            epochs=q_func_hyperparams["epochs"],
        )
        clf.train(
            X=X_tr,
            a=a_tr,
            y=y_tr,
            pi_e=pi_e_tr,
            pi_b_star=pi_b_star_tr,
            idx1=idx1_tr,
        )
        for a_idx in np.arange(bandit_feedback["n_class"]):
            estimated_q_func_for_a = clf.predict(X=X_ev, a_idx=a_idx)[:, a_idx]
            estimated_q_func[test_idx, a_idx] = estimated_q_func_for_a
        clf.s.close()
    return estimated_q_func


@dataclass
class QFuncEstimator:
    num_features: int
    num_classes: int
    eta: float = 0.01
    std: float = 0.01
    lam: float = 0.001
    batch_size: int = 256
    epochs: int = 200
    fitting_method: str = "stratified"

    def __post_init__(self) -> None:
        """Initialize Class."""
        tf.set_random_seed(0)
        self.s = tf.Session()
        self.create_placeholders()
        self.build_graph()
        self.create_losses()
        self.add_optimizer()

    def create_placeholders(self) -> None:
        """Create the placeholders to be used."""
        self.input_X = tf.placeholder(
            "float32", shape=(None, self.num_features), name="input_X"
        )
        self.input_A = tf.placeholder(
            "float32", shape=(None, self.num_classes), name="input_A"
        )
        self.input_R = tf.placeholder("float32", shape=(None,), name="input_R")
        self.input_pi_e = tf.placeholder(
            "float32", shape=(None, self.num_classes), name="input_pi_e"
        )
        self.input_pi_b_star = tf.placeholder(
            "float32", shape=(None, self.num_classes), name="input_pi_b_star"
        )
        self.input_idx1 = tf.placeholder("float32", shape=(None,), name="input_idx1")

    def build_graph(self) -> None:
        """Build the main tensorflow graph with embedding layers."""
        self.weights = tf.Variable(
            tf.random_normal(
                [self.num_features + self.num_classes, self.num_classes],
                stddev=self.std,
            )
        )
        self.bias = tf.Variable(tf.random_normal([self.num_classes], stddev=self.std))

        with tf.variable_scope("prediction"):
            input_X = tf.concat([self.input_X, self.input_A], axis=1)
            self.preds = tf.sigmoid(tf.matmul(input_X, self.weights) + self.bias)

    def create_losses(self) -> None:
        """Create the losses."""
        with tf.name_scope("loss"):
            shifted_rewards = self.input_R - tf.reduce_sum(
                self.preds * self.input_A, axis=1
            )
            if self.fitting_method == "normal":
                self.loss = tf.reduce_mean(tf.square(shifted_rewards))
            else:
                ratio1 = tf.reduce_mean(self.input_idx1)
                input_idx2 = tf.ones_like(self.input_idx1) - self.input_idx1
                ratio2 = tf.reduce_mean(input_idx2)
                pi_e = tf.reduce_sum(self.input_pi_e * self.input_A, 1)
                pi_b_star = tf.reduce_sum(self.input_pi_b_star * self.input_A, 1)
                v = tf.reduce_sum(self.input_pi_e * self.preds, 1)
                phi = (pi_e / pi_b_star) * shifted_rewards + v
                phi1 = self.input_idx1 * phi
                phi2 = input_idx2 * phi
                if self.fitting_method == "stratified":
                    self.loss = ratio1 * tf.reduce_mean(tf.square(phi1))
                    self.loss += ratio2 * tf.reduce_mean(tf.square(phi2))
                    self.loss -= ratio1 * tf.square(tf.reduce_mean(phi1))
                    self.loss -= ratio2 * tf.square(tf.reduce_mean(phi2))
                elif self.fitting_method == "naive":
                    self.loss = tf.reduce_mean(tf.square(phi))
                    self.loss -= tf.square(tf.reduce_mean(phi))

            self.var_list = [self.weights, self.bias]
            l2_reg = [tf.nn.l2_loss(v) for v in self.var_list]
            self.loss += self.lam * tf.add_n(l2_reg)

    def add_optimizer(self) -> None:
        """Add the required optimizer to the graph."""
        with tf.name_scope("optimizer"):
            self.apply_grads = tf.train.MomentumOptimizer(
                learning_rate=self.eta, momentum=0.8
            ).minimize(self.loss, var_list=self.var_list)

    def train(
        self,
        X: np.ndarray,
        a: np.ndarray,
        y: np.ndarray,
        pi_e: np.ndarray,
        pi_b_star: np.ndarray,
        idx1: np.ndarray,
    ) -> None:
        self.s.run(tf.global_variables_initializer())
        for _ in np.arange(self.epochs):
            arr = np.arange(X.shape[0])
            np.random.shuffle(arr)
            for idx in np.arange(0, X.shape[0], self.batch_size):
                arr_ = arr[idx : idx + self.batch_size]
                self.s.run(
                    self.apply_grads,
                    feed_dict={
                        self.input_X: X[arr_],
                        self.input_A: a[arr_],
                        self.input_R: y[arr_],
                        self.input_pi_e: pi_e[arr_],
                        self.input_pi_b_star: pi_b_star[arr_],
                        self.input_idx1: idx1[arr_],
                    },
                )

    def predict(self, X: np.ndarray, a_idx: int):
        a_ = np.zeros((X.shape[0], self.num_classes))
        a_[:, a_idx] = 1
        return self.s.run(self.preds, feed_dict={self.input_X: X, self.input_A: a_})


def estimate_pi_b(bandit_feedback, k_fold: int = 2) -> None:
    X = bandit_feedback["X_ev"]
    idx1 = bandit_feedback["idx1"]
    a = bandit_feedback["actions"]
    skf = StratifiedKFold(n_splits=k_fold, shuffle=True)
    skf.get_n_splits(X, a)
    estimated_pi_b1 = np.zeros((bandit_feedback["n_eval"], bandit_feedback["n_class"]))
    estimated_pi_b2 = np.zeros((bandit_feedback["n_eval"], bandit_feedback["n_class"]))
    estimated_pi_b_star = np.zeros(
        (bandit_feedback["n_eval"], bandit_feedback["n_class"])
    )
    for train_idx, test_idx in skf.split(X, a):
        X_tr, X_ev = X[train_idx], X[test_idx]
        idx1_tr, a_tr = idx1[train_idx], a[train_idx]
        clf = LogisticRegression(random_state=12345)
        clf.fit(X=X_tr[idx1_tr], y=a_tr[idx1_tr])
        estimated_pi_b1[test_idx, :] = clf.predict_proba(X_ev)
        clf = LogisticRegression(random_state=12345)
        clf.fit(X=X_tr[~idx1_tr], y=a_tr[~idx1_tr])
        estimated_pi_b2[test_idx, :] = clf.predict_proba(X_ev)
        clf = LogisticRegression(random_state=12345)
        clf.fit(X=X_tr, y=a_tr)
        estimated_pi_b_star[test_idx, :] = clf.predict_proba(X_ev)
    idx1 = np.expand_dims(idx1.astype(int), 1)
    bandit_feedback["pi_b"] = np.clip(
        idx1 * estimated_pi_b1 + (1 - idx1) * estimated_pi_b2, 1e-6, 1.0
    )
    bandit_feedback["pi_b_star"] = np.clip(estimated_pi_b_star, 1e-6, 1.0)
    return bandit_feedback
```

<!-- #region id="18Yu3VRbVvU1" -->
## Run
<!-- #endregion -->

```python id="Tc7u7QVsVvS-" executionInfo={"status": "ok", "timestamp": 1633519597856, "user_tz": -330, "elapsed": 504, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
def calc_rel_rmse(policy_value_true: float, policy_value_estimated: float) -> float:
    return np.sqrt(
        (((policy_value_true - policy_value_estimated) / policy_value_true) ** 2).mean()
    )
```

```python colab={"base_uri": "https://localhost:8080/"} id="kRdVkp4XWu2q" executionInfo={"status": "ok", "timestamp": 1633519776392, "user_tz": -330, "elapsed": 1038, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="2aa671b1-c82d-42ec-e164-4cd58f7a6022"
parser = argparse.ArgumentParser()
parser.add_argument("--num_sims", "-n", type=int, default=200)
parser.add_argument("--data", "-d", type=str, default='optdigits') # data in ['optdigits','pendigits','sat','letter']
parser.add_argument("--test_size", "-t", type=float, default=0.7)
parser.add_argument("--is_estimate_pi_b", "-i", default=True, action="store_true")
args = parser.parse_args(args={})
print(args)
```

```python colab={"base_uri": "https://localhost:8080/"} id="3fWlEeC0VvQH" executionInfo={"status": "ok", "timestamp": 1633539924046, "user_tz": -330, "elapsed": 19709985, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="642c85e2-7ef1-4552-d5d3-41db1b5f8c47"
# configurations
num_sims = args.num_sims
data = args.data
test_size = args.test_size
is_estimate_pi_b = args.is_estimate_pi_b
np.random.seed(12345)
ratio_list = [0.1, 0.2, 0.5, 1, 2, 4, 10]
estimator_names = [
    "ground_truth",
    "IS-Avg",
    "IS",
    "IS-PW(f)",
    "DR-Avg",
    "DR-PW",
    "DR",
    "MRDR",
    "SMRDR",
]
log_path = (
    Path("log") / data / f"test_size={test_size}" / "estimated_pi_b"
    if is_estimate_pi_b
    else Path("log") / data / f"test_size={test_size}" / "true_pi_b"
)
log_path.mkdir(parents=True, exist_ok=True)
raw_results_path = log_path / "raw_results"
raw_results_path.mkdir(parents=True, exist_ok=True)

rel_rmse_results = {
    name: {r: np.zeros(num_sims) for r in ratio_list} for name in estimator_names
}
for ratio in ratio_list:
    start = time.time()
    ope_results = {name: np.zeros(num_sims) for name in estimator_names}
    for sim_id in np.arange(num_sims):
        # load and split data
        data_dict = load_datasets(
            data=data, test_size=test_size, ratio=ratio, random_state=sim_id
        )
        # train eval and two behavior policies
        pi_e, pi_b1, pi_b2 = train_policies(
            data_dict=data_dict,
            random_state=sim_id,
        )
        # generate bandit feedback
        bandit_feedback_ = generate_bandit_feedback(
            data_dict=data_dict, pi_b1=pi_b1, pi_b2=pi_b2
        )
        # estimate pi_b1, pi_b2, and pi_b_star with 2-fold cross-fitting
        if is_estimate_pi_b:
            bandit_feedback = estimate_pi_b(bandit_feedback=bandit_feedback_)
        else:
            bandit_feedback = bandit_feedback_
        # estimate q-function with 2-fold cross-fitting
        estimated_q_func = estimate_q_func(
            bandit_feedback=bandit_feedback,
            pi_e=pi_e,
            fitting_method="normal",
        )
        estimated_q_func_with_mrdr_wrong = estimate_q_func(
            bandit_feedback=bandit_feedback,
            pi_e=pi_e,
            fitting_method="naive",
        )
        estimated_q_func_with_mrdr = estimate_q_func(
            bandit_feedback=bandit_feedback,
            pi_e=pi_e,
            fitting_method="stratified",
        )
        # off-policy evaluation
        ope_results["ground_truth"][sim_id] = calc_ground_truth(
            y_true=data_dict["y_ev"], pi=pi_e
        )
        ope_results["IS-Avg"][sim_id] = calc_ipw(
            rewards=bandit_feedback["rewards"],
            actions=bandit_feedback["actions"],
            pi_b=bandit_feedback["pi_b"],
            pi_e=pi_e,
        )
        ope_results["IS"][sim_id] = calc_ipw(
            rewards=bandit_feedback["rewards"],
            actions=bandit_feedback["actions"],
            pi_b=bandit_feedback["pi_b_star"],
            pi_e=pi_e,
        )
        ope_results["IS-PW(f)"][sim_id] = calc_weighted(
            rewards=bandit_feedback["rewards"],
            actions=bandit_feedback["actions"],
            idx1=bandit_feedback["idx1"],
            pi_b=bandit_feedback["pi_b"],
            pi_e=pi_e,
        )
        ope_results["DR-Avg"][sim_id] = calc_dr(
            rewards=bandit_feedback["rewards"],
            actions=bandit_feedback["actions"],
            estimated_q_func=estimated_q_func,
            pi_b=bandit_feedback["pi_b"],
            pi_e=pi_e,
        )
        ope_results["DR-PW"][sim_id] = calc_weighted(
            rewards=bandit_feedback["rewards"],
            actions=bandit_feedback["actions"],
            idx1=bandit_feedback["idx1"],
            pi_b=bandit_feedback["pi_b"],
            pi_e=pi_e,
            estimated_q_func=estimated_q_func,
        )
        ope_results["DR"][sim_id] = calc_dr(
            rewards=bandit_feedback["rewards"],
            actions=bandit_feedback["actions"],
            estimated_q_func=estimated_q_func,
            pi_b=bandit_feedback["pi_b_star"],
            pi_e=pi_e,
        )
        ope_results["MRDR"][sim_id] = calc_dr(
            rewards=bandit_feedback["rewards"],
            actions=bandit_feedback["actions"],
            estimated_q_func=estimated_q_func_with_mrdr_wrong,
            pi_b=bandit_feedback["pi_b_star"],
            pi_e=pi_e,
        )
        ope_results["SMRDR"][sim_id] = calc_dr(
            rewards=bandit_feedback["rewards"],
            actions=bandit_feedback["actions"],
            estimated_q_func=estimated_q_func_with_mrdr,
            pi_b=bandit_feedback["pi_b_star"],
            pi_e=pi_e,
        )
        if ((sim_id + 1) % 20) == 0:
            print(
                f"ratio={ratio}-{sim_id+1}th: {np.round((time.time() - start) / 60, 2)}min"
            )
    # save raw off-policy evaluation results.
    with open(raw_results_path / f"ratio={ratio}.pkl", mode="wb") as f:
        pickle.dump(ope_results, f)
    for estimator in estimator_names:
        rel_rmse_results[estimator][ratio] = calc_rel_rmse(
            policy_value_true=ope_results["ground_truth"],
            policy_value_estimated=ope_results[estimator],
        )
    print(f"finish ratio={ratio}: {np.round((time.time() - start) / 60, 2)}min")
    print("=" * 50)

# save results of the evaluation of OPE
rel_rmse_results_df = pd.DataFrame(rel_rmse_results).drop("ground_truth", 1)
rel_rmse_results_df.T.round(5).to_csv(log_path / f"rel_rmse.csv")
```

```python colab={"base_uri": "https://localhost:8080/", "height": 297} id="0acTR2zfka6Q" executionInfo={"status": "ok", "timestamp": 1633540012626, "user_tz": -330, "elapsed": 622, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="2caac3ef-1581-437b-8274-958be9889a68"
pd.read_csv('./log/optdigits/test_size=0.7/estimated_pi_b/rel_rmse.csv', index_col=0)
```

```python id="maBZT-D8kfe7"
!apt-get install tree
```

```python colab={"base_uri": "https://localhost:8080/"} id="54sMX48ekpRa" executionInfo={"status": "ok", "timestamp": 1633540067992, "user_tz": -330, "elapsed": 590, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="72288b9b-e2ec-4702-acba-114536cf5c92"
!tree --du -h ./log
```
