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

<!-- #region id="wL8SFvvcMI16" -->
# Evaluation of Multiple Off-Policy Estimators on Synthetic Dataset
<!-- #endregion -->

<!-- #region id="l6Z4nt5WmM__" -->
We evaluate the estimation performances of the following off-policy estimators using the ground-truth policy value of an evaluation policy calculable with synthetic data.

- Direct Method (DM)
- Inverse Probability Weighting (IPW)
- Self-Normalized Inverse Probability Weighting (SNIPW)
- Doubly Robust (DR)
- Self-Normalized Doubly Robust (SNDR)
- Switch Doubly Robust (Switch-DR)
- Doubly Robust with Optimistic Shrinkage (DRos)
<!-- #endregion -->

```python id="JMfMNBMDoqB4"
# !git clone https://github.com/st-tech/zr-obp.git
```

<!-- #region id="4bW6pkA9w0J3" -->
## Imports
<!-- #endregion -->

```python id="z5gPZJY4wZqx"
from abc import ABCMeta
from abc import abstractmethod
from dataclasses import dataclass
from typing import Dict
from typing import Optional

import numpy as np
from sklearn.utils import check_scalar

from typing import Dict
from typing import Optional
from typing import Union

import numpy as np
import pandas as pd
from sklearn.utils import check_random_state
from sklearn.utils import check_scalar
import torch

from typing import Optional

from abc import ABCMeta
from abc import abstractmethod

import numpy as np
from sklearn.utils import check_scalar

import argparse
from pathlib import Path

from joblib import delayed
from joblib import Parallel
import numpy as np
from pandas import DataFrame
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import yaml

from dataclasses import dataclass
from typing import Callable
from typing import Optional

import enum

import numpy as np
from scipy.stats import truncnorm
from sklearn.utils import check_random_state
from sklearn.utils import check_scalar

from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
from scipy.special import softmax
from sklearn.base import ClassifierMixin
from sklearn.base import clone
from sklearn.base import is_classifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils import check_random_state
from sklearn.utils import check_scalar
import torch
import torch.nn as nn
from torch.nn.functional import mse_loss
import torch.optim as optim
from tqdm import tqdm

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import clone
from sklearn.base import is_classifier
from sklearn.model_selection import KFold
from sklearn.utils import check_random_state
from sklearn.utils import check_scalar

from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame
import seaborn as sns
from sklearn.utils import check_scalar
```

<!-- #region id="8Ud34TJrwJ2Z" -->
## Utils
<!-- #endregion -->

```python cellView="form" id="29trTk5Jwr36"
#@markdown main_utils
def check_confidence_interval_arguments(
    alpha: float = 0.05,
    n_bootstrap_samples: int = 10000,
    random_state: Optional[int] = None,
) -> Optional[ValueError]:
    """Check confidence interval arguments.
    Parameters
    ----------
    alpha: float, default=0.05
        Significance level.
    n_bootstrap_samples: int, default=10000
        Number of resampling performed in the bootstrap procedure.
    random_state: int, default=None
        Controls the random seed in bootstrap sampling.
    Returns
    ----------
    estimated_confidence_interval: Dict[str, float]
        Dictionary storing the estimated mean and upper-lower confidence bounds.
    """
    check_random_state(random_state)
    check_scalar(alpha, "alpha", float, min_val=0.0, max_val=1.0)
    check_scalar(n_bootstrap_samples, "n_bootstrap_samples", int, min_val=1)


def estimate_confidence_interval_by_bootstrap(
    samples: np.ndarray,
    alpha: float = 0.05,
    n_bootstrap_samples: int = 10000,
    random_state: Optional[int] = None,
) -> Dict[str, float]:
    """Estimate confidence interval by nonparametric bootstrap-like procedure.
    Parameters
    ----------
    samples: array-like
        Empirical observed samples to be used to estimate cumulative distribution function.
    alpha: float, default=0.05
        Significance level.
    n_bootstrap_samples: int, default=10000
        Number of resampling performed in the bootstrap procedure.
    random_state: int, default=None
        Controls the random seed in bootstrap sampling.
    Returns
    ----------
    estimated_confidence_interval: Dict[str, float]
        Dictionary storing the estimated mean and upper-lower confidence bounds.
    """
    check_confidence_interval_arguments(
        alpha=alpha, n_bootstrap_samples=n_bootstrap_samples, random_state=random_state
    )

    boot_samples = list()
    random_ = check_random_state(random_state)
    for _ in np.arange(n_bootstrap_samples):
        boot_samples.append(np.mean(random_.choice(samples, size=samples.shape[0])))
    lower_bound = np.percentile(boot_samples, 100 * (alpha / 2))
    upper_bound = np.percentile(boot_samples, 100 * (1.0 - alpha / 2))
    return {
        "mean": np.mean(boot_samples),
        f"{100 * (1. - alpha)}% CI (lower)": lower_bound,
        f"{100 * (1. - alpha)}% CI (upper)": upper_bound,
    }


def sample_action_fast(
    action_dist: np.ndarray, random_state: Optional[int] = None
) -> np.ndarray:
    """Sample actions faster based on a given action distribution.
    Parameters
    ----------
    action_dist: array-like, shape (n_rounds, n_actions)
        Distribution over actions.
    random_state: Optional[int], default=None
        Controls the random seed in sampling synthetic bandit dataset.
    Returns
    ---------
    sampled_action: array-like, shape (n_rounds,)
        Actions sampled based on `action_dist`.
    """
    random_ = check_random_state(random_state)
    uniform_rvs = random_.uniform(size=action_dist.shape[0])[:, np.newaxis]
    cum_action_dist = action_dist.cumsum(axis=1)
    flg = cum_action_dist > uniform_rvs
    sampled_action = flg.argmax(axis=1)
    return sampled_action


def convert_to_action_dist(
    n_actions: int,
    selected_actions: np.ndarray,
) -> np.ndarray:
    """Convert selected actions (output of `run_bandit_simulation`) to distribution over actions.
    Parameters
    ----------
    n_actions: int
        Number of actions.
    selected_actions: array-like, shape (n_rounds, len_list)
            Sequence of actions selected by evaluation policy
            at each round in offline bandit simulation.
    Returns
    ----------
    action_dist: array-like, shape (n_rounds, n_actions, len_list)
        Action choice probabilities (can be deterministic).
    """
    n_rounds, len_list = selected_actions.shape
    action_dist = np.zeros((n_rounds, n_actions, len_list))
    for pos in np.arange(len_list):
        selected_actions_ = selected_actions[:, pos]
        action_dist[
            np.arange(n_rounds),
            selected_actions_,
            pos * np.ones(n_rounds, int),
        ] = 1
    return action_dist


def check_array(
    array: np.ndarray,
    name: str,
    expected_dim: int = 1,
) -> ValueError:
    """Input validation on an array.
    Parameters
    -------------
    array: object
        Input object to check.
    name: str
        Name of the input array.
    expected_dim: int, default=1
        Expected dimension of the input array.
    """
    if not isinstance(array, np.ndarray):
        raise ValueError(f"{name} must be {expected_dim}D array, but got {type(array)}")
    if array.ndim != expected_dim:
        raise ValueError(
            f"{name} must be {expected_dim}D array, but got {array.ndim}D array"
        )


def check_tensor(
    tensor: torch.tensor,
    name: str,
    expected_dim: int = 1,
) -> ValueError:
    """Input validation on a tensor.
    Parameters
    -------------
    array: object
        Input object to check.
    name: str
        Name of the input array.
    expected_dim: int, default=1
        Expected dimension of the input array.
    """
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(
            f"{name} must be {expected_dim}D tensor, but got {type(tensor)}"
        )
    if tensor.ndim != expected_dim:
        raise ValueError(
            f"{name} must be {expected_dim}D tensor, but got {tensor.ndim}D tensor"
        )


def check_bandit_feedback_inputs(
    context: np.ndarray,
    action: np.ndarray,
    reward: np.ndarray,
    expected_reward: Optional[np.ndarray] = None,
    position: Optional[np.ndarray] = None,
    pscore: Optional[np.ndarray] = None,
    action_context: Optional[np.ndarray] = None,
) -> Optional[ValueError]:
    """Check inputs for bandit learning or simulation.
    Parameters
    -----------
    context: array-like, shape (n_rounds, dim_context)
        Context vectors in each round, i.e., :math:`x_t`.
    action: array-like, shape (n_rounds,)
        Action sampled by behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.
    reward: array-like, shape (n_rounds,)
        Observed rewards (or outcome) in each round, i.e., :math:`r_t`.
    expected_reward: array-like, shape (n_rounds, n_actions), default=None
        Expected rewards (or outcome) in each round, i.e., :math:`\\mathbb{E}[r_t]`.
    position: array-like, shape (n_rounds,), default=None
        Position of recommendation interface where action was presented in each round of the given logged bandit data.
    pscore: array-like, shape (n_rounds,), default=None
        Propensity scores, the probability of selecting each action by behavior policy,
        in the given logged bandit data.
    action_context: array-like, shape (n_actions, dim_action_context)
        Context vectors characterizing each action.
    """
    check_array(array=context, name="context", expected_dim=2)
    check_array(array=action, name="action", expected_dim=1)
    check_array(array=reward, name="reward", expected_dim=1)
    if not (np.issubdtype(action.dtype, np.integer) and action.min() >= 0):
        raise ValueError("action elements must be non-negative integers")

    if expected_reward is not None:
        check_array(array=expected_reward, name="expected_reward", expected_dim=2)
        if not (
            context.shape[0]
            == action.shape[0]
            == reward.shape[0]
            == expected_reward.shape[0]
        ):
            raise ValueError(
                "Expected `context.shape[0] == action.shape[0] == reward.shape[0] == expected_reward.shape[0]`"
                ", but found it False"
            )
        if action.max() >= expected_reward.shape[1]:
            raise ValueError(
                "action elements must be smaller than `expected_reward.shape[1]`"
            )
    if pscore is not None:
        check_array(array=pscore, name="pscore", expected_dim=1)
        if not (
            context.shape[0] == action.shape[0] == reward.shape[0] == pscore.shape[0]
        ):
            raise ValueError(
                "Expected `context.shape[0] == action.shape[0] == reward.shape[0] == pscore.shape[0]`"
                ", but found it False"
            )
        if np.any(pscore <= 0):
            raise ValueError("pscore must be positive")

    if position is not None:
        check_array(array=position, name="position", expected_dim=1)
        if not (
            context.shape[0] == action.shape[0] == reward.shape[0] == position.shape[0]
        ):
            raise ValueError(
                "Expected `context.shape[0] == action.shape[0] == reward.shape[0] == position.shape[0]`"
                ", but found it False"
            )
        if not (np.issubdtype(position.dtype, np.integer) and position.min() >= 0):
            raise ValueError("position elements must be non-negative integers")
    else:
        if not (context.shape[0] == action.shape[0] == reward.shape[0]):
            raise ValueError(
                "Expected `context.shape[0] == action.shape[0] == reward.shape[0]`"
                ", but found it False"
            )
    if action_context is not None:
        check_array(array=action_context, name="action_context", expected_dim=2)
        if action.max() >= action_context.shape[0]:
            raise ValueError(
                "action elements must be smaller than `action_context.shape[0]`"
            )


def check_ope_inputs(
    action_dist: np.ndarray,
    position: Optional[np.ndarray] = None,
    action: Optional[np.ndarray] = None,
    reward: Optional[np.ndarray] = None,
    pscore: Optional[np.ndarray] = None,
    estimated_rewards_by_reg_model: Optional[np.ndarray] = None,
) -> Optional[ValueError]:
    """Check inputs for ope.
    Parameters
    -----------
    action_dist: array-like, shape (n_rounds, n_actions, len_list)
        Action choice probabilities of evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.
    position: array-like, shape (n_rounds,), default=None
        Position of recommendation interface where action was presented in each round of the given logged bandit data.
    action: array-like, shape (n_rounds,), default=None
        Action sampled by behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.
    reward: array-like, shape (n_rounds,), default=None
        Observed rewards (or outcome) in each round, i.e., :math:`r_t`.
    pscore: array-like, shape (n_rounds,), default=None
        Propensity scores, the probability of selecting each action by behavior policy,
        in the given logged bandit data.
    estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list), default=None
        Expected rewards given context, action, and position estimated by regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.
    """
    # action_dist
    check_array(array=action_dist, name="action_dist", expected_dim=3)
    if not np.allclose(action_dist.sum(axis=1), 1):
        raise ValueError("action_dist must be a probability distribution")

    # position
    if position is not None:
        check_array(array=position, name="position", expected_dim=1)
        if not (position.shape[0] == action_dist.shape[0]):
            raise ValueError(
                "Expected `position.shape[0] == action_dist.shape[0]`, but found it False"
            )
        if not (np.issubdtype(position.dtype, np.integer) and position.min() >= 0):
            raise ValueError("position elements must be non-negative integers")
        if position.max() >= action_dist.shape[2]:
            raise ValueError(
                "position elements must be smaller than `action_dist.shape[2]`"
            )
    elif action_dist.shape[2] > 1:
        raise ValueError(
            "position elements must be given when `action_dist.shape[2] > 1`"
        )

    # estimated_rewards_by_reg_model
    if estimated_rewards_by_reg_model is not None:
        if estimated_rewards_by_reg_model.shape != action_dist.shape:
            raise ValueError(
                "Expected `estimated_rewards_by_reg_model.shape == action_dist.shape`, but found it False"
            )

    # action, reward
    if action is not None or reward is not None:
        check_array(array=action, name="action", expected_dim=1)
        check_array(array=reward, name="reward", expected_dim=1)
        if not (action.shape[0] == reward.shape[0]):
            raise ValueError(
                "Expected `action.shape[0] == reward.shape[0]`, but found it False"
            )
        if not (np.issubdtype(action.dtype, np.integer) and action.min() >= 0):
            raise ValueError("action elements must be non-negative integers")
        if action.max() >= action_dist.shape[1]:
            raise ValueError(
                "action elements must be smaller than `action_dist.shape[1]`"
            )

    # pscore
    if pscore is not None:
        if pscore.ndim != 1:
            raise ValueError("pscore must be 1-dimensional")
        if not (action.shape[0] == reward.shape[0] == pscore.shape[0]):
            raise ValueError(
                "Expected `action.shape[0] == reward.shape[0] == pscore.shape[0]`, but found it False"
            )
        if np.any(pscore <= 0):
            raise ValueError("pscore must be positive")


def check_continuous_bandit_feedback_inputs(
    context: np.ndarray,
    action_by_behavior_policy: np.ndarray,
    reward: np.ndarray,
    expected_reward: Optional[np.ndarray] = None,
    pscore: Optional[np.ndarray] = None,
) -> Optional[ValueError]:
    """Check inputs for bandit learning or simulation with continuous actions.
    Parameters
    -----------
    context: array-like, shape (n_rounds, dim_context)
        Context vectors in each round, i.e., :math:`x_t`.
    action_by_behavior_policy: array-like, shape (n_rounds,)
        Continuous action values sampled by a behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.
    reward: array-like, shape (n_rounds,)
        Observed rewards (or outcome) in each round, i.e., :math:`r_t`.
    expected_reward: array-like, shape (n_rounds, n_actions), default=None
        Expected rewards (or outcome) in each round, i.e., :math:`\\mathbb{E}[r_t]`.
    pscore: array-like, shape (n_rounds,), default=None
        Probability densities of the continuous action values sampled by a behavior policy
        (generalized propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.
    """
    check_array(array=context, name="context", expected_dim=2)
    check_array(
        array=action_by_behavior_policy,
        name="action_by_behavior_policy",
        expected_dim=1,
    )
    check_array(array=reward, name="reward", expected_dim=1)

    if expected_reward is not None:
        check_array(array=expected_reward, name="expected_reward", expected_dim=1)
        if not (
            context.shape[0]
            == action_by_behavior_policy.shape[0]
            == reward.shape[0]
            == expected_reward.shape[0]
        ):
            raise ValueError(
                "Expected `context.shape[0] == action_by_behavior_policy.shape[0]"
                "== reward.shape[0] == expected_reward.shape[0]`, but found it False"
            )
    if pscore is not None:
        check_array(array=pscore, name="pscore", expected_dim=1)
        if not (
            context.shape[0]
            == action_by_behavior_policy.shape[0]
            == reward.shape[0]
            == pscore.shape[0]
        ):
            raise ValueError(
                "Expected `context.shape[0] == action_by_behavior_policy.shape[0]"
                "== reward.shape[0] == pscore.shape[0]`, but found it False"
            )
        if np.any(pscore <= 0):
            raise ValueError("pscore must be positive")


def check_continuous_ope_inputs(
    action_by_evaluation_policy: np.ndarray,
    action_by_behavior_policy: Optional[np.ndarray] = None,
    reward: Optional[np.ndarray] = None,
    pscore: Optional[np.ndarray] = None,
    estimated_rewards_by_reg_model: Optional[np.ndarray] = None,
) -> Optional[ValueError]:
    """Check inputs for OPE with continuous actions.
    Parameters
    -----------
    action_by_evaluation_policy: array-like, shape (n_rounds,)
        Continuous action values given by the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(x_t)`.
    action_by_behavior_policy: array-like, shape (n_rounds,), default=None
        Continuous action values sampled by a behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.
    reward: array-like, shape (n_rounds,), default=None
        Observed rewards (or outcome) in each round, i.e., :math:`r_t`.
    pscore: array-like, shape (n_rounds,), default=None
        Probability densities of the continuous action values sampled by a behavior policy
        (generalized propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.
    estimated_rewards_by_reg_model: array-like, shape (n_rounds,), default=None
            Expected rewards given context and action estimated by a regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.
    """
    # action_by_evaluation_policy
    check_array(
        array=action_by_evaluation_policy,
        name="action_by_evaluation_policy",
        expected_dim=1,
    )

    # estimated_rewards_by_reg_model
    if estimated_rewards_by_reg_model is not None:
        check_array(
            array=estimated_rewards_by_reg_model,
            name="estimated_rewards_by_reg_model",
            expected_dim=1,
        )
        if (
            estimated_rewards_by_reg_model.shape[0]
            != action_by_evaluation_policy.shape[0]
        ):
            raise ValueError(
                "Expected `estimated_rewards_by_reg_model.shape[0] == action_by_evaluation_policy.shape[0]`"
                ", but found if False"
            )

    # action, reward
    if action_by_behavior_policy is not None or reward is not None:
        check_array(
            array=action_by_behavior_policy,
            name="action_by_behavior_policy",
            expected_dim=1,
        )
        check_array(array=reward, name="reward", expected_dim=1)
        if not (action_by_behavior_policy.shape[0] == reward.shape[0]):
            raise ValueError(
                "Expected `action_by_behavior_policy.shape[0] == reward.shape[0]`"
                ", but found it False"
            )
        if not (
            action_by_behavior_policy.shape[0] == action_by_evaluation_policy.shape[0]
        ):
            raise ValueError(
                "Expected `action_by_behavior_policy.shape[0] == action_by_evaluation_policy.shape[0]`"
                ", but found it False"
            )

    # pscore
    if pscore is not None:
        check_array(array=pscore, name="pscore", expected_dim=1)
        if not (
            action_by_behavior_policy.shape[0] == reward.shape[0] == pscore.shape[0]
        ):
            raise ValueError(
                "Expected `action_by_behavior_policy.shape[0] == reward.shape[0] == pscore.shape[0]`"
                ", but found it False"
            )
        if np.any(pscore <= 0):
            raise ValueError("pscore must be positive")


def _check_slate_ope_inputs(
    slate_id: np.ndarray,
    reward: np.ndarray,
    position: np.ndarray,
    pscore: np.ndarray,
    evaluation_policy_pscore: np.ndarray,
    pscore_type: str,
) -> Optional[ValueError]:
    """Check inputs of Slate OPE estimators.
    Parameters
    -----------
    slate_id: array-like, shape (<= n_rounds * len_list,)
        Slate id observed in each round of the logged bandit feedback.
    reward: array-like, shape (<= n_rounds * len_list,)
        Reward observed at each slot in each round of the logged bandit feedback, i.e., :math:`r_{t}(k)`.
    position: array-like, shape (<= n_rounds * len_list,)
        Positions of each round and slot in the given logged bandit data.
    pscore: array-like, shape (<= n_rounds * len_list,)
        Action choice probabilities of behavior policy (propensity scores).
    evaluation_policy_pscore: array-like, shape (<= n_rounds * len_list,)
        Action choice probabilities of evaluation policy.
    pscore_type: str
        Either "pscore", "pscore_item_position", or "pscore_cascade".
    """
    # position
    check_array(array=position, name="position", expected_dim=1)
    if not (position.dtype == int and position.min() >= 0):
        raise ValueError("position elements must be non-negative integers")

    # reward
    check_array(array=reward, name="reward", expected_dim=1)

    # pscore
    check_array(array=pscore, name=f"{pscore_type}", expected_dim=1)
    if np.any(pscore <= 0) or np.any(pscore > 1):
        raise ValueError(f"{pscore_type} must be in the range of (0, 1]")

    # evaluation_policy_pscore
    check_array(
        array=evaluation_policy_pscore,
        name=f"evaluation_policy_{pscore_type}",
        expected_dim=1,
    )
    if np.any(evaluation_policy_pscore < 0) or np.any(evaluation_policy_pscore > 1):
        raise ValueError(
            f"evaluation_policy_{pscore_type} must be in the range of [0, 1]"
        )

    # slate id
    check_array(array=slate_id, name="slate_id", expected_dim=1)
    if not (slate_id.dtype == int and slate_id.min() >= 0):
        raise ValueError("slate_id elements must be non-negative integers")
    if not (
        slate_id.shape[0]
        == position.shape[0]
        == reward.shape[0]
        == pscore.shape[0]
        == evaluation_policy_pscore.shape[0]
    ):
        raise ValueError(
            f"slate_id, position, reward, {pscore_type}, and evaluation_policy_{pscore_type} "
            "must have the same number of samples."
        )


def check_sips_inputs(
    slate_id: np.ndarray,
    reward: np.ndarray,
    position: np.ndarray,
    pscore: np.ndarray,
    evaluation_policy_pscore: np.ndarray,
) -> Optional[ValueError]:
    """Check inputs of SlateStandardIPS.
    Parameters
    -----------
    slate_id: array-like, shape (<= n_rounds * len_list,)
        Slate id observed in each round of the logged bandit feedback.
    reward: array-like, shape (<= n_rounds * len_list,)
        Reward observed at each slot in each round of the logged bandit feedback, i.e., :math:`r_{t}(k)`.
    position: array-like, shape (<= n_rounds * len_list,)
        Positions of each round and slot in the given logged bandit data.
    pscore: array-like, shape (<= n_rounds * len_list,)
        Action choice probabilities of behavior policy (propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.
    evaluation_policy_pscore: array-like, shape (<= n_rounds * len_list,)
        Action choice probabilities of evaluation policy, i.e., :math:`\\pi_e(a_t|x_t)`.
    """
    _check_slate_ope_inputs(
        slate_id=slate_id,
        reward=reward,
        position=position,
        pscore=pscore,
        evaluation_policy_pscore=evaluation_policy_pscore,
        pscore_type="pscore",
    )

    bandit_feedback_df = pd.DataFrame()
    bandit_feedback_df["slate_id"] = slate_id
    bandit_feedback_df["reward"] = reward
    bandit_feedback_df["position"] = position
    bandit_feedback_df["pscore"] = pscore
    bandit_feedback_df["evaluation_policy_pscore"] = evaluation_policy_pscore
    # check uniqueness
    if bandit_feedback_df.duplicated(["slate_id", "position"]).sum() > 0:
        raise ValueError("position must not be duplicated in each slate")
    # check pscore uniqueness
    distinct_count_pscore_in_slate = bandit_feedback_df.groupby("slate_id").apply(
        lambda x: x["pscore"].unique().shape[0]
    )
    if (distinct_count_pscore_in_slate != 1).sum() > 0:
        raise ValueError("pscore must be unique in each slate")
    # check pscore uniqueness of evaluation policy
    distinct_count_evaluation_policy_pscore_in_slate = bandit_feedback_df.groupby(
        "slate_id"
    ).apply(lambda x: x["evaluation_policy_pscore"].unique().shape[0])
    if (distinct_count_evaluation_policy_pscore_in_slate != 1).sum() > 0:
        raise ValueError("evaluation_policy_pscore must be unique in each slate")


def check_iips_inputs(
    slate_id: np.ndarray,
    reward: np.ndarray,
    position: np.ndarray,
    pscore_item_position: np.ndarray,
    evaluation_policy_pscore_item_position: np.ndarray,
) -> Optional[ValueError]:
    """Check inputs of SlateIndependentIPS.
    Parameters
    -----------
    slate_id: array-like, shape (<= n_rounds * len_list,)
        Slate id observed in each round of the logged bandit feedback.
    reward: array-like, shape (<= n_rounds * len_list,)
        Reward observed at each slot in each round of the logged bandit feedback, i.e., :math:`r_{t}(k)`.
    position: array-like, shape (<= n_rounds * len_list,)
        Positions of each round and slot in the given logged bandit data.
    pscore_item_position: array-like, shape (<= n_rounds * len_list,)
        Marginal action choice probabilities of the slot (:math:`k`) by a behavior policy (propensity scores), i.e., :math:`\\pi_b(a_{t}(k) |x_t)`.
    evaluation_policy_pscore_item_position: array-like, shape (<= n_rounds * len_list,)
        Marginal action choice probabilities of the slot (:math:`k`) by the evaluation policy, i.e., :math:`\\pi_e(a_{t}(k) |x_t)`.
    """
    _check_slate_ope_inputs(
        slate_id=slate_id,
        reward=reward,
        position=position,
        pscore=pscore_item_position,
        evaluation_policy_pscore=evaluation_policy_pscore_item_position,
        pscore_type="pscore_item_position",
    )

    bandit_feedback_df = pd.DataFrame()
    bandit_feedback_df["slate_id"] = slate_id
    bandit_feedback_df["position"] = position
    # check uniqueness
    if bandit_feedback_df.duplicated(["slate_id", "position"]).sum() > 0:
        raise ValueError("position must not be duplicated in each slate")


def check_rips_inputs(
    slate_id: np.ndarray,
    reward: np.ndarray,
    position: np.ndarray,
    pscore_cascade: np.ndarray,
    evaluation_policy_pscore_cascade: np.ndarray,
) -> Optional[ValueError]:
    """Check inputs of SlateRewardInteractionIPS.
    Parameters
    -----------
    slate_id: array-like, shape (<= n_rounds * len_list,)
        Slate id observed in each round of the logged bandit feedback.
    reward: array-like, shape (<= n_rounds * len_list,)
        Reward observed at each slot in each round of the logged bandit feedback, i.e., :math:`r_{t}(k)`.
    position: array-like, shape (<= n_rounds * len_list,)
        Positions of each round and slot in the given logged bandit data.
    pscore_cascade: array-like, shape (<= n_rounds * len_list,)
        Action choice probabilities of behavior policy (propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.
    evaluation_policy_pscore_cascade: array-like, shape (<= n_rounds * len_list,)
        Action choice probabilities above the slot (:math:`k`) by the evaluation policy, i.e., :math:`\\pi_e(\\{a_{t, j}\\}_{j \\le k}|x_t)`.
    """
    _check_slate_ope_inputs(
        slate_id=slate_id,
        reward=reward,
        position=position,
        pscore=pscore_cascade,
        evaluation_policy_pscore=evaluation_policy_pscore_cascade,
        pscore_type="pscore_cascade",
    )

    bandit_feedback_df = pd.DataFrame()
    bandit_feedback_df["slate_id"] = slate_id
    bandit_feedback_df["reward"] = reward
    bandit_feedback_df["position"] = position
    bandit_feedback_df["pscore_cascade"] = pscore_cascade
    bandit_feedback_df[
        "evaluation_policy_pscore_cascade"
    ] = evaluation_policy_pscore_cascade
    # sort dataframe
    bandit_feedback_df = (
        bandit_feedback_df.sort_values(["slate_id", "position"])
        .reset_index(drop=True)
        .copy()
    )
    # check uniqueness
    if bandit_feedback_df.duplicated(["slate_id", "position"]).sum() > 0:
        raise ValueError("position must not be duplicated in each slate")
    # check pscore_cascade structure
    previous_minimum_pscore_cascade = (
        bandit_feedback_df.groupby("slate_id")["pscore_cascade"]
        .expanding()
        .min()
        .values
    )
    if (
        previous_minimum_pscore_cascade < bandit_feedback_df["pscore_cascade"]
    ).sum() > 0:
        raise ValueError("pscore_cascade must be non-increasing sequence in each slate")
    # check pscore_cascade structure of evaluation policy
    previous_minimum_evaluation_policy_pscore_cascade = (
        bandit_feedback_df.groupby("slate_id")["evaluation_policy_pscore_cascade"]
        .expanding()
        .min()
        .values
    )
    if (
        previous_minimum_evaluation_policy_pscore_cascade
        < bandit_feedback_df["evaluation_policy_pscore_cascade"]
    ).sum() > 0:
        raise ValueError(
            "evaluation_policy_pscore_cascade must be non-increasing sequence in each slate"
        )


def sigmoid(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Calculate sigmoid function."""
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Calculate softmax function."""
    b = np.max(x, axis=1)[:, np.newaxis]
    numerator = np.exp(x - b)
    denominator = np.sum(numerator, axis=1)[:, np.newaxis]
    return numerator / denominator
```

```python cellView="form" id="S-1fenfSxCLW"
#@markdown helpers
def estimate_bias_in_ope(
    reward: np.ndarray,
    iw: np.ndarray,
    iw_hat: np.ndarray,
    q_hat: Optional[np.ndarray] = None,
) -> float:
    """Helper to estimate a bias in OPE.
    Parameters
    ----------
    reward: array-like, shape (n_rounds,)
        Reward observed in each round of the logged bandit feedback, i.e., :math:`r_t`.
    iw: array-like, shape (n_rounds,)
        Importance weight in each round of the logged bandit feedback, i.e., :math:`w(x,a)=\\pi_e(a|x)/ \\pi_b(a|x)`.
    iw_hat: array-like, shape (n_rounds,)
        Importance weight (IW) modified by a hyparpareter. How IW is modified depends on the estimator as follows.
            - clipping: :math:`\\hat{w}(x,a) := \\min \\{ \\lambda, w(x,a) \\}`
            - switching: :math:`\\hat{w}(x,a) := w(x,a) \\cdot \\mathbb{I} \\{ w(x,a) < \\lambda \\}`
            - shrinkage: :math:`\\hat{w}(x,a) := (\\lambda w(x,a)) / (\\lambda + w^2(x,a))`
        where :math:`\\lambda` is a hyperparameter value.
    q_hat: array-like, shape (n_rounds,), default=None
        Estimated expected reward given context :math:`x_t` and action :math:`a_t`.
    Returns
    ----------
    estimated_bias: float
        Estimated the bias in OPE.
        This is based on the direct bias estimation stated on page 17 of Su et al.(2020).
    References
    ----------
    Yi Su, Maria Dimakopoulou, Akshay Krishnamurthy, and Miroslav Dudik.
    "Doubly Robust Off-Policy Evaluation with Shrinkage.", 2020.
    """
    n_rounds = reward.shape[0]
    if q_hat is None:
        q_hat = np.zeros(n_rounds)
    estimated_bias_arr = (iw - iw_hat) * (reward - q_hat)
    estimated_bias = np.abs(estimated_bias_arr.mean())

    return estimated_bias


def estimate_high_probability_upper_bound_bias(
    reward: np.ndarray,
    iw: np.ndarray,
    iw_hat: np.ndarray,
    q_hat: Optional[np.ndarray] = None,
    delta: float = 0.05,
) -> float:
    """Helper to estimate a high probability upper bound of bias in OPE.
    Parameters
    ----------
    reward: array-like, shape (n_rounds,)
        Reward observed in each round of the logged bandit feedback, i.e., :math:`r_t`.
    iw: array-like, shape (n_rounds,)
        Importance weight in each round of the logged bandit feedback, i.e., :math:`w(x,a)=\\pi_e(a|x)/ \\pi_b(a|x)`.
    iw_hat: array-like, shape (n_rounds,)
        Importance weight (IW) modified by a hyparpareter. How IW is modified depends on the estimator as follows.
            - clipping: :math:`\\hat{w}(x,a) := \\min \\{ \\lambda, w(x,a) \\}`
            - switching: :math:`\\hat{w}(x,a) := w(x,a) \\cdot \\mathbb{I} \\{ w(x,a) < \\lambda \\}`
            - shrinkage: :math:`\\hat{w}(x,a) := (\\lambda w(x,a)) / (\\lambda + w^2(x,a))`
        where :math:`\\lambda` and :math:`\\lambda` are hyperparameters.
    q_hat: array-like, shape (n_rounds,), default=None
        Estimated expected reward given context :math:`x_t` and action :math:`a_t`.
    delta: float, default=0.05
        A confidence delta to construct a high probability upper bound based on the Bernstein’s inequality.
    Returns
    ----------
    bias_upper_bound: float
        Estimated (high probability) upper bound of the bias.
        This upper bound is based on the direct bias estimation
        stated on page 17 of Su et al.(2020).
    References
    ----------
    Yi Su, Maria Dimakopoulou, Akshay Krishnamurthy, and Miroslav Dudik.
    "Doubly Robust Off-Policy Evaluation with Shrinkage.", 2020.
    """
    check_scalar(delta, "delta", (int, float), min_val=0.0, max_val=1.0)

    bias_upper_bound = estimate_bias_in_ope(
        reward=reward,
        iw=iw,
        iw_hat=iw_hat,
        q_hat=q_hat,
    )
    n_rounds = reward.shape[0]
    bias_upper_bound += np.sqrt((2 * (iw ** 2).mean() * np.log(2 / delta)) / n_rounds)
    bias_upper_bound += (2 * iw.max() * np.log(2 / delta)) / (3 * n_rounds)

    return bias_upper_bound
```

```python cellView="form" id="1uKenT50yI9P"
#@markdown reward type
class RewardType(enum.Enum):
    """Reward type.
    Attributes
    ----------
    BINARY:
        The reward type is binary.
    CONTINUOUS:
        The reward type is continuous.
    """

    BINARY = "binary"
    CONTINUOUS = "continuous"

    def __repr__(self) -> str:

        return str(self)
```

```python cellView="form" id="aZBo5vEuzYBP"
#@markdown policy type
class PolicyType(enum.Enum):
    """Policy type.
    Attributes
    ----------
    CONTEXT_FREE:
        The policy type is contextfree.
    CONTEXTUAL:
        The policy type is contextual.
    OFFLINE:
        The policy type is offline.
    """

    CONTEXT_FREE = enum.auto()
    CONTEXTUAL = enum.auto()
    OFFLINE = enum.auto()

    def __repr__(self) -> str:

        return str(self)
```

<!-- #region id="-Bd6ynsIxcCf" -->
## Dataset
<!-- #endregion -->

```python id="H45pS7eaxdF1"
# from ..types import BanditFeedback
# from ..utils import check_array
# from ..utils import sample_action_fast
# from ..utils import sigmoid
# from ..utils import softmax
# from .base import BaseBanditDataset
# from .reward_type import RewardType
```

```python id="owWf6vwYyCLg"
class BaseBanditDataset(metaclass=ABCMeta):
    """Base Class for Synthetic Bandit Dataset."""

    @abstractmethod
    def obtain_batch_bandit_feedback(self) -> None:
        """Obtain batch logged bandit feedback."""
        raise NotImplementedError


class BaseRealBanditDataset(BaseBanditDataset):
    """Base Class for Real-World Bandit Dataset."""

    @abstractmethod
    def load_raw_data(self) -> None:
        """Load raw dataset."""
        raise NotImplementedError

    @abstractmethod
    def pre_process(self) -> None:
        """Preprocess raw dataset."""
        raise NotImplementedError
```

```python id="dLvROuE_x0iJ"
# dataset type
BanditFeedback = Dict[str, Union[int, np.ndarray]]
```

```python cellView="form" id="hji2HuzuxkyI"
#@markdown synthetic dataset
@dataclass
class SyntheticBanditDataset(BaseBanditDataset):
    """Class for generating synthetic bandit dataset.
    Note
    -----
    By calling the `obtain_batch_bandit_feedback` method several times,
    we have different bandit samples with the same setting.
    This can be used to estimate confidence intervals of the performances of OPE estimators.
    If None is set as `behavior_policy_function`, the synthetic data will be context-free bandit feedback.
    Parameters
    -----------
    n_actions: int
        Number of actions.
    dim_context: int, default=1
        Number of dimensions of context vectors.
    reward_type: str, default='binary'
        Type of reward variable, which must be either 'binary' or 'continuous'.
        When 'binary' is given, rewards are sampled from the Bernoulli distribution.
        When 'continuous' is given, rewards are sampled from the truncated Normal distribution with `scale=1`.
        The mean parameter of the reward distribution is determined by the `reward_function` specified by the next argument.
    reward_function: Callable[[np.ndarray, np.ndarray], np.ndarray]], default=None
        Function generating expected reward for each given action-context pair,
        i.e., :math:`\\mu: \\mathcal{X} \\times \\mathcal{A} \\rightarrow \\mathbb{R}`.
        If None is set, context **independent** expected reward for each action will be
        sampled from the uniform distribution automatically.
    reward_std: float, default=1.0
        Standard deviation of the reward distribution.
        A larger value leads to a noisy reward distribution.
        This argument is valid only when `reward_type="continuous"`.
    behavior_policy_function: Callable[[np.ndarray, np.ndarray], np.ndarray], default=None
        Function generating probability distribution over action space,
        i.e., :math:`\\pi: \\mathcal{X} \\rightarrow \\Delta(\\mathcal{A})`.
        If None is set, context **independent** uniform distribution will be used (uniform random behavior policy).
    tau: float, default=1.0
        A temperature hyperparameer which controls the behavior policy.
        A large value leads to a near-uniform behavior policy,
        while a small value leads to a near-deterministic behavior policy.
    random_state: int, default=12345
        Controls the random seed in sampling synthetic bandit dataset.
    dataset_name: str, default='synthetic_bandit_dataset'
        Name of the dataset.
    Examples
    ----------
    .. code-block:: python
        >>> import numpy as np
        >>> from obp.dataset import (
            SyntheticBanditDataset,
            linear_reward_function,
            linear_behavior_policy
        )
        # generate synthetic contextual bandit feedback with 10 actions.
        >>> dataset = SyntheticBanditDataset(
                n_actions=10,
                dim_context=5,
                reward_function=logistic_reward_function,
                behavior_policy=linear_behavior_policy,
                random_state=12345
            )
        >>> bandit_feedback = dataset.obtain_batch_bandit_feedback(n_rounds=100000)
        >>> bandit_feedback
        {
            'n_rounds': 100000,
            'n_actions': 10,
            'context': array([[-0.20470766,  0.47894334, -0.51943872, -0.5557303 ,  1.96578057],
                    [ 1.39340583,  0.09290788,  0.28174615,  0.76902257,  1.24643474],
                    [ 1.00718936, -1.29622111,  0.27499163,  0.22891288,  1.35291684],
                    ...,
                    [ 1.36946256,  0.58727761, -0.69296769, -0.27519988, -2.10289159],
                    [-0.27428715,  0.52635353,  1.02572168, -0.18486381,  0.72464834],
                    [-1.25579833, -1.42455203, -0.26361242,  0.27928604,  1.21015571]]),
            'action_context': array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]),
            'action': array([7, 4, 0, ..., 7, 9, 6]),
            'position': None,
            'reward': array([0, 1, 1, ..., 0, 1, 0]),
            'expected_reward': array([[0.80210203, 0.73828559, 0.83199558, ..., 0.81190503, 0.70617705,
                    0.68985306],
                    [0.94119582, 0.93473317, 0.91345213, ..., 0.94140688, 0.93152449,
                    0.90132868],
                    [0.87248862, 0.67974991, 0.66965669, ..., 0.79229752, 0.82712978,
                    0.74923536],
                    ...,
                    [0.64856003, 0.38145901, 0.84476094, ..., 0.40962057, 0.77114661,
                    0.65752798],
                    [0.73208527, 0.82012699, 0.78161352, ..., 0.72361416, 0.8652249 ,
                    0.82571751],
                    [0.40348366, 0.24485417, 0.24037926, ..., 0.49613133, 0.30714854,
                    0.5527749 ]]),
            'pscore': array([0.05423855, 0.10339675, 0.09756788, ..., 0.05423855, 0.07250876,
                    0.14065505])
        }
    """

    n_actions: int
    dim_context: int = 1
    reward_type: str = RewardType.BINARY.value
    reward_function: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None
    reward_std: float = 1.0
    behavior_policy_function: Optional[
        Callable[[np.ndarray, np.ndarray], np.ndarray]
    ] = None
    tau: float = 1.0
    random_state: int = 12345
    dataset_name: str = "synthetic_bandit_dataset"

    def __post_init__(self) -> None:
        """Initialize Class."""
        check_scalar(self.n_actions, "n_actions", int, min_val=2)
        check_scalar(self.dim_context, "dim_context", int, min_val=1)
        if RewardType(self.reward_type) not in [
            RewardType.BINARY,
            RewardType.CONTINUOUS,
        ]:
            raise ValueError(
                f"reward_type must be either '{RewardType.BINARY.value}' or '{RewardType.CONTINUOUS.value}', but {self.reward_type} is given.'"
            )
        check_scalar(self.reward_std, "reward_std", (int, float), min_val=0)
        check_scalar(self.tau, "tau", (int, float), min_val=0)
        if self.random_state is None:
            raise ValueError("`random_state` must be given")
        self.random_ = check_random_state(self.random_state)
        if self.reward_function is None:
            self.expected_reward = self.sample_contextfree_expected_reward()
        if self.behavior_policy_function is None:
            self.behavior_policy = np.ones(self.n_actions) / self.n_actions
        if RewardType(self.reward_type) == RewardType.CONTINUOUS:
            self.reward_min = 0
            self.reward_max = 1e10
        # one-hot encoding representations characterizing each action
        self.action_context = np.eye(self.n_actions, dtype=int)

    @property
    def len_list(self) -> int:
        """Length of recommendation lists."""
        return 1

    def sample_contextfree_expected_reward(self) -> np.ndarray:
        """Sample expected reward for each action from the uniform distribution."""
        return self.random_.uniform(size=self.n_actions)

    def calc_expected_reward(self, context: np.ndarray) -> np.ndarray:
        """Sample expected rewards given contexts"""
        # sample reward for each round based on the reward function
        if self.reward_function is None:
            expected_reward_ = np.tile(self.expected_reward, (context.shape[0], 1))
        else:
            expected_reward_ = self.reward_function(
                context=context,
                action_context=self.action_context,
                random_state=self.random_state,
            )

        return expected_reward_

    def sample_reward_given_expected_reward(
        self,
        expected_reward: np.ndarray,
        action: np.ndarray,
    ) -> np.ndarray:
        """Sample reward given expected rewards"""
        expected_reward_factual = expected_reward[np.arange(action.shape[0]), action]
        if RewardType(self.reward_type) == RewardType.BINARY:
            reward = self.random_.binomial(n=1, p=expected_reward_factual)
        elif RewardType(self.reward_type) == RewardType.CONTINUOUS:
            mean = expected_reward_factual
            a = (self.reward_min - mean) / self.reward_std
            b = (self.reward_max - mean) / self.reward_std
            reward = truncnorm.rvs(
                a=a,
                b=b,
                loc=mean,
                scale=self.reward_std,
                random_state=self.random_state,
            )
        else:
            raise NotImplementedError

        return reward

    def sample_reward(self, context: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Sample rewards given contexts and actions, i.e., :math:`r \\sim p(r \\mid x, a)`.
        Parameters
        -----------
        context: array-like, shape (n_rounds, dim_context)
            Context vectors characterizing each round (such as user information).
        action: array-like, shape (n_rounds,)
            Selected actions to the contexts.
        Returns
        ---------
        reward: array-like, shape (n_rounds,)
            Sampled rewards given contexts and actions.
        """
        check_array(array=context, name="context", expected_dim=2)
        check_array(array=action, name="action", expected_dim=1)
        if context.shape[0] != action.shape[0]:
            raise ValueError(
                "Expected `context.shape[0] == action.shape[0]`, but found it False"
            )
        if not np.issubdtype(action.dtype, np.integer):
            raise ValueError("the dtype of action must be a subdtype of int")

        expected_reward_ = self.calc_expected_reward(context)

        return self.sample_reward_given_expected_reward(expected_reward_, action)

    def obtain_batch_bandit_feedback(self, n_rounds: int) -> BanditFeedback:
        """Obtain batch logged bandit feedback.
        Parameters
        ----------
        n_rounds: int
            Number of rounds for synthetic bandit feedback data.
        Returns
        ---------
        bandit_feedback: BanditFeedback
            Generated synthetic bandit feedback dataset.
        """
        check_scalar(n_rounds, "n_rounds", int, min_val=1)
        context = self.random_.normal(size=(n_rounds, self.dim_context))
        # sample actions for each round based on the behavior policy
        if self.behavior_policy_function is None:
            behavior_policy_ = np.tile(self.behavior_policy, (n_rounds, 1))
            behavior_policy_ = softmax(behavior_policy_ / self.tau)
            action = self.random_.choice(
                np.arange(self.n_actions), p=self.behavior_policy, size=n_rounds
            )
        else:
            behavior_policy_ = self.behavior_policy_function(
                context=context,
                action_context=self.action_context,
                random_state=self.random_state,
            )
            behavior_policy_ = softmax(behavior_policy_ / self.tau)
            action = sample_action_fast(
                behavior_policy_, random_state=self.random_state
            )
        pscore = behavior_policy_[np.arange(n_rounds), action]

        # sample reward based on the context and action
        expected_reward_ = self.calc_expected_reward(context)
        if RewardType(self.reward_type) == RewardType.CONTINUOUS:
            # correct expected_reward_, as we use truncated normal distribution here
            mean = expected_reward_
            a = (self.reward_min - mean) / self.reward_std
            b = (self.reward_max - mean) / self.reward_std
            expected_reward_ = truncnorm.stats(
                a=a, b=b, loc=mean, scale=self.reward_std, moments="m"
            )
        reward = self.sample_reward_given_expected_reward(expected_reward_, action)

        return dict(
            n_rounds=n_rounds,
            n_actions=self.n_actions,
            context=context,
            action_context=self.action_context,
            action=action,
            position=None,  # position effect is not considered in synthetic data
            reward=reward,
            expected_reward=expected_reward_,
            pscore=pscore,
        )

    def calc_ground_truth_policy_value(
        self, expected_reward: np.ndarray, action_dist: np.ndarray
    ) -> float:
        """Calculate the policy value of given action distribution on the given expected_reward.
        Parameters
        -----------
        expected_reward: array-like, shape (n_rounds, n_actions)
            Expected reward given context (:math:`x`) and action (:math:`a`), i.e., :math:`q(x,a):=\\mathbb{E}[r|x,a]`.
            This is often the expected_reward of the test set of logged bandit feedback data.
        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.
        Returns
        ----------
        policy_value: float
            The policy value of the given action distribution on the given bandit feedback data.
        """
        check_array(array=expected_reward, name="expected_reward", expected_dim=2)
        check_array(array=action_dist, name="action_dist", expected_dim=3)
        if expected_reward.shape[0] != action_dist.shape[0]:
            raise ValueError(
                "Expected `expected_reward.shape[0] = action_dist.shape[0]`, but found it False"
            )
        if expected_reward.shape[1] != action_dist.shape[1]:
            raise ValueError(
                "Expected `expected_reward.shape[1] = action_dist.shape[1]`, but found it False"
            )

        return np.average(expected_reward, weights=action_dist[:, :, 0], axis=1).mean()


def logistic_reward_function(
    context: np.ndarray,
    action_context: np.ndarray,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Logistic mean reward function for synthetic bandit datasets.
    Parameters
    -----------
    context: array-like, shape (n_rounds, dim_context)
        Context vectors characterizing each round (such as user information).
    action_context: array-like, shape (n_actions, dim_action_context)
        Vector representation for each action.
    random_state: int, default=None
        Controls the random seed in sampling dataset.
    Returns
    ---------
    expected_reward: array-like, shape (n_rounds, n_actions)
        Expected reward given context (:math:`x`) and action (:math:`a`), i.e., :math:`q(x,a):=\\mathbb{E}[r|x,a]`.
    """
    check_array(array=context, name="context", expected_dim=2)
    check_array(array=action_context, name="action_context", expected_dim=2)

    random_ = check_random_state(random_state)
    logits = np.zeros((context.shape[0], action_context.shape[0]))
    # each arm has different coefficient vectors
    coef_ = random_.uniform(size=(action_context.shape[0], context.shape[1]))
    action_coef_ = random_.uniform(size=action_context.shape[1])
    for d in np.arange(action_context.shape[0]):
        logits[:, d] = context @ coef_[d] + action_context[d] @ action_coef_

    return sigmoid(logits)


def linear_reward_function(
    context: np.ndarray,
    action_context: np.ndarray,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Linear mean reward function for synthetic bandit datasets.
    Parameters
    -----------
    context: array-like, shape (n_rounds, dim_context)
        Context vectors characterizing each round (such as user information).
    action_context: array-like, shape (n_actions, dim_action_context)
        Vector representation for each action.
    random_state: int, default=None
        Controls the random seed in sampling dataset.
    Returns
    ---------
    expected_reward: array-like, shape (n_rounds, n_actions)
        Expected reward given context (:math:`x`) and action (:math:`a`), i.e., :math:`q(x,a):=\\mathbb{E}[r|x,a]`.
    """
    check_array(array=context, name="context", expected_dim=2)
    check_array(array=action_context, name="action_context", expected_dim=2)

    random_ = check_random_state(random_state)
    expected_reward = np.zeros((context.shape[0], action_context.shape[0]))
    # each arm has different coefficient vectors
    coef_ = random_.uniform(size=(action_context.shape[0], context.shape[1]))
    action_coef_ = random_.uniform(size=action_context.shape[1])
    for d in np.arange(action_context.shape[0]):
        expected_reward[:, d] = context @ coef_[d] + action_context[d] @ action_coef_

    return expected_reward


def linear_behavior_policy(
    context: np.ndarray,
    action_context: np.ndarray,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Linear contextual behavior policy for synthetic bandit datasets.
    Parameters
    -----------
    context: array-like, shape (n_rounds, dim_context)
        Context vectors characterizing each round (such as user information).
    action_context: array-like, shape (n_actions, dim_action_context)
        Vector representation for each action.
    random_state: int, default=None
        Controls the random seed in sampling dataset.
    Returns
    ---------
    behavior_policy: array-like, shape (n_rounds, n_actions)
        Logit values given context (:math:`x`), i.e., :math:`\\pi: \\mathcal{X} \\rightarrow \\Delta(\\mathcal{A})`.
    """
    check_array(array=context, name="context", expected_dim=2)
    check_array(array=action_context, name="action_context", expected_dim=2)

    random_ = check_random_state(random_state)
    logits = np.zeros((context.shape[0], action_context.shape[0]))
    coef_ = random_.uniform(size=context.shape[1])
    action_coef_ = random_.uniform(size=action_context.shape[1])
    for d in np.arange(action_context.shape[0]):
        logits[:, d] = context @ coef_ + action_context[d] @ action_coef_

    return logits
```

<!-- #region id="jOiYdOljzxGO" -->
## Base Models
<!-- #endregion -->

```python cellView="form" id="1lsiTpWxzyqf"
#@markdown Regression model
@dataclass
class RegressionModel(BaseEstimator):
    """Machine learning model to estimate the mean reward function (:math:`q(x,a):= \\mathbb{E}[r|x,a]`).
    Note
    -------
    Reward (or outcome) :math:`r` must be either binary or continuous.
    Parameters
    ------------
    base_model: BaseEstimator
        A machine learning model used to estimate the mean reward function.
    n_actions: int
        Number of actions.
    len_list: int, default=1
        Length of a list of actions recommended in each impression.
        When Open Bandit Dataset is used, 3 should be set.
    action_context: array-like, shape (n_actions, dim_action_context), default=None
        Context vector characterizing action (i.e., vector representation of each action).
        If not given, one-hot encoding of the action variable is used as default.
    fitting_method: str, default='normal'
        Method to fit the regression model.
        Must be one of ['normal', 'iw', 'mrdr'] where 'iw' stands for importance weighting and
        'mrdr' stands for more robust doubly robust.
    References
    -----------
    Mehrdad Farajtabar, Yinlam Chow, and Mohammad Ghavamzadeh.
    "More Robust Doubly Robust Off-policy Evaluation.", 2018.
    Yi Su, Maria Dimakopoulou, Akshay Krishnamurthy, and Miroslav Dudik.
    "Doubly Robust Off-Policy Evaluation with Shrinkage.", 2020.
    Yusuke Narita, Shota Yasui, and Kohei Yata.
    "Off-policy Bandit and Reinforcement Learning.", 2020.
    """

    base_model: BaseEstimator
    n_actions: int
    len_list: int = 1
    action_context: Optional[np.ndarray] = None
    fitting_method: str = "normal"

    def __post_init__(self) -> None:
        """Initialize Class."""
        check_scalar(self.n_actions, "n_actions", int, min_val=2)
        check_scalar(self.len_list, "len_list", int, min_val=1)
        if not (
            isinstance(self.fitting_method, str)
            and self.fitting_method in ["normal", "iw", "mrdr"]
        ):
            raise ValueError(
                f"fitting_method must be one of 'normal', 'iw', or 'mrdr', but {self.fitting_method} is given"
            )
        if not isinstance(self.base_model, BaseEstimator):
            raise ValueError(
                "base_model must be BaseEstimator or a child class of BaseEstimator"
            )

        self.base_model_list = [
            clone(self.base_model) for _ in np.arange(self.len_list)
        ]
        if self.action_context is None:
            self.action_context = np.eye(self.n_actions, dtype=int)

    def fit(
        self,
        context: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        pscore: Optional[np.ndarray] = None,
        position: Optional[np.ndarray] = None,
        action_dist: Optional[np.ndarray] = None,
    ) -> None:
        """Fit the regression model on given logged bandit feedback data.
        Parameters
        ----------
        context: array-like, shape (n_rounds, dim_context)
            Context vectors observed in each round of the logged bandit feedback, i.e., :math:`x_t`.
        action: array-like, shape (n_rounds,)
            Action sampled by behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.
        reward: array-like, shape (n_rounds,)
            Reward observed in each round of the logged bandit feedback, i.e., :math:`r_t`.
        pscore: array-like, shape (n_rounds,)
            Action choice probabilities of behavior policy (propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.
            When None is given, behavior policy is assumed to be uniform.
        position: array-like, shape (n_rounds,), default=None
            Position of recommendation interface where action was presented in each round of the given logged bandit data.
            If None is set, a regression model assumes that there is only one position.
            When `len_list` > 1, this position argument has to be set.
        action_dist: array-like, shape (n_rounds, n_actions, len_list), default=None
            Action choice probabilities of evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.
            When either of 'iw' or 'mrdr' is used as the 'fitting_method' argument, then `action_dist` must be given.
        """
        check_bandit_feedback_inputs(
            context=context,
            action=action,
            reward=reward,
            pscore=pscore,
            position=position,
            action_context=self.action_context,
        )
        n_rounds = context.shape[0]

        if position is None or self.len_list == 1:
            position = np.zeros_like(action)
        else:
            if position.max() >= self.len_list:
                raise ValueError(
                    f"position elements must be smaller than len_list, but the maximum value is {position.max()} (>= {self.len_list})"
                )
        if self.fitting_method in ["iw", "mrdr"]:
            if not (isinstance(action_dist, np.ndarray) and action_dist.ndim == 3):
                raise ValueError(
                    "when fitting_method is either 'iw' or 'mrdr', action_dist (a 3-dimensional ndarray) must be given"
                )
            if action_dist.shape != (n_rounds, self.n_actions, self.len_list):
                raise ValueError(
                    f"shape of action_dist must be (n_rounds, n_actions, len_list)=({n_rounds, self.n_actions, self.len_list}), but is {action_dist.shape}"
                )
            if not np.allclose(action_dist.sum(axis=1), 1):
                raise ValueError("action_dist must be a probability distribution")
        if pscore is None:
            pscore = np.ones_like(action) / self.n_actions

        for position_ in np.arange(self.len_list):
            idx = position == position_
            X = self._pre_process_for_reg_model(
                context=context[idx],
                action=action[idx],
                action_context=self.action_context,
            )
            if X.shape[0] == 0:
                raise ValueError(f"No training data at position {position_}")
            # train the base model according to the given `fitting method`
            if self.fitting_method == "normal":
                self.base_model_list[position_].fit(X, reward[idx])
            else:
                action_dist_at_position = action_dist[
                    np.arange(n_rounds),
                    action,
                    position_ * np.ones(n_rounds, dtype=int),
                ][idx]
                if self.fitting_method == "iw":
                    sample_weight = action_dist_at_position / pscore[idx]
                    self.base_model_list[position_].fit(
                        X, reward[idx], sample_weight=sample_weight
                    )
                elif self.fitting_method == "mrdr":
                    sample_weight = action_dist_at_position
                    sample_weight *= 1.0 - pscore[idx]
                    sample_weight /= pscore[idx] ** 2
                    self.base_model_list[position_].fit(
                        X, reward[idx], sample_weight=sample_weight
                    )

    def predict(self, context: np.ndarray) -> np.ndarray:
        """Predict the mean reward function.
        Parameters
        -----------
        context: array-like, shape (n_rounds_of_new_data, dim_context)
            Context vectors of new data.
        Returns
        -----------
        estimated_rewards_by_reg_model: array-like, shape (n_rounds_of_new_data, n_actions, len_list)
            Expected rewards of new data estimated by the regression model.
        """
        n_rounds_of_new_data = context.shape[0]
        ones_n_rounds_arr = np.ones(n_rounds_of_new_data, int)
        estimated_rewards_by_reg_model = np.zeros(
            (n_rounds_of_new_data, self.n_actions, self.len_list)
        )
        for action_ in np.arange(self.n_actions):
            for position_ in np.arange(self.len_list):
                X = self._pre_process_for_reg_model(
                    context=context,
                    action=action_ * ones_n_rounds_arr,
                    action_context=self.action_context,
                )
                estimated_rewards_ = (
                    self.base_model_list[position_].predict_proba(X)[:, 1]
                    if is_classifier(self.base_model_list[position_])
                    else self.base_model_list[position_].predict(X)
                )
                estimated_rewards_by_reg_model[
                    np.arange(n_rounds_of_new_data),
                    action_ * ones_n_rounds_arr,
                    position_ * ones_n_rounds_arr,
                ] = estimated_rewards_
        return estimated_rewards_by_reg_model

    def fit_predict(
        self,
        context: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        pscore: Optional[np.ndarray] = None,
        position: Optional[np.ndarray] = None,
        action_dist: Optional[np.ndarray] = None,
        n_folds: int = 1,
        random_state: Optional[int] = None,
    ) -> np.ndarray:
        """Fit the regression model on given logged bandit feedback data and predict the reward function of the same data.
        Note
        ------
        When `n_folds` is larger than 1, then the cross-fitting procedure is applied.
        See the reference for the details about the cross-fitting technique.
        Parameters
        ----------
        context: array-like, shape (n_rounds, dim_context)
            Context vectors observed in each round of the logged bandit feedback, i.e., :math:`x_t`.
        action: array-like, shape (n_rounds,)
            Action sampled by behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.
        reward: array-like, shape (n_rounds,)
            Observed rewards (or outcome) in each round, i.e., :math:`r_t`.
        pscore: array-like, shape (n_rounds,), default=None
            Action choice probabilities (propensity score) of a behavior policy
            in the training logged bandit feedback.
            When None is given, the the behavior policy is assumed to be a uniform one.
        position: array-like, shape (n_rounds,), default=None
            Position of recommendation interface where action was presented in each round of the given logged bandit data.
            If None is set, a regression model assumes that there is only one position.
            When `len_list` > 1, this position argument has to be set.
        action_dist: array-like, shape (n_rounds, n_actions, len_list), default=None
            Action choice probabilities of evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.
            When either of 'iw' or 'mrdr' is used as the 'fitting_method' argument, then `action_dist` must be given.
        n_folds: int, default=1
            Number of folds in the cross-fitting procedure.
            When 1 is given, the regression model is trained on the whole logged bandit feedback data.
            Please refer to https://arxiv.org/abs/2002.08536 about the details of the cross-fitting procedure.
        random_state: int, default=None
            `random_state` affects the ordering of the indices, which controls the randomness of each fold.
            See https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html for the details.
        Returns
        -----------
        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list)
            Expected rewards of new data estimated by the regression model.
        """
        check_bandit_feedback_inputs(
            context=context,
            action=action,
            reward=reward,
            pscore=pscore,
            position=position,
            action_context=self.action_context,
        )
        n_rounds = context.shape[0]

        check_scalar(n_folds, "n_folds", int, min_val=1)
        check_random_state(random_state)

        if position is None or self.len_list == 1:
            position = np.zeros_like(action)
        else:
            if position.max() >= self.len_list:
                raise ValueError(
                    f"position elements must be smaller than len_list, but the maximum value is {position.max()} (>= {self.len_list})"
                )
        if self.fitting_method in ["iw", "mrdr"]:
            if not (isinstance(action_dist, np.ndarray) and action_dist.ndim == 3):
                raise ValueError(
                    "when fitting_method is either 'iw' or 'mrdr', action_dist (a 3-dimensional ndarray) must be given"
                )
            if action_dist.shape != (n_rounds, self.n_actions, self.len_list):
                raise ValueError(
                    f"shape of action_dist must be (n_rounds, n_actions, len_list)=({n_rounds, self.n_actions, self.len_list}), but is {action_dist.shape}"
                )
        if pscore is None:
            pscore = np.ones_like(action) / self.n_actions

        if n_folds == 1:
            self.fit(
                context=context,
                action=action,
                reward=reward,
                pscore=pscore,
                position=position,
                action_dist=action_dist,
            )
            return self.predict(context=context)
        else:
            estimated_rewards_by_reg_model = np.zeros(
                (n_rounds, self.n_actions, self.len_list)
            )
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        kf.get_n_splits(context)
        for train_idx, test_idx in kf.split(context):
            action_dist_tr = (
                action_dist[train_idx] if action_dist is not None else action_dist
            )
            self.fit(
                context=context[train_idx],
                action=action[train_idx],
                reward=reward[train_idx],
                pscore=pscore[train_idx],
                position=position[train_idx],
                action_dist=action_dist_tr,
            )
            estimated_rewards_by_reg_model[test_idx, :, :] = self.predict(
                context=context[test_idx]
            )
        return estimated_rewards_by_reg_model

    def _pre_process_for_reg_model(
        self,
        context: np.ndarray,
        action: np.ndarray,
        action_context: np.ndarray,
    ) -> np.ndarray:
        """Preprocess feature vectors to train a regression model.
        Note
        -----
        Please override this method if you want to use another feature enginnering
        for training the regression model.
        Parameters
        -----------
        context: array-like, shape (n_rounds,)
            Context vectors observed in each round of the logged bandit feedback, i.e., :math:`x_t`.
        action: array-like, shape (n_rounds,)
            Action sampled by behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.
        action_context: array-like, shape shape (n_actions, dim_action_context)
            Context vector characterizing action (i.e., vector representation of each action).
        """
        return np.c_[context, action_context[action]]
```

<!-- #region id="eJtM3cS7sNO0" -->
## Estimators
<!-- #endregion -->

```python id="mFQzKhqZwdqF"
# from ..utils import check_array
# from ..utils import check_ope_inputs
# from ..utils import estimate_confidence_interval_by_bootstrap
# from .helper import estimate_bias_in_ope
# from .helper import estimate_high_probability_upper_bound_bias
```

```python cellView="form" id="7DXsSwoewN3P"
#@markdown estimator classes
@dataclass
class BaseOffPolicyEstimator(metaclass=ABCMeta):
    """Base class for OPE estimators."""

    @abstractmethod
    def _estimate_round_rewards(self) -> np.ndarray:
        """Estimate round-wise (or sample-wise) rewards."""
        raise NotImplementedError

    @abstractmethod
    def estimate_policy_value(self) -> float:
        """Estimate the policy value of evaluation policy."""
        raise NotImplementedError

    @abstractmethod
    def estimate_interval(self) -> Dict[str, float]:
        """Estimate confidence interval of policy value by nonparametric bootstrap procedure."""
        raise NotImplementedError


@dataclass
class ReplayMethod(BaseOffPolicyEstimator):
    """Relpay Method (RM).

    Note
    -------
    Replay Method (RM) estimates the policy value of evaluation policy :math:`\\pi_e` by

    .. math::

        \\hat{V}_{\\mathrm{RM}} (\\pi_e; \\mathcal{D}) :=
        \\frac{\\mathbb{E}_{\\mathcal{D}}[\\mathbb{I} \\{ \\pi_e (x_t) = a_t \\} r_t ]}{\\mathbb{E}_{\\mathcal{D}}[\\mathbb{I} \\{ \\pi_e (x_t) = a_t \\}]},

    where :math:`\\mathcal{D}=\\{(x_t,a_t,r_t)\\}_{t=1}^{T}` is logged bandit feedback data with :math:`T` rounds collected by
    a behavior policy :math:`\\pi_b`. :math:`\\pi_e: \\mathcal{X} \\rightarrow \\mathcal{A}` is the function
    representing action choices by the evaluation policy realized during offline bandit simulation.
    :math:`\\mathbb{E}_{\\mathcal{D}}[\\cdot]` is the empirical average over :math:`T` observations in :math:`\\mathcal{D}`.

    Parameters
    ----------
    estimator_name: str, default='rm'.
        Name of the estimator.

    References
    ------------
    Lihong Li, Wei Chu, John Langford, and Xuanhui Wang.
    "Unbiased Offline Evaluation of Contextual-bandit-based News Article Recommendation Algorithms.", 2011.

    """

    estimator_name: str = "rm"

    def _estimate_round_rewards(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        action_dist: np.ndarray,
        position: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """Estimate round-wise (or sample-wise) rewards.

        Parameters
        ------------
        reward: array-like, shape (n_rounds,)
            Reward observed in each round of the logged bandit feedback, i.e., :math:`r_t`.

        action: array-like, shape (n_rounds,)
            Action sampled by behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of evaluation policy (must be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        position: array-like, shape (n_rounds,), default=None
            Position of recommendation interface where action was presented in each round of the given logged bandit data.
            When None is given, the effect of position on the reward will be ignored.
            (If only one action is chosen and there is no posion, then you can just ignore this argument.)

        Returns
        ----------
        estimated_rewards: array-like, shape (n_rounds,)
            Rewards of each round estimated by the Replay Method.

        """
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)
        action_match = np.array(
            action_dist[np.arange(action.shape[0]), action, position] == 1
        )
        estimated_rewards = np.zeros_like(action_match)
        if action_match.sum() > 0.0:
            estimated_rewards = action_match * reward / action_match.mean()
        return estimated_rewards

    def estimate_policy_value(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        action_dist: np.ndarray,
        position: Optional[np.ndarray] = None,
        **kwargs,
    ) -> float:
        """Estimate the policy value of evaluation policy.

        Parameters
        ------------
        reward: array-like, shape (n_rounds,)
            Reward observed in each round of the logged bandit feedback, i.e., :math:`r_t`.

        action: array-like, shape (n_rounds,)
            Action sampled by behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of evaluation policy (must be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        position: array-like, shape (n_rounds,), default=None
            Position of recommendation interface where action was presented in each round of the given logged bandit data.
            When None is given, the effect of position on the reward will be ignored.
            (If only one action is chosen and there is no posion, then you can just ignore this argument.)

        Returns
        ----------
        V_hat: float
            Estimated policy value (performance) of a given evaluation policy.

        """
        check_array(array=reward, name="reward", expected_dim=1)
        check_array(array=action, name="action", expected_dim=1)
        check_ope_inputs(
            action_dist=action_dist, position=position, action=action, reward=reward
        )
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)

        return self._estimate_round_rewards(
            reward=reward,
            action=action,
            position=position,
            action_dist=action_dist,
        ).mean()

    def estimate_interval(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        action_dist: np.ndarray,
        position: Optional[np.ndarray] = None,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 100,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """Estimate confidence interval of policy value by nonparametric bootstrap procedure.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Reward observed in each round of the logged bandit feedback, i.e., :math:`r_t`.

        action: array-like, shape (n_rounds,)
            Action sampled by behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of evaluation policy (must be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        position: array-like, shape (n_rounds,), default=None
            Position of recommendation interface where action was presented in each round of the given logged bandit data.
            When None is given, the effect of position on the reward will be ignored.
            (If only one action is chosen and there is no posion, then you can just ignore this argument.)

        alpha: float, default=0.05
            Significance level.

        n_bootstrap_samples: int, default=10000
            Number of resampling performed in the bootstrap procedure.

        random_state: int, default=None
            Controls the random seed in bootstrap sampling.

        Returns
        ----------
        estimated_confidence_interval: Dict[str, float]
            Dictionary storing the estimated mean and upper-lower confidence bounds.

        """
        check_array(array=reward, name="reward", expected_dim=1)
        check_array(array=action, name="action", expected_dim=1)
        check_ope_inputs(
            action_dist=action_dist, position=position, action=action, reward=reward
        )
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)

        estimated_round_rewards = self._estimate_round_rewards(
            reward=reward,
            action=action,
            position=position,
            action_dist=action_dist,
        )
        return estimate_confidence_interval_by_bootstrap(
            samples=estimated_round_rewards,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )


@dataclass
class InverseProbabilityWeighting(BaseOffPolicyEstimator):
    """Inverse Probability Weighting (IPW) Estimator.

    Note
    -------
    Inverse Probability Weighting (IPW) estimates the policy value of evaluation policy :math:`\\pi_e` by

    .. math::

        \\hat{V}_{\\mathrm{IPW}} (\\pi_e; \\mathcal{D}) := \\mathbb{E}_{\\mathcal{D}} [ w(x_t,a_t) r_t],

    where :math:`\\mathcal{D}=\\{(x_t,a_t,r_t)\\}_{t=1}^{T}` is logged bandit feedback data with :math:`T` rounds collected by
    a behavior policy :math:`\\pi_b`. :math:`w(x,a):=\\pi_e (a|x)/\\pi_b (a|x)` is the importance weight given :math:`x` and :math:`a`.
    :math:`\\mathbb{E}_{\\mathcal{D}}[\\cdot]` is the empirical average over :math:`T` observations in :math:`\\mathcal{D}`.
    When the weight-clipping is applied, a large importance weight is clipped as :math:`\\hat{w}(x,a) := \\min \\{ \\lambda, w(x,a) \\}`
    where :math:`\\lambda (>0)` is a hyperparameter that decides a maximum allowed importance weight.

    IPW re-weights the rewards by the ratio of the evaluation policy and behavior policy (importance weight).
    When the behavior policy is known, IPW is unbiased and consistent for the true policy value.
    However, it can have a large variance, especially when the evaluation policy significantly deviates from the behavior policy.

    Parameters
    ------------
    lambda_: float, default=np.inf
        A maximum possible value of the importance weight.
        When a positive finite value is given, importance weights larger than `lambda_` will be clipped.

    estimator_name: str, default='ipw'.
        Name of the estimator.

    References
    ------------
    Alex Strehl, John Langford, Lihong Li, and Sham M Kakade.
    "Learning from Logged Implicit Exploration Data"., 2010.

    Miroslav Dudík, Dumitru Erhan, John Langford, and Lihong Li.
    "Doubly Robust Policy Evaluation and Optimization.", 2014.

    Yi Su, Maria Dimakopoulou, Akshay Krishnamurthy, and Miroslav Dudik.
    "Doubly Robust Off-Policy Evaluation with Shrinkage.", 2020.

    """

    lambda_: float = np.inf
    estimator_name: str = "ipw"

    def __post_init__(self) -> None:
        """Initialize Class."""
        check_scalar(
            self.lambda_,
            name="lambda_",
            target_type=(int, float),
            min_val=0.0,
        )
        if self.lambda_ != self.lambda_:
            raise ValueError("lambda_ must not be nan")

    def _estimate_round_rewards(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        pscore: np.ndarray,
        action_dist: np.ndarray,
        position: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """Estimate round-wise (or sample-wise) rewards.

        Parameters
        ----------
        reward: array-like or Tensor, shape (n_rounds,)
            Reward observed in each round of the logged bandit feedback, i.e., :math:`r_t`.

        action: array-like or Tensor, shape (n_rounds,)
            Action sampled by behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        pscore: array-like or Tensor, shape (n_rounds,)
            Action choice probabilities of behavior policy (propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

        action_dist: array-like or Tensor, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        position: array-like or Tensor, shape (n_rounds,), default=None
            Position of recommendation interface where action was presented in each round of the given logged bandit data.
            When None is given, the effect of position on the reward will be ignored.
            (If only one action is chosen and there is no posion, then you can just ignore this argument.)

        Returns
        ----------
        estimated_rewards: array-like or Tensor, shape (n_rounds,)
            Rewards of each round estimated by IPW.

        """
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)
        iw = action_dist[np.arange(action.shape[0]), action, position] / pscore
        # weight clipping
        if isinstance(iw, np.ndarray):
            iw = np.minimum(iw, self.lambda_)
        return reward * iw

    def estimate_policy_value(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        pscore: np.ndarray,
        action_dist: np.ndarray,
        position: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """Estimate the policy value of evaluation policy.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Reward observed in each round of the logged bandit feedback, i.e., :math:`r_t`.

        action: array-like, shape (n_rounds,)
            Action sampled by behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        pscore: array-like, shape (n_rounds,)
            Action choice probabilities of behavior policy (propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        position: array-like, shape (n_rounds,), default=None
            Position of recommendation interface where action was presented in each round of the given logged bandit data.
            When None is given, the effect of position on the reward will be ignored.
            (If only one action is chosen and there is no posion, then you can just ignore this argument.)

        Returns
        ----------
        V_hat: float
            Estimated policy value (performance) of a given evaluation policy.

        """
        check_array(array=reward, name="reward", expected_dim=1)
        check_array(array=action, name="action", expected_dim=1)
        check_array(array=pscore, name="pscore", expected_dim=1)
        check_ope_inputs(
            action_dist=action_dist,
            position=position,
            action=action,
            reward=reward,
            pscore=pscore,
        )
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)

        return self._estimate_round_rewards(
            reward=reward,
            action=action,
            position=position,
            pscore=pscore,
            action_dist=action_dist,
        ).mean()

    def estimate_interval(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        pscore: np.ndarray,
        action_dist: np.ndarray,
        position: Optional[np.ndarray] = None,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 10000,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """Estimate confidence interval of policy value by nonparametric bootstrap procedure.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Reward observed in each round of the logged bandit feedback, i.e., :math:`r_t`.

        action: array-like, shape (n_rounds,)
            Action sampled by behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        pscore: array-like, shape (n_rounds,)
            Action choice probabilities of behavior policy (propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        position: array-like, shape (n_rounds,), default=None
            Position of recommendation interface where action was presented in each round of the given logged bandit data.
            When None is given, the effect of position on the reward will be ignored.
            (If only one action is chosen and there is no posion, then you can just ignore this argument.)

        alpha: float, default=0.05
            Significance level.

        n_bootstrap_samples: int, default=10000
            Number of resampling performed in the bootstrap procedure.

        random_state: int, default=None
            Controls the random seed in bootstrap sampling.

        Returns
        ----------
        estimated_confidence_interval: Dict[str, float]
            Dictionary storing the estimated mean and upper-lower confidence bounds.

        """
        check_array(array=reward, name="reward", expected_dim=1)
        check_array(array=action, name="action", expected_dim=1)
        check_array(array=pscore, name="pscore", expected_dim=1)
        check_ope_inputs(
            action_dist=action_dist,
            position=position,
            action=action,
            reward=reward,
            pscore=pscore,
        )
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)

        estimated_round_rewards = self._estimate_round_rewards(
            reward=reward,
            action=action,
            position=position,
            pscore=pscore,
            action_dist=action_dist,
        )
        return estimate_confidence_interval_by_bootstrap(
            samples=estimated_round_rewards,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )

    def _estimate_mse_score(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        pscore: np.ndarray,
        action_dist: np.ndarray,
        position: Optional[np.ndarray] = None,
        use_bias_upper_bound: bool = True,
        delta: float = 0.05,
        **kwargs,
    ) -> float:
        """Estimate the MSE score of a given clipping hyperparameter to conduct hyperparameter tuning.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Reward observed in each round of the logged bandit feedback, i.e., :math:`r_t`.

        action: array-like, shape (n_rounds,)
            Action sampled by behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        pscore: array-like, shape (n_rounds,)
            Action choice probabilities of behavior policy (propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        position: array-like, shape (n_rounds,), default=None
            Position of recommendation interface where action was presented in each round of the given logged bandit data.

        use_bias_upper_bound: bool, default=True
            Whether to use bias upper bound in hyperparameter tuning.
            If False, direct bias estimator is used to estimate the MSE.

        delta: float, default=0.05
            A confidence delta to construct a high probability upper bound based on the Bernstein’s inequality.

        Returns
        ----------
        estimated_mse_score: float
            Estimated MSE score of a given clipping hyperparameter `lambda_`.
            MSE score is the sum of (high probability) upper bound of bias and the sample variance.
            This is estimated using the automatic hyperparameter tuning procedure
            based on Section 5 of Su et al.(2020).

        """
        n_rounds = reward.shape[0]
        # estimate the sample variance of IPW with clipping
        sample_variance = np.var(
            self._estimate_round_rewards(
                reward=reward,
                action=action,
                pscore=pscore,
                action_dist=action_dist,
                position=position,
            )
        )
        sample_variance /= n_rounds

        # estimate the (high probability) upper bound of the bias of IPW with clipping
        iw = action_dist[np.arange(n_rounds), action, position] / pscore
        if use_bias_upper_bound:
            bias_term = estimate_high_probability_upper_bound_bias(
                reward=reward, iw=iw, iw_hat=np.minimum(iw, self.lambda_), delta=delta
            )
        else:
            bias_term = estimate_bias_in_ope(
                reward=reward,
                iw=iw,
                iw_hat=np.minimum(iw, self.lambda_),
            )
        estimated_mse_score = sample_variance + (bias_term ** 2)

        return estimated_mse_score


@dataclass
class SelfNormalizedInverseProbabilityWeighting(InverseProbabilityWeighting):
    """Self-Normalized Inverse Probability Weighting (SNIPW) Estimator.

    Note
    -------
    Self-Normalized Inverse Probability Weighting (SNIPW) estimates the policy value of evaluation policy :math:`\\pi_e` by

    .. math::

        \\hat{V}_{\\mathrm{SNIPW}} (\\pi_e; \\mathcal{D}) :=
        \\frac{\\mathbb{E}_{\\mathcal{D}} [w(x_t,a_t) r_t]}{ \\mathbb{E}_{\\mathcal{D}} [w(x_t,a_t)]},

    where :math:`\\mathcal{D}=\\{(x_t,a_t,r_t)\\}_{t=1}^{T}` is logged bandit feedback data with :math:`T` rounds collected by
    a behavior policy :math:`\\pi_b`. :math:`w(x,a):=\\pi_e (a|x)/\\pi_b (a|x)` is the importance weight given :math:`x` and :math:`a`.
    :math:`\\mathbb{E}_{\\mathcal{D}}[\\cdot]` is the empirical average over :math:`T` observations in :math:`\\mathcal{D}`.

    SNIPW re-weights the observed rewards by the self-normalized importance weihgt.
    This estimator is not unbiased even when the behavior policy is known.
    However, it is still consistent for the true policy value and increases the stability in some senses.
    See the references for the detailed discussions.

    Parameters
    ----------
    estimator_name: str, default='snipw'.
        Name of the estimator.

    References
    ----------
    Adith Swaminathan and Thorsten Joachims.
    "The Self-normalized Estimator for Counterfactual Learning.", 2015.

    Nathan Kallus and Masatoshi Uehara.
    "Intrinsically Efficient, Stable, and Bounded Off-Policy Evaluation for Reinforcement Learning.", 2019.

    """

    estimator_name: str = "snipw"

    def _estimate_round_rewards(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        pscore: np.ndarray,
        action_dist: np.ndarray,
        position: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """Estimate round-wise (or sample-wise) rewards.

        Parameters
        ----------
        reward: array-like or Tensor, shape (n_rounds,)
            Reward observed in each round of the logged bandit feedback, i.e., :math:`r_t`.

        action: array-like or Tensor, shape (n_rounds,)
            Action sampled by behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        pscore: array-like or Tensor, shape (n_rounds,)
            Action choice probabilities of behavior policy (propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

        action_dist: array-like or Tensor, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        position: array-like or Tensor, shape (n_rounds,), default=None
            Position of recommendation interface where action was presented in each round of the given logged bandit data.

        Returns
        ----------
        estimated_rewards: array-like or Tensor, shape (n_rounds,)
            Rewards of each round estimated by the SNIPW estimator.

        """
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)
        iw = action_dist[np.arange(action.shape[0]), action, position] / pscore
        return reward * iw / iw.mean()


@dataclass
class DirectMethod(BaseOffPolicyEstimator):
    """Direct Method (DM).

    Note
    -------
    DM first learns a supervised machine learning model, such as ridge regression and gradient boosting,
    to estimate the mean reward function (:math:`q(x,a) = \\mathbb{E}[r|x,a]`).
    It then uses it to estimate the policy value as follows.

    .. math::

        \\hat{V}_{\\mathrm{DM}} (\\pi_e; \\mathcal{D}, \\hat{q})
        &:= \\mathbb{E}_{\\mathcal{D}} \\left[ \\sum_{a \\in \\mathcal{A}} \\hat{q} (x_t,a) \\pi_e(a|x_t) \\right],    \\\\
        & =  \\mathbb{E}_{\\mathcal{D}}[\\hat{q} (x_t,\\pi_e)],

    where :math:`\\mathcal{D}=\\{(x_t,a_t,r_t)\\}_{t=1}^{T}` is logged bandit feedback data with :math:`T` rounds collected by
    a behavior policy :math:`\\pi_b`. :math:`\\mathbb{E}_{\\mathcal{D}}[\\cdot]` is the empirical average over :math:`T` observations in :math:`\\mathcal{D}`.
    :math:`\\hat{q} (x,a)` is an estimated expected reward given :math:`x` and :math:`a`.
    :math:`\\hat{q} (x_t,\\pi):= \\mathbb{E}_{a \\sim \\pi(a|x)}[\\hat{q}(x,a)]` is the expectation of the estimated reward function over :math:`\\pi`.
    To estimate the mean reward function, please use `obp.ope.regression_model.RegressionModel`, which supports several fitting methods specific to OPE.

    If the regression model (:math:`\\hat{q}`) is a good approximation to the true mean reward function,
    this estimator accurately estimates the policy value of the evaluation policy.
    If the regression function fails to approximate the mean reward function well,
    however, the final estimator is no longer consistent.

    Parameters
    ----------
    estimator_name: str, default='dm'.
        Name of the estimator.

    References
    ----------
    Alina Beygelzimer and John Langford.
    "The offset tree for learning with partial labels.", 2009.

    Miroslav Dudík, Dumitru Erhan, John Langford, and Lihong Li.
    "Doubly Robust Policy Evaluation and Optimization.", 2014.

    """

    estimator_name: str = "dm"

    def _estimate_round_rewards(
        self,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: np.ndarray,
        position: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """Estimate the policy value of evaluation policy.

        Parameters
        ----------
        action_dist: array-like or Tensor, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        estimated_rewards_by_reg_model: array-like or Tensor, shape (n_rounds, n_actions, len_list)
            Expected rewards given context, action, and position estimated by regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.

        position: array-like or Tensor, shape (n_rounds,), default=None
            Position of recommendation interface where action was presented in each round of the given logged bandit data.
            When None is given, the effect of position on the reward will be ignored.
            (If only one action is chosen and there is no posion, then you can just ignore this argument.)

        Returns
        ----------
        estimated_rewards: array-like or Tensor, shape (n_rounds,)
            Rewards of each round estimated by the DM estimator.

        """
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)
        n_rounds = position.shape[0]
        q_hat_at_position = estimated_rewards_by_reg_model[
            np.arange(n_rounds), :, position
        ]
        pi_e_at_position = action_dist[np.arange(n_rounds), :, position]

        if isinstance(action_dist, np.ndarray):
            return np.average(
                q_hat_at_position,
                weights=pi_e_at_position,
                axis=1,
            )
        else:
            raise ValueError("action must be 1D array")

    def estimate_policy_value(
        self,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: np.ndarray,
        position: Optional[np.ndarray] = None,
        **kwargs,
    ) -> float:
        """Estimate the policy value of evaluation policy.

        Parameters
        ----------
        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list)
            Expected rewards given context, action, and position estimated by regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.

        position: array-like, shape (n_rounds,), default=None
            Position of recommendation interface where action was presented in each round of the given logged bandit data.
            When None is given, the effect of position on the reward will be ignored.
            (If only one action is chosen and there is no posion, then you can just ignore this argument.)

        Returns
        ----------
        V_hat: float
            Estimated policy value (performance) of a given evaluation policy.

        """
        check_array(
            array=estimated_rewards_by_reg_model,
            name="estimated_rewards_by_reg_model",
            expected_dim=3,
        )
        check_ope_inputs(
            action_dist=action_dist,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
            position=position,
        )
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)

        return self._estimate_round_rewards(
            position=position,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
            action_dist=action_dist,
        ).mean()

    def estimate_interval(
        self,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: np.ndarray,
        position: Optional[np.ndarray] = None,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 10000,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """Estimate confidence interval of policy value by nonparametric bootstrap procedure.

        Parameters
        ----------
        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list)
            Expected rewards given context, action, and position estimated by regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.

        position: array-like, shape (n_rounds,), default=None
            Position of recommendation interface where action was presented in each round of the given logged bandit data.
            When None is given, the effect of position on the reward will be ignored.
            (If only one action is chosen and there is no posion, then you can just ignore this argument.)

        alpha: float, default=0.05
            Significance level.

        n_bootstrap_samples: int, default=10000
            Number of resampling performed in the bootstrap procedure.

        random_state: int, default=None
            Controls the random seed in bootstrap sampling.

        Returns
        ----------
        estimated_confidence_interval: Dict[str, float]
            Dictionary storing the estimated mean and upper-lower confidence bounds.

        """
        check_array(
            array=estimated_rewards_by_reg_model,
            name="estimated_rewards_by_reg_model",
            expected_dim=3,
        )
        check_ope_inputs(
            action_dist=action_dist,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
            position=position,
        )
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)

        estimated_round_rewards = self._estimate_round_rewards(
            position=position,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
            action_dist=action_dist,
        )
        return estimate_confidence_interval_by_bootstrap(
            samples=estimated_round_rewards,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )


@dataclass
class DoublyRobust(BaseOffPolicyEstimator):
    """Doubly Robust (DR) Estimator.

    Note
    -------
    Similar to DM, DR first learns a supervised machine learning model, such as ridge regression and gradient boosting,
    to estimate the mean reward function (:math:`q(x,a) = \\mathbb{E}[r|x,a]`).
    It then uses it to estimate the policy value as follows.

    .. math::

        \\hat{V}_{\\mathrm{DR}} (\\pi_e; \\mathcal{D}, \\hat{q})
        := \\mathbb{E}_{\\mathcal{D}}[\\hat{q}(x_t,\\pi_e) +  w(x_t,a_t) (r_t - \\hat{q}(x_t,a_t))],

    where :math:`\\mathcal{D}=\\{(x_t,a_t,r_t)\\}_{t=1}^{T}` is logged bandit feedback data with :math:`T` rounds collected by
    a behavior policy :math:`\\pi_b`.
    :math:`w(x,a):=\\pi_e (a|x)/\\pi_b (a|x)` is the importance weight given :math:`x` and :math:`a`.
    :math:`\\mathbb{E}_{\\mathcal{D}}[\\cdot]` is the empirical average over :math:`T` observations in :math:`\\mathcal{D}`.
    :math:`\\hat{q} (x,a)` is an estimated expected reward given :math:`x` and :math:`a`.
    :math:`\\hat{q} (x_t,\\pi):= \\mathbb{E}_{a \\sim \\pi(a|x)}[\\hat{q}(x,a)]` is the expectation of the estimated reward function over :math:`\\pi`.
    When the weight-clipping is applied, a large importance weight is clipped as :math:`\\hat{w}(x,a) := \\min \\{ \\lambda, w(x,a) \\}`
    where :math:`\\lambda (>0)` is a hyperparameter that decides a maximum allowed importance weight.

    To estimate the mean reward function, please use `obp.ope.regression_model.RegressionModel`,
    which supports several fitting methods specific to OPE such as *more robust doubly robust*.

    DR mimics IPW to use a weighted version of rewards, but DR also uses the estimated mean reward
    function (the regression model) as a control variate to decrease the variance.
    It preserves the consistency of IPW if either the importance weight or
    the mean reward estimator is accurate (a property called double robustness).
    Moreover, DR is semiparametric efficient when the mean reward estimator is correctly specified.

    Parameters
    ----------
    lambda_: float, default=np.inf
        A maximum possible value of the importance weight.
        When a positive finite value is given, importance weights larger than `lambda_` will be clipped.
        DoublyRobust with a finite positive `lambda_` corresponds to Doubly Robust with Pessimistic Shrinkage of Su et al.(2020) or CAB-DR of Su et al.(2019).

    estimator_name: str, default='dr'.
        Name of the estimator.

    References
    ----------
    Miroslav Dudík, Dumitru Erhan, John Langford, and Lihong Li.
    "Doubly Robust Policy Evaluation and Optimization.", 2014.

    Mehrdad Farajtabar, Yinlam Chow, and Mohammad Ghavamzadeh.
    "More Robust Doubly Robust Off-policy Evaluation.", 2018.

    Yi Su, Lequn Wang, Michele Santacatterina, and Thorsten Joachims.
    "CAB: Continuous Adaptive Blending Estimator for Policy Evaluation and Learning", 2019.

    Yi Su, Maria Dimakopoulou, Akshay Krishnamurthy, and Miroslav Dudík.
    "Doubly robust off-policy evaluation with shrinkage.", 2020.

    """

    lambda_: float = np.inf
    estimator_name: str = "dr"

    def __post_init__(self) -> None:
        """Initialize Class."""
        check_scalar(
            self.lambda_,
            name="lambda_",
            target_type=(int, float),
            min_val=0.0,
        )
        if self.lambda_ != self.lambda_:
            raise ValueError("lambda_ must not be nan")

    def _estimate_round_rewards(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        pscore: np.ndarray,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: np.ndarray,
        position: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """Estimate round-wise (or sample-wise) rewards.

        Parameters
        ----------
        reward: array-like or Tensor, shape (n_rounds,)
            Reward observed in each round of the logged bandit feedback, i.e., :math:`r_t`.

        action: array-like or Tensor, shape (n_rounds,)
            Action sampled by behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        pscore: array-like or Tensor, shape (n_rounds,)
            Action choice probabilities of behavior policy (propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

        action_dist: array-like or Tensor, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        estimated_rewards_by_reg_model or Tensor: array-like, shape (n_rounds, n_actions, len_list)
            Expected rewards given context, action, and position estimated by regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.

        position: array-like or Tensor, shape (n_rounds,), default=None
            Position of recommendation interface where action was presented in each round of the given logged bandit data.
            When None is given, the effect of position on the reward will be ignored.
            (If only one action is chosen and there is no posion, then you can just ignore this argument.)

        Returns
        ----------
        estimated_rewards: array-like or Tensor, shape (n_rounds,)
            Rewards of each round estimated by the DR estimator.

        """
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)
        n_rounds = action.shape[0]
        iw = action_dist[np.arange(n_rounds), action, position] / pscore
        # weight clipping
        if isinstance(iw, np.ndarray):
            iw = np.minimum(iw, self.lambda_)
        q_hat_at_position = estimated_rewards_by_reg_model[
            np.arange(n_rounds), :, position
        ]
        q_hat_factual = estimated_rewards_by_reg_model[
            np.arange(n_rounds), action, position
        ]
        pi_e_at_position = action_dist[np.arange(n_rounds), :, position]

        if isinstance(reward, np.ndarray):
            estimated_rewards = np.average(
                q_hat_at_position,
                weights=pi_e_at_position,
                axis=1,
            )
        else:
            raise ValueError("reward must be 1D array")

        estimated_rewards += iw * (reward - q_hat_factual)
        return estimated_rewards

    def estimate_policy_value(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        pscore: np.ndarray,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: np.ndarray,
        position: Optional[np.ndarray] = None,
    ) -> float:
        """Estimate the policy value of evaluation policy.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Reward observed in each round of the logged bandit feedback, i.e., :math:`r_t`.

        action: array-like, shape (n_rounds,)
            Action sampled by behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        pscore: array-like, shape (n_rounds,)
            Action choice probabilities of behavior policy (propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list)
            Expected rewards given context, action, and position estimated by regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.

        position: array-like, shape (n_rounds,), default=None
            Position of recommendation interface where action was presented in each round of the given logged bandit data.
            When None is given, the effect of position on the reward will be ignored.
            (If only one action is chosen and there is no posion, then you can just ignore this argument.)

        Returns
        ----------
        V_hat: float
            Policy value estimated by the DR estimator.

        """
        check_array(
            array=estimated_rewards_by_reg_model,
            name="estimated_rewards_by_reg_model",
            expected_dim=3,
        )
        check_array(array=reward, name="reward", expected_dim=1)
        check_array(array=action, name="action", expected_dim=1)
        check_array(array=pscore, name="pscore", expected_dim=1)
        check_ope_inputs(
            action_dist=action_dist,
            position=position,
            action=action,
            reward=reward,
            pscore=pscore,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        )
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)

        return self._estimate_round_rewards(
            reward=reward,
            action=action,
            position=position,
            pscore=pscore,
            action_dist=action_dist,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        ).mean()

    def estimate_interval(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        pscore: np.ndarray,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: np.ndarray,
        position: Optional[np.ndarray] = None,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 10000,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """Estimate confidence interval of policy value by nonparametric bootstrap procedure.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Reward observed in each round of the logged bandit feedback, i.e., :math:`r_t`.

        action: array-like, shape (n_rounds,)
            Action sampled by behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        pscore: array-like, shape (n_rounds,)
            Action choice probabilities of behavior policy (propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list)
            Expected rewards given context, action, and position estimated by regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.

        position: array-like, shape (n_rounds,), default=None
            Position of recommendation interface where action was presented in each round of the given logged bandit data.
            When None is given, the effect of position on the reward will be ignored.
            (If only one action is chosen and there is no posion, then you can just ignore this argument.)

        alpha: float, default=0.05
            Significance level.

        n_bootstrap_samples: int, default=10000
            Number of resampling performed in the bootstrap procedure.

        random_state: int, default=None
            Controls the random seed in bootstrap sampling.

        Returns
        ----------
        estimated_confidence_interval: Dict[str, float]
            Dictionary storing the estimated mean and upper-lower confidence bounds.

        """
        check_array(
            array=estimated_rewards_by_reg_model,
            name="estimated_rewards_by_reg_model",
            expected_dim=3,
        )
        check_array(array=reward, name="reward", expected_dim=1)
        check_array(array=action, name="action", expected_dim=1)
        check_array(array=pscore, name="pscore", expected_dim=1)
        check_ope_inputs(
            action_dist=action_dist,
            position=position,
            action=action,
            reward=reward,
            pscore=pscore,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        )
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)

        estimated_round_rewards = self._estimate_round_rewards(
            reward=reward,
            action=action,
            position=position,
            pscore=pscore,
            action_dist=action_dist,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        )
        return estimate_confidence_interval_by_bootstrap(
            samples=estimated_round_rewards,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )

    def _estimate_mse_score(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        pscore: np.ndarray,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: np.ndarray,
        position: Optional[np.ndarray] = None,
        use_bias_upper_bound: bool = True,
        delta: float = 0.05,
    ) -> float:
        """Estimate the MSE score of a given clipping hyperparameter to conduct hyperparameter tuning.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Reward observed in each round of the logged bandit feedback, i.e., :math:`r_t`.

        action: array-like, shape (n_rounds,)
            Action sampled by behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        pscore: array-like, shape (n_rounds,)
            Action choice probabilities of behavior policy (propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        position: array-like, shape (n_rounds,), default=None
            Position of recommendation interface where action was presented in each round of the given logged bandit data.
            When None is given, the effect of position on the reward will be ignored.
            (If only one action is chosen and there is no posion, then you can just ignore this argument.)

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list)
            Expected rewards given context, action, and position estimated by regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.

        use_bias_upper_bound: bool, default=True
            Whether to use bias upper bound in hyperparameter tuning.
            If False, direct bias estimator is used to estimate the MSE.

        delta: float, default=0.05
            A confidence delta to construct a high probability upper bound based on the Bernstein’s inequality.

        Returns
        ----------
        estimated_mse_score: float
            Estimated MSE score of a given clipping hyperparameter `lambda_`.
            MSE score is the sum of (high probability) upper bound of bias and the sample variance.
            This is estimated using the automatic hyperparameter tuning procedure
            based on Section 5 of Su et al.(2020).

        """
        n_rounds = reward.shape[0]
        # estimate the sample variance of DR with clipping
        sample_variance = np.var(
            self._estimate_round_rewards(
                reward=reward,
                action=action,
                pscore=pscore,
                action_dist=action_dist,
                estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
                position=position,
            )
        )
        sample_variance /= n_rounds

        # estimate the (high probability) upper bound of the bias of DR with clipping
        iw = action_dist[np.arange(n_rounds), action, position] / pscore
        if use_bias_upper_bound:
            bias_term = estimate_high_probability_upper_bound_bias(
                reward=reward,
                iw=iw,
                iw_hat=np.minimum(iw, self.lambda_),
                q_hat=estimated_rewards_by_reg_model[
                    np.arange(n_rounds), action, position
                ],
                delta=delta,
            )
        else:
            bias_term = estimate_bias_in_ope(
                reward=reward,
                iw=iw,
                iw_hat=np.minimum(iw, self.lambda_),
                q_hat=estimated_rewards_by_reg_model[
                    np.arange(n_rounds), action, position
                ],
            )
        estimated_mse_score = sample_variance + (bias_term ** 2)

        return estimated_mse_score


@dataclass
class SelfNormalizedDoublyRobust(DoublyRobust):
    """Self-Normalized Doubly Robust (SNDR) Estimator.

    Note
    -------
    Self-Normalized Doubly Robust estimates the policy value of evaluation policy :math:`\\pi_e` by

    .. math::

        \\hat{V}_{\\mathrm{SNDR}} (\\pi_e; \\mathcal{D}, \\hat{q}) :=
        \\mathbb{E}_{\\mathcal{D}} \\left[\\hat{q}(x_t,\\pi_e) +  \\frac{w(x_t,a_t) (r_t - \\hat{q}(x_t,a_t))}{\\mathbb{E}_{\\mathcal{D}}[ w(x_t,a_t) ]} \\right],

    where :math:`\\mathcal{D}=\\{(x_t,a_t,r_t)\\}_{t=1}^{T}` is logged bandit feedback data with :math:`T` rounds collected by
    a behavior policy :math:`\\pi_b`. :math:`w(x,a):=\\pi_e (a|x)/\\pi_b (a|x)` is the importance weight given :math:`x` and :math:`a`.
    :math:`\\mathbb{E}_{\\mathcal{D}}[\\cdot]` is the empirical average over :math:`T` observations in :math:`\\mathcal{D}`.
    :math:`\\hat{q} (x,a)` is an estimated expected reward given :math:`x` and :math:`a`.
    :math:`\\hat{q} (x_t,\\pi):= \\mathbb{E}_{a \\sim \\pi(a|x)}[\\hat{q}(x,a)]` is the expectation of the estimated reward function over :math:`\\pi`.
    To estimate the mean reward function, please use `obp.ope.regression_model.RegressionModel`.

    Similar to Self-Normalized Inverse Probability Weighting, SNDR estimator applies the self-normalized importance weighting technique to
    increase the stability of the original Doubly Robust estimator.

    Parameters
    ----------
    estimator_name: str, default='sndr'.
        Name of the estimator.

    References
    ----------
    Miroslav Dudík, Dumitru Erhan, John Langford, and Lihong Li.
    "Doubly Robust Policy Evaluation and Optimization.", 2014.

    Nathan Kallus and Masatoshi Uehara.
    "Intrinsically Efficient, Stable, and Bounded Off-Policy Evaluation for Reinforcement Learning.", 2019.

    """

    estimator_name: str = "sndr"

    def _estimate_round_rewards(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        pscore: np.ndarray,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: np.ndarray,
        position: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """Estimate round-wise (or sample-wise) rewards.

        Parameters
        ----------
        reward: array-like or Tensor, shape (n_rounds,)
            Reward observed in each round of the logged bandit feedback, i.e., :math:`r_t`.

        action: array-like or Tensor, shape (n_rounds,)
            Action sampled by behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        pscore: array-like or Tensor, shape (n_rounds,)
            Action choice probabilities of behavior policy (propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

        action_dist: array-like or Tensor, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        estimated_rewards_by_reg_model: array-like or Tensor, shape (n_rounds, n_actions, len_list)
            Expected rewards given context, action, and position estimated by regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.

        position: array-like or Tensor, shape (n_rounds,), default=None
            Position of recommendation interface where action was presented in each round of the given logged bandit data.
            When None is given, the effect of position on the reward will be ignored.
            (If only one action is chosen and there is no posion, then you can just ignore this argument.)

        Returns
        ----------
        estimated_rewards: array-like or Tensor, shape (n_rounds,)
            Rewards of each round estimated by the SNDR estimator.

        """
        n_rounds = action.shape[0]
        iw = action_dist[np.arange(n_rounds), action, position] / pscore
        q_hat_at_position = estimated_rewards_by_reg_model[
            np.arange(n_rounds), :, position
        ]
        pi_e_at_position = action_dist[np.arange(n_rounds), :, position]

        if isinstance(reward, np.ndarray):
            estimated_rewards = np.average(
                q_hat_at_position,
                weights=pi_e_at_position,
                axis=1,
            )
        else:
            raise ValueError("reward must be 1D array")

        q_hat_factual = estimated_rewards_by_reg_model[
            np.arange(n_rounds), action, position
        ]
        estimated_rewards += iw * (reward - q_hat_factual) / iw.mean()
        return estimated_rewards


@dataclass
class SwitchDoublyRobust(DoublyRobust):
    """Switch Doubly Robust (Switch-DR) Estimator.

    Note
    -------
    Switch-DR aims to reduce the variance of the DR estimator by using direct method when the importance weight is large.
    This estimator estimates the policy value of evaluation policy :math:`\\pi_e` by

    .. math::

        \\hat{V}_{\\mathrm{SwitchDR}} (\\pi_e; \\mathcal{D}, \\hat{q}, \\lambda)
        := \\mathbb{E}_{\\mathcal{D}} [\\hat{q}(x_t,\\pi_e) +  w(x_t,a_t) (r_t - \\hat{q}(x_t,a_t)) \\mathbb{I} \\{ w(x_t,a_t) \\le \\lambda \\}],

    where :math:`\\mathcal{D}=\\{(x_t,a_t,r_t)\\}_{t=1}^{T}` is logged bandit feedback data with :math:`T` rounds collected by
    a behavior policy :math:`\\pi_b`. :math:`w(x,a):=\\pi_e (a|x)/\\pi_b (a|x)` is the importance weight given :math:`x` and :math:`a`.
    :math:`\\mathbb{E}_{\\mathcal{D}}[\\cdot]` is the empirical average over :math:`T` observations in :math:`\\mathcal{D}`.
    :math:`\\lambda (\\ge 0)` is a switching hyperparameter, which decides the threshold for the importance weight.
    :math:`\\hat{q} (x,a)` is an estimated expected reward given :math:`x` and :math:`a`.
    :math:`\\hat{q} (x_t,\\pi):= \\mathbb{E}_{a \\sim \\pi(a|x)}[\\hat{q}(x,a)]` is the expectation of the estimated reward function over :math:`\\pi`.
    To estimate the mean reward function, please use `obp.ope.regression_model.RegressionModel`.

    Parameters
    ----------
    lambda_: float, default=np.inf
        Switching hyperparameter. When importance weight is larger than this parameter, DM is applied, otherwise DR is used.
        This hyperparameter should be larger than or equal to 0., otherwise it is meaningless.

    estimator_name: str, default='switch-dr'.
        Name of the estimator.

    References
    ----------
    Miroslav Dudík, Dumitru Erhan, John Langford, and Lihong Li.
    "Doubly Robust Policy Evaluation and Optimization.", 2014.

    Yu-Xiang Wang, Alekh Agarwal, and Miroslav Dudík.
    "Optimal and Adaptive Off-policy Evaluation in Contextual Bandits", 2016.

    Yi Su, Maria Dimakopoulou, Akshay Krishnamurthy, and Miroslav Dudik.
    "Doubly Robust Off-Policy Evaluation with Shrinkage.", 2020.

    """

    lambda_: float = np.inf
    estimator_name: str = "switch-dr"

    def __post_init__(self) -> None:
        """Initialize Class."""
        check_scalar(
            self.lambda_,
            name="lambda_",
            target_type=(int, float),
            min_val=0.0,
        )
        if self.lambda_ != self.lambda_:
            raise ValueError("lambda_ must not be nan")

    def _estimate_round_rewards(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        pscore: np.ndarray,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: np.ndarray,
        position: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """Estimate round-wise (or sample-wise) rewards.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Reward observed in each round of the logged bandit feedback, i.e., :math:`r_t`.

        action: array-like, shape (n_rounds,)
            Action sampled by behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        pscore: array-like, shape (n_rounds,)
            Action choice probabilities of behavior policy (propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list)
            Expected rewards given context, action, and position estimated by regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.

        position: array-like, shape (n_rounds,), default=None
            Position of recommendation interface where action was presented in each round of the given logged bandit data.
            When None is given, the effect of position on the reward will be ignored.
            (If only one action is chosen and there is no posion, then you can just ignore this argument.)

        Returns
        ----------
        estimated_rewards: array-like, shape (n_rounds,)
            Rewards of each round estimated by the Switch-DR estimator.

        """
        n_rounds = action.shape[0]
        iw = action_dist[np.arange(n_rounds), action, position] / pscore
        switch_indicator = np.array(iw <= self.lambda_, dtype=int)
        q_hat_at_position = estimated_rewards_by_reg_model[
            np.arange(n_rounds), :, position
        ]
        q_hat_factual = estimated_rewards_by_reg_model[
            np.arange(n_rounds), action, position
        ]
        pi_e_at_position = action_dist[np.arange(n_rounds), :, position]
        estimated_rewards = np.average(
            q_hat_at_position,
            weights=pi_e_at_position,
            axis=1,
        )
        estimated_rewards += switch_indicator * iw * (reward - q_hat_factual)
        return estimated_rewards

    def _estimate_mse_score(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        pscore: np.ndarray,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: np.ndarray,
        position: Optional[np.ndarray] = None,
        use_bias_upper_bound: bool = False,
        delta: float = 0.05,
    ) -> float:
        """Estimate the MSE score of a given switching hyperparameter to conduct hyperparameter tuning.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Reward observed in each round of the logged bandit feedback, i.e., :math:`r_t`.

        action: array-like, shape (n_rounds,)
            Action sampled by behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        pscore: array-like, shape (n_rounds,)
            Action choice probabilities of behavior policy (propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list)
            Expected rewards given context, action, and position estimated by regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.

        position: array-like, shape (n_rounds,), default=None
            Position of recommendation interface where action was presented in each round of the given logged bandit data.
            When None is given, the effect of position on the reward will be ignored.
            (If only one action is chosen and there is no posion, then you can just ignore this argument.)

        use_bias_upper_bound: bool, default=True
            Whether to use bias upper bound in hyperparameter tuning.
            If False, direct bias estimator is used to estimate the MSE.

        delta: float, default=0.05
            A confidence delta to construct a high probability upper bound based on the Bernstein’s inequality.

        Returns
        ----------
        estimated_mse_score: float
            Estimated MSE score of a given switching hyperparameter `lambda_`.
            MSE score is the sum of (high probability) upper bound of bias and the sample variance.
            This is estimated using the automatic hyperparameter tuning procedure
            based on Section 5 of Su et al.(2020).

        """
        n_rounds = reward.shape[0]
        # estimate the sample variance of Switch-DR (Eq.(8) of Wang et al.(2017))
        sample_variance = np.var(
            self._estimate_round_rewards(
                reward=reward,
                action=action,
                pscore=pscore,
                action_dist=action_dist,
                estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
                position=position,
            )
        )
        sample_variance /= n_rounds

        # estimate the (high probability) upper bound of the bias of Switch-DR
        iw = action_dist[np.arange(n_rounds), action, position] / pscore
        if use_bias_upper_bound:
            bias_term = estimate_high_probability_upper_bound_bias(
                reward=reward,
                iw=iw,
                iw_hat=iw * np.array(iw <= self.lambda_, dtype=int),
                q_hat=estimated_rewards_by_reg_model[
                    np.arange(n_rounds), action, position
                ],
                delta=delta,
            )
        else:
            bias_term = estimate_bias_in_ope(
                reward=reward,
                iw=iw,
                iw_hat=iw * np.array(iw <= self.lambda_, dtype=int),
                q_hat=estimated_rewards_by_reg_model[
                    np.arange(n_rounds), action, position
                ],
            )
        estimated_mse_score = sample_variance + (bias_term ** 2)

        return estimated_mse_score


@dataclass
class DoublyRobustWithShrinkage(DoublyRobust):
    """Doubly Robust with optimistic shrinkage (DRos) Estimator.

    Note
    ------
    DR with (optimistic) shrinkage replaces the importance weight in the original DR estimator with a new weight mapping
    found by directly optimizing sharp bounds on the resulting MSE.

    .. math::

        \\hat{V}_{\\mathrm{DRos}} (\\pi_e; \\mathcal{D}, \\hat{q}, \\lambda)
        := \\mathbb{E}_{\\mathcal{D}} [\\hat{q}(x_t,\\pi_e) +  w_o(x_t,a_t;\\lambda) (r_t - \\hat{q}(x_t,a_t))],

    where :math:`\\mathcal{D}=\\{(x_t,a_t,r_t)\\}_{t=1}^{T}` is logged bandit feedback data with :math:`T` rounds collected by
    a behavior policy :math:`\\pi_b`.
    :math:`w(x,a):=\\pi_e (a|x)/\\pi_b (a|x)` is the importance weight given :math:`x` and :math:`a`.
    :math:`\\hat{q} (x_t,\\pi):= \\mathbb{E}_{a \\sim \\pi(a|x)}[\\hat{q}(x,a)]` is the expectation of the estimated reward function over :math:`\\pi`.
    :math:`\\mathbb{E}_{\\mathcal{D}}[\\cdot]` is the empirical average over :math:`T` observations in :math:`\\mathcal{D}`.
    :math:`\\hat{q} (x,a)` is an estimated expected reward given :math:`x` and :math:`a`.
    To estimate the mean reward function, please use `obp.ope.regression_model.RegressionModel`.

    :math:`w_{o} (x_t,a_t;\\lambda)` is a new weight by the shrinkage technique which is defined as

    .. math::

        w_{o} (x_t,a_t;\\lambda) := \\frac{\\lambda}{w^2(x_t,a_t) + \\lambda} w(x_t,a_t).

    When :math:`\\lambda=0`, we have :math:`w_{o} (x,a;\\lambda)=0` corresponding to the DM estimator.
    In contrast, as :math:`\\lambda \\rightarrow \\infty`, :math:`w_{o} (x,a;\\lambda)` increases and in the limit becomes equal to the original importance weight, corresponding to the standard DR estimator.

    Parameters
    ----------
    lambda_: float
        Shrinkage hyperparameter.
        This hyperparameter should be larger than or equal to 0., otherwise it is meaningless.

    estimator_name: str, default='dr-os'.
        Name of the estimator.

    References
    ----------
    Miroslav Dudík, Dumitru Erhan, John Langford, and Lihong Li.
    "Doubly Robust Policy Evaluation and Optimization.", 2014.

    Yi Su, Maria Dimakopoulou, Akshay Krishnamurthy, and Miroslav Dudik.
    "Doubly Robust Off-Policy Evaluation with Shrinkage.", 2020.

    """

    lambda_: float = 0.0
    estimator_name: str = "dr-os"

    def __post_init__(self) -> None:
        """Initialize Class."""
        check_scalar(
            self.lambda_,
            name="lambda_",
            target_type=(int, float),
            min_val=0.0,
        )
        if self.lambda_ != self.lambda_:
            raise ValueError("lambda_ must not be nan")

    def _estimate_round_rewards(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        pscore: np.ndarray,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: np.ndarray,
        position: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """Estimate round-wise (or sample-wise) rewards.

        Parameters
        ----------
        reward: array-like or Tensor, shape (n_rounds,)
            Reward observed in each round of the logged bandit feedback, i.e., :math:`r_t`.

        action: array-like or Tensor, shape (n_rounds,)
            Action sampled by behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        pscore: array-like or Tensor, shape (n_rounds,)
            Action choice probabilities of behavior policy (propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

        action_dist: array-like or Tensor, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        estimated_rewards_by_reg_model: array-like or Tensor, shape (n_rounds, n_actions, len_list)
            Expected rewards given context, action, and position estimated by regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.

        position: array-like or Tensor, shape (n_rounds,), default=None
            Position of recommendation interface where action was presented in each round of the given logged bandit data.
            When None is given, the effect of position on the reward will be ignored.
            (If only one action is chosen and there is no posion, then you can just ignore this argument.)

        Returns
        ----------
        estimated_rewards: array-like or Tensor, shape (n_rounds,)
            Rewards of each round estimated by the DRos estimator.

        """
        n_rounds = action.shape[0]
        iw = action_dist[np.arange(n_rounds), action, position] / pscore
        if self.lambda_ < np.inf:
            iw_hat = (self.lambda_ * iw) / (iw ** 2 + self.lambda_)
        else:
            iw_hat = iw
        q_hat_at_position = estimated_rewards_by_reg_model[
            np.arange(n_rounds), :, position
        ]
        q_hat_factual = estimated_rewards_by_reg_model[
            np.arange(n_rounds), action, position
        ]
        pi_e_at_position = action_dist[np.arange(n_rounds), :, position]

        if isinstance(reward, np.ndarray):
            estimated_rewards = np.average(
                q_hat_at_position,
                weights=pi_e_at_position,
                axis=1,
            )
        else:
            raise ValueError("reward must be 1D array")

        estimated_rewards += iw_hat * (reward - q_hat_factual)
        return estimated_rewards

    def _estimate_mse_score(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        pscore: np.ndarray,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: np.ndarray,
        position: Optional[np.ndarray] = None,
        use_bias_upper_bound: bool = False,
        delta: float = 0.05,
    ) -> float:
        """Estimate the MSE score of a given shrinkage hyperparameter to conduct hyperparameter tuning.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Reward observed in each round of the logged bandit feedback, i.e., :math:`r_t`.

        action: array-like, shape (n_rounds,)
            Action sampled by behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        pscore: array-like, shape (n_rounds,)
            Action choice probabilities of behavior policy (propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list)
            Expected rewards given context, action, and position estimated by regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.

        position: array-like, shape (n_rounds,), default=None
            Position of recommendation interface where action was presented in each round of the given logged bandit data.

        use_bias_upper_bound: bool, default=True
            Whether to use bias upper bound in hyperparameter tuning.
            If False, direct bias estimator is used to estimate the MSE.

        delta: float, default=0.05
            A confidence delta to construct a high probability upper bound based on the Bernstein’s inequality.

        Returns
        ----------
        estimated_mse_score: float
            Estimated MSE score of a given shrinkage hyperparameter `lambda_`.
            MSE score is the sum of (high probability) upper bound of bias and the sample variance.
            This is estimated using the automatic hyperparameter tuning procedure
            based on Section 5 of Su et al.(2020).

        """
        n_rounds = reward.shape[0]
        # estimate the sample variance of DRos
        sample_variance = np.var(
            self._estimate_round_rewards(
                reward=reward,
                action=action,
                pscore=pscore,
                action_dist=action_dist,
                estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
                position=position,
            )
        )
        sample_variance /= n_rounds

        # estimate the (high probability) upper bound of the bias of DRos
        iw = action_dist[np.arange(n_rounds), action, position] / pscore
        if self.lambda_ < np.inf:
            iw_hat = (self.lambda_ * iw) / (iw ** 2 + self.lambda_)
        else:
            iw_hat = iw
        if use_bias_upper_bound:
            bias_term = estimate_high_probability_upper_bound_bias(
                reward=reward,
                iw=iw,
                iw_hat=iw_hat,
                q_hat=estimated_rewards_by_reg_model[
                    np.arange(n_rounds), action, position
                ],
                delta=0.05,
            )
        else:
            bias_term = estimate_bias_in_ope(
                reward=reward,
                iw=iw,
                iw_hat=iw_hat,
                q_hat=estimated_rewards_by_reg_model[
                    np.arange(n_rounds), action, position
                ],
            )
        estimated_mse_score = sample_variance + (bias_term ** 2)

        return estimated_mse_score

```

```python cellView="form" id="GTufzyQm0omB"
#@markdown off-policy estimator
@dataclass
class OffPolicyEvaluation:
    """Class to conduct OPE by multiple estimators simultaneously.
    Parameters
    -----------
    bandit_feedback: BanditFeedback
        Logged bandit feedback data used to conduct OPE.
    ope_estimators: List[BaseOffPolicyEstimator]
        List of OPE estimators used to evaluate the policy value of evaluation policy.
        Estimators must follow the interface of `obp.ope.BaseOffPolicyEstimator`.
    Examples
    ----------
    .. code-block:: python
        # a case for implementing OPE of the BernoulliTS policy
        # using log data generated by the Random policy
        >>> from obp.dataset import OpenBanditDataset
        >>> from obp.policy import BernoulliTS
        >>> from obp.ope import OffPolicyEvaluation, InverseProbabilityWeighting as IPW
        # (1) Data loading and preprocessing
        >>> dataset = OpenBanditDataset(behavior_policy='random', campaign='all')
        >>> bandit_feedback = dataset.obtain_batch_bandit_feedback()
        >>> bandit_feedback.keys()
        dict_keys(['n_rounds', 'n_actions', 'action', 'position', 'reward', 'pscore', 'context', 'action_context'])
        # (2) Off-Policy Learning
        >>> evaluation_policy = BernoulliTS(
            n_actions=dataset.n_actions,
            len_list=dataset.len_list,
            is_zozotown_prior=True, # replicate the policy in the ZOZOTOWN production
            campaign="all",
            random_state=12345
        )
        >>> action_dist = evaluation_policy.compute_batch_action_dist(
            n_sim=100000, n_rounds=bandit_feedback["n_rounds"]
        )
        # (3) Off-Policy Evaluation
        >>> ope = OffPolicyEvaluation(bandit_feedback=bandit_feedback, ope_estimators=[IPW()])
        >>> estimated_policy_value = ope.estimate_policy_values(action_dist=action_dist)
        >>> estimated_policy_value
        {'ipw': 0.004553...}
        # policy value improvement of BernoulliTS over the Random policy estimated by IPW
        >>> estimated_policy_value_improvement = estimated_policy_value['ipw'] / bandit_feedback['reward'].mean()
        # our OPE procedure suggests that BernoulliTS improves Random by 19.81%
        >>> print(estimated_policy_value_improvement)
        1.198126...
    """

    bandit_feedback: BanditFeedback
    ope_estimators: List[BaseOffPolicyEstimator]

    def __post_init__(self) -> None:
        """Initialize class."""
        for key_ in ["action", "position", "reward", "pscore"]:
            if key_ not in self.bandit_feedback:
                raise RuntimeError(f"Missing key of {key_} in 'bandit_feedback'.")
        self.ope_estimators_ = dict()
        self.is_model_dependent = False
        for estimator in self.ope_estimators:
            self.ope_estimators_[estimator.estimator_name] = estimator
            if isinstance(estimator, DirectMethod) or isinstance(estimator, DoublyRobust):
                self.is_model_dependent = True

    def _create_estimator_inputs(
        self,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: Optional[
            Union[np.ndarray, Dict[str, np.ndarray]]
        ] = None,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Create input dictionary to estimate policy value using subclasses of `BaseOffPolicyEstimator`"""
        check_array(array=action_dist, name="action_dist", expected_dim=3)
        if estimated_rewards_by_reg_model is None:
            pass
        elif isinstance(estimated_rewards_by_reg_model, dict):
            for estimator_name, value in estimated_rewards_by_reg_model.items():
                check_array(
                    array=value,
                    name=f"estimated_rewards_by_reg_model[{estimator_name}]",
                    expected_dim=3,
                )
                if value.shape != action_dist.shape:
                    raise ValueError(
                        f"Expected `estimated_rewards_by_reg_model[{estimator_name}].shape == action_dist.shape`, but found it False."
                    )
        elif estimated_rewards_by_reg_model.shape != action_dist.shape:
            raise ValueError(
                "Expected `estimated_rewards_by_reg_model.shape == action_dist.shape`, but found it False"
            )
        estimator_inputs = {
            estimator_name: {
                input_: self.bandit_feedback[input_]
                for input_ in ["reward", "action", "position", "pscore"]
            }
            for estimator_name in self.ope_estimators_
        }

        for estimator_name in self.ope_estimators_:
            estimator_inputs[estimator_name]["action_dist"] = action_dist
            if isinstance(estimated_rewards_by_reg_model, dict):
                if estimator_name in estimated_rewards_by_reg_model:
                    estimator_inputs[estimator_name][
                        "estimated_rewards_by_reg_model"
                    ] = estimated_rewards_by_reg_model[estimator_name]
                else:
                    estimator_inputs[estimator_name][
                        "estimated_rewards_by_reg_model"
                    ] = None
            else:
                estimator_inputs[estimator_name][
                    "estimated_rewards_by_reg_model"
                ] = estimated_rewards_by_reg_model

        return estimator_inputs

    def estimate_policy_values(
        self,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: Optional[
            Union[np.ndarray, Dict[str, np.ndarray]]
        ] = None,
    ) -> Dict[str, float]:
        """Estimate the policy value of evaluation policy.
        Parameters
        ------------
        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.
        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list) or Dict[str, array-like], default=None
            Expected rewards given each round, action, and position estimated by regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.
            When an array-like is given, all OPE estimators use it.
            When a dict is given, if the dict has the name of a estimator as a key, the corresponding value is used.
            When it is not given, model-dependent estimators such as DM and DR cannot be used.
        Returns
        ----------
        policy_value_dict: Dict[str, float]
            Dictionary containing estimated policy values by OPE estimators.
        """
        if self.is_model_dependent:
            if estimated_rewards_by_reg_model is None:
                raise ValueError(
                    "When model dependent estimators such as DM or DR are used, `estimated_rewards_by_reg_model` must be given"
                )

        policy_value_dict = dict()
        estimator_inputs = self._create_estimator_inputs(
            action_dist=action_dist,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        )
        for estimator_name, estimator in self.ope_estimators_.items():
            policy_value_dict[estimator_name] = estimator.estimate_policy_value(
                **estimator_inputs[estimator_name]
            )

        return policy_value_dict

    def estimate_intervals(
        self,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: Optional[
            Union[np.ndarray, Dict[str, np.ndarray]]
        ] = None,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 100,
        random_state: Optional[int] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Estimate confidence intervals of policy values using nonparametric bootstrap procedure.
        Parameters
        ------------
        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.
        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list) or Dict[str, array-like], default=None
            Expected rewards given context, action, and position estimated by regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.
            When an array-like is given, all OPE estimators use it.
            When a dict is given, if the dict has the name of a estimator as a key, the corresponding value is used.
            When it is not given, model-dependent estimators such as DM and DR cannot be used.
        alpha: float, default=0.05
            Significance level.
        n_bootstrap_samples: int, default=100
            Number of resampling performed in the bootstrap procedure.
        random_state: int, default=None
            Controls the random seed in bootstrap sampling.
        Returns
        ----------
        policy_value_interval_dict: Dict[str, Dict[str, float]]
            Dictionary containing confidence intervals of estimated policy value estimated
            using nonparametric bootstrap procedure.
        """
        if self.is_model_dependent:
            if estimated_rewards_by_reg_model is None:
                raise ValueError(
                    "When model dependent estimators such as DM or DR are used, `estimated_rewards_by_reg_model` must be given"
                )

        check_confidence_interval_arguments(
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )
        policy_value_interval_dict = dict()
        estimator_inputs = self._create_estimator_inputs(
            action_dist=action_dist,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        )
        for estimator_name, estimator in self.ope_estimators_.items():
            policy_value_interval_dict[estimator_name] = estimator.estimate_interval(
                **estimator_inputs[estimator_name],
                alpha=alpha,
                n_bootstrap_samples=n_bootstrap_samples,
                random_state=random_state,
            )

        return policy_value_interval_dict

    def summarize_off_policy_estimates(
        self,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: Optional[
            Union[np.ndarray, Dict[str, np.ndarray]]
        ] = None,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 100,
        random_state: Optional[int] = None,
    ) -> Tuple[DataFrame, DataFrame]:
        """Summarize policy values and their confidence intervals estimated by OPE estimators.
        Parameters
        ------------
        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.
        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list) or Dict[str, array-like], default=None
            Expected rewards given each round, action, and position estimated by regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.
            When an array-like is given, all OPE estimators use it.
            When a dict is given, if the dict has the name of a estimator as a key, the corresponding value is used.
            When it is not given, model-dependent estimators such as DM and DR cannot be used.
        alpha: float, default=0.05
            Significance level.
        n_bootstrap_samples: int, default=100
            Number of resampling performed in the bootstrap procedure.
        random_state: int, default=None
            Controls the random seed in bootstrap sampling.
        Returns
        ----------
        (policy_value_df, policy_value_interval_df): Tuple[DataFrame, DataFrame]
            Policy values and their confidence intervals Estimated by OPE estimators.
        """
        policy_value_df = DataFrame(
            self.estimate_policy_values(
                action_dist=action_dist,
                estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
            ),
            index=["estimated_policy_value"],
        )
        policy_value_interval_df = DataFrame(
            self.estimate_intervals(
                action_dist=action_dist,
                estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
                alpha=alpha,
                n_bootstrap_samples=n_bootstrap_samples,
                random_state=random_state,
            )
        )
        policy_value_of_behavior_policy = self.bandit_feedback["reward"].mean()
        policy_value_df = policy_value_df.T
        if policy_value_of_behavior_policy <= 0:
            logger.warning(
                f"Policy value of the behavior policy is {policy_value_of_behavior_policy} (<=0); relative estimated policy value is set to np.nan"
            )
            policy_value_df["relative_estimated_policy_value"] = np.nan
        else:
            policy_value_df["relative_estimated_policy_value"] = (
                policy_value_df.estimated_policy_value / policy_value_of_behavior_policy
            )
        return policy_value_df, policy_value_interval_df.T

    def visualize_off_policy_estimates(
        self,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: Optional[
            Union[np.ndarray, Dict[str, np.ndarray]]
        ] = None,
        alpha: float = 0.05,
        is_relative: bool = False,
        n_bootstrap_samples: int = 100,
        random_state: Optional[int] = None,
        fig_dir: Optional[Path] = None,
        fig_name: str = "estimated_policy_value.png",
    ) -> None:
        """Visualize policy values estimated by OPE estimators.
        Parameters
        ----------
        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.
        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list) or Dict[str, array-like], default=None
            Expected rewards given context, action, and position estimated by regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.
            When an array-like is given, all OPE estimators use it.
            When a dict is given, if the dict has the name of a estimator as a key, the corresponding value is used.
            When it is not given, model-dependent estimators such as DM and DR cannot be used.
        alpha: float, default=0.05
            Significance level.
        n_bootstrap_samples: int, default=100
            Number of resampling performed in the bootstrap procedure.
        random_state: int, default=None
            Controls the random seed in bootstrap sampling.
        is_relative: bool, default=False,
            If True, the method visualizes the estimated policy values of evaluation policy
            relative to the ground-truth policy value of behavior policy.
        fig_dir: Path, default=None
            Path to store the bar figure.
            If 'None' is given, the figure will not be saved.
        fig_name: str, default="estimated_policy_value.png"
            Name of the bar figure.
        """
        if fig_dir is not None:
            assert isinstance(fig_dir, Path), "fig_dir must be a Path"
        if fig_name is not None:
            assert isinstance(fig_name, str), "fig_dir must be a string"

        estimated_round_rewards_dict = dict()
        estimator_inputs = self._create_estimator_inputs(
            action_dist=action_dist,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        )
        for estimator_name, estimator in self.ope_estimators_.items():
            estimated_round_rewards_dict[
                estimator_name
            ] = estimator._estimate_round_rewards(**estimator_inputs[estimator_name])
        estimated_round_rewards_df = DataFrame(estimated_round_rewards_dict)
        estimated_round_rewards_df.rename(
            columns={key: key.upper() for key in estimated_round_rewards_dict.keys()},
            inplace=True,
        )
        if is_relative:
            estimated_round_rewards_df /= self.bandit_feedback["reward"].mean()

        plt.style.use("ggplot")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(
            data=estimated_round_rewards_df,
            ax=ax,
            ci=100 * (1 - alpha),
            n_boot=n_bootstrap_samples,
            seed=random_state,
        )
        plt.xlabel("OPE Estimators", fontsize=25)
        plt.ylabel(
            f"Estimated Policy Value (± {np.int(100*(1 - alpha))}% CI)", fontsize=20
        )
        plt.yticks(fontsize=15)
        plt.xticks(fontsize=25 - 2 * len(self.ope_estimators))

        if fig_dir:
            fig.savefig(str(fig_dir / fig_name))

    def evaluate_performance_of_estimators(
        self,
        ground_truth_policy_value: float,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: Optional[
            Union[np.ndarray, Dict[str, np.ndarray]]
        ] = None,
        metric: str = "relative-ee",
    ) -> Dict[str, float]:
        """Evaluate estimation performance of OPE estimators.
        Note
        ------
        Evaluate the estimation performance of OPE estimators by relative estimation error (relative-EE) or squared error (SE):
        .. math ::
            \\text{Relative-EE} (\\hat{V}; \\mathcal{D}) = \\left|  \\frac{\\hat{V}(\\pi; \\mathcal{D}) - V(\\pi)}{V(\\pi)} \\right|,
        .. math ::
            \\text{SE} (\\hat{V}; \\mathcal{D}) = \\left(\\hat{V}(\\pi; \\mathcal{D}) - V(\\pi) \\right)^2,
        where :math:`V({\\pi})` is the ground-truth policy value of the evalation policy :math:`\\pi_e` (often estimated using on-policy estimation).
        :math:`\\hat{V}(\\pi; \\mathcal{D})` is an estimated policy value by an OPE estimator :math:`\\hat{V}` and logged bandit feedback :math:`\\mathcal{D}`.
        Parameters
        ----------
        ground_truth policy value: float
            Ground_truth policy value of evaluation policy, i.e., :math:`V(\\pi_e)`.
            With Open Bandit Dataset, we use an on-policy estimate of the policy value as its ground-truth.
        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.
        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list) or Dict[str, array-like], default=None
            Expected rewards given context, action, and position estimated by regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.
            When an array-like is given, all OPE estimators use it.
            When a dict is given, if the dict has the name of a estimator as a key, the corresponding value is used.
            When it is not given, model-dependent estimators such as DM and DR cannot be used.
        metric: str, default="relative-ee"
            Evaluation metric used to evaluate and compare the estimation performance of OPE estimators.
            Must be "relative-ee" or "se".
        Returns
        ----------
        eval_metric_ope_dict: Dict[str, float]
            Dictionary containing evaluation metric for evaluating the estimation performance of OPE estimators.
        """
        check_scalar(
            ground_truth_policy_value,
            "ground_truth_policy_value",
            float,
        )
        if metric not in ["relative-ee", "se"]:
            raise ValueError(
                f"metric must be either 'relative-ee' or 'se', but {metric} is given"
            )
        if metric == "relative-ee" and ground_truth_policy_value == 0.0:
            raise ValueError(
                "ground_truth_policy_value must be non-zero when metric is relative-ee"
            )

        eval_metric_ope_dict = dict()
        estimator_inputs = self._create_estimator_inputs(
            action_dist=action_dist,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        )
        for estimator_name, estimator in self.ope_estimators_.items():
            estimated_policy_value = estimator.estimate_policy_value(
                **estimator_inputs[estimator_name]
            )
            if metric == "relative-ee":
                relative_ee_ = estimated_policy_value - ground_truth_policy_value
                relative_ee_ /= ground_truth_policy_value
                eval_metric_ope_dict[estimator_name] = np.abs(relative_ee_)
            elif metric == "se":
                se_ = (estimated_policy_value - ground_truth_policy_value) ** 2
                eval_metric_ope_dict[estimator_name] = se_
        return eval_metric_ope_dict

    def summarize_estimators_comparison(
        self,
        ground_truth_policy_value: float,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: Optional[
            Union[np.ndarray, Dict[str, np.ndarray]]
        ] = None,
        metric: str = "relative-ee",
    ) -> DataFrame:
        """Summarize performance comparisons of OPE estimators.
        Parameters
        ----------
        ground_truth policy value: float
            Ground_truth policy value of evaluation policy, i.e., :math:`V(\\pi_e)`.
            With Open Bandit Dataset, we use an on-policy estimate of the policy value as ground-truth.
        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.
        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list), default=None
            Expected rewards given context, action, and position estimated by regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.
            When it is not given, model-dependent estimators such as DM and DR cannot be used.
        metric: str, default="relative-ee"
            Evaluation metric used to evaluate and compare the estimation performance of OPE estimators.
            Must be either "relative-ee" or "se".
        Returns
        ----------
        eval_metric_ope_df: DataFrame
            Evaluation metric to evaluate and compare the estimation performance of OPE estimators.
        """
        eval_metric_ope_df = DataFrame(
            self.evaluate_performance_of_estimators(
                ground_truth_policy_value=ground_truth_policy_value,
                action_dist=action_dist,
                estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
                metric=metric,
            ),
            index=[metric],
        )
        return eval_metric_ope_df.T

    def visualize_off_policy_estimates_of_multiple_policies(
        self,
        policy_name_list: List[str],
        action_dist_list: List[np.ndarray],
        estimated_rewards_by_reg_model: Optional[
            Union[np.ndarray, Dict[str, np.ndarray]]
        ] = None,
        alpha: float = 0.05,
        is_relative: bool = False,
        n_bootstrap_samples: int = 100,
        random_state: Optional[int] = None,
        fig_dir: Optional[Path] = None,
        fig_name: str = "estimated_policy_value.png",
    ) -> None:
        """Visualize policy values estimated by OPE estimators.
        Parameters
        ----------
        policy_name_list: List[str]
            List of the names of evaluation policies.
        action_dist_list: List[array-like, shape (n_rounds, n_actions, len_list)]
            List of action choice probabilities by the evaluation policies (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.
        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list) or Dict[str, array-like], default=None
            Expected rewards given context, action, and position estimated by regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.
            When an array-like is given, all OPE estimators use it.
            When a dict is given, if the dict has the name of an estimator as a key, the corresponding value is used.
            When it is not given, model-dependent estimators such as DM and DR cannot be used.
        alpha: float, default=0.05
            Significance level.
        n_bootstrap_samples: int, default=100
            Number of resampling performed in the bootstrap procedure.
        random_state: int, default=None
            Controls the random seed in bootstrap sampling.
        is_relative: bool, default=False,
            If True, the method visualizes the estimated policy values of evaluation policy
            relative to the ground-truth policy value of behavior policy.
        fig_dir: Path, default=None
            Path to store the bar figure.
            If 'None' is given, the figure will not be saved.
        fig_name: str, default="estimated_policy_value.png"
            Name of the bar figure.
        """
        if len(policy_name_list) != len(action_dist_list):
            raise ValueError(
                "the length of policy_name_list must be the same as action_dist_list"
            )
        if fig_dir is not None:
            assert isinstance(fig_dir, Path), "fig_dir must be a Path"
        if fig_name is not None:
            assert isinstance(fig_name, str), "fig_dir must be a string"

        estimated_round_rewards_dict = {
            estimator_name: {} for estimator_name in self.ope_estimators_
        }

        for policy_name, action_dist in zip(policy_name_list, action_dist_list):
            estimator_inputs = self._create_estimator_inputs(
                action_dist=action_dist,
                estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
            )
            for estimator_name, estimator in self.ope_estimators_.items():
                estimated_round_rewards_dict[estimator_name][
                    policy_name
                ] = estimator._estimate_round_rewards(
                    **estimator_inputs[estimator_name]
                )

        plt.style.use("ggplot")
        fig = plt.figure(figsize=(8, 6.2 * len(self.ope_estimators_)))

        for i, estimator_name in enumerate(self.ope_estimators_):
            estimated_round_rewards_df = DataFrame(
                estimated_round_rewards_dict[estimator_name]
            )
            if is_relative:
                estimated_round_rewards_df /= self.bandit_feedback["reward"].mean()

            ax = fig.add_subplot(len(action_dist_list), 1, i + 1)
            sns.barplot(
                data=estimated_round_rewards_df,
                ax=ax,
                ci=100 * (1 - alpha),
                n_boot=n_bootstrap_samples,
                seed=random_state,
            )
            ax.set_title(estimator_name.upper(), fontsize=20)
            ax.set_ylabel(
                f"Estimated Policy Value (± {np.int(100*(1 - alpha))}% CI)", fontsize=20
            )
            plt.yticks(fontsize=15)
            plt.xticks(fontsize=25 - 2 * len(policy_name_list))

        if fig_dir:
            fig.savefig(str(fig_dir / fig_name))
```

<!-- #region id="zLDfki-FyOw2" -->
## Policy
<!-- #endregion -->

```python id="HjERr2N2ydAW"
@dataclass
class BaseOfflinePolicyLearner(metaclass=ABCMeta):
    """Base class for off-policy learners.
    Parameters
    -----------
    n_actions: int
        Number of actions.
    len_list: int, default=1
        Length of a list of actions recommended in each impression.
        When Open Bandit Dataset is used, 3 should be set.
    """

    n_actions: int
    len_list: int = 1

    def __post_init__(self) -> None:
        """Initialize class."""
        check_scalar(self.n_actions, "n_actions", int, min_val=2)
        check_scalar(self.len_list, "len_list", int, min_val=1, max_val=self.n_actions)

    @property
    def policy_type(self) -> PolicyType:
        """Type of the bandit policy."""
        return PolicyType.OFFLINE

    @abstractmethod
    def fit(
        self,
    ) -> None:
        """Fits an offline bandit policy using the given logged bandit feedback data."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, context: np.ndarray) -> np.ndarray:
        """Predict best action for new data.
        Parameters
        -----------
        context: array-like, shape (n_rounds_of_new_data, dim_context)
            Context vectors for new data.
        Returns
        -----------
        action: array-like, shape (n_rounds_of_new_data, n_actions, len_list)
            Action choices by a policy trained by calling the `fit` method.
        """
        raise NotImplementedError
```

```python cellView="form" id="wRWI4rL7yc8p"
#@markdown IPWLearner Policy

@dataclass
class IPWLearner(BaseOfflinePolicyLearner):
    """Off-policy learner with Inverse Probability Weighting.
    Parameters
    -----------
    n_actions: int
        Number of actions.
    len_list: int, default=1
        Length of a list of actions recommended in each impression.
        When Open Bandit Dataset is used, 3 should be set.
    base_classifier: ClassifierMixin
        Machine learning classifier used to train an offline decision making policy.
    References
    ------------
    Miroslav Dudík, Dumitru Erhan, John Langford, and Lihong Li.
    "Doubly Robust Policy Evaluation and Optimization.", 2014.
    Damien Lefortier, Adith Swaminathan, Xiaotao Gu, Thorsten Joachims, and Maarten de Rijke.
    "Large-scale Validation of Counterfactual Learning Methods: A Test-Bed.", 2016.
    """

    base_classifier: Optional[ClassifierMixin] = None

    def __post_init__(self) -> None:
        """Initialize class."""
        super().__post_init__()
        if self.base_classifier is None:
            self.base_classifier = LogisticRegression(random_state=12345)
        else:
            if not is_classifier(self.base_classifier):
                raise ValueError("base_classifier must be a classifier")
        self.base_classifier_list = [
            clone(self.base_classifier) for _ in np.arange(self.len_list)
        ]

    @staticmethod
    def _create_train_data_for_opl(
        context: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        pscore: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create training data for off-policy learning.
        Parameters
        -----------
        context: array-like, shape (n_rounds, dim_context)
            Context vectors in each round, i.e., :math:`x_t`.
        action: array-like, shape (n_rounds,)
            Action sampled by behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.
        reward: array-like, shape (n_rounds,)
            Observed rewards (or outcome) in each round, i.e., :math:`r_t`.
        pscore: array-like, shape (n_rounds,), default=None
            Propensity scores, the probability of selecting each action by behavior policy,
            in the given logged bandit data.
        Returns
        --------
        (X, sample_weight, y): Tuple[np.ndarray, np.ndarray, np.ndarray]
            Feature vectors, sample weights, and outcome for training the base machine learning model.
        """
        return context, (reward / pscore), action

    def fit(
        self,
        context: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        pscore: Optional[np.ndarray] = None,
        position: Optional[np.ndarray] = None,
    ) -> None:
        """Fits an offline bandit policy using the given logged bandit feedback data.
        Note
        --------
        This `fit` method trains a deterministic policy :math:`\\pi: \\mathcal{X} \\rightarrow \\mathcal{A}`
        via a cost-sensitive classification reduction as follows:
        .. math::
            \\hat{\\pi}
            & \\in \\arg \\max_{\\pi \\in \\Pi} \\hat{V}_{\\mathrm{IPW}} (\\pi ; \\mathcal{D}) \\\\
            & = \\arg \\max_{\\pi \\in \\Pi} \\mathbb{E}_{\\mathcal{D}} \\left[\\frac{\\mathbb{I} \\{\\pi (x_{i})=a_{i} \\}}{\\pi_{b}(a_{i} | x_{i})} r_{i} \\right] \\\\
            & = \\arg \\min_{\\pi \\in \\Pi} \\mathbb{E}_{\\mathcal{D}} \\left[\\frac{r_i}{\\pi_{b}(a_{i} | x_{i})} \\mathbb{I} \\{\\pi (x_{i}) \\neq a_{i} \\} \\right],
        where :math:`\\mathbb{E}_{\\mathcal{D}} [\cdot]` is the empirical average over observations in :math:`\\mathcal{D}`.
        See the reference for the details.
        Parameters
        -----------
        context: array-like, shape (n_rounds, dim_context)
            Context vectors in each round, i.e., :math:`x_t`.
        action: array-like, shape (n_rounds,)
            Action sampled by behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.
        reward: array-like, shape (n_rounds,)
            Observed rewards (or outcome) in each round, i.e., :math:`r_t`.
        pscore: array-like, shape (n_rounds,), default=None
            Action choice probabilities of behavior policy (propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.
        position: array-like, shape (n_rounds,), default=None
            Position of recommendation interface where action was presented in each round of the given logged bandit data.
            If None is given, a learner assumes that there is only one position.
            When `len_list` > 1, position has to be set.
        """
        check_bandit_feedback_inputs(
            context=context,
            action=action,
            reward=reward,
            pscore=pscore,
            position=position,
        )
        if (reward < 0).any():
            raise ValueError(
                "A negative value is found in `reward`."
                "`obp.policy.IPWLearner` cannot handle negative rewards,"
                "and please use `obp.policy.NNPolicyLearner` instead."
            )
        if pscore is None:
            n_actions = np.int(action.max() + 1)
            pscore = np.ones_like(action) / n_actions
        if self.len_list == 1:
            position = np.zeros_like(action, dtype=int)
        else:
            if position is None:
                raise ValueError("When `self.len_list=1`, `position` must be given.")

        for position_ in np.arange(self.len_list):
            X, sample_weight, y = self._create_train_data_for_opl(
                context=context[position == position_],
                action=action[position == position_],
                reward=reward[position == position_],
                pscore=pscore[position == position_],
            )
            self.base_classifier_list[position_].fit(
                X=X, y=y, sample_weight=sample_weight
            )

    def predict(self, context: np.ndarray) -> np.ndarray:
        """Predict best actions for new data.
        Note
        --------
        Action set predicted by this `predict` method can contain duplicate items.
        If you want a non-repetitive action set, then please use the `sample_action` method.
        Parameters
        -----------
        context: array-like, shape (n_rounds_of_new_data, dim_context)
            Context vectors for new data.
        Returns
        -----------
        action_dist: array-like, shape (n_rounds_of_new_data, n_actions, len_list)
            Action choices by a classifier, which can contain duplicate items.
            If you want a non-repetitive action set, please use the `sample_action` method.
        """
        check_array(array=context, name="context", expected_dim=2)

        n_rounds = context.shape[0]
        action_dist = np.zeros((n_rounds, self.n_actions, self.len_list))
        for position_ in np.arange(self.len_list):
            predicted_actions_at_position = self.base_classifier_list[
                position_
            ].predict(context)
            action_dist[
                np.arange(n_rounds),
                predicted_actions_at_position,
                np.ones(n_rounds, dtype=int) * position_,
            ] += 1
        return action_dist

    def predict_score(self, context: np.ndarray) -> np.ndarray:
        """Predict non-negative scores for all possible products of action and position.
        Parameters
        -----------
        context: array-like, shape (n_rounds_of_new_data, dim_context)
            Context vectors for new data.
        Returns
        -----------
        score_predicted: array-like, shape (n_rounds_of_new_data, n_actions, len_list)
            Scores for all possible pairs of action and position predicted by a classifier.
        """
        check_array(array=context, name="context", expected_dim=2)

        n_rounds = context.shape[0]
        score_predicted = np.zeros((n_rounds, self.n_actions, self.len_list))
        for position_ in np.arange(self.len_list):
            score_predicteds_at_position = self.base_classifier_list[
                position_
            ].predict_proba(context)
            score_predicted[:, :, position_] = score_predicteds_at_position
        return score_predicted

    def sample_action(
        self,
        context: np.ndarray,
        tau: Union[int, float] = 1.0,
        random_state: Optional[int] = None,
    ) -> np.ndarray:
        """Sample (non-repetitive) actions based on scores predicted by a classifier.
        Note
        --------
        This `sample_action` method samples a **non-repetitive** set of actions for new data :math:`x \\in \\mathcal{X}`
        by first computing non-negative scores for all possible candidate products of action and position
        :math:`(a, k) \\in \\mathcal{A} \\times \\mathcal{K}` (where :math:`\\mathcal{A}` is an action set and
        :math:`\\mathcal{K}` is a position set), and using softmax function as follows:
        .. math::
            & P (A_1 = a_1 | x) = \\frac{\\mathrm{exp}(f(x,a_1,1) / \\tau)}{\\sum_{a^{\\prime} \\in \\mathcal{A}} \\mathrm{exp}( f(x,a^{\\prime},1) / \\tau)} , \\\\
            & P (A_2 = a_2 | A_1 = a_1, x) = \\frac{\\mathrm{exp}(f(x,a_2,2) / \\tau)}{\\sum_{a^{\\prime} \\in \\mathcal{A} \\backslash \\{a_1\\}} \\mathrm{exp}(f(x,a^{\\prime},2) / \\tau )} ,
            \\ldots
        where :math:`A_k` is a random variable representing an action at a position :math:`k`.
        :math:`\\tau` is a temperature hyperparameter.
        :math:`f: \\mathcal{X} \\times \\mathcal{A} \\times \\mathcal{K} \\rightarrow \\mathbb{R}_{+}`
        is a scoring function which is now implemented in the `predict_score` method.
        Parameters
        ----------------
        context: array-like, shape (n_rounds_of_new_data, dim_context)
            Context vectors for new data.
        tau: int or float, default=1.0
            A temperature parameter, controlling the randomness of the action choice.
            As :math:`\\tau \\rightarrow \\infty`, the algorithm will select arms uniformly at random.
        random_state: int, default=None
            Controls the random seed in sampling actions.
        Returns
        -----------
        action: array-like, shape (n_rounds_of_new_data, n_actions, len_list)
            Action sampled by a trained classifier.
        """
        check_array(array=context, name="context", expected_dim=2)
        check_scalar(tau, name="tau", target_type=(int, float), min_val=0)

        n_rounds = context.shape[0]
        random_ = check_random_state(random_state)
        action = np.zeros((n_rounds, self.n_actions, self.len_list))
        score_predicted = self.predict_score(context=context)
        for i in tqdm(np.arange(n_rounds), desc="[sample_action]", total=n_rounds):
            action_set = np.arange(self.n_actions)
            for position_ in np.arange(self.len_list):
                score_ = softmax(score_predicted[i, action_set, position_] / tau)
                action_sampled = random_.choice(action_set, p=score_, replace=False)
                action[i, action_sampled, position_] = 1
                action_set = np.delete(action_set, action_set == action_sampled)
        return action

    def predict_proba(
        self,
        context: np.ndarray,
        tau: Union[int, float] = 1.0,
    ) -> np.ndarray:
        """Obtains action choice probabilities for new data based on scores predicted by a classifier.
        Note
        --------
        This `predict_proba` method obtains action choice probabilities for new data :math:`x \\in \\mathcal{X}`
        by first computing non-negative scores for all possible candidate actions
        :math:`a \\in \\mathcal{A}` (where :math:`\\mathcal{A}` is an action set),
        and using a Plackett-Luce ranking model as follows:
        .. math::
            P (A = a | x) = \\frac{\\mathrm{exp}(f(x,a) / \\tau)}{\\sum_{a^{\\prime} \\in \\mathcal{A}} \\mathrm{exp}(f(x,a^{\\prime}) / \\tau)},
        where :math:`A` is a random variable representing an action, and :math:`\\tau` is a temperature hyperparameter.
        :math:`f: \\mathcal{X} \\times \\mathcal{A} \\rightarrow \\mathbb{R}_{+}`
        is a scoring function which is now implemented in the `predict_score` method.
        **Note that this method can be used only when `len_list=1`, please use the `sample_action` method otherwise.**
        Parameters
        ----------------
        context: array-like, shape (n_rounds_of_new_data, dim_context)
            Context vectors for new data.
        tau: int or float, default=1.0
            A temperature parameter, controlling the randomness of the action choice.
            As :math:`\\tau \\rightarrow \\infty`, the algorithm will select arms uniformly at random.
        Returns
        -----------
        choice_prob: array-like, shape (n_rounds_of_new_data, n_actions, len_list)
            Action choice probabilities obtained by a trained classifier.
        """
        assert (
            self.len_list == 1
        ), "predict_proba method cannot be used when `len_list != 1`"
        check_array(array=context, name="context", expected_dim=2)
        check_scalar(tau, name="tau", target_type=(int, float), min_val=0)

        score_predicted = self.predict_score(context=context)
        choice_prob = softmax(score_predicted / tau, axis=1)
        return choice_prob
```

<!-- #region id="NVGmw7LrnkjN" -->
## Main
<!-- #endregion -->

```python id="Llj-xWcAnqqz"
# from obp.dataset import linear_behavior_policy
# from obp.dataset import logistic_reward_function
# from obp.dataset import SyntheticBanditDataset
# from obp.ope import DirectMethod
# from obp.ope import DoublyRobust
# from obp.ope import DoublyRobustWithShrinkage
# from obp.ope import InverseProbabilityWeighting
# from obp.ope import OffPolicyEvaluation
# from obp.ope import RegressionModel
# from obp.ope import SelfNormalizedDoublyRobust
# from obp.ope import SelfNormalizedInverseProbabilityWeighting
# from obp.ope import SwitchDoublyRobust
# from obp.policy import IPWLearner
```

<!-- #region id="vIFAFZLNn0bx" -->
### Hyperparams
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="rtjAGcXTnNGl" executionInfo={"status": "ok", "timestamp": 1632118263894, "user_tz": -330, "elapsed": 75, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="5ed32f66-b5b1-4980-b721-fa6a96fbcdde"
%%writefile hyperparams.yaml
lightgbm:
  n_estimators: 30
  learning_rate: 0.01
  max_depth: 5
  min_samples_leaf: 10
  random_state: 12345
logistic_regression:
  max_iter: 10000
  C: 100
  random_state: 12345
random_forest:
  n_estimators: 30
  max_depth: 5
  min_samples_leaf: 10
  random_state: 12345
```

```python id="j83Xy-9bn2k6"
# hyperparameters of the regression model used in model dependent OPE estimators
with open("hyperparams.yaml", "rb") as f:
    hyperparams = yaml.safe_load(f)
```

<!-- #region id="yI9sfhIXoBfU" -->
### Base Models
<!-- #endregion -->

```python id="ZyCKvJT_oBDW"
base_model_dict = dict(
    logistic_regression=LogisticRegression,
    lightgbm=GradientBoostingClassifier,
    random_forest=RandomForestClassifier,
)
```

<!-- #region id="WTLlloiaoLyI" -->
### OPE Estimators
<!-- #endregion -->

```python id="BpcgNZv9oOp_"
# compared OPE estimators
ope_estimators = [
    DirectMethod(),
    InverseProbabilityWeighting(),
    SelfNormalizedInverseProbabilityWeighting(),
    DoublyRobust(),
    SelfNormalizedDoublyRobust(),
    SwitchDoublyRobust(lambda_=1.0, estimator_name="switch-dr (lambda=1)"),
    SwitchDoublyRobust(lambda_=100.0, estimator_name="switch-dr (lambda=100)"),
    DoublyRobustWithShrinkage(lambda_=1.0, estimator_name="dr-os (lambda=1)"),
    DoublyRobustWithShrinkage(lambda_=100.0, estimator_name="dr-os (lambda=100)"),
]
```

<!-- #region id="BAWQv_YtpOjm" -->
### Arg Parse
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="yHozhJsHpR33" executionInfo={"status": "ok", "timestamp": 1632118263901, "user_tz": -330, "elapsed": 57, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="77e15fbb-6776-4fb1-addc-66383840a652"
    parser = argparse.ArgumentParser(
        description="evaluate off-policy estimators with synthetic bandit data."
    )
    parser.add_argument(
        "--n_runs", type=int, default=1, help="number of simulations in the experiment."
    )
    parser.add_argument(
        "--n_rounds",
        type=int,
        default=10000,
        help="number of rounds for synthetic bandit feedback.",
    )
    parser.add_argument(
        "--n_actions",
        type=int,
        default=10,
        help="number of actions for synthetic bandit feedback.",
    )
    parser.add_argument(
        "--dim_context",
        type=int,
        default=5,
        help="dimensions of context vectors characterizing each round.",
    )
    parser.add_argument(
        "--base_model_for_evaluation_policy",
        type=str,
        choices=["logistic_regression", "lightgbm", "random_forest"],
        default='random_forest',
        help="base ML model for evaluation policy, logistic_regression, random_forest or lightgbm.",
    )
    parser.add_argument(
        "--base_model_for_reg_model",
        type=str,
        choices=["logistic_regression", "lightgbm", "random_forest"],
        default='logistic_regression',
        help="base ML model for regression model, logistic_regression, random_forest or lightgbm.",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=2,
        help="the maximum number of concurrently running jobs.",
    )
    parser.add_argument("--random_state", type=int, default=12345)
    args = parser.parse_args(args={})
    print(args)
```

```python id="aEbvz_7VpuE9"
# configurations
n_runs = args.n_runs
n_rounds = args.n_rounds
n_actions = args.n_actions
dim_context = args.dim_context
base_model_for_evaluation_policy = args.base_model_for_evaluation_policy
base_model_for_reg_model = args.base_model_for_reg_model
n_jobs = args.n_jobs
random_state = args.random_state
```

<!-- #region id="s5VxXYFopxns" -->
### Process
<!-- #endregion -->

```python id="3BDjUIP8py9d"
def process(i: int):
    # synthetic data generator
    dataset = SyntheticBanditDataset(
        n_actions=n_actions,
        dim_context=dim_context,
        reward_function=logistic_reward_function,
        behavior_policy_function=linear_behavior_policy,
        random_state=i,
    )
    # define evaluation policy using IPWLearner
    evaluation_policy = IPWLearner(
        n_actions=dataset.n_actions,
        base_classifier=base_model_dict[base_model_for_evaluation_policy](
            **hyperparams[base_model_for_evaluation_policy]
        ),
    )
    # sample new training and test sets of synthetic logged bandit feedback
    bandit_feedback_train = dataset.obtain_batch_bandit_feedback(n_rounds=n_rounds)
    bandit_feedback_test = dataset.obtain_batch_bandit_feedback(n_rounds=n_rounds)
    # train the evaluation policy on the training set of the synthetic logged bandit feedback
    evaluation_policy.fit(
        context=bandit_feedback_train["context"],
        action=bandit_feedback_train["action"],
        reward=bandit_feedback_train["reward"],
        pscore=bandit_feedback_train["pscore"],
    )
    # predict the action decisions for the test set of the synthetic logged bandit feedback
    action_dist = evaluation_policy.predict(
        context=bandit_feedback_test["context"],
    )
    # estimate the mean reward function of the test set of synthetic bandit feedback with ML model
    regression_model = RegressionModel(
        n_actions=dataset.n_actions,
        action_context=dataset.action_context,
        base_model=base_model_dict[base_model_for_reg_model](
            **hyperparams[base_model_for_reg_model]
        ),
    )
    estimated_rewards_by_reg_model = regression_model.fit_predict(
        context=bandit_feedback_test["context"],
        action=bandit_feedback_test["action"],
        reward=bandit_feedback_test["reward"],
        n_folds=3,  # 3-fold cross-fitting
        random_state=random_state,
    )
    # evaluate estimators' performances using relative estimation error (relative-ee)
    ope = OffPolicyEvaluation(
        bandit_feedback=bandit_feedback_test,
        ope_estimators=ope_estimators,
    )
    relative_ee_i = ope.evaluate_performance_of_estimators(
        ground_truth_policy_value=dataset.calc_ground_truth_policy_value(
            expected_reward=bandit_feedback_test["expected_reward"],
            action_dist=action_dist,
        ),
        action_dist=action_dist,
        estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
    )

    return relative_ee_i
```

<!-- #region id="O5apkt21p5cW" -->
### Run
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="7KYAR6Z_nanI" executionInfo={"status": "ok", "timestamp": 1632118266121, "user_tz": -330, "elapsed": 2260, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="0ff50898-0f7d-4831-feea-34bb27a2a5ae"
processed = Parallel(
    n_jobs=n_jobs,
    verbose=50,
)([delayed(process)(i) for i in np.arange(n_runs)])
relative_ee_dict = {est.estimator_name: dict() for est in ope_estimators}
for i, relative_ee_i in enumerate(processed):
    for (
        estimator_name,
        relative_ee_,
    ) in relative_ee_i.items():
        relative_ee_dict[estimator_name][i] = relative_ee_
relative_ee_df = DataFrame(relative_ee_dict).describe().T.round(6)

print("=" * 45)
print(f"random_state={random_state}")
print("-" * 45)
print(relative_ee_df[["mean", "std"]])
print("=" * 45)
```

```python id="brtzYQVKp2oK"
# save results of the evaluation of off-policy estimators in './logs' directory.
log_path = Path("./logs")
log_path.mkdir(exist_ok=True, parents=True)
relative_ee_df.to_csv(log_path / "relative_ee_of_ope_estimators.csv")
```
