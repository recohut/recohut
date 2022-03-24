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

<!-- #region id="KKN-9w0bLG5U" -->
# Evaluating the Robustness of Off-Policy Evaluation
<!-- #endregion -->

<!-- #region id="8fZr3WnuLFqI" -->
## Setup
<!-- #endregion -->

```python id="aIsmJfbS739X"
!pip install -q obp
```

```python id="Pe-0qKN9BsYv"
!pip install matplotlib==3.1.1
```

```python id="klvJDHIcCPEz"
!pip install -U pandas
```

<!-- #region id="Omda8tuDqjkd" -->
## Imports
<!-- #endregion -->

```python id="O8GhX4Mr7jp0" executionInfo={"status": "ok", "timestamp": 1633531025565, "user_tz": -330, "elapsed": 1790, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
from abc import ABCMeta, abstractmethod
from typing import Union
from scipy.stats import loguniform

from inspect import isclass
from pathlib import Path
from typing import Optional, List, Tuple, Union, Dict
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection._search import BaseSearchCV
from sklearn.metrics import (
    roc_auc_score,
    log_loss,
    mean_squared_error as calc_mse,
    mean_absolute_error as calc_mae,
)
from obp.ope import (
    RegressionModel,
    OffPolicyEvaluation,
    BaseOffPolicyEstimator,
)
from obp.types import BanditFeedback

import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier as RandomForest
from sklearn.model_selection import RandomizedSearchCV

import obp
from obp.dataset import (
    SyntheticBanditDataset,
    logistic_reward_function,
    linear_behavior_policy
)
from obp.policy import IPWLearner
from obp.ope import (
    DirectMethod,
    DoublyRobust,
    DoublyRobustWithShrinkage,
    InverseProbabilityWeighting,
)
from obp.dataset import MultiClassToBanditReduction
```

```python id="RdPPTf39GUny"
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
```

<!-- #region id="aKsAy3S87vB9" -->
## Utils
<!-- #endregion -->

```python id="ZZNJjEpH7wGb" executionInfo={"status": "ok", "timestamp": 1633531025566, "user_tz": -330, "elapsed": 6, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
def _choose_uniform(
    s: int,
    lower: Union[int, float],
    upper: Union[int, float],
    type_: type,
) -> Union[int, float]:
    np.random.seed(seed=s)
    assert lower <= upper, "`upper` must be larger than or equal to `lower`"
    assert type_ in [int, float], f"`type_` must be int or float but {type_} is given"
    if lower == upper:
        return lower
    if type_ == int:
        return np.random.randint(lower, upper, dtype=type_)
    else:  # type_ == float:
        return np.random.uniform(lower, upper)


def _choose_log_uniform(
    s: int,
    lower: Union[int, float],
    upper: Union[int, float],
    type_: type,
) -> Union[int, float]:
    assert (
        lower > 0
    ), f"`lower` must be greater than 0 when drawing from log uniform distribution but {lower} is given"
    assert lower <= upper, "`upper` must be larger than or equal to `lower`"
    assert type_ in [int, float], f"`type_` must be int or float but {type_} is given"
    if lower == upper:
        return lower
    if type_ == int:
        return int(loguniform.rvs(lower, upper, random_state=s))
    else:  # type_ == float:
        return loguniform.rvs(lower, upper, random_state=s)
```

<!-- #region id="G4WkQNyr7kJb" -->
## OPE Evaluators
<!-- #endregion -->

```python id="An3QvViM7osQ" executionInfo={"status": "ok", "timestamp": 1633531025567, "user_tz": -330, "elapsed": 6, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
class BaseOPEEvaluator(metaclass=ABCMeta):
    @abstractmethod
    def estimate_policy_value(self) -> None:
        """Estimate policy values."""
        raise NotImplementedError

    @abstractmethod
    def calculate_squared_error(self) -> None:
        """Calculate squared errors."""
        raise NotImplementedError

    @abstractmethod
    def visualize_cdf(self) -> None:
        """Create graph of cumulative distribution function of an estimator."""
        raise NotImplementedError

    @abstractmethod
    def visualize_cdf_aggregate(self) -> None:
        """Create graph of cumulative distribution function of all estimators."""
        raise NotImplementedError

    @abstractmethod
    def save_policy_value(self) -> None:
        """Save estimate policy values to csv file."""
        raise NotImplementedError

    @abstractmethod
    def save_squared_error(self) -> None:
        """Save squared errors to csv file."""
        raise NotImplementedError

    @abstractmethod
    def calculate_au_cdf_score(self) -> None:
        """Calculate AU-CDF score."""
        raise NotImplementedError

    @abstractmethod
    def calculate_cvar_score(self) -> None:
        """Calculate CVaR score."""
        raise NotImplementedError
```

```python id="bF-Fjzis8G5V" executionInfo={"status": "ok", "timestamp": 1633531029291, "user_tz": -330, "elapsed": 3730, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
COLORS = [
    "lightcoral",
    "plum",
    "lightgreen",
    "lightskyblue",
    "lightsalmon",
    "orange",
    "forestgreen",
    "royalblue",
    "gold",
    "blueviolet",
    "fuchsia",
    "lightpink",
    "firebrick",
    "peru",
    "darkkhaki",
    "darkolivegreen",
    "navy",
    "deeppink",
    "black",
    "silver",
]

LINESTYLES = [
    "solid",
    (0, (1, 0.6)),
    (0, (1, 1.2)),
    (0, (1, 1.8)),
    (0, (1, 2.4)),
    (0, (1, 3)),
]


@dataclass
class InterpretableOPEEvaluator(BaseOPEEvaluator):
    """Class to carry out Interpretable OPE Evaluation.
    Parameters
    ----------
    random_states: np.ndarray
        list of integers representing random states
        length of random_states corresponds to the number of runs
    bandit_feedbacks: List[BanditFeedback]
        list of bandit feedbacks
    evaluation_policies: List[Tuple[float, np.ndarray]]
        list of tuples representing evaluation policies
        first entry in tuple represents the ground truth policy value
        second entry in tuple represents action distribution of evaluation policy
    ope_estimators: List[BaseOffPolicyEstimator]
        list of ope estimators from obp.ope
    ope_estimator_hyperparams: dict
        dictionary storing hyperparameters for ope estimators
        must be in the following format
            ope_estimator_hyperparams = dict(
                [OffPolicyEstimator].estimator_name = dict(
                    [parameter_name] = dict(
                        "lower":
                        "upper":
                        "log":
                        "type":
                    )
                ),
            )
    regression_models: Optional[List[Union[BaseEstimator, BaseSearchCV]]]
        list of regression models to be used in off policy evaluation
        each element must either be of type BaseEstimator or BaseSearchCV
    regression_model_hyperparams: dict
        dictionary storing hyperparameters for regression models
        must be in the following format
            regression_model_hyperparams = dict(
                [model_name] = dict(
                    [parameter_name] = dict(
                        "lower":
                        "upper":
                        "log":
                        "type":
                    )
                ),
            )
    pscore_estimators: Optional[List[Union[BaseEstimator, BaseSearchCV]]]
        list of classification models to be used in estimating propensity scores of behavior policy
        each element must either be of type BaseEstimator or BaseSearchCV
    pscore_estimator_hyperparams: dict
        dictionary storing hyperparameters for pscore estimators
        must be in the following format
            pscore_estimator_hyperparams = dict(
                [model_name] = dict(
                    [parameter_name] = dict(
                        "lower":
                        "upper":
                        "log":
                        "type":
                    )
                ),
            )
    """

    random_states: np.ndarray
    ope_estimators: List[BaseOffPolicyEstimator]
    bandit_feedbacks: List[BanditFeedback]
    evaluation_policies: List[Tuple[float, np.ndarray]]
    ope_estimator_hyperparams: Optional[dict] = None
    regression_models: Optional[List[Union[BaseEstimator, BaseSearchCV]]] = None
    regression_model_hyperparams: Optional[dict] = None
    pscore_estimators: Optional[List[Union[BaseEstimator, BaseSearchCV]]] = None
    pscore_estimator_hyperparams: Optional[dict] = None

    def __post_init__(self) -> None:
        self.estimator_names = [est.estimator_name for est in self.ope_estimators]
        self.policy_value = None
        for i in np.arange(len(self.bandit_feedbacks)):
            if self.bandit_feedbacks[i]["position"] is None:
                self.bandit_feedbacks[i]["position"] = np.zeros_like(
                    self.bandit_feedbacks[i]["action"],
                    dtype=int,
                )
        if self.reward_type == "binary":
            self.reg_model_metric_names = ["auc", "rel_ce"]
        else:
            self.reg_model_metric_names = ["rel_mse", "rel_mae"]

        if not self.ope_estimator_hyperparams:
            self.ope_estimator_hyperparams = {
                estimator_name: dict() for estimator_name in self.estimator_names
            }

        if not self.regression_model_hyperparams:
            self.regression_model_hyperparams = {
                regression_model: dict() for regression_model in self.regression_models
            }

        if self.pscore_estimators and not self.pscore_estimator_hyperparams:
            self.pscore_estimator_hyperparams = {
                pscore_estimator: dict() for pscore_estimator in self.pscore_estimators
            }

    @property
    def n_runs(self) -> int:
        """Number of iterations."""
        return self.random_states.shape[0]

    @property
    def n_rounds(self) -> np.ndarray:
        """Number of observations in each given bandit_feedback in self.bandit_feedbacks"""
        return np.asarray(
            [bandit_feedback["n_rounds"] for bandit_feedback in self.bandit_feedbacks]
        )

    @property
    def n_actions(self) -> np.ndarray:
        """Number of actions in each given bandit_feedback in self.bandit_feedbacks"""
        return np.asarray(
            [bandit_feedback["n_actions"] for bandit_feedback in self.bandit_feedbacks]
        )

    @property
    def reward_type(self) -> np.ndarray:
        """Whether the reward is binary or continuous"""
        if np.unique(self.bandit_feedbacks[0]["reward"]).shape[0] == 2:
            return "binary"
        else:
            return "continuous"

    @property
    def len_list(self) -> np.ndarray:
        """Number of positions in each given bandit_feedback in self.bandit_feedbacks"""
        return np.asarray(
            [
                int(bandit_feedback["position"].max() + 1)
                for bandit_feedback in self.bandit_feedbacks
            ]
        )

    def estimate_policy_value(
        self,
        n_folds_: Union[int, Optional[dict]] = 2,
        sample_size: Optional[int] = None,
    ) -> dict:
        """Estimates the policy values using selected ope estimators under a range of environments."""
        # initialize dictionaries to store results
        self.policy_value = {est: np.zeros(self.n_runs) for est in self.estimator_names}
        self.squared_error = {
            est: np.zeros(self.n_runs) for est in self.estimator_names
        }
        self.reg_model_metrics = {
            metric: np.zeros(self.n_runs) for metric in self.reg_model_metric_names
        }
        for i, s in enumerate(tqdm(self.random_states)):
            np.random.seed(seed=s)
            # randomly select bandit_feedback
            self.bandit_feedback = self._choose_bandit_feedback(s)

            if self.pscore_estimators is not None:
                # randomly choose pscore estimator
                pscore_estimator = np.random.choice(self.pscore_estimators)
                # randomly choose hyperparameters of pscore estimator
                if isinstance(pscore_estimator, BaseEstimator):
                    classifier = pscore_estimator
                    setattr(classifier, "random_state", s)
                elif isclass(pscore_estimator) and issubclass(
                    pscore_estimator, BaseEstimator
                ):
                    pscore_estimator_hyperparam = (
                        self._choose_pscore_estimator_hyperparam(s, pscore_estimator)
                    )
                    classifier = clone(pscore_estimator(**pscore_estimator_hyperparam))
                else:
                    raise ValueError(
                        f"pscore_estimator must be BaseEstimator or BaseSearchCV, but {type(pscore_estimator)} is given."
                    )
                # fit classifier
                classifier.fit(
                    self.bandit_feedback["context"], self.bandit_feedback["action"]
                )
                estimated_pscore = classifier.predict_proba(
                    self.bandit_feedback["context"]
                )
                # replace pscore in bootstrap bandit feedback with estimated pscore
                self.bandit_feedback["pscore"] = estimated_pscore[
                    np.arange(self.bandit_feedback["n_rounds"]),
                    self.bandit_feedback["action"],
                ]

            # randomly sample from selected bandit_feedback
            bootstrap_bandit_feedback = self._sample_bootstrap_bandit_feedback(
                s, sample_size
            )
            # randomly choose hyperparameters of ope estimators
            self._choose_ope_estimator_hyperparam(s)
            # randomly choose regression model
            regression_model = self._choose_regression_model(s)
            # randomly choose hyperparameters of regression models
            if isinstance(regression_model, BaseEstimator):
                setattr(regression_model, "random_state", s)
            elif isclass(regression_model) and issubclass(
                regression_model, BaseEstimator
            ):
                regression_model_hyperparam = self._choose_regression_model_hyperparam(
                    s, regression_model
                )
                regression_model = regression_model(**regression_model_hyperparam)
            else:
                raise ValueError(
                    f"regression_model must be BaseEstimator or BaseSearchCV, but {type(regression_model)} is given."
                )
            # randomly choose evaluation policy
            ground_truth, bootstrap_action_dist = self._choose_evaluation_policy(s)
            # randomly choose number of folds
            if isinstance(n_folds_, dict):
                n_folds = _choose_uniform(
                    s,
                    n_folds_["lower"],
                    n_folds_["upper"],
                    n_folds_["type"],
                )
            else:
                n_folds = n_folds_
            # estimate policy value using each ope estimator under setting s
            (
                policy_value_s,
                estimated_rewards_by_reg_model_s,
            ) = self._estimate_policy_value_s(
                s,
                bootstrap_bandit_feedback,
                regression_model,
                bootstrap_action_dist,
                n_folds,
            )
            # calculate squared error for each ope estimator
            squared_error_s = self._calculate_squared_error_s(
                policy_value_s,
                ground_truth,
            )
            # evaluate the performance of reg_model
            r_pred = estimated_rewards_by_reg_model_s[
                np.arange(bootstrap_bandit_feedback["n_rounds"]),
                bootstrap_bandit_feedback["action"],
                bootstrap_bandit_feedback["position"],
            ]
            reg_model_metrics = self._calculate_rec_model_performance_s(
                r_true=bootstrap_bandit_feedback["reward"],
                r_pred=r_pred,
            )
            # store results
            for est in self.estimator_names:
                self.policy_value[est][i] = policy_value_s[est]
                self.squared_error[est][i] = squared_error_s[est]
            for j, metric in enumerate(self.reg_model_metric_names):
                self.reg_model_metrics[metric][i] = reg_model_metrics[j].mean()
        return self.policy_value

    def calculate_squared_error(self) -> dict:
        """Calculates the squared errors using selected ope estimators under a range of environments."""
        if not self.policy_value:
            _ = self.estimate_policy_value()
        return self.squared_error

    def calculate_variance(self, scale: bool = False, std: bool = True) -> dict:
        """Calculates the variance of squared errors."""
        if not self.policy_value:
            _ = self.estimate_policy_value()
        if std:
            self.variance = {
                key: np.sqrt(np.var(val)) for key, val in self.squared_error.items()
            }
        else:
            self.variance = {
                key: np.var(val) for key, val in self.squared_error.items()
            }
        variance = self.variance.copy()

        if scale:
            c = min(variance.values())
            for est in self.estimator_names:
                if type(variance[est]) != str:
                    variance[est] = variance[est] / c
        return variance

    def calculate_mean(self, scale: bool = False, root: bool = False) -> dict:
        """Calculates the mean of squared errors."""
        if not self.policy_value:
            _ = self.estimate_policy_value()
        if root:  # root mean squared error
            self.mean = {
                key: np.sqrt(np.mean(val)) for key, val in self.squared_error.items()
            }
        else:  # mean squared error
            self.mean = {key: np.mean(val) for key, val in self.squared_error.items()}
        mean = self.mean.copy()

        if scale:
            c = min(mean.values())
            for est in self.estimator_names:
                if type(mean[est]) != str:
                    mean[est] = mean[est] / c
        return mean

    def save_policy_value(
        self,
        file_dir: str = "results",
        file_name: str = "ieoe_policy_value.csv",
    ) -> None:
        """Save policy_value to csv file."""
        path = Path(file_dir)
        path.mkdir(exist_ok=True, parents=True)
        ieoe_policy_value_df = pd.DataFrame(self.policy_value, self.random_states)
        ieoe_policy_value_df.to_csv(f"{file_dir}/{file_name}")

    def save_squared_error(
        self,
        file_dir: str = "results",
        file_name: str = "ieoe_squared_error.csv",
    ) -> None:
        """Save squared_error to csv file."""
        path = Path(file_dir)
        path.mkdir(exist_ok=True, parents=True)
        ieoe_squared_error_df = pd.DataFrame(self.squared_error, self.random_states)
        ieoe_squared_error_df.to_csv(f"{file_dir}/{file_name}")

    def save_variance(
        self,
        file_dir: str = "results",
        file_name: str = "ieoe_variance.csv",
    ) -> None:
        """Save squared_error to csv file."""
        path = Path(file_dir)
        path.mkdir(exist_ok=True, parents=True)
        ieoe_variance_df = pd.DataFrame(self.variance.values(), self.variance.keys())
        ieoe_variance_df.to_csv(f"{file_dir}/{file_name}")

    def visualize_cdf(
        self,
        fig_dir: str = "figures",
        fig_name: str = "cdf.png",
        font_size: int = 12,
        fig_width: float = 8,
        fig_height: float = 6,
        kde: Optional[bool] = False,
    ) -> None:
        """Create a cdf graph for each ope estimator."""
        path = Path(fig_dir)
        path.mkdir(exist_ok=True, parents=True)
        for est in self.estimator_names:
            plt.clf()
            plt.style.use("ggplot")
            plt.rcParams.update({"font.size": font_size})
            _, ax = plt.subplots(figsize=(fig_width, fig_height))
            if kde:
                sns.kdeplot(
                    x=self.squared_error[est],
                    kernel="gaussian",
                    cumulative=True,
                    ax=ax,
                    label=est,
                    linewidth=3.0,
                    bw_method=0.05,
                )
            else:
                sns.ecdfplot(
                    self.squared_error[est],
                    ax=ax,
                    label=est,
                    linewidth=3.0,
                )
            plt.legend()
            plt.title(f"{est}: Cumulative distribution of squared error")
            plt.xlabel("Squared error")
            plt.ylabel("Cumulative probability")
            plt.xlim(0, None)
            plt.ylim(0, 1.1)
            plt.savefig(f"{fig_dir}/{est}_{fig_name}", dpi=100)
            plt.show()

    def visualize_cdf_aggregate(
        self,
        fig_dir: str = "figures",
        fig_name: str = "cdf.png",
        font_size: int = 12,
        fig_width: float = 8,
        fig_height: float = 6,
        xmax: Optional[float] = None,
        kde: Optional[bool] = False,
        linestyles: Optional[bool] = False,
    ) -> None:
        """Create a graph containing the cdf of all ope estimators."""
        path = Path(fig_dir)
        path.mkdir(exist_ok=True, parents=True)
        plt.clf()
        plt.style.use("ggplot")
        plt.rcParams.update({"font.size": font_size})
        _, ax = plt.subplots(figsize=(fig_width, fig_height))
        for i, est in enumerate(self.estimator_names):
            if i < len(COLORS):
                color = COLORS[i]
            else:
                color = np.random.rand(
                    3,
                )
            if linestyles:
                linestyle = LINESTYLES[i % len(LINESTYLES)]
            else:
                linestyle = "solid"
            if kde:
                sns.kdeplot(
                    x=self.squared_error[est],
                    kernel="gaussian",
                    cumulative=True,
                    ax=ax,
                    label=est,
                    linewidth=3.0,
                    bw_method=0.05,
                    alpha=0.7,
                    c=color,
                    linestyle=linestyle,
                )
            else:
                sns.ecdfplot(
                    self.squared_error[est],
                    ax=ax,
                    label=est,
                    linewidth=3.0,
                    alpha=0.7,
                    c=color,
                    linestyle=linestyle,
                )
        plt.legend(loc="lower right")
        plt.title("Cumulative distribution of squared error")
        plt.xlabel("Squared error")
        plt.ylabel("Cumulative probability")
        plt.xlim(0, xmax)
        plt.ylim(0, 1.1)
        plt.savefig(f"{fig_dir}/{fig_name}", dpi=100)
        plt.show()

    def visualize_squared_error_density(
        self,
        fig_dir: str = "figures",
        fig_name: str = "squared_error_density_estimation.png",
        font_size: int = 12,
        fig_width: float = 8,
        fig_height: float = 6,
    ) -> None:
        """Create a graph based on kernel density estimation of squared error for each ope estimator."""
        path = Path(fig_dir)
        path.mkdir(exist_ok=True, parents=True)
        for est in self.estimator_names:
            plt.clf()
            plt.style.use("ggplot")
            plt.rcParams.update({"font.size": font_size})
            _, ax = plt.subplots(figsize=(fig_width, fig_height))
            sns.kdeplot(
                self.squared_error[est],
                ax=ax,
                label=est,
                linewidth=3.0,
            )
            plt.legend()
            plt.title(f"{est}: Graph of estimated density of squared error")
            plt.xlabel(
                "Squared error",
            )
            plt.savefig(f"{fig_dir}/{est}_{fig_name}", dpi=100)
            plt.show()

    def calculate_au_cdf_score(
        self,
        threshold: float,
        scale: bool = False,
    ) -> dict:
        """Calculate AU-CDF score."""
        au_cdf_score = {est: None for est in self.estimator_names}
        for est in self.estimator_names:
            au_cdf_score[est] = np.mean(
                np.clip(threshold - self.squared_error[est], 0, None)
            )
        if scale:
            c = max(au_cdf_score.values())
            for est in self.estimator_names:
                au_cdf_score[est] = au_cdf_score[est] / c
        return au_cdf_score

    def calculate_cvar_score(
        self,
        alpha: float,
        scale: bool = False,
    ) -> dict:
        """Calculate CVaR score."""
        cvar_score = {est: None for est in self.estimator_names}
        for est in self.estimator_names:
            threshold = np.percentile(self.squared_error[est], alpha)
            bool_ = self.squared_error[est] >= threshold
            if any(bool_):
                cvar_score[est] = np.sum(self.squared_error[est] * bool_) / np.sum(
                    bool_
                )
            else:
                cvar_score[
                    est
                ] = f"the largest squared error is less than the threshold value {threshold}"
        if scale:
            c = min(cvar_score.values())
            for est in self.estimator_names:
                if type(cvar_score[est]) != str:
                    cvar_score[est] = cvar_score[est] / c
        return cvar_score

    def set_ope_estimator_hyperparam_space(
        self,
        ope_estimator_name: str,
        param_name: str,
        lower: Union[int, float],
        upper: Union[int, float],
        log: Optional[bool] = False,
        type_: Optional[type] = int,
    ) -> None:
        """Specify sampling method of hyperparameter of ope estimator."""
        assert type_ in [
            int,
            float,
        ], f"`type_` must be int or float but {type_} is given"
        dic = {
            "lower": lower,
            "upper": upper,
            "log": log,
            "type": type_,
        }
        self.ope_estimator_hyperparams[ope_estimator_name][param_name] = dic

    def set_regression_model_hyperparam_space(
        self,
        regression_model: Union[BaseEstimator, BaseSearchCV],
        param_name: str,
        lower: Union[int, float],
        upper: Union[int, float],
        log: Optional[bool] = False,
        type_: Optional[type] = int,
    ) -> None:
        """Specify sampling method of hyperparameter of regression model."""
        assert type_ in [
            int,
            float,
        ], f"`type_` must be int or float but {type_} is given"
        dic = {
            "lower": lower,
            "upper": upper,
            "log": log,
            "type": type_,
        }
        self.regression_model_hyperparams[regression_model][param_name] = dic

    def _choose_bandit_feedback(
        self,
        s: int,
    ) -> BanditFeedback:
        """Randomly select bandit_feedback."""
        np.random.seed(seed=s)
        idx = np.random.choice(len(self.bandit_feedbacks))
        return self.bandit_feedbacks[idx]

    def _sample_bootstrap_bandit_feedback(
        self, s: int, sample_size: Optional[int]
    ) -> BanditFeedback:
        """Randomly sample bootstrap data from bandit_feedback."""
        bootstrap_bandit_feedback = self.bandit_feedback.copy()
        np.random.seed(seed=s)
        if sample_size is None:
            sample_size = self.bandit_feedback["n_rounds"]
        self.bootstrap_idx = np.random.choice(
            np.arange(sample_size), size=sample_size, replace=True
        )
        for key_ in self.bandit_feedback.keys():
            # if the size of a certain key_ is not equal to n_rounds,
            # we should not resample that certain key_
            # e.g. we want to resample action and reward, but not n_rounds
            if (
                not isinstance(self.bandit_feedback[key_], np.ndarray)
                or len(self.bandit_feedback[key_]) != self.bandit_feedback["n_rounds"]
            ):
                continue
            bootstrap_bandit_feedback[key_] = bootstrap_bandit_feedback[key_][
                self.bootstrap_idx
            ]
        bootstrap_bandit_feedback["n_rounds"] = sample_size
        return bootstrap_bandit_feedback

    def _choose_ope_estimator_hyperparam(
        self,
        s: int,
    ) -> None:
        """Randomly choose hyperparameters for ope estimators."""
        for i, est in enumerate(self.ope_estimators):
            hyperparam = self.ope_estimator_hyperparams.get(est.estimator_name, None)
            if not hyperparam:
                continue
            for p in hyperparam:
                if hyperparam[p].get("log", False):
                    val = _choose_log_uniform(
                        s,
                        hyperparam[p]["lower"],
                        hyperparam[p]["upper"],
                        hyperparam[p].get("type", int),
                    )
                else:
                    val = _choose_uniform(
                        s,
                        hyperparam[p]["lower"],
                        hyperparam[p]["upper"],
                        hyperparam[p].get("type", int),
                    )
                setattr(est, p, val)
            self.ope_estimators[i] = est

    def _choose_regression_model(
        self,
        s: int,
    ) -> Union[BaseEstimator, BaseSearchCV]:
        """Randomly choose regression model."""
        idx = np.random.choice(len(self.regression_models))
        return self.regression_models[idx]

    def _choose_regression_model_hyperparam(
        self,
        s: int,
        regression_model: Union[BaseEstimator, BaseSearchCV],
    ) -> dict:
        """Randomly choose hyperparameters for regression model."""
        hyperparam = dict(
            random_state=s,
        )
        hyperparam_set = self.regression_model_hyperparams.get(regression_model, None)
        if not hyperparam_set:
            return hyperparam
        for p in hyperparam_set:
            if hyperparam_set[p].get("log", False):
                val = _choose_log_uniform(
                    s,
                    hyperparam_set[p]["lower"],
                    hyperparam_set[p]["upper"],
                    hyperparam_set[p].get("type", int),
                )
            else:
                val = _choose_uniform(
                    s,
                    hyperparam_set[p]["lower"],
                    hyperparam_set[p]["upper"],
                    hyperparam_set[p].get("type", int),
                )
            hyperparam[p] = val
        return hyperparam

    def _choose_pscore_estimator_hyperparam(
        self,
        s: int,
        pscore_estimator: Union[BaseEstimator, BaseSearchCV],
    ) -> dict:
        """Randomly choose hyperparameters for pscore estimator."""
        hyperparam = dict(
            random_state=s,
        )
        hyperparam_set = self.pscore_estimator_hyperparams.get(pscore_estimator, None)
        if not hyperparam_set:
            return hyperparam
        for p in hyperparam_set:
            if hyperparam_set[p].get("log", False):
                val = _choose_log_uniform(
                    s,
                    hyperparam_set[p]["lower"],
                    hyperparam_set[p]["upper"],
                    hyperparam_set[p].get("type", int),
                )
            else:
                val = _choose_uniform(
                    s,
                    hyperparam_set[p]["lower"],
                    hyperparam_set[p]["upper"],
                    hyperparam_set[p].get("type", int),
                )
            hyperparam[p] = val
        return hyperparam

    def _choose_evaluation_policy(
        self,
        s: int,
    ) -> Tuple[float, np.ndarray]:
        """Randomly choose evaluation policy and resample using bootstrap."""
        np.random.seed(seed=s)
        idx = np.random.choice(len(self.evaluation_policies))
        ground_truth, action_dist = self.evaluation_policies[idx]
        action_dist = action_dist[self.bootstrap_idx]
        return ground_truth, action_dist

    def _estimate_policy_value_s(
        self,
        s: int,
        bootstrap_bandit_feedback: BanditFeedback,
        _regression_model: Union[BaseEstimator, BaseSearchCV],
        bootstrap_action_dist: np.ndarray,
        n_folds: int,
    ) -> Tuple[dict, np.ndarray]:
        """Estimates the policy values using selected ope estimators under a particular environments."""
        # prepare regression model for ope
        regression_model = RegressionModel(
            n_actions=self.bandit_feedback["n_actions"],
            len_list=int(self.bandit_feedback["position"].max() + 1),
            base_model=_regression_model,
            fitting_method="normal",
        )
        estimated_reward_by_reg_model = regression_model.fit_predict(
            context=bootstrap_bandit_feedback["context"],
            action=bootstrap_bandit_feedback["action"],
            reward=bootstrap_bandit_feedback["reward"],
            position=bootstrap_bandit_feedback["position"],
            pscore=bootstrap_bandit_feedback["pscore"],
            action_dist=bootstrap_action_dist,
            n_folds=n_folds,
            random_state=int(s),
        )

        # estimate policy value using ope
        ope = OffPolicyEvaluation(
            bandit_feedback=bootstrap_bandit_feedback,
            ope_estimators=self.ope_estimators,
        )
        estimated_policy_value = ope.estimate_policy_values(
            action_dist=bootstrap_action_dist,
            estimated_rewards_by_reg_model=estimated_reward_by_reg_model,
        )

        return estimated_policy_value, estimated_reward_by_reg_model

    def _calculate_squared_error_s(
        self,
        policy_value: dict,
        ground_truth: float,
    ) -> dict:
        """Calculate squared error."""
        squared_error = {
            est: np.square(policy_value[est] - ground_truth)
            for est in self.estimator_names
        }
        return squared_error

    def _calculate_rec_model_performance_s(
        self,
        r_true: np.ndarray,
        r_pred: np.ndarray,
    ) -> Tuple[float, float]:
        """Calculate performance of reg model."""
        r_naive_pred = np.ones_like(r_true) * r_true.mean()
        if self.reward_type == "binary":
            auc = roc_auc_score(r_true, r_pred)
            ce = log_loss(r_true, r_pred)
            ce_naive = log_loss(r_true, r_naive_pred)
            rel_ce = 1 - (ce / ce_naive)
            return auc, rel_ce

        elif self.reward_type == "continuous":
            mse = calc_mse(r_true, r_pred)
            mse_naive = calc_mse(r_true, r_naive_pred)
            rel_mse = 1 - (mse / mse_naive)
            mae = calc_mae(r_true, r_pred)
            mae_naive = calc_mae(r_true, r_naive_pred)
            rel_mae = 1 - (mae / mse_naive)
            return rel_mse, rel_mae

    def load_squared_error(
        self,
        file_dir: str,
        file_name: str,
    ) -> None:
        df = pd.read_csv(f"{file_dir}/{file_name}")
        self.squared_error = {est: None for est in self.estimator_names}
        for est in self.estimator_names:
            self.squared_error[est] = df[est].values
```

<!-- #region id="ZkHWycdN8L3j" -->
## Example 1 - Synthetic dataset
<!-- #endregion -->

<!-- #region id="-2MW7AG28mcW" -->
This section demonstrates an example of conducting Interpretable Evaluation for Off-Policy Evaluation (IEOE). We use synthetic logged bandit feedback data generated using [`obp`](https://github.com/st-tech/zr-obp) and evaluate the performance of Direct Method (DM), Doubly Robust (DR), Doubly Robust with Shrinkage (DRos), and Inverse Probability Weighting (IPW).

Our example contains the following three major steps:

1. Data Preparation
2. Setting Hyperparameter Spaces for Off-Policy Evaluation
3. Interpretable Evaluation for Off-Policy Evaluation
<!-- #endregion -->

<!-- #region id="8LIsXGoo8mcc" -->
### Data Preparation

In order to conduct IEOE using `pyieoe`, we need to prepare logged bandit feedback data, action distributions of evaluation policies, and ground truth policy values of evaluation policies. Because `pyieoe` is built with the intention of being used with `obp`, these inputs must follow the conventions in `obp`. Specifically, logged bandit feedback data must be of type `BanditFeedback`, action distributions must be of type `np.ndarray`, and ground truth policy values must be of type `float` (or `int`). 

In this example, we generate synthetic logged bandit feedback data and perform off-policy learning to obtain two sets of evaluation policies along with their action distributions and ground truth policy values using `obp`. For a detailed explanation of this process, please refer to the [official obp example](https://github.com/st-tech/zr-obp/blob/master/examples/quickstart/quickstart_synthetic.ipynb).
<!-- #endregion -->

```python id="JqPDekwo8mcd" executionInfo={"status": "ok", "timestamp": 1633531029293, "user_tz": -330, "elapsed": 11, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
# generate a synthetic bandit dataset with 10 actions
# we use `logistic function` as the reward function and `linear_behavior_policy` as the behavior policy.
# one can define their own reward function and behavior policy such as nonlinear ones. 
dataset = SyntheticBanditDataset(
    n_actions=10,
    dim_context=5,
    reward_type="binary", # "binary" or "continuous"
    reward_function=logistic_reward_function,
    behavior_policy_function=linear_behavior_policy,
    random_state=12345
)
# obtain training and test sets of synthetic logged bandit feedback
n_rounds_train, n_rounds_test = 10000, 10000
bandit_feedback_train = dataset.obtain_batch_bandit_feedback(n_rounds=n_rounds_train)
bandit_feedback_test = dataset.obtain_batch_bandit_feedback(n_rounds=n_rounds_test)

# define IPWLearner with Logistic Regression as its base ML model
evaluation_policy_a = IPWLearner(
    n_actions=dataset.n_actions,
    len_list=dataset.len_list,
    base_classifier=LogisticRegression(C=100, random_state=12345)
)
# train IPWLearner on the training set of the synthetic logged bandit feedback
evaluation_policy_a.fit(
    context=bandit_feedback_train["context"],
    action=bandit_feedback_train["action"],
    reward=bandit_feedback_train["reward"],
    pscore=bandit_feedback_train["pscore"]
)
# obtains action choice probabilities for the test set of the synthetic logged bandit feedback
action_dist_a = evaluation_policy_a.predict_proba(
    context=bandit_feedback_test["context"],
    tau=0.1 # temperature hyperparameter
)

# define IPWLearner with Random Forest as its base ML model
evaluation_policy_b = IPWLearner(
    n_actions=dataset.n_actions,
    len_list=dataset.len_list,
    base_classifier=RandomForest(n_estimators=30, min_samples_leaf=10, random_state=12345)
)
# train IPWLearner on the training set of the synthetic logged bandit feedback
evaluation_policy_b.fit(
    context=bandit_feedback_train["context"],
    action=bandit_feedback_train["action"],
    reward=bandit_feedback_train["reward"],
    pscore=bandit_feedback_train["pscore"]
)
# obtains action choice probabilities for the test set of the synthetic logged bandit feedback
action_dist_b = evaluation_policy_b.predict_proba(
    context=bandit_feedback_test["context"],
    tau=0.1 # temperature hyperparameter
)

# obtain ground truth policy value for each action choice probabilities
expected_rewards = bandit_feedback_test["expected_reward"]
ground_truth_a = np.average(expected_rewards, weights=action_dist_a[:, :, 0], axis=1).mean()
ground_truth_b = np.average(expected_rewards, weights=action_dist_b[:, :, 0], axis=1).mean()
```

<!-- #region id="q9ekJZa38mch" -->
### Setting Hyperparameter Spaces for Off-Policy Evaluation

An integral aspect of IEOE is the different sources of variance. The main sources of variance are evaluation policies, random states, hyperparameters of OPE estimators, and hyperparameters of regression models. 

In this step, we define the spaces from which the hyperparameters of OPE estimators / regression models are chosen. (The evaluation policy space is defined in the previous step, and the random state space will be defined in the next step.)
<!-- #endregion -->

```python id="39YU1CiI8mci" executionInfo={"status": "ok", "timestamp": 1633531029293, "user_tz": -330, "elapsed": 8, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
# set hyperparameter space for ope estimators

# set hyperparameter space for the doubly robust with shrinkage estimator
# with the following code, lambda_ will be chosen from a logarithm uniform distribution over the interval [0.001, 1000]
lambda_ = {
    "lower": 1e-3,
    "upper": 1e3,
    "log": True,
    "type": float
}
dros_param = {"lambda_": lambda_}
```

```python id="MghLFZFN8mck" executionInfo={"status": "ok", "timestamp": 1633531029294, "user_tz": -330, "elapsed": 8, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
# set hyperparameter space for regression models

# set hyperparameter space for logistic regression
# with the following code, C will be chosen from a logarithm uniform distribution over the interval [0.001, 100]
C = {
    "lower": 1e-3,
    "upper": 1e2,
    "log": True,
    "type": float
}
# create a dictionary mapping hyperparamter names to hyperparamter spaces
logistic_regression_param = {"C": C}

# set hyperparameter space for the random forest classifier
# with the following code, n_estimators will be chosen from a logarithm uniform distribution over the interval [50, 100]
# the chosen value will be of type int
n_estimators = {
    "lower": 5e1,
    "upper": 1e2,
    "log": True,
    "type": int
}
# with the following code, max_depth will be chosen from a uniform distribution over the interval [2, 10]
# the chosen value will be of type int
max_depth = {
    "lower": 2,
    "upper": 10,
    "log": False,
    "type": int
}
# with the following code, min_samples_split will be chosen from a uniform distribution over the interval [2, 10]
# the chosen value will be of type int
min_samples_split = {
    "lower": 2,
    "upper": 10,
    "log": False,
    "type": int
}
# create a dictionary mapping hyperparamter names to hyperparamter spaces
random_forest_param = {
    "n_estimators": n_estimators, 
    "max_depth": max_depth, 
    "min_samples_split": min_samples_split
}
```

<!-- #region id="as9sJoQ18mck" -->
### Interpretable Evaluation for Off-Policy Evaluation

With the above steps completed, we can finally conduct IEOE by utilizing the `InterpretableOPEEvaluator` class.

Here is a brief description for each parameter that can be passed into `InterpretableOPEEvaluator`:

- `random_states`: a list of integers representing the random_state used when performing OPE; corresponds to the number of iterations
- `bandit_feedback`: a list of logged bandit feedback data
- `evaluation_policies`: a list of tuples representing (ground truth policy value, action distribution)
- `ope_estimators`: a list of OPE ope_estimators
- `ope_estimator_hyperparams`: a dictionary mapping OPE estimator names to OPE estimator hyperparameter spaces defined in step 2
- `regression_models`: a list of regression_models
- `regression_model_hyperparams`: a dictionary mapping regression models to regression model hyperparameter spaces defined in step 2
<!-- #endregion -->

```python id="9zS8aK0j8mcl" executionInfo={"status": "ok", "timestamp": 1633531030670, "user_tz": -330, "elapsed": 6, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
# initializing class
evaluator = InterpretableOPEEvaluator(
    random_states=np.arange(1000),
    bandit_feedbacks=[bandit_feedback_test],
    evaluation_policies=[
        (ground_truth_a, action_dist_a), 
        (ground_truth_b, action_dist_b)
    ],
    ope_estimators=[
        DirectMethod(),
        DoublyRobust(),
        DoublyRobustWithShrinkage(),
        InverseProbabilityWeighting(), 
    ],
    ope_estimator_hyperparams={
        DoublyRobustWithShrinkage.estimator_name: dros_param,
    },
    regression_models=[
        LogisticRegression,
        RandomForest
    ],
    regression_model_hyperparams={
        LogisticRegression: logistic_regression_param,
        RandomForest: random_forest_param
    }
)
```

<!-- #region id="MJi8Sk2y8mcm" -->
We can set the hyperparameters of OPE estimators / regression models after initializing `InterpretableOPEEvaluator` as well. Below is an example:
<!-- #endregion -->

```python id="aNK67ojt8mcm" executionInfo={"status": "ok", "timestamp": 1633531031323, "user_tz": -330, "elapsed": 3, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
# re-set hyperparameter space for the doubly robust with shrinkage estimator
# with the following code, lambda_ will now be chosen from a logarithm uniform distribution over the interval [0.001, 100]
evaluator.set_ope_estimator_hyperparam_space(
    DoublyRobustWithShrinkage.estimator_name,
    param_name="lambda_",
    lower=1e-3,
    upper=1e2,
    log=True,
    type_=float,
)

# re-set hyperparameter space for logistic regression
# with the following code, C will now be chosen from a logarithm uniform distribution over the interval [0.001, 100]
evaluator.set_regression_model_hyperparam_space(
    LogisticRegression,
    param_name="C",
    lower=1e-2,
    upper=1e2,
    log=True,
    type_=float,
)
```

<!-- #region id="3E7M-Sp88mcn" -->
Once we have initialized `InterpretableOPEEvaluator`, we can call implemented methods to perform IEOE.
<!-- #endregion -->

```python id="sinQVE1H8mcn" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1633531875551, "user_tz": -330, "elapsed": 843510, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="d63812df-ac84-41b4-ea3a-20f395165641"
# estimate policy values
# we obtain a dictionary mapping ope estimator names to np.ndarray storing the estimated policy value for each iteration
policy_value = evaluator.estimate_policy_value()
```

```python id="_xap440_8mco" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1633531875553, "user_tz": -330, "elapsed": 108, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="4b6176b4-5caf-45d9-b990-8fdb0c1cfa42"
print("dm:", policy_value["dm"][:3])
print("dr:", policy_value["dr"][:3])
print("dr-os:", policy_value["dr-os"][:3])
print("ipw:", policy_value["ipw"][:3])
```

```python id="N8_NPhB68mco" executionInfo={"status": "ok", "timestamp": 1633531875553, "user_tz": -330, "elapsed": 63, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
# compute squared errors
# we obtain a dictionary mapping ope estimator names to np.ndarray storing the calculated squared error for each iteration
squared_error = evaluator.calculate_squared_error()
```

```python id="WcqDkW4f8mcp" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1633531875554, "user_tz": -330, "elapsed": 62, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="8c10f4c7-f998-486f-eee4-6c6e203241fa"
print("dm:", squared_error["dm"][:3])
print("dr:", squared_error["dr"][:3])
print("dr-os:", squared_error["dr-os"][:3])
print("ipw:", squared_error["ipw"][:3])
```

```python id="7JrCAlGP8mcq" colab={"base_uri": "https://localhost:8080/", "height": 432} executionInfo={"status": "ok", "timestamp": 1633531875555, "user_tz": -330, "elapsed": 51, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="b2029a47-41ba-4813-caac-d1bf542e0ac7"
# visualize cdf of squared errors for all ope estimators
evaluator.visualize_cdf_aggregate()
```

```python id="Y4whQskM8mcr" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1633531875556, "user_tz": -330, "elapsed": 44, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="d50c50ec-01e2-45e3-b804-06147bc4cfd9"
# compute the au-cdf score (area under cdf of squared error over interval [0, thershold]), higher score is better
# we obtain a dictionary mapping ope estimator names to cvar scores 
au_cdf = evaluator.calculate_au_cdf_score(threshold=0.0004)
au_cdf
```

```python id="mrJDfPla8mcs" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1633531876454, "user_tz": -330, "elapsed": 928, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="6f0236fa-36d2-4ab0-9883-5b173458a247"
# by activating the `scale` option, 
# we obtain the au_cdf scores where the highest score is scaled to 1
au_cdf_scaled = evaluator.calculate_au_cdf_score(threshold=0.0004, scale=True)
au_cdf_scaled
```

```python id="QEvejCKy8mcs" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1633531876456, "user_tz": -330, "elapsed": 39, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="320351c3-8e28-465e-bbaf-dfaf818be8b2"
# compute the cvar score (expected value of squared error above probability alpha), lower score is better
# we obtain a dictionary mapping ope estimator names to cvar scores 
cvar = evaluator.calculate_cvar_score(alpha=90)
cvar
```

```python id="G4NXJqVA8mct" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1633531876457, "user_tz": -330, "elapsed": 27, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="053bb41d-9725-49e5-f9bb-39dcce530fb5"
# by activating the `scale` option, 
# we obtain the cvar scores where the lowest score is scaled to 1
cvar_scaled = evaluator.calculate_cvar_score(alpha=90, scale=True)
cvar_scaled
```

<!-- #region id="skeoq1N28mct" -->
## Example 2 - Synthetic P-Score
<!-- #endregion -->

<!-- #region id="e1gRO7rYEY_L" -->
A quickstart guide of pyIEOE using synthetic logged bandit feedback data and using estimated propensity scores of the behavior policy instead of the ground truth values.
<!-- #endregion -->

<!-- #region id="AmhBtw3nDu6d" -->
This section demonstrates an example of conducting Interpretable Evaluation for Off-Policy Evaluation (IEOE). We use synthetic logged bandit feedback data generated using [`obp`](https://github.com/st-tech/zr-obp) and evaluate the performance of Direct Method (DM), Doubly Robust (DR), Doubly Robust with Shrinkage (DRos), and Inverse Probability Weighting (IPW).

Our example contains the following three major steps:

1. Data Preparation
2. Setting Hyperparameter Spaces for Off-Policy Evaluation
3. Interpretable Evaluation for Off-Policy Evaluation
<!-- #endregion -->

<!-- #region id="4z-SwDXnDu6q" -->
### Data Preparation

In order to conduct IEOE using `pyieoe`, we need to prepare logged bandit feedback data, action distributions of evaluation policies, and ground truth policy values of evaluation policies. Because `pyieoe` is built with the intention of being used with `obp`, these inputs must follow the conventions in `obp`. Specifically, logged bandit feedback data must be of type `BanditFeedback`, action distributions must be of type `np.ndarray`, and ground truth policy values must be of type `float` (or `int`). 

In this example, we generate synthetic logged bandit feedback data and perform off-policy learning to obtain two sets of evaluation policies along with their action distributions and ground truth policy values using `obp`. For a detailed explanation of this process, please refer to the [official obp example](https://github.com/st-tech/zr-obp/blob/master/examples/quickstart/quickstart_synthetic.ipynb).
<!-- #endregion -->

```python id="jx284y0CDu6t" executionInfo={"status": "ok", "timestamp": 1633531907767, "user_tz": -330, "elapsed": 1147, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
# generate a synthetic bandit dataset with 10 actions
# we use `logistic function` as the reward function and `linear_behavior_policy` as the behavior policy.
# one can define their own reward function and behavior policy such as nonlinear ones. 
dataset = SyntheticBanditDataset(
    n_actions=10,
    dim_context=5,
    reward_type="binary", # "binary" or "continuous"
    reward_function=logistic_reward_function,
    behavior_policy_function=linear_behavior_policy,
    random_state=12345
)
# obtain training and test sets of synthetic logged bandit feedback
n_rounds_train, n_rounds_test = 10000, 10000
bandit_feedback_train = dataset.obtain_batch_bandit_feedback(n_rounds=n_rounds_train)
bandit_feedback_test = dataset.obtain_batch_bandit_feedback(n_rounds=n_rounds_test)

# define IPWLearner with Logistic Regression as its base ML model
evaluation_policy_a = IPWLearner(
    n_actions=dataset.n_actions,
    len_list=dataset.len_list,
    base_classifier=LogisticRegression(C=100, random_state=12345)
)
# train IPWLearner on the training set of the synthetic logged bandit feedback
evaluation_policy_a.fit(
    context=bandit_feedback_train["context"],
    action=bandit_feedback_train["action"],
    reward=bandit_feedback_train["reward"],
    pscore=bandit_feedback_train["pscore"]
)
# obtains action choice probabilities for the test set of the synthetic logged bandit feedback
action_dist_a = evaluation_policy_a.predict_proba(
    context=bandit_feedback_test["context"],
    tau=0.1 # temperature hyperparameter
)

# define IPWLearner with Random Forest as its base ML model
evaluation_policy_b = IPWLearner(
    n_actions=dataset.n_actions,
    len_list=dataset.len_list,
    base_classifier=RandomForest(n_estimators=30, min_samples_leaf=10, random_state=12345)
)
# train IPWLearner on the training set of the synthetic logged bandit feedback
evaluation_policy_b.fit(
    context=bandit_feedback_train["context"],
    action=bandit_feedback_train["action"],
    reward=bandit_feedback_train["reward"],
    pscore=bandit_feedback_train["pscore"]
)
# obtains action choice probabilities for the test set of the synthetic logged bandit feedback
action_dist_b = evaluation_policy_b.predict_proba(
    context=bandit_feedback_test["context"],
    tau=0.1 # temperature hyperparameter
)

# obtain ground truth policy value for each action choice probabilities
expected_rewards = bandit_feedback_test["expected_reward"]
ground_truth_a = np.average(expected_rewards, weights=action_dist_a[:, :, 0], axis=1).mean()
ground_truth_b = np.average(expected_rewards, weights=action_dist_b[:, :, 0], axis=1).mean()
```

<!-- #region id="oKC-lDodDu60" -->
### Setting Hyperparameter Spaces for Off-Policy Evaluation

An integral aspect of IEOE is the different sources of variance. The main sources of variance are evaluation policies, random states, hyperparameters of OPE estimators, and hyperparameters of regression models. 

In this step, we define the spaces from which the hyperparameters of OPE estimators / regression models are chosen. (The evaluation policy space is defined in the previous step, and the random state space will be defined in the next step.)
<!-- #endregion -->

```python id="c2jGLE-3Du64" executionInfo={"status": "ok", "timestamp": 1633531909694, "user_tz": -330, "elapsed": 6, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
# set hyperparameter space for ope estimators

# set hyperparameter space for the doubly robust with shrinkage estimator
# with the following code, lambda_ will be chosen from a logarithm uniform distribution over the interval [0.001, 1000]
lambda_ = {
    "lower": 1e-3,
    "upper": 1e3,
    "log": True,
    "type": float
}
dros_param = {"lambda_": lambda_}
```

```python id="siofnHwHDu66" executionInfo={"status": "ok", "timestamp": 1633531911713, "user_tz": -330, "elapsed": 414, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
# set hyperparameter space for regression models

# set hyperparameter space for logistic regression
# with the following code, C will be chosen from a logarithm uniform distribution over the interval [0.001, 100]
C = {
    "lower": 1e-3,
    "upper": 1e2,
    "log": True,
    "type": float
}
# create a dictionary mapping hyperparamter names to hyperparamter spaces
logistic_regression_param = {"C": C}

# set hyperparameter space for the random forest classifier
# with the following code, n_estimators will be chosen from a logarithm uniform distribution over the interval [50, 100]
# the chosen value will be of type int
n_estimators = {
    "lower": 5e1,
    "upper": 1e2,
    "log": True,
    "type": int
}
# with the following code, max_depth will be chosen from a uniform distribution over the interval [2, 10]
# the chosen value will be of type int
max_depth = {
    "lower": 2,
    "upper": 10,
    "log": False,
    "type": int
}
# with the following code, min_samples_split will be chosen from a uniform distribution over the interval [2, 10]
# the chosen value will be of type int
min_samples_split = {
    "lower": 2,
    "upper": 10,
    "log": False,
    "type": int
}
# create a dictionary mapping hyperparamter names to hyperparamter spaces
random_forest_param = {
    "n_estimators": n_estimators, 
    "max_depth": max_depth, 
    "min_samples_split": min_samples_split
}
```

<!-- #region id="AqoaVgxHDu68" -->
### Interpretable Evaluation for Off-Policy Evaluation

With the above steps completed, we can finally conduct IEOE by utilizing the `InterpretableOPEEvaluator` class.

Here is a brief description for each parameter that can be passed into `InterpretableOPEEvaluator`:

- `random_states`: a list of integers representing the random_state used when performing OPE; corresponds to the number of iterations
- `bandit_feedback`: a list of logged bandit feedback data
- `evaluation_policies`: a list of tuples representing (ground truth policy value, action distribution)
- `ope_estimators`: a list of OPE ope_estimators
- `ope_estimator_hyperparams`: a dictionary mapping OPE estimator names to OPE estimator hyperparameter spaces defined in step 2
- `regression_models`: a list of regression_models
- `regression_model_hyperparams`: a dictionary mapping regression models to regression model hyperparameter spaces defined in step 2
<!-- #endregion -->

```python id="blW97KoKDu6-" executionInfo={"status": "ok", "timestamp": 1633531914852, "user_tz": -330, "elapsed": 442, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
# initializing class
evaluator = InterpretableOPEEvaluator(
    random_states=np.arange(1000),
    bandit_feedbacks=[bandit_feedback_test],
    evaluation_policies=[
        (ground_truth_a, action_dist_a), 
        (ground_truth_b, action_dist_b)
    ],
    ope_estimators=[
        DirectMethod(),
        DoublyRobust(),
        DoublyRobustWithShrinkage(),
        InverseProbabilityWeighting(), 
    ],
    ope_estimator_hyperparams={
        DoublyRobustWithShrinkage.estimator_name: dros_param,
    },
    regression_models=[
        LogisticRegression,
        RandomForest
    ],
    regression_model_hyperparams={
        LogisticRegression: logistic_regression_param,
        RandomForest: random_forest_param
    },
    pscore_estimators=[
        LogisticRegression,
        RandomForest
    ],
    pscore_estimator_hyperparams={
        LogisticRegression: logistic_regression_param,
        RandomForest: random_forest_param
    }
)
```

<!-- #region id="i6cfcKBvDu7A" -->
We can set the hyperparameters of OPE estimators / regression models after initializing `InterpretableOPEEvaluator` as well. Below is an example:
<!-- #endregion -->

```python id="nk6WSm2BDu7B" executionInfo={"status": "ok", "timestamp": 1633531917054, "user_tz": -330, "elapsed": 454, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
# re-set hyperparameter space for the doubly robust with shrinkage estimator
# with the following code, lambda_ will now be chosen from a logarithm uniform distribution over the interval [0.001, 100]
evaluator.set_ope_estimator_hyperparam_space(
    DoublyRobustWithShrinkage.estimator_name,
    param_name="lambda_",
    lower=1e-3,
    upper=1e2,
    log=True,
    type_=float,
)

# re-set hyperparameter space for logistic regression
# with the following code, C will now be chosen from a logarithm uniform distribution over the interval [0.001, 100]
evaluator.set_regression_model_hyperparam_space(
    LogisticRegression,
    param_name="C",
    lower=1e-2,
    upper=1e2,
    log=True,
    type_=float,
)
```

<!-- #region id="Hk4SExejDu7C" -->
Once we have initialized `InterpretableOPEEvaluator`, we can call implemented methods to perform IEOE.
<!-- #endregion -->

```python id="P7iKs11nDu7D" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1633533386212, "user_tz": -330, "elapsed": 1467183, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="b653b918-70fc-49a7-8f4d-aefaa4fb5797"
# estimate policy values
# we obtain a dictionary mapping ope estimator names to np.ndarray storing the estimated policy value for each iteration
policy_value = evaluator.estimate_policy_value()
```

```python id="7Z_welpvDu7F" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1633533386213, "user_tz": -330, "elapsed": 32, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="19682ffa-e010-4e08-e0e1-9ab069d65abf"
print("dm:", policy_value["dm"][:3])
print("dr:", policy_value["dr"][:3])
print("dr-os:", policy_value["dr-os"][:3])
print("ipw:", policy_value["ipw"][:3])
```

```python id="0ujRjvG5Du7G" executionInfo={"status": "ok", "timestamp": 1633533403052, "user_tz": -330, "elapsed": 1379, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
# compute squared errors
# we obtain a dictionary mapping ope estimator names to np.ndarray storing the calculated squared error for each iteration
squared_error = evaluator.calculate_squared_error()
```

```python id="vP_vo9hcDu7H" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1633533405765, "user_tz": -330, "elapsed": 21, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="2edba37d-678c-4f5c-b168-cf943f4e3441"
print("dm:", squared_error["dm"][:3])
print("dr:", squared_error["dr"][:3])
print("dr-os:", squared_error["dr-os"][:3])
print("ipw:", squared_error["ipw"][:3])
```

```python id="SZbxYSCjDu7H" colab={"base_uri": "https://localhost:8080/", "height": 432} executionInfo={"status": "ok", "timestamp": 1633533409081, "user_tz": -330, "elapsed": 1090, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="5c514cab-442c-4b37-d22d-27c3a6eb1363"
# visualize cdf of squared errors for all ope estimators
evaluator.visualize_cdf_aggregate(xmax=0.002)
```

```python id="36naCb1EDu7I" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1633533412696, "user_tz": -330, "elapsed": 620, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="de522526-3696-4abc-d458-11e226b41693"
# compute the au-cdf score (area under cdf of squared error over interval [0, thershold]), higher score is better
# we obtain a dictionary mapping ope estimator names to cvar scores 
au_cdf = evaluator.calculate_au_cdf_score(threshold=0.0004)
au_cdf
```

```python id="09pr3_UuDu7J" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1633533414676, "user_tz": -330, "elapsed": 21, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="91bbe2ff-ef6f-4bc9-c6eb-46c03c075feb"
# by activating the `scale` option, 
# we obtain the au_cdf scores where the highest score is scaled to 1
au_cdf_scaled = evaluator.calculate_au_cdf_score(threshold=0.0004, scale=True)
au_cdf_scaled
```

```python id="787ebELZDu7J" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1633533416444, "user_tz": -330, "elapsed": 17, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="42e97017-91d9-4e6c-f109-e09f7578215a"
# compute the cvar score (expected value of squared error above probability alpha), lower score is better
# we obtain a dictionary mapping ope estimator names to cvar scores 
cvar = evaluator.calculate_cvar_score(alpha=90)
cvar
```

```python id="b2GuPiBiDu7K" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1633533418752, "user_tz": -330, "elapsed": 18, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="aca49a97-6dce-4b3c-f618-6da009d69d9e"
# by activating the `scale` option, 
# we obtain the cvar scores where the lowest score is scaled to 1
cvar_scaled = evaluator.calculate_cvar_score(alpha=90, scale=True)
cvar_scaled
```

<!-- #region id="M91PB3GDDu7O" -->
## Example 3 - Synthetic RSCV
<!-- #endregion -->

<!-- #region id="f35iP_S8FB0C" -->
A quickstart guide of pyIEOE using synthetic logged bandit feedback data and using RandomizedSearchCV for regression models and pscore estimators.
<!-- #endregion -->

<!-- #region id="uwJs5p_NE8Kh" -->
### Data Preparation

In order to conduct IEOE using `pyieoe`, we need to prepare logged bandit feedback data, action distributions of evaluation policies, and ground truth policy values of evaluation policies. Because `pyieoe` is built with the intention of being used with `obp`, these inputs must follow the conventions in `obp`. Specifically, logged bandit feedback data must be of type `BanditFeedback`, action distributions must be of type `np.ndarray`, and ground truth policy values must be of type `float` (or `int`). 

In this example, we generate synthetic logged bandit feedback data and perform off-policy learning to obtain two sets of evaluation policies along with their action distributions and ground truth policy values using `obp`. For a detailed explanation of this process, please refer to the [official obp example](https://github.com/st-tech/zr-obp/blob/master/examples/quickstart/quickstart_synthetic.ipynb).
<!-- #endregion -->

```python id="hiKvoK-cE8Ki"
# generate a synthetic bandit dataset with 10 actions
# we use `logistic function` as the reward function and `linear_behavior_policy` as the behavior policy.
# one can define their own reward function and behavior policy such as nonlinear ones. 
dataset = SyntheticBanditDataset(
    n_actions=10,
    dim_context=5,
    reward_type="binary", # "binary" or "continuous"
    reward_function=logistic_reward_function,
    behavior_policy_function=linear_behavior_policy,
    random_state=12345
)
# obtain training and test sets of synthetic logged bandit feedback
n_rounds_train, n_rounds_test = 10000, 10000
bandit_feedback_train = dataset.obtain_batch_bandit_feedback(n_rounds=n_rounds_train)
bandit_feedback_test = dataset.obtain_batch_bandit_feedback(n_rounds=n_rounds_test)

# define IPWLearner with Logistic Regression as its base ML model
evaluation_policy_a = IPWLearner(
    n_actions=dataset.n_actions,
    len_list=dataset.len_list,
    base_classifier=LogisticRegression(C=100, random_state=12345)
)
# train IPWLearner on the training set of the synthetic logged bandit feedback
evaluation_policy_a.fit(
    context=bandit_feedback_train["context"],
    action=bandit_feedback_train["action"],
    reward=bandit_feedback_train["reward"],
    pscore=bandit_feedback_train["pscore"]
)
# obtains action choice probabilities for the test set of the synthetic logged bandit feedback
action_dist_a = evaluation_policy_a.predict_proba(
    context=bandit_feedback_test["context"],
    tau=0.1 # temperature hyperparameter
)

# define IPWLearner with Random Forest as its base ML model
evaluation_policy_b = IPWLearner(
    n_actions=dataset.n_actions,
    len_list=dataset.len_list,
    base_classifier=RandomForest(n_estimators=30, min_samples_leaf=10, random_state=12345)
)
# train IPWLearner on the training set of the synthetic logged bandit feedback
evaluation_policy_b.fit(
    context=bandit_feedback_train["context"],
    action=bandit_feedback_train["action"],
    reward=bandit_feedback_train["reward"],
    pscore=bandit_feedback_train["pscore"]
)
# obtains action choice probabilities for the test set of the synthetic logged bandit feedback
action_dist_b = evaluation_policy_b.predict_proba(
    context=bandit_feedback_test["context"],
    tau=0.1 # temperature hyperparameter
)

# obtain ground truth policy value for each action choice probabilities
expected_rewards = bandit_feedback_test["expected_reward"]
ground_truth_a = np.average(expected_rewards, weights=action_dist_a[:, :, 0], axis=1).mean()
ground_truth_b = np.average(expected_rewards, weights=action_dist_b[:, :, 0], axis=1).mean()
```

<!-- #region id="Gz7FkzuFE8Kl" -->
### Setting Hyperparameter Spaces for Off-Policy Evaluation

An integral aspect of IEOE is the different sources of variance. The main sources of variance are evaluation policies, random states, hyperparameters of OPE estimators, and hyperparameters of regression models. 

In this step, we define the spaces from which the hyperparameters of OPE estimators / regression models are chosen. (The evaluation policy space is defined in the previous step, and the random state space will be defined in the next step.)
<!-- #endregion -->

```python id="NVH2dZzYE8Km"
# set hyperparameter space for ope estimators

# set hyperparameter space for the doubly robust with shrinkage estimator
# with the following code, lambda_ will be chosen from a logarithm uniform distribution over the interval [0.001, 1000]
lambda_ = {
    "lower": 1e-3,
    "upper": 1e3,
    "log": True,
    "type": float
}
dros_param = {"lambda_": lambda_}
```

```python id="auB7GcSYE8Kn"
# set hyperparameter space for logistic regression using RandomizedSearchCV

from sklearn.utils.fixes import loguniform
logistic = LogisticRegression()
distributions = {
    "C": loguniform(1e-2, 1e2)
}
clf_logistic = RandomizedSearchCV(logistic, distributions, random_state=0, n_iter=5)
```

```python id="H20quziGE8Ko"
# set hyperparameter space for random forest classifier using RandomizedSearchCV

from scipy.stats import randint
randforest = RandomForest()
distributions = {
    # n_estimators will be chosen from a uniform distribution over the interval [50, 100)
    "n_estimators": randint(5e1, 1e2), 
    # max_depth will be chosen from a uniform distribution over the interval [2, 10)
    "max_depth": randint(2, 10), 
    # min_samples_split will be chosen from a uniform distribution over the interval [2, 10)
    "min_samples_split": randint(2, 10)
}
clf_randforest = RandomizedSearchCV(randforest, distributions, random_state=0, n_iter=5)
```

<!-- #region id="YhIFJTDdE8Ko" -->
### Interpretable Evaluation for Off-Policy Evaluation

With the above steps completed, we can finally conduct IEOE by utilizing the `InterpretableOPEEvaluator` class.

Here is a brief description for each parameter that can be passed into `InterpretableOPEEvaluator`:

- `random_states`: a list of integers representing the random_state used when performing OPE; corresponds to the number of iterations
- `bandit_feedback`: a list of logged bandit feedback data
- `evaluation_policies`: a list of tuples representing (ground truth policy value, action distribution)
- `ope_estimators`: a list of OPE ope_estimators
- `ope_estimator_hyperparams`: a dictionary mapping OPE estimator names to OPE estimator hyperparameter spaces defined in step 2
- `regression_models`: a list of regression_models
- `regression_model_hyperparams`: a dictionary mapping regression models to regression model hyperparameter spaces defined in step 2
<!-- #endregion -->

```python id="favcoQnZE8Kp"
# initializing class
evaluator = InterpretableOPEEvaluator(
    random_states=np.arange(100),
    bandit_feedbacks=[bandit_feedback_test],
    evaluation_policies=[
        (ground_truth_a, action_dist_a), 
        (ground_truth_b, action_dist_b)
    ],
    ope_estimators=[
        DirectMethod(),
        DoublyRobust(),
        DoublyRobustWithShrinkage(),
        InverseProbabilityWeighting(), 
    ],
    ope_estimator_hyperparams={
        DoublyRobustWithShrinkage.estimator_name: dros_param,
    },
    regression_models=[
        clf_logistic,
        clf_randforest
    ],
    pscore_estimators=[
        clf_logistic,
        clf_randforest
    ]
)
```

<!-- #region id="aZ_pxjj-E8Kp" -->
Once we have initialized `InterpretableOPEEvaluator`, we can call implemented methods to perform IEOE.
<!-- #endregion -->

```python id="-WMc3JMAE8Kq" outputId="6e0d2868-c3a5-4db0-bff3-e8123a46934a"
# estimate policy values
# we obtain a dictionary mapping ope estimator names to np.ndarray storing the estimated policy value for each iteration
policy_value = evaluator.estimate_policy_value()
```

```python id="NV5s_KQNE8Kr" outputId="d9479f4e-67de-4ebe-ed30-52246595e4be"
print("dm:", policy_value["dm"][:3])
print("dr:", policy_value["dr"][:3])
print("dr-os:", policy_value["dr-os"][:3])
print("ipw:", policy_value["ipw"][:3])
```

```python id="2YpdxF85E8Kr"
# compute squared errors
# we obtain a dictionary mapping ope estimator names to np.ndarray storing the calculated squared error for each iteration
squared_error = evaluator.calculate_squared_error()
```

```python id="dZbd-eVbE8Ks" outputId="46f571d5-d382-45aa-c011-62a6c2f59251"
print("dm:", squared_error["dm"][:3])
print("dr:", squared_error["dr"][:3])
print("dr-os:", squared_error["dr-os"][:3])
print("ipw:", squared_error["ipw"][:3])
```

```python id="TAzYFpRRE8Ks" outputId="b1679b55-9f65-45f8-b927-00eaece7a139"
# visualize cdf of squared errors for all ope estimators
evaluator.visualize_cdf_aggregate(xmax=0.002)
```

```python id="NxYS2lquE8Kt" outputId="af7834ca-acf9-434b-9f44-48468e567960"
# compute the au-cdf score (area under cdf of squared error over interval [0, thershold]), higher score is better
# we obtain a dictionary mapping ope estimator names to cvar scores 
au_cdf = evaluator.calculate_au_cdf_score(threshold=0.0004)
au_cdf
```

```python id="pGJMvH_vE8Ku" outputId="274f7992-a883-4030-eab3-ae0cead1e172"
# by activating the `scale` option, 
# we obtain the au_cdf scores where the highest score is scaled to 1
au_cdf_scaled = evaluator.calculate_au_cdf_score(threshold=0.0004, scale=True)
au_cdf_scaled
```

```python id="064QHL4xE8Kv" outputId="e00e4aef-bc51-4790-896f-6407535169b9"
# compute the cvar score (expected value of squared error above probability alpha), lower score is better
# we obtain a dictionary mapping ope estimator names to cvar scores 
cvar = evaluator.calculate_cvar_score(alpha=90)
cvar
```

```python id="vAAvr2xbE8Kv" outputId="f65d2728-b8db-4053-fcef-82f30c711d9f"
# by activating the `scale` option, 
# we obtain the cvar scores where the lowest score is scaled to 1
cvar_scaled = evaluator.calculate_cvar_score(alpha=90, scale=True)
cvar_scaled
```

<!-- #region id="vfgNKQaTF3f0" -->
## Example 4 - Multiclass dataset
<!-- #endregion -->

<!-- #region id="prn-5aqKF58Z" -->
A quickstart guide of pyIEOE using multiclass classification data as logged bandit feedback data.
<!-- #endregion -->

<!-- #region id="v_FGRdjYGiJn" -->
This section demonstrates an example of conducting Interpretable Evaluation for Off-Policy Evaluation (IEOE). We use logged bandit feedback data generated by modifying multiclass classification data using [`obp`](https://github.com/st-tech/zr-obp) and evaluate the performance of Direct Method (DM), Doubly Robust (DR), Doubly Robust with Shrinkage (DRos), and Inverse Probability Weighting (IPW).

Our example contains the following three major steps:

1. Data Preparation
2. Setting Hyperparameter Spaces for Off-Policy Evaluation
3. Interpretable Evaluation for Off-Policy Evaluation
<!-- #endregion -->

<!-- #region id="PYJ2uM8bGe5L" -->
### Data Preparation

In order to conduct IEOE using `pyieoe`, we need to prepare logged bandit feedback data, action distributions of evaluation policies, and ground truth policy values of evaluation policies. Because `pyieoe` is built with the intention of being used with `obp`, these inputs must follow the conventions in `obp`. Specifically, logged bandit feedback data must be of type `BanditFeedback`, action distributions must be of type `np.ndarray`, and ground truth policy values must be of type `float` (or `int`). 

In this example, we generate logged bandit feedback data by modifying multiclass classification data and obtain two sets of evaluation policies along with their action distributions and ground truth policy values using `obp`. For a detailed explanation of this process, please refer to the [official docs](https://zr-obp.readthedocs.io/en/latest/_autosummary/obp.dataset.multiclass.html#module-obp.dataset.multiclass).
<!-- #endregion -->

```python id="BlxiQlSqGe5M"
# load raw digits data
X, y = load_digits(return_X_y=True)
# convert the raw classification data into the logged bandit dataset
dataset = MultiClassToBanditReduction(
    X=X,
    y=y,
    base_classifier_b=LogisticRegression(random_state=12345),
    alpha_b=0.8,
    dataset_name="digits"
)
# split the original data into the training and evaluation sets
dataset.split_train_eval(eval_size=0.7, random_state=12345)
# obtain logged bandit feedback generated by the behavior policy
bandit_feedback = dataset.obtain_batch_bandit_feedback(random_state=12345)

# obtain action choice probabilities by an evaluation policy and its ground-truth policy value
action_dist_a = dataset.obtain_action_dist_by_eval_policy(
    base_classifier_e=LogisticRegression(C=100, random_state=12345),
    alpha_e=0.9
)
ground_truth_a = dataset.calc_ground_truth_policy_value(action_dist=action_dist_a)
action_dist_b = dataset.obtain_action_dist_by_eval_policy(
    base_classifier_e=RandomForest(n_estimators=100, min_samples_split=5, random_state=12345),
    alpha_e=0.9
)
ground_truth_b = dataset.calc_ground_truth_policy_value(action_dist=action_dist_b)
```

<!-- #region id="D1WCrmkSGe5M" -->
### Setting Hyperparameter Spaces for Off-Policy Evaluation

An integral aspect of IEOE is the different sources of variance. The main sources of variance are evaluation policies, random states, hyperparameters of OPE estimators, and hyperparameters of regression models. 

In this step, we define the spaces from which the hyperparameters of OPE estimators / regression models are chosen. (The evaluation policy space is defined in the previous step, and the random state space will be defined in the next step.)
<!-- #endregion -->

```python id="1qeIQd79Ge5N"
# set hyperparameter space for ope estimators

# set hyperparameter space for the doubly robust with shrinkage estimator
# with the following code, lambda_ will be chosen from a logarithm uniform distribution over the interval [0.001, 1000]
lambda_ = {
    "lower": 1e-3,
    "upper": 1e3,
    "log": True,
    "type": float
}
dros_param = {"lambda_": lambda_}
```

```python id="LOpaUfGYGe5N"
# set hyperparameter space for regression models

# set hyperparameter space for logistic regression
# with the following code, C will be chosen from a logarithm uniform distribution over the interval [0.001, 100]
C = {
    "lower": 1e-3,
    "upper": 1e2,
    "log": True,
    "type": float
}
# with the following code, max_iter will be fixed at 10000 and of type int
max_iter = {
    "lower": 1e4,
    "upper": 1e4,
    "log": False,
    "type": int
}
# create a dictionary mapping hyperparamter names to hyperparamter spaces
logistic_regression_param = {
    "C": C,
    "max_iter": max_iter
}

# set hyperparameter space for random forest classifier
# with the following code, n_estimators will be chosen from a logarithm uniform distribution over the interval [50, 100]
# the chosen value will be of type int
n_estimators = {
    "lower": 5e1,
    "upper": 1e2,
    "log": True,
    "type": int
}
# with the following code, max_depth will be chosen from a uniform distribution over the interval [2, 10]
# the chosen value will be of type int
max_depth = {
    "lower": 2,
    "upper": 10,
    "log": False,
    "type": int
}
# with the following code, min_samples_split will be chosen from a uniform distribution over the interval [2, 10]
# the chosen value will be of type int
min_samples_split = {
    "lower": 2,
    "upper": 10,
    "log": False,
    "type": int
}
# create a dictionary mapping hyperparamter names to hyperparamter spaces
random_forest_param = {
    "n_estimators": n_estimators, 
    "max_depth": max_depth, 
    "min_samples_split": min_samples_split
}
```

<!-- #region id="jLOs3fUrGe5O" -->
### Interpretable Evaluation for Off-Policy Evaluation

With the above steps completed, we can finally conduct IEOE by utilizing the `InterpretableOPEEvaluator` class.

Here is a brief description for each parameter that can be passed into `InterpretableOPEEvaluator`:

- `random_states`: a list of integers representing the random_state used when performing OPE; corresponds to the number of iterations
- `bandit_feedback`: a list of logged bandit feedback data
- `evaluation_policies`: a list of tuples representing (ground truth policy value, action distribution)
- `ope_estimators`: a list of OPE ope_estimators
- `ope_estimator_hyperparams`: a dictionary mapping OPE estimator names to OPE estimator hyperparameter spaces defined in step 2
- `regression_models`: a list of regression regression_models
- `regression_model_hyperparams`: a dictionary mapping regression models to regression model hyperparameter spaces defined in step 2
<!-- #endregion -->

```python id="RpAqLCmUGe5R"
# initializing class
evaluator = InterpretableOPEEvaluator(
    random_states=np.arange(1000),
    bandit_feedbacks=[bandit_feedback],
    evaluation_policies=[
        (ground_truth_a, action_dist_a), 
        (ground_truth_b, action_dist_b)
    ],
    ope_estimators=[
        DirectMethod(),
        DoublyRobust(),
        DoublyRobustWithShrinkage(),
        InverseProbabilityWeighting(), 
    ],
    ope_estimator_hyperparams={
        DoublyRobustWithShrinkage.estimator_name: dros_param,
    },
    regression_models=[
        LogisticRegression,
        RandomForest
    ],
    regression_model_hyperparams={
        LogisticRegression: logistic_regression_param,
        RandomForest: random_forest_param
    }
)
```

<!-- #region id="28uy9Gh6Ge5R" -->
We can set the hyperparameters of OPE estimators / regression models after initializing `InterpretableOPEEvaluator` as well. Below is an example:
<!-- #endregion -->

```python id="PTyLRuKKGe5S"
# re-set hyperparameter space for doubly robust with shrinkage estimator
# with the following code, lambda_ will now be chosen from a logarithm uniform distribution over the interval [0.001, 100]
evaluator.set_ope_estimator_hyperparam_space(
    DoublyRobustWithShrinkage.estimator_name,
    param_name="lambda_",
    lower=1e-3,
    upper=1e2,
    log=True,
    type_=float,
)

# re-set hyperparameter space for logistic regression
# with the following code, C will now be chosen from a logarithm uniform distribution over the interval [0.001, 100]
evaluator.set_regression_model_hyperparam_space(
    LogisticRegression,
    param_name="C",
    lower=1e-2,
    upper=1e2,
    log=True,
    type_=float,
)
```

<!-- #region id="UEnmiLzmGe5S" -->
Once we have initialized `InterpretableOPEEvaluator`, we can call implemented methods to perform IEOE.
<!-- #endregion -->

```python id="XzZGsJyJGe5T" outputId="4359a6d9-05ab-4985-eddb-cf230d39d7d4"
# estimate policy values
# we obtain a dictionary mapping ope estimator names to np.ndarray storing the estimated policy value for each iteration
policy_value = evaluator.estimate_policy_value()
```

```python id="05QPkmrRGe5T" outputId="518cf6f5-ca32-42c5-f9b8-e8eac321614f"
print("dm:", policy_value["dm"][:3])
print("dr:", policy_value["dr"][:3])
print("dr-os:", policy_value["dr-os"][:3])
print("ipw:", policy_value["ipw"][:3])
```

```python id="LR8geYTkGe5U"
# compute squared errors
# we obtain a dictionary mapping ope estimator names to np.ndarray storing the calculated squared error for each iteration
squared_error = evaluator.calculate_squared_error()
```

```python id="fWQUMjXlGe5U" outputId="b67a1f8c-2077-4bed-9b9d-cd3413d18fc8"
print("dm:", squared_error["dm"][:3])
print("dr:", squared_error["dr"][:3])
print("dr-os:", squared_error["dr-os"][:3])
print("ipw:", squared_error["ipw"][:3])
```

```python id="vFxyRkNDGe5V" outputId="a55e6f4c-58ca-4c0e-ec2a-606075da3257"
# visualize cdf of squared errors for all ope estimators
evaluator.visualize_cdf_aggregate()
```

```python id="KzfGNZ-sGe5V" outputId="1a8f4342-c6fe-4758-fc3c-859bf51d99a0"
# compute the au-cdf score (area under cdf of squared error over interval [0, thershold]), higher score is better
# we obtain a dictionary mapping ope estimator names to cvar scores 
au_cdf = evaluator.calculate_au_cdf_score(threshold=0.004)
au_cdf
```

```python id="kZgtGa7wGe5W" outputId="9a3b80f6-bc89-44c4-f669-1a19fbdeed05"
# by activating the `scale` option, 
# we obtain au_cdf scores where the highest score is scaled to 1
au_cdf_scaled = evaluator.calculate_au_cdf_score(threshold=0.004, scale=True)
au_cdf_scaled
```

```python id="wrNEepNHGe5X" outputId="6b5755f8-d508-4a90-f334-0dc8f5ad28e1"
# compute the cvar score (expected value of squared error above probability alpha), lower score is better
# we obtain a dictionary mapping ope estimator names to cvar scores 
cvar = evaluator.calculate_cvar_score(alpha=90)
cvar
```

```python id="JnFZf7znGe5c" outputId="813a83cc-5b7f-447c-d846-ccfa9be56539"
# by activating the `scale` option, 
# we obtain cvar scores where the lowest score is scaled to 1
cvar_scaled = evaluator.calculate_cvar_score(alpha=90, scale=True)
cvar_scaled
```

<!-- #region id="-XznTeoAG_qy" -->
## Example 5 - Multiclass P-Score
<!-- #endregion -->

<!-- #region id="9WghWS-xJdEy" -->
A quickstart guide of pyIEOE using multiclass classification data and using estimated propensity scores of the behavior policy instead of the ground truth values.
<!-- #endregion -->

<!-- #region id="bRJuPPCDG6uA" -->
This notebook demonstrates an example of conducting Interpretable Evaluation for Off-Policy Evaluation (IEOE). We use logged bandit feedback data generated by modifying multiclass classification data using [`obp`](https://github.com/st-tech/zr-obp) and evaluate the performance of Direct Method (DM), Doubly Robust (DR), Doubly Robust with Shrinkage (DRos), and Inverse Probability Weighting (IPW).

Our example contains the following three major steps:

1. Data Preparation
2. Setting Hyperparameter Spaces for Off-Policy Evaluation
3. Interpretable Evaluation for Off-Policy Evaluation
<!-- #endregion -->

<!-- #region id="UDuL-UTxG6uL" -->
### Data Preparation

In order to conduct IEOE using `pyieoe`, we need to prepare logged bandit feedback data, action distributions of evaluation policies, and ground truth policy values of evaluation policies. Because `pyieoe` is built with the intention of being used with `obp`, these inputs must follow the conventions in `obp`. Specifically, logged bandit feedback data must be of type `BanditFeedback`, action distributions must be of type `np.ndarray`, and ground truth policy values must be of type `float` (or `int`). 

In this example, we generate logged bandit feedback data by modifying multiclass classification data and obtain two sets of evaluation policies along with their action distributions and ground truth policy values using `obp`. For a detailed explanation of this process, please refer to the [official docs](https://zr-obp.readthedocs.io/en/latest/_autosummary/obp.dataset.multiclass.html#module-obp.dataset.multiclass).
<!-- #endregion -->

```python id="EgLP5CSjG6uM"
# load raw digits data
X, y = load_digits(return_X_y=True)
# convert the raw classification data into the logged bandit dataset
dataset = MultiClassToBanditReduction(
    X=X,
    y=y,
    base_classifier_b=LogisticRegression(random_state=12345),
    alpha_b=0.8,
    dataset_name="digits"
)
# split the original data into the training and evaluation sets
dataset.split_train_eval(eval_size=0.7, random_state=12345)
# obtain logged bandit feedback generated by the behavior policy
bandit_feedback = dataset.obtain_batch_bandit_feedback(random_state=12345)

# obtain action choice probabilities by an evaluation policy and its ground-truth policy value
action_dist_a = dataset.obtain_action_dist_by_eval_policy(
    base_classifier_e=LogisticRegression(C=100, random_state=12345, max_iter=10000),
    alpha_e=0.9
)
ground_truth_a = dataset.calc_ground_truth_policy_value(action_dist=action_dist_a)
action_dist_b = dataset.obtain_action_dist_by_eval_policy(
    base_classifier_e=RandomForest(n_estimators=100, min_samples_split=5, random_state=12345),
    alpha_e=0.9
)
ground_truth_b = dataset.calc_ground_truth_policy_value(action_dist=action_dist_b)
```

<!-- #region id="xBPAdys_G6uP" -->
### Setting Hyperparameter Spaces for Off-Policy Evaluation

An integral aspect of IEOE is the different sources of variance. The main sources of variance are evaluation policies, random states, hyperparameters of OPE estimators, and hyperparameters of regression models. 

In this step, we define the spaces from which the hyperparameters of OPE estimators / regression models are chosen. (The evaluation policy space is defined in the previous step, and the random state space will be defined in the next step.)
<!-- #endregion -->

```python id="nvJYaTFVG6uR"
# set hyperparameter space for ope estimators

# set hyperparameter space for the doubly robust with shrinkage estimator
# with the following code, lambda_ will be chosen from a logarithm uniform distribution over the interval [0.001, 1000]
lambda_ = {
    "lower": 1e-3,
    "upper": 1e3,
    "log": True,
    "type": float
}
dros_param = {"lambda_": lambda_}
```

```python id="Tf--fTxVG6uU"
# set hyperparameter space for regression models

# set hyperparameter space for logistic regression
# with the following code, C will be chosen from a logarithm uniform distribution over the interval [0.001, 100]
# the chosen value will be of type float
C = {
    "lower": 1e-3,
    "upper": 1e2,
    "log": True,
    "type": float
}
# with the following code, max_iter will be fixed at 10000 and of type int
max_iter = {
    "lower": 1e4,
    "upper": 1e4,
    "log": False,
    "type": int
}
# create a dictionary mapping hyperparamter names to hyperparamter spaces
logistic_regression_param = {
    "C": C,
    "max_iter": max_iter
}

# set hyperparameter space for random forest classifier
# with the following code, n_estimators will be chosen from a logarithm uniform distribution over the interval [50, 100]
# the chosen value will be of type int
n_estimators = {
    "lower": 5e1,
    "upper": 1e2,
    "log": True,
    "type": int
}
# with the following code, max_depth will be chosen from a uniform distribution over the interval [2, 10]
# the chosen value will be of type int
max_depth = {
    "lower": 2,
    "upper": 10,
    "log": False,
    "type": int
}
# with the following code, min_samples_split will be chosen from a uniform distribution over the interval [2, 10]
# the chosen value will be of type int
min_samples_split = {
    "lower": 2,
    "upper": 10,
    "log": False,
    "type": int
}
# create a dictionary mapping hyperparamter names to hyperparamter spaces
random_forest_param = {
    "n_estimators": n_estimators, 
    "max_depth": max_depth, 
    "min_samples_split": min_samples_split
}
```

<!-- #region id="huLEWqzjG6uW" -->
### Interpretable Evaluation for Off-Policy Evaluation

With the above steps completed, we can finally conduct IEOE by utilizing the `InterpretableOPEEvaluator` class.

Here is a brief description for each parameter that can be passed into `InterpretableOPEEvaluator`:

- `random_states`: a list of integers representing the random_state used when performing OPE; corresponds to the number of iterations
- `bandit_feedback`: a list of logged bandit feedback data
- `evaluation_policies`: a list of tuples representing (ground truth policy value, action distribution)
- `ope_estimators`: a list of OPE ope_estimators
- `ope_estimator_hyperparams`: a dictionary mapping OPE estimator names to OPE estimator hyperparameter spaces defined in step 2
- `regression_models`: a list of regression regression_models
- `regression_model_hyperparams`: a dictionary mapping regression models to regression model hyperparameter spaces defined in step 2
<!-- #endregion -->

```python id="A8WeqUEWG6uX"
# initializing class
evaluator = InterpretableOPEEvaluator(
    random_states=np.arange(1000),
    bandit_feedbacks=[bandit_feedback],
    evaluation_policies=[
        (ground_truth_a, action_dist_a), 
        (ground_truth_b, action_dist_b)
    ],
    ope_estimators=[
        DirectMethod(),
        DoublyRobust(),
        DoublyRobustWithShrinkage(),
        InverseProbabilityWeighting(), 
    ],
    ope_estimator_hyperparams={
        DoublyRobustWithShrinkage.estimator_name: dros_param,
    },
    regression_models=[
        LogisticRegression,
        RandomForest
    ],
    regression_model_hyperparams={
        LogisticRegression: logistic_regression_param,
        RandomForest: random_forest_param
    },
    pscore_estimators=[
        LogisticRegression,
        RandomForest
    ],
    pscore_estimator_hyperparams={
        LogisticRegression: logistic_regression_param,
        RandomForest: random_forest_param
    }
)
```

<!-- #region id="EBeRcYo-G6uX" -->
We can set the hyperparameters of OPE estimators / regression models after initializing `InterpretableOPEEvaluator` as well. Below is an example:
<!-- #endregion -->

```python id="bHC2WIxbG6uY"
# re-set hyperparameter space for doubly robust with shrinkage estimator
# with the following code, lambda_ will now be chosen from a logarithm uniform distribution over the interval [0.001, 100]
evaluator.set_ope_estimator_hyperparam_space(
    DoublyRobustWithShrinkage.estimator_name,
    param_name="lambda_",
    lower=1e-3,
    upper=1e2,
    log=True,
    type_=float,
)

# re-set hyperparameter space for logistic regression
# with the following code, C will now be chosen from a logarithm uniform distribution over the interval [0.001, 100]
evaluator.set_regression_model_hyperparam_space(
    LogisticRegression,
    param_name="C",
    lower=1e-2,
    upper=1e2,
    log=True,
    type_=float,
)
```

<!-- #region id="GGwQNCVyG6uY" -->
Once we have initialized `InterpretableOPEEvaluator`, we can call implemented methods to perform IEOE.
<!-- #endregion -->

```python id="uNzEIUMWG6uZ" outputId="8c914282-e8c9-4fe3-e627-7b02f154ab29"
# estimate policy values
# we obtain a dictionary mapping ope estimator names to np.ndarray storing the estimated policy value for each iteration
policy_value = evaluator.estimate_policy_value()
```

```python id="LwUCCvvwG6uZ" outputId="486f4818-1c81-4fc4-e82c-18585cecc42d"
print("dm:", policy_value["dm"][:3])
print("dr:", policy_value["dr"][:3])
print("dr-os:", policy_value["dr-os"][:3])
print("ipw:", policy_value["ipw"][:3])
```

```python id="0NPL_UYUG6ua"
# compute squared errors
# we obtain a dictionary mapping ope estimator names to np.ndarray storing the calculated squared error for each iteration
squared_error = evaluator.calculate_squared_error()
```

```python id="-ophfTl6G6ua" outputId="751cb388-ce5b-4d5b-d7ba-c814a8e5669f"
print("dm:", squared_error["dm"][:3])
print("dr:", squared_error["dr"][:3])
print("dr-os:", squared_error["dr-os"][:3])
print("ipw:", squared_error["ipw"][:3])
```

```python id="pROmONbVG6ua" outputId="c826906d-328c-4f06-8ac7-9188aff7a8d3"
# visualize cdf of squared errors for all ope estimators
evaluator.visualize_cdf_aggregate(xmax=0.04)
```

```python id="nhNcA5sWG6ug" outputId="dfd454dd-caee-4436-bcc6-bb9a4ad45c07"
# compute the au-cdf score (area under cdf of squared error over interval [0, thershold]), higher score is better
# we obtain a dictionary mapping ope estimator names to cvar scores 
au_cdf = evaluator.calculate_au_cdf_score(threshold=0.004)
au_cdf
```

```python id="zIC_frEzG6uj" outputId="69cadae3-361c-444c-f257-ba98fa7519aa"
# by activating the `scale` option, 
# we obtain au_cdf scores where the highest score is scaled to 1
au_cdf_scaled = evaluator.calculate_au_cdf_score(threshold=0.004, scale=True)
au_cdf_scaled
```

```python id="4ysQqt_OG6up" outputId="299a7247-eb15-417e-8b31-5788a890868f"
# compute the cvar score (expected value of squared error above probability alpha), lower score is better
# we obtain a dictionary mapping ope estimator names to cvar scores 
cvar = evaluator.calculate_cvar_score(alpha=90)
cvar
```

```python id="MD-2jhGtG6ut" outputId="1fb3cdc7-f7e8-464d-f113-abf117b472cd"
# by activating the `scale` option, 
# we obtain cvar scores where the lowest score is scaled to 1
cvar_scaled = evaluator.calculate_cvar_score(alpha=90, scale=True)
cvar_scaled
```

<!-- #region id="Cl59UFs_G6uu" -->
## Example 6 - Multiclass RSCV
<!-- #endregion -->

<!-- #region id="7qGIsmOAJmG_" -->
A quickstart guide of pyIEOE using multiclass classification data and using RandomizedSearchCV for regression models and pscore estimators.
<!-- #endregion -->

<!-- #region id="wY466oehJqXZ" -->
This section demonstrates an example of conducting Interpretable Evaluation for Off-Policy Evaluation (IEOE). We use logged bandit feedback data generated by modifying multiclass classification data using [`obp`](https://github.com/st-tech/zr-obp) and evaluate the performance of Direct Method (DM), Doubly Robust (DR), Doubly Robust with Shrinkage (DRos), and Inverse Probability Weighting (IPW).

Our example contains the following three major steps:

1. Data Preparation
2. Setting Hyperparameter Spaces for Off-Policy Evaluation
3. Interpretable Evaluation for Off-Policy Evaluation
<!-- #endregion -->

<!-- #region id="Uqy99AyZJqXf" -->
### Data Preparation

In order to conduct IEOE using `pyieoe`, we need to prepare logged bandit feedback data, action distributions of evaluation policies, and ground truth policy values of evaluation policies. Because `pyieoe` is built with the intention of being used with `obp`, these inputs must follow the conventions in `obp`. Specifically, logged bandit feedback data must be of type `BanditFeedback`, action distributions must be of type `np.ndarray`, and ground truth policy values must be of type `float` (or `int`). 

In this example, we generate logged bandit feedback data by modifying multiclass classification data and obtain two sets of evaluation policies along with their action distributions and ground truth policy values using `obp`. For a detailed explanation of this process, please refer to the [official docs](https://zr-obp.readthedocs.io/en/latest/_autosummary/obp.dataset.multiclass.html#module-obp.dataset.multiclass).
<!-- #endregion -->

```python id="n1EjJoJ_JqXg"
# load raw digits data
X, y = load_digits(return_X_y=True)
# convert the raw classification data into the logged bandit dataset
dataset = MultiClassToBanditReduction(
    X=X,
    y=y,
    base_classifier_b=LogisticRegression(random_state=12345),
    alpha_b=0.8,
    dataset_name="digits"
)
# split the original data into the training and evaluation sets
dataset.split_train_eval(eval_size=0.7, random_state=12345)
# obtain logged bandit feedback generated by the behavior policy
bandit_feedback = dataset.obtain_batch_bandit_feedback(random_state=12345)

# obtain action choice probabilities by an evaluation policy and its ground-truth policy value
action_dist_a = dataset.obtain_action_dist_by_eval_policy(
    base_classifier_e=LogisticRegression(C=100, random_state=12345, max_iter=10000),
    alpha_e=0.9
)
ground_truth_a = dataset.calc_ground_truth_policy_value(action_dist=action_dist_a)
action_dist_b = dataset.obtain_action_dist_by_eval_policy(
    base_classifier_e=RandomForest(n_estimators=100, min_samples_split=5, random_state=12345),
    alpha_e=0.9
)
ground_truth_b = dataset.calc_ground_truth_policy_value(action_dist=action_dist_b)
```

<!-- #region id="xDmLkCTnJqXi" -->
### Setting Hyperparameter Spaces for Off-Policy Evaluation

An integral aspect of IEOE is the different sources of variance. The main sources of variance are evaluation policies, random states, hyperparameters of OPE estimators, and hyperparameters of regression models. 

In this step, we define the spaces from which the hyperparameters of OPE estimators / regression models are chosen. (The evaluation policy space is defined in the previous step, and the random state space will be defined in the next step.)
<!-- #endregion -->

```python id="FG_hpxWcJqXj"
# set hyperparameter space for ope estimators

# set hyperparameter space for the doubly robust with shrinkage estimator
# with the following code, lambda_ will be chosen from a logarithm uniform distribution over the interval [0.001, 1000]
lambda_ = {
    "lower": 1e-3,
    "upper": 1e3,
    "log": True,
    "type": float
}
dros_param = {"lambda_": lambda_}
```

```python id="fy3DSPK9JqXk"
# set hyperparameter space for logistic regression using RandomizedSearchCV

from sklearn.utils.fixes import loguniform
logistic = LogisticRegression()
distributions = {
    "C": loguniform(1e-2, 1e2)
}
clf_logistic = RandomizedSearchCV(logistic, distributions, random_state=0, n_iter=5)
```

```python id="z6_SJ6vrJqXl"
# set hyperparameter space for random forest classifier using RandomizedSearchCV

from scipy.stats import randint
randforest = RandomForest()
distributions = {
    # n_estimators will be chosen from a uniform distribution over the interval [50, 100)
    "n_estimators": randint(5e1, 1e2), 
    # max_depth will be chosen from a uniform distribution over the interval [2, 10)
    "max_depth": randint(2, 10), 
    # min_samples_split will be chosen from a uniform distribution over the interval [2, 10)
    "min_samples_split": randint(2, 10)
}
clf_randforest = RandomizedSearchCV(randforest, distributions, random_state=0, n_iter=5)
```

<!-- #region id="7T7Rg-jXJqXl" -->
### Interpretable Evaluation for Off-Policy Evaluation

With the above steps completed, we can finally conduct IEOE by utilizing the `InterpretableOPEEvaluator` class.

Here is a brief description for each parameter that can be passed into `InterpretableOPEEvaluator`:

- `random_states`: a list of integers representing the random_state used when performing OPE; corresponds to the number of iterations
- `bandit_feedback`: a list of logged bandit feedback data
- `evaluation_policies`: a list of tuples representing (ground truth policy value, action distribution)
- `ope_estimators`: a list of OPE ope_estimators
- `ope_estimator_hyperparams`: a dictionary mapping OPE estimator names to OPE estimator hyperparameter spaces defined in step 2
- `regression_models`: a list of regression regression_models
- `regression_model_hyperparams`: a dictionary mapping regression models to regression model hyperparameter spaces defined in step 2
<!-- #endregion -->

```python id="1xSx2ZwvJqXm"
# initializing class
evaluator = InterpretableOPEEvaluator(
    random_states=np.arange(100),
    bandit_feedbacks=[bandit_feedback],
    evaluation_policies=[
        (ground_truth_a, action_dist_a), 
        (ground_truth_b, action_dist_b)
    ],
    ope_estimators=[
        DirectMethod(),
        DoublyRobust(),
        DoublyRobustWithShrinkage(),
        InverseProbabilityWeighting(), 
    ],
    ope_estimator_hyperparams={
        DoublyRobustWithShrinkage.estimator_name: dros_param,
    },
    regression_models=[
        clf_logistic,
        clf_randforest
    ],
    pscore_estimators=[
        clf_logistic,
        clf_randforest
    ]
)
```

<!-- #region id="sv_kCRYGJqXm" -->
Once we have initialized `InterpretableOPEEvaluator`, we can call implemented methods to perform IEOE.
<!-- #endregion -->

```python id="jGujF_feJqXn" outputId="642095bc-6cdc-473f-b143-3613441a320e"
# estimate policy values
# we obtain a dictionary mapping ope estimator names to np.ndarray storing the estimated policy value for each iteration
policy_value = evaluator.estimate_policy_value()
```

```python id="yfJ6E95TJqXn" outputId="ada81cd1-9fc5-407f-c03b-0f14b8b159d6"
print("dm:", policy_value["dm"][:3])
print("dr:", policy_value["dr"][:3])
print("dr-os:", policy_value["dr-os"][:3])
print("ipw:", policy_value["ipw"][:3])
```

```python id="KKQtHYxCJqXo"
# compute squared errors
# we obtain a dictionary mapping ope estimator names to np.ndarray storing the calculated squared error for each iteration
squared_error = evaluator.calculate_squared_error()
```

```python id="Ni5mcDu7JqXo" outputId="91e336ff-5915-4842-d6f0-c1255b36d016"
print("dm:", squared_error["dm"][:3])
print("dr:", squared_error["dr"][:3])
print("dr-os:", squared_error["dr-os"][:3])
print("ipw:", squared_error["ipw"][:3])
```

```python id="2aXY2t2FJqXp" outputId="69ab5fea-137b-44b6-a162-b7f147cf52c7"
# visualize cdf of squared errors for all ope estimators
evaluator.visualize_cdf_aggregate(xmax=0.04)
```

```python id="l_JfY0PVJqXq" outputId="68a06243-7692-494f-b723-92447c860e4d"
# compute the au-cdf score (area under cdf of squared error over interval [0, thershold]), higher score is better
# we obtain a dictionary mapping ope estimator names to cvar scores 
au_cdf = evaluator.calculate_au_cdf_score(threshold=0.004)
au_cdf
```

```python id="HMMlyuvZJqXq" outputId="17fcbef6-98ca-4194-c67f-36533024a6a5"
# by activating the `scale` option, 
# we obtain au_cdf scores where the highest score is scaled to 1
au_cdf_scaled = evaluator.calculate_au_cdf_score(threshold=0.004, scale=True)
au_cdf_scaled
```

```python id="Hbzzn13kJqXr" outputId="6a61c824-45ab-4cc2-e6a8-5116080942ee"
# compute the cvar score (expected value of squared error above probability alpha), lower score is better
# we obtain a dictionary mapping ope estimator names to cvar scores 
cvar = evaluator.calculate_cvar_score(alpha=90)
cvar
```

```python id="ZvO-fTAUJqXr" outputId="f7093127-78a5-4f96-db68-2a08d8503e67"
# by activating the `scale` option, 
# we obtain cvar scores where the lowest score is scaled to 1
cvar_scaled = evaluator.calculate_cvar_score(alpha=90, scale=True)
cvar_scaled
```

```python id="wgF3P3ZWJqXs"

```
