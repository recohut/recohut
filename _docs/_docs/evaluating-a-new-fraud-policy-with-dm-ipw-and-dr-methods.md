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

<!-- #region id="PEXQVw-snLo0" -->
# Evaluating a New Fraud Policy with DM, IPW, and DR Methods
<!-- #endregion -->

<!-- #region id="V93SrzBK3Exx" -->
## IPS
<!-- #endregion -->

```python id="FMAnagid3SmI"
from typing import Callable, Dict, List
import pandas as pd
import statistics


P95_Z_SCORE = 1.96


def compute_list_stats(input: List):
    """Compute mean and P95 CI of mean for a list of floats."""
    mean = statistics.mean(input)
    std_dev = statistics.stdev(input) if len(input) > 1 else None
    ci_low = round(mean - P95_Z_SCORE * std_dev, 2) if std_dev else None
    ci_high = round(mean + P95_Z_SCORE * std_dev, 2) if std_dev else None

    return {"mean": round(mean, 2), "ci_low": ci_low, "ci_high": ci_high}


def evaluate(
    df: pd.DataFrame, action_prob_function: Callable, num_bootstrap_samples: int = 0
) -> Dict[str, Dict[str, float]]:

    results = [
        evaluate_raw(df, action_prob_function, sample=True)
        for _ in range(num_bootstrap_samples)
    ]

    if not results:
        results = [evaluate_raw(df, action_prob_function, sample=False)]

    logging_policy_rewards = [result["logging_policy"] for result in results]
    new_policy_rewards = [result["new_policy"] for result in results]

    return {
        "expected_reward_logging_policy": compute_list_stats(logging_policy_rewards),
        "expected_reward_new_policy": compute_list_stats(new_policy_rewards),
    }


def evaluate_raw(
    df: pd.DataFrame, action_prob_function: Callable, sample: bool
) -> Dict[str, float]:

    tmp_df = df.sample(df.shape[0], replace=True) if sample else df

    cum_reward_new_policy = 0
    for _, row in tmp_df.iterrows():
        action_probabilities = action_prob_function(row["context"])
        cum_reward_new_policy += (
            action_probabilities[row["action"]] / row["action_prob"]
        ) * row["reward"]

    return {
        "logging_policy": tmp_df.reward.sum() / len(tmp_df),
        "new_policy": cum_reward_new_policy / len(tmp_df),
    }
```

<!-- #region id="VHRaYu-q3Ghn" -->
### Scenario
<!-- #endregion -->

<!-- #region id="bddlO29L0Cn9" -->
Assume we have a fraud model in production that blocks transactions if the P(fraud) > 0.05.

Let's build some sample logs from that policy running in production. One thing to note, we need some basic exploration in the production logs (e.g. epsilon-greedy w/ε = 0.1). That is, 10% of the time we take a random action. Rewards represent revenue gained from allowing the transaction. A negative reward indicates the transaction was fraud and resulted in a chargeback.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 175} id="8X8VoM5g005u" executionInfo={"status": "ok", "timestamp": 1632587985909, "user_tz": -330, "elapsed": 615, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="fd6607eb-c4c4-44cd-f26b-c7c364bac579"
import pandas as pd

logs_df = pd.DataFrame([
    {"context": {"p_fraud": 0.08}, "action": "blocked", "action_prob": 0.90, "reward": 0},
    {"context": {"p_fraud": 0.03}, "action": "allowed", "action_prob": 0.90, "reward": 20},
    {"context": {"p_fraud": 0.01}, "action": "allowed", "action_prob": 0.90, "reward": 10},    
    {"context": {"p_fraud": 0.09}, "action": "allowed", "action_prob": 0.10, "reward": -20}, # only allowed due to exploration 
])

logs_df
```

<!-- #region id="e3Ef6Seb033L" -->
Now let's use IPS to score a more lenient fraud model that blocks transactions only if the P(fraud) > 0.10.

IPS requires that we know P(action | context) for the new policy. We can easily describe our new policy:
<!-- #endregion -->

```python id="IpDL7Rr51YXS"
def action_probabilities(context):
    epsilon = 0.10
    if context["p_fraud"] > 0.10:
        return {"allowed": epsilon, "blocked": 1 - epsilon}    
    
    return {"allowed": 1 - epsilon, "blocked": epsilon}
```

<!-- #region id="Xnnc9EmI1bfa" -->
We can now get the probability that the new policy takes the same action that was taken in the production logs above.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 175} id="h9eiFqfl2MJI" executionInfo={"status": "ok", "timestamp": 1632588323194, "user_tz": -330, "elapsed": 770, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="4af191f3-5f2f-452b-8264-b02703dbbaf8"
logs_df["new_action_prob"] = logs_df.apply(
    lambda row: action_probabilities(row["context"])[row["action"]],
    axis=1
)
logs_df
```

<!-- #region id="QjIYi90P2NMh" -->
We see that the new policy lets through a fraud example (row: 3) at a much higher probability. This should make the new model get penalized in offline evaluation. We also see that for row: 0, the new model has a 90% chance of allowing the transaction, but we don't have the counterfactual knowledge of whether or not this would have been a non-fraud transaction since in production this transaction was blocked. This demonstrates one of the drawbacks of offline policy evaluation, but with more data we'd ideally see a different action taken in the same situation (due to exploration).
<!-- #endregion -->

<!-- #region id="CRssjM_a2ssR" -->
Now we will score the new model using IPS:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="aRjUmUWC2vRd" executionInfo={"status": "ok", "timestamp": 1632588917379, "user_tz": -330, "elapsed": 772, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="22993b72-4df7-4b3a-b608-67194ea61044"
evaluate(logs_df, action_probabilities, num_bootstrap_samples=100)
```

<!-- #region id="5RSi1iVB4eNJ" -->
The expected reward per observation for the new policy is much worse than the logging policy (due to the observation that allowed fraud to go through (row: 3)) so we wouldn't roll out this new policy into an A/B test or production and instead should test some different policies offline.

However, the confidence intervals around the expected rewards for our old and new policies overlap. If we want to be really certain, it's might be best to gather some more data to ensure the difference is signal and not noise. In this case, fortunately, we have strong reason to suspect the new policy is worse, but these confidence intervals can be important in cases where we have less prior certainty.
<!-- #endregion -->

<!-- #region id="51nqgdt-4qLQ" -->
## DM
<!-- #endregion -->

```python id="KuBPMOyS4yTw"
from typing import NoReturn, Tuple, Dict, Callable

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn import ensemble
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.linear_model import Ridge


def fit_gbdt_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray = None,
    y_test: np.ndarray = None,
) -> Dict:
    """Off the shelf sklearn GBDT regressor."""

    clf = ensemble.GradientBoostingRegressor()
    clf.fit(X_train, y_train)

    mse_train = mean_squared_error(y_train, clf.predict(X_train))

    mse_test = None
    if X_test and y_test:
        mse_test = mean_squared_error(y_test, clf.predict(X_test))

    return {"model": clf, "mse_train": mse_train, "mse_test": mse_test}


def fit_gbdt_classification(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray = None,
    y_test: np.ndarray = None,
) -> Dict:
    """Off the shelf sklearn GBDT classifier."""

    clf = ensemble.GradientBoostingClassifier()
    clf.fit(X_train, y_train)

    acc_train = accuracy_score(y_train, clf.predict(X_train))

    acc_test = None
    if X_test and y_test:
        acc_test = accuracy_score(y_test, clf.predict(X_test))

    return {"model": clf, "acc_train": acc_train, "acc_test": acc_test}


def fit_ridge_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray = None,
    y_test: np.ndarray = None,
) -> Dict:
    """Off the shelf sklearn Ridge regression."""

    clf = Ridge()
    clf.fit(X_train, y_train)

    mse_train = mean_squared_error(y_train, clf.predict(X_train))

    mse_test = None
    if X_test and y_test:
        mse_test = mean_squared_error(y_test, clf.predict(X_test))

    return {"model": clf, "mse_train": mse_train, "mse_test": mse_test}


def fit_ridge_classification(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray = None,
    y_test: np.ndarray = None,
) -> Dict:
    """Off the shelf sklearn Ridge classifier."""

    clf = RidgeClassifier()
    clf.fit(X_train, y_train)

    acc_train = accuracy_score(y_train, clf.predict(X_train))

    acc_test = None
    if X_test and y_test:
        acc_test = accuracy_score(y_test, clf.predict(X_test))

    return {"model": clf, "acc_train": acc_train, "acc_test": acc_test}


class Predictor:
    def _preprocess_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        # preprocess context
        context_df = df.context.apply(pd.Series)
        self.context_column_order = list(context_df.columns)

        # preprocess actions
        self.action_preprocessor = OneHotEncoder(sparse=False)
        action_values = df.action.values.reshape(-1, 1)
        self.possible_actions = set(action_values.squeeze().tolist())
        one_hot_action_values = self.action_preprocessor.fit_transform(action_values)

        X_train = np.concatenate((context_df.values, one_hot_action_values), axis=1)
        y_train = df.reward.values

        return X_train, y_train

    def fit(self, df: pd.DataFrame) -> NoReturn:
        X_train, y_train = self._preprocess_data(df)
        results = fit_gbdt_regression(X_train, y_train)
        self.model = results.pop("model")
        self.training_stats = results

    def predict(self, input: np.ndarray) -> float:
        return self.model.predict(input)[0]


def evaluate(
    df: pd.DataFrame, action_prob_function: Callable, num_bootstrap_samples: int = 0
) -> Dict[str, Dict[str, float]]:

    # train a model that predicts reward given (context, action)
    reward_model = Predictor()
    reward_model.fit(df)

    results = [
        evaluate_raw(df, action_prob_function, sample=True, reward_model=reward_model)
        for _ in range(num_bootstrap_samples)
    ]

    if not results:
        results = [
            evaluate_raw(
                df, action_prob_function, sample=False, reward_model=reward_model
            )
        ]

    logging_policy_rewards = [result["logging_policy"] for result in results]
    new_policy_rewards = [result["new_policy"] for result in results]

    return {
        "expected_reward_logging_policy": compute_list_stats(logging_policy_rewards),
        "expected_reward_new_policy": compute_list_stats(new_policy_rewards),
    }


def evaluate_raw(
    df: pd.DataFrame,
    action_prob_function: Callable,
    sample: bool,
    reward_model: Predictor,
) -> Dict[str, float]:

    tmp_df = df.sample(df.shape[0], replace=True) if sample else df

    context_df = tmp_df.context.apply(pd.Series)
    context_array = context_df[reward_model.context_column_order].values
    cum_reward_new_policy = 0

    for idx, row in tmp_df.iterrows():
        observation_expected_reward = 0
        action_probabilities = action_prob_function(row["context"])
        for action, action_probability in action_probabilities.items():
            one_hot_action = reward_model.action_preprocessor.transform(
                np.array(action).reshape(-1, 1)
            )
            observation = np.concatenate((context_array[idx], one_hot_action.squeeze()))
            predicted_reward = reward_model.predict(observation.reshape(1, -1))
            observation_expected_reward += action_probability * predicted_reward
        cum_reward_new_policy += observation_expected_reward

    return {
        "logging_policy": tmp_df.reward.sum() / len(tmp_df),
        "new_policy": cum_reward_new_policy / len(tmp_df),
    }
```

<!-- #region id="PRqNkkRA4w2o" -->
### Scenario
<!-- #endregion -->

<!-- #region id="3j7dWVMK-pRV" -->
Assume we have a fraud model in production that blocks transactions if the P(fraud) > 0.05.

Let's build some sample logs from that policy running in production. One thing to note, we need some basic exploration in the production logs (e.g. epsilon-greedy w/ε = 0.1). That is, 10% of the time we take a random action. Rewards represent revenue gained from allowing the transaction. A negative reward indicates the transaction was fraud and resulted in a chargeback.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 238} id="Vh4ybGYJ-aOQ" executionInfo={"status": "ok", "timestamp": 1632590505641, "user_tz": -330, "elapsed": 917, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="17ee1273-e949-4a04-be3e-705da5c5ed40"
logs_df = pd.DataFrame([
    {"context": {"p_fraud": 0.08}, "action": "blocked", "action_prob": 0.90, "reward": 0},
    {"context": {"p_fraud": 0.03}, "action": "allowed", "action_prob": 0.90, "reward": 20},
    {"context": {"p_fraud": 0.02}, "action": "allowed", "action_prob": 0.90, "reward": 10}, 
    {"context": {"p_fraud": 0.01}, "action": "allowed", "action_prob": 0.90, "reward": 20},     
    {"context": {"p_fraud": 0.09}, "action": "allowed", "action_prob": 0.10, "reward": -20}, # only allowed due to exploration 
    {"context": {"p_fraud": 0.40}, "action": "allowed", "action_prob": 0.10, "reward": -10}, # only allowed due to exploration     
])

logs_df
```

<!-- #region id="ORLinrUq-h_B" -->
Now let's use the direct method to score a more lenient fraud model that blocks transactions only if the P(fraud) > 0.10.

The direct method requires that we have a function that computes P(action | context)for all possible actions under our new policy. We can define that for our new policy easily here:
<!-- #endregion -->

```python id="oWecXmKE-xCu"
def action_probabilities(context):
    epsilon = 0.10
    if context["p_fraud"] > 0.10:
        return {"allowed": epsilon, "blocked": 1 - epsilon}    
    
    return {"allowed": 1 - epsilon, "blocked": epsilon}
```

<!-- #region id="SZRzqWLY-2zX" -->
We will use the same production logs above and run them through the new policy:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="XRzWhvd_-34u" executionInfo={"status": "ok", "timestamp": 1632590710659, "user_tz": -330, "elapsed": 1563, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="fd5c2a95-a9bb-4799-b2be-873c5d79947f"
evaluate(logs_df, action_probabilities, num_bootstrap_samples=100)
```

<!-- #region id="99nE9Gyi-6HQ" -->
The direct method estimates that the expected reward per observation for the new policy is slightly better than the logging policy so we would think about rolling out this new policy into an A/B test or production.

However, the confidence intervals around the expected rewards for our old and new policies overlap heavily. If we want to be really certain, it's might be best to gather some more data to ensure the difference is signal and not noise. In this case, fortunately, we have strong reason to suspect the new policy is worse, but these confidence intervals can be important in cases where we have less prior certainty.
<!-- #endregion -->

<!-- #region id="3vep9LPS_5fa" -->
## DR
<!-- #endregion -->

```python id="SbZtj3rLAAG5"
from typing import NoReturn, Tuple, Dict, Callable

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn import ensemble
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.linear_model import Ridge


def fit_gbdt_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray = None,
    y_test: np.ndarray = None,
) -> Dict:
    """Off the shelf sklearn GBDT regressor."""

    clf = ensemble.GradientBoostingRegressor()
    clf.fit(X_train, y_train)

    mse_train = mean_squared_error(y_train, clf.predict(X_train))

    mse_test = None
    if X_test and y_test:
        mse_test = mean_squared_error(y_test, clf.predict(X_test))

    return {"model": clf, "mse_train": mse_train, "mse_test": mse_test}


def fit_gbdt_classification(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray = None,
    y_test: np.ndarray = None,
) -> Dict:
    """Off the shelf sklearn GBDT classifier."""

    clf = ensemble.GradientBoostingClassifier()
    clf.fit(X_train, y_train)

    acc_train = accuracy_score(y_train, clf.predict(X_train))

    acc_test = None
    if X_test and y_test:
        acc_test = accuracy_score(y_test, clf.predict(X_test))

    return {"model": clf, "acc_train": acc_train, "acc_test": acc_test}


def fit_ridge_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray = None,
    y_test: np.ndarray = None,
) -> Dict:
    """Off the shelf sklearn Ridge regression."""

    clf = Ridge()
    clf.fit(X_train, y_train)

    mse_train = mean_squared_error(y_train, clf.predict(X_train))

    mse_test = None
    if X_test and y_test:
        mse_test = mean_squared_error(y_test, clf.predict(X_test))

    return {"model": clf, "mse_train": mse_train, "mse_test": mse_test}


def fit_ridge_classification(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray = None,
    y_test: np.ndarray = None,
) -> Dict:
    """Off the shelf sklearn Ridge classifier."""

    clf = RidgeClassifier()
    clf.fit(X_train, y_train)

    acc_train = accuracy_score(y_train, clf.predict(X_train))

    acc_test = None
    if X_test and y_test:
        acc_test = accuracy_score(y_test, clf.predict(X_test))

    return {"model": clf, "acc_train": acc_train, "acc_test": acc_test}


class Predictor:
    def _preprocess_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        # preprocess context
        context_df = df.context.apply(pd.Series)
        self.context_column_order = list(context_df.columns)

        # preprocess actions
        self.action_preprocessor = OneHotEncoder(sparse=False)
        action_values = df.action.values.reshape(-1, 1)
        self.possible_actions = set(action_values.squeeze().tolist())
        one_hot_action_values = self.action_preprocessor.fit_transform(action_values)

        X_train = np.concatenate((context_df.values, one_hot_action_values), axis=1)
        y_train = df.reward.values

        return X_train, y_train

    def fit(self, df: pd.DataFrame) -> NoReturn:
        X_train, y_train = self._preprocess_data(df)
        results = fit_gbdt_regression(X_train, y_train)
        self.model = results.pop("model")
        self.training_stats = results

    def predict(self, input: np.ndarray) -> float:
        return self.model.predict(input)[0]


def evaluate(
    df: pd.DataFrame, action_prob_function: Callable, num_bootstrap_samples: int = 0
) -> Dict[str, Dict[str, float]]:

    # train a model that predicts reward given (context, action)
    reward_model = Predictor()
    reward_model.fit(df)

    results = [
        evaluate_raw(df, action_prob_function, sample=True, reward_model=reward_model)
        for _ in range(num_bootstrap_samples)
    ]

    if not results:
        results = [
            evaluate_raw(
                df, action_prob_function, sample=False, reward_model=reward_model
            )
        ]

    logging_policy_rewards = [result["logging_policy"] for result in results]
    new_policy_rewards = [result["new_policy"] for result in results]

    return {
        "expected_reward_logging_policy": compute_list_stats(logging_policy_rewards),
        "expected_reward_new_policy": compute_list_stats(new_policy_rewards),
    }


def evaluate_raw(
    df: pd.DataFrame,
    action_prob_function: Callable,
    sample: bool,
    reward_model: Predictor,
) -> Dict[str, float]:

    tmp_df = df.sample(df.shape[0], replace=True) if sample else df

    context_df = tmp_df.context.apply(pd.Series)
    context_array = context_df[reward_model.context_column_order].values
    cum_reward_new_policy = 0

    for idx, row in tmp_df.iterrows():
        observation_expected_reward = 0
        processed_context = context_array[idx]

        # first compute the left hand term, which is the direct method
        action_probabilities = action_prob_function(row["context"])
        for action, action_probability in action_probabilities.items():
            one_hot_action = reward_model.action_preprocessor.transform(
                np.array(action).reshape(-1, 1)
            )
            observation = np.concatenate((processed_context, one_hot_action.squeeze()))
            predicted_reward = reward_model.predict(observation.reshape(1, -1))
            observation_expected_reward += action_probability * predicted_reward

        # then compute the right hand term, which is similar to IPS
        logged_action = row["action"]
        new_action_probability = action_probabilities[logged_action]
        weight = new_action_probability / row["action_prob"]
        one_hot_action = reward_model.action_preprocessor.transform(
            np.array(row["action"]).reshape(-1, 1)
        )
        observation = np.concatenate((processed_context, one_hot_action.squeeze()))
        predicted_reward = reward_model.predict(observation.reshape(1, -1))
        observation_expected_reward += weight * (row["reward"] - predicted_reward)

        cum_reward_new_policy += observation_expected_reward

    return {
        "logging_policy": tmp_df.reward.sum() / len(tmp_df),
        "new_policy": cum_reward_new_policy / len(tmp_df),
    }
```

<!-- #region id="cztyBE9Z_-ej" -->
### Scenario
<!-- #endregion -->

<!-- #region id="nE4_lmcBAY1r" -->
Assume we have a fraud model in production that blocks transactions if the P(fraud) > 0.05.

Let's build some sample logs from that policy running in production. One thing to note, we need some basic exploration in the production logs (e.g. epsilon-greedy w/ε = 0.1). That is, 10% of the time we take a random action. Rewards represent revenue gained from allowing the transaction. A negative reward indicates the transaction was fraud and resulted in a chargeback.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 238} id="_MN5k5qJAY18" executionInfo={"status": "ok", "timestamp": 1632590505641, "user_tz": -330, "elapsed": 917, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="17ee1273-e949-4a04-be3e-705da5c5ed40"
logs_df = pd.DataFrame([
    {"context": {"p_fraud": 0.08}, "action": "blocked", "action_prob": 0.90, "reward": 0},
    {"context": {"p_fraud": 0.03}, "action": "allowed", "action_prob": 0.90, "reward": 20},
    {"context": {"p_fraud": 0.02}, "action": "allowed", "action_prob": 0.90, "reward": 10}, 
    {"context": {"p_fraud": 0.01}, "action": "allowed", "action_prob": 0.90, "reward": 20},     
    {"context": {"p_fraud": 0.09}, "action": "allowed", "action_prob": 0.10, "reward": -20}, # only allowed due to exploration 
    {"context": {"p_fraud": 0.40}, "action": "allowed", "action_prob": 0.10, "reward": -10}, # only allowed due to exploration     
])

logs_df
```

<!-- #region id="HlboiuS9AZ9w" -->
Now let's use the doubly robust method to score a more lenient fraud model that blocks transactions only if the P(fraud) > 0.10.

The doubly robust method requires that we have a function that computes P(action | context)for all possible actions under our new policy. We can define that for our new policy easily here:
<!-- #endregion -->

```python id="PQc2rRODAd3y"
def action_probabilities(context):
    epsilon = 0.10
    if context["p_fraud"] > 0.10:
        return {"allowed": epsilon, "blocked": 1 - epsilon}    
    
    return {"allowed": 1 - epsilon, "blocked": epsilon}
```

<!-- #region id="173ugbysAfM7" -->
We will use the same production logs above and run them through the new policy.


<!-- #endregion -->

```python id="byvcpV4KAhaD" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1632591034455, "user_tz": -330, "elapsed": 1751, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="d2d08b58-08f6-4ee8-d8aa-cc6c6053af31"
evaluate(logs_df, action_probabilities, num_bootstrap_samples=50)
```

<!-- #region id="35UdAVZVAi4c" -->
The doubly robust method estimates that the expected reward per observation for the new policy is much worse than the logging policy so we wouldn't roll out this new policy into an A/B test or production and instead should test some different policies offline.

However, the confidence intervals around the expected rewards for our old and new policies overlap heavily. If we want to be really certain, it's might be best to gather some more data to ensure the difference is signal and not noise. In this case, fortunately, we have strong reason to suspect the new policy is worse, but these confidence intervals can be important in cases where we have less prior certainty.
<!-- #endregion -->
