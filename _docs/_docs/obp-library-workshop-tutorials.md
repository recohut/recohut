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

<!-- #region id="jnjxGf9X3X-x" -->
# OBP Library Workshop Tutorials
<!-- #endregion -->

<!-- #region id="XHI4nfS4x27P" -->
## Synthetic Data
<!-- #endregion -->

```python id="tXGoiwPewTHR"
!pip install obp==0.5.1
!pip install matplotlib==3.1.1
```

```python id="99mNopXTwaIk" executionInfo={"status": "ok", "timestamp": 1633593740260, "user_tz": -330, "elapsed": 4867, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
from sklearn.linear_model import LogisticRegression

import obp
from obp.dataset import (
    SyntheticBanditDataset,
    logistic_reward_function,
    linear_behavior_policy,
)
from obp.policy import IPWLearner
from obp.ope import (
    OffPolicyEvaluation, 
    RegressionModel,
    InverseProbabilityWeighting as IPS,
    DirectMethod as DM,
    DoublyRobust as DR,
)
```

<!-- #region id="SpnDzvB9xDcX" -->
### Generate Synthetic Dataset
<!-- #endregion -->

<!-- #region id="Uwz5ez9kz66R" -->
`obp.dataset.SyntheticBanditDataset` is an easy-to-use synthetic data generator.

It takes 
- number of actions (`n_actions`, $|\mathcal{A}|$)
- dimension of context vectors (`dim_context`, $d$)
- reward function (`reward_function`, $q(x,a)=\mathbb{E}[r \mid x,a]$)
- behavior policy (`behavior_policy_function`, $\pi_b(a|x)$) 

as inputs and generates synthetic logged bandit data that can be used to evaluate the performance of decision making policies (obtained by `off-policy learning`) and OPE estimators.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="_t-rilQ5xkiJ" executionInfo={"status": "ok", "timestamp": 1633593760583, "user_tz": -330, "elapsed": 2296, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="7b485b88-295c-468a-a181-083a7b22ea5c"
# generate synthetic logged bandit data with 10 actions
# we use `logistic function` as the reward function and `linear_behavior_policy` as the behavior policy.
# one can define their own reward function and behavior policy such as nonlinear ones. 
dataset = SyntheticBanditDataset(
    n_actions=10, # number of actions; |A|
    dim_context=5, # number of dimensions of context vector
    reward_function=logistic_reward_function, # mean reward function; q(x,a)
    behavior_policy_function=linear_behavior_policy, # behavior policy; \pi_b
    random_state=12345,

)
training_bandit_data = dataset.obtain_batch_bandit_feedback(n_rounds=10000)
test_bandit_data = dataset.obtain_batch_bandit_feedback(n_rounds=1000000)

training_bandit_data
```

<!-- #region id="vDkPg2moz66b" -->
the logged bandit feedback is collected by the behavior policy as follows.

$ \mathcal{D}_b := \{(x_i,a_i,r_i)\}$  where $(x,a,r) \sim p(x)\pi_b(a \mid x)p(r \mid x,a) $
<!-- #endregion -->

<!-- #region id="vanoqJyHxo90" -->
### Train Bandit Policies (OPL)
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="Ka081Ovyx9um" executionInfo={"status": "ok", "timestamp": 1633593867524, "user_tz": -330, "elapsed": 731, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="d9de27e0-6bd8-4e30-afbd-e66af1fec406"
ipw_learner = IPWLearner(
    n_actions=dataset.n_actions, # number of actions; |A|
    base_classifier=LogisticRegression(C=100, random_state=12345) # any sklearn classifier
)

# fit
ipw_learner.fit(
    context=training_bandit_data["context"], # context; x
    action=training_bandit_data["action"], # action; a
    reward=training_bandit_data["reward"], # reward; r
    pscore=training_bandit_data["pscore"], # propensity score; pi_b(a|x)
)

# predict (action dist = action distribution)
action_dist_ipw = ipw_learner.predict(
    context=test_bandit_data["context"], # context in the test data
)

action_dist_ipw[:, :, 0] # which action to take for each context 
```

<!-- #region id="yGntfcS0yDas" -->
### Approximate the Ground-truth Policy Value
<!-- #endregion -->

<!-- #region id="wY_JqooAyH7a" -->
$$ V(\pi) \approx \frac{1}{|\mathcal{D}_{te}|} \sum_{i=1}^{|\mathcal{D}_{te}|} \mathbb{E}_{a \sim \pi(a|x_i)} [r(x_i, a)], \; \, where \; \, r(x,a) := \mathbb{E}_{r \sim p(r|x,a)} [r] $$
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="k2L3ERg3yN_d" executionInfo={"status": "ok", "timestamp": 1633593938688, "user_tz": -330, "elapsed": 497, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="37c95dc6-ff5e-4c67-8410-2deeff888e70"
policy_value_of_ipw = dataset.calc_ground_truth_policy_value(
    expected_reward=test_bandit_data["expected_reward"], # expected rewards; q(x,a)
    action_dist=action_dist_ipw, # action distribution of IPWLearner
)

# ground-truth policy value of `IPWLearner`
policy_value_of_ipw
```

<!-- #region id="0CQOwojeylfs" -->
### Obtain a Reward Estimator

obp.ope.RegressionModel simplifies the process of reward modeling

$ r(x,a)=E[r∣x,a]≈r^(x,a) $
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="35Ox3I5yyy5M" executionInfo={"status": "ok", "timestamp": 1633594069697, "user_tz": -330, "elapsed": 6624, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="f7a778d9-2dae-4d35-920f-a124d9d6d45f"
# estimate the expected reward by using an ML model (Logistic Regression here)
# the estimated rewards are used by model-dependent estimators such as DM and DR
regression_model = RegressionModel(
    n_actions=dataset.n_actions, # number of actions; |A|
    base_model=LogisticRegression(C=100, random_state=12345) # any sklearn classifier
)

estimated_rewards = regression_model.fit_predict(
    context=test_bandit_data["context"], # context; x
    action=test_bandit_data["action"], # action; a
    reward=test_bandit_data["reward"], # reward; r
    random_state=12345,
)

estimated_rewards[:, :, 0] # \hat{q}(x,a)
```

<!-- #region id="xk7mBN1Az66y" -->
please refer to https://arxiv.org/abs/2002.08536 about the details of the cross-fitting procedure.
<!-- #endregion -->

<!-- #region id="f-GTRNZ7yU41" -->
### Off-Policy Evaluation (OPE)

obp.ope.OffPolicyEvaluation simplifies the OPE process

$ V(πe)≈\hat{V}(πe;D0,θ) $ using DM, IPS, and DR
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="gdlxOm2BycmY" executionInfo={"status": "ok", "timestamp": 1633594141946, "user_tz": -330, "elapsed": 1204, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="514b8ab8-81a4-4b30-8488-a41894fa466b"
ope = OffPolicyEvaluation(
    bandit_feedback=test_bandit_data, # test data
    ope_estimators=[
        IPS(estimator_name="IPS"), 
        DM(estimator_name="DM"), 
        DR(estimator_name="DR"),
    ] # used estimators
)

estimated_policy_value = ope.estimate_policy_values(
    action_dist=action_dist_ipw, # \pi_e(a|x)
    estimated_rewards_by_reg_model=estimated_rewards, # \hat{q}(x,a)
)

# OPE results given by the three estimators
estimated_policy_value
```

```python colab={"base_uri": "https://localhost:8080/", "height": 505} id="MxgHeuJW02ta" executionInfo={"status": "ok", "timestamp": 1633594774614, "user_tz": -330, "elapsed": 76248, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="45927ff5-db15-4508-8920-d4cb7fd2727b"
# estimate the policy value of IPWLearner with Logistic Regression
estimated_policy_value_a, estimated_interval_a = ope.summarize_off_policy_estimates(
    action_dist=action_dist_ipw,
    estimated_rewards_by_reg_model=estimated_rewards
)
print(estimated_interval_a, '\n')

# visualize policy values of IPWLearner with Logistic Regression estimated by the three OPE estimators
ope.visualize_off_policy_estimates(
    action_dist=action_dist_ipw,
    estimated_rewards_by_reg_model=estimated_rewards,
    n_bootstrap_samples=1000, # number of resampling performed in the bootstrap procedure
    random_state=12345,
)
```

<!-- #region id="Sr7mcmE0zFwL" -->
### Evaluation of OPE

Now, let's evaluate the OPE performance (estimation accuracy) of the three estimators

$ V(\pi_e) \approx \hat{V} (\pi_e; \mathcal{D}_0, \theta) $
<!-- #endregion -->

<!-- #region id="jsb6iebez67A" -->
We can then evaluate the estimation performance of OPE estimators by comparing the estimated policy values of the evaluation with its ground-truth as follows.

- $\textit{relative-ee} (\hat{V}; \mathcal{D}_b) := \left| \frac{V(\pi_e) - \hat{V} (\pi_e; \mathcal{D}_b)}{V(\pi_e)} \right|$ (relative estimation error; relative-ee)
- $\textit{SE} (\hat{V}; \mathcal{D}_b) := \left( V(\pi_e) - \hat{V} (\pi_e; \mathcal{D}_b) \right)^2$ (squared error; se)
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="MivIYtwQzKQN" executionInfo={"status": "ok", "timestamp": 1633594240398, "user_tz": -330, "elapsed": 1357, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="f81ec81f-3cd3-40b9-f1c5-6610f19b0fdf"
squared_errors = ope.evaluate_performance_of_estimators(
    ground_truth_policy_value=policy_value_of_ipw, # V(\pi_e)
    action_dist=action_dist_ipw, # \pi_e(a|x)
    estimated_rewards_by_reg_model=estimated_rewards, # \hat{q}(x,a)
    metric="se", # squared error
)

squared_errors # DR is the most accurate
```

<!-- #region id="jqCPYLv3zeV_" -->
We can iterate the above process several times and calculate the following MSE

$ MSE (\hat{V}) := T^{-1} \sum_{t=1}^T SE (\hat{V}; \mathcal{D}_0^{(t)}) $ 

where $ \mathcal{D}_0^{(t)} $ is the synthetic data in the t-th iteration
<!-- #endregion -->

<!-- #region id="_bdOKiBPN0sJ" -->
## Off-Policy Simulation Data
<!-- #endregion -->

<!-- #region id="xG5Qn1-xOJ0W" -->
### Imports
<!-- #endregion -->

```python id="Uy5zbTVsOQbk"
import time

import numpy as np
import pandas as pd
from pandas import DataFrame
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import obp
from obp.dataset import (
    SyntheticBanditDataset,
    logistic_reward_function,
    linear_behavior_policy
)
from obp.policy import IPWLearner
from obp.ope import (
    OffPolicyEvaluation, 
    RegressionModel,
    InverseProbabilityWeighting as IPS,
    SelfNormalizedInverseProbabilityWeighting as SNIPS,
    DirectMethod as DM,
    DoublyRobust as DR,
    DoublyRobustWithShrinkage as DRos,
    DoublyRobustWithShrinkageTuning as DRosTuning,
)
from obp.utils import softmax

import warnings
warnings.simplefilter('ignore')
```

<!-- #region id="8La7ngvUN32f" -->
### Evaluatin of OPE estimators (Part 1; easy setting)
<!-- #endregion -->

```python id="5_b3-qQPN33w"
### configurations
num_runs = 200
num_data_list = [100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200]
```

```python id="0wQwntVDN330"
### define a dataset class
dataset = SyntheticBanditDataset(
    n_actions=10, 
    dim_context=10,
    tau=0.2, 
    reward_function=logistic_reward_function,
    behavior_policy_function=linear_behavior_policy,
    random_state=12345,
)
#### training data is used to train an evaluation policy
train_bandit_data = dataset.obtain_batch_bandit_feedback(n_rounds=5000)
#### test bandit data is used to approximate the ground-truth policy value
test_bandit_data = dataset.obtain_batch_bandit_feedback(n_rounds=100000)
```

```python id="ZvwAtO8SN333"
### evaluation policy training
ipw_learner = IPWLearner(
    n_actions=dataset.n_actions,
    base_classifier=RandomForestClassifier(random_state=12345),
)
ipw_learner.fit(
    context=train_bandit_data["context"], 
    action=train_bandit_data["action"], 
    reward=train_bandit_data["reward"], 
    pscore=train_bandit_data["pscore"], 
)
action_dist_ipw_test = ipw_learner.predict(
    context=test_bandit_data["context"],
)
policy_value_of_ipw = dataset.calc_ground_truth_policy_value(
    expected_reward=test_bandit_data["expected_reward"], 
    action_dist=action_dist_ipw_test, 
)
```

```python id="WqZmJyC3N33_" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1633535191802, "user_tz": -330, "elapsed": 825141, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="190688ce-d2ab-4225-8915-b336f366253b"
### evaluation of OPE estimators
se_df_list = []
for num_data in num_data_list:
    se_list = []
    for _ in tqdm(range(num_runs), desc=f"num_data={num_data}..."):
        ### generate validation data
        validation_bandit_data = dataset.obtain_batch_bandit_feedback(
            n_rounds=num_data
        )

        ### make decisions on vlidation data
        action_dist_ipw_val = ipw_learner.predict(
            context=validation_bandit_data["context"],
        )

        ### OPE using validation data
        regression_model = RegressionModel(
            n_actions=dataset.n_actions, 
            base_model=LogisticRegression(C=100, max_iter=10000, random_state=12345),
        )
        estimated_rewards = regression_model.fit_predict(
            context=validation_bandit_data["context"], # context; x
            action=validation_bandit_data["action"], # action; a
            reward=validation_bandit_data["reward"], # reward; r
            n_folds=2, # 2-fold cross fitting
            random_state=12345,
        )
        ope = OffPolicyEvaluation(
            bandit_feedback=validation_bandit_data,
            ope_estimators=[
                IPS(estimator_name="IPS"), 
                DM(estimator_name="DM"), 
                IPS(lambda_=5, estimator_name="CIPS"), 
                SNIPS(estimator_name="SNIPS"),
                DR(estimator_name="DR"), 
                DRos(lambda_=500, estimator_name="DRos"), 
            ]
        )
        squared_errors = ope.evaluate_performance_of_estimators(
            ground_truth_policy_value=policy_value_of_ipw, # V(\pi_e)
            action_dist=action_dist_ipw_val, # \pi_e(a|x)
            estimated_rewards_by_reg_model=estimated_rewards, # \hat{q}(x,a)
            metric="se", # squared error
        )
        se_list.append(squared_errors)
    ### maximum importance weight in the validation data
    #### a larger value indicates that the logging and evaluation policies are greatly different
    max_iw = (action_dist_ipw_val[
        np.arange(validation_bandit_data["n_rounds"]), 
        validation_bandit_data["action"], 
        0
    ] / validation_bandit_data["pscore"]).max()
    tqdm.write(f"maximum importance weight={np.round(max_iw, 5)}\n")

    ### summarize results
    se_df = DataFrame(DataFrame(se_list).stack())\
        .reset_index(1).rename(columns={"level_1": "est", 0: "se"})
    se_df["num_data"] = num_data
    se_df_list.append(se_df)
    tqdm.write("=====" * 15)
    time.sleep(0.5)

# aggregate all results 
result_df = pd.concat(se_df_list).reset_index(level=0)
```

<!-- #region id="As9PXk4pN34H" -->
### Visualize Results
<!-- #endregion -->

```python id="F0VUMWW7N34R"
# figure configs
query = "(est == 'DM' or est == 'IPS') and num_data <= 6400"
xlabels = [100, 400, 1600, 6400]

plt.style.use('ggplot')
fig, ax = plt.subplots(figsize=(10, 7), tight_layout=True)
sns.lineplot(
    linewidth=5,
    dashes=False,
    legend=False,
    x="num_data",
    y="se",
    hue="est",
    ax=ax,
    data=result_df.query("(est == 'DM' or est == 'IPS') and num_data <= 12800"),
)
# title and legend
ax.legend(["IPS", "DM"], loc="upper right", fontsize=25)
# yaxis
ax.set_yscale("log")
ax.set_ylabel("mean squared error (MSE)", fontsize=25)
ax.tick_params(axis="y", labelsize=15)
ax.yaxis.set_label_coords(-0.08, 0.5)
# xaxis
ax.set_xscale("log")
ax.set_xlabel("number of samples in the log data", fontsize=25)
ax.set_xticks(xlabels)
ax.set_xticklabels(xlabels, fontsize=15)
ax.xaxis.set_label_coords(0.5, -0.1)
```

```python id="M656Z7u2N34W" outputId="58ccd008-3213-44b4-d1eb-f5ce037421ee"
# figure configs
query = "(est == 'DM' or est == 'CIPS' or est == 'IPS' or est == 'SNIPS')" 
query += "and num_data <= 6400"
xlabels = [100, 400, 1600, 6400]

plt.style.use('ggplot')
fig, ax = plt.subplots(figsize=(10, 7), tight_layout=True)
sns.lineplot(
    linewidth=5,
    dashes=False,
    legend=False,
    x="num_data",
    y="se",
    hue="est",
    ax=ax,
    data=result_df.query(query),
)
# title and legend
ax.legend(["IPS", "DM", "CIPS", "SNIPS"], loc="upper right", fontsize=22)
# yaxis
ax.set_yscale("log")
ax.set_ylabel("mean squared error (MSE)", fontsize=25)
ax.tick_params(axis="y", labelsize=15)
ax.yaxis.set_label_coords(-0.08, 0.5)
# xaxis
ax.set_xscale("log")
ax.set_xlabel("number of samples in the log data", fontsize=25)
ax.set_xticks(xlabels)
ax.set_xticklabels(xlabels, fontsize=15)
ax.xaxis.set_label_coords(0.5, -0.1)
```

```python id="_3jY_yPdN34b" outputId="e52f8ac4-f341-462e-d999-e858443f667a"
# figure configs
query = "(est == 'DM' or est == 'IPS' or est == 'SNIPS' or est == 'CIPS' or est == 'DR')"
query += "and num_data <= 6400"
xlabels = [100, 400, 1600, 6400]

plt.style.use('ggplot')
fig, ax = plt.subplots(figsize=(10, 7), tight_layout=True)
sns.lineplot(
    linewidth=5,
    dashes=False,
    legend=False,
    x="num_data",
    y="se",
    hue="est",
    ax=ax,
    data=result_df.query(query),
)
# title and legend
ax.legend(["IPS", "DM", "CIPS", "SNIPS", "DR"], loc="upper right", fontsize=22)
# yaxis
ax.set_yscale("log")
ax.set_ylim(1e-4, 1)
ax.set_ylabel("mean squared error (MSE)", fontsize=25)
ax.tick_params(axis="y", labelsize=15)
ax.yaxis.set_label_coords(-0.08, 0.5)
# xaxis
ax.set_xscale("log")
ax.set_xlabel("number of samples in the log data", fontsize=25)
ax.set_xticks(xlabels)
ax.set_xticklabels(xlabels, fontsize=15)
ax.xaxis.set_label_coords(0.5, -0.1)
```

```python id="WfMzllbBN34f" outputId="53561d46-166f-4428-98f5-74f7e53ac3c3"
# figure configs
query = "num_data <= 6400"
xlabels = [100, 400, 1600, 6400]

plt.style.use('ggplot')
fig, ax = plt.subplots(figsize=(10, 7), tight_layout=True)
sns.lineplot(
    linewidth=4,
    dashes=False,
    legend=False,
    x="num_data",
    y="se",
    hue="est",
    ax=ax,
    data=result_df.query(query),
)
# title and legend
ax.legend(["IPS", "DM", "CIPS", "SNIPS", "DR", "DRos"], loc="upper right", fontsize=20)
# yaxis
ax.set_yscale("log")
ax.set_ylim(1e-4, 1)
ax.set_ylabel("mean squared error (MSE)", fontsize=25)
ax.tick_params(axis="y", labelsize=15)
ax.yaxis.set_label_coords(-0.08, 0.5)
# xaxis
ax.set_xscale("log")
ax.set_xlabel("number of samples in the log data", fontsize=25)
ax.set_xticks(xlabels)
ax.set_xticklabels(xlabels, fontsize=15)
ax.xaxis.set_label_coords(0.5, -0.1)
```

```python id="12FHzJyUN34h" outputId="2f5c9863-89c8-47c8-e001-f46fdfbdb04d"
xlabels = [100, 6400, 25600, 51200]

plt.style.use('ggplot')
fig, ax = plt.subplots(figsize=(10, 7), tight_layout=True)
sns.lineplot(
    linewidth=5,
    dashes=False,
    legend=False,
    x="num_data",
    y="se",
    hue="est",
    ax=ax,
    data=result_df,
)
# title and legend
ax.legend(["IPS", "DM", "CIPS", "SNIPS", "DR", "DRos"], loc="upper right", fontsize=20)
# yaxis
ax.set_yscale("log")
ax.set_ylabel("mean squared error (MSE)", fontsize=25)
ax.tick_params(axis="y", labelsize=15)
ax.yaxis.set_label_coords(-0.08, 0.5)
# xaxis
ax.set_xlabel("number of samples in the log data", fontsize=25)
ax.set_xticks(xlabels)
ax.set_xticklabels(xlabels, fontsize=15)
ax.xaxis.set_label_coords(0.5, -0.1)
```

<!-- #region id="Ovbakc27N4GS" -->
### Evaluation of OPE estimators (Part 2; challenging setting)
<!-- #endregion -->

```python id="SUVh7rtZN4Gg"
### configurations
num_runs = 200
num_data_list = [100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200]
```

```python id="EyDcDyWjN4Gh"
### define a dataset class
dataset = SyntheticBanditDataset(
    n_actions=10, 
    dim_context=10,
    tau=0.2, 
    reward_function=logistic_reward_function,
    behavior_policy_function=linear_behavior_policy,
    random_state=12345,
)
#### training data is used to train an evaluation policy
train_bandit_data = dataset.obtain_batch_bandit_feedback(n_rounds=5000)
#### test bandit data is used to approximate the ground-truth policy value
test_bandit_data = dataset.obtain_batch_bandit_feedback(n_rounds=100000)
```

```python id="jG_tgi1qN4Gi"
### evaluation policy training
ipw_learner = IPWLearner(
    n_actions=dataset.n_actions,
    base_classifier=LogisticRegression(C=100, max_iter=10000, random_state=12345)
)
ipw_learner.fit(
    context=train_bandit_data["context"], 
    action=train_bandit_data["action"], 
    reward=train_bandit_data["reward"], 
    pscore=train_bandit_data["pscore"], 
)
action_dist_ipw_test = ipw_learner.predict(
    context=test_bandit_data["context"],
)
policy_value_of_ipw = dataset.calc_ground_truth_policy_value(
    expected_reward=test_bandit_data["expected_reward"], 
    action_dist=action_dist_ipw_test, 
)
```

```python id="B2SBnbEZN4Gk" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1633535954534, "user_tz": -330, "elapsed": 220837, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="dc62a94a-b5be-4b1d-a04e-1675e8ea2114"
### evaluation of OPE estimators
se_df_list = []
for num_data in num_data_list:
    se_list = []
    for _ in tqdm(range(num_runs), desc=f"num_data={num_data}..."):
        ### generate validation data
        validation_bandit_data = dataset.obtain_batch_bandit_feedback(
            n_rounds=num_data
        )

        ### make decisions on vlidation data
        action_dist_ipw_val = ipw_learner.predict(
            context=validation_bandit_data["context"],
        )

        ### OPE using validation data
        regression_model = RegressionModel(
            n_actions=dataset.n_actions, 
            base_model=LogisticRegression(C=100, max_iter=10000, random_state=12345),
        )
        estimated_rewards = regression_model.fit_predict(
            context=validation_bandit_data["context"], # context; x
            action=validation_bandit_data["action"], # action; a
            reward=validation_bandit_data["reward"], # reward; r
            n_folds=2, # 2-fold cross fitting
            random_state=12345,
        )
        ope = OffPolicyEvaluation(
            bandit_feedback=validation_bandit_data,
            ope_estimators=[
                IPS(estimator_name="IPS"), 
                DM(estimator_name="DM"), 
                IPS(lambda_=100, estimator_name="CIPS"), 
                SNIPS(estimator_name="SNIPS"),
                DR(estimator_name="DR"), 
                DRos(lambda_=500, estimator_name="DRos"), 
            ]
        )
        squared_errors = ope.evaluate_performance_of_estimators(
            ground_truth_policy_value=policy_value_of_ipw, # V(\pi_e)
            action_dist=action_dist_ipw_val, # \pi_e(a|x)
            estimated_rewards_by_reg_model=estimated_rewards, # \hat{q}(x,a)
            metric="se", # squared error
        )
        se_list.append(squared_errors)
    ### maximum importance weight in the validation data
    #### a larger value indicates that the logging and evaluation policies are greatly different
    max_iw = (action_dist_ipw_val[
        np.arange(validation_bandit_data["n_rounds"]), 
        validation_bandit_data["action"], 
        0
    ] / validation_bandit_data["pscore"]).max()
    tqdm.write(f"maximum importance weight={np.round(max_iw, 5)}\n")

    ### summarize results
    se_df = DataFrame(DataFrame(se_list).stack())\
        .reset_index(1).rename(columns={"level_1": "est", 0: "se"})
    se_df["num_data"] = num_data
    se_df_list.append(se_df)
    tqdm.write("=====" * 15)
    time.sleep(0.5)

# aggregate all results 
result_df = pd.concat(se_df_list).reset_index(level=0)
```

<!-- #region id="ppBjbg1IN4Gp" -->
### Visualize Results
<!-- #endregion -->

```python id="OynQO-ZBN4Gq"
# figure configs
query = "(est == 'DM' or est == 'IPS' or est == 'SNIPS' or est == 'CIPS' or est == 'DR')"
query += "and num_data <= 6400"
xlabels = [100, 400, 1600, 6400]

plt.style.use('ggplot')
fig, ax = plt.subplots(figsize=(10, 7), tight_layout=True)
sns.lineplot(
    linewidth=5,
    dashes=False,
    legend=False,
    x="num_data",
    y="se",
    hue="est",
    ax=ax,
    data=result_df.query(query),
)
# title and legend
ax.legend(["IPS", "DM", "CIPS", "SNIPS", "DR"], loc="upper right", fontsize=22)
# yaxis
ax.set_yscale("log")
ax.set_ylim(1e-4, 1)
ax.set_ylabel("mean squared error (MSE)", fontsize=25)
ax.tick_params(axis="y", labelsize=15)
ax.yaxis.set_label_coords(-0.08, 0.5)
# xaxis
ax.set_xscale("log")
ax.set_xlabel("number of samples in the log data", fontsize=25)
ax.set_xticks(xlabels)
ax.set_xticklabels(xlabels, fontsize=15)
ax.xaxis.set_label_coords(0.5, -0.1)
```

```python id="Qp4Jft3nN4Gs" outputId="99e84cb1-0f49-43c7-9f65-4fafc700360a"
query = "num_data <= 6400"
xlabels = [100, 400, 1600, 6400]

plt.style.use('ggplot')
fig, ax = plt.subplots(figsize=(10, 7), tight_layout=True)
sns.lineplot(
    linewidth=5,
    dashes=False,
    legend=False,
    x="num_data",
    y="se",
    hue="est",
    ax=ax,
    data=result_df.query(query),
)
# title and legend
ax.legend(["IPS", "DM", "CIPS", "SNIPS", "DR", "DRos"], loc="upper right", fontsize=20)
# yaxis
ax.set_yscale("log")
ax.set_ylim(1e-4, 1)
ax.set_ylabel("mean squared error (MSE)", fontsize=25)
ax.tick_params(axis="y", labelsize=15)
ax.yaxis.set_label_coords(-0.08, 0.5)
# xaxis
ax.set_xscale("log")
ax.set_xlabel("number of samples in the log data", fontsize=25)
ax.set_xticks(xlabels)
ax.set_xticklabels(xlabels, fontsize=15)
ax.xaxis.set_label_coords(0.5, -0.1)
```

```python id="amsFq5EoN4Gt" outputId="b7b43e7c-1425-4209-b767-5d2d3f18544d"
xlabels = [100, 6400, 25600, 51200]

plt.style.use('ggplot')
fig, ax = plt.subplots(figsize=(10, 7), tight_layout=True)
sns.lineplot(
    linewidth=5,
    dashes=False,
    legend=False,
    x="num_data",
    y="se",
    hue="est",
    ax=ax,
    data=result_df,
)
# title and legend
ax.legend(["IPS", "DM", "CIPS", "SNIPS", "DR", "DRos"], loc="upper right", fontsize=20)
# yaxis
ax.set_yscale("log")
ax.set_ylabel("mean squared error (MSE)", fontsize=25)
ax.tick_params(axis="y", labelsize=15)
ax.yaxis.set_label_coords(-0.08, 0.5)
# xaxis
ax.set_xlabel("number of samples in the log data", fontsize=25)
ax.set_xticks(xlabels)
ax.set_xticklabels(xlabels, fontsize=15)
ax.xaxis.set_label_coords(0.5, -0.1)
```

<!-- #region id="0NtaEmu4N4MV" -->
### Hyperparameter Tuning of OPE estimators
<!-- #endregion -->

```python id="Prq1tmxoN4Mr"
### configurations
num_runs = 100
num_data_list = [100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200]
```

```python id="0rGRc44xN4Mt"
### define a dataset class
dataset = SyntheticBanditDataset(
    n_actions=10, 
    dim_context=10,
    tau=0.2, 
    reward_function=logistic_reward_function,
    behavior_policy_function=linear_behavior_policy,
    random_state=12345,
)
#### training data is used to train an evaluation policy
train_bandit_data = dataset.obtain_batch_bandit_feedback(n_rounds=5000)
#### test bandit data is used to approximate the ground-truth policy value
test_bandit_data = dataset.obtain_batch_bandit_feedback(n_rounds=100000)
```

```python id="A1nHObdBN4Mz"
### evaluation policy training
ipw_learner = IPWLearner(
    n_actions=dataset.n_actions,
    base_classifier=LogisticRegression(C=100, max_iter=10000, random_state=12345)
)
ipw_learner.fit(
    context=train_bandit_data["context"], 
    action=train_bandit_data["action"], 
    reward=train_bandit_data["reward"], 
    pscore=train_bandit_data["pscore"], 
)
action_dist_ipw_test = ipw_learner.predict(
    context=test_bandit_data["context"],
)
policy_value_of_ipw = dataset.calc_ground_truth_policy_value(
    expected_reward=test_bandit_data["expected_reward"], 
    action_dist=action_dist_ipw_test, 
)
```

```python id="Ya1-bwhaN4M2" outputId="32790975-5646-46b5-a70c-12a11c5ffa4b"
### evaluation of OPE estimators
se_df_list = []
for num_data in num_data_list:
    se_list = []
    for _ in tqdm(range(num_runs), desc=f"num_data={num_data}..."):
        ### generate validation data
        validation_bandit_data = dataset.obtain_batch_bandit_feedback(
            n_rounds=num_data
        )

        ### make decisions on vlidation data
        action_dist_ipw_val = ipw_learner.predict(
            context=validation_bandit_data["context"],
        )

        ### OPE using validation data
        regression_model = RegressionModel(
            n_actions=dataset.n_actions, 
            base_model=LogisticRegression(C=100, max_iter=10000, random_state=12345)
        )
        estimated_rewards = regression_model.fit_predict(
            context=validation_bandit_data["context"], # context; x
            action=validation_bandit_data["action"], # action; a
            reward=validation_bandit_data["reward"], # reward; r
            n_folds=2, # 2-fold cross fitting
            random_state=12345,
        )
        ope = OffPolicyEvaluation(
            bandit_feedback=validation_bandit_data,
            ope_estimators=[
                DRos(lambda_=1, estimator_name="DRos (1)"), 
                DRos(lambda_=100, estimator_name="DRos (100)"), 
                DRos(lambda_=10000, estimator_name="DRos (10000)"),
                DRosTuning(
                    use_bias_upper_bound=False,
                    lambdas=np.arange(1, 10002, 100).tolist(), 
                    estimator_name="DRos (tuning)"
                ), 
            ]
        )
        squared_errors = ope.evaluate_performance_of_estimators(
            ground_truth_policy_value=policy_value_of_ipw, # V(\pi_e)
            action_dist=action_dist_ipw_val, # \pi_e(a|x)
            estimated_rewards_by_reg_model=estimated_rewards, # \hat{q}(x,a)
            metric="se", # squared error
        )
        se_list.append(squared_errors)
    ### maximum importance weight in the validation data
    #### a larger value indicates that the logging and evaluation policies are greatly different
    max_iw = (action_dist_ipw_val[
        np.arange(validation_bandit_data["n_rounds"]), 
        validation_bandit_data["action"], 
        0
    ] / validation_bandit_data["pscore"]).max()
    tqdm.write(f"maximum importance weight={np.round(max_iw, 5)}\n")

    ### summarize results
    se_df = DataFrame(DataFrame(se_list).stack())\
        .reset_index(1).rename(columns={"level_1": "est", 0: "se"})
    se_df["num_data"] = num_data
    se_df_list.append(se_df)
    tqdm.write("=====" * 15)
    time.sleep(0.5)

# aggregate all results 
result_df = pd.concat(se_df_list).reset_index(level=0)
```

<!-- #region id="dOuGBvgkN4NA" -->
### Visualize Results
<!-- #endregion -->

```python id="mcSL-5CkN4NC" outputId="58be630b-eb3f-4fea-fd20-14481ddb08c9"
query = "est == 'DRos (1)' or est == 'DRos (100)' or est == 'DRos (10000)'"
xlabels = [100, 6400, 25600, 51200]

plt.style.use('ggplot')
fig, ax = plt.subplots(figsize=(10, 7), tight_layout=True)
sns.lineplot(
    linewidth=5,
    dashes=False,
    legend=False,
    x="num_data",
    y="se",
    hue="est",
    ax=ax,
    data=result_df.query(query),
)
# title and legend
ax.legend(
    ["DRos (1)", "DRos (100)", "DRos (10000)"], 
    loc="upper right", fontsize=25,
)
# yaxis
ax.set_yscale("log")
ax.set_ylim(3e-4, 0.05)
ax.set_ylabel("mean squared error (MSE)", fontsize=25)
ax.tick_params(axis="y", labelsize=15)
ax.yaxis.set_label_coords(-0.1, 0.5)
# xaxis
# ax.set_xscale("log")
ax.set_xlabel("number of samples in the log data", fontsize=25)
ax.set_xticks(xlabels)
ax.set_xticklabels(xlabels, fontsize=15)
ax.xaxis.set_label_coords(0.5, -0.1)
```

```python id="OdaRivnJN4NE" outputId="8ec6265d-c37c-47d9-e76e-99593b79bd68"
xlabels = [100, 6400, 25600, 51200]

plt.style.use('ggplot')
fig, ax = plt.subplots(figsize=(10, 7), tight_layout=True)
sns.lineplot(
    linewidth=5,
    dashes=False,
    legend=False,
    x="num_data",
    y="se",
    hue="est",
    ax=ax,
    data=result_df,
)
# title and legend
ax.legend(
    ["DRos (1)", "DRos (100)", "DRos (10000)", "DRos (tuning)"], 
    loc="upper right", fontsize=22,
)
# yaxis
ax.set_yscale("log")
ax.set_ylim(3e-4, 0.05)
ax.set_ylabel("mean squared error (MSE)", fontsize=25)
ax.tick_params(axis="y", labelsize=15)
ax.yaxis.set_label_coords(-0.1, 0.5)
# xaxis
# ax.set_xscale("log")
ax.set_xlabel("number of samples in the log data", fontsize=25)
ax.set_xticks(xlabels)
ax.set_xticklabels(xlabels, fontsize=15)
ax.xaxis.set_label_coords(0.5, -0.1)
```

<!-- #region id="g0WakpbD3zud" -->
## Multi-class Classificatoin Data
<!-- #endregion -->

<!-- #region id="pVYkSySo4Ekw" -->
This section provides an example of conducting OPE of an evaluation policy using classification data as logged bandit data.
It is quite common to conduct OPE experiments using classification data. Appendix G of [Farajtabar et al.(2018)](https://arxiv.org/abs/1802.03493) describes how to conduct OPE experiments with classification data in detail.
<!-- #endregion -->

```python id="lF6UlZCf4K5l" executionInfo={"status": "ok", "timestamp": 1633595545532, "user_tz": -330, "elapsed": 424, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import obp
from obp.dataset import MultiClassToBanditReduction
from obp.ope import (
    OffPolicyEvaluation, 
    RegressionModel,
    InverseProbabilityWeighting as IPS,
    DirectMethod as DM,
    DoublyRobust as DR, 
)
```

<!-- #region tags=[] id="B82mCB3D4Ek-" -->
### Bandit Reduction
`obp.dataset.MultiClassToBanditReduction` is an easy-to-use for transforming classification data to bandit data.
It takes 
- feature vectors (`X`)
- class labels (`y`)
- classifier to construct behavior policy (`base_classifier_b`) 
- paramter of behavior policy (`alpha_b`) 

as its inputs and generates a bandit data that can be used to evaluate the performance of decision making policies (obtained by `off-policy learning`) and OPE estimators.
<!-- #endregion -->

```python id="u0JW7ncq4Ek_" executionInfo={"status": "ok", "timestamp": 1633595547274, "user_tz": -330, "elapsed": 6, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
# load raw digits data
# `return_X_y` splits feature vectors and labels, instead of returning a Bunch object
X, y = load_digits(return_X_y=True)
```

```python tags=[] id="9qcJx4qe4ElA" executionInfo={"status": "ok", "timestamp": 1633595549136, "user_tz": -330, "elapsed": 8, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
# convert the raw classification data into a logged bandit dataset
# we construct a behavior policy using Logistic Regression and parameter alpha_b
# given a pair of a feature vector and a label (x, c), create a pair of a context vector and reward (x, r)
# where r = 1 if the output of the behavior policy is equal to c and r = 0 otherwise
# please refer to https://zr-obp.readthedocs.io/en/latest/_autosummary/obp.dataset.multiclass.html for the details
dataset = MultiClassToBanditReduction(
    X=X,
    y=y,
    base_classifier_b=LogisticRegression(max_iter=10000, random_state=12345),
    alpha_b=0.8,
    dataset_name="digits",
)
```

```python id="r4Q489zK4ElC" executionInfo={"status": "ok", "timestamp": 1633595551923, "user_tz": -330, "elapsed": 408, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
# split the original data into training and evaluation sets
dataset.split_train_eval(eval_size=0.7, random_state=12345)
```

```python id="Xp-wV8o34ElE" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1633595555132, "user_tz": -330, "elapsed": 2496, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="030e302f-144e-4979-9be3-35469dfbd2f8"
# obtain logged bandit data generated by behavior policy
bandit_data = dataset.obtain_batch_bandit_feedback(random_state=12345)

# `bandit_data` is a dictionary storing logged bandit feedback
bandit_data
```

<!-- #region id="KwOsOHSy4ElG" -->
### Off-Policy Learning
After generating logged bandit data, we now obtain an evaluation policy using the training set.
<!-- #endregion -->

```python id="tP-QRRYO4ElH" executionInfo={"status": "ok", "timestamp": 1633595562717, "user_tz": -330, "elapsed": 654, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
# obtain action choice probabilities by an evaluation policy
# we construct an evaluation policy using Random Forest and parameter alpha_e
action_dist = dataset.obtain_action_dist_by_eval_policy(
    base_classifier_e=RandomForestClassifier(random_state=12345),
    alpha_e=0.9,
)
```

```python id="2E_PdWhs4ElJ" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1633595563429, "user_tz": -330, "elapsed": 12, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="792d66f3-eccd-4877-e48e-43f09bfc75ee"
# which action to take for each context (a probability distribution over actions)
action_dist[:, :, 0]
```

<!-- #region id="iSDndg5F4ElL" -->
### Obtain a Reward Estimator
`obp.ope.RegressionModel` simplifies the process of reward modeling

$r(x,a) = \mathbb{E} [r \mid x, a] \approx \hat{r}(x,a)$
<!-- #endregion -->

```python id="4YDcPw5X4ElL" executionInfo={"status": "ok", "timestamp": 1633595608029, "user_tz": -330, "elapsed": 3437, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
regression_model = RegressionModel(
    n_actions=dataset.n_actions, # number of actions; |A|
    base_model=LogisticRegression(C=100, max_iter=10000, random_state=12345), # any sklearn classifier
)
```

```python id="_RQygmVb4ElM" executionInfo={"status": "ok", "timestamp": 1633595608917, "user_tz": -330, "elapsed": 22, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
estimated_rewards = regression_model.fit_predict(
    context=bandit_data["context"],
    action=bandit_data["action"],
    reward=bandit_data["reward"],
    position=bandit_data["position"],
    random_state=12345,
)
```

```python id="kVKN-yFJ4ElM" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1633595608919, "user_tz": -330, "elapsed": 22, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="1cf829df-6423-4ff0-fbe8-5b6265cdd536"
estimated_rewards[:, :, 0] # \hat{q}(x,a)
```

<!-- #region tags=[] id="iSHMA1Zq4ElK" -->
### Off-Policy Evaluation (OPE)
OPE attempts to estimate the performance of evaluation policies using their action choice probabilities.

Here, we evaluate/compare the OPE performance (estimation accuracy) of 
- **Inverse Propensity Score (IPS)**
- **DirectMethod (DM)**
- **Doubly Robust (DR)**
<!-- #endregion -->

<!-- #region id="XJri6i8T4ElN" -->
`obp.ope.OffPolicyEvaluation` simplifies the OPE process

$V(\pi_e) \approx \hat{V} (\pi_e; \mathcal{D}_0, \theta)$ using DM, IPS, and DR
<!-- #endregion -->

```python tags=[] id="ao74MKZh4ElN" executionInfo={"status": "ok", "timestamp": 1633595612620, "user_tz": -330, "elapsed": 414, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
ope = OffPolicyEvaluation(
    bandit_feedback=bandit_data, # bandit data
    ope_estimators=[
        IPS(estimator_name="IPS"), 
        DM(estimator_name="DM"), 
        DR(estimator_name="DR"),
    ] # used estimators
)
```

```python tags=[] id="j_9e42PN4ElN" executionInfo={"status": "ok", "timestamp": 1633595615759, "user_tz": -330, "elapsed": 622, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
estimated_policy_value = ope.estimate_policy_values(
    action_dist=action_dist, # \pi_e(a|x)
    estimated_rewards_by_reg_model=estimated_rewards, # \hat{q}
)
```

```python id="og587yT84ElO" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1633595616213, "user_tz": -330, "elapsed": 18, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="28d4c8fe-f715-4c89-bb3e-a05051b58425"
# OPE results given by the three estimators
estimated_policy_value
```

```python tags=[] id="GSw4hMKq4BVO" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1633595773361, "user_tz": -330, "elapsed": 405, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="14eb20fc-b8ac-4e6a-f760-1577512c9d98"
# estimate the policy value of IPWLearner with Logistic Regression
estimated_policy_value, estimated_interval = ope.summarize_off_policy_estimates(
    action_dist=action_dist,
    estimated_rewards_by_reg_model=estimated_rewards
)
print(estimated_interval, '\n')
```

```python id="ElRnFn0z4BVO" colab={"base_uri": "https://localhost:8080/", "height": 420} executionInfo={"status": "ok", "timestamp": 1633595783284, "user_tz": -330, "elapsed": 2464, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="bb7053e3-4108-42e9-d09a-c9dadbe8d9fb"
# visualize policy values of the evaluation policy estimated by the three OPE estimators
ope.visualize_off_policy_estimates(
    action_dist=action_dist,
    estimated_rewards_by_reg_model=estimated_rewards,
    n_bootstrap_samples=10000, # number of resampling performed in the bootstrap procedure
    random_state=12345,
)
```

<!-- #region tags=[] id="nlu-3nP44ElO" -->
### Evaluation of OPE estimators
Our final step is **the evaluation of OPE**, which evaluates and compares the estimation accuracy of OPE estimators.

With the multi-class classification data, we can calculate the ground-truth policy value of the evaluation policy. 
Therefore, we can compare the policy values estimated by OPE estimators with the ground-turth to evaluate OPE estimators.
<!-- #endregion -->

<!-- #region id="eic3oLtg4ElP" -->
**Approximate the Ground-truth Policy Value**

$V(\pi) \approx \frac{1}{|\mathcal{D}_{te}|} \sum_{i=1}^{|\mathcal{D}_{te}|} \mathbb{E}_{a \sim \pi(a|x_i)} [r(x_i, a)], \; \, where \; \, r(x,a) := \mathbb{E}_{r \sim p(r|x,a)} [r]$
<!-- #endregion -->

```python id="HmToQUAI4ElP" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1633595639110, "user_tz": -330, "elapsed": 714, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="75ed5c69-18d2-4e49-a6cb-e68318f26d07"
# calculate the ground-truth performance of the evaluation policy
true_policy_value = dataset.calc_ground_truth_policy_value(action_dist=action_dist)

true_policy_value
```

<!-- #region id="ZWMpEjoj4ElQ" -->
**Evaluation of OPE**

Now, let's evaluate the OPE performance (estimation accuracy) of the three estimators 

$SE (\hat{V}; \mathcal{D}_0) := \left( V(\pi_e) - \hat{V} (\pi_e; \mathcal{D}_0, \theta) \right)^2$,     (squared error of $\hat{V}$)
<!-- #endregion -->

```python id="IxIAxIPr4ElQ" executionInfo={"status": "ok", "timestamp": 1633595649753, "user_tz": -330, "elapsed": 484, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
squared_errors = ope.evaluate_performance_of_estimators(
    ground_truth_policy_value=true_policy_value,
    action_dist=action_dist,
    estimated_rewards_by_reg_model=estimated_rewards,
    metric="se", # squared error
)
```

```python id="f4wwafQJ4ElR" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1633595650483, "user_tz": -330, "elapsed": 10, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="803f6c7f-4af5-4668-9598-a85304882c7a"
squared_errors # DR is the most accurate 
```

<!-- #region id="RAlNdEJO4ElS" -->
We can iterate the above process several times and calculate the following MSE

$MSE (\hat{V}) := T^{-1} \sum_{t=1}^T SE (\hat{V}; \mathcal{D}_0^{(t)}) $

where $\mathcal{D}_0^{(t)}$ is the synthetic data in the $t$-th iteration
<!-- #endregion -->

<!-- #region id="DbPhRfv_42vM" -->
## OBP Library Workshop Tutorial on ZOZOTOWN Open-Bandit Dataset
<!-- #endregion -->

<!-- #region id="1HVz8Zvk6Vmm" -->
This section demonstrates an example of conducting OPE of Bernoulli Thompson Sampling (BernoulliTS) as an evaluation policy. We use some OPE estimators and logged bandit feedback generated by running the Random policy (behavior policy) on the ZOZOTOWN platform. We also evaluate and compare the OPE performance (accuracy) of several estimators.

The example consists of the follwoing four major steps:
- (1) Data Loading and Preprocessing
- (2) Replicating Production Policy
- (3) Off-Policy Evaluation (OPE)
- (4) Evaluation of OPE
<!-- #endregion -->

```python id="GL3Rgwjd61lT" executionInfo={"status": "ok", "timestamp": 1633596178033, "user_tz": -330, "elapsed": 406, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
from sklearn.linear_model import LogisticRegression

import obp
from obp.dataset import OpenBanditDataset
from obp.policy import BernoulliTS
from obp.ope import (
    OffPolicyEvaluation, 
    RegressionModel,
    DirectMethod,
    InverseProbabilityWeighting,
    DoublyRobust
)
```

<!-- #region id="Vb5es4ED6Vm0" -->
### Data Loading and Preprocessing

`obp.dataset.OpenBanditDataset` is an easy-to-use data loader for Open Bandit Dataset. 

It takes behavior policy ('bts' or 'random') and campaign ('all', 'men', or 'women') as inputs and provides dataset preprocessing.
<!-- #endregion -->

```python tags=[] id="r8Y5fmAe6Vm0"
# load and preprocess raw data in "All" campaign collected by the Random policy (behavior policy here)
# When `data_path` is not given, this class downloads the small-sized version of the Open Bandit Dataset.
dataset = OpenBanditDataset(behavior_policy='random', campaign='all')

# obtain logged bandit feedback generated by behavior policy
bandit_feedback = dataset.obtain_batch_bandit_feedback()
```

<!-- #region id="gCcVNKgU6Vm1" -->
the logged bandit feedback is collected by the behavior policy as follows.

$ \mathcal{D}_b := \{(x_i,a_i,r_i)\}$  where $(x,a,r) \sim p(x)\pi_b(a \mid x)p(r \mid x,a) $
<!-- #endregion -->

```python id="QnkylN1j6Vm2" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1633596246968, "user_tz": -330, "elapsed": 17, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="c2d601e1-2958-4490-fc39-bde7eaea3043"
# `bandit_feedback` is a dictionary storing logged bandit feedback
bandit_feedback.keys()
```

<!-- #region id="80E1jcy46Vm3" -->
### let's see some properties of the dataset class
<!-- #endregion -->

```python id="MyGk1RzM6Vm4" colab={"base_uri": "https://localhost:8080/", "height": 35} executionInfo={"status": "ok", "timestamp": 1633596251736, "user_tz": -330, "elapsed": 796, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="382b1e2c-303e-44ee-8e34-080fcca47c96"
# name of the dataset is 'obd' (open bandit dataset)
dataset.dataset_name
```

```python id="qxwfstvo6Vm5" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1633596253226, "user_tz": -330, "elapsed": 19, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="fff28382-229f-4792-ec77-445da30cabba"
# number of actions of the "All" campaign is 80
dataset.n_actions
```

```python id="Pdx2T2mM6Vm5" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1633596253229, "user_tz": -330, "elapsed": 14, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="0697e567-ac39-4944-e72b-185e19867ad4"
# small sample example data has 10,000 samples (or rounds)
dataset.n_rounds
```

```python id="a_KKfmDK6Vm6" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1633596256334, "user_tz": -330, "elapsed": 731, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="ec3902f5-9c82-4fd8-d4cb-1f45f48f9a21"
# default context (feature) engineering creates context vector with 20 dimensions
dataset.dim_context
```

```python id="Id-YjQBu6Vm8" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1633596256335, "user_tz": -330, "elapsed": 11, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="55bc294d-8880-4efb-8f8b-4ca4d8d0dba8"
# ZOZOTOWN recommendation interface has three positions
# (please see https://github.com/st-tech/zr-obp/blob/master/images/recommended_fashion_items.png)
dataset.len_list
```

<!-- #region id="z16WFQhM6Vm9" -->
### Replicating Production Policy

After preparing the dataset, we now replicate the BernoulliTS policy implemented on the ZOZOTOWN recommendation interface during the data collection period.

Here, we use `obp.policy.BernoulliTS` as an evaluation policy. 
By activating its `is_zozotown_prior` argument, we can replicate (the policy parameters of) BernoulliTS used in the ZOZOTOWN production.

(When `is_zozotown_prior=False`, non-informative prior distribution is used.)
<!-- #endregion -->

```python tags=[] id="sumivwdz6VnA" executionInfo={"status": "ok", "timestamp": 1633596274350, "user_tz": -330, "elapsed": 8411, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
# define BernoulliTS as an evaluation policy
evaluation_policy = BernoulliTS(
    n_actions=dataset.n_actions, 
    len_list=dataset.len_list, 
    is_zozotown_prior=True, # replicate the BernoulliTS policy in the ZOZOTOWN production
    campaign="all",
    random_state=12345,
)

# compute the action choice probabilities of the evaluation policy using Monte Carlo simulation
action_dist = evaluation_policy.compute_batch_action_dist(
    n_sim=100000, n_rounds=bandit_feedback["n_rounds"],
)
```

```python id="l05xdN2L6VnB" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1633596274354, "user_tz": -330, "elapsed": 51, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="d3c66953-24dc-406b-c085-b5a83c1d1cb1"
# action_dist is an array of shape (n_rounds, n_actions, len_list) 
# representing the distribution over actions by the evaluation policy
action_dist
```

<!-- #region id="bGoDVDue6VnE" -->
### Obtaining a Reward Estimator
A reward estimator $\hat{q}(x,a)$ is needed for model dependent estimators such as DM or DR.

$\hat{q}(x,a) \approx \mathbb{E} [r \mid x,a]$
<!-- #endregion -->

```python id="PuJpSxLF6VnE" executionInfo={"status": "ok", "timestamp": 1633596319228, "user_tz": -330, "elapsed": 947, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
# estimate the expected reward by using an ML model (Logistic Regression here)
# the estimated rewards are used by model-dependent estimators such as DM and DR
regression_model = RegressionModel(
    n_actions=dataset.n_actions,
    len_list=dataset.len_list,
    action_context=dataset.action_context,
    base_model=LogisticRegression(max_iter=1000, random_state=12345),
)
```

```python id="PcKW-mKD6VnF" executionInfo={"status": "ok", "timestamp": 1633596322063, "user_tz": -330, "elapsed": 999, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
estimated_rewards_by_reg_model = regression_model.fit_predict(
    context=bandit_feedback["context"],
    action=bandit_feedback["action"],
    reward=bandit_feedback["reward"],
    position=bandit_feedback["position"],
    pscore=bandit_feedback["pscore"],
    n_folds=3, # use 3-fold cross-fitting
    random_state=12345,
)
```

<!-- #region id="Ou9KAYVK6VnF" -->
please refer to https://arxiv.org/abs/2002.08536 about the details of the cross-fitting procedure.
<!-- #endregion -->

<!-- #region id="MABMbS-z6VnD" -->
### Off-Policy Evaluation (OPE)
Our next step is OPE, which attempts to estimate the performance of evaluation policies using the logged bandit feedback and OPE estimators.

Here, we use 
- `obp.ope.InverseProbabilityWeighting` (IPW)
- `obp.ope.DirectMethod` (DM)
- `obp.ope.DoublyRobust` (DR)

as estimators and visualize the OPE results.
<!-- #endregion -->

<!-- #region id="r3JQSihf6VnG" -->
$V(\pi_e) \approx \hat{V} (\pi_e; \mathcal{D}_b, \theta)$ using DM, IPW, and DR
<!-- #endregion -->

```python tags=[] id="W4DGTTzL6VnG" executionInfo={"status": "ok", "timestamp": 1633596333413, "user_tz": -330, "elapsed": 7318, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
# estimate the policy value of BernoulliTS based on its action choice probabilities
# it is possible to set multiple OPE estimators to the `ope_estimators` argument
ope = OffPolicyEvaluation(
    bandit_feedback=bandit_feedback,
    ope_estimators=[InverseProbabilityWeighting(), DirectMethod(), DoublyRobust()]
)

# `summarize_off_policy_estimates` returns pandas dataframes including the OPE results
estimated_policy_value, estimated_interval = ope.summarize_off_policy_estimates(
    action_dist=action_dist, 
    estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
    n_bootstrap_samples=10000, # number of resampling performed in the bootstrap procedure.
    random_state=12345,
)
```

```python id="U7d-t-op6VnH" colab={"base_uri": "https://localhost:8080/", "height": 142} executionInfo={"status": "ok", "timestamp": 1633596333440, "user_tz": -330, "elapsed": 43, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="779f07f4-291f-48e7-c63c-ff2b232d56db"
# the estimated policy value of the evaluation policy (the BernoulliTS policy)
# relative_estimated_policy_value is the policy value of the evaluation policy 
# relative to the ground-truth policy value of the behavior policy (the Random policy here)
estimated_policy_value
```

```python id="DDog6nOz6VnH" colab={"base_uri": "https://localhost:8080/", "height": 142} executionInfo={"status": "ok", "timestamp": 1633596333448, "user_tz": -330, "elapsed": 42, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="b18a63fd-c2c4-45d2-8f07-eeb2c5b5f903"
# confidence intervals of policy value of BernoulliTS estimated by OPE estimators
# (`mean` values in this dataframe is also estimated via the non-parametric bootstrap procedure 
# and is a bit different from the above values in `estimated_policy_value`)
estimated_interval
```

```python id="qAemnF5s6VnI" colab={"base_uri": "https://localhost:8080/", "height": 420} executionInfo={"status": "ok", "timestamp": 1633596343326, "user_tz": -330, "elapsed": 5175, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="3e7af000-faff-41e8-f35f-1008f2625233"
# visualize the policy values of BernoulliTS estimated by the three OPE estimators
# and their 95% confidence intervals (estimated by nonparametric bootstrap method)
ope.visualize_off_policy_estimates(
    action_dist=action_dist,
    estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
    n_bootstrap_samples=10000, # number of resampling performed in the bootstrap procedure
    random_state=12345,
)
```

```python id="q_boKCey6VnI" colab={"base_uri": "https://localhost:8080/", "height": 420} executionInfo={"status": "ok", "timestamp": 1633596347268, "user_tz": -330, "elapsed": 3951, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="429a8744-deb7-48c8-8450-f270b0753b38"
# by activating the `is_relative` option
# we can visualize the estimated policy value of the evaluation policy
# relative to the ground-truth policy value of the behavior policy
ope.visualize_off_policy_estimates(
    action_dist=action_dist,
    estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
    n_bootstrap_samples=10000, # number of resampling performed in the bootstrap procedure
    is_relative=True,
    random_state=12345,
)
```

<!-- #region id="3qZOJfFA6VnJ" -->
Note that the OPE demonstration here is with the small size example version of our dataset. 

Please use its full size version (https://research.zozo.com/data.html) to produce more reasonable results.
<!-- #endregion -->

<!-- #region id="vi8gUQii6VnJ" -->
### Evaluation of OPE

Our final step is the **evaluation of OPE**, which evaluates the estimation accuracy of OPE estimators.

Specifically, we asses the accuracy of the estimator such as DM, IPW, and DR by comparing its estimation with the ground-truth policy value estimated via the on-policy estimation from the Open Bandit Dataset.

This type of evaluation of OPE is possible, because Open Bandit Dataset contains a set of *multiple* different logged bandit feedback datasets collected by running different policies on the same platform at the same time.

Please refer to [the documentation](https://zr-obp.readthedocs.io/en/latest/evaluation_ope.html) for the details about the evaluation of OPE protocol.
<!-- #endregion -->

<!-- #region id="9UOhpsA56VnK" -->
**Approximate the Ground-truth Policy Value**
With Open Bandit Dataset, we can estimate the ground-truth policy value of the evaluation policy in an on-policy manner as follows.

$V(\pi_e) \approx \frac{1}{|\mathcal{D}_{e}|} \sum_{i=1}^{|\mathcal{D}_{e}|} \mathbb{E}_{n} [r_i]$

$ \mathcal{D}_e := \{(x_i,a_i,r_i)\} $ ($(x,a,r) \sim p(x)\pi_e(a \mid x)p(r \mid x,a) $) is the log data collected by the evaluation policy (, which is used only for approximating the ground-truth policy value).

We can compare the policy values estimated by OPE estimators with this on-policy estimate to evaluate the accuracy of OPE.
<!-- #endregion -->

```python id="IZk_JW4L6VnK"
# we first calculate the ground-truth policy value of the evaluation policy
# , which is estimated by averaging the factual (observed) rewards contained in the dataset (on-policy estimation)
policy_value_bts = OpenBanditDataset.calc_on_policy_policy_value_estimate(
    behavior_policy='bts', campaign='all'
)
```

<!-- #region id="2zGoaMKD6VnL" -->
**Evaluation of OPE**

We can evaluate the estimation performance of OPE estimators by comparing the estimated policy values of the evaluation with its ground-truth as follows.

- $\textit{relative-ee} (\hat{V}; \mathcal{D}_b) := \left| \frac{V(\pi_e) - \hat{V} (\pi_e; \mathcal{D}_b)}{V(\pi_e)} \right|$ (relative estimation error; relative-ee)
- $\textit{SE} (\hat{V}; \mathcal{D}_b) := \left( V(\pi_e) - \hat{V} (\pi_e; \mathcal{D}_b) \right)^2$ (squared error; se)
<!-- #endregion -->

```python id="QEZwyYMa6VnL" colab={"base_uri": "https://localhost:8080/", "height": 142} executionInfo={"status": "ok", "timestamp": 1633596366167, "user_tz": -330, "elapsed": 488, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="9ae11ec7-3569-49b5-a524-7cea9389a832"
# evaluate the estimation performance of OPE estimators 
# `evaluate_performance_of_estimators` returns a dictionary containing estimation performance of given estimators 
relative_ee = ope.summarize_estimators_comparison(
    ground_truth_policy_value=policy_value_bts,
    action_dist=action_dist,
    estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
    metric="relative-ee", # "relative-ee" (relative estimation error) or "se" (squared error)
)

# estimation performances of the three estimators (lower means accurate)
relative_ee
```

<!-- #region id="dLR9yQ4Z6VnM" -->
We can iterate the above process several times to get more relibale results.

Please see [examples/obd](../obd) for a more sophisticated example of the evaluation of OPE with the Open Bandit Dataset.
<!-- #endregion -->

<!-- #region id="E58KUB0_8kCm" -->
## Real-world Example
<!-- #endregion -->

<!-- #region id="n1lh-txJ8tth" -->
**"What is the best new policy for the ZOZOTOWN recommendation interface?"**
<!-- #endregion -->

<!-- #region id="58Gl4El856zr" -->
1. Data Loading and Preprocessing (Random Bucket of Open Bandit Dataset)
2. Off-Policy Learning (IPWLearner and NNPolicyLearner)
3. Off-Policy Evaluation (IPWLearner vs NNPolicyLearner)
<!-- #endregion -->

```python id="RnS0ERLI8v_E" executionInfo={"status": "ok", "timestamp": 1633596679941, "user_tz": -330, "elapsed": 575, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import obp
from obp.dataset import OpenBanditDataset
from obp.policy import IPWLearner, NNPolicyLearner
from obp.ope import (
    RegressionModel,
    OffPolicyEvaluation,
    SelfNormalizedInverseProbabilityWeighting as SNIPS,
    DoublyRobust as DR,
)
```

<!-- #region id="shwV9APY56z3" -->
### Data Loading and Preprocessing

Here we use a random bucket of the Open Bandit Pipeline. We can download this by using `obp.dataset.OpenBanditDataset`.
<!-- #endregion -->

```python id="pAbFEGCd56z4"
# define OpenBanditDataset class to handle the real bandit data
dataset = OpenBanditDataset(
    behavior_policy="random", campaign="all"
)
```

```python id="NkujJdES56z5" executionInfo={"status": "ok", "timestamp": 1633596752025, "user_tz": -330, "elapsed": 686, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
# logged bandit data collected by the uniform random policy
training_bandit_data, test_bandit_data = dataset.obtain_batch_bandit_feedback(
    test_size=0.3, is_timeseries_split=True,
)
```

```python id="Op3FDL2v56z6" executionInfo={"status": "ok", "timestamp": 1633596752995, "user_tz": -330, "elapsed": 25, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
# ignore the position effect for a demo purpose
training_bandit_data["position"] = None 
test_bandit_data["position"] = None
```

```python id="EwrljvAA56z8" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1633596752997, "user_tz": -330, "elapsed": 25, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="4fd747fa-4734-42ce-e5cc-49ac438dbd68"
# number of actions
dataset.n_actions
```

```python id="s48veLd856z9" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1633596753551, "user_tz": -330, "elapsed": 15, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="ee00573f-cfbd-49b7-dbe2-4421a0f2fb5f"
# sample size
dataset.n_rounds
```

<!-- #region id="c-L4xGR556z-" -->
### Off-Policy Learning (OPL)

Train two new policies: `obp.policy.IPWLearner` and `obp.policy.NNPolicyLearner`.
<!-- #endregion -->

<!-- #region id="gV3apX5_560A" -->
#### IPWLearner
<!-- #endregion -->

```python id="Dw2YCUE2560C" executionInfo={"status": "ok", "timestamp": 1633596766687, "user_tz": -330, "elapsed": 1266, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
ipw_learner = IPWLearner(
    n_actions=dataset.n_actions,
    base_classifier=RandomForestClassifier(
        n_estimators=100, max_depth=5, min_samples_leaf=10, random_state=12345
    ),
)
```

```python id="a5XUmBmy560H" executionInfo={"status": "ok", "timestamp": 1633596768216, "user_tz": -330, "elapsed": 4, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
# fit
ipw_learner.fit(
    context=training_bandit_data["context"], # context; x
    action=training_bandit_data["action"], # action; a
    reward=training_bandit_data["reward"], # reward; r
    pscore=training_bandit_data["pscore"], # propensity score; pi_b(a|x)
)
```

```python id="so3rm4Z7560Q" executionInfo={"status": "ok", "timestamp": 1633596770713, "user_tz": -330, "elapsed": 22, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
# predict (make new decisions)
action_dist_ipw = ipw_learner.predict(
    context=test_bandit_data["context"]
)
```

<!-- #region id="valx_rT5560S" -->
#### NNPolicyLearner
<!-- #endregion -->

```python id="ZFDKi7br560T" executionInfo={"status": "ok", "timestamp": 1633596776980, "user_tz": -330, "elapsed": 690, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
nn_learner = NNPolicyLearner(
    n_actions=dataset.n_actions,
    dim_context=dataset.dim_context,
    solver="adam",
    off_policy_objective="ipw", # = ips
    batch_size=32, 
    random_state=12345,
)
```

```python id="rOlbBIEt560T" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1633596778140, "user_tz": -330, "elapsed": 498, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="eb5a0c52-4c39-42dc-fedb-2bf931a118a7"
# fit
nn_learner.fit(
    context=training_bandit_data["context"], # context; x
    action=training_bandit_data["action"], # action; a
    reward=training_bandit_data["reward"], # reward; r
    pscore=training_bandit_data["pscore"], # propensity score; pi_b(a|x)
)
```

```python id="r4MM6dZi560V" executionInfo={"status": "ok", "timestamp": 1633596784212, "user_tz": -330, "elapsed": 454, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
# predict (make new decisions)
action_dist_nn = nn_learner.predict(
    context=test_bandit_data["context"]
)
```

<!-- #region id="4cnhCxFD560W" -->
### Obtain a Reward Estimator
`obp.ope.RegressionModel` simplifies the process of reward modeling

$r(x,a) = \mathbb{E} [r \mid x, a] \approx \hat{r}(x,a)$
<!-- #endregion -->

```python id="5fLcJQd5560X" executionInfo={"status": "ok", "timestamp": 1633596816837, "user_tz": -330, "elapsed": 721, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
regression_model = RegressionModel(
    n_actions=dataset.n_actions, 
    base_model=LogisticRegression(C=100, max_iter=500, random_state=12345),
)
```

```python id="bEh9IBT0560X" executionInfo={"status": "ok", "timestamp": 1633596817550, "user_tz": -330, "elapsed": 717, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
estimated_rewards = regression_model.fit_predict(
    context=test_bandit_data["context"], 
    action=test_bandit_data["action"], 
    reward=test_bandit_data["reward"], 
    random_state=12345,
)
```

<!-- #region id="ZEMfZ5Oa560V" -->
### Off-Policy Evaluation

Estimating the performance of IPWLearner and NNPolicyLearner via OPE.
<!-- #endregion -->

<!-- #region id="kKQ3Edm-560Y" -->
`obp.ope.OffPolicyEvaluation` simplifies the OPE process

$V(\pi_e) \approx \hat{V} (\pi_e; \mathcal{D}_0, \theta)$ using DM, IPS, and DR
<!-- #endregion -->

```python id="iVqAiUZq560Z" executionInfo={"status": "ok", "timestamp": 1633596820507, "user_tz": -330, "elapsed": 496, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
ope = OffPolicyEvaluation(
    bandit_feedback=test_bandit_data,
    ope_estimators=[
        SNIPS(estimator_name="SNIPS"),
        DR(estimator_name="DR"),
    ]
)
```

<!-- #region id="Ez6lEok2560Z" -->
### Visualize the OPE results

Output the relative performances of the trained policies compared to the logging policy (uniform random)
<!-- #endregion -->

```python id="DvU-i9v39mg0" outputId="bf0cec7c-1ded-4dc8-938b-6e2d2500706a"
ope.visualize_off_policy_estimates_of_multiple_policies(
    policy_name_list=["IPWLearner", "NNPolicyLearner"],
    action_dist_list=[action_dist_ipw, action_dist_nn],
    estimated_rewards_by_reg_model=estimated_rewards,
    n_bootstrap_samples=100,
    is_relative=True,
    random_state=12345,
)
```

<!-- #region id="BNlhWgkG560b" -->
Both policy learner outperforms the random baseline. In particular, NNPolicyLearner seems to be the best, improving the random baseline by about 70\%. It also outperforms IPWLearner in both SNIPS and DR.
<!-- #endregion -->

<!-- #region id="9mpbTSak9WLK" -->
## Synthetic Slate Dataset
<!-- #endregion -->

<!-- #region id="rchVYeEE95fx" -->
This section provides an example of conducting OPE of several different evaluation policies with synthetic slate bandit feedback data.

Our example with synthetic bandit data contains the follwoing four major steps:
- (1) Synthetic Slate Data Generation
- (2) Defining Evaluation Policy
- (3) Off-Policy Evaluation
- (4) Evaluation of OPE Estimators

The second step could be replaced by some Off-Policy Learning (OPL) step, but obp still does not implement any OPL module for slate bandit data. Implementing OPL for slate bandit data is our future work.
<!-- #endregion -->

```python id="otRDX7Ae95f7" executionInfo={"status": "ok", "timestamp": 1633597018179, "user_tz": -330, "elapsed": 1487, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
import numpy as np
import pandas as pd
import obp

from obp.ope import SlateStandardIPS, SlateIndependentIPS, SlateRewardInteractionIPS, SlateOffPolicyEvaluation
from obp.dataset import (
    logistic_reward_function,
    SyntheticSlateBanditDataset,
)
```

```python id="u5zod2mV95f8" executionInfo={"status": "ok", "timestamp": 1633597019342, "user_tz": -330, "elapsed": 5, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
from itertools import product
from copy import deepcopy
```

```python id="tUIq56WA95f9" executionInfo={"status": "ok", "timestamp": 1633597020880, "user_tz": -330, "elapsed": 3, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
import matplotlib.pyplot as plt
import seaborn as sns
```

```python id="9r_7NhFZ95f_" executionInfo={"status": "ok", "timestamp": 1633597026876, "user_tz": -330, "elapsed": 7, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
import warnings
warnings.filterwarnings('ignore')
```

<!-- #region id="wAyG9yNw95gB" -->
### Synthetic Slate Data Generation
We prepare easy-to-use synthetic slate data generator: `SyntheticSlateBanditDataset` class in the dataset module.

It takes the following arguments as inputs and generates a synthetic bandit dataset that can be used to evaluate the performance of decision making policies (obtained by `off-policy learning`) and OPE estimators.
- length of a list of actions recommended in each slate. (`len_list`)
- number of unique actions (`n_unique_actions`)
- dimension of context vectors (`dim_context`)
- reward type (`reward_type`)
- reward structure (`reward_structure`)
- click model (`click_model`)
- base reward function (`base_reward_function`)
- behavior policy (`behavior_policy_function`)

We use a uniform random policy as a behavior policy here.
<!-- #endregion -->

```python tags=[] id="tTLwXe8s95gC" executionInfo={"status": "ok", "timestamp": 1633597038196, "user_tz": -330, "elapsed": 1791, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
# generate a synthetic bandit dataset with 10 actions
# we use `logistic_reward_function` as the reward function and `linear_behavior_policy_logit` as the behavior policy.
# one can define their own reward function and behavior policy such as nonlinear ones. 

n_unique_action=10
len_list = 3
dim_context = 2
reward_type = "binary"
reward_structure="cascade_additive"
click_model=None
random_state=12345
base_reward_function=logistic_reward_function

# obtain  test sets of synthetic logged bandit feedback
n_rounds_test = 10000
```

```python id="-pNeWEze95gD" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1633597044687, "user_tz": -330, "elapsed": 4577, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="849784f0-8f63-49c7-eabf-6f030aa068f1"
# define Uniform Random Policy as a baseline behavior policy
dataset_with_random_behavior = SyntheticSlateBanditDataset(
    n_unique_action=n_unique_action,
    len_list=len_list,
    dim_context=dim_context,
    reward_type=reward_type,
    reward_structure=reward_structure,
    click_model=click_model,
    random_state=random_state,
    behavior_policy_function=None,  # set to uniform random
    base_reward_function=base_reward_function,
)

# compute the factual action choice probabililties for the test set of the synthetic logged bandit feedback
bandit_feedback_with_random_behavior = dataset_with_random_behavior.obtain_batch_bandit_feedback(
    n_rounds=n_rounds_test,
    return_pscore_item_position=True,
)

# print policy value
random_policy_value = dataset_with_random_behavior.calc_on_policy_policy_value(
    reward=bandit_feedback_with_random_behavior["reward"],
    slate_id=bandit_feedback_with_random_behavior["slate_id"],
)
print(random_policy_value)
```

<!-- #region id="BeXxTJJV95gF" -->
### Evaluation Policy Definition (Off-Policy Learning)
 After generating synthetic data, we now define the evaluation policy as follows:
 
1. Generate logit values of three valuation policies (`random`, `optimal`, and `anti-optimal`).
  - A `optimal` policy is defined by a policy that samples actions using`3 * base_expected_reward`.
  - An `anti-optimal` policy is defined by a policy that samples actions using the sign inversion of `-3 * base_expected_reward`.
2. Obtain pscores of the evaluation policies by `obtain_pscore_given_evaluation_policy_logit` method.
<!-- #endregion -->

```python id="sORmtxg595gG" executionInfo={"status": "ok", "timestamp": 1633597054289, "user_tz": -330, "elapsed": 943, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
random_policy_logit_ = np.zeros((n_rounds_test, n_unique_action))
```

```python id="U7G1m2WZ95gG" executionInfo={"status": "ok", "timestamp": 1633597055055, "user_tz": -330, "elapsed": 6, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
base_expected_reward = dataset_with_random_behavior.base_reward_function(
    context=bandit_feedback_with_random_behavior["context"],
    action_context=dataset_with_random_behavior.action_context,
    random_state=dataset_with_random_behavior.random_state,
)
```

```python id="HNbbnAWg95gH" executionInfo={"status": "ok", "timestamp": 1633597055056, "user_tz": -330, "elapsed": 6, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
optimal_policy_logit_ = base_expected_reward * 3
anti_optimal_policy_logit_ = -3 * base_expected_reward
```

```python id="FjIWT5m395gH" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1633597079787, "user_tz": -330, "elapsed": 23396, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="010f202e-2c90-4785-feda-a5fd14b1e559"
random_policy_pscores = dataset_with_random_behavior.obtain_pscore_given_evaluation_policy_logit(
    action=bandit_feedback_with_random_behavior["action"],
    evaluation_policy_logit_=random_policy_logit_
)
```

```python id="vBx8pnnJ95gI" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1633597107902, "user_tz": -330, "elapsed": 28134, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="c74783ab-0752-4362-95f3-9f14b17d3615"
optimal_policy_pscores = dataset_with_random_behavior.obtain_pscore_given_evaluation_policy_logit(
    action=bandit_feedback_with_random_behavior["action"],
    evaluation_policy_logit_=optimal_policy_logit_
)
```

```python id="gFKKNUQ895gK" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1633597136586, "user_tz": -330, "elapsed": 28721, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="43507e0c-118d-47e8-cc9c-f049b7322a9a"
anti_optimal_policy_pscores = dataset_with_random_behavior.obtain_pscore_given_evaluation_policy_logit(
    action=bandit_feedback_with_random_behavior["action"],
    evaluation_policy_logit_=anti_optimal_policy_logit_
)
```

<!-- #region id="98oHt2rR95gL" -->
### Off-Policy Evaluation (OPE)
Our next step is OPE which attempts to estimate the performance of evaluation policies using the logged bandit feedback and OPE estimators.

Here, we use the **SlateStandardIPS (SIPS)**, **SlateIndependentIPS (IIPS)**, and **SlateRewardInteractionIPS (RIPS)** estimators and visualize the OPE results.
<!-- #endregion -->

```python tags=[] id="aRQVUoP-95gL" executionInfo={"status": "ok", "timestamp": 1633597136587, "user_tz": -330, "elapsed": 19, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
# estimate the policy value of the evaluation policies based on their action choice probabilities
# it is possible to set multiple OPE estimators to the `ope_estimators` argument

sips = SlateStandardIPS(len_list=len_list)
iips = SlateIndependentIPS(len_list=len_list)
rips = SlateRewardInteractionIPS(len_list=len_list)

ope = SlateOffPolicyEvaluation(
    bandit_feedback=bandit_feedback_with_random_behavior,
    ope_estimators=[sips, iips, rips]
)
```

```python id="bG9zOeZS95gM" colab={"base_uri": "https://localhost:8080/", "height": 505} executionInfo={"status": "ok", "timestamp": 1633597156033, "user_tz": -330, "elapsed": 19463, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="fe5d64a1-d5d6-4e35-a024-212659ee7c12"
_, estimated_interval_random = ope.summarize_off_policy_estimates(
    evaluation_policy_pscore=random_policy_pscores[0],
    evaluation_policy_pscore_item_position=random_policy_pscores[1],
    evaluation_policy_pscore_cascade=random_policy_pscores[2],
    alpha=0.05,
    n_bootstrap_samples=1000,
    random_state=dataset_with_random_behavior.random_state,
)
estimated_interval_random["policy_name"] = "random"

print(estimated_interval_random, '\n')
# visualize estimated policy values of Uniform Random by the three OPE estimators
# and their 95% confidence intervals (estimated by nonparametric bootstrap method)
ope.visualize_off_policy_estimates(
    evaluation_policy_pscore=random_policy_pscores[0],
    evaluation_policy_pscore_item_position=random_policy_pscores[1],
    evaluation_policy_pscore_cascade=random_policy_pscores[2],
    alpha=0.05,
    n_bootstrap_samples=1000, # number of resampling performed in the bootstrap procedure
    random_state=dataset_with_random_behavior.random_state,
)
```

```python id="tkKXMt_595gM" colab={"base_uri": "https://localhost:8080/", "height": 505} executionInfo={"status": "ok", "timestamp": 1633597172894, "user_tz": -330, "elapsed": 16871, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="281ceef0-44c7-4aee-8ca8-a314da8a20e8"
_, estimated_interval_optimal = ope.summarize_off_policy_estimates(
    evaluation_policy_pscore=optimal_policy_pscores[0],
    evaluation_policy_pscore_item_position=optimal_policy_pscores[1],
    evaluation_policy_pscore_cascade=optimal_policy_pscores[2],
    alpha=0.05,
    n_bootstrap_samples=1000,
    random_state=dataset_with_random_behavior.random_state,
)

estimated_interval_optimal["policy_name"] = "optimal"

print(estimated_interval_optimal, '\n')
# visualize estimated policy values of Optimal by the three OPE estimators
# and their 95% confidence intervals (estimated by nonparametric bootstrap method)
ope.visualize_off_policy_estimates(
    evaluation_policy_pscore=optimal_policy_pscores[0],
    evaluation_policy_pscore_item_position=optimal_policy_pscores[1],
    evaluation_policy_pscore_cascade=optimal_policy_pscores[2],
    alpha=0.05,
    n_bootstrap_samples=1000, # number of resampling performed in the bootstrap procedure
    random_state=dataset_with_random_behavior.random_state,
)
```

```python id="h7zkG-QW95gN" colab={"base_uri": "https://localhost:8080/", "height": 505} executionInfo={"status": "ok", "timestamp": 1633597190549, "user_tz": -330, "elapsed": 17664, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="57773436-eed9-4fc0-d3af-336a5cb783d2"
_, estimated_interval_anti_optimal = ope.summarize_off_policy_estimates(
    evaluation_policy_pscore=anti_optimal_policy_pscores[0],
    evaluation_policy_pscore_item_position=anti_optimal_policy_pscores[1],
    evaluation_policy_pscore_cascade=anti_optimal_policy_pscores[2],
    alpha=0.05,
    n_bootstrap_samples=1000,
    random_state=dataset_with_random_behavior.random_state,
)
estimated_interval_anti_optimal["policy_name"] = "anti-optimal"

print(estimated_interval_anti_optimal, '\n')
# visualize estimated policy values of Anti-optimal by the three OPE estimators
# and their 95% confidence intervals (estimated by nonparametric bootstrap method)
ope.visualize_off_policy_estimates(
    evaluation_policy_pscore=anti_optimal_policy_pscores[0],
    evaluation_policy_pscore_item_position=anti_optimal_policy_pscores[1],
    evaluation_policy_pscore_cascade=anti_optimal_policy_pscores[2],
    alpha=0.05,
    n_bootstrap_samples=1000, # number of resampling performed in the bootstrap procedure
    random_state=dataset_with_random_behavior.random_state,
)
```

<!-- #region id="EcxJNdkx95gN" -->
### Evaluation of OPE estimators
Our final step is **the evaluation of OPE**, which evaluates and compares the estimation accuracy of OPE estimators.

With synthetic slate data, we can calculate the policy value of the evaluation policies. 
Therefore, we can compare the policy values estimated by OPE estimators with the ground-turths to evaluate the accuracy of OPE.
<!-- #endregion -->

```python id="k8oluKJ995gO" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1633597207415, "user_tz": -330, "elapsed": 14864, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="b60ac058-bbd5-49bc-ac09-e4adb76308da"
ground_truth_policy_value_random = dataset_with_random_behavior.calc_ground_truth_policy_value(
    context=bandit_feedback_with_random_behavior["context"],
    evaluation_policy_logit_=random_policy_logit_
)
ground_truth_policy_value_random
```

```python id="I4RfED3s95gO" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1633597224763, "user_tz": -330, "elapsed": 17393, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="10156f83-3c76-498f-e3bd-79ced7e09506"
ground_truth_policy_value_optimal = dataset_with_random_behavior.calc_ground_truth_policy_value(
    context=bandit_feedback_with_random_behavior["context"],
    evaluation_policy_logit_=optimal_policy_logit_
)
ground_truth_policy_value_optimal
```

```python id="VddS4sUJ95gP" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1633597240472, "user_tz": -330, "elapsed": 15741, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="d0096c92-61ba-46df-b660-b46ebe6084d2"
ground_truth_policy_value_anti_optimal = dataset_with_random_behavior.calc_ground_truth_policy_value(
    context=bandit_feedback_with_random_behavior["context"],
    evaluation_policy_logit_=anti_optimal_policy_logit_
)
ground_truth_policy_value_anti_optimal
```

```python id="sTQ28sHP95gP" executionInfo={"status": "ok", "timestamp": 1633597240474, "user_tz": -330, "elapsed": 15, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
estimated_interval_random["ground_truth"] = ground_truth_policy_value_random
estimated_interval_optimal["ground_truth"] = ground_truth_policy_value_optimal
estimated_interval_anti_optimal["ground_truth"] = ground_truth_policy_value_anti_optimal

estimated_intervals = pd.concat(
    [
        estimated_interval_random,
        estimated_interval_optimal,
        estimated_interval_anti_optimal
    ]
)
```

```python id="AWVpIyMN95gQ" colab={"base_uri": "https://localhost:8080/", "height": 328} executionInfo={"status": "ok", "timestamp": 1633597240475, "user_tz": -330, "elapsed": 14, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="75146e86-bfba-4634-e6bd-90a391d31b9f"
estimated_intervals
```

<!-- #region id="gjIuIbbT95gQ" -->
We can confirm that the three OPE estimators return the same results when the behavior policy and the evaluation policy is the same, and the estimates are quite similar to the `random_policy_value` calcurated above.

We can also observe that the performance of OPE estimators are as follows in this simulation: `IIPS > RIPS > SIPS`.
<!-- #endregion -->

```python id="AwjV4mwv95gQ" colab={"base_uri": "https://localhost:8080/", "height": 142} executionInfo={"status": "ok", "timestamp": 1633597244161, "user_tz": -330, "elapsed": 3697, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="9a2acace-da67-460a-bdcb-cd4abc8a70cb"
# evaluate the estimation performances of OPE estimators 
# by comparing the estimated policy values and its ground-truth.
# `summarize_estimators_comparison` returns a pandas dataframe containing estimation performances of given estimators 

relative_ee_for_random_evaluation_policy = ope.summarize_estimators_comparison(
    ground_truth_policy_value=ground_truth_policy_value_random,
    evaluation_policy_pscore=random_policy_pscores[0],
    evaluation_policy_pscore_item_position=random_policy_pscores[1],
    evaluation_policy_pscore_cascade=random_policy_pscores[2],
)
relative_ee_for_random_evaluation_policy
```

```python id="eDZt2HWL95gR" colab={"base_uri": "https://localhost:8080/", "height": 142} executionInfo={"status": "ok", "timestamp": 1633597247599, "user_tz": -330, "elapsed": 3444, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="cbb49c21-e9cc-4311-aef9-60565abd79cc"
# evaluate the estimation performances of OPE estimators 
# by comparing the estimated policy values and its ground-truth.
# `summarize_estimators_comparison` returns a pandas dataframe containing estimation performances of given estimators 

relative_ee_for_optimal_evaluation_policy = ope.summarize_estimators_comparison(
    ground_truth_policy_value=ground_truth_policy_value_optimal,
    evaluation_policy_pscore=optimal_policy_pscores[0],
    evaluation_policy_pscore_item_position=optimal_policy_pscores[1],
    evaluation_policy_pscore_cascade=optimal_policy_pscores[2],
)
relative_ee_for_optimal_evaluation_policy
```

```python id="otFmNfhB95gR" colab={"base_uri": "https://localhost:8080/", "height": 142} executionInfo={"status": "ok", "timestamp": 1633597251591, "user_tz": -330, "elapsed": 3997, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="fbabbc4d-a019-4317-8019-ddf6a78851bb"
# evaluate the estimation performances of OPE estimators 
# by comparing the estimated policy values and its ground-truth.
# `summarize_estimators_comparison` returns a pandas dataframe containing estimation performances of given estimators 

relative_ee_for_anti_optimal_evaluation_policy = ope.summarize_estimators_comparison(
    ground_truth_policy_value=ground_truth_policy_value_anti_optimal,
    evaluation_policy_pscore=anti_optimal_policy_pscores[0],
    evaluation_policy_pscore_item_position=anti_optimal_policy_pscores[1],
    evaluation_policy_pscore_cascade=anti_optimal_policy_pscores[2],
)
relative_ee_for_anti_optimal_evaluation_policy
```

<!-- #region id="s8vP0gyR95gS" -->
The variance of OPE estimators is as follows: `SIPS > RIPS > IIPS`.
<!-- #endregion -->

```python id="3DOoUt8E95gS" executionInfo={"status": "ok", "timestamp": 1633597251592, "user_tz": -330, "elapsed": 58, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
estimated_intervals["errbar_length"] = (
    estimated_intervals.drop(["mean", "policy_name", "ground_truth"], axis=1).diff(axis=1).iloc[:, -1].abs()
)
```

```python id="bpEn23fy95gS" colab={"base_uri": "https://localhost:8080/", "height": 242} executionInfo={"status": "ok", "timestamp": 1633597252297, "user_tz": -330, "elapsed": 754, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="e21bbae1-8e6e-43a6-8213-3f160132972f"
alpha = 0.05
plt.style.use("ggplot")

def errplot(x, y, yerr, **kwargs):
    ax = plt.gca()
    data = kwargs.pop("data")
    data.plot(x=x, y=y, yerr=yerr, kind="bar", ax=ax, **kwargs)
    ax.hlines(data["ground_truth"].iloc[0], -1, len(x)+1)
#     ax.set_xlabel("OPE estimator")
    
g = sns.FacetGrid(
    estimated_intervals.reset_index().rename(columns={"index": "OPE estimator", "mean": "Policy value"}),
    col="policy_name"
)
g.map_dataframe(errplot, "OPE estimator", "Policy value", "errbar_length")
plt.ylim((1.7, 1.9))
```

<!-- #region id="Vkp5EDoY95gT" -->
It is surprising that `RIPS` estimator does not achieve the best performance even if the reward structure is not independent. If we run a simulation where the reward of each position depends heavily on those of other positions, `RIPS`estimator could achieve the best performance.
<!-- #endregion -->

<!-- #region id="5DR5Riz595gT" -->
## Off-Policy Evaluation of Online Bandit Algorithms
<!-- #endregion -->

<!-- #region id="_TH7R-pd_OXM" -->
This section provides an example of conducting OPE of online bandit algorithms using Replay Method (RM) with synthetic bandit feedback data.
RM uses a subset of the logged bandit feedback data where actions selected by the behavior policy are the same as that of the evaluation policy.
Theoretically, RM is unbiased when the behavior policy is uniformly random and the evaluation policy is fixed.
However, empirically, RM works well when evaluation policies are learning algorithms.
Please refer to https://arxiv.org/abs/1003.5956 about the details of RM.

Our example with online bandit algorithms contains the follwoing three major steps:
- (1) Synthetic Data Generation
- (2) Off-Policy Evaluation (OPE)
- (3) Evaluation of OPE
<!-- #endregion -->

```python id="frUel1pK_OXV" executionInfo={"status": "ok", "timestamp": 1633597392332, "user_tz": -330, "elapsed": 499, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
import obp
from obp.dataset import (
    SyntheticBanditDataset,
    logistic_reward_function
)
from obp.policy import EpsilonGreedy, LinTS, LinUCB
from obp.ope import (
    OffPolicyEvaluation, 
    ReplayMethod
)
from obp.simulator import (
    calc_ground_truth_policy_value,
    run_bandit_simulation
)
```

<!-- #region id="kt2m4T3I_OXZ" -->
### Synthetic Data Generation
We prepare easy-to-use synthetic data generator: `SyntheticBanditDataset` class in the dataset module.

It takes number of actions (`n_actions`), dimension of context vectors (`dim_context`), reward function (`reward_function`), and behavior policy (`behavior_policy_function`) as inputs and generates a synthetic bandit dataset that can be used to evaluate the performance of decision making policies (obtained by `off-policy learning`) and OPE estimators.
<!-- #endregion -->

```python tags=[] id="14i72_Va_OXa" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1633597405684, "user_tz": -330, "elapsed": 1156, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="98e6e899-d77e-4afa-9580-672c80bb86c4"
# generate a synthetic bandit dataset with 10 actions
# we use `logistic function` as the reward function
# we use the uniformly random behavior policy because it is desriable for RM
# one can define their own reward function and behavior policy such as nonlinear ones. 
dataset = SyntheticBanditDataset(
    n_actions=10,
    dim_context=5,
    reward_type="binary", # "binary" or "continuous"
    reward_function=logistic_reward_function,
    behavior_policy_function=None, # uniformly random
    random_state=12345,
)
# obtain a set of synthetic logged bandit feedback
n_rounds = 10000
bandit_feedback = dataset.obtain_batch_bandit_feedback(n_rounds=n_rounds)

# `bandit_feedback` is a dictionary storing synthetic logged bandit feedback
bandit_feedback
```

<!-- #region id="Oll2aqRm_OXd" -->
### Off-Policy Evaluation (OPE)
Our next step is OPE which attempts to estimate the performance of online bandit algorithms using the logged bandit feedback and RM.

Here, we visualize the OPE results.
<!-- #endregion -->

```python id="iOklgfLI_OXf" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1633597431631, "user_tz": -330, "elapsed": 19327, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="d23056d0-41d7-4064-cb45-5134637088c0"
# simulations of online bandit algorithms
# obtain a deterministic action distribution representing which action is selected at each round in the simulation
# policies are updated only when the selected action is the same as that of the logged data
epsilon_greedy = EpsilonGreedy(
    n_actions=dataset.n_actions,
    epsilon=0.1,
    random_state=12345
)
action_dist_epsilon_greedy = run_bandit_simulation(
    bandit_feedback=bandit_feedback,
    policy=epsilon_greedy
)

lin_ts = LinTS(
    dim=dataset.dim_context,
    n_actions=dataset.n_actions,
    random_state=12345
)
action_dist_lin_ts = run_bandit_simulation(
    bandit_feedback=bandit_feedback,
    policy=lin_ts
)

lin_ucb = LinUCB(
    dim=dataset.dim_context,
    n_actions=dataset.n_actions,
    random_state=12345
)
action_dist_lin_ucb = run_bandit_simulation(
    bandit_feedback=bandit_feedback,
    policy=lin_ucb
)
```

```python tags=[] id="7vSP3XE4_OXh" executionInfo={"status": "ok", "timestamp": 1633597431632, "user_tz": -330, "elapsed": 31, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
# estimate the policy value of the online bandit algorithms using RM
ope = OffPolicyEvaluation(
    bandit_feedback=bandit_feedback,
    ope_estimators=[ReplayMethod()]
)
```

```python tags=[] id="Pwhm2PIN_OXh" colab={"base_uri": "https://localhost:8080/", "height": 475} executionInfo={"status": "ok", "timestamp": 1633597433767, "user_tz": -330, "elapsed": 2164, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="4c0ad809-c2a9-4405-b436-8c87939ad47e"
# estimate the policy value of EpsilonGreedy
estimated_policy_value_epsilon_greedy, estimated_interval_epsilon_greedy = ope.summarize_off_policy_estimates(
    action_dist=action_dist_epsilon_greedy
)
print(estimated_interval_epsilon_greedy, '\n')

# visualize estimated policy values of EpsilonGreedy by the three OPE estimators
# and their 95% confidence intervals (estimated by nonparametric bootstrap method)
ope.visualize_off_policy_estimates(
    action_dist=action_dist_epsilon_greedy,
    n_bootstrap_samples=10000, # number of resampling performed in the bootstrap procedure
    random_state=12345,
)
```

```python id="MN1N8bVx_OXj" colab={"base_uri": "https://localhost:8080/", "height": 475} executionInfo={"status": "ok", "timestamp": 1633597435984, "user_tz": -330, "elapsed": 2234, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="6f9a3aa9-3eb4-4340-ef21-14418468ee95"
# estimate the policy value of LinTS
estimated_policy_value_lin_ts, estimated_interval_lin_ts = ope.summarize_off_policy_estimates(
    action_dist=action_dist_lin_ts
)
print(estimated_interval_lin_ts, '\n')

# visualize estimated policy values of LinTS by the three OPE estimators
# and their 95% confidence intervals (estimated by nonparametric bootstrap method)
ope.visualize_off_policy_estimates(
    action_dist=action_dist_lin_ts,
    n_bootstrap_samples=10000, # number of resampling performed in the bootstrap procedure
    random_state=12345,
)
```

```python id="IL0UWk8I_OXj" colab={"base_uri": "https://localhost:8080/", "height": 475} executionInfo={"status": "ok", "timestamp": 1633597437865, "user_tz": -330, "elapsed": 1899, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="d722d805-c779-4720-94eb-1665a62ccd63"
# estimate the policy value of LinUCB
estimated_policy_value_lin_ucb, estimated_interval_lin_ucb = ope.summarize_off_policy_estimates(
    action_dist=action_dist_lin_ucb
)
print(estimated_interval_lin_ucb, '\n')

# visualize estimated policy values of LinUCB by the three OPE estimators
# and their 95% confidence intervals (estimated by nonparametric bootstrap method)
ope.visualize_off_policy_estimates(
    action_dist=action_dist_lin_ucb,
    n_bootstrap_samples=10000, # number of resampling performed in the bootstrap procedure
    random_state=12345,
)
```

<!-- #region id="HMF8sJnY_OXk" -->
RM estimates that LinTS is the best policy.
<!-- #endregion -->

<!-- #region id="9rNmK3js_OXl" -->
### Evaluation of OPE
Our final step is **the evaluation of OPE**, which evaluates and compares the estimation accuracy of OPE estimators.

With synthetic data, we can calculate the policy value of the evaluation policies. 
Therefore, we can compare the policy values estimated by RM with the ground-turths to evaluate the accuracy of OPE.
<!-- #endregion -->

```python tags=[] id="Xjni-SCg_OXl" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1633597541472, "user_tz": -330, "elapsed": 103624, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="04d5d639-86f0-4201-c075-07b164b7ba3d"
# we first calculate the policy values of the three evaluation policies
# in synthetic data, we know p(r|x,a), the reward distribution, so we can perform simulations
# here, a policy is updated at each round according to actions and rewards sampled from the policy and p(r|x,a)
# the policy values are averaged over `n_sim` simulations
policy_value_epsilon_greedy = calc_ground_truth_policy_value(
    bandit_feedback=bandit_feedback,
    reward_sampler=dataset.sample_reward, # p(r|x,a)
    policy=epsilon_greedy,
    n_sim=3 # the number of simulations
)
policy_value_lin_ts = calc_ground_truth_policy_value(
    bandit_feedback=bandit_feedback,
    reward_sampler=dataset.sample_reward, # p(r|x,a)
    policy=lin_ts,
    n_sim=3 # the number of simulations
)
policy_value_lin_ucb = calc_ground_truth_policy_value(
    bandit_feedback=bandit_feedback,
    reward_sampler=dataset.sample_reward, # p(r|x,a)
    policy=lin_ucb,
    n_sim=3 # the number of simulations
)

print(f'policy value of EpsilonGreedy: {policy_value_epsilon_greedy}')
print(f'policy value of LinTS: {policy_value_lin_ts}')
print(f'policy value of LinUCB: {policy_value_lin_ucb}')
```

<!-- #region id="IgTay9Tx_OXm" -->
In fact, LinTS reveals the best performance among the three evaluation policies.

Using the above policy values, we evaluate the estimation accuracy of the OPE estimators.
<!-- #endregion -->

```python id="NNLv1aNV_OXm" colab={"base_uri": "https://localhost:8080/", "height": 80} executionInfo={"status": "ok", "timestamp": 1633597541473, "user_tz": -330, "elapsed": 22, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="7a178279-2b0e-4193-a96a-ed54b1011688"
# evaluate the estimation performances of OPE estimators 
# by comparing the estimated policy values of EpsilonGreedy and its ground-truth.
# `summarize_estimators_comparison` returns a pandas dataframe containing estimation performances of given estimators 
relative_ee_epsilon_greedy = ope.summarize_estimators_comparison(
    ground_truth_policy_value=policy_value_epsilon_greedy,
    action_dist=action_dist_epsilon_greedy,
    metric="relative-ee", # "relative-ee" (relative estimation error) or "se" (squared error)
)

# estimation performances of the three estimators (lower means accurate)
relative_ee_epsilon_greedy
```

```python id="6ZJScQdy_OXo" colab={"base_uri": "https://localhost:8080/", "height": 80} executionInfo={"status": "ok", "timestamp": 1633597544648, "user_tz": -330, "elapsed": 19, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="4c4a54bb-94cc-475c-d58d-bffdd1e53ab5"
# evaluate the estimation performance of OPE estimators 
# by comparing the estimated policy values of LinTS t and its ground-truth.
# `summarize_estimators_comparison` returns a pandas dataframe containing estimation performances of given estimators 
relative_ee_lin_ts = ope.summarize_estimators_comparison(
    ground_truth_policy_value=policy_value_lin_ts,
    action_dist=action_dist_lin_ts,
    metric="relative-ee", # "relative-ee" (relative estimation error) or "se" (squared error)
)

# estimation performances of the three estimators (lower means accurate)
relative_ee_lin_ts
```

```python id="7dXzbbB2_OXp" colab={"base_uri": "https://localhost:8080/", "height": 80} executionInfo={"status": "ok", "timestamp": 1633597544651, "user_tz": -330, "elapsed": 17, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="4f918b29-b6ac-4242-df26-8464d051fb9b"
# evaluate the estimation performance of OPE estimators 
# by comparing the estimated policy values of LinUCB and its ground-truth.
# `summarize_estimators_comparison` returns a pandas dataframe containing estimation performances of given estimators 
relative_ee_lin_ucb = ope.summarize_estimators_comparison(
    ground_truth_policy_value=policy_value_lin_ucb,
    action_dist=action_dist_lin_ucb,
    metric="relative-ee", # "relative-ee" (relative estimation error) or "se" (squared error)
)

# estimation performances of the three estimators (lower means accurate)
relative_ee_lin_ucb
```

<!-- #region id="yXoMpQFu_OXr" -->
## Off-Policy Learners
<!-- #endregion -->

<!-- #region id="uEgxPU4Y_1ZU" -->
This section provides an example of implementing several off-policy learning methods with synthetic logged bandit data.

The example consists of the follwoing four major steps:
- (1) Generating Synthetic Data
- (2) Off-Policy Learning
- (3) Evaluation of Off-Policy Learners
<!-- #endregion -->

```python id="Qh_encBN_1Zh" executionInfo={"status": "ok", "timestamp": 1633597549771, "user_tz": -330, "elapsed": 10, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
from sklearn.ensemble import RandomForestClassifier as RandomForest
from sklearn.linear_model import LogisticRegression

import obp
from obp.dataset import (
    SyntheticBanditDataset,
    logistic_reward_function,
    linear_reward_function,
    linear_behavior_policy
)
from obp.policy import (
    IPWLearner, 
    NNPolicyLearner, 
    Random
)
```

<!-- #region id="KDWmCdg__1Zn" -->
### Generating Synthetic Data
`obp.dataset.SyntheticBanditDataset` is an easy-to-use synthetic data generator.

It takes 
- number of actions (`n_actions`, $|\mathcal{A}|$)
- dimension of context vectors (`dim_context`, $d$)
- reward function (`reward_function`, $q(x,a)=\mathbb{E}[r \mid x,a]$)
- behavior policy (`behavior_policy_function`, $\pi_b(a|x)$) 

as inputs and generates a synthetic logged bandit data that can be used to evaluate the performance of decision making policies (obtained by `off-policy learning`).
<!-- #endregion -->

```python tags=[] id="CApFXCdh_1Zw" executionInfo={"status": "ok", "timestamp": 1633597556758, "user_tz": -330, "elapsed": 884, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
# generate a synthetic bandit dataset with 10 actions
# we use `logistic function` as the reward function and `linear_behavior_policy` as the behavior policy.
# one can define their own reward function and behavior policy such as nonlinear ones. 
dataset = SyntheticBanditDataset(
    n_actions=10,
    dim_context=5,
    tau=0.2, # temperature hyperparameter to control the entropy of the behavior policy
    reward_type="binary", # "binary" or "continuous"
    reward_function=logistic_reward_function,
    behavior_policy_function=linear_behavior_policy,
    random_state=12345,
)
```

```python id="q3bApAj8_1Zz" executionInfo={"status": "ok", "timestamp": 1633597557549, "user_tz": -330, "elapsed": 5, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
# obtain training and test sets of synthetic logged bandit data
n_rounds_train, n_rounds_test = 10000, 10000
bandit_feedback_train = dataset.obtain_batch_bandit_feedback(n_rounds=n_rounds_train)
bandit_feedback_test = dataset.obtain_batch_bandit_feedback(n_rounds=n_rounds_test)
```

<!-- #region id="05y20gQh_1Z2" -->
the logged bandit data is collected by the behavior policy as follows.

$ \mathcal{D}_b := \{(x_i,a_i,r_i)\}_{i=1}^n$  where $(x,a,r) \sim p(x)\pi_b(a \mid x)p(r \mid x,a) $
<!-- #endregion -->

```python id="cZpfXLoo_1Z6" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1633597558349, "user_tz": -330, "elapsed": 14, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="fb689c50-6783-4571-f2ca-d932d181f71f"
# `bandit_feedback` is a dictionary storing synthetic logged bandit feedback
bandit_feedback_train
```

<!-- #region id="YfdDMdAj_1Z-" -->
### Off-Policy Learning
After generating synthetic data, we now train some decision making policies.

To train policies, we use

- `obp.policy.NNPolicyLearner` (Neural Network Policy Learner)
- `obp.policy.IPWLearner`

For NN Learner, we use 
- Direct Method ("dm")
- InverseProbabilityWeighting ("ipw")
- DoublyRobust ("dr") 

as its objective functions (`off_policy_objective`). 

For IPW Learner, we use *RandomForestClassifier* and *LogisticRegression* implemented in scikit-learn for base machine learning methods.
<!-- #endregion -->

<!-- #region id="qYZ0kfkg_1aA" -->
A policy is trained by maximizing an OPE estimator as an objective function as follows.

$$ \hat{\pi} \in \arg \max_{\pi \in \Pi} \hat{V} (\pi; \mathcal{D}_{tr}) - \lambda \cdot \Omega (\pi)  $$

where $\hat{V}(\cdot; \mathcal{D})$ is an off-policy objective and $\mathcal{D}_{tr}$ is a training bandit dataset. $\Omega (\cdot)$ is a regularization term.
<!-- #endregion -->

```python id="EKoUT-6S_1aB" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1633597658367, "user_tz": -330, "elapsed": 91532, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="e3044c08-f7dc-428c-9e14-b2f5d05f5c97"
# define NNPolicyLearner with DM as its objective function
nn_dm = NNPolicyLearner(
    n_actions=dataset.n_actions,
    dim_context=dataset.dim_context,
    off_policy_objective="dm",
    batch_size=64,
    random_state=12345,
)

# train NNPolicyLearner on the training set of logged bandit data
nn_dm.fit(
    context=bandit_feedback_train["context"],
    action=bandit_feedback_train["action"],
    reward=bandit_feedback_train["reward"],
)

# obtains action choice probabilities for the test set
action_dist_nn_dm = nn_dm.predict_proba(
    context=bandit_feedback_test["context"]
)
```

```python id="s1gkS-Ta_1aD" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1633597717211, "user_tz": -330, "elapsed": 58875, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="5fd4fa7e-5b18-4f54-e9a4-51e51ab46314"
# define NNPolicyLearner with IPW as its objective function
nn_ipw = NNPolicyLearner(
    n_actions=dataset.n_actions,
    dim_context=dataset.dim_context,
    off_policy_objective="ipw",
    batch_size=64,
    random_state=12345,
)

# train NNPolicyLearner on the training set of logged bandit data
nn_ipw.fit(
    context=bandit_feedback_train["context"],
    action=bandit_feedback_train["action"],
    reward=bandit_feedback_train["reward"],
    pscore=bandit_feedback_train["pscore"],
)

# obtains action choice probabilities for the test set
action_dist_nn_ipw = nn_ipw.predict_proba(
    context=bandit_feedback_test["context"]
)
```

```python id="zPwoKJlB_1aE" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1633597823777, "user_tz": -330, "elapsed": 106598, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="59d77af1-d012-45dc-d245-687879232666"
# define NNPolicyLearner with DR as its objective function
nn_dr = NNPolicyLearner(
    n_actions=dataset.n_actions,
    dim_context=dataset.dim_context,
    off_policy_objective="dr",
    batch_size=64,
    random_state=12345,
)

# train NNPolicyLearner on the training set of logged bandit data
nn_dr.fit(
    context=bandit_feedback_train["context"],
    action=bandit_feedback_train["action"],
    reward=bandit_feedback_train["reward"],
    pscore=bandit_feedback_train["pscore"],
)

# obtains action choice probabilities for the test set
action_dist_nn_dr = nn_dr.predict_proba(
    context=bandit_feedback_test["context"]
)
```

```python tags=[] id="BsyJ1z56_1aF" executionInfo={"status": "ok", "timestamp": 1633597823778, "user_tz": -330, "elapsed": 37, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
# define IPWLearner with Logistic Regression as its base ML model
ipw_lr = IPWLearner(
    n_actions=dataset.n_actions,
    base_classifier=LogisticRegression(C=100, random_state=12345)
)

# train IPWLearner on the training set of logged bandit data
ipw_lr.fit(
    context=bandit_feedback_train["context"],
    action=bandit_feedback_train["action"],
    reward=bandit_feedback_train["reward"],
    pscore=bandit_feedback_train["pscore"]
)

# obtains action choice probabilities for the test set
action_dist_ipw_lr = ipw_lr.predict(
    context=bandit_feedback_test["context"]
)
```

```python tags=[] id="B8SinaaU_1aH" executionInfo={"status": "ok", "timestamp": 1633597824612, "user_tz": -330, "elapsed": 868, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
# define IPWLearner with Random Forest as its base ML model
ipw_rf = IPWLearner(
    n_actions=dataset.n_actions,
    base_classifier=RandomForest(
        n_estimators=30, min_samples_leaf=10, random_state=12345
    )
)

# train IPWLearner on the training set of logged bandit data
ipw_rf.fit(
    context=bandit_feedback_train["context"],
    action=bandit_feedback_train["action"],
    reward=bandit_feedback_train["reward"],
    pscore=bandit_feedback_train["pscore"]
)

# obtains action choice probabilities for the test set
action_dist_ipw_rf = ipw_rf.predict(
    context=bandit_feedback_test["context"]
)
```

```python id="IdwjuJrt_1aI" executionInfo={"status": "ok", "timestamp": 1633597824614, "user_tz": -330, "elapsed": 49, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
# define Uniform Random Policy as a baseline evaluation policy
random = Random(n_actions=dataset.n_actions,)

# compute the action choice probabilities for the test set
action_dist_random = random.compute_batch_action_dist(
    n_rounds=bandit_feedback_test["n_rounds"]
)
```

```python id="xAuWBR6I_1aJ" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1633597824615, "user_tz": -330, "elapsed": 48, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="697a320a-2f68-4250-a625-a0e587afb135"
# action_dist is a probability distribution over actions (can be deterministic)
action_dist_ipw_lr[:, :, 0]
```

<!-- #region id="ndKWYwWJ_1aL" -->
### Evaluation of Off-Policy Learners
Our final step is the evaluation and comparison of the off-policy learnres.

With synthetic data, we can calculate the policy value of the off-policy learners as follows. 

$$V(\pi_e) \approx \frac{1}{|\mathcal{D}_{te}|} \sum_{i=1}^{|\mathcal{D}_{te}|} \mathbb{E}_{a \sim \pi_e(a|x_i)} [q(x_i, a)], \; \, where \; \, q(x,a) := \mathbb{E}_{r \sim p(r|x,a)} [r]$$

where $\mathcal{D}_{te}$ is the test set of logged bandit data.
<!-- #endregion -->

```python tags=[] id="mToOwsjc_1aM" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1633597824616, "user_tz": -330, "elapsed": 27, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="c9c49768-9bbb-4e7e-8f54-2247f02ab33b"
# we calculate the policy values of the trained policies based on the expected rewards of the test data
policy_names = [
    "NN Policy Learner with DM",
    "NN Policy Learner with IPW",
    "NN Policy Learner with DR",
    "IPW Learner with Logistic Regression",
    "IPW Learner with Random Forest",
    "Unifrom Random"
]
action_dist_list = [
    action_dist_nn_dm,
    action_dist_nn_ipw,
    action_dist_nn_dr,
    action_dist_ipw_lr,
    action_dist_ipw_rf,
    action_dist_random
]

for name, action_dist in zip(policy_names, action_dist_list):
    true_policy_value = dataset.calc_ground_truth_policy_value(
        expected_reward=bandit_feedback_test["expected_reward"],
        action_dist=action_dist,
    )
    print(f'policy value of {name}: {true_policy_value}')
```

<!-- #region id="C-1gl4-b_1aN" -->
In fact, NNPolicyLearner maximizing the DM estimator seems the best in this simple setting.
<!-- #endregion -->

<!-- #region id="uIvyAF2y_1aO" -->
We can iterate the above process several times to get more relibale results.
<!-- #endregion -->
