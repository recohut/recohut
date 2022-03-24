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

<!-- #region id="WG4fo-sbXvi3" -->
# Offline Policy Evaluation with VW Command Line
<!-- #endregion -->

<!-- #region id="JLa718n7Un3r" -->
VW implements several estimators to reduce policy evaluation to supervised learning-type evaluation. The simplest method, the direct method (DM), simply trains a regression model that estimates the cost (negative reward) of an (action, context) pair. As you might suspect, this method is generally biased, because the partial information problem means you typically see many more rewards for good actions than bad ones (assuming your production policy is working normally). Biased estimators should not be used for offline policy evaluation, but VW implements provably unbiased estimators like inverse propensity weighting (IPS) and doubly robust (DR) that can be used for this purpose.
<!-- #endregion -->

<!-- #region id="FON2pvvuUqDs" -->
## Batch scenario: policy evaluation with a pre-trained VW policy, cb-format data
<!-- #endregion -->

<!-- #region id="7OnRU1F3UzuV" -->
Let’s say you have collected the following bandit data from your production policy, and that the data is ordered such that the oldest data is first (don’t worry about the actual numbers in this toy example):
<!-- #endregion -->

<!-- #region id="sxq7udK1U3eV" -->
```text
1:1:0.5 | user_age:25
2:0:0.5 | user_age:25
2:1:0.5 | user_age:55
2:1:0.5 | user_age:56
2:1:0.5 | user_age:55
2:1:0.5 | user_age:56
1:1:0.5 | user_age:27
2:0:0.5 | user_age:21
2:0:0.5 | user_age:23
2:0:0.5 | user_age:56
2:1:0.5 | user_age:55
2:1:0.5 | user_age:36
1:1:0.5 | user_age:25
2:0:0.5 | user_age:25
2:1:0.5 | user_age:55
2:1:0.5 | user_age:56
2:1:0.5 | user_age:55
2:1:0.5 | user_age:56
1:1:0.5 | user_age:27
2:0:0.5 | user_age:21
2:0:0.5 | user_age:23
2:0:0.5 | user_age:56
2:1:0.5 | user_age:55
2:1:0.5 | user_age:36
```
<!-- #endregion -->

<!-- #region id="0OcD8EngVEd8" -->
In order to do OPE, it is useful to think carefully about what you wish to evaluate. In a batch setting, you are interested in training a candidate policy and evaluating its performance on unseen data that is fresher than what you trained on. So, let’s split our data into two files. Starting from the oldest data first, we do e.g. and 70%/30% split, and save the results as train.dat and test.dat:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="-KeLIaPtVFKr" executionInfo={"status": "ok", "timestamp": 1633686951622, "user_tz": -330, "elapsed": 729, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="64462adc-8abe-4b63-a439-7b664dfa2f17"
%%writefile train.dat
1:1:0.5 | user_age:25
2:0:0.5 | user_age:25
2:1:0.5 | user_age:55
2:1:0.5 | user_age:56
2:1:0.5 | user_age:55
2:1:0.5 | user_age:56
1:1:0.5 | user_age:27
2:0:0.5 | user_age:21
2:0:0.5 | user_age:23
2:0:0.5 | user_age:56
2:1:0.5 | user_age:55
2:1:0.5 | user_age:36
1:1:0.5 | user_age:25
2:0:0.5 | user_age:25
2:1:0.5 | user_age:55
2:1:0.5 | user_age:56
```

```python colab={"base_uri": "https://localhost:8080/"} id="6ilx6kZzVJB7" executionInfo={"status": "ok", "timestamp": 1633686964218, "user_tz": -330, "elapsed": 594, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="747194a0-ad50-4bdf-b90a-cf1e4f60a786"
%%writefile test.dat
2:1:0.5 | user_age:55
2:1:0.5 | user_age:56
1:1:0.5 | user_age:27
2:0:0.5 | user_age:21
2:0:0.5 | user_age:23
2:0:0.5 | user_age:56
2:1:0.5 | user_age:55
2:1:0.5 | user_age:36
```

<!-- #region id="fIRQjPR6VL0x" -->
Before continuing, it is worth understanding that policy value estimators such as IPS, DM and DR aren’t only useful for policy value estimation. Since they provide us a way to fill in fake rewards for untaken actions, they allow us to reduce bandit learning to supervised learning, and used to train policies. For example, say you have a (biased) DM estimator. For each untaken action per round, you can predict a reward, thus forming a supervised learning example where the loss of each action is known (estimated). You can then train an importance-weighted classification model, or even a regression model that estimates costs of arms given contexts, and use these models as policies. This is, in fact, what VW does: estimators serve a dual purpose and are used not only for evaluation, but also optimisation/training.

Now that we know that the same estimators that are used for OPE can also be used to train policies, let’s train a new candidate policy on the train set using e.g. IPS, and save the model as candidate-model.vw. In this instance we have two arms, so the command will look something like:
<!-- #endregion -->

```python id="_0trP5tiV-Q2"
!sudo apt-get install vowpal-wabbit
```

```python colab={"base_uri": "https://localhost:8080/"} id="4SjIK64pVv4h" executionInfo={"status": "ok", "timestamp": 1633687188688, "user_tz": -330, "elapsed": 597, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="ad34c5f8-e8ac-4e28-c7b4-e61f98a73e63"
!vw --cb 2 --cb_type ips -d train.dat -f candidate-model.vw
```

<!-- #region id="qOK1ks71V13z" -->
The average loss above is of less interest to us in this scenario since we have a separate test set. Let’s load candidate-model.vw using -i and test against our test set. Remember to use the -t flag to disable learning, and to use the same cb_type as you did when training:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="E84AqVqdWPzx" executionInfo={"status": "ok", "timestamp": 1633687248908, "user_tz": -330, "elapsed": 618, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="255b8a4f-654f-4eb4-a54a-b5644d7d47e7"
!vw -i candidate-model.vw --cb_type ips -t -d test.dat
```

<!-- #region id="FzqaRfkXWRpm" -->
The average loss reported is the OPE estimate for this policy, and in this case, calculated against our test set. Since we specified --cb_type ips, the IPS estimator is used, which is unbiased. Feel free to use dr, too, but note that although VW will allow it, the use of dm is discouraged for OPE since it is biased. If you are unsure, we suggest using dr. Note that VW will complain if you train using one cb_type but test using another; mixing estimators in training and evaluation is currently not supported.

Now that you have an OPE estimate of 0.250000, how does it compare to the production policy in production? This comparison is easy to make since, for the same time period, we have the ground truth in the test.dat file. If we sum the costs in that file and divide by the number of examples, we get 0.625. Generally, our toy example has far too little data with which to perform reliable estimates, but the principle applied: lower is better, so your candidate policy is estimated to perform better than the production policy in production. The exact definition of OPE is important: in this case, it means that had you deployed the candidate policy, with no exploration, instead of the production policy you could have expected to see the average cost reduce from 0.625 to 0.250000 for the period of time covered in the test set.

Feel free to gridsearch several candidate policies using the same setup, to determine a combination of hyperparameters that work well for your use case. Note however that mixing different cb_type options when gridsearching is discouraged, even if all specified estimators are unbiased. Choose either IPS or DR beforehand.
<!-- #endregion -->

<!-- #region id="ctZCTSn5WaBb" -->
## Batch scenario: policy evaluation with a pre-trained VW policy, cb_adf-format data
The cb_adf format is especially useful if you have rich features associated with an arm, or a variable number of arms per round. If you have adf-format data, the same procedure as above applies – just change cb to cb_adf in the corresponding commands.
<!-- #endregion -->

<!-- #region id="GTMLnQKnWfhD" -->
## Online scenario: policy evaluation with an incrementally trained VW policy, cb-format data
In the online scenario, when you deploy a new policy behind e.g. a REST endpoint, that policy will continue to update itself from second to second, as and when new examples come in. Learning is incremental: you don’t iterate over the same training examples more than once.

Online learning is particularly useful in settings where you need to react to changes in the world as fast as possible. From the point of view of OPE, the objective is the same: determining if a candidate policy is better than the one currently in production. But the setup differs slightly: since any policy deployed will continue to learn online, the key question is how well it will generalise to new examples coming in, considering that the policy is constantly evolving. So in this case, our candidate policy is in fact not a fixed policy, but one that changes constantly. It may help to think of it as a set of hyperparameters instead. The aim is to find out which set of these hyperparameters learns best in an online fashion.

To answer this question, we leverage progressive validation (PV), a validation process implemented in VW. PV is explained in detail elsewhere (see e.g. John Langford’s talk on real world interactive learning), but for this tutorial, it is enough to know that for one-pass learning, the loss reported by progressive validation deviates like a test set yet allows you to train on all of your data. It’s a good indicator of the generalisation performance for online learning.

VW reports PV loss automatically if you only iterate once over your data when training, i.e. --passes is 1 (the default). In this case, obtaining an OPE estimate is simply a matter of taking the bandit data from the production policy, again ordered oldest example first, and training as normal – no train/test split is needed.

First, save your bandit data to a file, e.g data.dat:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="W0Ds4d8VWm8Y" executionInfo={"status": "ok", "timestamp": 1633687355120, "user_tz": -330, "elapsed": 648, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="1447610f-3aad-4d43-f927-8d5ab224350f"
%%writefile data.dat
1:1:0.5 | user_age:25
2:0:0.5 | user_age:25
2:1:0.5 | user_age:55
2:1:0.5 | user_age:56
2:1:0.5 | user_age:55
2:1:0.5 | user_age:56
1:1:0.5 | user_age:27
2:0:0.5 | user_age:21
2:0:0.5 | user_age:23
2:0:0.5 | user_age:56
2:1:0.5 | user_age:55
2:1:0.5 | user_age:36
1:1:0.5 | user_age:25
2:0:0.5 | user_age:25
2:1:0.5 | user_age:55
2:1:0.5 | user_age:56
2:1:0.5 | user_age:55
2:1:0.5 | user_age:56
1:1:0.5 | user_age:27
2:0:0.5 | user_age:21
2:0:0.5 | user_age:23
2:0:0.5 | user_age:56
2:1:0.5 | user_age:55
2:1:0.5 | user_age:36
```

<!-- #region id="3-brQqWoWrMQ" -->
Then, train a policy incrementally. For the above data, with 2 possible actions and an IPS estimator, the command would be vw --cb 2 --cb_type ips -d data.dat (plus any additional options).
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="UCSQBLkGWxvS" executionInfo={"status": "ok", "timestamp": 1633687381999, "user_tz": -330, "elapsed": 699, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="b7b872c4-16d3-4815-8854-db6881d3bb8f"
!vw --cb 2 --cb_type ips -d data.dat 
```

<!-- #region id="pls7CM55WyHW" -->
The average loss here is the OPE estimate, calculated using progressive validation, with the cb_type estimator you specified. The same caveats as before apply: if the specified cb_type is biased, it is generally no recommended to use the average loss as an OPE estimate. Again, if you are unsure, we recommend using the default by omitting cb_type altogether.

To compare the OPE estimate of the candidate policy to the production policy, calculate the realised average cost in the data.dat file. In this case, it is 0.6667, and since our candidate policy’s 0.416667 is lower, we have found a set of hyperparameters estimated to perform better than our production online learner were it deployed at the time covered by the data.
<!-- #endregion -->

<!-- #region id="QQ9Jz3IjW-hb" -->
## Online scenario: policy evaluation with an incrementally trained VW policy, cb_adf-format data
If you have adf-format data, the same procedure as above applies – just change cb to cb_adf in the corresponding commands.
<!-- #endregion -->

<!-- #region id="tU2RgRFxXB_C" -->
## Legacy: policy evaluation with cb-format data, using a pre-trained policy
If your production policy produces bandit data in the standard cb format, and you already have a candidate policy even one trained outside VW, you can use the legacy --eval option to perform OPE. It is not recommended to use --eval if you are able to use any of the other methods described in this tutorial.

First, create a new file, e.g. eval.dat. Then, for each instance of your production policy’s bandit data, write it to eval.dat but prepend the line with the action your candidate policy would have chosen given the same context. For example, if your current instance is 1:2:0.5 | feature_a feature_b and your candidate policy chooses action 2 instead given the same context feature_a feature_b, write the line 2 1:2:0.5 | feature_a feature_b (note the space!).
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="Ee6PXNr5XQQe" executionInfo={"status": "ok", "timestamp": 1633687520047, "user_tz": -330, "elapsed": 600, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="a09a1d24-d43d-4189-c08b-e594389cf3fb"
%%writefile eval.dat
2 1:2:0.5 | feature_a feature_b
2 2:2:0.4 | feature_a feature_c
1 1:2:0.1 | feature_b feature_c
```

<!-- #region id="9IulJtDpXT1w" -->
In the toy example above, the candidate agreed with the production policy for the second and third instances, but disagreed on the first instance.

You are now ready to run policy evaluation using the command vw --cb <number_of_arms> --eval -d <dataset>. In our example, we have two possible actions, so the command is vw --cb 2 --eval -d eval.dat. This produced the following output (your results might differ based on VW version, or the seed):
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="RrI8WDDdXWVL" executionInfo={"status": "ok", "timestamp": 1633687564804, "user_tz": -330, "elapsed": 14, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="ea23e238-bd27-46bf-ac86-35fdc9a1e9c5"
!vw --cb 2 --eval -d eval.dat
```

<!-- #region id="ryK-zXbQXeu7" -->
Again, average loss is the OPE estimate, and can be compared against the production policy’s realised average loss to determine if your candidate policy is estimated to work better than the policy in production.

Note what happens if we try to run --eval with an estimator we know is biased, vw --cb 2 --eval -d eval.dat --cb_type dm. You will end up with an error, to prevent you from making a mistake:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="uXkJRXNuXlXB" executionInfo={"status": "ok", "timestamp": 1633687608024, "user_tz": -330, "elapsed": 780, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="8a81116b-57a9-4fb4-d435-f819cccd0cb8"
!vw --cb 2 --eval -d eval.dat --cb_type dm
```
