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

<!-- #region id="pIHP196mEjmJ" -->
# Dask Incremental Wrapper

Some estimators can be trained incrementally – without seeing the entire dataset at once. Scikit-Learn provides the partial_fit API to stream batches of data to an estimator that can be fit in batches.

Normally, if you pass a Dask Array to an estimator expecting a NumPy array, the Dask Array will be converted to a single, large NumPy array. On a single machine, you’ll likely run out of RAM and crash the program. On a distributed cluster, all the workers will send their data to a single machine and crash it.

`dask_ml.wrappers.Incremental` provides a bridge between Dask and Scikit-Learn estimators supporting the partial_fit API. You wrap the underlying estimator in Incremental. Dask-ML will sequentially pass each block of a Dask Array to the underlying estimator’s partial_fit method.

`dask_ml.wrappers.Incremental` is a meta-estimator (an estimator that takes another estimator) that bridges scikit-learn estimators expecting NumPy arrays, and users with large Dask Arrays.

Each block of a Dask Array is fed to the underlying estimator’s partial_fit method. The training is entirely sequential, so you won’t notice massive training time speedups from parallelism. In a distributed environment, you should notice some speedup from avoiding extra IO, and the fact that models are typically much smaller than data, and so faster to move between machines.
<!-- #endregion -->

```python id="Ey0Jpw-8lZwT"
!pip install -q -U dask-ml distributed
```

```python id="pTNPnXdgE4sP" colab={"base_uri": "https://localhost:8080/", "height": 197} executionInfo={"status": "ok", "timestamp": 1635251750153, "user_tz": -330, "elapsed": 1938, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="e06f0da2-ea5c-4bee-eeef-ea242a7b7828"
from dask_ml.datasets import make_classification
from dask_ml.wrappers import Incremental
from sklearn.linear_model import SGDClassifier

X, y = make_classification(chunks=25)
X
```

```python colab={"base_uri": "https://localhost:8080/"} id="Lzfx3kg9lWRf" executionInfo={"status": "ok", "timestamp": 1635251781117, "user_tz": -330, "elapsed": 689, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="0c2f36d5-4622-4ea8-e22d-bcb95238dc2a"
estimator = SGDClassifier(random_state=10, max_iter=100)

clf = Incremental(estimator)

clf.fit(X, y, classes=[0, 1])
```

<!-- #region id="rwDTTeelmfiR" -->
In this example, we make a (small) random Dask Array. It has 100 samples, broken in the 4 blocks of 25 samples each. The chunking is only along the first axis (the samples). There is no chunking along the features.

You instantiate the underlying estimator as usual. It really is just a scikit-learn compatible estimator, and will be trained normally via its partial_fit.

Notice that we call the regular .fit method, not partial_fit for training. Dask-ML takes care of passing each block to the underlying estimator for you.

Just like `sklearn.linear_model.SGDClassifier.partial_fit()`, we need to pass the classes argument to fit. In general, any argument that is required for the underlying estimators partial_fit becomes required for the wrapped fit.


<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="ndGWcfQ0mqRY" executionInfo={"status": "ok", "timestamp": 1635251826333, "user_tz": -330, "elapsed": 721, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="18cc4837-0317-4797-d55a-d7b911c31cfe"
clf.score(X, y)
```

```python colab={"base_uri": "https://localhost:8080/"} id="oTHNAKk4mqet" executionInfo={"status": "ok", "timestamp": 1635251831767, "user_tz": -330, "elapsed": 525, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="dbdf0f83-706a-4e6c-c48a-b6dcd463a980"
clf.coef_
```

```python colab={"base_uri": "https://localhost:8080/"} id="b7O3Nm-Umr6s" executionInfo={"status": "ok", "timestamp": 1635251841924, "user_tz": -330, "elapsed": 539, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="5536aa42-81f7-4583-e1bb-4446816b1aa1"
clf.estimator_
```
