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

<!-- #region id="x8qxp9NDVSdb" -->
Installation
<!-- #endregion -->

```python id="n-FdknIAVXOF"
!pip install -q pytorch-lightning
!pip install -q -U git+https://github.com/RecoHut-Projects/recohut.git@v0.0.11.post4
!apt-get -qq install tree
```

<!-- #region id="Y0OWOHWnDHp3" -->
ML1m Dataset
<!-- #endregion -->

```python id="0x34s2OcDZij" executionInfo={"status": "ok", "timestamp": 1642093018445, "user_tz": -330, "elapsed": 540, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
from recohut.datasets.movielens import ML1mDataset_v4
```

```python id="QQa-DJrvPTHm" executionInfo={"status": "ok", "timestamp": 1642093019968, "user_tz": -330, "elapsed": 721, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
ds = ML1mDataset_v4(data_dir='/content/data')
data = ds.load()
x_train = data["x_train"]
y_train = None
x_test = data["x_test"]
y_test = data["y_test"]
```

<!-- #region id="yRrsibBmDBcL" -->
Prod2vec Model
<!-- #endregion -->

```python id="Ir0I0XDFAl5_" executionInfo={"status": "ok", "timestamp": 1642093020649, "user_tz": -330, "elapsed": 684, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
from recohut.models.prod2vec import Prod2Vec_v2
```

```python colab={"base_uri": "https://localhost:8080/"} id="9fA-5ZZpG7R-" executionInfo={"status": "ok", "timestamp": 1642093033579, "user_tz": -330, "elapsed": 12934, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="e01fd1bd-f36e-4ac1-98c5-c4a0a8d845dd"
model = Prod2Vec_v2()
model.train(x_train)
y_pred = model.predict(x_test)
```

<!-- #region id="M5uqPycUTWix" -->
Behavioral Testing
<!-- #endregion -->

```python id="fVMQd846TZJi" executionInfo={"status": "ok", "timestamp": 1642093033580, "user_tz": -330, "elapsed": 11, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
from recohut.evaluation.behavioral import *
```

```python colab={"base_uri": "https://localhost:8080/", "height": 436} executionInfo={"status": "ok", "timestamp": 1642093037146, "user_tz": -330, "elapsed": 3576, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="a95eac81-1563-4b21-fe8d-e4e154a4f401" id="NSAAZQzrT4UM"
rec_list = SimilarItemEvaluation(
    x_train=x_train,
    y_train=y_train,
    x_test=x_test,
    y_test=y_test,
    y_preds=y_pred,
)

# invoke rec_list to run tests
rec_list(verbose=True)
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1642093093119, "user_tz": -330, "elapsed": 482, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="4a1331d5-b243-4664-87c7-fe3f91354c4b" id="2i4iFsgjT4UQ"
!tree -h --du -C ./SimilarItemEvaluation
```

```python colab={"base_uri": "https://localhost:8080/", "height": 419} executionInfo={"status": "ok", "timestamp": 1642093046659, "user_tz": -330, "elapsed": 9525, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="18d126a8-9da1-4949-902b-a65d6fe31c06" id="hKLNakaAT4UR"
rec_list = ComplementaryItemEvaluation(
    x_train=x_train,
    y_train=y_train,
    x_test=x_test,
    y_test=y_test,
    y_preds=y_pred,
    model=model
)

# invoke rec_list to run tests
rec_list(verbose=True)
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1642093097559, "user_tz": -330, "elapsed": 568, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="17ffb148-3724-40ef-da87-48adcaffe4d7" id="e7TgRgl7T4UR"
!tree -h --du -C ./ComplementaryItemEvaluation
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1642093050121, "user_tz": -330, "elapsed": 3472, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="83d2a91c-8d63-4315-cc47-6f5e10473d3a" id="wTeo-PlrT4US"
rec_list = SessionItemEvaluation(
    x_train=x_train,
    y_train=y_train,
    x_test=x_test,
    y_test=y_test,
    y_preds=y_pred,
    model=model
)

# invoke rec_list to run tests
rec_list(verbose=True)
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1642093104218, "user_tz": -330, "elapsed": 707, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="f0b8503f-b822-4e04-9695-68e3185392b6" id="wF_iZO_lT4US"
!tree -h --du -C ./SessionItemEvaluation
```
