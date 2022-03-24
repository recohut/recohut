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

<!-- #region id="ACNt2GFq6ItJ" -->
# MovieLens Data Splitting
<!-- #endregion -->

<!-- #region id="Hx4U0cpYvhih" -->
### Data Split
<!-- #endregion -->

```python id="9hZnrRahvp2b"
import time
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
```

```python id="GwEypSUXwO6B"
def _check_and_convert_ratio(test_size, multi_ratios):
    if not test_size and not multi_ratios:
        raise ValueError("must provide either 'test_size' or 'multi_ratios'")

    elif test_size is not None:
        assert isinstance(test_size, float), "test_size must be float value"
        assert 0.0 < test_size < 1.0, "test_size must be in (0.0, 1.0)"
        ratios = [1 - test_size, test_size]
        return ratios, 2

    elif isinstance(multi_ratios, (list, tuple)):
        assert len(multi_ratios) > 1, (
            "multi_ratios must at least have two elements")
        assert all([r > 0.0 for r in multi_ratios]), (
            "ratios should be positive values")
        if math.fsum(multi_ratios) != 1.0:
            ratios = [r / math.fsum(multi_ratios) for r in multi_ratios]
        else:
            ratios = multi_ratios
        return ratios, len(ratios)

    else:
        raise ValueError("multi_ratios should be list or tuple")


def _filter_unknown_user_item(data_list):
    train_data = data_list[0]
    unique_values = dict(user=set(train_data.user.tolist()),
                         item=set(train_data.item.tolist()))

    split_data_all = [train_data]
    for i, test_data in enumerate(data_list[1:], start=1):
        # print(f"Non_train_data {i} size before filtering: {len(test_data)}")
        out_of_bounds_row_indices = set()
        for col in ["user", "item"]:
            for j, val in enumerate(test_data[col]):
                if val not in unique_values[col]:
                    out_of_bounds_row_indices.add(j)

        mask = np.arange(len(test_data))
        test_data_clean = test_data[~np.isin(
            mask, list(out_of_bounds_row_indices))]
        split_data_all.append(test_data_clean)
        # print(f"Non_train_data {i} size after filtering: "
        #      f"{len(test_data_clean)}")
    return split_data_all


def _pad_unknown_user_item(data_list):
    train_data, test_data = data_list
    n_users = train_data.user.nunique()
    n_items = train_data.item.nunique()
    unique_users = set(train_data.user.tolist())
    unique_items = set(train_data.item.tolist())

    split_data_all = [train_data]
    for i, test_data in enumerate(data_list[1:], start=1):
        test_data.loc[~test_data.user.isin(unique_users), "user"] = n_users
        test_data.loc[~test_data.item.isin(unique_items), "item"] = n_items
        split_data_all.append(test_data)
    return split_data_all


def _groupby_user(user_indices, order):
    sort_kind = "mergesort" if order else "quicksort"
    users, user_position, user_counts = np.unique(user_indices,
                                                  return_inverse=True,
                                                  return_counts=True)
    user_split_indices = np.split(np.argsort(user_position, kind=sort_kind),
                                  np.cumsum(user_counts)[:-1])
    return user_split_indices
```

```python id="SeQjDRWjwBqP"
def random_split(data, test_size=None, multi_ratios=None, shuffle=True,
                 filter_unknown=True, pad_unknown=False, seed=42):
    ratios, n_splits = _check_and_convert_ratio(test_size, multi_ratios)
    if not isinstance(ratios, list):
        ratios = list(ratios)

    # if we want to split data in multiple folds,
    # then iteratively split data based on modified ratios
    train_data = data.copy()
    split_data_all = []
    for i in range(n_splits - 1):
        size = ratios.pop(-1)
        ratios = [r / math.fsum(ratios) for r in ratios]
        train_data, split_data = train_test_split(train_data,
                                                  test_size=size,
                                                  shuffle=shuffle,
                                                  random_state=seed)
        split_data_all.insert(0, split_data)
    split_data_all.insert(0, train_data)  # insert final fold of data

    if filter_unknown:
        split_data_all = _filter_unknown_user_item(split_data_all)
    elif pad_unknown:
        split_data_all = _pad_unknown_user_item(split_data_all)
    return split_data_all


def split_by_ratio(data, order=True, shuffle=False, test_size=None,
                   multi_ratios=None, filter_unknown=True, pad_unknown=False,
                   seed=42):
    np.random.seed(seed)
    assert ("user" in data.columns), "data must contains user column"
    ratios, n_splits = _check_and_convert_ratio(test_size, multi_ratios)

    n_users = data.user.nunique()
    user_indices = data.user.to_numpy()
    user_split_indices = _groupby_user(user_indices, order)

    cum_ratios = np.cumsum(ratios).tolist()[:-1]
    split_indices_all = [[] for _ in range(n_splits)]
    for u in range(n_users):
        u_data = user_split_indices[u]
        u_data_len = len(u_data)
        if u_data_len <= 3:   # keep items of rare users in trainset
            split_indices_all[0].extend(u_data)
        else:
            u_split_data = np.split(u_data, [
                round(cum * u_data_len) for cum in cum_ratios
            ])
            for i in range(n_splits):
                split_indices_all[i].extend(list(u_split_data[i]))

    if shuffle:
        split_data_all = tuple(
            np.random.permutation(data[idx]) for idx in split_indices_all)
    else:
        split_data_all = list(data.iloc[idx] for idx in split_indices_all)

    if filter_unknown:
        split_data_all = _filter_unknown_user_item(split_data_all)
    elif pad_unknown:
        split_data_all = _pad_unknown_user_item(split_data_all)
    return split_data_all


def split_by_num(data, order=True, shuffle=False, test_size=1,
                 filter_unknown=True, pad_unknown=False, seed=42):
    np.random.seed(seed)
    assert ("user" in data.columns), "data must contains user column"
    assert isinstance(test_size, int), "test_size must be int value"
    assert 0 < test_size < len(data), "test_size must be in (0, len(data))"

    n_users = data.user.nunique()
    user_indices = data.user.to_numpy()
    user_split_indices = _groupby_user(user_indices, order)

    train_indices = []
    test_indices = []
    for u in range(n_users):
        u_data = user_split_indices[u]
        u_data_len = len(u_data)
        if u_data_len <= 3:   # keep items of rare users in trainset
            train_indices.extend(u_data)
        elif u_data_len <= test_size:
            train_indices.extend(u_data[:-1])
            test_indices.extend(u_data[-1:])
        else:
            k = test_size
            train_indices.extend(u_data[:(u_data_len-k)])
            test_indices.extend(u_data[-k:])

    if shuffle:
        train_indices = np.random.permutation(train_indices)
        test_indices = np.random.permutation(test_indices)

    split_data_all = (data.iloc[train_indices], data.iloc[test_indices])
    if filter_unknown:
        split_data_all = _filter_unknown_user_item(split_data_all)
    elif pad_unknown:
        split_data_all = _pad_unknown_user_item(split_data_all)
    return split_data_all


def split_by_ratio_chrono(data, order=True, shuffle=False, test_size=None,
                          multi_ratios=None, seed=42):
    assert all([
        "user" in data.columns,
        "time" in data.columns
    ]), "data must contains user and time column"

    data.sort_values(by=["time"], inplace=True)
    data.reset_index(drop=True, inplace=True)
    return split_by_ratio(**locals())


def split_by_num_chrono(data, order=True, shuffle=False, test_size=1, seed=42):
    assert all([
        "user" in data.columns,
        "time" in data.columns
    ]), "data must contains user and time column"

    data.sort_values(by=["time"], inplace=True)
    data.reset_index(drop=True, inplace=True)
    return split_by_num(**locals())
```

```python colab={"base_uri": "https://localhost:8080/", "height": 886} id="_ANES5MHySvI" executionInfo={"status": "ok", "timestamp": 1630675873079, "user_tz": -330, "elapsed": 919, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="1a6b0dbf-4973-473e-a325-3c1e41a4d6dd"
data = pd.read_csv('sample_movielens_merged.csv')
data
```

```python colab={"base_uri": "https://localhost:8080/"} id="NhRMOUyKydVz" executionInfo={"status": "ok", "timestamp": 1630675845213, "user_tz": -330, "elapsed": 723, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="978b98a0-88d6-42f8-b2f8-1710916e071e"
data.info()
```

```python colab={"base_uri": "https://localhost:8080/"} id="JFq9ehHuymz4" executionInfo={"status": "ok", "timestamp": 1630676818903, "user_tz": -330, "elapsed": 727, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="3f2178df-e54d-447b-f754-b178bcb64869"
train_data, eval_data, test_data = random_split(data, multi_ratios=[0.5, 0.1, 0.1], seed=42,
                                                filter_unknown=False)

train_data.shape, eval_data.shape, test_data.shape
```

```python colab={"base_uri": "https://localhost:8080/", "height": 173} id="Qp7UwylEy4v6" executionInfo={"status": "ok", "timestamp": 1630676823222, "user_tz": -330, "elapsed": 13, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="9df2c1ee-401a-4535-b945-8417c6cf7bf1"
test_data.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="tnAklPyo6ha2" executionInfo={"status": "ok", "timestamp": 1630677356407, "user_tz": -330, "elapsed": 1731, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="cf8774b7-b70a-4912-b122-bcda12f51f6b"
train_data, eval_data, test_data = random_split(data,
                                                multi_ratios=[0.8, 0.1, 0.1],
                                                seed=42,
                                                filter_unknown=True,
                                                pad_unknown=False)

train_data.shape, eval_data.shape, test_data.shape
```

```python colab={"base_uri": "https://localhost:8080/", "height": 49} id="wq2Q5IRq6hbM" executionInfo={"status": "ok", "timestamp": 1630677358093, "user_tz": -330, "elapsed": 1693, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="e40fe4dd-c2d3-4576-feb3-cdd4f6815620"
eval_data.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="JFTnd-bHzMaF" executionInfo={"status": "ok", "timestamp": 1630674989174, "user_tz": -330, "elapsed": 725, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="18727508-ec39-4c12-885b-5532c051c87a"
train_data, eval_data = split_by_ratio(data, test_size=0.2)

train_data.shape, eval_data.shape
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="S82MAfhVzvRz" executionInfo={"status": "ok", "timestamp": 1630675079062, "user_tz": -330, "elapsed": 507, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="eb0c8abb-1ff5-460e-aa1d-28e3a97d2307"
eval_data.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="kQddKIvbzh8k" executionInfo={"status": "ok", "timestamp": 1630675027448, "user_tz": -330, "elapsed": 686, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="b1c8e59f-6f37-4746-b086-b680ae326aee"
train_data, eval_data = split_by_num(data, test_size=1)

train_data.shape, eval_data.shape
```

```python colab={"base_uri": "https://localhost:8080/"} id="n01ZszMgzmpP" executionInfo={"status": "ok", "timestamp": 1630675046442, "user_tz": -330, "elapsed": 515, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="1019c798-da35-4782-e511-bcc5d74a425c"
train_data, eval_data = split_by_ratio_chrono(data, test_size=0.2)

train_data.shape, eval_data.shape
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="vJieaGB10En2" executionInfo={"status": "ok", "timestamp": 1630675169223, "user_tz": -330, "elapsed": 698, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="128d265c-a156-4440-9346-28a14f8ee5be"
eval_data.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="a16L78A9zsC6" executionInfo={"status": "ok", "timestamp": 1630675067401, "user_tz": -330, "elapsed": 824, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="f07810ba-4b43-45ee-e555-2aed83654a95"
train_data, eval_data = split_by_num_chrono(data, test_size=1)

train_data.shape, eval_data.shape
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="jl2LGa51vjBZ" executionInfo={"status": "ok", "timestamp": 1630675191287, "user_tz": -330, "elapsed": 497, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="19d1f059-d07d-405d-f1a4-a3be71ffa551"
eval_data.head()
```
