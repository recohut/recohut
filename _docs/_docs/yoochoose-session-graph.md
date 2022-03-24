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

<!-- #region id="yG-gp8EAaB85" -->
# Conversion of Yoochoose Sessions into Graphs
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="1jsUW40FeO6X" executionInfo={"status": "ok", "timestamp": 1637816560387, "user_tz": -330, "elapsed": 3491, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="3337a8a1-76ab-44af-f098-48b4958af43d"
!wget -q --show-progress https://github.com/sparsh-ai/stanza/raw/S969796/datasets/yoochoose1_64/raw/train.txt
!wget -q --show-progress https://github.com/sparsh-ai/stanza/raw/S969796/datasets/yoochoose1_64/raw/test.txt
```

```python id="S1payE7KeXH1"
import pickle

train_data = pickle.load(open('train.txt', 'rb'))
```

```python colab={"base_uri": "https://localhost:8080/"} id="cyVzkq0leZ1P" executionInfo={"status": "ok", "timestamp": 1637816634911, "user_tz": -330, "elapsed": 504, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="1d112100-3eef-47a9-d208-676a28e56206"
type(train_data)
```

```python colab={"base_uri": "https://localhost:8080/"} id="xIWwTak6hX-D" executionInfo={"status": "ok", "timestamp": 1637817377156, "user_tz": -330, "elapsed": 509, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="99a1d563-360d-4dd9-b376-7aefced62d13"
train_data[0][:10]
```

```python colab={"base_uri": "https://localhost:8080/"} id="BrX8iHs0hlN4" executionInfo={"status": "ok", "timestamp": 1637817411947, "user_tz": -330, "elapsed": 698, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="3fe16ab9-ea83-48fc-c47f-6172afae60d1"
train_data[1][:10]
```

```python colab={"base_uri": "https://localhost:8080/"} id="8fXhhodkfFKb" executionInfo={"status": "ok", "timestamp": 1637817497288, "user_tz": -330, "elapsed": 13, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="a3260f4d-7277-4d01-c796-7a3485fda4ee"
len(train_data[0])
```

```python colab={"base_uri": "https://localhost:8080/", "height": 34} id="jmYxWhz5jAFm" executionInfo={"status": "ok", "timestamp": 1637817826792, "user_tz": -330, "elapsed": 526, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="af93427f-4609-4e4f-c1b4-8eb32925a584"
us_lens = [len(upois) for upois in train_data[0]] # user-session lengths
us_lens[:10]
```

```python colab={"base_uri": "https://localhost:8080/", "height": 34} id="QsZGf0gAiAKi" executionInfo={"status": "ok", "timestamp": 1637817968162, "user_tz": -330, "elapsed": 461, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="df89e1c1-9a96-4407-9d8a-3317824cdfad"
len_max = max(us_lens) # maximum session length
len_max
```

```python colab={"base_uri": "https://localhost:8080/", "height": 503} id="m7RHTMmAjUhF" executionInfo={"status": "ok", "timestamp": 1637818027777, "user_tz": -330, "elapsed": 484, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="95edd41b-f5c5-4573-b82f-819415a00c4c"
us_pois = [upois + [0] * (len_max - le) for upois, le in zip(train_data[0], us_lens)]
print(*us_pois[:10])
```

```python colab={"base_uri": "https://localhost:8080/", "height": 503} id="e1tGiKdukyzP" executionInfo={"status": "ok", "timestamp": 1637818298440, "user_tz": -330, "elapsed": 3412, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="c750dab4-b57e-4ed0-fca7-1ebd19bc6b44"
us_msks = [[1] * le + [0] * (len_max - le) for le in us_lens] # user session mask
print(*us_msks[:10])
```

```python colab={"base_uri": "https://localhost:8080/", "height": 17} id="tP9Zw64eiAH1" executionInfo={"status": "ok", "timestamp": 1637819520826, "user_tz": -330, "elapsed": 9122, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="cd0a38ed-980b-4c66-d9c9-10cc8e97f48e"
inputs = np.asarray(us_pois)
mask = np.asarray(us_msks)
len_max = len_max
targets = np.asarray(train_data[1])
length = len(inputs)
```

<!-- #region id="qHncDQNLl7qD" -->
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 34} id="cTVpjEFeqomj" executionInfo={"status": "ok", "timestamp": 1637820454433, "user_tz": -330, "elapsed": 478, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="30de193d-f27e-48c5-d46b-24bc826189b7"
i = 2
j = 11

_inputs, _mask, _targets = inputs[i:j], mask[i:j], targets[i:j]

items, n_node, A, alias_inputs = [], [], [], []

for u_input in _inputs:
    n_node.append(len(np.unique(u_input)))

print(*n_node)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 17} id="4dTnV3qNqEWW" executionInfo={"status": "ok", "timestamp": 1637820456310, "user_tz": -330, "elapsed": 14, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="e5ecdf3c-5776-4035-da54-e4445d3be542"
max_n_node = np.max(n_node)

for u_input in _inputs:
    node = np.unique(u_input)
    items.append(node.tolist() + (max_n_node - len(node)) * [0])
    u_A = np.zeros((max_n_node, max_n_node))
    for i in np.arange(len(u_input) - 1):
        if u_input[i + 1] == 0:
            break
        u = np.where(node == u_input[i])[0][0]
        v = np.where(node == u_input[i + 1])[0][0]
        u_A[u][v] = 1
    u_sum_in = np.sum(u_A, 0)
    u_sum_in[np.where(u_sum_in == 0)] = 1
    u_A_in = np.divide(u_A, u_sum_in)
    u_sum_out = np.sum(u_A, 1)
    u_sum_out[np.where(u_sum_out == 0)] = 1
    u_A_out = np.divide(u_A.transpose(), u_sum_out)
    u_A = np.concatenate([u_A_in, u_A_out]).transpose()
    A.append(u_A)
    alias_inputs.append([np.where(node == i)[0][0] for i in u_input])
```

```python colab={"base_uri": "https://localhost:8080/", "height": 451} id="KDnOq7Y4pOA1" executionInfo={"status": "ok", "timestamp": 1637820456311, "user_tz": -330, "elapsed": 11, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="513acf4f-01e4-4ecb-9689-5dbe8b178ea4"
print(*alias_inputs)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 173} id="TDI0buIXrzIs" executionInfo={"status": "ok", "timestamp": 1637820456311, "user_tz": -330, "elapsed": 10, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="f6fe83bd-6f80-48ed-a574-a6cc46722ac2"
items
```

```python colab={"base_uri": "https://localhost:8080/", "height": 34} id="SjL0UzKqsEJs" executionInfo={"status": "ok", "timestamp": 1637820460665, "user_tz": -330, "elapsed": 7, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="d87cc416-a129-4d33-fb1b-f6c9541a5524"
np.array(A).shape
```

```python colab={"base_uri": "https://localhost:8080/", "height": 642} id="odO6ccCNsE57" executionInfo={"status": "ok", "timestamp": 1637820526472, "user_tz": -330, "elapsed": 745, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="b1f516e8-90b8-4068-f1e8-5dda8a953f89"
A
```
