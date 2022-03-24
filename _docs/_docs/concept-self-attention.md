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
    language: python
    name: python3
---

<!-- #region id="0XQ6NsIuDtgr" -->
# Concept - Self-Attention
> Step-by-step guide to self-attention with illustrations and code

- toc: true
- badges: true
- comments: true
- categories: [Concept, SelfAttention, NLP]
- author: "<a href='https://towardsdatascience.com/illustrated-self-attention-2d627e33b20a'>Manuel Romero</a>"
- image:
<!-- #endregion -->

<!-- #region id="U76qWlrbOmx7" -->
![texto alternativo](https://miro.medium.com/max/1973/1*_92bnsMJy8Bl539G4v93yg.gif)
<!-- #endregion -->

<!-- #region id="wOkXKd60Q_Iu" -->
What do *BERT, RoBERTa, ALBERT, SpanBERT, DistilBERT, SesameBERT, SemBERT, MobileBERT, TinyBERT and CamemBERT* all have in common? And I’m not looking for the answer “BERT”.
Answer: **self-attention**. We are not only talking about architectures bearing the name “BERT’, but more correctly **Transformer-based architectures**. Transformer-based architectures, which are primarily used in modelling language understanding tasks, eschew the use of recurrence in neural network (RNNs) and instead trust entirely on self-attention mechanisms to draw global dependencies between inputs and outputs. But what’s the math behind this?
<!-- #endregion -->

<!-- #region id="yozzTBjBRbAA" -->
The main content of this kernel is to walk you through the mathematical operations involved in a self-attention module.
<!-- #endregion -->

<!-- #region id="atUYzU3TSD9z" -->
### Step 0. What is self-attention?

If you’re thinking if self-attention is similar to attention, then the answer is yes! They fundamentally share the same concept and many common mathematical operations.
A self-attention module takes in n inputs, and returns n outputs. What happens in this module? In layman’s terms, the self-attention mechanism allows the inputs to interact with each other (“self”) and find out who they should pay more attention to (“attention”). The outputs are aggregates of these interactions and attention scores.
<!-- #endregion -->

<!-- #region id="SDMmHAaSTE6P" -->
Following, we are going to explain and implement:
- Prepare inputs
- Initialise weights
- Derive key, query and value
- Calculate attention scores for Input 1
- Calculate softmax
- Multiply scores with values
- Sum weighted values to get Output 1
- Repeat steps 4–7 for Input 2 & Input 3
<!-- #endregion -->

```python id="u1UxPJlHBVmS"
import torch
```

<!-- #region id="ENdzUZqSBsiB" -->
### Step 1: Prepare inputs

For this tutorial, for the shake of simplicity, we start with 3 inputs, each with dimension 4.

![texto alternativo](https://miro.medium.com/max/1973/1*hmvdDXrxhJsGhOQClQdkBA.png)

<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 70} id="jKYrJsljBhnv" outputId="7b865905-2151-4a6a-a899-5439aa429af4"
x = [
  [1, 0, 1, 0], # Input 1
  [0, 2, 0, 2], # Input 2
  [1, 1, 1, 1]  # Input 3
 ]
x = torch.tensor(x, dtype=torch.float32)
x
```

<!-- #region id="DZ96EoE1Bvat" -->
### Step 2: Initialise weights

Every input must have three representations (see diagram below). These representations are called **key** (orange), **query** (red), and **value** (purple). For this example, let’s take that we want these representations to have a dimension of 3. Because every input has a dimension of 4, this means each set of the weights must have a shape of 4×3.

![texto del enlace](https://miro.medium.com/max/1975/1*VPvXYMGjv0kRuoYqgFvCag.gif)
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 284} id="jUTNr15JBkSG" outputId="baa4c379-6174-4990-8cd2-51191e904550"
w_key = [
  [0, 0, 1],
  [1, 1, 0],
  [0, 1, 0],
  [1, 1, 0]
]
w_query = [
  [1, 0, 1],
  [1, 0, 0],
  [0, 0, 1],
  [0, 1, 1]
]
w_value = [
  [0, 2, 0],
  [0, 3, 0],
  [1, 0, 3],
  [1, 1, 0]
]
w_key = torch.tensor(w_key, dtype=torch.float32)
w_query = torch.tensor(w_query, dtype=torch.float32)
w_value = torch.tensor(w_value, dtype=torch.float32)

print("Weights for key: \n", w_key)
print("Weights for query: \n", w_query)
print("Weights for value: \n", w_value)
```

<!-- #region id="8pr9XZF9X_Ed" -->
Note: *In a neural network setting, these weights are usually small numbers, initialised randomly using an appropriate random distribution like Gaussian, Xavier and Kaiming distributions.*
<!-- #endregion -->

<!-- #region id="UxGT5awVB1Xw" -->
### Step 3: Derive key, query and value

Now that we have the three sets of weights, let’s actually obtain the **key**, **query** and **value** representations for every input.
<!-- #endregion -->

<!-- #region id="VQwhDIi7aGXp" -->
Obtaining the keys:
```
               [0, 0, 1]
[1, 0, 1, 0]   [1, 1, 0]   [0, 1, 1]
[0, 2, 0, 2] x [0, 1, 0] = [4, 4, 0]
[1, 1, 1, 1]   [1, 1, 0]   [2, 3, 1]
```
![texto alternativo](https://miro.medium.com/max/1975/1*dr6NIaTfTxEWzxB2rc0JWg.gif)
<!-- #endregion -->

<!-- #region id="Qi0EblXTamFz" -->
Obtaining the values:
```
               [0, 2, 0]
[1, 0, 1, 0]   [0, 3, 0]   [1, 2, 3] 
[0, 2, 0, 2] x [1, 0, 3] = [2, 8, 0]
[1, 1, 1, 1]   [1, 1, 0]   [2, 6, 3]
```
![texto alternativo](https://miro.medium.com/max/1975/1*5kqW7yEwvcC0tjDOW3Ia-A.gif)

<!-- #endregion -->

<!-- #region id="GTp2izu1bLNq" -->
Obtaining the querys:
```
               [1, 0, 1]
[1, 0, 1, 0]   [1, 0, 0]   [1, 0, 2]
[0, 2, 0, 2] x [0, 0, 1] = [2, 2, 2]
[1, 1, 1, 1]   [0, 1, 1]   [2, 1, 3]
```
![texto alternativo](https://miro.medium.com/max/1975/1*wO_UqfkWkv3WmGQVHvrMJw.gif)
<!-- #endregion -->

<!-- #region id="qegb9M0KbnRK" -->
> Note: In practice, a bias vector may be added to the product of matrix multiplication.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 230} id="rv2NXynOB7oG" outputId="a2656b52-4b1d-4726-9d42-522f941b3126"
keys = x @ w_key
querys = x @ w_query
values = x @ w_value

print("Keys: \n", keys)
# tensor([[0., 1., 1.],
#         [4., 4., 0.],
#         [2., 3., 1.]])

print("Querys: \n", querys)
# tensor([[1., 0., 2.],
#         [2., 2., 2.],
#         [2., 1., 3.]])
print("Values: \n", values)
# tensor([[1., 2., 3.],
#         [2., 8., 0.],
#         [2., 6., 3.]])
```

<!-- #region id="3pmf0OQhCnD8" -->
### Step 4: Calculate attention scores
![texto alternativo](https://miro.medium.com/max/1973/1*u27nhUppoWYIGkRDmYFN2A.gif)

To obtain **attention scores**, we start off with taking a dot product between Input 1’s **query** (red) with **all keys** (orange), including itself. Since there are 3 key representations (because we have 3 inputs), we obtain 3 attention scores (blue).

```
            [0, 4, 2]
[1, 0, 2] x [1, 4, 3] = [2, 4, 4]
            [1, 0, 1]
```
Notice that we only use the query from Input 1. Later we’ll work on repeating this same step for the other querys.

Note: *The above operation is known as dot product attention, one of the several score functions. Other score functions include scaled dot product and additive/concat.*            
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 70} id="6GDhKEl0Cokw" outputId="c91356df-202c-4816-e98d-eefd1e1031d3"
attn_scores = querys @ keys.T
print(attn_scores)

# tensor([[ 2.,  4.,  4.],  # attention scores from Query 1
#         [ 4., 16., 12.],  # attention scores from Query 2
#         [ 4., 12., 10.]]) # attention scores from Query 3
```

<!-- #region id="bO3NmnbvCxpX" -->
### Step 5: Calculate softmax
![texto alternativo](https://miro.medium.com/max/1973/1*jf__2D8RNCzefwS0TP1Kyg.gif)

Take the **softmax** across these **attention scores** (blue).
```
softmax([2, 4, 4]) = [0.0, 0.5, 0.5]
```
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 124} id="PDNzdZHVC1ys" outputId="c528a7be-5c26-46a9-8fdb-1f2b029b6b93"
from torch.nn.functional import softmax

attn_scores_softmax = softmax(attn_scores, dim=-1)
print(attn_scores_softmax)
# tensor([[6.3379e-02, 4.6831e-01, 4.6831e-01],
#         [6.0337e-06, 9.8201e-01, 1.7986e-02],
#         [2.9539e-04, 8.8054e-01, 1.1917e-01]])

# For readability, approximate the above as follows
attn_scores_softmax = [
  [0.0, 0.5, 0.5],
  [0.0, 1.0, 0.0],
  [0.0, 0.9, 0.1]
]
attn_scores_softmax = torch.tensor(attn_scores_softmax)
print(attn_scores_softmax)
```

<!-- #region id="iBe71nseDBhb" -->
### Step 6: Multiply scores with values
![texto alternativo](https://miro.medium.com/max/1973/1*9cTaJGgXPbiJ4AOCc6QHyA.gif)

The softmaxed attention scores for each input (blue) is multiplied with its corresponding **value** (purple). This results in 3 alignment vectors (yellow). In this tutorial, we’ll refer to them as **weighted values**.
```
1: 0.0 * [1, 2, 3] = [0.0, 0.0, 0.0]
2: 0.5 * [2, 8, 0] = [1.0, 4.0, 0.0]
3: 0.5 * [2, 6, 3] = [1.0, 3.0, 1.5]
``` 
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 212} id="tNnx-Fx5DFDi" outputId="abc7a8ec-f964-483a-9bfb-2848f0e8e592"
weighted_values = values[:,None] * attn_scores_softmax.T[:,:,None]
print(weighted_values)
```

<!-- #region id="gU6w0U9ADQIc" -->
### Step 7: Sum weighted values
![texto alternativo](https://miro.medium.com/max/1973/1*1je5TwhVAwwnIeDFvww3ew.gif)

Take all the **weighted values** (yellow) and sum them element-wise:

```
  [0.0, 0.0, 0.0]
+ [1.0, 4.0, 0.0]
+ [1.0, 3.0, 1.5]
-----------------
= [2.0, 7.0, 1.5]
```

The resulting vector ```[2.0, 7.0, 1.5]``` (dark green) **is Output 1**, which is based on the **query representation from Input 1** interacting with all other keys, including itself.

<!-- #endregion -->

<!-- #region id="P3yNYDUEgAos" -->
### Step 8: Repeat for Input 2 & Input 3
![texto alternativo](https://miro.medium.com/max/1973/1*G8thyDVqeD8WHim_QzjvFg.gif)

Note: *The dimension of **query** and **key** must always be the same because of the dot product score function. However, the dimension of **value** may be different from **query** and **key**. The resulting output will consequently follow the dimension of **value**.*
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 70} id="R6excNSUDRRj" outputId="e5161fbe-05a5-41d2-da1e-5951ce8b1674"
outputs = weighted_values.sum(dim=0)
print(outputs)

# tensor([[2.0000, 7.0000, 1.5000],  # Output 1
#         [2.0000, 8.0000, 0.0000],  # Output 2
#         [2.0000, 7.8000, 0.3000]]) # Output 3
```

<!-- #region id="oavQirdbhAK7" -->
### Bonus: Tensorflow 2 implementation
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 35} id="575q0u_ahP-6" outputId="867a4e88-2223-41e4-ccd5-dbc47f580c83"
%tensorflow_version 2.x
import tensorflow as tf
```

```python colab={"base_uri": "https://localhost:8080/", "height": 88} id="0vjwwEKMhqmZ" outputId="56e5ed58-e100-434d-a8b2-00325bfc0d40"
x = [
  [1, 0, 1, 0], # Input 1
  [0, 2, 0, 2], # Input 2
  [1, 1, 1, 1]  # Input 3
 ]

x = tf.convert_to_tensor(x, dtype=tf.float32)
print(x)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 337} id="TN-pri7rhwJ-" outputId="aa8b1395-80a3-41e1-b544-beb06ce65a96"
w_key = [
  [0, 0, 1],
  [1, 1, 0],
  [0, 1, 0],
  [1, 1, 0]
]
w_query = [
  [1, 0, 1],
  [1, 0, 0],
  [0, 0, 1],
  [0, 1, 1]
]
w_value = [
  [0, 2, 0],
  [0, 3, 0],
  [1, 0, 3],
  [1, 1, 0]
]
w_key = tf.convert_to_tensor(w_key, dtype=tf.float32)
w_query = tf.convert_to_tensor(w_query, dtype=tf.float32)
w_value = tf.convert_to_tensor(w_value, dtype=tf.float32)
print("Weights for key: \n", w_key)
print("Weights for query: \n", w_query)
print("Weights for value: \n", w_value)

```

```python colab={"base_uri": "https://localhost:8080/", "height": 230} id="Jp2DP46Sh19r" outputId="5c1befaf-e096-454c-8402-885f049752e0"
keys = tf.matmul(x, w_key)
querys = tf.matmul(x, w_query)
values = tf.matmul(x, w_value)
print(keys)
print(querys)
print(values)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 88} id="tLJDo_bFigkm" outputId="b5d8e02d-9531-49c8-a587-7a6e0b6f884d"
attn_scores = tf.matmul(querys, keys, transpose_b=True)
print(attn_scores)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 159} id="8QY858MEiibV" outputId="2e84f48b-a4ed-4116-8655-21cbb9de8358"
attn_scores_softmax = tf.nn.softmax(attn_scores, axis=-1)
print(attn_scores_softmax)

# For readability, approximate the above as follows
attn_scores_softmax = [
  [0.0, 0.5, 0.5],
  [0.0, 1.0, 0.0],
  [0.0, 0.9, 0.1]
]
attn_scores_softmax = tf.convert_to_tensor(attn_scores_softmax)
print(attn_scores_softmax)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 230} id="TOJMfkFpi0KQ" outputId="8de18989-50d7-4534-cf5c-2711c66d17ce"
weighted_values = values[:,None] * tf.transpose(attn_scores_softmax)[:,:,None]
print(weighted_values)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 88} id="jan_cyy7i-s7" outputId="09b1406f-3a08-47e2-8dee-d4d6334ef1de"
outputs = tf.reduce_sum(weighted_values, axis=0)  # 6
print(outputs)
```
