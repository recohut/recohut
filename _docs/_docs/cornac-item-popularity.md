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

<!-- #region id="xyM7qLhOvaBn" -->
# Popularity based movie recommender
> A simple popular movie recommender and eda on movielens-100k dataset with the help of cornac library

- toc: true
- badges: true
- comments: true
- categories: [Movie, Cornac]
- author: "<a href='https://nbviewer.jupyter.org/github/PreferredAI/tutorials/tree/master/recommender-systems/'>Cornac</a>"
- image:
<!-- #endregion -->

<!-- #region id="DIyIbFovlp07" -->
## Setup
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="JaastgN_i3ry" outputId="bf29e875-2bda-4cad-d4cd-cb96860b7d3c"
!pip install -q cornac==1.4.1
```

```python colab={"base_uri": "https://localhost:8080/"} id="zj1_hePrjSqk" outputId="6eced395-0fd0-42cc-b87d-e8d5f025bb6e"
import os
import sys

import cornac
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, sparse

%matplotlib inline

print(f"System version: {sys.version}")
print(f"Cornac version: {cornac.__version__}")
```

<!-- #region id="jU6acUDBm3mz" -->
## Movielens Data
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 272} id="Nb26UGqJjaZ1" outputId="1d7605fe-e7f6-4021-852b-c34e9811a2ca"
data = cornac.datasets.movielens.load_feedback(variant="100K")
df = pd.DataFrame(data, columns=["user_id", "item_id", "rating"])
df.head()
```

<!-- #region id="Myjwm4yOlr3U" -->
## EDA
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="prPyDl_akPBd" outputId="24b7ff94-e26e-4836-ece5-542f04b08e95"
n_users = df.user_id.nunique()
n_items = df.item_id.nunique()
n_ratings = len(df)
rating_matrix_size = n_users * n_items
sparsity = 1 - n_ratings / rating_matrix_size

print(f"Number of users: {n_users}")
print(f"Number of items: {n_items}")
print(f"Number of available ratings: {n_ratings}")
print(f"Number of all possible ratings: {rating_matrix_size}")
print("-" * 40)
print(f"SPARSITY: {sparsity * 100.0:.2f}%")
```

<!-- #region id="PFB2XowglBUr" -->
> Note: For this MovieLens dataset, the data has been prepared in such a way that each user has at least 20 ratings. As a result, it's relatively dense as compared to many other recommendation datasets that are usually much sparser (often 99% or more).
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="J0h3BpYskbe2" outputId="dc79c715-3386-4db8-d525-6e06e6aca577"
df.rating.describe()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 351} id="PEFekQEhkgJ3" outputId="19fe5a42-3304-4d44-ef29-f6d7ff669b9d"
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))
sns.countplot(x="rating", data=df, palette="ch:.25", ax=axes[0])
sns.boxplot(x="rating", data=df, palette="ch:.25", ax=axes[1])
```

```python colab={"base_uri": "https://localhost:8080/", "height": 350} id="DYWZ43fSkvNj" outputId="70602415-2e27-4310-b332-2374f773d43e"
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))

item_rate_count = df.groupby("item_id")["user_id"].nunique().sort_values(ascending=False)

axes[0].bar(x=item_rate_count.index, height=item_rate_count.values, width=1.0, align="edge")
axes[0].set_xticks([])
axes[0].set(title="long tail of rating frequency", 
            xlabel="item ordered by decreasing frequency", 
            ylabel="#ratings")

count = item_rate_count.value_counts()
sns.scatterplot(x=np.log(count.index), y=np.log(count.values), ax=axes[1])
axes[1].set(title="log-log plot", xlabel="#ratings (log scale)", ylabel="#items (log scale)");
```

<!-- #region id="W5R5FnFBltdk" -->
## Recommendation Based on Item Popularity
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="QtLW0z5flccx" outputId="de49d19f-f802-4ec5-8339-ba422d483bbe"
# 5-fold cross validation
eval_method = cornac.eval_methods.CrossValidation(data, n_folds=5, seed=42)

# recommender system based on item popularity
most_pop = cornac.models.MostPop()

# recall@20 metric
rec_20 = cornac.metrics.Recall(k=20)

# put everything together into an experiment
cornac.Experiment(eval_method=eval_method, models=[most_pop], metrics=[rec_20]).run()
```
