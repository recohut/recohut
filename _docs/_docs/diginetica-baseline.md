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

# Diginetica Baseline Recommender

```python executionInfo={"elapsed": 728, "status": "ok", "timestamp": 1631276623063, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}, "user_tz": -330} id="aVC5awU1jAzf"
import os
project_name = "chef-session"; branch = "main"; account = "sparsh-ai"
project_path = os.path.join('/content', project_name)
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 13, "status": "ok", "timestamp": 1631276623065, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}, "user_tz": -330} id="8WqnA1xDjAzj" outputId="abc73563-fa3c-4af1-daf9-27783f9794ed"
if not os.path.exists(project_path):
    !pip install -U -q dvc dvc[gdrive]
    !cp -r /content/drive/MyDrive/git_credentials/. ~
    path = "/content/" + project_name; 
    !mkdir "{path}"
    %cd "{path}"
    !git init
    !git remote add origin https://github.com/"{account}"/"{project_name}".git
    !git pull origin "{branch}"
    !git checkout "{branch}"
else:
    %cd "{project_path}"
```

```python id="ZoaSr0CxjAzk"
!git status
```

```python id="fBsuxRGMjAzl"
!git add . && git commit -m 'commit' && git push origin "{branch}"
```

```python id="jo4GmFulkgAJ"
# This is sample baseline for CIKM Personalization Cup 2016
# by Alexander Laktionov & Vladislav Grozin

import numpy as np
import pandas as pd
import datetime

start_time = datetime.datetime.now()
print("Running baseline. Now it's", start_time.isoformat())

# Loading queries (assuming data placed in <dataset-train/>
queries = pd.read_csv('dataset-train/train-queries.csv', sep=';')[['queryId', 'items', 'is.test']]
print('Total queries', len(queries))

# Leaving only test queries (the ones which items we have to sort)
queries = queries[queries['is.test'] == True][['queryId', 'items']]
print('Test queries', len(queries))
queries.reset_index(inplace=True)
queries.drop(['index'], axis=1, inplace=True)

# Loading item views; taking itemId column
item_views = pd.read_csv('dataset-train/train-item-views.csv', sep=';')[['itemId']]
print('Item views', len(item_views))

# Loading clicks; taking itemId column
clicks = pd.read_csv('dataset-train/train-clicks.csv', sep=';')[['itemId']]
print('Clicks', len(clicks))

# Loading purchases; taking itemId column
purchases = pd.read_csv('dataset-train/train-purchases.csv', sep=';')[['itemId']]
print('Purchases', len(purchases))

# Calculating popularity as [Amount of views] * 1 + Amount of clicks * 2 + [Amount of purchases] * 3
print('Scoring popularity for each item ...')
prod_pop = {}
for cost, container in enumerate([item_views, clicks, purchases]):
    for prod in container.values:
        product = str(prod[0])
        if product not in prod_pop:
            prod_pop[product] = cost
        else:
            prod_pop[product] += cost

print('Popularity scored for', len(prod_pop), 'products')

# For each query:
#   parse items (comma-separated values in last column)
#   sort them by score;
#   write them to the submission file.
# This is longest part; it usually takes around 5 minutes.
print('Sorting items per query by popularity...')

answers = []
step = int(len(queries) / 20)

with open('submission.txt', 'w+') as submission:
    for i, q in enumerate(queries.values):

        # Fancy progressbar
        if i % step == 0:
            print(5 * i / step, '%...')

        # Splitting last column which contains comma-separated items
        items = q[-1].split(',')
        # Getting scores for each item. Also, inverting scores here, so we can use argsort
        items_scores = list(map(lambda x: -prod_pop.get(x, 0), items))
        # Sorting items using items_scores order permutation
        sorted_items = np.array(items)[np.array(items_scores).argsort()]
        # Squashing items together
        s = ','.join(sorted_items)
        # and writing them to submission
        submission.write(str(q[0]) + " " + s + "\n")

end_time = datetime.datetime.now()
print("Done. Now it's ", end_time.isoformat())
print("Calculated baseline in ", (end_time - start_time).seconds, " seconds")

```
