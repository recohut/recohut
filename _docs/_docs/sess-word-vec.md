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

<!-- #region id="0M2nRjhs2bhj" -->
# Training Session-based Product Recommender using Word2vec on Retail data
<!-- #endregion -->

<!-- #region id="eZrZz_8u2_yL" -->
## Executive summary

| | |
| --- | --- |
| Problem | A key trend over the past few years has been session-based recommendation algorithms that provide recommendations solely based on a user’s interactions in an ongoing session, and which do not require the existence of user profiles or their entire historical preferences. This tutorial explores a simple, yet powerful, NLP-based approach (word2vec) to recommend a next item to a user. While NLP-based approaches are generally employed for linguistic tasks, can we exploit them to learn the structure induced by a user’s behavior or an item’s nature. |
| Prblm Stmnt. | Given a series of events (e.g. user's browsing history), the task is to predict the next event. |
| Solution | We will implement a simple, yet powerful, NLP-based approach (word2vec) to recommend a next item to a user. While NLP-based approaches are generally employed for linguistic tasks, here we exploit them to learn the structure induced by a user’s behavior or an item’s nature. |
| Dataset | Retail Session Data |
| Preprocessing | There are some rows with missing information, so we'll filter those out. Since we want to define customer sessions, we'll use group by CustomerID field and filter out any customer entries that have fewer than three purchased items. We used withholding the last element of the session for sessionization.  |
| Metrics | Recall, MRR |
| Models | Prod2vec |
| Cluster | Python 3.6+, RayTune |
| Tags | `SessionRecommender`, `Word2vec`, `HyperParamOptimization`, `RayTune` |
| Credits | Cloudera Fast Forward Labs |
<!-- #endregion -->

<!-- #region id="PVgIuKuy6rYX" -->
## Process flow

![](https://github.com/RecoHut-Stanzas/S810511/raw/main/images/S810511%20%7C%20Retail%20Session-based%20recommendations.drawio.svg)
<!-- #endregion -->

<!-- #region id="DKJ0AqIzuXJ5" -->
A user’s choice of items not only depends on long-term historical preference, but also on short-term and more recent preferences. Choices almost always have time-sensitive context; for instance, “recently viewed” or “recently purchased” items may actually be more relevant than others. These short-term preferences are embedded in the user’s most recent interactions, but may account for only a small proportion of historical interactions. In addition, a user’s preference towards certain items can tend to be dynamic rather than static; it often evolves over time.

A key trend over the past few years has been session-based recommendation algorithms that provide recommendations solely based on a user’s interactions in an ongoing session, and which do not require the existence of user profiles or their entire historical preferences.

We will implement a simple, yet powerful, NLP-based approach (word2vec) to recommend a next item to a user. While NLP-based approaches are generally employed for linguistic tasks, here we exploit them to learn the structure induced by a user’s behavior or an item’s nature.
<!-- #endregion -->

<!-- #region id="XIY8wbf-VFyT" -->
## Install/import libraries
<!-- #endregion -->

```python id="5pz-XFyoj5is"
!sudo apt-get install -y ray
!pip install ray
!pip install ray[default]
!pip install ray[tune]
```

```python id="zV-QhI_1UkjA"
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from numpy.random import default_rng
import collections
import itertools
from copy import deepcopy 

from gensim.models.word2vec import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from ray import tune

import os
import argparse
import ray
import time

MODEL_DIR = "/content"
```

```python id="XDOxi0wabyQs"
plt.style.use("seaborn-white")
cldr_colors = ['#00b6b5', '#f7955b','#6c8cc7', '#828282']#
cldr_green = '#a4d65d'
color_palette = "viridis"

rng = default_rng(123)

ECOMM_PATH = "/content"
ECOMM_FILENAME = "OnlineRetail.csv"

%load_ext tensorboard
```

<!-- #region id="4UPgoIvlU9z2" -->
## Load data
<!-- #endregion -->

<!-- #region id="6RToYkyYuUsw" -->
We chose an open domain e-commerce dataset from a UK-based online boutique selling specialty gifts. This dataset was collected between 12/01/2010 and 12/09/2011 and contains purchase histories for 4,372 customers and 3,684 unique products. These purchase histories record transactions for each customer and detail the items that were purchased in each transaction. This is a bit different from a browsing history, as it does not contain the order of items clicked while perusing the website; it only includes the items that were eventually purchased in each transaction. However, the transactions are ordered in time, so we can treat a customer’s full transaction history as a session. Instead of predicting recommendations for what a customer might click on next, we’ll be predicting recommendations for what that customer might actually buy next. Session definitions are flexible, and care must be taken in order to properly interpret the results.

The dataset is composed of the following columns:

- **InvoiceNo**: Invoice number. Nominal, a 6-digit integral number uniquely assigned to each transaction. If this code starts with letter 'c', it indicates a cancellation.
- **StockCode**: Product (item) code. Nominal, a 5-digit integral number uniquely assigned to each distinct product.
- **Description**: Product (item) name. Nominal.
- **Quantity**: The quantities of each product (item) per transaction. Numeric.
- **InvoiceDate**: Invice Date and time. Numeric, the day and time when each transaction was generated.
- **UnitPrice**: Unit price. Numeric, Product price per unit in sterling.
- **CustomerID**: Customer number. Nominal, a 5-digit integral number uniquely assigned to each customer.
- **Country**: Country name. Nominal, the name of the country where each customer resides.
<!-- #endregion -->

```python id="t5DGOggiTryP" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1639397725768, "user_tz": -330, "elapsed": 2418, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="42c4d353-8dc5-49df-d01f-9c08b3c51ded"
!wget -O data.zip https://github.com/RecoHut-Datasets/retail_session/raw/v1/onlineretail.zip
!unzip data.zip
```

```python id="Hgc7vOvTV7y9"
def load_original_ecomm(pathname=ECOMM_PATH):
    df = pd.read_csv(os.path.join(pathname, ECOMM_FILENAME),
        encoding="ISO-8859-1",
        parse_dates=["InvoiceDate"],
    )
    return df
```

```python colab={"base_uri": "https://localhost:8080/", "height": 293} id="TmsQtNH4dEwb" outputId="88389534-aedf-4c66-9d27-cfaff583d2bd" executionInfo={"status": "ok", "timestamp": 1639397728832, "user_tz": -330, "elapsed": 3070, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
df = load_original_ecomm()
df.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="QnlEr3cngRbq" outputId="fafe3543-51e9-4290-bf18-0f14063879b3" executionInfo={"status": "ok", "timestamp": 1639397728833, "user_tz": -330, "elapsed": 11, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
df.isnull().sum()
```

<!-- #region id="XedYV2JkWigc" -->
## Preprocess
<!-- #endregion -->

<!-- #region id="474PEoLOgFI_" -->
There are some rows with missing information, so we'll filter those out. Since we want to define customer sessions, we'll use group by CustomerID field and filter out any customer entries that have fewer than three purchased items.
<!-- #endregion -->

<!-- #region id="NLatMazxvKAt" -->
- Personally identifying information has already been removed
- We removed entries that did not contain a customer ID number (which is how we define a session)
- We removed sessions that contain fewer than three purchased items. A session with only two, for instance, is just a [query item, ground truth item] pair and does not give us any examples for training
<!-- #endregion -->

```python id="3KVsMP58gp5h"
def preprocess_ecomm(df, min_session_count=3):

    df.dropna(inplace=True)
    item_counts = df.groupby(["CustomerID"]).count()["StockCode"]
    df = df[df["CustomerID"].isin(item_counts[item_counts >= min_session_count].index)].reset_index(drop=True)
    
    return df
```

```python colab={"base_uri": "https://localhost:8080/", "height": 293} id="5138zkW_g1g0" outputId="0f6eceb6-f226-43d0-c32e-9483841b2ab5" executionInfo={"status": "ok", "timestamp": 1639397729609, "user_tz": -330, "elapsed": 22, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
df = preprocess_ecomm(df)
df.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="0P1BcYYYhDqI" outputId="f54b9869-1131-4708-dc37-17d30321d43a" executionInfo={"status": "ok", "timestamp": 1639397729611, "user_tz": -330, "elapsed": 21, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
# Number of unique customers after preprocessing
df.CustomerID.nunique()
```

```python colab={"base_uri": "https://localhost:8080/"} id="wq63Y-D-hWKR" outputId="fd780147-969a-4570-e330-e3b678847fb4" executionInfo={"status": "ok", "timestamp": 1639397729612, "user_tz": -330, "elapsed": 17, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
# Number of unique stock codes (products)
df.StockCode.nunique()
```

<!-- #region id="lKpBJdaOiExk" -->
## Product popularity
Here we plot the frequency by which each product is purchased (occurs in a transaction). Most products are not very popular and are only purchased a handful of times. On the other hand, a few products are wildly popular and purchased thousands of times.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 441} id="TZLOGjaPiHPY" outputId="ae45988a-c017-445d-cfaf-a459dd150fc9" executionInfo={"status": "ok", "timestamp": 1639397731128, "user_tz": -330, "elapsed": 1528, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
plt.style.use("seaborn-white")

# Number of unique customer IDs
product_counts = df.groupby(['StockCode']).count()['InvoiceNo'].values

fig = plt.figure(figsize=(8,6))
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)

plt.semilogy(sorted(product_counts))
plt.ylabel("Product counts", fontsize=16);
plt.xlabel("Product index", fontsize=16);

plt.tight_layout()
```

<!-- #region id="I4evk-MwiT-f" -->
The left side of the figure corresponds to products that are not very popular (because they aren't purchased very often), while the far right side indicates that some products are extremely popular and have been purchased hundreds of times.

## Customer session lengths
We define a customer's "session" as all the products they purchased in each transaction, in the order in which they were purchased (ordered InvoiceDate). We can then examine statistics regarding the length of these sessions. Below is a boxplot of all customer session lengths.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 441} id="0nlaj3ieiILl" outputId="c235e944-39e9-4eba-9ba4-242765988731" executionInfo={"status": "ok", "timestamp": 1639397731129, "user_tz": -330, "elapsed": 15, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
session_lengths = df.groupby("CustomerID").count()['InvoiceNo'].values

fig = plt.figure(figsize=(8,6))
plt.xticks(fontsize=14)

ax = sns.boxplot(x=session_lengths, color=cldr_colors[2])

for patch in ax.artists:
    r, g, b, a = patch.get_facecolor()
    patch.set_facecolor((r, g, b, .7))
    
plt.xlim(0,600)
plt.xlabel("Session length (# of products purchased)", fontsize=16);

plt.tight_layout()
plt.savefig("session_lengths.png", transparent=True, dpi=150)
```

```python colab={"base_uri": "https://localhost:8080/"} id="a6aVEx4uiIIi" outputId="6d7db7d5-7bcb-4d8a-93ef-eab20c357406" executionInfo={"status": "ok", "timestamp": 1639397731131, "user_tz": -330, "elapsed": 13, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
print("Minimum session length: \t", min(session_lengths))
print("Maximum session length: \t", max(session_lengths))
print("Mean session length: \t \t", np.mean(session_lengths))
print("Median session length: \t \t", np.median(session_lengths))
print("Total number of purchases: \t", np.sum(session_lengths))
```

<!-- #region id="0-wyj-SCtGNr" -->
The median customer purchased 44 products over the course of the dataset, while the average customer purchased 96 products.
<!-- #endregion -->

<!-- #region id="pJ7qK-MCXBwx" -->
## Sessionization
<!-- #endregion -->

<!-- #region id="YYeVyqfsvRtt" -->
The effectiveness of an algorithm is measured by its ability to predict items withheld from the session. There are a variety of withholding strategies:

- withholding the last element of each session
- iteratively revealing each interaction in a session
- in cases where each user has multiple sessions, withholding the entire final session

In this tutorial, we have employed withholding the last element of the session.


<!-- #endregion -->

```python id="V_tNkOldV7t6"
def construct_session_sequences(df, sessionID, itemID, save_filename):
    """
    Given a dataset in pandas df format, construct a list of lists where each sublist
    represents the interactions relevant to a specific session, for each sessionID. 
    These sublists are composed of a series of itemIDs (str) and are the core training 
    data used in the Word2Vec algorithm. 
    This is performed by first grouping over the SessionID column, then casting to list
    each group's series of values in the ItemID column. 
    INPUTS
    ------------
    df:                 pandas dataframe
    sessionID: str      column name in the df that represents invididual sessions
    itemID: str         column name in the df that represents the items within a session
    save_filename: str  output filename 
  
    Example:
    Given a df that looks like 
    SessionID |   ItemID 
    ----------------------
        1     |     111
        1     |     123
        1     |     345
        2     |     045 
        2     |     334
        2     |     342
        2     |     8970
        2     |     345
    
    Retrun a list of lists like this: 
    sessions = [
            ['111', '123', '345'],
            ['045', '334', '342', '8970', '345'],
        ]
    """
    grp_by_session = df.groupby([sessionID])

    session_sequences = []
    for name, group in grp_by_session:
        session_sequences.append(list(group[itemID].values))

    pickle.dump(session_sequences, open(save_filename, "wb"))
    return session_sequences
```

```python colab={"base_uri": "https://localhost:8080/", "height": 171} id="jiswHnavV7wt" outputId="a4078ea5-9b1c-4253-aa38-97ab6628ced3"
filename = os.path.join(ECOMM_PATH, ECOMM_FILENAME.replace(".csv", "_sessions.pkl"))
sessions = construct_session_sequences(df, "CustomerID", "StockCode", save_filename=filename)
' --> '.join(sessions[0])
```

```python id="7nvizM2Tl4_n"
def load_ecomm(filename=None):
    """
    Checks to see if the processed Online Retail ecommerce session sequence file exists
        If True: loads and returns the session sequences
        If False: creates and returns the session sequences constructed from the original data file
    """
    original_filename = os.path.join(ECOMM_PATH, ECOMM_FILENAME)
    if filename is None:
        processed_filename = original_filename.replace(".csv", "_sessions.pkl")
        if os.path.exists(processed_filename):
            return pickle.load(open(processed_filename,'rb'))

    df = load_original_ecomm(original_filename)
    df = preprocess_ecomm(df)
    session_sequences = construct_session_sequences(df, "CustomerID", "StockCode",
                                                    save_filename=original_filename)
    return session_sequences
```

<!-- #region id="4IgJEMr7XHB2" -->
## Splitting
<!-- #endregion -->

<!-- #region id="s1vAtXtVvYhp" -->
Wherein the first n-1 items highlighted in a green box act as part of the training set, while the item outside is used as ground truth for the recommendations generated.

For each customer in the Online Retail Data Set, we construct the training set from the first n-1 purchased items. We construct test and validation sets as a series of [query item, ground truth item] pairs. The test and validation sets must be disjoint—that is, each set is composed of pairs with no pairs shared between the two sets (or else we would leak information from our validation into the final test set!).
<!-- #endregion -->

```python id="GpIVmG6QV7rG"
def train_test_split(session_sequences, test_size: int = 10000, rng=rng):
    """
    Next Event Prediction (NEP) does not necessarily follow the traditional train/test split. 
    Instead training is perform on the first n-1 items in a session sequence of n items. 
    The test set is constructed of (n-1, n) "query" pairs where the n-1 item is used to generate 
    recommendation predictions and it is checked whether the nth item is included in those recommendations. 
    Example:
        Given a session sequence ['045', '334', '342', '8970', '128']
        Training is done on ['045', '334', '342', '8970']
        Testing (and validation) is done on ['8970', '128']
    
    Test and Validation sets are constructed to be disjoint. 
    """

    ## Construct training set
    # use (1 st, ..., n-1 th) items from each session sequence to form the train set (drop last item)
    train = [sess[:-1] for sess in session_sequences]

    if test_size > len(train):
        print(
            f"Test set cannot be larger than train set. Train set contains {len(train)} sessions."
        )
        return

    ## Construct test and validation sets
    # sub-sample 10k sessions, and use (n-1 th, n th) pairs of items from session_squences to form the
    # disjoint validaton and test sets
    test_validation = [sess[-2:] for sess in session_sequences]
    index = np.random.choice(range(len(test_validation)), test_size * 2, replace=False)
    test = np.array(test_validation)[index[:test_size]].tolist()
    validation = np.array(test_validation)[index[test_size:]].tolist()

    return train, test, validation
```

```python colab={"base_uri": "https://localhost:8080/"} id="dR2iVWuscfvt" outputId="218a61ca-53c1-4381-8af0-b00651e97fe5"
print(len(sessions))
train, test, valid = train_test_split(sessions, test_size=1000)
print(len(train), len(valid), len(test))
```

<!-- #region id="nrfW72QYXL3Z" -->
## Metrics
<!-- #endregion -->

<!-- #region id="b-5ri_xSwfok" -->
For a sequence containing n interactions, we use the first (n-1) items in that sequence as part of the model training set. We randomly sample (n-1th, nth) pairs from these sequences for the validation and test sets. For prediction, we use the last item in the training sequence (the n-1th item) as the query item, and predict the K closest items to the query item using cosine similarity between the vector representations. We can then evaluate with the following metrics:

- Recall at K (Recall@K) defined as the proportion of cases in which the ground truth item is among the top K recommendations for all test cases (that is, a test example is assigned a score of 1 if the nth item appears in the list, and 0 otherwise).
- Mean Reciprocal Rank at K (MRR@K), takes the average of the reciprocal ranks of the ground truth items within the top K recommendations for all test cases (that is, if the nth item was second in the list of recommendations, its reciprocal rank would be 1/2). This metric measures and favors higher ranks in the ordered list of recommendation results.
<!-- #endregion -->

```python id="9gZKiSz4fALx"
def recall_at_k(test, embeddings, k: int = 10) -> float:
    """
    test must be a list of (query, ground truth) pairs
    embeddings must be a gensim.word2vec.wv thingy
    """
    ratk_score = 0
    for query_item, ground_truth in test:
        # get the k most similar items to the query item (computes cosine similarity)
        neighbors = embeddings.similar_by_vector(query_item, topn=k)
        # clean up the list
        recommendations = [item for item, score in neighbors]
        # check if ground truth is in the recommedations
        if ground_truth in recommendations:
            ratk_score += 1
    ratk_score /= len(test)
    return ratk_score


def recall_at_k_baseline(test, comatrix, k: int = 10) -> float:
    """
    test must be a list of (query, ground truth) pairs
    embeddings must be a gensim.word2vec.wv thingy
    """
    ratk_score = 0
    for query_item, ground_truth in test:
        # get the k most similar items to the query item (computes cosine similarity)
        try:
            co_occ = collections.Counter(comatrix[query_item])
            items_and_counts = co_occ.most_common(k)
            recommendations = [item for (item, counts) in items_and_counts]
            if ground_truth in recommendations: 
                ratk_score +=1
        except:
            pass
    ratk_score /= len(test)
    return ratk_score


def hitratio_at_k(test, embeddings, k: int = 10) -> float:
    """
    Implemented EXACTLY as was done in the Hyperparameters Matter paper. 
    In the paper this metric is described as 
        • Hit ratio at K (HR@K). It is equal to 1 if the test item appears
        in the list of k predicted items and 0 otherwise [13]. 
    
    But this is not what they implement, where they instead divide by k. 
    What they have actually implemented is more like Precision@k.
    However, Precision@k doesn't make a lot of sense in this context because
    there is only ONE possible correct answer in the list of generated 
    recommendations.  I don't think this is the best metric to use but 
    I'll keep it here for posterity. 
    test must be a list of (query, ground truth) pairs
    embeddings must be a gensim.word2vec.wv thingy
    """
    hratk_score = 0
    for query_item, ground_truth in test:
        # If the query item and next item are the same, prediction is automatically correct
        if query_item == ground_truth:
            hratk_score += 1 / k
        else:
            # get the k most similar items to the query item (computes cosine similarity)
            neighbors = embeddings.similar_by_vector(query_item, topn=k)
            # clean up the list
            recommendations = [item for item, score in neighbors]
            # check if ground truth is in the recommedations
            if ground_truth in recommendations:
                hratk_score += 1 / k
    hratk_score /= len(test)
    return hratk_score*1000


def mrr_at_k(test, embeddings, k: int) -> float:
    """
    Mean Reciprocal Rank. 
    test must be a list of (query, ground truth) pairs
    embeddings must be a gensim.word2vec.wv thingy
    """
    mrratk_score = 0
    for query_item, ground_truth in test:
        # get the k most similar items to the query item (computes cosine similarity)
        neighbors = embeddings.similar_by_vector(query_item, topn=k)
        # clean up the list
        recommendations = [item for item, score in neighbors]
        # check if ground truth is in the recommedations
        if ground_truth in recommendations:
            # identify where the item is in the list
            rank_idx = (
                np.argwhere(np.array(recommendations) == ground_truth)[0][0] + 1
            )
            # score higher-ranked ground truth higher than lower-ranked ground truth
            mrratk_score += 1 / rank_idx
    mrratk_score /= len(test)
    return mrratk_score


def mrr_at_k_baseline(test, comatrix, k: int = 10) -> float:
    """
    Mean Reciprocal Rank. 
    test must be a list of (query, ground truth) pairs
    embeddings must be a gensim.word2vec.wv thingy
    """
    mrratk_score = 0
    for query_item, ground_truth in test:
        # get the k most similar items to the query item (computes cosine similarity)
        try:
            co_occ = collections.Counter(comatrix[query_item])
            items_and_counts = co_occ.most_common(k)
            recommendations = [item for (item, counts) in items_and_counts]
            if ground_truth in recommendations: 
                rank_idx = (
                    np.argwhere(np.array(recommendations) == ground_truth)[0][0] + 1
                )
                mrratk_score += 1 / rank_idx
        except:
            pass
    mrratk_score /= len(test)
    return mrratk_score
```

<!-- #region id="rxY8Dbfzshsv" -->
## Baseline analysis
<!-- #endregion -->

<!-- #region id="XOkNZ1yevgeB" -->
There are many baselines for the next event prediction (NEP) task. The simplest and most common are designed to recommend the item that most frequently co-occurs with the last item in the session. Known as “Association Rules,” this heuristic is straightforward, but doesn’t capture the complexity of the user’s session history.
<!-- #endregion -->

```python id="ZPmpUqm_smzm"
def association_rules_baseline(train_sessions):
    """
    Constructs a co-occurence matrix that counts how frequently each item 
    co-occurs with any other item in a given session. This matrix can 
    then be used to generate a list of recommendations according to the most
    frequently co-occurring items for the item in question. 

    These recommendations must be evaluated using the "_baseline"  recall/mrr functions in metrics.py
    """
    comatrix = collections.defaultdict(list)
    for session in train_sessions:
        for (x, y) in itertools.permutations(session, 2):
            comatrix[x].append(y)
    return comatrix
```

```python colab={"base_uri": "https://localhost:8080/"} id="dCUFeOw4smwI" outputId="bb1aaf8c-b440-4e45-f69c-a6f473e193b6"
# Construct a co-occurrence matrix containing how frequently 
# each item is found in the same session as any other item
comatrix = association_rules_baseline(train)

# Recommendations are generated as the top K most frequently co-occurring items
# Compute metrics on these recommendations for each (query item, ground truth item)
# pair in the test set
recall_at_10 = recall_at_k_baseline(test, comatrix, k=10)
mrr_at_10 = mrr_at_k_baseline(test, comatrix, k=10)

print("Recall@10:", recall_at_10)
print("MRR@10:", mrr_at_10)
```

<!-- #region id="sfCnGNBUXT-V" -->
## Initializing Ray
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="9uT3Woact705" outputId="36390597-5e7d-482b-ba7d-702bff389666"
ray.init(num_cpus=4, ignore_reinit_error=True)
```

<!-- #region id="nEn7YBp94I3K" -->
## Train word2vec with logging
<!-- #endregion -->

<!-- #region id="k24DXm6fvmfs" -->
More recently, deep learning approaches have begun to make waves. Variations of graph neural networks and recurrent neural networks have been applied to the problem with promising results, and currently represent the state of the art in NEP for several use cases. However, while these algorithms capture complexity, they can also be difficult to understand, unintuitive in their recommendations, and not always better than comparably simple algorithms (in terms of prediction accuracy).

There is still another option, though, that sits between simple heuristics and deep learning algorithms. It’s a model that can capture semantic complexity with only a single layer: word2vec.

We can treat each session as a sentence, with each item or product in the session representing a “word.” A website’s collection of user browser histories will act as the corpus. Word2vec will crunch over the entire corpus, learning relationships between products in the context of user browsing behavior. The result will be a collection of embeddings: one for each product. The idea is that these learned product embeddings will contain more information than a simple heuristic, and training the word2vec algorithm is typically faster and easier than training more complex, data-hungry deep learning algorithms.


<!-- #endregion -->

```python id="C_6624rRjSdW"
def train_w2v(train_data, params:dict, callbacks=None, model_name=None):
    if model_name: 
        # Load a model for additional training. 
        model = Word2Vec.load(model_name)
    else: 
        # train model
        if callbacks:
            model = Word2Vec(callbacks=callbacks, **params)
        else:
            model = Word2Vec(**params)
        model.build_vocab(train_data)

    model.train(train_data, total_examples=model.corpus_count, epochs=model.epochs, compute_loss=True)
    vectors = model.wv
    return vectors
    

def tune_w2v(config):
    ratk_logger = RecallAtKLogger(valid, k=config['k'], ray_tune=True)

    # remove keys from config that aren't hyperparameters of word2vec
    config.pop('dataset')
    config.pop('k')
    train_w2v(train, params=config, callbacks=[ratk_logger])


class RecallAtKLogger(CallbackAny2Vec):
    '''Report Recall@K at each epoch'''
    def __init__(self, validation_set, k, ray_tune=False, save_model=False):
        self.epoch = 0
        self.recall_scores = []
        self.validation = validation_set
        self.k = k
        self.tune = ray_tune
        self.save = save_model

    def on_epoch_begin(self, model):
        if not self.tune:
            print(f'Epoch: {self.epoch}', end='\t')

    def on_epoch_end(self, model):
        # method 1: deepcopy the model and set the model copy's wv to None
        mod = deepcopy(model)
        mod.wv.norms = None # will cause it recalculate norms? 
        
        # Every 10 epochs, save the model 
        if self.epoch%10 == 0 and self.save: 
            # method 2: save and reload the model
            model.save(f"{MODEL_DIR}w2v_{self.epoch}.model")
            #mod = Word2Vec.load(f"w2v_{self.epoch}.model")
        
        ratk_score = recall_at_k(self.validation, mod.wv, self.k)  

        if self.tune: 
            tune.report(recall_at_k = ratk_score)    
        else:
            self.recall_scores.append(ratk_score)
            print(f' Recall@10: {ratk_score}')
        self.epoch += 1


class LossLogger(CallbackAny2Vec):
    '''Report training loss at each epoch'''
    def __init__(self):
        self.epoch = 0
        self.previous_loss = 0
        self.training_loss = []

    def on_epoch_end(self, model):
        # the loss output by Word2Vec is more akin to a cumulative loss and increases each epoch
        # to get a value closer to loss per epoch, we subtract
        cumulative_loss = model.get_latest_training_loss()
        loss = cumulative_loss - self.previous_loss
        self.previous_loss = cumulative_loss
        self.training_loss.append(loss)
        print(f' Loss: {loss}')
        self.epoch += 1
```

```python id="1TeCHqd30rcy"
expt_dir = '/content/big_HPO_no_distributed'
```

```python colab={"base_uri": "https://localhost:8080/", "height": 466} id="xEJnbuOgsmLQ" outputId="29669c60-14ec-444c-831e-3c05cc101d3c"
use_saved_expt = False
if use_saved_expt:
  analysis = Analysis(expt_dir, default_metric="recall_at_k", default_mode="max")
  w2v_params = analysis.get_best_config()
else:
  w2v_params = {
          "min_count": 1,
          "iter": 5,
          "workers": 10,
          "sg": 1,
      }

# Instantiate callback to measurs Recall@K on the validation set after each epoch of training
ratk_logger = RecallAtKLogger(valid, k=10, save_model=True)
# Instantiate callback to compute Word2Vec's training loss on the training set after each epoch of training
loss_logger = LossLogger()
# Train Word2Vec model and retrieve trained embeddings
embeddings = train_w2v(train, w2v_params, [ratk_logger, loss_logger])

# Save results
pickle.dump(ratk_logger.recall_scores, open(os.path.join("/content", f"recall@k_per_epoch.pkl"), "wb"))
pickle.dump(loss_logger.training_loss, open(os.path.join("/content", f"trainloss_per_epoch.pkl"), "wb"))

# Save trained embeddings
embeddings.save(os.path.join("/content", f"embeddings.wv"))

# Visualize metrics as a function of epoch
plt.plot(np.array(ratk_logger.recall_scores)/np.max(ratk_logger.recall_scores))
plt.plot(np.array(loss_logger.training_loss)/np.max(loss_logger.training_loss))
plt.show()

# Print results on the test set
print(recall_at_k(test, embeddings, k=10))
print(mrr_at_k(test, embeddings, k=10))
```

<!-- #region id="6F0OaTwC4NJK" -->
## Tune word2vec with ray
<!-- #endregion -->

<!-- #region id="84gKhlj9v3uN" -->
In the previous section, we simply trained word2vec using the default hyperparameters-but hyperparameters matter! In addition to the learning rate or the embedding size (hyperparameters likely familiar to many), word2vec has several others which have considerable impact on the resulting embeddings. Let’s see how.

The default word2vec's parameters in Gensim library were found to produce semantically meaningful representations for words in documents, but we are learning embeddings for products in online sessions. The order of products in online sessions will not have the same structure as words in sentences, so we’ll need to consider adjusting word2vec’s hyperparameters to be more appropriate to the task.
<!-- #endregion -->

<!-- #region id="aAcFrEVPv5dO" -->
<!-- #endregion -->

<!-- #region id="pqWaBbQxv7Em" -->
This table shows the main hyperparameters we tuned over. For each one, we show the starting and ending values we tried, along with the step size we used. The total number of trials is computed by multiplying each value in the Configurations column.

We trained a word2vec model using the best hyperparameters found above, which resulted in a Recall@10 score of 25.18±0.19 on the validation set, and 25.21±.26 for the test set. These scores may not seem immediately impressive, but if we consider that there are more than 3600 different products to recommend, this is far better than random chance!
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 258} id="W-rvETTgsepP" outputId="93a8ae6c-cc79-45df-8b08-899c017f8135"
from ray.tune import Analysis
from ray.tune.schedulers import ASHAScheduler

# Define the hyperparameter search space for Word2Vec algorithm
search_space = {
    "dataset": "ecomm",
    "k": 10,
    "size": tune.grid_search(list(np.arange(10,106, 6))),
    "window": tune.grid_search(list(np.arange(1,22, 3))),
    "ns_exponent": tune.grid_search(list(np.arange(-1, 1.2, .2))),
    "alpha": tune.grid_search([0.001, 0.01, 0.1]),
    "negative": tune.grid_search(list(np.arange(1,22, 3))),
    "iter": 10,
    "min_count": 1,
    "workers": 6,
    "sg": 1,
}

use_asha = True
smoke_test = False

# The ASHA Scheduler will stop underperforming trials in a principled fashion
asha_scheduler = ASHAScheduler(max_t=100, grace_period=10) if use_asha else None

# Set the stopping critera -- use the smoke-test arg to test the system 
stopping_criteria = {"training_iteration": 1 if smoke_test else 9999}

# Perform hyperparamter sweep with Ray Tune
analysis = tune.run(
    tune_w2v,
    name=expt_dir,
    local_dir="ray_results",
    metric="recall_at_k",
    mode="max",
    scheduler=asha_scheduler,
    stop=stopping_criteria,
    num_samples=1,
    verbose=1,
    resources_per_trial={
        "cpu": 1,
        "gpu": 0
    },
    config=search_space,
)
print("Best hyperparameters found were: ", analysis.best_config)
```

<!-- #region id="jjdH1oSHrt2p" -->
Ray Tune saves the results of each trial in the ray_results directory. Each time Ray Tune performs an HPO sweep, the results for that run are saved under a unique subdirectory. In this case, we named that subdirectory big_HPO_no_distributed. Ray Tune provides methods for interacting with these results, starting with the Analysis class that loads the results from each trial, including performance metrics as a function of training time and tons of metadata.

These results are stored as JSON but the Analysis class provides a nice wrapper for converting those results in a pandas dataframe.
<!-- #endregion -->

<!-- #region id="RexEQbOd3TNR" -->
## Explore the results of the full hyperparameter sweep
Next, we're going to look at how the Recall@10 score changes as a function of various hyperparameter configurations that we tuned over. We tuned over three hyperparameters: the context window size, negative sampling exponent, and the number of negative samples.

We want to look at the Recall@10 scores for all of these configurations but this is a 3-dimensional space and, as such, will be difficult to visualize. Instead, we'll "collapse" one dimension, while examining the other two. To do this, we aggregate the Recall@10 scores (taking the mean) along the "collapsed" dimension.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 949} id="O7lbr2-5sebq" outputId="a2704370-f4b4-464a-87ee-6b4b6f7ed347"
analysis = Analysis("big_HPO_no_distributed/", 
                    default_metric="recall_at_k",
                    default_mode="max")

results = analysis.dataframe()
results
```

<!-- #region id="FHKEwe9u3LDv" -->
The Analysis objects also has methods to quickly retrieve the best configuration found during the HPO sweep.


<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="WnKAoaER2vkp" outputId="1f50520a-3267-4d0c-fb99-d7a566fbc95b"
best_config = analysis.get_best_config()
best_config
```

<!-- #region id="tGk8zNkp3Pkr" -->
While the results dataframe contains the final Recall@10 scores for each of the 539 trials, it's also nice to explore how those scores evolved as a function of training for any given trial. Again, the Analysis class delivers, providing the ability to access the full training results for any of the trials. Below we plot the Recall@10 score as a function of training epochs for the best configuration.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 276} id="y8QF0YZY3GXZ" outputId="01f09f26-14ee-4003-d621-7135cf1f0861"
best_path = analysis.get_best_logdir()
dfs = analysis.fetch_trial_dataframes()

plt.plot(dfs[best_path]['recall_at_k']);
plt.xlabel("Epoch")
plt.ylabel("Recall@10");
```

```python id="z6sCIiAK3Rmt"
def aggregate_z(x_name, y_name):
    grouped = results.groupby([f"config/{x_name}", f"config/{y_name}"])
    x_values = []
    y_values = []
    mean_recall_values = []
    
    for name, grp in grouped:
        x_values.append(name[0])
        y_values.append(name[1])
        mean_recall_values.append(grp['recall_at_k'].mean())
    return x_values, y_values, mean_recall_values
```

```python colab={"base_uri": "https://localhost:8080/", "height": 327} id="6MDZY3uQ3xDA" outputId="11d3cdd2-949c-4bc7-ee82-24ee63c5ba46"
fig = plt.figure(figsize=(15,5))

ax = fig.add_subplot(131)
negative, ns_exp, recall = aggregate_z("negative", "ns_exponent")
cm = sns.scatterplot(x=ns_exp, y=negative, hue=recall, palette=color_palette, legend=None)
ax.set_xlabel("Negative sampling exponent", fontsize=16)
ax.set_ylabel("Number of negative samples", fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
ax.plot(0.75, 5, 
         marker='*', 
         color=cldr_colors[1],
         markersize=10)
ax.plot(best_config['ns_exponent'], 
        best_config['negative'], 
         marker="o", 
         fillstyle='none', 
         color=cldr_colors[0],
         markersize=15)
ax = fig.add_subplot(132)

window, ns_exp, recall = aggregate_z("window", "ns_exponent")
cm = sns.scatterplot(x=ns_exp, y=window, hue=recall, palette=color_palette, legend=None)
ax.set_xlabel("Negative sampling exponent", fontsize=16)
ax.set_ylabel("Context window size", fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
ax.plot(0.75, 5, 
           marker='*', 
           color=cldr_colors[1],
           markersize=10)
ax.plot(best_config['ns_exponent'], 
         best_config['window'], 
         marker="o", 
         fillstyle='none', 
         color=cldr_colors[0],
         markersize=15)

ax = fig.add_subplot(133)
window, negative, recall = aggregate_z("window", "negative")
cm = sns.scatterplot(x=window, y=negative, hue=recall, palette=color_palette, legend=None)
ax.set_xlabel("Number of negative examples", fontsize=16)
ax.set_ylabel("Context window size", fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
ax.plot(5, 5, 
        marker='*',
        color=cldr_colors[1],
        markersize=10)
ax.plot(best_config['window'], 
         best_config['negative'], 
         marker="o", 
         fillstyle='none', 
         color=cldr_colors[0],
         markersize=15);

plt.tight_layout()
plt.savefig("hpsweep_results.png", transparent=True, dpi=150)
```

<!-- #region id="-5KwIKIl36gA" -->
And there we have it! Each panel shows the Recall@10 scores (where yellow is a high score and purple is a low score) associated with a unique configuration of hyperparameters. The best hyperparameter values for the Online Retail Data Set are denoted by the light blue circle. Word2vec’s default values are shown by the orange star. In all cases, the orange star is nowhere near the light blue circle, indicating that the default values are not optimal for this dataset.
<!-- #endregion -->

<!-- #region id="wYL9GUoPwSh0" -->
## Online Evaluation

Offline evaluations rarely inform us about the quality of recommendations as perceived by the users. In one study of e-commerce session-based recommenders, it was observed that offline evaluation metrics often fall short because they tend to reward an algorithm when they predict the exact item that the user clicked or purchased. In real life, though, there are identical products that could potentially make equally good recommendations. To overcome these limitations, we suggest incorporating human feedback on the recommendations from offline evaluations before conducting A/B tests.


<!-- #endregion -->

<!-- #region id="p4qrxHCzwJxR" -->
## Summary

- We experimented with an NLP-based algorithm—word2vec—which is known for learning low-dimensional word representations that are contextual in nature.
- We applied it to an e-commerce dataset containing historical purchase transactions, to learn the structure induced by both the user’s behavior and the product’s nature to recommend the next item to be purchased.
- While doing so, we learned that hyperparameter choices are data- and task-dependent, and especially, that they differ from linguistic tasks; what works for language models does not necessarily work for recommendation tasks.
- That said, our experiments indicate that in addition to specific parameters (like negative sampling exponent, the number of negative samples, and context window size), the number of training epochs greatly influences model performance. We recommend that word2vec be trained for as many epochs as computational resources allow, or until performance on a downstream recommendation metric has plateaued.
- We also realized during our experimentation that performing hyperparameter search over just a handful of parameters can be time consuming and computationally expensive; hence, it could be a bottleneck to developing a real-life recommendation solution (with word2vec). While scaling hyperparameter optimization is possible through tools like Ray Tune, we envision additional research into algorithmic approaches to solve this problem would pave the way in developing more scalable (and less complex) solutions.
<!-- #endregion -->

<!-- #region id="chZ1lJ_guGQQ" -->
---
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="l9S61GlZuGQR" executionInfo={"status": "ok", "timestamp": 1639397783258, "user_tz": -330, "elapsed": 3838, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="77933f06-a621-4e44-b690-7f1da6a4dbec"
!pip install -q watermark
%reload_ext watermark
%watermark -a "Sparsh A." -m -iv -u -t -d -p gensim
```

<!-- #region id="z-Ixq7XguGQR" -->
---
<!-- #endregion -->

<!-- #region id="Nn1dX3S7uGQS" -->
**END**
<!-- #endregion -->
