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

<!-- #region id="b2V_6GjQrQxL" -->
# Evaluating Implicit Models on LastFM Music Data
> Comparing iALS, BPR, ItemPop, and Random baseline models on LastFM-250K music dataset

- toc: true
- badges: true
- comments: true
- categories: [Implicit, Music, Evaluation]
- author: "<a href='https://github.com/david-cortes'>David Cortes</a>"
- image:
<!-- #endregion -->

<!-- #region id="8kIUn4kj1pQU" -->
This vignette is an introduction to the Python package
[recometrics](https://www.github.com/david-cortes/recometrics)
for evaluating recommender systems built with implicit-feedback data, assuming
that the recommendation models are based on low-rank matrix factorization
(example such packages:
[implicit](https://github.com/benfred/implicit),
[lightfm](https://github.com/lyst/lightfm),
[cmfrec](https://github.com/david-cortes/cmfrec),
among many others), or assuming that it is possible to compute a user-item
score as a dot product of user and item factors/components/attributes.
<!-- #endregion -->

<!-- #region id="Ri9oKyHk1t0q" -->
## Implicit-feedback data

Historically, many models for recommender systems were designed by approaching the
problem as regression or rating prediction, by taking as input a matrix
$\mathbf{X}_{ui}$ denoting user likes and dislikes of items in a scale
(e.g. users giving a 1-to-5 star rating to different movies), and evaluating such
models by seeing how well they predict these ratings on hold-out data.

In many cases, it is impossible or very expensive to obtain such data, but one
has instead so called "implicit-feedback" records: that is, observed logs of user
interactions with items (e.g. number of times that a user played each
song in a music service), which do not signal dislikes in the same way as a
1-star rating would, but can still be used for building and evaluating
recommender systems.

In the latter case, the problem is approached more as ranking or classification
instead of regression, with the models being evaluated not by how well they
perform at predicting ratings, but by how good they are at scoring the observed
interactions higher than the non-observed interactions for each user, using
metrics more typical of information retrieval.

Generating a ranked list of items for each user according to their predicted
score and comparing such lists against hold-out data can nevertheless be very
slow (might even be slower than fitting the model itself), and this is where
`recometrics` comes in: it provides efficient routines for calculating many
implicit-feedback recommendation quality metrics, which exploit multi-threading,
SIMD instructions, and efficient sorted search procedures.
<!-- #endregion -->

<!-- #region id="WuMfpVOF1w87" -->
## Matrix factorization models

The perhaps most common approach towards building a recommendation model is by
trying to approximate the matrix $\mathbf{X}_{mn}$ as the product of two
lower-dimensional matrices $\mathbf{A}_{mk}$ and $\mathbf{B}_{nk}$ (with
$k \ll m$ and $k \ll n$), representing latent user and item factors/components,
respectively (which are the model parameters to estimate) - i.e.
$$
\mathbf{X} \approx \mathbf{A} \mathbf{B}^T
$$
In the explicit-feedback setting (e.g. movie ratings), this is typically done by
trying to minimize squared errors with respect to the **observed** entries in
$\mathbf{X}$, while in implicit-feedback settings this is typically done by turning the
$\mathbf{X}$ matrix into a binary matrix which has a one if the observation is observed
and a zero if not, using the actual values (e.g. number of times that a song was played)
instead as weights for the positive entries, thereby looking at **all** entries rather
than just the observed (non-zero) values - e.g.:
$$
\min_{\mathbf{A}, \mathbf{B}} \sum_{u=1}^{m} \sum_{i=1}^{n} x_{ui} (I_{x_{ui}>0} - \mathbf{a}_u \cdot \mathbf{b}_i)^2
$$

The recommendations for a given user are then produced by calculating the full products
between that user vector $\mathbf{a}_u$ and the $\mathbf{B}$ matrix, sorting these
predicted scores in descending order.

For a better overview of implicit-feedback matrix factorization, see the paper
_Hu, Yifan, Yehuda Koren, and Chris Volinsky. "Collaborative filtering for implicit feedback datasets." 2008 Eighth IEEE International Conference on Data Mining. Ieee, 2008._
<!-- #endregion -->

<!-- #region id="90WK68R20qTF" -->
## Evaluating recommendation models

Such matrix factorization models are commonly evaluated by setting aside a small amount
of users as hold-out for evaluation, fitting a model to all the remaining users and
items. Then, from the evaluation users, a fraction of their interactions data is set as a
hold-out test set, while their latent factors are computed using the rest of the data
and the previously fitted model from the other users.

Then, top-K recommendations for each user are produced, discarding the non-hold-out
items with which their latent factors were just determined, and these top-K lists are
compared against the hold-out test items, seeing how well they do at ranking them near
the top vs. how they rank the remainder of the items.

** *

This package can be used to calculate many recommendation quality metrics given the
user and item factors and the train-test data split that was used, including:

* **P\@K** ("precision-at-k"): this is the most intuitive metric. It calculates the
proportion of the top-K recommendations that include items from the test set for
a given user - i.e.
$$
P@K = \frac{1}{k} \sum_{i=1}^k
\begin{cases}
    1, & r_i \in \mathcal{T}\\
    0, & \text{otherwise}
\end{cases}
$$
Where $r_i$ is the item ranked at position $i$ by the model (sorting the predicted
scores in descending order, after excluding the items in the training data for that 
user), and $\mathcal{T}$ is the set of items that are in the test set for that user.

    Note that some papers and libraries define $P@K$ differently, see the second
version below.

* **TP\@K** (truncated $P@K$): same calculation as $P@K$, but will instead divide by
the minimum between $k$ and the number of test items:
$$
TP@K = \frac{1}{\min\{k, |\mathcal{T}|\}} \sum_{i=1}^k
\begin{cases}
    1, & r_i \in \mathcal{T}\\
    0, & \text{otherwise}
\end{cases}
$$

    The "truncated" prefix is a non-standard nomenclature introduced here to
differentiate it from the other $P@K$ metric.

* **R\@K** ("recall-at-k"): while $P@K$ offers an intuitive metric that captures what
a recommender system aims at being good at, it does not capture the fact that,
the more test items there are, the higher the chances that they will be included in the
top-K recommendations. Recall instead looks at what proportion of the test
items would have been retrieved with the top-K recommended list:
$$
R@K = \frac{1}{|\mathcal{T}|} \sum_{i=1}^k
\begin{cases}
    1, & r_i \in \mathcal{T}\\
    0, & \text{otherwise}
\end{cases}
$$

* **AP\@K** ("average precision-at-k"): precision and recall look at all the items
in the top-K equally, whereas one might want to take into account also the ranking
within this top-K list, for which this metric comes in handy.
"Average Precision" tries to reflect the precisions that would be obtained at
different recalls:
$$
AP@K = \frac{1}{|\mathcal{T}|} \sum_{i=1}^k
\begin{cases}
    P@i, & r_i \in \mathcal{T}\\
    0, & \text{otherwise}
\end{cases}
$$
$AP@K$ is a metric which to some degree considers precision, recall, and rank within
top-K. Intuitively, it tries to approximate the are under a precision-recall tradeoff
curve. Its average across users is typically called "MAP\@K" or "Mean Average Precision".

    **Important:** many authors define $AP@K$ differently, such as dividing by the minimum
between $k$ and $|\mathcal{T}|$ instead, or as the average for P\@1..P\@K (either as-is
or stopping the calculation after already retrieving all test items).
See below for the other version.

* **TAP\@K** (truncated $AP@K$): a truncated version of the
$AP@K$ metric, which will instead divide it by the minimum between $k$ and the
number of test items. Just like for $TP@K$, the "truncated" prefix is a non-standard
nomenclature used here to differentiate it from the other more typical $AP@K$.

* **NDCG\@K** ("normalized discounted cumulative gain at k"): while the earlier metrics
look at just the presence of an item in the test set, these items might not all be as
good, with some of them having higher observed values than others. NDCG aims at
judging these values, but discounted according to the rank in the top-K list. First
it calculates the unstandardized discounted cumulative gain:
$$
DCG@K = \sum_{i=1}^{k} \frac{C_{r_i}}{log_2 (1+i)}
$$
Where $C_{r_i}$ indicates the observed interaction value in the test data for item
$r_i$, and is zero if the item was not in the test data. The DCG\@K metric is then
standardized by dividing it by the maximum achievable DCG\@K for the test data:
$$
NDCG@K = \frac{DCG@K}{\max DCG@K}
$$

    Unlike the other metrics, NDCG can handle data which contains "dislikes" in the
form of negative values. If there are no negative values in the test data, it will
be bounded between zero and one.

* **Hit\@K** (from which "Hit Rate" is calculated): this is a simpler yes/no metric
that looks at whether any of the top-K recommended items were in the test set for
a given user:
$$
Hit@K = \max_{i=1..K}
\begin{cases}
    1, & r_i \in \mathcal{T}\\
    0, & \text{otherwise}
\end{cases}
$$
The average of this metric across users is typically called "Hit Rate".

* **RR\@K** ("reciprocal rank at k", from which "MRR" or "mean reciprocal rank"
is calculated):
this metric only looks at the rank of the first recommended item that is in the test set,
and outputs its inverse:
$$
RR@K = \max_{i=1..K} \frac{1}{i} \:\:\:\text{s.t.}\:\:\: r_i \in \mathcal{T}
$$
The average of this metric across users is typically called "Mean Reciprocal Rank".

* **ROC AUC** ("area under the receiver-operating characteristic curve"): see the
[Wikipedia entry](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)
for details. While the metrics above only looked at the top-K
recommended items, this metric looks at the full ranking of items instead, and
produces a standardized number between zero and one in which 0.5 denotes random
predictions.

* **PR AUC** ("area under the precision-recall curve"): while ROC AUC provides an
overview of the overall ranking, one is typically only interested in how well it
retrieves test items within top ranks, and for this the area under the
precision-recall curve can do a better job at judging rankings, albeit the metric
itself is not standardized and its minimum does not go as low as zero.

    The metric is calculated using the fast but not-so-precise rectangular method,
whose formula corresponds to the AP\@K metric with K=N. Some papers and libraries
call this the average of this metric the "MAP" or "Mean Average Precision" instead
(without the "\@K").

_(For more details about the metrics, see the [package documentation](https://recometrics.readthedocs.io))_

**NOT** covered by this package:

* Metrics that look at the rareness of the items recommended (to evaluate so-called
"serendipity").

* Metrics that look at "discoverability".

* Metrics that take into account the diversity of the ranked lists.

** *

Now a practical example with the [LastFM-360K](http://ocelma.net/MusicRecommendationDataset/lastfm-360K.html) dataset, which contains the number of times that different users played different songs from the Last.FM service.


The example will compare different models from two popular libraries for recommender systems: [implicit](https://github.com/benfred/implicit) and [lightfm](https://github.com/lyst/lightfm). This library (`recosystem`) is able to work with any other library that would produce user and item embeddings, but for speed purposes the comparison will be limited to those two, as other popular libraries such as e.g. `spotlight` or `cornac` can be a few orders of magnitude slower in large datasets.

For better results, one might want to apply transformations to these counts before fitting ALS models, such as taking logarithms and/or dividing the counts by some larger number, but for simplicity purposes, this notebook will use them as-is.
<!-- #endregion -->

<!-- #region id="YEu-p7Hf14c6" -->
## Loading the data

Loading the data and converting the triplets to sparse matrices:
<!-- #endregion -->

```python id="f63LDg100qTW" outputId="829dbb50-cae2-4486-8dc5-30f9c9009d2c"
import numpy as np, pandas as pd
from scipy.sparse import coo_matrix

lfm = pd.read_table('usersha1-artmbid-artname-plays.tsv',
                    sep='\t', header=None,
                    names=['UserId','ItemId', 'Artist','Count'])
lfm.head(3)
```

```python id="GWfw4nSU0qTZ" outputId="ae6d409d-d21f-4816-bbca-2b06968e517d"
lfm = lfm.drop('Artist', axis=1)
lfm = lfm.loc[(lfm.Count > 0) & (lfm.UserId.notnull()) & (lfm.ItemId.notnull())]
lfm['UserId'] = pd.Categorical(lfm.UserId).codes
lfm['ItemId'] = pd.Categorical(lfm.ItemId).codes
lfm.head(3)
```

```python id="NN2C5l3o0qTa" outputId="8d4bd03f-af8e-4773-8c81-08b003ce1c2c"
X = coo_matrix((lfm.Count, (lfm.UserId, lfm.ItemId)))
X
```

<!-- #region id="zxgMUIp30qTb" -->
## Creating a train-test split

Now leaving aside a random sample of 10,000 users for model evaluation, for whom 30%
of the data will be left as a hold-out test set.

**Important:** `recometrics` can produce train-test splits that are intended to work in 2 possible ways:

1. Selecting a sample of test users, then for each of those users selecting train and test items for each, while fitting the model **to the remainder of the users**, and then using that fitted model on the train data for these test users to produce new factors.
2. Selecting a sample of test users, then for each of those users selecting train and test items for each, while fitting the model **to the remainder of the users PLUS the training data of the test users**, and using the obtained user factors directly.

The first approach is more representative of real model usage and it is recommended to follow, but many popular Python libraries for recommender systems lack the functionality for calculating new user factors after the model is already fitted (for example, packages `implicit` and `cmfrec` have such functionality, but packages `cornac` and `lightfm` do not).

As this notebook compares different libraries, it follows instead the second approach, despite not being ideal.
<!-- #endregion -->

```python id="Xbu1jS7U0qTc" outputId="27242705-798f-4f3b-de97-3c0cbd446e2a"
import recometrics

X_train, X_test, users_test = \
    recometrics.split_reco_train_test(
        X, split_type="joined",
        users_test_fraction = None,
        max_test_users = 10000,
        items_test_fraction = 0.3
    )
X_test
```

<!-- #region id="S-44e1Xj0qTd" -->
## Establishing baselines

In order to determine if a personalized recommendation model is bringing value or not,
it's logical to compare such model against the simplest possible ways of making
recommendations, such as:

* Making random predictions.
* Always predicting the same score for each item regardless of the
user (non-personalized).

This section creates such baselines to compare against.
<!-- #endregion -->

```python id="EOo6dOaS0qTe" outputId="9ed87573-08fe-49f7-dc88-aa6b75794f37"
from cmfrec import MostPopular

### Random recommendations (random latent factors)
rng = np.random.default_rng(seed=1)
UserFactors_random = rng.standard_normal(size=(X_test.shape[0], 5))
ItemFactors_random = rng.standard_normal(size=(X_test.shape[1], 5))

### Non-personalized recommendations
model_baseline = MostPopular(implicit=True, user_bias=False).fit(X_train.tocoo())
item_biases = model_baseline.item_bias_
item_biases
```

<!-- #region id="41kcBfl30qTf" -->
## Fitting models

This section will fit two models from different software libraries that are based on different optimization criteria:

* The typical implicit-feedback matrix factorization model described at the beginning,
which considers all the entries in the matrix as zero or one with weights, minimizing
squared error across all of them. This is known as the "weighted regularized
matrix factorization" (WRMF) model or the implicit-ALS ("iALS") model.
* The "Bayesian Personalized Ranking" model, which instead sub-samples negative items (user-item interactions that have not been observed) and minimizes an optimization objective that approximates the goodness of the relative ranking of positive/negative items.
<!-- #endregion -->

```python id="zwnRTlK70qTg" outputId="462871db-b131-4188-ed33-9bcaddeba2fe" colab={"referenced_widgets": ["e481a386e76a4b56acf8ef89f40f87e0"]}
from implicit.als import AlternatingLeastSquares
from lightfm import LightFM

### Fitting WRMF model
wrmf = AlternatingLeastSquares(factors=50, regularization=1, random_state=123)
wrmf.fit(X_train.T)

### Fitting BPR model with WARP loss
bpr_warp = LightFM(no_components=50, loss="warp", random_state=123)
bpr_warp.fit(X_train.tocoo())
```

<!-- #region id="HsVnMnP30qTi" -->
## Calculating metrics

Finally, calculating recommendation quality metrics for all these models:
<!-- #endregion -->

```python id="qu4FiORl0qTi"
k = 5 ## Top-K recommendations to evaluate

metrics_random = recometrics.calc_reco_metrics(
    X_train[:X_test.shape[0]], X_test,
    UserFactors_random, ItemFactors_random,
    k=k, all_metrics=True
)

metrics_baseline = recometrics.calc_reco_metrics(
    X_train[:X_test.shape[0]], X_test,
    None, None, item_biases=item_biases,
    k=k, all_metrics=True
)

metrics_wrmf = recometrics.calc_reco_metrics(
    X_train[:X_test.shape[0]], X_test,
    wrmf.user_factors[:X_test.shape[0]], wrmf.item_factors,
    k=k, all_metrics=True
)

metrics_bpr_warp = recometrics.calc_reco_metrics(
    X_train[:X_test.shape[0]], X_test,
    bpr_warp.user_embeddings[:X_test.shape[0]], bpr_warp.item_embeddings,
    item_biases=bpr_warp.item_biases,
    k=k, all_metrics=True
)
```

<!-- #region id="uXcl2jJQ0qTj" -->
These metrics are by default returned as a data frame, with each user representing
a row and each metric a column - example:
<!-- #endregion -->

```python id="Kf_EDHMp0qTj" outputId="8013c28d-e0ef-466a-fc4f-f438b89654f7"
metrics_baseline.head()
```

<!-- #region id="vPBbNeER0qTk" -->
## Comparing models

In order to compare models, one can instead summarize these metrics across users:
<!-- #endregion -->

```python id="OkLEQ3o40qTl" outputId="22f30458-d9bf-4934-8f5a-df8b3dcdcd32"
all_metrics = [
    metrics_random,
    metrics_baseline,
    metrics_wrmf,
    metrics_bpr_warp
]
all_metrics = pd.concat([m.mean(axis=0).to_frame().T for m in all_metrics], axis=0)
all_metrics.index = [
    "Random",
    "Non-personalized",
    "WRMF (a.k.a. iALS)",
    "BPR-WARP"
]
all_metrics
```

<!-- #region id="A5Fy6gZT0qTm" -->
From these metrics, the better-performing model under every criteria seems to be the WRMF model (weighted regularized matrix factorization , a.k.a. implicit-ALS) from the package `implicit`, achieving significantly better results than non-personalized recommendations and than the BPR (Bayesian Personalized Ranking) model with WARP loss from the `lightfm` package.
<!-- #endregion -->
