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

<!-- #region id="GZLHDPT7RTqF" -->
# Incremental Matrix Factorization on ML-100k using River library
<!-- #endregion -->

<!-- #region id="m0akdB4nRYML" -->
## Setup
<!-- #endregion -->

```python id="It-vcW2kIjMr"
!pip install -U river numpy
```

<!-- #region id="XifIn1BoVT7e" -->
Restart the session at this point!
<!-- #endregion -->

```python id="sw24c-Z0JpdE"
import json
import river
from river.evaluate import progressive_val_score
```

<!-- #region id="jzdwEeHDVYfI" -->
## Data
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="yWmyT_dDVZ2Y" executionInfo={"status": "ok", "timestamp": 1635163478992, "user_tz": -330, "elapsed": 1244, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="53639655-0df0-4119-d8be-4d9b70892bc0"
!wget -q --show-progress https://github.com/sparsh-ai/model-retraining/raw/main/data/bronze/ml_100k.csv
```

```python id="3J9ke37XjTh-"
def get_data_stream():
    data_stream = river.stream.iter_csv('ml_100k.csv',
                                        target="rating",
                                        delimiter="\t",
                                        converters={
                                            "timestamp": int,
                                            "release_date": int,
                                            "age": float,
                                            "rating": float,
                                        })
    return data_stream 
```

```python colab={"base_uri": "https://localhost:8080/"} id="CMmCuzL2jw3I" executionInfo={"status": "ok", "timestamp": 1635163502404, "user_tz": -330, "elapsed": 781, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="e8b9180a-4866-45bd-9272-e96c43a6d652"
for x, y in get_data_stream():
    print(f'x = {json.dumps(x, indent=4)}\ny = {y}')
    break
```

<!-- #region id="SGeFxD0Nj39U" -->
Let's define a routine to evaluate our different models on MovieLens 100K. Mean Absolute Error and Root Mean Squared Error will be our metrics printed alongside model's computation time and memory usage:
<!-- #endregion -->

```python id="Komj40c3j5rF"
def evaluate(model):
    X_y = get_data_stream()
    metric = river.metrics.MAE() + river.metrics.RMSE()
    _ = progressive_val_score(X_y, model, metric, print_every=25_000, show_time=True, show_memory=True)
```

<!-- #region id="NlElS4wAod4w" -->
## Naive prediction
<!-- #endregion -->

<!-- #region id="b-B8KiIQogjf" -->
It's good practice in machine learning to start with a naive baseline and then iterate from simple things to complex ones observing progress incrementally. Let's start by predicing the target running mean as a first shot:


<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="LE1p044Loq9S" executionInfo={"status": "ok", "timestamp": 1635163509164, "user_tz": -330, "elapsed": 2100, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="648b01c1-6fcf-47d8-b8af-508a3b24e89b"
mean = river.stats.Mean()
metric = river.metrics.MAE() + river.metrics.RMSE()

for i, x_y in enumerate(get_data_stream(), start=1):
    _, y = x_y
    metric.update(y, mean.get())
    mean.update(y)

    if not i % 25_000:
        print(f'[{i:,d}] {metric}')
```

<!-- #region id="1ZMTNzTotngT" -->
## Parameters
<!-- #endregion -->

<!-- #region id="bl8vnJkFto1C" -->
let's review the important parameters to tune when dealing with this family of methods:

- n_factors: the number of latent factors. The more you set, the more items aspects and users preferences you are going to learn. Too many will cause overfitting, l2 regularization could help.
- *_optimizer: the optimizers. Classic stochastic gradient descent performs well, finding the good learning rate will make the difference.
- initializer: the latent weights initialization. Latent vectors have to be initialized with non-constant values. We generally sample them from a zero-mean normal distribution with small standard deviation.
<!-- #endregion -->

<!-- #region id="n7dSQ7vqptqI" -->
## Baseline model
<!-- #endregion -->

<!-- #region id="eK4MEPR0qCr1" -->
Now we can do machine learning and explore available models in river.reco module starting with the baseline model. It extends our naive prediction by adding to the global running mean two bias terms characterizing the user and the item discrepancy from the general tendency. This baseline model can be viewed as a linear regression where the intercept is replaced by the target running mean with the users and the items one hot encoded.

All machine learning models in river expect dicts as input with feature names as keys and feature values as values. Specifically, models from river.reco expect a 'user' and an 'item' entries without any type constraint on their values (i.e. can be strings or numbers). Other entries, if exist, are simply ignored. This is quite useful as we don't need to spend time and storage doing one hot encoding.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="aqGrK2mUqFSZ" executionInfo={"status": "ok", "timestamp": 1630505280987, "user_tz": -330, "elapsed": 5870, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="6a25accf-a33d-4932-8b40-035505ff96f7"
baseline_params = {
    'optimizer': river.optim.SGD(0.025),
    'l2': 0.,
    'initializer': river.optim.initializers.Zeros()
}

model = river.meta.PredClipper(
    regressor=river.reco.Baseline(**baseline_params),
    y_min=1,
    y_max=5
)

evaluate(model)
```

<!-- #region id="ZPwZh_-2vDTl" -->
## Matrix Factorization
<!-- #endregion -->

<!-- #region id="bTRwnZ4cq2Ak" -->
### Funk Matrix Factorization (FunkMF)
<!-- #endregion -->

<!-- #region id="u1J08n5xsWQm" -->
It's the pure form of matrix factorization consisting of only learning the users and items latent representations. Simon Funk popularized its stochastic gradient descent optimization in 2006 during the Netflix Prize. FunkMF is sometimes referred as Probabilistic Matrix Factorization which is an extended probabilistic version.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="uVM_icBosiQp" executionInfo={"status": "ok", "timestamp": 1630505461379, "user_tz": -330, "elapsed": 9948, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="431f7e59-283d-4bb4-9fa2-f8b118decb21"
funk_mf_params = {
    'n_factors': 10,
    'optimizer': river.optim.SGD(0.05),
    'l2': 0.1,
    'initializer': river.optim.initializers.Normal(mu=0., sigma=0.1, seed=73)
}

model = river.meta.PredClipper(
    regressor = river.reco.FunkMF(**funk_mf_params),
    y_min=1,
    y_max=5
)

evaluate(model)
```

<!-- #region id="lo6llniWsrv2" -->
Results are equivalent to our naive prediction (0.9448 vs 0.9421). By only focusing on the users preferences and the items characteristics, the model is limited in his ability to capture different views of the problem. Despite its poor performance alone, this algorithm is quite useful combined in other models or when we need to build dense representations for other tasks.
<!-- #endregion -->

<!-- #region id="YeGHoup8tFDz" -->
### Biased Matrix Factorization (BiasedMF)
It's the combination of the Baseline model and FunkMF. Biased Matrix Factorization name is used by some people but some others refer to it by SVD or Funk SVD. It's the case of Yehuda Koren and Robert Bell in Recommender Systems Handbook (Chapter 5 Advances in Collaborative Filtering) and of surprise library. Nevertheless, SVD could be confused with the original Singular Value Decomposition from which it's derived from, and Funk SVD could also be misleading because of the biased part of the model equation which doesn't come from Simon Funk's work. For those reasons, we chose to side with Biased Matrix Factorization which fits more naturally to it.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="fjYfzdJctT0s" executionInfo={"status": "ok", "timestamp": 1630505661712, "user_tz": -330, "elapsed": 10811, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="d4524c7e-8308-4698-ebd2-d900bebe579b"
biased_mf_params = {
    'n_factors': 10,
    'bias_optimizer': river.optim.SGD(0.025),
    'latent_optimizer': river.optim.SGD(0.05),
    'weight_initializer': river.optim.initializers.Zeros(),
    'latent_initializer': river.optim.initializers.Normal(mu=0., sigma=0.1, seed=73),
    'l2_bias': 0.,
    'l2_latent': 0.
}

model = river.meta.PredClipper(
    regressor = river.reco.BiasedMF(**biased_mf_params),
    y_min=1,
    y_max=5
)

evaluate(model)
```

<!-- #region id="FI7KDXJytcdW" -->
Results improved (0.7485 vs 0.7546) demonstrating that users and items latent representations bring additional information.
<!-- #endregion -->

<!-- #region id="lKUG4EHRuZO5" -->
## Factorization Machines
Steffen Rendel came up in 2010 with Factorization Machines, an algorithm able to handle any real valued feature vector, combining the advantages of general predictors with factorization models. It became quite popular in the field of online advertising, notably after winning several Kaggle competitions. The modeling technique starts with a linear regression to capture the effects of each variable individually. 

Then are added interaction terms to learn features relations. Instead of learning a single and specific weight per interaction (as in polynomial regression), a set of latent factors is learnt per feature (as in MF). An interaction is calculated by multiplying involved features product with their latent vectors dot product. The degree of factorization — or model order — represents the maximum number of features per interaction considered.

Strong emphasis must be placed on feature engineering as it allows FM to mimic most factorization models and significantly impact its performance. High cardinality categorical variables one hot encoding is the most frequent step before feeding the model with data. For more efficiency, river FM implementation considers string values as categorical variables and automatically one hot encode them. FM models have their own module ```river.facto```.
<!-- #endregion -->

<!-- #region id="IInnUCXFvrDT" -->
### Mimic Biased Matrix Factorization (BiasedMF)
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="EWh1rRqtu6zE" executionInfo={"status": "ok", "timestamp": 1630506217489, "user_tz": -330, "elapsed": 28155, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="6acae318-dc2d-4d7d-eab4-988a730b0b17"
fm_params = {
    'n_factors': 10,
    'weight_optimizer': river.optim.SGD(0.025),
    'latent_optimizer': river.optim.SGD(0.05),
    'sample_normalization': False,
    'l1_weight': 0.,
    'l2_weight': 0.,
    'l1_latent': 0.,
    'l2_latent': 0.,
    'intercept': 3,
    'intercept_lr': .01,
    'weight_initializer': river.optim.initializers.Zeros(),
    'latent_initializer': river.optim.initializers.Normal(mu=0., sigma=0.1, seed=73),
}

regressor = river.compose.Select('user', 'item')
regressor |= river.facto.FMRegressor(**fm_params)

model = river.meta.PredClipper(
    regressor=regressor,
    y_min=1,
    y_max=5
)

evaluate(model)
```

<!-- #region id="Xkuh5fLJvdph" -->
Both MAE are very close to each other (0.7486 vs 0.7485) showing that we almost reproduced reco.BiasedMF algorithm. The cost is a naturally slower running time as FM implementation offers more flexibility.
<!-- #endregion -->

<!-- #region id="wAw-3iwEvz2x" -->
### Feature engineering for FM models
<!-- #endregion -->

```python id="8u2s-zO4zd_T"
def split_genres(x):
    genres = x['genres'].split(', ')
    return {f'genre_{genre}': 1 / len(genres) for genre in genres}
    

def bin_age(x):
    if x['age'] <= 18:
        return {'age_0-18': 1}
    elif x['age'] <= 32:
        return {'age_19-32': 1}
    elif x['age'] < 55:
        return {'age_33-54': 1}
    else:
        return {'age_55-100': 1}
```

```python colab={"base_uri": "https://localhost:8080/"} id="CcWdtd4Pv2ME" executionInfo={"status": "ok", "timestamp": 1630507305520, "user_tz": -330, "elapsed": 61569, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="03855c42-e7e7-4136-ce55-20d54c7d45a7"
fm_params = {
    'n_factors': 14,
    'weight_optimizer': river.optim.SGD(0.01),
    'latent_optimizer': river.optim.SGD(0.025),
    'intercept': 3,
    'latent_initializer': river.optim.initializers.Normal(mu=0., sigma=0.05, seed=73),
}

regressor = river.compose.Select('user', 'item')
regressor += (
    river.compose.Select('genres') |
    river.compose.FuncTransformer(split_genres)
)
regressor += (
    river.compose.Select('age') |
    river.compose.FuncTransformer(bin_age)
)
regressor |= river.facto.FMRegressor(**fm_params)

model = river.meta.PredClipper(
    regressor=regressor,
    y_min=1,
    y_max=5
)

evaluate(model)
```

<!-- #region id="xKlpR-0izg-m" -->
### Higher-Order Factorization Machines (HOFM)
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="jPAmK0pbzyJC" executionInfo={"status": "ok", "timestamp": 1630507708525, "user_tz": -330, "elapsed": 275003, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="83cada53-2c99-4391-f58e-9aea8d07ee94"
hofm_params = {
    'degree': 3,
    'n_factors': 12,
    'weight_optimizer': river.optim.SGD(0.01),
    'latent_optimizer': river.optim.SGD(0.025),
    'intercept': 3,
    'latent_initializer': river.optim.initializers.Normal(mu=0., sigma=0.05, seed=73),
}

regressor = river.compose.Select('user', 'item')
regressor += (
    river.compose.Select('genres') |
    river.compose.FuncTransformer(split_genres)
)
regressor += (
    river.compose.Select('age') |
    river.compose.FuncTransformer(bin_age)
)
regressor |= river.facto.HOFMRegressor(**hofm_params)

model = river.meta.PredClipper(
    regressor=regressor,
    y_min=1,
    y_max=5
)

evaluate(model)
```

<!-- #region id="I6p052um0Prv" -->
High-order interactions are often hard to estimate due to too much sparsity, that's why we won't spend too much time here.
<!-- #endregion -->

<!-- #region id="k4iZ6JzG0Z89" -->
### Field-aware Factorization Machines (FFM)
Field-aware variant of FM (FFM) improved the original method by adding the notion of "fields". A "field" is a group of features that belong to a specific domain (e.g. the "users" field, the "items" field, or the "movie genres" field).

FFM restricts itself to pairwise interactions and factorizes separated latent spaces — one per combination of fields (e.g. users/items, users/movie genres, or items/movie genres) — instead of a common one shared by all fields. Therefore, each feature has one latent vector per field it can interact with — so that it can learn the specific effect with each different field.
<!-- #endregion -->

<!-- #region id="X32Q02Vo1iSy" -->
Note that FFM usually needs to learn smaller number of latent factors than FM as each latent vector only deals with one field.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="pNY_JYWG0rZM" executionInfo={"status": "ok", "timestamp": 1630507857712, "user_tz": -330, "elapsed": 85809, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="a7d91a45-3af6-456d-ef99-198aecb7d945"
ffm_params = {
    'n_factors': 8,
    'weight_optimizer': river.optim.SGD(0.01),
    'latent_optimizer': river.optim.SGD(0.025),
    'intercept': 3,
    'latent_initializer': river.optim.initializers.Normal(mu=0., sigma=0.05, seed=73),
}

regressor = river.compose.Select('user', 'item')
regressor += (
    river.compose.Select('genres') |
    river.compose.FuncTransformer(split_genres)
)
regressor += (
    river.compose.Select('age') |
    river.compose.FuncTransformer(bin_age)
)
regressor |= river.facto.FFMRegressor(**ffm_params)

model = river.meta.PredClipper(
    regressor=regressor,
    y_min=1,
    y_max=5
)

evaluate(model)
```

<!-- #region id="m1Sj4JNo1yH4" -->
### Field-weighted Factorization Machines (FwFM)
Field-weighted Factorization Machines (FwFM) address FFM memory issues caused by its large number of parameters, which is in the order of feature number times field number. As FFM, FwFM is an extension of FM restricted to pairwise interactions, but instead of factorizing separated latent spaces, it learns a specific weight for each field combination modelling the interaction strength.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="eU2b7vvc2D95" executionInfo={"status": "ok", "timestamp": 1630508080551, "user_tz": -330, "elapsed": 104684, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="8b0dbbcd-c134-4f2f-944b-fab60572af6f"
fwfm_params = {
    'n_factors': 10,
    'weight_optimizer': river.optim.SGD(0.01),
    'latent_optimizer': river.optim.SGD(0.025),
    'intercept': 3,
    'seed': 73,
}

regressor = river.compose.Select('user', 'item')
regressor += (
    river.compose.Select('genres') |
    river.compose.FuncTransformer(split_genres)
)
regressor += (
    river.compose.Select('age') |
    river.compose.FuncTransformer(bin_age)
)
regressor |= river.facto.FwFMRegressor(**fwfm_params)

model = river.meta.PredClipper(
    regressor=regressor,
    y_min=1,
    y_max=5
)

evaluate(model)
```
