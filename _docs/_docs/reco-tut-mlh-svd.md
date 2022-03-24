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

```python id="UV_mis-jdwLd" executionInfo={"status": "ok", "timestamp": 1628676056412, "user_tz": -330, "elapsed": 34, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import os
project_name = "reco-tut-mlh"; branch = "main"; account = "sparsh-ai"
project_path = os.path.join('/content', project_name)
```

```python id="KRGLEjqMd3dV" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628676078654, "user_tz": -330, "elapsed": 3123, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="0fbd6fd7-0120-4e9b-a166-a6cebbc0d5bc"
if not os.path.exists(project_path):
    !cp /content/drive/MyDrive/mykeys.py /content
    import mykeys
    !rm /content/mykeys.py
    path = "/content/" + project_name; 
    !mkdir "{path}"
    %cd "{path}"
    import sys; sys.path.append(path)
    !git config --global user.email "recotut@recohut.com"
    !git config --global user.name  "reco-tut"
    !git init
    !git remote add origin https://"{mykeys.git_token}":x-oauth-basic@github.com/"{account}"/"{project_name}".git
    !git pull origin "{branch}"
    !git checkout main
else:
    %cd "{project_path}"
```

```python id="HWliEWwod3dX"
!git status
```

```python id="dGCJpyjLd3dY"
!git add . && git commit -m 'commit' && git push origin "{branch}"
```

```python id="ClFYXwQcqWub" executionInfo={"status": "ok", "timestamp": 1628676134321, "user_tz": -330, "elapsed": 992, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import sys
sys.path.insert(0, './code')
```

<!-- #region id="t0QPGVexfcaM" -->
---
<!-- #endregion -->

<!-- #region id="Ws5ay6Ihpr5n" -->
# Singular Value Decomposition (SVD & SVD++)

SVD was heavily used in Netflix's Prize Competition in 2009. The grand prize of $1,000,000 was won by BellKor's Pragmatic Chaos. SVD utilizes stochastic gradient descent to attempt to decompose the original sparse matrices into lower ranking user and item factors (matrix factorization). These two matrices are then multiplied together to predict unknown values in the original sparse martix.

SVD++ adds a new  factor, the effect of implicit information instead of just the explicit information.
<!-- #endregion -->

<!-- #region id="l0Ta0GcZpr5r" -->
# Imports
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="6kXQbS-Aqjve" executionInfo={"status": "ok", "timestamp": 1628676223545, "user_tz": -330, "elapsed": 39606, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="2bf73f7d-f8d2-46ad-d652-7bd3fad828f3"
!pip install -q surprise
```

```python id="iIXubwDEtd8C" executionInfo={"status": "ok", "timestamp": 1628676223546, "user_tz": -330, "elapsed": 29, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import os
import pandas as pd
import surprise

from utils import stratified_split
import metrics
```

<!-- #region id="SVbsGYixpr5v" -->
# Prepare data
<!-- #endregion -->

<!-- #region id="HnxA2uo9pr5w" -->
## Load data
<!-- #endregion -->

```python id="YfwQBC0rxJIb" colab={"base_uri": "https://localhost:8080/", "height": 376} executionInfo={"status": "ok", "timestamp": 1628676223548, "user_tz": -330, "elapsed": 27, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="7abb1c22-55b6-4e18-9f2a-d4b408c4cae5"
fp = os.path.join('./data/bronze', 'u.data')
raw_data = pd.read_csv(fp, sep='\t', names=['userId', 'movieId', 'rating', 'timestamp'])
print(f'Shape: {raw_data.shape}')
raw_data.sample(10, random_state=123)
```

<!-- #region id="QUdJOLg6pr50" -->
## Train test split
<!-- #endregion -->

```python id="2YcvleME0GuI" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628676244221, "user_tz": -330, "elapsed": 3313, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="9ecf73c4-84dc-4530-9a27-05226c7f1744"
train_size = 0.75
train, test = stratified_split(raw_data, 'userId', train_size)

print(f'Train Shape: {train.shape}')
print(f'Test Shape: {test.shape}')
print(f'Do they have the same users?: {set(train.userId) == set(test.userId)}')
```

<!-- #region id="uTgjQl7epr54" -->
# SVD and SVD++
<!-- #endregion -->

```python id="axvmlSz_tkxb" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628676412114, "user_tz": -330, "elapsed": 167155, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="321c7be3-ada3-4475-e514-424875d11b04"
# Drop 'Timestamp' because surprise only takes dataframes with 3 columns in this order: userid, itemid, rating.
surprise_train = surprise.Dataset.load_from_df(train.drop('timestamp', axis=1), reader=surprise.Reader('ml-100k')).build_full_trainset()

# Instantiate models.
svd = surprise.SVD(random_state=0, n_factors=64, n_epochs=10, verbose=True)
svdpp = surprise.SVDpp(random_state=0, n_factors=64, n_epochs=10, verbose=True)
models = [svd, svdpp]

# Fit.
for model in models:
    model.fit(surprise_train)
```

<!-- #region id="gVuIvYE9pr57" -->
## Recommend
<!-- #endregion -->

```python id="ezdwIwhUpr57" executionInfo={"status": "ok", "timestamp": 1628676615377, "user_tz": -330, "elapsed": 203274, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
all_preds = []
for model in models:
    # Predict ratings for ALL movies for all users
    predictions = []
    users = train['userId'].unique()
    items = train['movieId'].unique()

    for user in users:
            for item in items:
                predictions.append([user, item, model.predict(user, item).est])
    
    predictions = pd.DataFrame(predictions, columns=['userId', 'movieId', 'prediction'])
    
    # Remove movies already seen by users
    # Create column of all 1s
    temp = train[['userId', 'movieId']].copy()
    temp['seen'] = 1

    # Outer join and remove movies that have alread been seen (seen=1)
    merged = pd.merge(temp, predictions, on=['userId', 'movieId'], how="outer")
    merged = merged[merged['seen'].isnull()].drop('seen', axis=1)
    
    all_preds.append(merged)
```

```python id="Royf4K0jpr58" executionInfo={"status": "ok", "timestamp": 1628676616911, "user_tz": -330, "elapsed": 1551, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
recommendations = []
for predictions in all_preds:
    # Create filter for users that appear in both the train and test set
    common_users = set(test['userId']).intersection(set(predictions['userId']))
    
    # Filter the test and predictions so they have the same users between them
    test_common = test[test['userId'].isin(common_users)]
    svd_pred_common = predictions[predictions['userId'].isin(common_users)]
    
    if len(set(predictions['userId'])) != len(set(test['userId'])):
        print('Number of users in train and test are NOT equal')
        print(f"# of users in train and test respectively: {len(set(predictions['userId']))}, {len(set(test['userId']))}")
        print(f"# of users in BOTH train and test: {len(set(svd_pred_common['userId']))}")
        continue
        
    # From the predictions, we want only the top k for each user,
    # not all the recommendations.
    # Extract the top k recommendations from the predictions
    top_movies = svd_pred_common.groupby('userId', as_index=False).apply(lambda x: x.nlargest(10, 'prediction')).reset_index(drop=True)
    top_movies['rank'] = top_movies.groupby('userId', sort=False).cumcount() + 1
    
    recommendations.append(top_movies)
```

<!-- #region id="p7XS4HTJpr59" -->
# Evaluation metrics

We see how SVD++ performs better than normal SVD in all metrics.
<!-- #endregion -->

```python id="xj8s-nNnpr59" executionInfo={"status": "ok", "timestamp": 1628676616913, "user_tz": -330, "elapsed": 21, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
model_metrics = {'svd':{}, 'svd++':{}}
for recommendation, model in zip(recommendations, model_metrics):
    # Create column with the predicted movie's rank for each user.
    top_k = recommendation.copy()
    top_k['rank'] = recommendation.groupby('userId', sort=False).cumcount() + 1  # For each user, only include movies recommendations that are also in the test set
    
    # Metrics.
    precision_at_k = metrics.precision_at_k(top_k, test, 'userId', 'movieId', 'rank')
    recall_at_k = metrics.recall_at_k(top_k, test, 'userId', 'movieId', 'rank')
    mean_average_precision = metrics.mean_average_precision(top_k, test, 'userId', 'movieId', 'rank')
    ndcg = metrics.ndcg(top_k, test, 'userId', 'movieId', 'rank')

    model_metrics[model]['precision'] = precision_at_k
    model_metrics[model]['recall'] = recall_at_k
    model_metrics[model]['MAP'] = mean_average_precision
    model_metrics[model]['NDCG'] = ndcg
```

```python id="g1mqYD7C3MiB" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628676616915, "user_tz": -330, "elapsed": 20, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="1c581b93-04c7-4d0a-b40f-a0c91e608f59"
for model, values in model_metrics.items():
    print(f'------ {model} -------',
          f'Precision: {values["precision"]:.6f}',
          f'Recall: {values["recall"]:.6f}',
          f'MAP: {values["MAP"]:.6f} ',
          f'NDCG: {values["NDCG"]:.6f}',
          '', sep='\n')
```
