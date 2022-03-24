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

<!-- #region id="hZzLBRl0iNfE" -->
# Movie Recommender System
<!-- #endregion -->

<!-- #region id="g0mwPHs1itx9" -->
## Collaborative filtering (matrix factorization)

You are an online retailer/travel agent/movie review website, and you would like to help the visitors of your website to explore more of your products/destinations/movies. You got data which either describe the different products/destinations/films, or past transactions/trips/views (or preferences) of your visitors (or both!). You decide to leverage that data to provide relevant and meaningful recommendations.

This notebook implements a simple collaborative system using  factorization of the user-item matrix.
<!-- #endregion -->

```python id="J8u8ZXvkhkfY"
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
```

```python id="CPVCIslerd5e"
ratings="https://github.com/couturierc/tutorials/raw/master/recommender_system/data/ratings.csv"
movies="https://github.com/couturierc/tutorials/raw/master/recommender_system/data/movies.csv"

# If data stored locally
# ratings="./data/ratings.csv"
# movies="./data/movies.csv"

df_ratings = pd.read_csv(ratings, sep=',')
df_ratings.columns = ['userId', 'itemId', 'rating', 'timestamp']
df_movies = pd.read_csv(movies, sep=',')
df_movies.columns = ['itemId', 'title', 'genres']
```

```python id="IvyAYay5rzcS"
df_movies.head()
```

```python id="R4K08KX3sYhr"
df_ratings.head()
```

<!-- #region id="2hUKyFxYdsT5" -->
## Quick exploration

Hints: use df.describe(), df.column_name.hist(), scatterplot matrix (sns.pairplot(df[column_range])), correlation matrix (sns.heatmap(df.corr()) ), check duplicates, ...
<!-- #endregion -->

```python id="LVqBtDNmJ5vL"
# Start your exploration -- use as many cells as you need !

```

<!-- #region id="MffuKcE5s8fQ" -->
## Obtain the user-item matrice by pivoting df_ratings
<!-- #endregion -->

```python id="qOt3GI3zs2Ts"
##### FILL HERE (1 line) ######
df_user_item = NULL # Use df.pivot, rows ~ userId's, columns ~ itemId's
################################

# Sort index/rows (userId's) and columns (itemId's)
df_user_item.sort_index(axis=0, inplace=True)
df_user_item.sort_index(axis=1, inplace=True)
```

<!-- #region id="90Q7L3SQtc1t" -->
This matrix has **many** missing values:
<!-- #endregion -->

```python id="P6tkf_s3tgsL"
df_user_item.head()
```

```python id="J0EfDXLIRWaG"
df_user_item.describe()
```

<!-- #region id="HXanXrqI4xJ4" -->
For instance, rating for userId=1 for movies with itemId 1 to 10:
<!-- #endregion -->

```python id="QLI0gnwT4obE"
df_user_item.loc[1][:10]
```

```python id="3SM4RU3njy2K"
# df_user_item.loc[1].dropna().sort_values(ascending=False)
```

<!-- #region id="-dtJPkm1knNC" -->
Save the movie ids for user 1 for later:
<!-- #endregion -->

```python id="C05fKcNrkmYv"
item_rated_user_1 = df_user_item.loc[1].dropna().index
item_rated_user_1
```

<!-- #region id="oR-pEwd5thyy" -->
We want to find the matrix of rank $k$ which is closest to the original matrix.


<!-- #endregion -->

<!-- #region id="gAUU_b5ma5bA" -->
## What not to do: Fill with 0's or mean values, then Singular Value Decomposition (SVD)
<!-- #endregion -->

<!-- #region id="5ixiAfGIH6VU" -->
(Adapted from https://github.com/beckernick/matrix_factorization_recommenders/blob/master/matrix_factorization_recommender.ipynb)

Singular Value Decomposition decomposes a matrix $R$ into the best lower rank (i.e. smaller/simpler) approximation of the original matrix $R$. Mathematically, it decomposes R into a two unitary matrices and a diagonal matrix:

$$\begin{equation}
R = U\Sigma V^{T}
\end{equation}$$

where: 
- R is users's ratings matrix, 
- $U$ is the user "features" matrix, it represents how much users "like" each feature,
- $\Sigma$ is the diagonal matrix of singular values (essentially weights), 
- $V^{T}$ is the movie "features" matrix, it represents how relevant each feature is to each movie,

with $U$ and $V^{T}$ orthogonal.
<!-- #endregion -->

```python id="MMVe_feVQQK_"
df_user_item = df_user_item.fillna(0)
df_user_item.head()
```

```python id="Pz16Rlw4tlom"
R = df_user_item.values
```

```python id="_R9inUPkH1Hm"
R
```

<!-- #region id="gypFSYCYHg63" -->
Apply SVD to R (e.g. using NumPy or SciPy)
<!-- #endregion -->

```python id="XGSFlWxLHYVE"
from scipy.sparse.linalg import svds
U, sigma, Vt = svds(R, k = 50)
```

<!-- #region id="slRJZ23uIVLt" -->
What do $U$, $\Sigma$, $V^T$ look like?
<!-- #endregion -->

```python id="jfifORX6IIga"
U
```

```python id="nXkKnGWcISzH"
sigma
```

```python id="v0H56AlQIUTM"
Vt
```

<!-- #region id="baQzWyVHKQVN" -->
Get recommendations:
<!-- #endregion -->

```python id="CyzbchyIKnkW"
# First make sigma a diagonal matrix:
sigma = np.diag(sigma)
```

```python id="uouELHsfKtOU"
R_after_svd = np.dot(np.dot(U, sigma), Vt)
R_after_svd
```

```python id="mFID_6eWKskb"

```

<!-- #region id="z6NRarPjJ0DI" -->
Drawbacks of this approach: 
- the missing values (here filled with 0's) is feedback that the user did not give, we should not cannot consider it negative/null rating.
- the dense matrix is huge, applying SVD is not scalable.
<!-- #endregion -->

<!-- #region id="Keb06kCFbIPl" -->
## Approximate SVD with stochastic gradient descend (SGD)


This time, we do **not** fill missing values. 

We inject $\Sigma$ into U and V, and try to find P and q such that $\widehat{R} = P Q^{T}$ is close to  $R$ **for the item-user pairs already rated**.

<!-- #endregion -->

<!-- #region id="tkr8jfzbVS_R" -->
A first function to simplify the entries (userId/itemId) : we map the set of 
<!-- #endregion -->

```python id="F_HgEkPAQSTG"
def encode_ids(data):
    '''Takes a rating dataframe and return: 
    - a simplified rating dataframe with ids in range(nb unique id) for users and movies
    - 2 mapping disctionaries
    
    '''

    data_encoded = data.copy()
    
    users = pd.DataFrame(data_encoded.userId.unique(),columns=['userId'])  # df of all unique users
    dict_users = users.to_dict()    
    inv_dict_users = {v: k for k, v in dict_users['userId'].items()}

    items = pd.DataFrame(data_encoded.itemId.unique(),columns=['itemId']) # df of all unique items
    dict_items = items.to_dict()    
    inv_dict_items = {v: k for k, v in dict_items['itemId'].items()}

    data_encoded.userId = data_encoded.userId.map(inv_dict_users)
    data_encoded.itemId = data_encoded.itemId.map(inv_dict_items)

    return data_encoded, dict_users, dict_items
  
```

<!-- #region id="Yt6SYVvAX3Di" -->
Here is the procedure we would like to implement in the function SGD():

1.   itinialize P and Q to random values

2.   for $n_{epochs}$ passes on the data:

    *   for all known ratings $r_{ui}$
        *   compute the error between the predicted rating $p_u \cdot q_i$ and the known ratings $r_{ui}$:
        $$ err = r_{ui} - p_u \cdot q_i $$
        *   update $p_u$ and $q_i$ with the following rule:
        $$ p_u \leftarrow p_u + \alpha \cdot err \cdot q_i  $$
        $$ q_i \leftarrow q_i + \alpha \cdot err \cdot p_u$$






<!-- #endregion -->

```python id="iA0tyBHJ5xyI"
# Adapted from http://nicolas-hug.com/blog/matrix_facto_4
def SGD(data,           # dataframe containing 1 user|item|rating per row
        n_factors = 10, # number of factors
        alpha = .01,    # number of factors
        n_epochs = 3,   # number of iteration of the SGD procedure
       ):
    '''Learn the vectors P and Q (ie all the weights p_u and q_i) with SGD.
    '''

    # Encoding userId's and itemId's in data
    data, dict_users, dict_items = encode_ids(data)
    
    ##### FILL HERE (2 lines) ######
    n_users = NULL  # number of unique users
    n_items = NULL  # number of unique items
    ################################
    
    # Randomly initialize the user and item factors.
    p = np.random.normal(0, .1, (n_users, n_factors))
    q = np.random.normal(0, .1, (n_items, n_factors))

    # Optimization procedure
    for epoch in range(n_epochs):
        print ('epoch: ', epoch)
        # Loop over the rows in data
        for index in range(data.shape[0]):
            row = data.iloc[[index]]
            u = int(row.userId)      # current userId = position in the p vector (thanks to the encoding)
            i = int(row.itemId)      # current itemId = position in the q vector
            r_ui = float(row.rating) # rating associated to the couple (user u , item i)
            
            ##### FILL HERE (1 line) ######
            err = NULL    # difference between the predicted rating (p_u . q_i) and the known ratings r_ui
            ################################
            
            # Update vectors p_u and q_i
            ##### FILL HERE (2 lines) ######
            p[u] = NULL  # cf. update rule above 
            q[i] = NULL
            ################################
            
    return p, q
    
    
def estimate(u, i, p, q):
    '''Estimate rating of user u for item i.'''
    ##### FILL HERE (1 line) ######
    return NULL             #scalar product of p[u] and q[i] /!\ dimensions
    ################################  
```

```python id="_MYUUm18-id6"
p, q = SGD(df_ratings)
```

<!-- #region id="qJd80gNgNuUR" -->
## Get the estimate for all user-item pairs:
<!-- #endregion -->

<!-- #region id="hj4Pc-FjPJK6" -->
Get the user-item matrix filled with predicted ratings:
<!-- #endregion -->

```python id="YRCg3k2IPMSc"
df_user_item_filled = pd.DataFrame(np.dot(p, q.transpose()))
df_user_item_filled.head()
```

<!-- #region id="LLHPMdpyN96R" -->
However, it is using the encode ids ; we need to retrieve the association of encoded ids to original ids, and apply it:
<!-- #endregion -->

```python id="cuft25TRN4CY"
df_ratings_encoded, dict_users, dict_items = encode_ids(df_ratings)
```

```python id="mCidjCrUl2tx"
df_user_item_filled.rename(columns=(dict_items['itemId']), inplace=True)
df_user_item_filled.rename(index=(dict_users['userId']), inplace=True)

# Sort index/rows (userId's) and columns (itemId's)
df_user_item_filled.sort_index(axis=0, inplace=True)
df_user_item_filled.sort_index(axis=1, inplace=True)

df_user_item_filled.head()
```

<!-- #region id="AVXIqXAdOPzX" -->
Originally available ratings for user 1:
<!-- #endregion -->

```python id="iyka6nXcOPo4"
df_user_item.loc[1][:10]
```

<!-- #region id="Pphixa2wOPeh" -->
Estimated ratings after the approximate SVD:
<!-- #endregion -->

```python id="YDczh7x5Q6in"
df_user_item_filled.loc[1][:10]
```

<!-- #region id="Uk8zB0HCmLvk" -->
## Give recommendation to a user

For instance 10 recommended movies for user 1
<!-- #endregion -->

```python id="G8zxuZ2VmaIs"
recommendations = list((df_user_item_filled.loc[10]).sort_values(ascending=False)[:10].index)
recommendations
```

```python id="5U7R7lyTuOy_"
df_movies[df_movies.itemId.isin(recommendations)]
```

<!-- #region id="3fhXmfLeuDZo" -->
vs the ones that were rated initially:
<!-- #endregion -->

```python id="4ooeCcRnuI8y"
already_rated = list((df_user_item.loc[10]).sort_values(ascending=False)[:10].index)
already_rated
```

```python id="0SM3mJYwyF1g"
df_movies[df_movies.itemId.isin(already_rated)]
```

<!-- #region id="qKarQdgbm4tw" -->
This is all the movies in descending order of predicted rating. Let's remove the ones that where alread rated.
<!-- #endregion -->

<!-- #region id="hkvVcbTALIji" -->



---



To put this into production, you'd first separate data into a training and validation set and optimize the number of latent factors (n_factors) by minimizing the Root Mean Square Error. 
It is easier to use a framework that allows to do this, do cross-validation, grid search, etc.
<!-- #endregion -->

<!-- #region id="nMdbrNdLldG9" -->
## Gradient Descent SVD using Surprise
<!-- #endregion -->

```python id="4VdMT5PnbIn9"
!pip install surprise
#!pip install scikit-surprise # if the first line does not work
```

```python id="Ed0lnuff4NOw"
# from surprise import Reader, Dataset, SVD, evaluate

# Following Surprise documentation examples 
# https://surprise.readthedocs.io/en/stable/getting_started.html

from surprise import Reader, Dataset, SVD, evaluate, NormalPredictor
from surprise.model_selection import cross_validate
from collections import defaultdict

# As we're loading a custom dataset, we need to define a reader.
reader = Reader(rating_scale=(0.5, 5))

# The columns must correspond to user id, item id and ratings (in that order).
data = Dataset.load_from_df(df_ratings[['userId', 'itemId', 'rating']], reader)

# We'll use the famous SVD algorithm.
algo = SVD()

# Run 5-fold cross-validation and print results
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
```

<!-- #region id="YyciPjWI4Q94" -->
#### Tune algorithm parameters with GridSearchCV


<!-- #endregion -->

```python id="tG3nlrAKzLZg"
from surprise.model_selection import GridSearchCV

param_grid = {'n_epochs': [5, 10], 'lr_all': [0.002, 0.005],
              'reg_all': [0.4, 0.6]}
gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)

gs.fit(data)

# best RMSE score
print(gs.best_score['rmse'])

# combination of parameters that gave the best RMSE score
print(gs.best_params['rmse'])
```

```python id="LnfvwVPvzUsw"
# We can now use the algorithm that yields the best rmse:
algo = gs.best_estimator['rmse']
trainset = data.build_full_trainset()
algo.fit(trainset)
```

```python id="JVAeYFgTzppL"
algo.predict(621,1)
```

```python id="li7UhY6fz1oG"
df_data = data.df
df_data = df_data.join(df_movies,how="left", on='itemId',rsuffix='_', lsuffix='')
df_data[df_data['userId']==1].sort_values(by = 'rating',ascending=False)[:10]
```

```python id="CRm97oJVz8wG"
# From Surprise documentation: https://surprise.readthedocs.io/en/stable/FAQ.html
def get_top_n(predictions, n=10):
    '''Return the top-N recommendation for each user from a set of predictions.

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.

    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    '''

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n
```

```python id="poADsLk634aR"
# Predict ratings for all pairs (u, i) that are NOT in the training set.
testset = trainset.build_anti_testset()
predictions = algo.test(testset)
```

```python id="zn3AViRh19eR"
top_n = get_top_n(predictions, n=10)
```

```python id="igRXlPxr4gCH"
top_n.items()
```

```python id="U2ElCZzT4EC1"
# Print the recommended items for all user 1
for uid, user_ratings in top_n.items():
    print(uid, [iid for (iid, _) in user_ratings])
    if uid == 1:
        break
```

```python id="3OVCCW1C4ziF"
df_movies[df_movies.itemId.isin([318, 750, 1204, 858, 904, 48516, 1221, 912, 1276, 4973])]
```

```python id="uNVZSfS35PSo"

```
