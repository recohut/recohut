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

<!-- #region id="5MzSv49iMF1_" -->
# CF Part 7 - Explainable Matrix factorization method
> Collaborative Filtering on MovieLens Latest-small Part 7 - Finding recommendations using model based explainable matrix factorization method

- toc: true
- badges: true
- comments: true
- categories: [movie, collaborative]
- image:
<!-- #endregion -->

<!-- #region id="qEF8lEcAgDm6" -->
### How to quantify explainability ?

- Use the rating distribution within the active user’s neighborhood. 
- If many neighbors have rated the recommended item, then this can provide a basis upon which to explain the recommendations, using neighborhood style explanation mechanisms
<!-- #endregion -->

<!-- #region id="9NVH7eX8gDm8" -->
According to [(Abdollahi and Nasraoui, 2016)](https://www.researchgate.net/publication/301616080_Explainable_Matrix_Factorization_for_Collaborative_Filtering), an item $i$ is consider to be explainable for user $u$ if a considerable number of its neighbors rated item $i$. The explainability score $E_{ui}$ is the percentage of user $u$'s neighbors who have rated item $i$.

\begin{equation}
E_{ui} = \frac{|N_k^{(i)}(u)|}{|N_k(u)|},
\end{equation}

where $N_k(u)$ is the set of $k$ nearest neighbors of user $u$ and $N_k^{(i)}(u)$ is the set of user $u$'s neighbors who have rated item $i$. However, only explainable scores above an optimal threshold $\theta$ are accepted.

\begin{equation}
W_{ui} = \begin{cases} E_{ui} \text{  } if \text{  } E_{ui} > \theta \\ 0 \text{ } otherwise \end{cases},
\end{equation}
<!-- #endregion -->

<!-- #region id="xFxef1q6gDm-" -->
By including explainability weight in the training algorithm, the new objective function, to be minimized over the set of known ratings, has been formulated by [(Abdollahi and Nasraoui, 2016)](https://www.researchgate.net/publication/301616080_Explainable_Matrix_Factorization_for_Collaborative_Filtering) as:

\begin{equation}
 J = \sum_{(u,i)\in \kappa} (R_{ui} - \hat{R}_{ui})^2 +\frac{\beta}{2}(||P_u||^2 + ||Q_i||^2) + \frac{\lambda}{2}(P_u-Q_i)^2W_{ui},
\end{equation}

here, $\frac{\beta}{2}(||P_u||^2 + ||Q_i||^2)$ is the $L_2$ regularization term weighted by the coefficient $\beta$, and $\lambda$ is an explainability regularization coefficient that controls the smoothness of the new representation and tradeoff between explainability and accuracy. The idea here is that if item $i$ is explainable for user $u$, then their representations in the latent space, $Q_i$ and $P_u$, should be close to each other. Stochastic Gradient descent can be used to optimize the objectve function.

\begin{equation}
P_u \leftarrow P_u + \alpha\left(2(R_{u,i}-P_uQ_i^{\top})Q_i - \beta P_u - \lambda(P_u-Q_i)W_{ui}\right)
\end{equation}
\begin{equation}
Q_i \leftarrow Q_i + \alpha\left(2(R_{u,i}-P_uQ_i^{\top})P_u - \beta Q_i + \lambda(P_u-Q_i)W_{ui}\right)
\end{equation}
<!-- #endregion -->

<!-- #region id="C4k3EF8VMF2D" -->
## Explainable Matrix Factorization : Model definition
<!-- #endregion -->

```python id="iU7wjIFHgDnB"
import os

if not (os.path.exists("recsys.zip") or os.path.exists("recsys")):
    !wget https://github.com/nzhinusoftcm/review-on-collaborative-filtering/raw/master/recsys.zip    
    !unzip recsys.zip
```

<!-- #region id="TR5BlBqGgDnH" -->
### requirements

```
matplotlib==3.2.2
numpy==1.18.1
pandas==1.0.5
python==3.6.10
scikit-learn==0.23.1
scipy==1.5.0
```
<!-- #endregion -->

```python id="x_ohlWURMF2F"
from recsys.memories.UserToUser import UserToUser

from recsys.preprocessing import mean_ratings
from recsys.preprocessing import normalized_ratings
from recsys.preprocessing import ids_encoder
from recsys.preprocessing import train_test_split
from recsys.preprocessing import rating_matrix
from recsys.preprocessing import get_examples

from recsys.datasets import ml100k, ml1m

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import sys
import os
```

<!-- #region id="r8AwCSWVgDnp" -->
### Compute Explainable Scores

Explainable score are computed using neighborhood based similarities. Here, we are using the user based algorithme to compute similarities
<!-- #endregion -->

```python id="gYEoCr_RgDnr"
def explainable_score(user2user, users, items, theta=0):
    
    def _progress(count):
        sys.stdout.write('\rCompute Explainable score. Progress status : %.1f%%'%(float(count/len(users))*100.0))
        sys.stdout.flush()
    # initialize explainable score to zeros
    W = np.zeros((len(users), len(items)))

    for count, u in enumerate(users):            
        candidate_items = user2user.find_user_candidate_items(u)        
        for i in candidate_items:                
            user_who_rated_i, similar_user_who_rated_i = \
                user2user.similar_users_who_rated_this_item(u, i)
            if user_who_rated_i.shape[0] == 0:
                w = 0.0
            else:
                w = similar_user_who_rated_i.shape[0] / user_who_rated_i.shape[0]
            W[u,i] =  w  if w > theta else 0.0
        _progress(count)
    return W
```

<!-- #region id="TH6uvGObgDns" -->
### Explainable Matrix Factorization Model
<!-- #endregion -->

```python id="O2bSuu0jgDnw"
class ExplainableMatrixFactorization:
    
    def __init__(self, m, n, W, alpha=0.001, beta=0.01, lamb=0.1, k=10):
        """
            - R : Rating matrix of shape (m,n) 
            - W : Explainability Weights of shape (m,n)
            - k : number of latent factors
            - beta : L2 regularization parameter
            - lamb : explainability regularization coefficient
            - theta : threshold above which an item is explainable for a user
        """
        self.W = W
        self.m = m
        self.n = n
        
        np.random.seed(64)
        
        # initialize the latent factor matrices P and Q (of shapes (m,k) and (n,k) respectively) that will be learnt
        self.k = k
        self.P = np.random.normal(size=(self.m,k))
        self.Q = np.random.normal(size=(self.n,k))
        
        # hyperparameter initialization
        self.alpha = alpha
        self.beta = beta
        self.lamb = lamb
        
        # training history
        self.history = {
            "epochs":[],
            "loss":[],
            "val_loss":[],
        }
        
    def print_training_parameters(self):
        print('Training EMF')
        print(f'k={self.k} \t alpha={self.alpha} \t beta={self.beta} \t lambda={self.lamb}')
        
    def update_rule(self, u, i, error):
        self.P[u] = self.P[u] + \
            self.alpha*(2 * error*self.Q[i] - self.beta*self.P[u] - self.lamb*(self.P[u] - self.Q[i]) * self.W[u,i])
        
        self.Q[i] = self.Q[i] + \
            self.alpha*(2 * error*self.P[u] - self.beta*self.Q[i] + self.lamb*(self.P[u] - self.Q[i]) * self.W[u,i])
        
    def mae(self,  x_train, y_train):
        """
        returns the Mean Absolute Error
        """
        # number of training exemples
        M = x_train.shape[0]
        error = 0
        for pair, r in zip(x_train, y_train):
            u, i = pair
            error += np.absolute(r - np.dot(self.P[u], self.Q[i]))
        return error/M
    
    def print_training_progress(self, epoch, epochs, error, val_error, steps=5):
        if epoch == 1 or epoch % steps == 0 :
                print(f"epoch {epoch}/{epochs} - loss : {round(error,3)} - val_loss : {round(val_error,3)}")
                
    def learning_rate_schedule(self, epoch, target_epochs = 20):
        if (epoch >= target_epochs) and (epoch % target_epochs == 0):
                factor = epoch // target_epochs
                self.alpha = self.alpha * (1 / (factor * 20))
                print("\nLearning Rate : {}\n".format(self.alpha))
        
    def fit(self, x_train, y_train, validation_data, epochs=10):
        """
        Train latent factors P and Q according to the training set
        
        :param
            - x_train : training pairs (u,i) for which rating r_ui is known
            - y_train : set of ratings r_ui for all training pairs (u,i)
            - validation_data : tuple (x_test, y_test)
            - epochs : number of time to loop over the entire training set. 
            10 epochs by default
            
        Note that u and i are encoded values of userid and itemid
        """
        self.print_training_parameters()
        
        # get validation data
        x_test, y_test = validation_data
        
        for epoch in range(1, epochs+1):
            for pair, r in zip(x_train, y_train):                
                u,i = pair                
                r_hat = np.dot(self.P[u], self.Q[i])                
                e = r - r_hat
                self.update_rule(u, i, error=e)
                
            # training and validation error  after this epochs
            error = self.mae(x_train, y_train)
            val_error = self.mae(x_test, y_test)
            self.update_history(epoch, error, val_error)            
            self.print_training_progress(epoch, epochs, error, val_error, steps=1)
        
        return self.history
    
    def update_history(self, epoch, error, val_error):
        self.history['epochs'].append(epoch)
        self.history['loss'].append(error)
        self.history['val_loss'].append(val_error)
    
    def evaluate(self, x_test, y_test):
        """
        compute the global error on the test set
        
        :param
            - x_test : test pairs (u,i) for which rating r_ui is known
            - y_test : set of ratings r_ui for all test pairs (u,i)
        """
        error = self.mae(x_test, y_test)
        print(f"validation error : {round(error,3)}")
      
    def predict(self, userid, itemid):
        """
        Make rating prediction for a user on an item

        :param
        - userid
        - itemid

        :return
        - r : predicted rating
        """
        # encode user and item ids to be able to access their latent factors in
        # matrices P and Q
        u = uencoder.transform([userid])[0]
        i = iencoder.transform([itemid])[0]

        # rating prediction using encoded ids. Dot product between P_u and Q_i
        r = np.dot(self.P[u], self.Q[i])

        return r

    def recommend(self, userid, N=30):
        """
        make to N recommendations for a given user

        :return 
        - (top_items,preds) : top N items with the highest predictions 
        """
        # encode the userid
        u = uencoder.transform([userid])[0]

        # predictions for this user on all product
        predictions = np.dot(self.P[u], self.Q.T)

        # get the indices of the top N predictions
        top_idx = np.flip(np.argsort(predictions))[:N]

        # decode indices to get their corresponding itemids
        top_items = iencoder.inverse_transform(top_idx)

        # take corresponding predictions for top N indices
        preds = predictions[top_idx]

        return top_items, preds
```

```python id="f8WgfBjPgDnz"
epochs = 10
```

<!-- #region id="-9IDXOqRgDn1" -->
# Model Evaluation

## 1. MovieLens 100K

### 1.1. Evaluation on raw data
<!-- #endregion -->

```python id="zVfuOHTPgDn3"
# load data
ratings, movies = ml100k.load()

# encode users and items ids
ratings, uencoder, iencoder = ids_encoder(ratings)

users = sorted(ratings.userid.unique())
items = sorted(ratings.itemid.unique())

m = len(users)
n = len(items)

# get examples as tuples of userids and itemids and labels from normalize ratings
raw_examples, raw_labels = get_examples(ratings)

# train test split
(x_train, x_test), (y_train, y_test) = train_test_split(examples=raw_examples, labels=raw_labels)
```

```python id="kJ9MFwmdgDn4" outputId="25d631d3-6f45-447c-96b0-8ef5e8ab2fdb"
# create the user to user model for similarity measure
usertouser = UserToUser(ratings, movies)

# compute explainable score
W = explainable_score(usertouser, users, items)
```

```python id="CQGoZQ1ogDoU" outputId="c31487e4-be7b-4b3e-a182-c9077982ad81"
# initialize the model
EMF = ExplainableMatrixFactorization(m, n, W, alpha=0.01, beta=0.4, lamb=0.01, k=10)

history = EMF.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test))
```

```python colab={"base_uri": "https://localhost:8080/", "height": 35} id="Bc6F0HWMWbLd" outputId="8e271e4a-ca53-4d05-eac3-90d18baf4ba4"
EMF.evaluate(x_test, y_test)
```

<!-- #region id="HP7FBr2HgDoe" -->
### 1.2. Evaluation on normalized data
<!-- #endregion -->

```python id="Mp1XcoWTgDog"
# load data
ratings, movies = ml100k.load()

# encode users and items ids
ratings, uencoder, iencoder = ids_encoder(ratings)

users = sorted(ratings.userid.unique())
items = sorted(ratings.itemid.unique())

m = len(users)
n = len(items)

# normalize ratings by substracting means
normalized_column_name = "norm_rating"
ratings = normalized_ratings(ratings, norm_column=normalized_column_name)

# get examples as tuples of userids and itemids and labels from normalize ratings
raw_examples, raw_labels = get_examples(ratings, labels_column=normalized_column_name)

# train test split
(x_train, x_test), (y_train, y_test) = train_test_split(examples=raw_examples, labels=raw_labels)
```

```python id="nKL1O2SsgDoh" outputId="d00bd004-c568-4f84-adea-988b9ff145c1"
# initialize the model
EMF = ExplainableMatrixFactorization(m, n, W, alpha=0.022, beta=0.65, lamb=0.01, k=10)

history = EMF.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test))
```

<!-- #region id="O1IYzbV3gDoi" -->
## 2. MovieLens 1M

### 2.1. Evaluation on raw data
<!-- #endregion -->

```python id="o9ozKw3ogDoj"
# load data
ratings, movies = ml1m.load()

# encode users and items ids
ratings, uencoder, iencoder = ids_encoder(ratings)

users = sorted(ratings.userid.unique())
items = sorted(ratings.itemid.unique())

m = len(users)
n = len(items)

# get examples as tuples of userids and itemids and labels from normalize ratings
raw_examples, raw_labels = get_examples(ratings)

# train test split
(x_train, x_test), (y_train, y_test) = train_test_split(examples=raw_examples, labels=raw_labels)
```

```python id="h9IBKiwAgDok" outputId="1163ad2a-9403-43bc-b015-3c2263da5eb7"
# create the user to user model for similarity measure
usertouser = UserToUser(ratings, movies)

# compute explainable score
W = explainable_score(usertouser, users, items)
```

```python id="QOyz8rXzgDor" outputId="ed3142e5-5f69-49df-d458-583095db531e"
# initialize the model
EMF = ExplainableMatrixFactorization(m, n, W, alpha=0.01, beta=0.4, lamb=0.01, k=10)

history = EMF.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test))
```

<!-- #region id="VtD4fCSigDot" -->
### 2.2. Evaluation on normalized data
<!-- #endregion -->

```python id="FEWsFmKNgDot"
# load data
ratings, movies = ml1m.load()

# encode users and items ids
ratings, uencoder, iencoder = ids_encoder(ratings)

# normalize ratings by substracting means
normalized_column_name = "norm_rating"
ratings = normalized_ratings(ratings, norm_column=normalized_column_name)

# get examples as tuples of userids and itemids and labels from normalize ratings
raw_examples, raw_labels = get_examples(ratings, labels_column=normalized_column_name)

# train test split
(x_train, x_test), (y_train, y_test) = train_test_split(examples=raw_examples, labels=raw_labels)
```

```python id="1pUkEWRogDo3" outputId="8b63f85c-3ad8-4c01-b283-cf5ee7a518db"
# initialize the model
EMF = ExplainableMatrixFactorization(m, n, W, alpha=0.023, beta=0.59, lamb=0.01, k=10)

history = EMF.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test))
```

<!-- #region id="_K9PoEXsgDo3" -->
# Ratings prediction
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 979} id="M5FpA7T3uA2v" outputId="35730e8b-e6d8-4f56-f6f4-161e09128412"
# get list of top N items with their corresponding predicted ratings
userid = 42
recommended_items, predictions = EMF.recommend(userid=userid)

# find corresponding movie titles
top_N = list(zip(recommended_items,predictions))
top_N = pd.DataFrame(top_N, columns=['itemid','predictions'])
top_N.predictions = top_N.predictions + ratings.loc[ratings.userid==userid].rating_mean.values[0]
List = pd.merge(top_N, movies, on='itemid', how='inner')

# show the list
List
```

<!-- #region id="ZPeVMap_wu2T" -->
**Note**: The recommendation list may content items already purchased by the user. This is just an illustration of how to implement matrix factorization recommender system. You can optimize the recommended list and return the top rated items that the user has not already purchased.
<!-- #endregion -->

<!-- #region id="FP65IDNNgDo7" -->
## Putting all together

This repository covers the following algorithms for collaborative filtering

1. User-based CF
2. Item-based CF
3. Singular Value Decomposition (SVD)
4. Matrix Factorization (MF)
5. Non-negative Matrix Factorization (NMF)
6. Explainable Matrix Factorization (EMF)

In the [next notebook](https://github.com/nzhinusoftcm/review-on-collaborative-filtering/blob/master/8.Performances_measure.ipynb), we present performances comparison of all these models.
<!-- #endregion -->

<!-- #region id="B-jjxDhex5Gs" -->
## Reference

1. Yehuda Koren et al. (2009). <a href='https://ieeexplore.ieee.org/document/5197422'>Matrix Factorization Techniques for Recommender Systems</a>
2. Abdollahi and Nasraoui (2016). [Explainable Matrix Factorization for Collaborative Filtering](https://www.researchgate.net/publication/301616080_Explainable_Matrix_Factorization_for_Collaborative_Filtering)
3. Abdollahi and Nasraoui (2017). [Using Explainability for Constrained Matrix Factorization](https://dl.acm.org/doi/abs/10.1145/3109859.3109913)
4. Shuo Wang et al, (2018). [Explainable Matrix Factorization with Constraints on Neighborhood in the Latent Space](https://dl.acm.org/doi/abs/10.1145/3109859.3109913)
<!-- #endregion -->

<!-- #region id="jV3Jj17ox_UR" -->
## Author

[Carmel WENGA](https://www.linkedin.com/in/carmel-wenga-871876178/), <br>
PhD student at Université de la Polynésie Française, <br> 
Applied Machine Learning Research Engineer, <br>
[ShoppingList](https://shoppinglist.cm), NzhinuSoft.
<!-- #endregion -->
