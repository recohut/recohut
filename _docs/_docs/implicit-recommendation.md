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

<!-- #region id="OFSvPOkId0aZ" -->
# Implicit Recommendation
<!-- #endregion -->

<!-- #region id="ZhFt8K-vHVE3" -->
### Framework
1. Data type : 'Unary & Implicit data'
  - Be careful not to confuse binary and unary data: unary data means that you have information that a user consumed something (which is coded as 1, much like binary data), but you have no information about whether a user didn't like or consume something (which is coded as NULL instead of binary data's 0).
  - Implicit data is data we gather from the users behaviour, with no ratings or specific actions needed. It could be what items a user purchased, how many times they played a song or watched a movie, how long they’ve spent reading a specific article etc. 
2. Algorithms
 1. Traditional algorithm - Item-item nearest neighbor models - a. Cosine distance metric, b. TF IDF, c.	BM25, Popularity based recommendation (baseline).
 2. ALS (Alternating Least Squares) Matrix Factorization
    - Original paper: http://yifanhu.net/PUB/cf.pdf 
    - We can use matrix factorization to mathematically reduce the dimensionality of our original “all users by all items” matrix into something much smaller that represents “all items by some taste dimensions” and “all users by some taste dimensions”. These dimensions are called latent or hidden features and we learn them from our data.
    - There are different ways to factor a matrix, like Singular Value Decomposition (SVD) or Probabilistic Latent Semantic Analysis (PLSA) if we’re dealing with explicit data.With implicit data the difference lies in how we deal with all the missing data in our very sparse matrix. For explicit data we treat them as just unknown fields that we should assign some predicted rating to. But for implicit we can’t just assume the same since there is information in these unknown values as well. As stated before we don’t know if a missing value means the user disliked something, or if it means they love it but just don’t know about it. Basically we need some way to learn from the missing data. So we’ll need a different approach to get us there.
    - <img src='https://jessesw.com/images/Rec_images/ALS_Image_Test.png' width=600>
  3. Bayesian Personalized Ranking (BPR)
     - Original paper: https://arxiv.org/ftp/arxiv/papers/1205/1205.2618.pdf
  4. Logistic Matrix Factorization
     - Original paper: http://stanford.edu/~rezab/nips2014workshop/submits/logmat.pdf
  5. Collaborative Less-Is-More Filtering
     - Original paper: https://www.ijcai.org/Proceedings/13/Papers/460.pdf 

    
<!-- #endregion -->

```python id="RNVoFwWx1Ml_"
!pip install implicit

import numpy as np
import pandas as pd
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
from sklearn import metrics
import time
import random

from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking
```

```python id="dx6FW6dxKley" outputId="2079ad12-46eb-44f5-dda5-c71cb88f18bd" colab={"base_uri": "https://localhost:8080/", "height": 204}
data = pd.read_csv('https://raw.githubusercontent.com/sparsh9012/Recommendation-Engine/master/data/data.csv')
data.head()
```

```python id="BAzjx2jtK72h" outputId="2dabdd2a-e0ec-48ff-fef6-544a249fd7c3" colab={"base_uri": "https://localhost:8080/", "height": 298}
matrix = pd.crosstab(data.customer_id, data.item_id)
sparsed = sparse.csr_matrix(matrix.values)
print('matrix shape: ',matrix.shape)
print('sparse shape: ',sparsed.shape)
matrix.head()
```

```python id="GqFo9H8bMWx9" outputId="b54fc9d6-04cf-4725-8ce9-5018da449a79" colab={"base_uri": "https://localhost:8080/", "height": 34}
#dropping customers with very-high-frequency purchase
matrix1 = matrix.loc[matrix.sum(axis=1).values<5000,:]

#dropping customers with very-low-frequency purchase
matrix1 = matrix1.loc[matrix1.sum(axis=1).values>2,:]

#dropping products with very-low-frequency purchase
matrix1 = matrix1.loc[:,matrix1.sum(axis=0).values>2]

sparsed1 = sparse.csr_matrix(matrix1.values)
matrix1.shape
sparsed1.shape
```

```python id="VOPWzjCHtV43" outputId="63270596-541d-435b-99d6-3e4dabc02ba6" colab={"base_uri": "https://localhost:8080/", "height": 51}
#check sparsity
sparsity = round(1.0 - len(data) / float(matrix.shape[0] * matrix.shape[1]), 3)
print('The sparsity level of dataset is ' +  str(sparsity * 100) + '%')

sparsity = round(1.0 - len(data) / float(matrix1.shape[0] * matrix1.shape[1]), 3)
print('The sparsity level of filtered dataset is ' +  str(sparsity * 100) + '%')
```

```python id="crEk0SRQF3vt"
item_dictionary = { i : matrix.columns[i] for i in range(0, len(matrix.columns) ) }
customer_dictionary = { i : matrix.index[i] for i in range(0, len(matrix.index) ) }
```

```python id="s8cbODa4BMtD"
def calculate_recommendations(df_new, model_name='als', factors=32, regularization=0.01, iterations=10):
    
    # initialize models
    if model_name=='als':
        model = AlternatingLeastSquares(factors=32, regularization = 0.02, iterations = 50)
    elif model_name=='bpr':
        model = BayesianPersonalizedRanking(factors=factors, learning_rate=0.01, regularization=regularization, iterations=iterations)
    
    '''item_users (csr_matrix) – Matrix of confidences for the liked items. 
    This matrix should be a csr_matrix where the rows of the matrix are the item, 
    the columns are the users that liked that item, and the value is the confidence 
    that the user liked the item.'''
    
    model.fit(sparsed.T)
    
    '''user_items (csr_matrix) – A sparse matrix of shape (number_users, number_items). 
    This lets us look up the liked items and their weights for the user. 
    This is used to filter out items that have already been liked from the output, 
    and to also potentially calculate the best items for this user.'''
    
    user_items = sparsed.tocsr()
    
    result = pd.DataFrame(columns=['customer_id', 'recommendation'])

    # Calculates the N best recommendations for a user, and returns a list of itemids, score
    for i in range(matrix.shape[0]):
        rc = model.recommend(i, user_items, N=10)
          result.loc[i,'customer_id'] = matrix.index[i]
        result.loc[i,'recommendation'] = rc

    x = pd.DataFrame(result.recommendation.tolist(), index=result.customer_id).stack().reset_index(level=1, drop=True).reset_index(name='recommendation')
    df_new['customer_id '+model_name] = x['customer_id']
    df_new['recommendation '+model_name] = x['recommendation'].apply(lambda x: x[0])
    df_new['score '+model_name] = x['recommendation'].apply(lambda x: x[1])

    df_new = df_new.replace({'recommendation '+model_name: item_dictionary})
    return df_new
```

```python id="mGdme_RVZmq4" outputId="7c5866d7-3ab9-47c3-bea4-9344ca07efef" colab={"base_uri": "https://localhost:8080/", "height": 221}
calculate_recommendations(pd.DataFrame(), model_name='als').head()
```

```python id="-c53tFWrQBHO" outputId="95edf1bf-259c-4049-ea6c-64314680d2eb" colab={"base_uri": "https://localhost:8080/", "height": 238}
df_n = pd.DataFrame()
df_n = calculate_recommendations(df_n, model_name='als')
df_n = calculate_recommendations(df_n, model_name='bpr')
df_n.head()
```

<!-- #region id="Upz1WgvsOMjD" -->
### Model Evaluation
- It is important to realize that we do not have a reliable feedback regarding which items are disliked. The absence of a favorite item indicator can be related to multiple reasons. We also can't track user reactions to our recommendations. Thus, precision based metrics, such as RMSE and MSE, are not very appropriate, as they require knowing which items users dislike for it to make sense. 
In addition, we are currently unable to track user reactions to our recommendations. Thus, precision based metrics are not very appropriate, as they require knowing which programs are undesired to a user. However, watching a program is an indication of liking it, making recall-oriented measures applicable.
1.	Random masking and measuring predicted vs. actual values of masked values – ROC AUC score
   <img src='https://jessesw.com/images/Rec_images/MaskTrain.png' width=600>
2.	Recall based evaluation ranking – **Mean Percentage Ranking (MPR)** a.k.a. expected percentile ranking.  Lower values of MPR are more desirable. The expected value of MPR for random predictions is 50%, and thus MPR > 50% indicates an algorithm no better than random.

<!-- #endregion -->

```python id="fpnoGvWz2JiU"
def make_train(ratings, pct_test = 0.2):
    '''
    This function will take in the original user-item matrix and "mask" a percentage of the original ratings where a
    user-item interaction has taken place for use as a test set. The test set will contain all of the original ratings, 
    while the training set replaces the specified percentage of them with a zero in the original ratings matrix. 
    
    parameters: 
    
    ratings - the original ratings matrix from which you want to generate a train/test set. Test is just a complete
    copy of the original set. This is in the form of a sparse csr_matrix. 
    
    pct_test - The percentage of user-item interactions where an interaction took place that you want to mask in the 
    training set for later comparison to the test set, which contains all of the original ratings. 
    
    returns:
    
    training_set - The altered version of the original data with a certain percentage of the user-item pairs 
    that originally had interaction set back to zero.
    
    test_set - A copy of the original ratings matrix, unaltered, so it can be used to see how the rank order 
    compares with the actual interactions.
    
    user_inds - From the randomly selected user-item indices, which user rows were altered in the training data.
    This will be necessary later when evaluating the performance via AUC.
    '''
    test_set = ratings.copy() # Make a copy of the original set to be the test set. 
    test_set[test_set != 0] = 1 # Store the test set as a binary preference matrix
    training_set = ratings.copy() # Make a copy of the original data we can alter as our training set. 
    nonzero_inds = training_set.nonzero() # Find the indices in the ratings data where an interaction exists
    nonzero_pairs = list(zip(nonzero_inds[0], nonzero_inds[1])) # Zip these pairs together of user,item index into list
    random.seed(0) # Set the random seed to zero for reproducibility
    num_samples = int(np.ceil(pct_test*len(nonzero_pairs))) # Round the number of samples needed to the nearest integer
    samples = random.sample(nonzero_pairs, num_samples) # Sample a random number of user-item pairs without replacement
    user_inds = [index[0] for index in samples] # Get the user row indices
    item_inds = [index[1] for index in samples] # Get the item column indices
    training_set[user_inds, item_inds] = 0 # Assign all of the randomly chosen user-item pairs to zero
    training_set.eliminate_zeros() # Get rid of zeros in sparse array storage after update to save space
    return training_set, test_set, list(set(user_inds)) # Output the unique list of user rows that were altered 
```

```python id="ruzywaoFCJ2r"
def auc_score(predictions, test):
    '''
    This simple function will output the area under the curve using sklearn's metrics. 
    
    parameters:
    
    - predictions: your prediction output
    
    - test: the actual target result you are comparing to
    
    returns:
    
    - AUC (area under the Receiver Operating Characterisic curve)
    '''
    fpr, tpr, thresholds = metrics.roc_curve(test, predictions)
    return metrics.auc(fpr, tpr)
  

  
def calc_mean_auc(training_set, altered_users, predictions, test_set):
    '''
    This function will calculate the mean AUC by user for any user that had their user-item matrix altered. 
    
    parameters:
    
    training_set - The training set resulting from make_train, where a certain percentage of the original
    user/item interactions are reset to zero to hide them from the model 
    
    predictions - The matrix of your predicted ratings for each user/item pair as output from the implicit MF.
    These should be stored in a list, with user vectors as item zero and item vectors as item one. 
    
    altered_users - The indices of the users where at least one user/item pair was altered from make_train function
    
    test_set - The test set constucted earlier from make_train function
    
    

    returns:
    
    The mean AUC (area under the Receiver Operator Characteristic curve) of the test set only on user-item interactions
    there were originally zero to test ranking ability in addition to the most popular items as a benchmark.
    '''
    
    
    store_auc = [] # An empty list to store the AUC for each user that had an item removed from the training set
    popularity_auc = [] # To store popular AUC scores
    pop_items = np.array(test_set.sum(axis = 0)).reshape(-1) # Get sum of item iteractions to find most popular
    item_vecs = predictions[1]
    for user in altered_users: # Iterate through each user that had an item altered
        training_row = training_set[user,:].toarray().reshape(-1) # Get the training set row
        zero_inds = np.where(training_row == 0) # Find where the interaction had not yet occurred
        # Get the predicted values based on our user/item vectors
        user_vec = predictions[0][user,:]
        pred = user_vec.dot(item_vecs).toarray()[0,zero_inds].reshape(-1)
        # Get only the items that were originally zero
        # Select all ratings from the MF prediction for this user that originally had no iteraction
        actual = test_set[user,:].toarray()[0,zero_inds].reshape(-1) 
        # Select the binarized yes/no interaction pairs from the original full data
        # that align with the same pairs in training 
        pop = pop_items[zero_inds] # Get the item popularity for our chosen items
        store_auc.append(auc_score(pred, actual)) # Calculate AUC for the given user and store
        popularity_auc.append(auc_score(pop, actual)) # Calculate AUC using most popular and score
    # End users iteration
    
    # Return the mean AUC rounded to three decimal places for both test and popularity benchmark
    return float('%.3f'%np.mean(store_auc)), float('%.3f'%np.mean(popularity_auc))  
```

```python id="2oga2G5W6kFR" outputId="0b50bac1-d680-4dbc-d4b7-481d928ef3be" colab={"base_uri": "https://localhost:8080/", "height": 2465}
### ALS ###

# hyperparameters

PCT = [0.2,0.3]
factors = [32,64,128]
regularization = [0.01,0.05,0.1,0.2]
iterations = [10,20,50]

# PCT = [0.2]
# factors = [32]
# regularization = [0.01]
# iterations = [10]

scores = pd.DataFrame(columns = ['PCT','factors','regularization','iterations','score'])

# Grid-search hyperparameter optimization
for i in PCT:
  for j in factors:
    for k in regularization: 
      for index,l in enumerate(iterations):
        # creating train, test and altered sets
        train, test, altered = make_train(sparsed, pct_test = i)

        # calculate the confidence by multiplying it by our alpha value
        alpha_val = 15
        train = (train * alpha_val).astype('double')

        # defining the model
        model = implicit.als.AlternatingLeastSquares(factors=j, regularization = k, iterations = l)

        # training the model
        model.fit(train.T)

        # AUC for our recommender system
        score = calc_mean_auc(train, altered, [sparse.csr_matrix(model.user_factors), sparse.csr_matrix(model.item_factors.T)], test)
        print(model.user_factors.shape)
        # saving in a dataframe
        scores.loc[index,'PCT'] = i
        scores.loc[index,'factors'] = j
        scores.loc[index,'regularization'] = k
        scores.loc[index,'iterations'] = l
        scores.loc[index,'score'] = score
        
```

```python id="sAVu1MxoNV0D" outputId="6e7aa946-fea7-47ea-d738-0c265c8bf1eb" colab={"base_uri": "https://localhost:8080/", "height": 142}
pd.DataFrame(scores.score.tolist(), index=scores[['PCT','factors','regularization','iterations']], columns=['ALS','Popularity']).sort_values(by='ALS', ascending=False)
```

```python id="Zlt-ZLVRPrPz" outputId="a5e0d693-719e-47b4-8e89-2d27e1ec6871" colab={"base_uri": "https://localhost:8080/", "height": 1547}
### BPR ###

# hyperparameters

PCT = [0.2, 0.3]
factors = [31,63,127]
regularization = [0.01, 0.03, 0.05, 0.1, 0.2]
# learning_rate = [0.01, 0.05, 0.1]
iterations = [10,20,50]

scores = pd.DataFrame(columns = ['PCT','factors','regularization','iterations','score'])

# Grid-search hyperparameter optimization
for i in PCT:
  for j in factors:
    for k in regularization: 
      for index,l in enumerate(iterations):
        # creating train, test and altered sets
        train, test, altered = make_train(sparsed, pct_test = i)

        # calculate the confidence by multiplying it by our alpha value
        alpha_val = 15
        train = (train * alpha_val).astype('double')

        # defining the model
        model = BayesianPersonalizedRanking(factors=j, regularization = k, iterations = l)

        # training the model
        model.fit(train.T)

        # AUC for our recommender system
        score = calc_mean_auc(train, altered, [sparse.csr_matrix(model.user_factors), sparse.csr_matrix(model.item_factors.T)], test)
        
        # saving in a dataframe
        scores.loc[index,'PCT'] = i
        scores.loc[index,'factors'] = j
        scores.loc[index,'regularization'] = k
        scores.loc[index,'iterations'] = l
        scores.loc[index,'score'] = score
```

```python id="HaC4SPQhRBXh" outputId="74b7476b-b233-40cf-bc92-749d9e3c57b9" colab={"base_uri": "https://localhost:8080/", "height": 142}
pd.DataFrame(scores.score.tolist(), index=scores[['PCT','factors','regularization','iterations']], columns=['ALS','Popularity']).sort_values(by='ALS', ascending=False)
```

```python id="UY-3a4BgPrDw" outputId="4f0ba189-4311-4aa3-8b2c-6b44247186db" colab={"base_uri": "https://localhost:8080/", "height": 238}
# ALS best parameters - (0.2, 128, 0.05, 20)	
# BPR best parameters - (0.2, 127, 0.05, 20)	

df_n = pd.DataFrame()
df_n = calculate_recommendations(df_n, model_name='als', factors=128, regularization=0.05, iterations=20)
df_n = calculate_recommendations(df_n, model_name='bpr', factors=127, regularization=0.05, iterations=20)
df_n.to_csv('recommendations.csv')
df_n.head()
```

<!-- #region id="23_0lqemJ7te" -->
### References
1.	https://www.benfrederickson.com/distance-metrics/
2.	https://github.com/benfred/bens-blog-code/blob/master/distance-metrics/calculate_similar.py
3.	https://www.benfrederickson.com/approximate-nearest-neighbours-for-recommender-systems/
4.	https://www.benfrederickson.com/fast-implicit-matrix-factorization/
5.	https://www.ethanrosenthal.com/2016/10/19/implicit-mf-part-1/
6.	https://jessesw.com/Rec-System/
7.	https://github.com/benfred/implicit/blob/master/examples/lastfm.py
8.	https://github.com/benfred/implicit
9.	https://towardsdatascience.com/large-scale-jobs-recommendation-engine-using-implicit-data-in-pyspark-ccf8df5d910e
10.	http://activisiongamescience.github.io/2016/01/11/Implicit-Recommender-Systems-Biased-Matrix-Factorization/
11.	https://arxiv.org/pdf/1705.00105.pdf
12.	https://www.ijcai.org/Proceedings/15/Papers/255.pdf
13.	https://www.kaggle.com/c/msdchallenge/
14.	https://stanford.edu/~rezab/nips2014workshop/submits/logmat.pdf
15.	http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.167.5120&rep=rep1&type=pdf 
16.	http://adrem.uantwerpen.be/bibrem/pubs/verstrepen15PhDthesis.pdf 
17.	https://pdfs.semanticscholar.org/eb95/7789f53814a290bc0f8bb01dd01cbd0746cc.pdf 
18.	https://implicit.readthedocs.io/en/latest/ 
19.	https://github.com/akhilesh-reddy/Implicit-data-based-recommendation-system/blob/master/Implicit%20data%20based%20recommendation%20system%20using%20ALS.ipynb 

<!-- #endregion -->
