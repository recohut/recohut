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

<!-- #region id="NTCf7vHJPFfG" -->
# Electronics Product Recommender
> Training electronics product recommender on Amazon electronics product rating data using turicreate and surprise library based popularity and collaborative filtering models and evaluating on RMSE

- toc: true
- badges: true
- comments: true
- categories: [Turicreate, Surprise, HyperTuning, Electronics, Retail, ECommerce]
- image:
<!-- #endregion -->

<!-- #region id="bDPrFtE46oqx" -->
### Install libraries
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 1000} id="JkOocUyH6sCe" outputId="e423af7a-1c87-49bd-be24-f5625639af37"
#hide-output
!pip install surprise
!pip install turicreate
```

<!-- #region id="L3bBe1l76sqr" -->
### Import libraries
<!-- #endregion -->

```python id="QvWFMAll6zxM"
import numpy as np
import pandas as pd
import math
import joblib 
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
import scipy.sparse
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

from surprise import Dataset
from surprise import Reader
from surprise import accuracy                                              
from surprise.model_selection import cross_validate
from surprise.model_selection import GridSearchCV
from surprise import KNNBasic, KNNWithMeans, KNNWithZScore
from surprise import SVD, SVDpp, NMF
from surprise import SlopeOne, CoClustering

import turicreate

%matplotlib inline

import warnings
warnings.simplefilter('ignore')
```

<!-- #region id="QCCETtQO6W9q" -->
### Load the data
We will download the data from Kaggle, unzip it, and load it into pandas dataframe.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="wH0ilH4q6bLa" outputId="a3f33f95-0168-406e-90d4-f5ce09801cfd"
#hide-output
!pip install -q -U kaggle
!pip install --upgrade --force-reinstall --no-deps kaggle
!mkdir ~/.kaggle
!cp /content/drive/MyDrive/kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d pritech/ratings-electronics
!unzip ratings-electronics.zip
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="LnA4beZm7Ntd" outputId="71950b32-5475-4b3a-9d0a-ec96867e8586"
columns=['userId', 'productId', 'ratings','timestamp']
electronics_df = pd.read_csv('ratings_Electronics.csv', names=columns)
electronics_df.head()
```

<!-- #region id="zftWmsuj8IyL" -->
### Pre-processing
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="nPG8Do5L7_Fg" outputId="1bfac607-612d-4ee1-8086-918a4b6cf673"
electronics_df.info()
```

```python colab={"base_uri": "https://localhost:8080/"} id="cWbXQdxP8DQt" outputId="3f56a119-f5e4-4b9b-cc16-0214e3c67468"
electronics_df.isnull().sum()
```

```python id="FH_lQqNn8MOt"
electronics_df.drop('timestamp',axis=1,inplace=True)
```

```python colab={"base_uri": "https://localhost:8080/"} id="l9cSRrgs8deI" outputId="8477108d-f9b6-468d-b8fc-dc490200309d"
print("Electronic Data Summary")
print("="*100)
print("\nTotal # of Ratings :",electronics_df.shape[0])
print("Total # of Users   :", len(np.unique(electronics_df.userId)))
print("Total # of Products  :", len(np.unique(electronics_df.productId)))
print("\n")
print("="*100)
```

<!-- #region id="dkcYSRV18fyM" -->
> Note: Number of products is less than number of users, so item-item colaborative filtering would make sense instead of user-user colaborative filtering.
<!-- #endregion -->

<!-- #region id="M1PcB-c4AMi8" -->
### Sampling
<!-- #endregion -->

<!-- #region id="-Tx7_xRv9VRl" -->
There are more than 4.2 million users. It is big for a prototype version like ours. So we will select ~10K users randomly, and assume that the sample would represent the population.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 142} id="rAt714XK8r6r" outputId="21160894-b62c-4d99-f69e-fb73e4aea927"
userids_random10k = np.random.choice(electronics_df.userId.unique(), int(1e4))
electronics_df = electronics_df.loc[electronics_df.userId.isin(userids_random10k)]
electronics_df.describe(include='all').T
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="LS34LlBiA85-" outputId="a0d213ab-fc5d-49f7-9354-1535a61bd0ed"
electronics_df.head()
```

<!-- #region id="q_Q4x0I9BDSJ" -->
### Encoding

Label encode userid and productid and int for ratings.
<!-- #endregion -->

```python id="sgMym_doBL_X"
le_userid = LabelEncoder()
electronics_df.userId = le_userid.fit_transform(electronics_df.userId)

le_prodid = LabelEncoder()
electronics_df.productId = le_prodid.fit_transform(electronics_df.productId)

electronics_df.ratings = electronics_df.ratings.astype('int32')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="j-nplbmmCHQq" outputId="3d481b03-0a98-4e42-eb29-7c6a8863bde6"
electronics_df.head()
```

<!-- #region id="tri4sUTWAR3F" -->
### Matrix
Constructing pivot table on `userId` and `productId` column based on the value of `ratings`

<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 284} id="XtFRDpK6_j_U" outputId="b70f83e0-db4a-4a1c-c4b0-c9d34df0012c"
final_ratings_matrix = electronics_df.pivot(index = 'userId', columns ='productId', values = 'ratings')
final_ratings_matrix.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="kXWyZPxmAeCf" outputId="ab1881d0-774b-4a7b-9854-28b90a16d1ad"
#Calucating the density of the rating marix
given_num_of_ratings = final_ratings_matrix.count().sum()
print('given_num_of_ratings = ', given_num_of_ratings)
possible_num_of_ratings = final_ratings_matrix.shape[0] * final_ratings_matrix.shape[1]
print('possible_num_of_ratings = ', possible_num_of_ratings)
density = (given_num_of_ratings/possible_num_of_ratings)
density *= 100
print ('density: {:4.2f}%'.format(density))
```

<!-- #region id="WbQJzBiiDp82" -->
### Split
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="cVJ9iz5sDrDL" outputId="b83e70c3-0ca4-4d09-9cf7-511a0aadac1a"
#Split the data randomnly into train and test datasets into 70:30 ratio
train_data, test_data = train_test_split(electronics_df, test_size = 0.3, random_state=42)
train_data.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="gWUKf6vaEfuu" outputId="f172826c-b102-41f2-eb0a-de7be422c565"
print('Shape of training data: ', train_data.shape)
print('Shape of testing data: ', test_data.shape)
```

<!-- #region id="m5soQKXZEjab" -->
## Models
<!-- #endregion -->

<!-- #region id="HBf7r28JEmR5" -->
### Popularity (manual)
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="XRFw2GYcEgLf" outputId="04ccb6f7-0f75-4946-82e3-a0f45770482f"
#Count of user_id for each unique product as recommendation score 
train_data_grouped = train_data.groupby('productId').agg({'userId': 'count'}).reset_index()
train_data_grouped.rename(columns = {'userId': 'score'},inplace=True)

#Sort the products on recommendation score 
train_data_sort = train_data_grouped.sort_values(['score', 'productId'], ascending = [0,1]) 
      
#Generate a recommendation rank based upon score 
train_data_sort['rank'] = train_data_sort['score'].rank(ascending=0, method='first') 
          
#Get the top 5 recommendations 
popularity_recommendations = train_data_sort.head(5)
popularity_recommendations
```

```python id="zNhNIXG4F6Bu"
# Sample fuction to use manual popularity based recommender model to make predictions
def recommend(user_id):     
    user_recommendations = popularity_recommendations 
          
    #Add user_id column for which the recommendations are being generated 
    user_recommendations['userId'] = user_id 
      
    #Bring user_id column to the front 
    cols = user_recommendations.columns.tolist() 
    cols = cols[-1:] + cols[:-1] 
    user_recommendations = user_recommendations[cols] 
          
    return user_recommendations 
```

```python colab={"base_uri": "https://localhost:8080/"} id="Za3wSsruE-Qh" outputId="9e292fa8-026d-494d-e50e-36ae27fbc801"
find_recom = [10,100,150]   # This list is user choice.
for i in find_recom:
    print("The list of recommendations for the userId: %d\n" %(i))
    print(recommend(i))    
    print("\n") 
```

<!-- #region id="-uylt0CAFAGA" -->
Since, it is a Popularity recommender model, so, all the three users are given the same recommendations. Here, we predict the products based on the popularity. It is not personalized to particular user. It is a non-personalized recommender system.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="EIl2Cs-cFEEQ" outputId="caf5a3c1-b62e-4b03-b30d-c4d1972cbf1a"
#Calculating the RMSE of the popularity based recommendation system
#Rating present in the test data is the actual rating (Act_rating)
test_data2 = test_data.copy()
test_data2.drop(['userId'],axis=1,inplace=True)
test_data2.rename(columns = {'ratings':'Act_rating'}, inplace = True)

#Count of user_id for each unique product as recommendation score

train_data_grouped2 = train_data.groupby('productId').agg({'ratings': 'sum'}).reset_index()
train_data_grouped2.rename(columns = {'ratings': 'Sum_rating'},inplace=True)
train_data_grouped2.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="iIgvoP8rGnr7" outputId="f751492a-5a3f-4ac5-b32b-ac79776c5e99"
train_data_inner = pd.merge(train_data_grouped2, train_data_sort)
train_data_inner.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="1zzZcV6GGsHY" outputId="348bb81e-b873-44e1-8656-50133be27edc"
#Obtain the average rating of the product across users
train_data_inner["Avg_Rating"] = train_data_inner["Sum_rating"]/train_data_inner["score"]
train_data_inner.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="QbyI7TjTG4jY" outputId="cc34bb10-d351-4c3b-cc83-91d37ca17876"
#Merge the train data having average rating with the test data having actual rating at product level
test_data_inner = pd.merge(train_data_inner, test_data2)
test_data_inner.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="HdURdJMaHAJO" outputId="18e1d33e-0302-446e-8027-8069d7860aa6"
#RMSE for popularity based recommender system
mse = mean_squared_error(test_data_inner["Act_rating"], test_data_inner["Avg_Rating"])
rmse = math.sqrt(mse)
print("RMSE for popularity based recommendation system:", rmse)
```

<!-- #region id="AP8YnZQoHIBJ" -->
### Popularity (turicreate)
<!-- #endregion -->

```python id="LDyB9r6bHCOO"
train_data2 = turicreate.SFrame(train_data)
test_data2 = turicreate.SFrame(test_data)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 85} id="zbQefJldHcqp" outputId="e53c0c10-8a5f-4686-fb29-ffcbecec795f"
popularity_model = turicreate.popularity_recommender.create(train_data2, user_id='userId', item_id='productId', target='ratings')
```

```python colab={"base_uri": "https://localhost:8080/"} id="sFEE61ObHol5" outputId="9275215a-aeef-4c9c-e288-c22e020f0003"
random_5users = np.random.choice(userids_random10k, 5)
random_5users
```

```python colab={"base_uri": "https://localhost:8080/"} id="ihnkUdK8HcoH" outputId="82a5fee3-e7d5-4f32-bcbd-33bfcec63d4f"
#Recommend for a given set of users, since there are top 5 recommendation for 5 users, total rows will be 25
popularity_recomm = popularity_model.recommend(users=le_userid.transform(random_5users),k=5)
popularity_recomm.print_rows(num_rows=25)
```

```python colab={"base_uri": "https://localhost:8080/"} id="6FlTeLDBHcl5" outputId="d663fc30-b5a0-4256-ed08-6cd126cdda98"
pop_rmse = popularity_model.evaluate_rmse(test_data2,'ratings')
print(pop_rmse)
#The RMSE value for the popularity based model
pop_rmse["rmse_overall"]
```

<!-- #region id="XLOQvP3eIrdn" -->
### Item-item similarity (turicreate)
<!-- #endregion -->

<!-- #region id="wmFprRLOIyWq" -->
This model first computes the similarity between items using the observations of users who have interacted with both items. Given a similarity between item i and j, S(i,j), it scores an item j for user u using a weighted average of the user’s previous observations Iu.

<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 391} id="YzQg1pwgHcih" outputId="975458b4-8425-429b-dd04-40bd4fc1b2aa"
#Training the model for item-item similarity recommender
item_sim_model = turicreate.item_similarity_recommender.create(train_data2, user_id='userId',
                                                               item_id='productId', target='ratings',
                                                               similarity_type='cosine')
```

```python colab={"base_uri": "https://localhost:8080/"} id="-iAXMZ_sI5gx" outputId="dfb60990-03f8-4966-bc63-cf723806f7bb"
#Get the item-item recommender RMSE value
item_rmse = item_sim_model.evaluate_rmse(test_data2,'ratings')
item_rmse["rmse_overall"]
```

<!-- #region id="z_-rJXvRJMRc" -->
### Matrix factorization (turicreate)
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 765} id="J8p5TV3lJDlN" outputId="9d76c254-c207-4950-93c5-ff852b875b8b"
#Build a matrix based factorization model recommender system
factorization_model = turicreate.factorization_recommender.create(train_data2, user_id='userId',
                                                                  item_id='productId', target='ratings')
```

```python colab={"base_uri": "https://localhost:8080/"} id="yaNIZ8m6JVNT" outputId="66b9af14-f5a9-40ba-cc01-047662956f66"
fcm_rmse2 = factorization_model.evaluate_rmse(test_data2,'ratings')
fcm_rmse2["rmse_overall"]
```

<!-- #region id="PAj4f6FzK2h2" -->
### Multiple models (surprise library)
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="1u3yFxtQJVjq" outputId="7742e15f-d39b-41ba-bead-30a94166b2d6"
rts_gp = electronics_df.groupby(by=['ratings']).agg({'userId': 'count'}).reset_index()
rts_gp.columns = ['ratings', 'Count']

#Subsetting the data to keep products having at least x=1 ratings
prod_ge=electronics_df.groupby("productId").filter(lambda x:x['ratings'].count() >= 2)
#Subsetting the data to keep users who have given at least y=1 ratings
user_ge=electronics_df.groupby("userId").filter(lambda x:x['ratings'].count() >= 2)
user_ge.drop(['ratings'],inplace=True,axis=1)
user_prod_ge = pd.merge(prod_ge,user_ge)
user_prod_ge.shape
```

<!-- #region id="kG0Yst-XKOL8" -->
> Tip: As we already sampled the data, so filtering by min. rating count of 2 here, but if you are not using sampling, feel free to raise this x,y bar to 5 or maybe 25.
<!-- #endregion -->

```python id="gjJeSaA7J1IZ"
# Set Rating Scale from 1 to 5
#We are running basic algorithms to check which one works best
reader = Reader(rating_scale=(1, 5))

# Load data with rating scale
#data = Dataset.load_from_df(new_df, reader)
data = Dataset.load_from_df(user_prod_ge,reader)
```

```python colab={"base_uri": "https://localhost:8080/"} id="D3vzPXiMML4o" outputId="64e3d80c-8f98-4a13-df2a-6f34323e7590"
knnbasic_cv = cross_validate(KNNBasic(), data, cv=5, n_jobs=5, verbose=True)
knnmeans_cv = cross_validate(KNNWithMeans(), data, cv=5, n_jobs=5, verbose=True)
knnz_cv = cross_validate(KNNWithZScore(), data, cv=5, n_jobs=5, verbose=True)
svd_cv = cross_validate(SVD(), data, cv=5, n_jobs=5, verbose=True)
svdpp_cv = cross_validate(SVDpp(), data, cv=5, n_jobs=5, verbose=True)
nmf_cv = cross_validate(NMF(), data, cv=5, n_jobs=5, verbose=True)
slope_cv = cross_validate(SlopeOne(), data, cv=5, n_jobs=5, verbose=True)
coclus_cv = cross_validate(CoClustering(), data, cv=5, n_jobs=5, verbose=True)
```

```python colab={"base_uri": "https://localhost:8080/"} id="Xso3cLSELBBX" outputId="927ca51a-76d0-4563-f128-8f03eae38096"
print('Algorithm\t RMSE\t\t MAE')
print()
print('KNN Basic', '\t', round(knnbasic_cv['test_rmse'].mean(), 4), '\t', round(knnbasic_cv['test_mae'].mean(), 4))
print('KNN Means', '\t', round(knnmeans_cv['test_rmse'].mean(), 4), '\t', round(knnmeans_cv['test_mae'].mean(), 4))
print('KNN ZScore', '\t', round(knnz_cv['test_rmse'].mean(), 4), '\t', round(knnz_cv['test_mae'].mean(), 4))
print()
print('SVD', '\t\t', round(svd_cv['test_rmse'].mean(), 4), '\t', round(svd_cv['test_mae'].mean(), 4))
print('SVDpp', '\t\t', round(svdpp_cv['test_rmse'].mean(), 4), '\t', round(svdpp_cv['test_mae'].mean(), 4))
print('NMF', '\t\t', round(nmf_cv['test_rmse'].mean(), 4), '\t', round(nmf_cv['test_mae'].mean(), 4))
print()
print('SlopeOne', '\t', round(slope_cv['test_rmse'].mean(), 4), '\t', round(slope_cv['test_mae'].mean(), 4))
print('CoClustering', '\t', round(coclus_cv['test_rmse'].mean(), 4), '\t', round(coclus_cv['test_mae'].mean(), 4))
print()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 287} id="kYTCMgdQLFdw" outputId="008f72a6-531c-4746-e3b3-fd0dce68f5c8"
x_algo = ['KNN Basic', 'KNN Means', 'KNN ZScore', 'SVD', 'SVDpp', 'NMF', 'SlopeOne', 'CoClustering']
all_algos_cv = [knnbasic_cv, knnmeans_cv, knnz_cv, svd_cv, svdpp_cv, nmf_cv, slope_cv, coclus_cv]

rmse_cv = [round(res['test_rmse'].mean(), 4) for res in all_algos_cv]
mae_cv = [round(res['test_mae'].mean(), 4) for res in all_algos_cv]

plt.figure(figsize=(20,5))

plt.subplot(1, 2, 1)
plt.title('Comparison of Algorithms on RMSE', loc='center', fontsize=15)
plt.plot(x_algo, rmse_cv, label='RMSE', color='darkgreen', marker='o')
plt.xlabel('Algorithms', fontsize=15)
plt.ylabel('RMSE Value', fontsize=15)
plt.legend()
plt.grid(ls='dashed')

plt.subplot(1, 2, 2)
plt.title('Comparison of Algorithms on MAE', loc='center', fontsize=15)
plt.plot(x_algo, mae_cv, label='MAE', color='navy', marker='o')
plt.xlabel('Algorithms', fontsize=15)
plt.ylabel('MAE Value', fontsize=15)
plt.legend()
plt.grid(ls='dashed')

plt.show()
```

<!-- #region id="x583EFjEMsNb" -->
## Hyperparameter tuning
Grid Searching on SVD and SVDPP algorithms
<!-- #endregion -->

```python id="7_H_NncdMeb8"
raw_ratings = data.raw_ratings                         
# 90% trainset, 10% testset                                                
threshold = int(.9 * len(raw_ratings))                                     
trainset_raw_ratings = raw_ratings[:threshold]                             
test_raw_ratings = raw_ratings[threshold:]             
data.raw_ratings = trainset_raw_ratings        
```

```python colab={"base_uri": "https://localhost:8080/"} id="8Cf06sJIM4OF" outputId="9fb7f24b-5bff-48c4-bed1-b78a84a2a02e"
# Parameter space
svd_param_grid = {'n_epochs': [20, 25], 
                  'lr_all': [0.007, 0.009, 0.01],
                  'reg_all': [0.4, 0.6]}

svdpp_gs = GridSearchCV(SVDpp, svd_param_grid, measures=['rmse', 'mae'], cv=5, n_jobs=5)
svdpp_gs.fit(data)
algo_svdpp = svdpp_gs.best_estimator['rmse']   

# retrain on the whole train set                      
trainset = data.build_full_trainset()                 
algo_svdpp.fit(trainset)

# now test on the trainset                                                 
testset = data.construct_testset(trainset_raw_ratings)                     
predictions_train = algo_svdpp.test(testset)                                           
print('Accuracy on the trainset:')                                         
accuracy.rmse(predictions_train)                                                 

# now test on the testset                                                  
testset = data.construct_testset(test_raw_ratings)                         
pred_svdpp=algo_svdpp.test(testset)
print('Accuracy on the testset:')                                          
accuracy.rmse(pred_svdpp)  


svd_gs = GridSearchCV(SVD, svd_param_grid, measures=['rmse', 'mae'], cv=5, n_jobs=5)
svd_gs.fit(data)
algo_svd = svd_gs.best_estimator['rmse']    
# retrain on the whole train set                      
trainset = data.build_full_trainset()                  
algo_svd.fit(trainset)
# now test on the trainset                                                 
testset = data.construct_testset(trainset_raw_ratings)                     
predictions_train = algo_svd.test(testset)                                           
print('Accuracy on the trainset:')                                         
accuracy.rmse(predictions_train)                                                 

# now test on the testset                                                  
testset = data.construct_testset(test_raw_ratings)                         
pred_svd=algo_svd.test(testset)
print('Accuracy on the testset:')                                          
accuracy.rmse(pred_svd) 
```

```python colab={"base_uri": "https://localhost:8080/"} id="L2fkUf3AM9Xb" outputId="03a86e10-bfa2-4c5d-9b95-dfe492e927c5"
print('SVDpp - RMSE:', round(svdpp_gs.best_score['rmse'], 4), '; MAE:', round(svdpp_gs.best_score['mae'], 4))
print('SVD   - RMSE:', round(svd_gs.best_score['rmse'], 4), '; MAE:', round(svd_gs.best_score['mae'], 4))
print('RMSE =', svdpp_gs.best_params['rmse'])
print('MAE =', svdpp_gs.best_params['mae'])
print('RMSE =', svd_gs.best_params['rmse'])
print('MAE =', svd_gs.best_params['mae'])
```

```python id="qotSUqw7NQhu"
def get_top_n_recommendations(reccomemndations, n=5):
    # First map the reccommendations to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in reccomemndations:
        inverselabel_iid = le_prodid.inverse_transform([iid])[0]
        inverselabel_uid = le_userid.inverse_transform([uid])[0]
        top_n[inverselabel_uid].append((inverselabel_iid, est))

    #sort predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n
```

```python colab={"base_uri": "https://localhost:8080/"} id="7ljTg--YNS-n" outputId="e6ff053c-ae30-4530-87bb-388eb3433913"
#collapse-output
top_5 = get_top_n_recommendations(pred_svd, n=5)
for uid, user_ratings in top_5.items():
    print(uid, [iid for (iid, _) in user_ratings])
```

<!-- #region id="57n4lXTmNXzt" -->
**Observation :**
* The Popularity-based recommender system is a non-personalized recommender system and these are based on frequency counts, which may be not suitable to the user.
* Model-based Collaborative Filtering is a personalized recommender system, the recommendations are based on the past behavior of the user and it is not dependent on any additional information.
* Item Similarity Based model performed worse than even simple popularity based models.
* Hyper parameter tuning with GridSearch improve model performance.
<!-- #endregion -->
