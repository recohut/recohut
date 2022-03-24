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

<!-- #region id="gROX3NK6Ebdu" -->
# AyuHealth Healthy Food Recommendation
> Healthy Food Recommendation System using Content Based (KNN) & Collaborative Filtering (matrix-factorization) Method

- toc: true
- badges: true
- comments: true
- categories: [Tensorflow, Health&Fitness, Food, KNN]
- image:
<!-- #endregion -->

<!-- #region id="eHC4w_dxDu8j" -->
## Setup
<!-- #endregion -->

```python id="EaBBBRse1Zhm"
!wget https://github.com/sparsh-ai/reco-data/raw/master/ayuhealth/ayuhealth.zip
!unzip ayuhealth.zip
```

```python id="l2MrmKHl2RSz"
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from sklearn import preprocessing
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from datetime import datetime
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Embedding, Reshape, Dot, Flatten, concatenate, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import model_to_dot
from IPython.display import SVG
from tensorflow.keras.models import load_model
```

```python colab={"base_uri": "https://localhost:8080/"} id="9wuaw_qTDLz8" outputId="33bc6919-70cb-45a4-a048-6bf145220413"
!pip install -q watermark
%reload_ext watermark
%watermark -m -iv
```

```python id="eiwIZI085seA"
RATED_FOOD_PATH = 'food.csv'
RATED_FOOD_PATH = 'rated_food.csv'
RATING_PATH = 'ratings.csv'
USER_PATH = 'users.csv'
```

<!-- #region id="t-3z57kQ25Gp" -->
## KNN model
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 598} id="JDFqxjsp2nvq" outputId="ca49b72f-74f4-41d5-8a79-986baa65d8f8"
df = pd.read_csv(RATED_FOOD_PATH)
df.head()
```

<!-- #region id="kLiTMyLk3AQY" -->
### Feature selection
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="yhnNg79t2uEs" outputId="0f81d5ed-e583-44a9-ee22-12221c84c503"
knn_df = df.copy()
knn_df.drop(['food_id','food_code','name','category','type'], axis =1, inplace = True)
knn_df.head()
```

<!-- #region id="6WieV0cw3CJF" -->
### Scaling
<!-- #endregion -->

```python id="a0sYchvI3C6H"
min_max_scaler = MinMaxScaler()
knn_df = min_max_scaler.fit_transform(knn_df)
```

<!-- #region id="fvf-G9Yz3G_O" -->
### Model fitting
<!-- #endregion -->

```python id="goxcs_Vv3Gbp"
nbrs = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(knn_df)
distances, indices = nbrs.kneighbors(knn_df)
all_code = list(df['food_code'].values)
```

<!-- #region id="lOpgBgcW3Rxf" -->
### Testing
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 923} id="yWpJFduz3QyL" outputId="e9170bf9-ee74-4213-ee4b-d32faa511311"
# select a random food
random_food_code = np.random.choice(df.food_code.values,1)
random_food_name = df.loc[df.food_code==random_food_code[0],'name'].tolist()[0]

# find 10 similar items
searched_food_code = all_code.index(random_food_code[0])
similar_foods = df.loc[indices[searched_food_code]]

# print results
print("FnB that related to : ", random_food_name)
similar_foods
```

<!-- #region id="CQ_1teGV5I98" -->
### Save and load the model
<!-- #endregion -->

```python id="7AvEEsah4tFq"
import pickle
 
pickle.dump(nbrs, open('content_based_knn_model.pkl', 'wb'))
 
model = pickle.load(open('content_based_knn_model.pkl', 'rb'))
distances, indices = model.kneighbors(knn_df)
```

<!-- #region id="xrwZ8dPf5fwW" -->
## Collaborative-filtering model
<!-- #endregion -->

<!-- #region id="PM7T2DrtysYY" -->
**1. Read Exported CSV From DB**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 598} id="pUDOUFlF6ang" outputId="24f1486e-a82c-4751-d423-3a3c43c90cf3"
food_df = pd.read_csv(RATED_FOOD_PATH)
food_df.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="jLN660OzmDCM" outputId="c0d95d93-3489-45d9-8b43-14e3f516b7e4"
rating = pd.read_csv(RATING_PATH)
rating.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="swh8pshZy9tl" outputId="cced7ceb-a174-4866-d179-ac900237ace5"
user_df = pd.read_csv(USER_PATH)
user_df.head()
```

```python id="3HoaOpG3nanS"
rating.dropna(inplace=True)
```

```python colab={"base_uri": "https://localhost:8080/"} id="k32gmQa_nIXk" outputId="cee43021-3a26-4844-f852-906da0fe00c5"
rating.isna().sum()
```

<!-- #region id="60OUHLSkzlug" -->
### Encode user_id and food_code
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="jfbv-ozpmGFG" outputId="31f24fc8-e587-422f-96ba-ee352cbed998"
user_enc = preprocessing.LabelEncoder()
rating['User_ID'] = user_enc.fit_transform(rating['user_id'])

food_enc = preprocessing.LabelEncoder()
rating['Food_ID'] = food_enc.fit_transform(rating['food_code'])
rating.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="3KHOxY-JmaSg" outputId="6a1989c4-d53b-4008-8749-2a6ce4c60a51"
userid_nunique = rating['User_ID'].nunique()
food_unique = rating['Food_ID'].nunique()

print('User_id total unique:', userid_nunique)
print('Food_id total unique:', food_unique)
```

<!-- #region id="Z-fgIdJj0ZYU" -->
### Build Tensorflow Recommendation model
<!-- #endregion -->

```python id="Gg0HiLTCcYtu"
def RecommenderV2(n_users, n_food, n_dim):
    
    # User Embedding
    user = Input(shape=(1,))
    U = Embedding(n_users, n_dim)(user)
    U = Flatten()(U)
    
    # Food Rating Embedding
    food = Input(shape=(1,))
    M = Embedding(n_food, n_dim)(food)
    M = Flatten()(M)
    

    merged_vector = concatenate([U, M])
    dense_1 = Dense(128, activation='relu')(merged_vector)
    dense_2 = Dense(64, activation='relu')(dense_1)
    final = Dense(1)(dense_2)
    
    model = Model(inputs=[user, food], outputs=final)
    
    model.compile(optimizer=Adam(0.001), loss='mean_squared_error')
    
    return model
```

<!-- #region id="BGZ5sUTO09iD" -->
### Model structure
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 1000} id="JbMwnM_2mwst" outputId="e3faa89e-4f5e-4398-9832-5a870dfdb331"
model = RecommenderV2(userid_nunique, food_unique, 32)

SVG(model_to_dot(model, show_shapes=True, show_layer_names=True).create(prog='dot', format='svg'))
```

```python colab={"base_uri": "https://localhost:8080/"} id="T0gdRW2xm1xm" outputId="1bdc2747-929a-4ef2-fb62-a4031109f14a"
model.summary()
```

<!-- #region id="PMYjbgkF1E3U" -->
### Train/validation split
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="mDeLADZnm8vV" outputId="417589a8-ed0e-463b-f7ef-3372e7832c30"
X = rating.drop(['rating'], axis=1)
y = rating['rating']

X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                  test_size=.2,
                                                  stratify=y,
                                                  random_state=87)

X_train.shape, X_val.shape, y_train.shape, y_val.shape
```

<!-- #region id="cLBnlN011Mzj" -->
### Setting the callback
<!-- #endregion -->

```python id="fDj-7Uht0zzs"
checkpoint = ModelCheckpoint('model1.h5', monitor='val_loss', verbose=0, save_best_only=True)
val_loss_cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
```

<!-- #region id="Ykn8Wl221Qvh" -->
### Model training
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="CxXhNvQZnCuU" outputId="fa3c2b32-eb60-43e4-e2d1-e3ab8fcd0527"
history = model.fit(x=[X_train['User_ID'], X_train['Food_ID']],
                    y=y_train,
                    batch_size=64,
                    epochs=100,
                    verbose=1,
                    validation_data=([X_val['User_ID'], X_val['Food_ID']], y_val),
                    callbacks=[val_loss_cb,checkpoint])
```

<!-- #region id="lBAG5KH01sEN" -->
### Plot results
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 279} id="RgxguAGe1GWu" outputId="3fc817f0-26b1-4ffb-b03e-ccca12898c0a"
training_loss2 = history.history['loss']
test_loss2 = history.history['val_loss']

# Create count of the number of epochs
epoch_count = range(1, len(training_loss2) + 1)

# Visualize loss history
plt.plot(epoch_count, training_loss2, 'r--')
plt.plot(epoch_count, test_loss2, 'b-')
plt.legend(['Training Loss', 'Test Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
```

<!-- #region id="f4vhOOJC5rx1" -->
### Recommendation function
<!-- #endregion -->

```python id="HYeAWTvJ1hUV"
def make_pred(user_id, food_id, model):
    return model.predict([np.array([user_id]), np.array([food_id])])[0][0]
```

```python id="__0vSY0b6XKp"
def get_topN_rec(user_id, model):
    user_ratings = rating[rating['user_id'] == user_id]
    converted_id = rating[rating['user_id'] == user_id].head(1).User_ID.tolist()

    #remove food that user already rated
    recommendation = rating[~rating['Food_ID'].isin(user_ratings['Food_ID'])][['Food_ID','food_code']].drop_duplicates()
    #predict user rating for every user non rated food
    recommendation['rating_predict'] = recommendation.apply(lambda x: make_pred(converted_id[0], x['Food_ID'], model), axis=1)
    
    #create DF user non rated food sorted by rating prediction descending
    final_rec = recommendation.sort_values(by='rating_predict', ascending=False).merge(food_df[['food_code', 'name','type','category']],on='food_code')
    return final_rec.sort_values('rating_predict', ascending=False)[['name','food_code', 'rating_predict','type','category']]
```

<!-- #region id="NO3292dB9V2i" -->
### Recommending for a single user
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 447} id="iLwHSWUzfhv5" outputId="d5f79a44-8dc4-42e4-cd0f-247dc49fc7e4"
model_name = 'model1.h5'
model = load_model(model_name)

USER_ID = np.random.choice(rating.user_id.values,1)[0]
print('USER ID: {} RATED FOOD'.format(USER_ID))
# food_df.loc[food_df['Food_ID'].isin(rating.loc[rating['User_ID']==2]['Food_ID'].to_list())]
rating.loc[rating['user_id']==USER_ID].merge(food_df[['food_code', 'name','category','type']],on='food_code').sort_values(by='rating',ascending=False)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 436} id="rtcCchP88Sk_" outputId="899d0da3-e7e4-41ab-a449-930d180cd8d3"
print('RECOMENDATION FOR USER ID : ',USER_ID)
rec_result = get_topN_rec(USER_ID, model)
rec_result
```

<!-- #region id="0J8WiMBE6C6_" -->
### Recommending for all users
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 439} id="RB_Y92G1jDTA" outputId="7dea2a3a-fd82-4805-9247-fa8cdcdc6bf6"
# create zeros DF that index representing user_id and columns represending food_code
zeros_df = pd.DataFrame(0,index=sorted(rating['user_id'].unique()),columns=food_df['food_code'].tolist())
zeros_df
```

```python id="JZqIYI8G7HG7"
#create dictionary for every data to labelencoder data
user_dict = {}
food_dict = {}
for index, row in rating.iterrows():
  if(row['user_id'] not in user_dict):
    user_dict[row['user_id']] = row['User_ID']

  if(row['food_code'] not in food_dict):
    food_dict[row['food_code']] = row['Food_ID']
```

```python id="MepXY8Apnzjk"
total_user = len(zeros_df)
for index, row in tqdm(zeros_df.iterrows(), total=zeros_df.shape[0]):
  arr_rated = rating.loc[rating['user_id']==index,'food_code'].tolist()
  for rate in arr_rated:
    zeros_df.loc[index,rate]=rating.loc[(rating['user_id']==index)&(rating['food_code']==rate),'rating'].tolist()[0]

  not_rated = food_df.loc[(~food_df['food_code'].isin(arr_rated)),'food_code'].tolist()
  for rate in not_rated:
    try:

      zeros_df.loc[index,rate]=model.predict([np.array([user_dict[index]]), np.array([food_dict[rate]])])[0][0]
    except:
      zeros_df.loc[index,rate]=0
```

```python id="N9Pw901I3utK" colab={"base_uri": "https://localhost:8080/", "height": 439} outputId="77db1b4f-cd6e-4143-a46c-13a71915abab"
zeros_df
```

```python id="BctjcZSWEeTp"
food_name_dict = {}
for index, row in food_df.iterrows():
    if(row['food_code'] not in food_name_dict):
        food_name_dict[row['food_code']] = row['name']
```

<!-- #region id="bWlvSeeFE5iC" -->
**Give Recommendation to One User**
<!-- #endregion -->

```python id="NMM2alEtD2ay"
USER_ID = 129
food_code_arr = []
name_arr = []
prediction_arr = []
user_rated_food_code = rating.loc[rating['user_id']==USER_ID,'food_code'].tolist()
for key, value in zeros_df.loc[int(USER_ID)].items():
  if(int(key) not in user_rated_food_code):
    food_code_arr.append(int(key))
    name_arr.append(food_name_dict[int(key)])
    prediction_arr.append(value)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 385} id="gPs2bt2gFUMa" outputId="ea445ed1-900e-493a-cea7-bbb1c9c6268e"
print('Recommendation for user id : {}'.format(USER_ID))
rating.loc[rating['user_id']==USER_ID].merge(food_df[['food_code', 'name','category','type']],on='food_code').sort_values(by='rating',ascending=False)
```

```python id="Hq0wulL7FWNr" colab={"base_uri": "https://localhost:8080/", "height": 419} outputId="2a1099a2-64a4-4cae-a6bd-d9ce2bee425b"
recommendation_columns = {'food_code': food_code_arr,
                          'name': name_arr,
                          'predicted_rating': prediction_arr,
                          }

rec_for_user = pd.DataFrame(data=recommendation_columns)

# SORTED RATING FOR ONE USER
rec_for_user.sort_values(by='predicted_rating', ascending=False)
```

<!-- #region id="avntEJI_G9KD" -->
### Save predicted data to csv file
<!-- #endregion -->

```python id="v8p6Z5HR0_Be"
zeros_df.to_csv('recs.csv', index=True)
```
