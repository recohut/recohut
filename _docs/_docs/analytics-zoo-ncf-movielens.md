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

<!-- #region id="vWAyqrazbVkU" -->
# Analytics Zoo Recommendation Part 1
> Applying NCF on Movielens using Analytics Zoo library

- toc: true
- badges: true
- comments: true
- categories: [Movie, BigData, PySpark, AnalyticsZoo, NCF]
- author: "<a href='https://nbviewer.jupyter.org/github/intel-analytics/analytics-zoo/blob/master/apps/recommendation-ncf/ncf-explicit-feedback.ipynb'>Analytics Zoo</a>"
- image:
<!-- #endregion -->

<!-- #region id="5tJLYN9yYJxO" -->
## Introduction
<!-- #endregion -->

<!-- #region id="f4e-a8D8YSXJ" -->
NCF Recommender with Explict Feedback

In this notebook we demostrate how to build a neural network recommendation system, Neural Collaborative Filtering(NCF) with explict feedback. We use Recommender API in Analytics Zoo to build a model, and use optimizer of BigDL to train the model. 

The system ([Recommendation systems: Principles, methods and evaluation](http://www.sciencedirect.com/science/article/pii/S1110866515000341)) normally prompts the user through the system interface to provide ratings for items in order to construct and improve the model. The accuracy of recommendation depends on the quantity of ratings provided by the user.  

NCF([He, 2015](https://www.comp.nus.edu.sg/~xiangnan/papers/ncf.pdf)) leverages a multi-layer perceptrons to learn the user–item interaction function, at the mean time, NCF can express and generalize matrix factorization under its framework. includeMF(Boolean) is provided for users to build a NCF with or without matrix factorization. 

Data: 
* The dataset we used is movielens-1M ([link](https://grouplens.org/datasets/movielens/1m/)), which contains 1 million ratings from 6000 users on 4000 movies.  There're 5 levels of rating. We will try classify each (user,movie) pair into 5 classes and evaluate the effect of algortithms using Mean Absolute Error.  
  
References: 
* A Keras implementation of Movie Recommendation([notebook](https://github.com/ririw/ririw.github.io/blob/master/assets/Recommending%20movies.ipynb)) from the [blog](http://blog.richardweiss.org/2016/09/25/movie-embeddings.html).
* Nerual Collaborative filtering ([He, 2015](https://www.comp.nus.edu.sg/~xiangnan/papers/ncf.pdf))

Python interface:

```python
ncf = NeuralCF(user_count, item_count, class_num, user_embed=20, item_embed=20, hidden_layers=(40, 20, 10), include_mf=True, mf_embed=20)
```

- `user_count`: The number of users. Positive int.
- `item_count`: The number of classes. Positive int.
- `class_num`: The number of classes. Positive int.
- `user_embed`: Units of user embedding. Positive int. Default is 20.
- `item_embed`: itemEmbed Units of item embedding. Positive int. Default is 20.
- `hidden_layers`: Units of hidden layers for MLP. Tuple of positive int. Default is (40, 20, 10).
- `include_mf`: Whether to include Matrix Factorization. Boolean. Default is True.
- `mf_embed`: Units of matrix factorization embedding. Positive int. Default is 20.
<!-- #endregion -->

<!-- #region id="IPqTsm9pSckt" -->
## Installation
<!-- #endregion -->

<!-- #region id="u5UkOC-3SuwP" -->
### Install Java 8
<!-- #endregion -->

<!-- #region id="nY8_OcwpSyha" -->
Run the command on the colaboratory file to install jdk 1.8
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="8oH5g_58SScM" outputId="37ff9435-54d8-4846-f6b4-3ee62f6def22"
# Install jdk8
!apt-get install openjdk-8-jdk-headless -qq > /dev/null
# Set jdk environment path which enables you to run Pyspark in your Colab environment.
import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
!update-alternatives --set java /usr/lib/jvm/java-8-openjdk-amd64/jre/bin/java
```

<!-- #region id="zPqef37WS2r-" -->
### Install Analytics Zoo from pip
<!-- #endregion -->

<!-- #region id="Mo4najsbS5RX" -->
You can add the following command on your colab file to install the analytics-zoo via pip easily:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="RtHKGTcSS1cD" outputId="262ef946-24f1-419e-9fd8-b9276f29b3d9"
# Install latest release version of analytics-zoo 
# Installing analytics-zoo from pip will automatically install pyspark, bigdl, and their dependencies.
!pip install analytics-zoo
```

<!-- #region id="M8KiG7WhTAPa" -->
### Initialize context
<!-- #endregion -->

<!-- #region id="ZWWxCdrkTCqY" -->
Call init_nncontext() that will create a SparkContext with optimized performance configurations.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="Wy6CnH-8S81r" outputId="b36c7b07-c603-4619-d32c-16c27724ed05"
from zoo.common.nncontext import*

sc = init_nncontext()
```

<!-- #region id="nLQymd-_TbKx" -->
Analytics Zoo provides three Recommenders, including Wide and Deep (WND) model, Neural network-based Collaborative Filtering (NCF) model and Session Recommender model. Easy-to-use Keras-Style defined models which provides compile and fit methods for training. Alternatively, they could be fed into NNFrames or BigDL Optimizer.

WND and NCF recommenders can handle either explict or implicit feedback, given corresponding features.
<!-- #endregion -->

<!-- #region id="_KTBMy52T5he" -->
## Imports
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="nsib3Wg_T7E5" outputId="0e1481ac-014e-45a1-ec48-1156aae11038"
from zoo.pipeline.api.keras.layers import *
from zoo.models.recommendation import UserItemFeature
from zoo.models.recommendation import NeuralCF
from zoo.common.nncontext import init_nncontext
import matplotlib
from sklearn import metrics
from operator import itemgetter
from bigdl.util.common import *

import os
import numpy as np

matplotlib.use('agg')
import matplotlib.pyplot as plt
%pylab inline
```

<!-- #region id="Cl6H99sBTgOt" -->
## Download movielens dataset
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="wkKAeEORTPZ_" outputId="37526f44-10ec-4969-f223-9da3f3236fa6"
!wget http://files.grouplens.org/datasets/movielens/ml-1m.zip
!unzip ml-1m.zip
```

<!-- #region id="cizJHBZhVV15" -->
## Read the dataset
<!-- #endregion -->

```python id="vT8aZM0BT2oi"
def read_data_sets(data_dir):
  rating_files = os.path.join(data_dir,"ratings.dat")
  rating_list = [i.strip().split("::") for i in open(rating_files,"r").readlines()]    
  movielens_data = np.array(rating_list).astype(int)
  return movielens_data 

def get_id_pairs(data_dir):
	movielens_data = read_data_sets(data_dir)
	return movielens_data[:, 0:2]

def get_id_ratings(data_dir):
	movielens_data = read_data_sets(data_dir)
	return movielens_data[:, 0:3]
```

```python id="7n6vQz-3VR4I"
movielens_data = get_id_ratings("/content/ml-1m")
```

<!-- #region id="GCRIqd0hV2-5" -->
## Understand the data
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="nY3Bbk1xV8wL" outputId="45879150-5873-4857-e06c-313fcdebd89b"
min_user_id = np.min(movielens_data[:,0])
max_user_id = np.max(movielens_data[:,0])
min_movie_id = np.min(movielens_data[:,1])
max_movie_id = np.max(movielens_data[:,1])
rating_labels= np.unique(movielens_data[:,2])

print(movielens_data.shape)
print(min_user_id, max_user_id, min_movie_id, max_movie_id, rating_labels)
```

<!-- #region id="Vrs_dyQcVz8z" -->
Each record is in format of (userid, movieid, rating_score). UserIDs range between 1 and 6040. MovieIDs range between 1 and 3952. Ratings are made on a 5-star scale (whole-star ratings only). Counts of users and movies are recorded for later use.
<!-- #endregion -->

<!-- #region id="O-Ks_RL2WMT7" -->
## Transformation
<!-- #endregion -->

<!-- #region id="cnUhH-ksWOxh" -->
Transform original data into RDD of sample. We use optimizer of BigDL directly to train the model, it requires data to be provided in format of RDD(Sample). A Sample is a BigDL data structure which can be constructed using 2 numpy arrays, feature and label respectively. The API interface is Sample.from_ndarray(feature, label) Here, labels are tranformed into zero-based since original labels start from 1.
<!-- #endregion -->

```python id="gmD0lS0eVnK7"
def build_sample(user_id, item_id, rating):
    sample = Sample.from_ndarray(np.array([user_id, item_id]), np.array([rating]))
    return UserItemFeature(user_id, item_id, sample)
```

```python colab={"base_uri": "https://localhost:8080/"} id="BcwsB2SsWVBZ" outputId="e7a3f1f0-174a-4392-816d-6d5f9e4d6d17"
pairFeatureRdds = sc.parallelize(movielens_data).map(lambda x: build_sample(x[0], x[1], x[2]-1))
pairFeatureRdds.take(3)
```

<!-- #region id="wfqh_7FsWkrw" -->
## Split
<!-- #endregion -->

<!-- #region id="qw37Flz4Wm0F" -->
Randomly split the data into train (80%) and validation (20%)
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="plRhKQ0BWdD5" outputId="1629568b-f74c-4a67-cb88-774a265963a6"
trainPairFeatureRdds, valPairFeatureRdds = pairFeatureRdds.randomSplit([0.8, 0.2], seed= 1)
valPairFeatureRdds.cache()

train_rdd= trainPairFeatureRdds.map(lambda pair_feature: pair_feature.sample)
val_rdd= valPairFeatureRdds.map(lambda pair_feature: pair_feature.sample)
val_rdd.persist()
```

```python colab={"base_uri": "https://localhost:8080/"} id="PCWetBlaWuyb" outputId="bcaeaa9c-ef6e-4169-8eb1-6ea542a88e0f"
train_rdd.count()
```

```python colab={"base_uri": "https://localhost:8080/"} id="czo63lHuW2DB" outputId="655ac868-afc6-4dc1-bf94-92bb3feda2d4"
train_rdd.take(3)
```

<!-- #region id="KnONxSMsXARF" -->
## Build model
<!-- #endregion -->

<!-- #region id="IhpOgCvFXG2e" -->
In Analytics Zoo, it is simple to build NCF model by calling NeuralCF API. You need specify the user count, item count and class number according to your data, then add hidden layers as needed, you can also choose to include matrix factorization in the network. The model could be fed into an Optimizer of BigDL or NNClassifier of analytics-zoo. Please refer to the document for more details. In this example, we demostrate how to use optimizer of BigDL.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="4xMI7FZqW3zv" outputId="29be18e7-4be2-4ef4-8cdf-8dd642716e33"
ncf = NeuralCF(user_count=max_user_id, 
               item_count=max_movie_id, 
               class_num=5, 
               hidden_layers=[20, 10], 
               include_mf = False)
```

<!-- #region id="BVfHXNF2XMLp" -->
## Compile model
<!-- #endregion -->

<!-- #region id="R_Xgwv_MXUUI" -->
Compile model given specific optimizers, loss, as well as metrics for evaluation. Optimizer tries to minimize the loss of the neural net with respect to its weights/biases, over the training set. To create an Optimizer in BigDL, you want to at least specify arguments: model(a neural network model), criterion(the loss function), traing_rdd(training dataset) and batch size. Please refer to [ProgrammingGuide](https://bigdl-project.github.io/master/#ProgrammingGuide/optimization/) and [Optimizer](https://bigdl-project.github.io/master/#APIGuide/Optimizers/Optimizer/) for more details to create efficient optimizers.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="EpiU9RG_XKKM" outputId="7fbae0d6-4c77-437e-bfde-a2597a0d0655"
ncf.compile(optimizer= "adam",
            loss= "sparse_categorical_crossentropy",
            metrics=['accuracy'])
```

<!-- #region id="pOvfFZP3Xk0r" -->
## Collect logs
<!-- #endregion -->

<!-- #region id="ult1tu6QXnaX" -->
You can leverage tensorboard to see the summaries.
<!-- #endregion -->

```python id="E44IwOlzXjYW"
tmp_log_dir = create_tmp_path()
ncf.set_tensorboard(tmp_log_dir, "training_ncf")
```

<!-- #region id="IKipPFhsXqxS" -->
## Train the model
<!-- #endregion -->

```python id="4IHzRgUyXpu8"
ncf.fit(train_rdd, 
        nb_epoch= 10, 
        batch_size= 8000, 
        validation_data=val_rdd)
```

<!-- #region id="YLj6L9JPXxri" -->
## Prediction
<!-- #endregion -->

<!-- #region id="uzDas0oIX11i" -->
Zoo models make inferences based on the given data using model.predict(val_rdd) API. A result of RDD is returned. predict_class returns the predicted label.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="gUkaK6TUXuDJ" outputId="d2698771-6f5f-411c-f784-5f832806f570"
results = ncf.predict(val_rdd)
results.take(5)
```

```python colab={"base_uri": "https://localhost:8080/"} id="aLhHUm7sZm1Y" outputId="f85dcae9-e6eb-46b7-e748-27b031bff564"
results_class = ncf.predict_class(val_rdd)
results_class.take(5)
```

<!-- #region id="WqOfNL7gZrG6" -->
In Analytics Zoo, Recommender has provied 3 unique APIs to predict user-item pairs and make recommendations for users or items given candidates.
<!-- #endregion -->

<!-- #region id="9ffsw5z7aBlO" -->
### Predict for user item pairs
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="NXHMNrSsZnOD" outputId="6115e527-45ab-4a7b-f145-83f37839cda4"
userItemPairPrediction = ncf.predict_user_item_pair(valPairFeatureRdds)
for result in userItemPairPrediction.take(5): print(result)
```

<!-- #region id="SCgDNVQaZ-mJ" -->
### Recommend 3 items for each user given candidates in the feature RDDs
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="W2uMAeiOZ5RJ" outputId="7a558bf0-9d57-4922-a487-d1fe67c4f3a0"
userRecs = ncf.recommend_for_user(valPairFeatureRdds, 3)
for result in userRecs.take(5): print(result)
```

<!-- #region id="kOHBbcDAaH8K" -->
### Recommend 3 users for each item given candidates in the feature RDDs
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="CLp29k3kaHB-" outputId="ba1442fc-6485-4871-952a-e4c6195f1855"
itemRecs = ncf.recommend_for_item(valPairFeatureRdds, 3)
for result in itemRecs.take(5): print(result)
```

<!-- #region id="FRJoSBJGaO43" -->
## Evaluation
<!-- #endregion -->

<!-- #region id="t8AOXDp9aSwD" -->
Plot the train and validation loss curves
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 407} id="MbS3TGZbaKva" outputId="344d1ab3-415e-4c82-a2a5-3b17988ee2f5"
#retrieve train and validation summary object and read the loss data into ndarray's. 
train_loss = np.array(ncf.get_train_summary("Loss"))
val_loss = np.array(ncf.get_validation_summary("Loss"))
#plot the train and validation curves
# each event data is a tuple in form of (iteration_count, value, timestamp)
plt.figure(figsize = (12,6))
plt.plot(train_loss[:,0],train_loss[:,1],label='train loss')
plt.plot(val_loss[:,0],val_loss[:,1],label='val loss',color='green')
plt.scatter(val_loss[:,0],val_loss[:,1],color='green')
plt.legend();
plt.xlim(0,train_loss.shape[0]+10)
plt.grid(True)
plt.title("loss")
```

<!-- #region id="-Yh804RoaW8M" -->
Plot accuracy
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 390} id="F0JtfjEMaVLc" outputId="de3e140b-b46c-4e32-ca39-680be11f689c"
plt.figure(figsize = (12,6))
top1 = np.array(ncf.get_validation_summary("Top1Accuracy"))
plt.plot(top1[:,0],top1[:,1],label='top1')
plt.title("top1 accuracy")
plt.grid(True)
plt.legend();
```
