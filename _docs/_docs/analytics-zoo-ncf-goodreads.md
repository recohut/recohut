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
# Analytics Zoo Recommendation Part 2
> Applying NCF on Goodreads using Analytics Zoo library

- toc: true
- badges: true
- comments: true
- categories: [Book, BigData, PySpark, AnalyticsZoo, NCF]
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
* Goodreads book ratings dataset 
  
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

```python colab={"base_uri": "https://localhost:8080/"} id="8oH5g_58SScM" outputId="dfb99804-1975-4159-ef51-d23e021ec621"
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

```python colab={"base_uri": "https://localhost:8080/"} id="RtHKGTcSS1cD" outputId="00364579-40ef-4fb9-e770-70428dc04ddd"
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

```python colab={"base_uri": "https://localhost:8080/"} id="Wy6CnH-8S81r" outputId="06a6e670-b16f-487d-f2a4-857f6d157603"
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

```python colab={"base_uri": "https://localhost:8080/"} id="nsib3Wg_T7E5" outputId="7ebed2eb-383e-4dda-b8c9-6a8b14fa9495"
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
from sklearn import preprocessing

matplotlib.use('agg')
import matplotlib.pyplot as plt
%pylab inline
```

<!-- #region id="Cl6H99sBTgOt" -->
## Download goodreads dataset
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="wkKAeEORTPZ_" outputId="17ec66dd-dda9-4934-e57f-99692399c955"
!wget https://github.com/sparsh-ai/reco-data/raw/master/goodreads/ratings.csv
```

<!-- #region id="cizJHBZhVV15" -->
## Read the dataset
<!-- #endregion -->

```python id="vT8aZM0BT2oi"
def read_data_sets(data_dir):
  rating_files = os.path.join(data_dir,"ratings.csv")
  rating_list = [i.strip().split(",") for i in open(rating_files,"r").readlines()]    
  goodreads_data = np.array(rating_list[1:]).astype(int)
  return goodreads_data 

def get_id_pairs(data_dir):
	goodreads_data = read_data_sets(data_dir)
	return goodreads_data[:, 0:2]

def get_id_ratings(data_dir):
  goodreads_data = read_data_sets(data_dir)
  le_user = preprocessing.LabelEncoder()
  goodreads_data[:, 0] = le_user.fit_transform(goodreads_data[:, 0])
  le_item = preprocessing.LabelEncoder()
  goodreads_data[:, 1] = le_item.fit_transform(goodreads_data[:, 1])
  return goodreads_data[:, 0:3]
```

```python id="7n6vQz-3VR4I"
goodreads_data = get_id_ratings("/content")
```

<!-- #region id="GCRIqd0hV2-5" -->
## Understand the data
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="nY3Bbk1xV8wL" outputId="61215aa9-a1a2-4f01-9de1-798c0316afe0"
min_user_id = np.min(goodreads_data[:,0])
max_user_id = np.max(goodreads_data[:,0])
min_book_id = np.min(goodreads_data[:,1])
max_book_id = np.max(goodreads_data[:,1])
rating_labels= np.unique(goodreads_data[:,2])

print(goodreads_data.shape)
print(min_user_id, max_user_id, min_book_id, max_book_id, rating_labels)
```

<!-- #region id="Vrs_dyQcVz8z" -->
Each record is in format of (userid, bookid, rating_score). Both UserIDs and BookIDs range between 0 and 4999. Ratings are made on a 5-star scale (whole-star ratings only). Counts of users and books are recorded for later use.
<!-- #endregion -->

<!-- #region id="O-Ks_RL2WMT7" -->
## Transformation
<!-- #endregion -->

<!-- #region id="cnUhH-ksWOxh" -->
Transform original data into RDD of sample. We use optimizer of BigDL directly to train the model, it requires data to be provided in format of RDD(Sample). A Sample is a BigDL data structure which can be constructed using 2 numpy arrays, feature and label respectively. The API interface is Sample.from_ndarray(feature, label).
<!-- #endregion -->

```python id="gmD0lS0eVnK7"
def build_sample(user_id, item_id, rating):
    sample = Sample.from_ndarray(np.array([user_id, item_id]), np.array([rating]))
    return UserItemFeature(user_id, item_id, sample)
```

```python colab={"base_uri": "https://localhost:8080/"} id="BcwsB2SsWVBZ" outputId="fc09f520-5afa-4e58-dc2a-bd7c2340ffcd"
pairFeatureRdds = sc.parallelize(goodreads_data).map(lambda x: build_sample(x[0], x[1], x[2]-1))
pairFeatureRdds.take(3)
```

<!-- #region id="wfqh_7FsWkrw" -->
## Split
<!-- #endregion -->

<!-- #region id="qw37Flz4Wm0F" -->
Randomly split the data into train (80%) and validation (20%)
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="plRhKQ0BWdD5" outputId="d7d7cafb-82f8-46fd-9537-aac438ce823c"
trainPairFeatureRdds, valPairFeatureRdds = pairFeatureRdds.randomSplit([0.8, 0.2], seed= 1)
valPairFeatureRdds.cache()

train_rdd= trainPairFeatureRdds.map(lambda pair_feature: pair_feature.sample)
val_rdd= valPairFeatureRdds.map(lambda pair_feature: pair_feature.sample)
val_rdd.persist()
```

```python colab={"base_uri": "https://localhost:8080/"} id="PCWetBlaWuyb" outputId="e1774d82-1356-4222-830e-5f182c495ccc"
train_rdd.count()
```

```python colab={"base_uri": "https://localhost:8080/"} id="czo63lHuW2DB" outputId="caf5f0d3-427d-4ffe-df95-9d930908e300"
train_rdd.take(3)
```

<!-- #region id="KnONxSMsXARF" -->
## Build model
<!-- #endregion -->

<!-- #region id="IhpOgCvFXG2e" -->
In Analytics Zoo, it is simple to build NCF model by calling NeuralCF API. You need specify the user count, item count and class number according to your data, then add hidden layers as needed, you can also choose to include matrix factorization in the network. The model could be fed into an Optimizer of BigDL or NNClassifier of analytics-zoo. Please refer to the document for more details. In this example, we demostrate how to use optimizer of BigDL.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="4xMI7FZqW3zv" outputId="f3f248e2-c626-4a4a-ec98-3bd58c96436c"
ncf = NeuralCF(user_count=max_user_id, 
               item_count=max_book_id, 
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

```python colab={"base_uri": "https://localhost:8080/"} id="EpiU9RG_XKKM" outputId="d730c086-e91e-44f3-b0fc-81d66a32d509"
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
        batch_size= 5000,
        validation_data=val_rdd)
```

<!-- #region id="YLj6L9JPXxri" -->
## Prediction
<!-- #endregion -->

<!-- #region id="uzDas0oIX11i" -->
Zoo models make inferences based on the given data using model.predict(val_rdd) API. A result of RDD is returned. predict_class returns the predicted label.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="gUkaK6TUXuDJ" outputId="16018ece-c981-4a17-ad43-014a1cae71e6"
results = ncf.predict(val_rdd)
results.take(5)
```

```python colab={"base_uri": "https://localhost:8080/"} id="aLhHUm7sZm1Y" outputId="b36d73bf-cf28-4197-9f5d-db752ddc7ace"
results_class = ncf.predict_class(val_rdd)
results_class.take(5)
```

<!-- #region id="WqOfNL7gZrG6" -->
In Analytics Zoo, Recommender has provied 3 unique APIs to predict user-item pairs and make recommendations for users or items given candidates.
<!-- #endregion -->

<!-- #region id="9ffsw5z7aBlO" -->
### Predict for user item pairs
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="NXHMNrSsZnOD" outputId="df07c238-f264-4550-cfe8-2c09b9eb8073"
userItemPairPrediction = ncf.predict_user_item_pair(valPairFeatureRdds)
for result in userItemPairPrediction.take(5): print(result)
```

<!-- #region id="SCgDNVQaZ-mJ" -->
### Recommend 3 items for each user given candidates in the feature RDDs
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="W2uMAeiOZ5RJ" outputId="e09e990f-e600-4541-ba87-77f455cea036"
userRecs = ncf.recommend_for_user(valPairFeatureRdds, 3)
for result in userRecs.take(5): print(result)
```

<!-- #region id="kOHBbcDAaH8K" -->
### Recommend 3 users for each item given candidates in the feature RDDs
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="CLp29k3kaHB-" outputId="faf205ed-dcbe-4e27-a988-fa83f56fc0ef"
itemRecs = ncf.recommend_for_item(valPairFeatureRdds, 3)
for result in itemRecs.take(5): print(result)
```

<!-- #region id="FRJoSBJGaO43" -->
## Evaluation
<!-- #endregion -->

<!-- #region id="t8AOXDp9aSwD" -->
Plot the train and validation loss curves
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 407} id="MbS3TGZbaKva" outputId="1f7a9581-9feb-4022-95a3-bf013ab06fb7"
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

```python colab={"base_uri": "https://localhost:8080/", "height": 390} id="F0JtfjEMaVLc" outputId="2c22ed0b-8858-4180-bbe8-3f2d230210fd"
plt.figure(figsize = (12,6))
top1 = np.array(ncf.get_validation_summary("Top1Accuracy"))
plt.plot(top1[:,0],top1[:,1],label='top1')
plt.title("top1 accuracy")
plt.grid(True)
plt.legend();
```
