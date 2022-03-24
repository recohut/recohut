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
# Analytics Zoo Recommendation Part 3
> Applying Wide&Deep on MovieLens using Analytics Zoo library

- toc: true
- badges: true
- comments: true
- categories: [movie, BigData, PySpark, AnalyticsZoo, Wide&Deep]
- author: "<a href='https://nbviewer.jupyter.org/github/intel-analytics/analytics-zoo/blob/master/apps/recommendation-wide-n-deep/wide_n_deep.ipynb'>Analytics Zoo</a>"
- image:
<!-- #endregion -->

<!-- #region id="5tJLYN9yYJxO" -->
## Introduction
<!-- #endregion -->

<!-- #region id="f4e-a8D8YSXJ" -->
Wide and Deep Learning Model, proposed by Google in 2016, is a DNN-Linear mixed model. Wide and deep learning has been used for Google App Store for their app recommendation.

In this tutorial, we use Recommender API of Analytics Zoo to build a wide linear model and a deep neural network, which is called Wide&Deep model, and use optimizer of BigDL to train the neural network. Wide&Deep model combines the strength of memorization and generalization. It's useful for generic large-scale regression and classification problems with sparse input features (e.g., categorical features with a large number of possible feature values).

Python interface:

```python
wide_and_deep = WideAndDeep(class_num, column_info, model_type="wide_n_deep", hidden_layers=(40, 20, 10))
```

- `class_num`: The number of classes. Positive int.
- `column_info`: An instance of ColumnFeatureInfo.
- `model_type`: String. 'wide', 'deep' and 'wide_n_deep' are supported. Default is 'wide_n_deep'.
- `hidden_layers`: Units of hidden layers for the deep model. Tuple of positive int. Default is (40, 20, 10).
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

```python colab={"base_uri": "https://localhost:8080/"} id="8oH5g_58SScM" outputId="3e9869fd-973c-49f4-81a1-e9ac91fa9417"
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

```python colab={"base_uri": "https://localhost:8080/"} id="RtHKGTcSS1cD" outputId="5c5d968e-63b3-45fb-c9b4-8015af886e62"
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

```python colab={"base_uri": "https://localhost:8080/"} id="Wy6CnH-8S81r" outputId="ca275bf1-95fc-4557-b72a-f146b05d1b5d"
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

```python colab={"base_uri": "https://localhost:8080/"} id="nsib3Wg_T7E5" outputId="242f446c-1e3c-4f69-ac50-1ddf5fb504cb"
from zoo.models.recommendation import *
from zoo.models.recommendation.utils import *
from zoo.common.nncontext import init_nncontext

sqlContext = SQLContext(sc)
from pyspark.sql.types import *
from pyspark.sql import Row

import os
import sys
import datetime as dt
import numpy as np
from sklearn import preprocessing

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
%pylab inline
```

<!-- #region id="Cl6H99sBTgOt" -->
## Download movielens dataset
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="wkKAeEORTPZ_" outputId="39ad851a-50ed-4082-f5d2-1ae9473bbdc3"
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
	return movielens_data[:100000, 0:3]
```

```python id="7n6vQz-3VR4I"
movielens_data = get_id_ratings("/content/ml-1m")
```

<!-- #region id="GCRIqd0hV2-5" -->
## Understand the data
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="nY3Bbk1xV8wL" outputId="03c131c6-3a21-40d1-f3e0-cc74b9f14b2c"
min_user_id = np.min(movielens_data[:,0])
max_user_id = np.max(movielens_data[:,0])
min_movie_id = np.min(movielens_data[:,1])
max_movie_id = np.max(movielens_data[:,1])
rating_labels= np.unique(movielens_data[:,2])

print(movielens_data.shape)
print(min_user_id, max_user_id, min_movie_id, max_movie_id, rating_labels)
```

<!-- #region id="O-Ks_RL2WMT7" -->
## Transformation
<!-- #endregion -->

<!-- #region id="cnUhH-ksWOxh" -->
Transform ratings into dataframe, read user and item data into dataframes. Transform labels to zero-based since the original labels start from 1.
<!-- #endregion -->

```python id="gmD0lS0eVnK7"
Rating = Row("userId", "itemId", "label")
User = Row("userId", "gender", "age" ,"occupation")
Item = Row("itemId", "title" ,"genres")

ratings = sc.parallelize(movielens_data)\
    .map(lambda l: (int(l[0]), int(l[1]), int(l[2])-1))\
    .map(lambda r: Rating(*r))
ratingDF = sqlContext.createDataFrame(ratings)

users= sc.textFile("/content/ml-1m/users.dat")\
    .map(lambda l: l.split("::")[0:4])\
    .map(lambda l: (int(l[0]), l[1], int(l[2]), int(l[3])))\
    .map(lambda r: User(*r))
userDF = sqlContext.createDataFrame(users)

items = sc.textFile("/content/ml-1m/movies.dat")\
    .map(lambda l: l.split("::")[0:3])\
    .map(lambda l: (int(l[0]), l[1], l[2].split('|')[0]))\
    .map(lambda r: Item(*r))
itemDF = sqlContext.createDataFrame(items)
```

<!-- #region id="5GF3VrUfrGUj" -->
Join data together, and transform data. For example, gender is going be used as categorical feature, occupation and gender will be used as crossed features.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="3vIUyR2YrKi0" outputId="f10cc813-17dc-4c24-e2f2-54ed69216e34"
from pyspark.sql.functions import col, udf

gender_udf = udf(lambda gender: categorical_from_vocab_list(gender, ["F", "M"], start=1))
bucket_cross_udf = udf(lambda feature1, feature2: hash_bucket(str(feature1) + "_" + str(feature2), bucket_size=100))
genres_list = ["Crime", "Romance", "Thriller", "Adventure", "Drama", "Children's",
      "War", "Documentary", "Fantasy", "Mystery", "Musical", "Animation", "Film-Noir", "Horror",
      "Western", "Comedy", "Action", "Sci-Fi"]
genres_udf = udf(lambda genres: categorical_from_vocab_list(genres, genres_list, start=1))
     
allDF = ratingDF.join(userDF, ["userId"]).join(itemDF, ["itemId"]) \
        .withColumn("gender", gender_udf(col("gender")).cast("int")) \
        .withColumn("age-gender", bucket_cross_udf(col("age"), col("gender")).cast("int")) \
        .withColumn("genres", genres_udf(col("genres")).cast("int"))
allDF.show(5)
```

<!-- #region id="pi1uCpp3rRrG" -->
Speficy data feature information shared by the WideAndDeep model and its feature generation. Here, we use occupation gender for wide base part, age and gender crossed as wide cross part, genres and gender as indicators, userid and itemid for embedding.
<!-- #endregion -->

```python id="xyc85kqqrQ5y"
bucket_size = 100
column_info = ColumnFeatureInfo(
            wide_base_cols=["occupation", "gender"],
            wide_base_dims=[21, 3],
            wide_cross_cols=["age-gender"],
            wide_cross_dims=[bucket_size],
            indicator_cols=["genres", "gender"],
            indicator_dims=[19, 3],
            embed_cols=["userId", "itemId"],
            embed_in_dims=[max_user_id, max_movie_id],
            embed_out_dims=[64, 64],
            continuous_cols=["age"])
```

<!-- #region id="4Ct0py6Qrjki" -->
Transform data to RDD of Sample. We use optimizer of BigDL directly to train the model, it requires data to be provided in format of RDD(Sample). A Sample is a BigDL data structure which can be constructed using 2 numpy arrays, feature and label respectively. The API interface is Sample.from_ndarray(feature, label). Wide&Deep model need two input tensors, one is SparseTensor for the Wide model, another is a DenseTensor for the Deep model.
<!-- #endregion -->

<!-- #region id="qw37Flz4Wm0F" -->
Randomly split the data into train (80%) and validation (20%)
<!-- #endregion -->

```python id="Q2XqW5wMrlWR"
rdds = allDF.rdd.map(lambda row: to_user_item_feature(row, column_info))
trainPairFeatureRdds, valPairFeatureRdds = rdds.randomSplit([0.8, 0.2], seed= 1)
valPairFeatureRdds.persist()

train_data= trainPairFeatureRdds.map(lambda pair_feature: pair_feature.sample)
test_data= valPairFeatureRdds.map(lambda pair_feature: pair_feature.sample)
```

```python colab={"base_uri": "https://localhost:8080/"} id="PCWetBlaWuyb" outputId="7202974b-48b9-45c4-e1a3-f661a0e6d74b"
train_data.count()
```

```python colab={"base_uri": "https://localhost:8080/"} id="czo63lHuW2DB" outputId="088e28e5-ee19-4adc-c374-f1720c4264a1"
train_data.take(3)
```

<!-- #region id="KnONxSMsXARF" -->
## Build model
<!-- #endregion -->

<!-- #region id="IhpOgCvFXG2e" -->
In Analytics Zoo, it is simple to build Wide&Deep model by calling WideAndDeep API. You need specify model type, and class number, as well as column information of features according to your data. You can also change other default parameters in the network, like hidden layers. The model could be fed into an Optimizer of BigDL or NNClassifier of analytics-zoo. Please refer to the document for more details. In this example, we demostrate how to use optimizer of BigDL.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="4xMI7FZqW3zv" outputId="b6d03687-d5f8-4be9-a982-24d7be4aff18"
wide_n_deep = WideAndDeep(5, column_info, "wide_n_deep")
```

<!-- #region id="AgFrVcNusKGX" -->
### Create optimizer and train the model
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="4IHzRgUyXpu8" outputId="52a786bc-29c0-4c49-dc86-f1a20326adb5"
wide_n_deep.compile(optimizer = "adam",
                    loss= "sparse_categorical_crossentropy",
                    metrics=['accuracy'])
```

```python id="pBFimH_GsOPP"
tmp_log_dir = create_tmp_path()
wide_n_deep.set_tensorboard(tmp_log_dir, "training_wideanddeep")
```

<!-- #region id="l0Ev3lGSsP3d" -->
Train the network. Wait some time till it finished.. Voila! You've got a trained model


<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="nnlnakrOsRkx" outputId="b4dc683e-0e02-44ad-b5c0-99e342911f01"
%%time
# Boot training process
wide_n_deep.fit(train_data,
                batch_size = 20000,
                nb_epoch = 10,
                validation_data = test_data)
print("Optimization Done.")
```

<!-- #region id="YLj6L9JPXxri" -->
## Prediction
<!-- #endregion -->

<!-- #region id="uzDas0oIX11i" -->
Zoo models make inferences based on the given data using model.predict(val_rdd) API. A result of RDD is returned. predict_class returns the predicted label.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="gUkaK6TUXuDJ" outputId="125955be-84b4-4b3a-ae93-7a1e8452c33f"
results = wide_n_deep.predict(test_data)
results.take(5)
```

```python colab={"base_uri": "https://localhost:8080/"} id="aLhHUm7sZm1Y" outputId="f5052c54-7263-4f1d-a5fe-6789ccf56137"
results_class = wide_n_deep.predict_class(test_data)
results_class.take(5)
```

<!-- #region id="WqOfNL7gZrG6" -->
In Analytics Zoo, Recommender has provied 3 unique APIs to predict user-item pairs and make recommendations for users or items given candidates.
<!-- #endregion -->

<!-- #region id="9ffsw5z7aBlO" -->
### Predict for user item pairs
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="NXHMNrSsZnOD" outputId="1b587a28-d9a4-4dcb-b4f1-c0ad6594c968"
userItemPairPrediction = wide_n_deep.predict_user_item_pair(valPairFeatureRdds)
for result in userItemPairPrediction.take(5): print(result)
```

<!-- #region id="SCgDNVQaZ-mJ" -->
### Recommend 3 items for each user given candidates in the feature RDDs
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="W2uMAeiOZ5RJ" outputId="10dd1a15-c7c9-4afc-d27b-10077807ed43"
userRecs = wide_n_deep.recommend_for_user(valPairFeatureRdds, 3)
for result in userRecs.take(5): print(result)
```

<!-- #region id="kOHBbcDAaH8K" -->
### Recommend 3 users for each item given candidates in the feature RDDs
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="CLp29k3kaHB-" outputId="2b140c1f-7438-40da-84c9-1f5e6dc62acb"
itemRecs = wide_n_deep.recommend_for_item(valPairFeatureRdds, 3)
for result in itemRecs.take(5): print(result)
```

<!-- #region id="FRJoSBJGaO43" -->
## Evaluation
<!-- #endregion -->

<!-- #region id="t8AOXDp9aSwD" -->
Plot the train and validation loss curves
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 407} id="MbS3TGZbaKva" outputId="4d13a6e9-4b08-41e5-d167-98869c796441"
#retrieve train and validation summary object and read the loss data into ndarray's. 
train_loss = np.array(wide_n_deep.get_train_summary("Loss"))
val_loss = np.array(wide_n_deep.get_validation_summary("Loss"))
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

```python colab={"base_uri": "https://localhost:8080/", "height": 407} id="F0JtfjEMaVLc" outputId="6f4ac310-aee2-4988-867a-2b25f988e33e"
plt.figure(figsize = (12,6))
top1 = np.array(wide_n_deep.get_validation_summary("Top1Accuracy"))
plt.plot(top1[:,0],top1[:,1],label='top1')
plt.title("top1 accuracy")
plt.grid(True)
plt.legend();
plt.xlim(0,train_loss.shape[0]+10)
```

```python id="4WwZJ_OB01SN"
# %reload_ext tensorboard
# %tensorboard --logdir $tmp_log_dir/training_wideanddeep/
```

<!-- #region id="gK07c-Mh2j2X" -->
<!-- #endregion -->
