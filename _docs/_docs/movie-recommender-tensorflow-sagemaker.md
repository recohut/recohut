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

<!-- #region id="Cl3onQXd-n8o" -->
# Movie recommender using Tensorflow in Sagemaker
> Tutorial on ow to build and deploy a customized recommender system using Neural Collaborative Filtering model in TensorFlow 2.0 on Amazon SageMaker, based on which you can customize further accordingly.

- toc: true
- badges: true
- comments: true
- categories: [Sagemaker, AWS, Tensorflow, NCF, Movie]
- image:
<!-- #endregion -->

<!-- #region id="DRXewMvhwdso" -->
Recommender systems have been used to tailor customer experience on online platforms. Amazon Personalize is a fully-managed service that makes it easy to develop recommender system solutions; it automatically examines the data, performs feature and algorithm selection, optimizes the model based on your data, and deploys and hosts the model for real-time recommendation inference. However, due to unique constraints in some domains, sometimes recommender systems need to be custom-built.

In this project, I will walk you through how to build and deploy a customized recommender system using Neural Collaborative Filtering model in TensorFlow 2.0 on Amazon SageMaker, based on which you can customize further accordingly.
<!-- #endregion -->

<!-- #region id="VS3yvTpWwmWM" -->
## Data Preparation

1. download MovieLens dataset into ml-latest-small directory
2. split the data into training and testing sets
3. perform negative sampling
4. calculate statistics needed to train the NCF model
5. upload data onto S3 bucket
<!-- #endregion -->

<!-- #region id="wa_uFMxHxpI0" -->
## Download dataset
<!-- #endregion -->

```bash id="Dc_xnzoZws8I"
# delete the data directory if exists
rm -r ml-latest-small

# download movielens small dataset
curl -O http://files.grouplens.org/datasets/movielens/ml-latest-small.zip

# unzip into data directory
unzip ml-latest-small.zip
rm ml-latest-small.zip
```

```python id="LabiJ35Mws4s"
!cat ml-latest-small/README.txt
```

<!-- #region id="N03Sx493xsWq" -->
## Read data and perform train and test split
<!-- #endregion -->

```python id="GcVGr7tJws1q"
# requirements
import os
import boto3
import sagemaker
import numpy as np
import pandas as pd
```

```python id="WVpC-4a5wsyf"
# read rating data
fpath = './ml-latest-small/ratings.csv'
df = pd.read_csv(fpath)
```

```python id="jN_Ztxgnw5n3"
# let's see what the data look like
df.head(2)
```

```python id="p3FkjDvbw8aH"
# understand what's the maximum number of hold out portion should be
df.groupby('userId').movieId.nunique().min()
```

<!-- #region id="f3zNnfwUw-YZ" -->
Note: Since the "least active" user has 20 ratings, for our testing set, let's hold out 10 items for every user so that the max test set portion is 50%.


<!-- #endregion -->

```python id="39uMdiAyw8Wl"
def train_test_split(df, holdout_num):
    """ perform training/testing split
    
    @param df: dataframe
    @param holdhout_num: number of items to be held out
    
    @return df_train: training data
    @return df_test testing data
    
    """
    # first sort the data by time
    df = df.sort_values(['userId', 'timestamp'], ascending=[True, False])
    
    # perform deep copy on the dataframe to avoid modification on the original dataframe
    df_train = df.copy(deep=True)
    df_test = df.copy(deep=True)
    
    # get test set
    df_test = df_test.groupby(['userId']).head(holdout_num).reset_index()
    
    # get train set
    df_train = df_train.merge(
        df_test[['userId', 'movieId']].assign(remove=1),
        how='left'
    ).query('remove != 1').drop('remove', 1).reset_index(drop=True)
    
    # sanity check to make sure we're not duplicating/losing data
    assert len(df) == len(df_train) + len(df_test)
    
    return df_train, df_test
```

```python id="C6eTtL5Bw8Td"
df_train, df_test = train_test_split(df, 10)
```

<!-- #region id="AMWzwhwbxHMy" -->
## Perform negative sampling
Assuming if a user rating an item is a positive label, there is no negative sample in the dataset, which is not possible for model training. Therefore, we random sample n items from the unseen movie list for every user to provide the negative samples.
<!-- #endregion -->

```python id="LS3i-bR2w8Qs"
def negative_sampling(user_ids, movie_ids, items, n_neg):
    """This function creates n_neg negative labels for every positive label
    
    @param user_ids: list of user ids
    @param movie_ids: list of movie ids
    @param items: unique list of movie ids
    @param n_neg: number of negative labels to sample
    
    @return df_neg: negative sample dataframe
    
    """
    
    neg = []
    ui_pairs = zip(user_ids, movie_ids)
    records = set(ui_pairs)
    
    # for every positive label case
    for (u, i) in records:
        # generate n_neg negative labels
        for _ in range(n_neg):
            # if the randomly sampled movie exists for that user
            j = np.random.choice(items)
            while(u, j) in records:
                # resample
                j = np.random.choice(items)
            neg.append([u, j, 0])
    # conver to pandas dataframe for concatenation later
    df_neg = pd.DataFrame(neg, columns=['userId', 'movieId', 'rating'])
    
    return df_neg
```

```python id="wlv_i-pXxK-x"
# create negative samples for training set
neg_train = negative_sampling(
    user_ids=df_train.userId.values, 
    movie_ids=df_train.movieId.values,
    items=df.movieId.unique(),
    n_neg=5
)
```

```python id="KzlSvLQRxK7J"
print(f'created {neg_train.shape[0]:,} negative samples')
```

```python id="exvuVgpBxK3y"
df_train = df_train[['userId', 'movieId']].assign(rating=1)
df_test = df_test[['userId', 'movieId']].assign(rating=1)

df_train = pd.concat([df_train, neg_train], ignore_index=True)
```

<!-- #region id="MalrZRfNxRSK" -->
## Calulate statistics for our understanding and model training
<!-- #endregion -->

```python id="3vPf9fFIxOEi"
def get_unique_count(df):
    """calculate unique user and movie counts"""
    return df.userId.nunique(), df.movieId.nunique()
```

```python id="E-b56beyxOBI"
# unique number of user and movie in the whole dataset
get_unique_count(df)
```

```python id="E16VzAefxUNd"
print('training set shape', get_unique_count(df_train))
print('testing set shape', get_unique_count(df_test))
```

```python id="jmo4i1RAxUKB"
# number of unique user and number of unique item/movie
n_user, n_item = get_unique_count(df_train)

print("number of unique users", n_user)
print("number of unique items", n_item)
```

```python id="RZmbSlwmxUGT"
# save the variable for the model training notebook
# -----
# read about `store` magic here: 
# https://ipython.readthedocs.io/en/stable/config/extensions/storemagic.html

%store n_user
%store n_item
```

<!-- #region id="vHYszqdExcQ1" -->
## Preprocess data and upload them onto S3
<!-- #endregion -->

```python id="pr_bRHSKxZt7"
# get current session region
session = boto3.session.Session()
region = session.region_name
print(f'currently in {region}')
```

```python id="dsex89BwxZpl"
# use the default sagemaker s3 bucket to store processed data
# here we figure out what that default bucket name is 
sagemaker_session = sagemaker.Session()
bucket_name = sagemaker_session.default_bucket()
print(bucket_name)  # bucket name format: "sagemaker-{region}-{aws_account_id}"
```

<!-- #region id="DD3cv6V0xhbX" -->
upload data to the bucket


<!-- #endregion -->

```python id="fbkVA2M-xUCP"
# save data locally first
dest = 'ml-latest-small/s3'
train_path = os.path.join(dest, 'train.npy')
test_path = os.path.join(dest, 'test.npy')

!mkdir {dest}
np.save(train_path, df_train.values)
np.save(test_path, df_test.values)

# upload to S3 bucket (see the bucket name above)
sagemaker_session.upload_data(train_path, key_prefix='data')
sagemaker_session.upload_data(test_path, key_prefix='data')
```

<!-- #region id="yGO5-59mxjrY" -->
## Train and Deploy a Neural Collaborative Filtering Model

1. inspect the training script ncf.py
2. train a model using Tensorflow Estimator
3. deploy and host the trained model as an endpoint using Amazon SageMaker Hosting Services
4. perform batch inference by calling the model endpoint
<!-- #endregion -->

```python id="eK_p08Lbx901"
# import requirements
import os
import json
import sagemaker
import numpy as np
import pandas as pd
import tensorflow as tf
from sagemaker import get_execution_role
from sagemaker.tensorflow import TensorFlow

# get current SageMaker session's execution role and default bucket name
sagemaker_session = sagemaker.Session()

role = get_execution_role()
print("execution role ARN:", role)

bucket_name = sagemaker_session.default_bucket()
print("default bucket name:", bucket_name)
```

```python id="dzlD1MeTx9xO"
# specify the location of the training data
training_data_uri = os.path.join(f's3://{bucket_name}', 'data')
```

```python id="I7OD4cNfe7H6"
%%writefile ncf.py

"""

 Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 SPDX-License-Identifier: MIT-0
 
 Permission is hereby granted, free of charge, to any person obtaining a copy of this
 software and associated documentation files (the "Software"), to deal in the Software
 without restriction, including without limitation the rights to use, copy, modify,
 merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
 INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
 PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""


import tensorflow as tf
import argparse
import os
import numpy as np
import json


# for data processing
def _load_training_data(base_dir):
    """ load training data """
    df_train = np.load(os.path.join(base_dir, 'train.npy'))
    user_train, item_train, y_train = np.split(np.transpose(df_train).flatten(), 3)
    return user_train, item_train, y_train


def batch_generator(x, y, batch_size, n_batch, shuffle, user_dim, item_dim):
    """ batch generator to supply data for training and testing """

    user_df, item_df = x

    counter = 0
    training_index = np.arange(user_df.shape[0])

    if shuffle:
        np.random.shuffle(training_index)

    while True:
        batch_index = training_index[batch_size*counter:batch_size*(counter+1)]
        user_batch = tf.one_hot(user_df[batch_index], depth=user_dim)
        item_batch = tf.one_hot(item_df[batch_index], depth=item_dim)
        y_batch = y[batch_index]
        counter += 1
        yield [user_batch, item_batch], y_batch

        if counter == n_batch:
            if shuffle:
                np.random.shuffle(training_index)
            counter = 0


# network
def _get_user_embedding_layers(inputs, emb_dim):
    """ create user embeddings """
    user_gmf_emb = tf.keras.layers.Dense(emb_dim, activation='relu')(inputs)

    user_mlp_emb = tf.keras.layers.Dense(emb_dim, activation='relu')(inputs)

    return user_gmf_emb, user_mlp_emb


def _get_item_embedding_layers(inputs, emb_dim):
    """ create item embeddings """
    item_gmf_emb = tf.keras.layers.Dense(emb_dim, activation='relu')(inputs)

    item_mlp_emb = tf.keras.layers.Dense(emb_dim, activation='relu')(inputs)

    return item_gmf_emb, item_mlp_emb


def _gmf(user_emb, item_emb):
    """ general matrix factorization branch """
    gmf_mat = tf.keras.layers.Multiply()([user_emb, item_emb])

    return gmf_mat


def _mlp(user_emb, item_emb, dropout_rate):
    """ multi-layer perceptron branch """
    def add_layer(dim, input_layer, dropout_rate):
        hidden_layer = tf.keras.layers.Dense(dim, activation='relu')(input_layer)

        if dropout_rate:
            dropout_layer = tf.keras.layers.Dropout(dropout_rate)(hidden_layer)
            return dropout_layer

        return hidden_layer

    concat_layer = tf.keras.layers.Concatenate()([user_emb, item_emb])

    dropout_l1 = tf.keras.layers.Dropout(dropout_rate)(concat_layer)

    dense_layer_1 = add_layer(64, dropout_l1, dropout_rate)

    dense_layer_2 = add_layer(32, dense_layer_1, dropout_rate)

    dense_layer_3 = add_layer(16, dense_layer_2, None)

    dense_layer_4 = add_layer(8, dense_layer_3, None)

    return dense_layer_4


def _neuCF(gmf, mlp, dropout_rate):
    concat_layer = tf.keras.layers.Concatenate()([gmf, mlp])

    output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(concat_layer)

    return output_layer


def build_graph(user_dim, item_dim, dropout_rate=0.25):
    """ neural collaborative filtering model """

    user_input = tf.keras.Input(shape=(user_dim))
    item_input = tf.keras.Input(shape=(item_dim))

    # create embedding layers
    user_gmf_emb, user_mlp_emb = _get_user_embedding_layers(user_input, 32)
    item_gmf_emb, item_mlp_emb = _get_item_embedding_layers(item_input, 32)

    # general matrix factorization
    gmf = _gmf(user_gmf_emb, item_gmf_emb)

    # multi layer perceptron
    mlp = _mlp(user_mlp_emb, item_mlp_emb, dropout_rate)

    # output
    output = _neuCF(gmf, mlp, dropout_rate)

    # create the model
    model = tf.keras.Model(inputs=[user_input, item_input], outputs=output)

    return model


def model(x_train, y_train, n_user, n_item, num_epoch, batch_size):

    num_batch = np.ceil(x_train[0].shape[0]/batch_size)

    # build graph
    model = build_graph(n_user, n_item)

    # compile and train
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])

    model.fit_generator(
        generator=batch_generator(
            x=x_train, y=y_train,
            batch_size=batch_size, n_batch=num_batch,
            shuffle=True, user_dim=n_user, item_dim=n_item),
        epochs=num_epoch,
        steps_per_epoch=num_batch,
        verbose=2
    )

    return model


def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--sm-model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ.get('SM_HOSTS')))
    parser.add_argument('--current-host', type=str, default=os.environ.get('SM_CURRENT_HOST'))
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--n_user', type=int)
    parser.add_argument('--n_item', type=int)

    return parser.parse_known_args()


if __name__ == "__main__":
    args, unknown = _parse_args()

    # load data
    user_train, item_train, train_labels = _load_training_data(args.train)

    # build model
    ncf_model = model(
        x_train=[user_train, item_train],
        y_train=train_labels,
        n_user=args.n_user,
        n_item=args.n_item,
        num_epoch=args.epochs,
        batch_size=args.batch_size
    )

    if args.current_host == args.hosts[0]:
        # save model to an S3 directory with version number '00000001'
        ncf_model.save(os.path.join(args.sm_model_dir, '000000001'), 'neural_collaborative_filtering.h5')
```

```python id="rd5IfsHFx9uC"
# specify training instance type and model hyperparameters
# note that for the demo purpose, the number of epoch is set to 1

num_of_instance = 1                 # number of instance to use for training
instance_type = 'ml.c5.2xlarge'     # type of instance to use for training

training_script = 'ncf.py'

training_parameters = {
    'epochs': 1,
    'batch_size': 256, 
    'n_user': n_user, 
    'n_item': n_item
}

# training framework specs
tensorflow_version = '2.1.0'
python_version = 'py3'
distributed_training_spec = {'parameter_server': {'enabled': True}}
```

```python id="S4uO4rj5x9qL"
# initiate the training job using Tensorflow estimator
ncf_estimator = TensorFlow(
    entry_point=training_script,
    role=role,
    train_instance_count=num_of_instance,
    train_instance_type=instance_type,
    framework_version=tensorflow_version,
    py_version=python_version,
    distributions=distributed_training_spec,
    hyperparameters=training_parameters
)
```

```python id="qK30wSs8wIGs"
# kick off the training job
ncf_estimator.fit(training_data_uri)
```

```python id="1yMadQjvwID2"
# once the model is trained, we can deploy the model using Amazon SageMaker Hosting Services
# Here we deploy the model using one ml.c5.xlarge instance as a tensorflow-serving endpoint
# This enables us to invoke the endpoint like how we use Tensorflow serving
# Read more about Tensorflow serving using the link below
# https://www.tensorflow.org/tfx/tutorials/serving/rest_simple

endpoint_name = 'neural-collaborative-filtering-model-demo'

predictor = ncf_estimator.deploy(initial_instance_count=1, 
                                 instance_type='ml.c5.xlarge', 
                                 endpoint_type='tensorflow-serving',
                                 endpoint_name=endpoint_name)
```

```python id="Rn1BcXivwIA6"
# To use the endpoint in another notebook, we can initiate a predictor object as follows
from sagemaker.tensorflow import TensorFlowPredictor

predictor = TensorFlowPredictor(endpoint_name)
```

```python id="kiNdqQofwH9D"
# Define a function to read testing data
def _load_testing_data(base_dir):
    """ load testing data """
    df_test = np.load(os.path.join(base_dir, 'test.npy'))
    user_test, item_test, y_test = np.split(np.transpose(df_test).flatten(), 3)
    return user_test, item_test, y_test
```

```python id="wf7PoSRgyTEX"
# read testing data from local
user_test, item_test, test_labels = _load_testing_data('./ml-latest-small/s3/')

# one-hot encode the testing data for model input
with tf.Session() as tf_sess:
    test_user_data = tf_sess.run(tf.one_hot(user_test, depth=n_user)).tolist()
    test_item_data = tf_sess.run(tf.one_hot(item_test, depth=n_item)).tolist()
    
# if you're using Tensorflow 2.0 for one hot encoding
# you can convert the tensor to list using:
# tf.one_hot(uuser_test, depth=n_user).numpy().tolist()
```

```python id="O7M6DdZhyTAj"
# make batch prediction
batch_size = 100
y_pred = []
for idx in range(0, len(test_user_data), batch_size):
    # reformat test samples into tensorflow serving acceptable format
    input_vals = {
     "instances": [
         {'input_1': u, 'input_2': i} 
         for (u, i) in zip(test_user_data[idx:idx+batch_size], test_item_data[idx:idx+batch_size])
    ]}
 
    # invoke model endpoint to make inference
    pred = predictor.predict(input_vals)
    
    # store predictions
    y_pred.extend([i[0] for i in pred['predictions']])
```

```python id="GdBQdptUyS8w"
# let's see some prediction examples, assuming the threshold 
# --- prediction probability view ---
print('This is what the prediction output looks like')
print(y_pred[:5], end='\n\n\n')

# --- user item pair prediction view, with threshold of 0.5 applied ---
pred_df = pd.DataFrame([
    user_test,
    item_test,
    (np.array(y_pred) >= 0.5).astype(int)],
).T

pred_df.columns = ['userId', 'movieId', 'prediction']

print('We can convert the output to user-item pair as shown below')
print(pred_df.head(), end='\n\n\n')

# --- aggregated prediction view, by user ---
print('Lastly, we can roll up the prediction list by user and view it that way')
print(pred_df.query('prediction == 1').groupby('userId').movieId.apply(list).head().to_frame(), end='\n\n\n')
```

```python id="Kdw2dv4zyS5P"
# delete endpoint at the end of the demo
predictor.delete_endpoint(delete_endpoint_config=True)
```
