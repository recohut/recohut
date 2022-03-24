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

# NCF Tensorflow

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 5941, "status": "ok", "timestamp": 1617996579075, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="XF-QSGraGsHd" outputId="8b2c76ed-a252-4842-ade3-135565c71726"
%tensorflow_version 1.x
```

```python executionInfo={"elapsed": 5957, "status": "ok", "timestamp": 1617996579074, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="Dko49M8sCFpd"
import numpy as np
import math
import pandas as pd
from ast import literal_eval as make_tuple
from time import time
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Embedding, Flatten, Concatenate, Multiply, Dot
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy
import matplotlib.pyplot as plt
```

```python executionInfo={"elapsed": 2144, "status": "ok", "timestamp": 1617996579075, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="WVm6nP-0CFpf"
data_path = "https://raw.githubusercontent.com/sparsh-ai/rec-data-public/master/dump/"
train_file_path_movielens = "ml-1m.train.rating"
negative_file_path_movielens = "ml-1m.test.negative"

train_file_path_pinterest = "pinterest-20.train.rating"
negative_file_path_pinterest = "pinterest-20.test.negative"
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 1464, "status": "ok", "timestamp": 1617996580321, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="rps8X5UpFW_0" outputId="cfe2a773-6ed8-416d-b3e3-488427b7adac"
!wget -nc {data_path+train_file_path_movielens}
!wget -nc {data_path+negative_file_path_movielens}
!wget -nc {data_path+train_file_path_pinterest}
!wget -nc {data_path+negative_file_path_pinterest}
```

```python executionInfo={"elapsed": 1534, "status": "ok", "timestamp": 1617996585860, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="lJmiyvRBCFpg"
batch_size = 2**13 #(8192)
K = 10
epochs = 10
```

```python executionInfo={"elapsed": 1131, "status": "ok", "timestamp": 1617996585862, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="Bw9XBh1TCFpg"
def number_users_and_items(path, sep='\t', title_columns=False):
    file = open(file=path, mode='r')
    num_users, num_items = 0, 0
    if title_columns:
        file.readline()
    for line in file.readlines():
        split_line = line.split(sep=sep)
        userID, itemID = int(split_line[0]), int(split_line[1])
        if userID > num_users:
            num_users = userID
        if itemID > num_items:
            num_items = itemID
    file.close()
    return num_users+1, num_items+1 # we add one to take into account userID 0 and itemID 0
```

```python executionInfo={"elapsed": 889, "status": "ok", "timestamp": 1617996585864, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="Rb0dNDqzCFpi"
def get_train_set(path, num_negative_instances, num_items):
    data = []
    file = open(file=path, mode='r')
    current_user = 0
    interactions_per_user = []
    for line in file:
        split_line = line.split('\t')[:-1]
        user, item = int(split_line[0]), int(split_line[1])
        if user == current_user:
            interactions_per_user.append(item)
            data.append([user, item, 1])
        else:
            num_interactions = len(interactions_per_user)
            for i in range(num_negative_instances * num_interactions):
                random_item = np.random.randint(num_items)
                while (random_item in interactions_per_user):
                    random_item = np.random.randint(num_items)
                data.append([current_user, random_item, 0])
            data.append([user, item, 1])
            current_user = user
            interactions_per_user = [item]
    num_interactions = len(interactions_per_user)
    for i in range(num_negative_instances * num_interactions):
        random_item = np.random.randint(num_items)
        while (random_item in interactions_per_user):
            random_item = np.random.randint(num_items)
        data.append([current_user, random_item, 0])
    file.close()
    return np.array(data)

def get_test_set(negative_filepath):
    test_set = []
    negative_file = open(file=negative_filepath, mode="r")
    for line in negative_file:
        split_line = line.split(sep="\t")
        (user, positive_item) = make_tuple(split_line[0])
        items = [positive_item]+[int(split_line[i]) for i in range(1,len(split_line))]
        test_set.append(items)
    negative_file.close()
    return np.array(test_set)
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 35583, "status": "ok", "timestamp": 1617996687939, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="tru2Y1S8CFpj" outputId="16a730d1-f5e0-4400-a732-22f26547024f"
print("\n*** Preprocessing of data ***")

start = time()

num_users_movielens, num_items_movielens = number_users_and_items(path=train_file_path_movielens)
train_set_movielens = get_train_set(path=train_file_path_movielens, num_negative_instances=4, num_items=num_items_movielens)
test_set_movielens = get_test_set(negative_filepath=negative_file_path_movielens)
print("\nDataset MovieLens")
print("Number of users : {}".format(num_users_movielens))
print("Number of items : {}".format(num_items_movielens))
print("Number of training interactions : {}".format(len(train_set_movielens)))
train_features_movielens = [train_set_movielens[:,0], train_set_movielens[:,1]]
train_labels_movielens = train_set_movielens[:,2]

num_users_pinterest, num_items_pinterest = number_users_and_items(path=train_file_path_pinterest)
train_set_pinterest = get_train_set(path=train_file_path_pinterest, num_negative_instances=4, num_items=num_items_pinterest)
test_set_pinterest = get_test_set(negative_filepath=negative_file_path_pinterest)
print("\nDataset Pinterest")
print("Number of users : {}".format(num_users_pinterest))
print("Number of items : {}".format(num_items_pinterest))
print("Number of training interactions : {}".format(len(train_set_pinterest)))
train_features_pinterest = [train_set_pinterest[:,0], train_set_pinterest[:,1]]
train_labels_pinterest = train_set_pinterest[:,2]

run_time = time()-start
print("Running time : {:.0f} min {:.0f} sec".format(run_time//60, run_time%60))

print("\nDatasets loaded successfully !")
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 23, "status": "ok", "timestamp": 1617996687941, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="dJ1ryBnmCFpn" outputId="08372182-e159-42c1-a433-8f42f770c055"
def get_model_MF(num_users, num_items, factors=8):
    d = factors
    
    user_input_MF = Input(shape=(1,), dtype='int32', name='user_input_MF')
    item_input_MF = Input(shape=(1,), dtype='int32', name='item_input_MF')

    user_embedding_MF = Embedding(input_dim=num_users, output_dim=d, name='user_embedding_MF')(user_input_MF)
    item_embedding_MF = Embedding(input_dim=num_items, output_dim=d, name='item_embedding_MF')(item_input_MF)

    user_latent_MF = Flatten()(user_embedding_MF)
    item_latent_MF = Flatten()(item_embedding_MF)
    
    dot = Dot(axes=1)([user_latent_MF, item_latent_MF])

    MF = Model(inputs=[user_input_MF, item_input_MF], outputs=dot)
    MF.compile(optimizer=tf.train.AdamOptimizer(), loss=binary_crossentropy)
    return MF

MF = get_model_MF(num_users=num_users_movielens, num_items=num_items_movielens)
MF.summary()
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 1283, "status": "ok", "timestamp": 1617996727328, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="i4jtAmLRCFpq" outputId="c40f2429-d2d2-4b65-d708-d692dc3ed0aa"
## GMF model
def get_model_GMF(num_users, num_items, factors=8):
    d=factors
    user_input_GMF = Input(shape=(1,), dtype='int32', name='user_input_GMF')
    item_input_GMF = Input(shape=(1,), dtype='int32', name='item_input_GMF')

    user_embedding_GMF = Embedding(input_dim=num_users, output_dim=d, name='user_embedding_GMF')
    item_embedding_GMF = Embedding(input_dim=num_items, output_dim=d, name='item_embedding_GMF')

    user_latent_GMF = Flatten()(user_embedding_GMF(user_input_GMF))
    item_latent_GMF = Flatten()(item_embedding_GMF(item_input_GMF))

    mul = Multiply()([user_latent_GMF, item_latent_GMF]) # len = factors

    prediction_GMF = Dense(units=1, activation='sigmoid', name='prediction')(mul)

    GMF = Model(inputs=[user_input_GMF, item_input_GMF], outputs=prediction_GMF)
    GMF.compile(optimizer=tf.train.AdamOptimizer(), loss=binary_crossentropy)
    return GMF

GMF_movielens = get_model_GMF(num_users=num_users_movielens, num_items=num_items_movielens)
GMF_movielens.summary()
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 1340, "status": "ok", "timestamp": 1617996730163, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="lq2fU-WUCFpu" outputId="a7057a1d-34db-4cf2-c953-2b3f92bd2713"
def get_model_MLP(num_users, num_items, num_layers=3, factors=8):
    if num_layers==0:
        d = int(factors/2)
    else:
        d = int((2**(num_layers-2))*factors)
        
    user_input_MLP = Input(shape=(1,), dtype='int32', name='user_input_MLP')
    item_input_MLP = Input(shape=(1,), dtype='int32', name='item_input_MLP')
    
    user_embedding_MLP = Embedding(input_dim=num_users, output_dim=d, name='user_embedding_MLP')(user_input_MLP)
    item_embedding_MLP = Embedding(input_dim=num_items, output_dim=d, name='item_embedding_MLP')(item_input_MLP)
    
    user_latent_MLP = Flatten()(user_embedding_MLP)
    item_latent_MLP = Flatten()(item_embedding_MLP)
    
    concatenation = Concatenate()([user_latent_MLP, item_latent_MLP])
    output = concatenation
    layer_name = 0
    for i in range(num_layers-1,-1,-1):
        layer = Dense(units=(2**i)*factors, activation='relu', name='layer%d' %(layer_name+1))
        output = layer(output)
        layer_name += 1
    prediction_MLP = Dense(units=1, activation='sigmoid', name='prediction_MLP')(output)
    MLP = Model(inputs=[user_input_MLP, item_input_MLP], outputs=prediction_MLP)
    MLP.compile(optimizer=tf.train.AdamOptimizer(), loss=binary_crossentropy)
    return MLP

MLP_movielens = get_model_MLP(num_users=num_users_movielens, num_items=num_items_movielens)
MLP_movielens.summary()
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 1271, "status": "ok", "timestamp": 1617996732785, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="9Uu8ulVQCFpw" outputId="0d18cffa-03de-4477-e914-a23dd42ae06f"
def get_model_NeuMF(num_users, num_items, num_layers_MLP_part=3, factors=8):
    assert (factors%2)==0
    if num_layers_MLP_part==0:
        d_MLP = int(factors/4)
    else:
        d_MLP = (2**(num_layers_MLP_part-3))*factors

    user_input = Input(shape=(1,), dtype='int32', name='user_input_NeuMF')
    item_input = Input(shape=(1,), dtype='int32', name='item_input_NeuMF')

    ## MLP part
    user_embedding_MLP = Embedding(input_dim=num_users, output_dim=d_MLP, name='user_embedding_MLP')(user_input)
    item_embedding_MLP = Embedding(input_dim=num_items, output_dim=d_MLP, name='item_embedding_MLP')(item_input)

    user_latent_MLP = Flatten()(user_embedding_MLP)
    item_latent_MLP = Flatten()(item_embedding_MLP)

    concatenation_embeddings = Concatenate()([user_latent_MLP, item_latent_MLP])
    
    output_MLP = concatenation_embeddings  
    layer_name = 0
    for i in range(num_layers_MLP_part-2,-2,-1):
        layer = Dense(units=(2**i)*factors, activation='relu', name='layer%d' %(layer_name+1))
        output_MLP = layer(output_MLP)
        layer_name += 1
    
    d_GMF = int(factors/2)
    ## GMF part
    user_embedding_GMF = Embedding(input_dim=num_users, output_dim=d_GMF, name='user_embedding_GMF')
    item_embedding_GMF = Embedding(input_dim=num_items, output_dim=d_GMF, name='item_embedding_GMF')

    user_latent_GMF = Flatten()(user_embedding_GMF(user_input))
    item_latent_GMF = Flatten()(item_embedding_GMF(item_input))

    mul = Multiply()([user_latent_GMF, item_latent_GMF])

    concatenation_of_models = Concatenate(name='final_concatenation')([mul, output_MLP]) # len = factors
    prediction_NeuMF = Dense(units=1, activation='sigmoid', name='prediction')(concatenation_of_models)

    NeuMF = Model(inputs=[user_input, item_input], outputs=prediction_NeuMF)
    NeuMF.compile(optimizer=tf.train.AdamOptimizer(), loss=binary_crossentropy)
    return NeuMF

print()
NeuMF_movielens = get_model_NeuMF(num_users=num_users_movielens, num_items=num_items_movielens)
NeuMF_movielens.summary()
```

```python executionInfo={"elapsed": 1089, "status": "ok", "timestamp": 1617996735333, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="ENVhruEaCFpx"
def getHitRatio(ranklist, K, positive_item): # à optimiser avec un dictionnaire
    if positive_item in ranklist[:K]:
        return 1
    else:
        return 0

def getNDCG(ranklist, K, positive_item): # à optimiser avec un dictionnaire
    if positive_item in ranklist[:K]:
        ranking_of_positive_item = np.where(ranklist == positive_item)[0][0]
        return math.log(2)/math.log(2+ranking_of_positive_item)
    else:
        return 0
```

```python executionInfo={"elapsed": 1387, "status": "ok", "timestamp": 1617996738368, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="2H3DkqjjCFpy"
def rank(item_scores):
    list_item_scores = item_scores.tolist()
    ranklist = sorted(list_item_scores, key=lambda item_score: item_score[1], reverse=True)
    ranklist = np.array(ranklist)[:,0].astype('int64')
    return ranklist

def evaluate_models(models, test_set, K):
    hits, ndcgs = [], []
    users = np.array([user for user in range(len(test_set)) for i in range(100)])
    items = test_set.reshape(-1,)
    for model in models:
        hits_model, ndcgs_model = [], []
        predictions = model.predict(x=[users, items], batch_size=len(test_set), verbose=0)
        map_item_scores = np.concatenate((items.reshape((100*len(test_set), 1)), predictions), axis=1)
        for user in range(len(test_set)):
            ranklist_items = rank(item_scores=map_item_scores[100*user:100*(user+1)])
            positive_item = items[100*user]
            hr = getHitRatio(ranklist=ranklist_items, K=K, positive_item=positive_item)
            ndcg = getNDCG(ranklist=ranklist_items, K=K, positive_item=positive_item)
            hits_model.append(hr)
            ndcgs_model.append(ndcg)
        hits.append(np.array(hits_model).mean())
        ndcgs.append(np.array(ndcgs_model).mean())
    return hits, ndcgs

def train_models(models, train_features, train_labels, test_set, batch_size, epochs, K, verbose=2):
    first_hits, first_ndcgs = evaluate_models(models=models, test_set=test_set, K=K)
    first_losses = []
    for model in models:
        loss = model.evaluate(x=train_features, y=train_labels, batch_size=batch_size, verbose=0)
        first_losses.append(loss)
    losses, hits, ndcgs = [first_losses], [first_hits], [first_ndcgs]
    for e in range(epochs):
        print("\nEpoch n°{}/{}".format(e+1, epochs))
        losses_of_this_epoch = []
        for model in models:
            history = model.fit(x=train_features, y=train_labels, batch_size=batch_size, epochs=1, verbose=verbose, shuffle=True)
            losses_of_this_epoch.append(history.history["loss"][0])
        hits_of_this_epoch, ndcgs_of_this_epoch = evaluate_models(models=models, test_set=test_set, K=K)
        hits.append(hits_of_this_epoch)
        ndcgs.append(ndcgs_of_this_epoch)
        losses.append(losses_of_this_epoch)
    return np.array(hits), np.array(ndcgs), np.array(losses)
```

<!-- #region id="UfmeEHhrCFp0" -->
#### Dataset MovieLens
<!-- #endregion -->

```python id="4qHGSplvCFp0" outputId="cafab9ff-ec1d-4227-fc65-934f656874a1"
start = time()

MF_movielens = get_model_MF(num_users=num_users_movielens, num_items=num_items_movielens)
GMF_movielens = get_model_GMF(num_users=num_users_movielens, num_items=num_items_movielens)
MLP_movielens = get_model_MLP(num_users=num_users_movielens, num_items=num_items_movielens)
NeuMF_movielens = get_model_NeuMF(num_users=num_users_movielens, num_items=num_items_movielens)

models_movielens = [MF_movielens, GMF_movielens, MLP_movielens, NeuMF_movielens]

hits, ndcgs, losses = train_models(models=models_movielens, train_features=train_features_movielens, train_labels=train_labels_movielens, 
                                   test_set=test_set_movielens, batch_size=batch_size, epochs=epochs, K=K)

loss_MF, loss_GMF, loss_MLP, loss_NeuMF = losses[:,0], losses[:,1], losses[:,2], losses[:,3]
hits_MF, hits_GMF, hits_MLP, hits_NeuMF = hits[:,0], hits[:,1], hits[:,2], hits[:,3]
ndcgs_MF, ndcgs_GMF, ndcgs_MLP, ndcgs_NeuMF = ndcgs[:,0], ndcgs[:,1], ndcgs[:,2], ndcgs[:,3]

run_time = time()-start
print("\nRunning time : {:.0f} min {:.0f} sec".format(run_time//60, run_time%60))
```

```python id="uaT2FcK2CFp1" outputId="968fc86c-8ee2-4547-a5f0-fbe8a1e178bd"
iterations = [e for e in range(epochs+1)]
color_MF, color_GMF, color_MLP, color_NeuMF = "green", "mediumblue", "rebeccapurple", "red"
path_to_save_figures = "Figures/"

plt.figure("loss_movieLens")
plt.plot(iterations, loss_MF, color=color_MF)
plt.plot(iterations, loss_GMF, color=color_GMF, linestyle="dashed")
plt.plot(iterations, loss_MLP, color=color_MLP)
plt.plot(iterations, loss_NeuMF, color=color_NeuMF)
plt.xlabel("Iteration")
plt.ylabel("Training loss")
plt.legend(("MF", "GMF","MLP","NeuMF",))
plt.title("MovieLens")
plt.grid()
plt.savefig(fname=path_to_save_figures+"experiment1_loss_movielens.png")

plt.figure("hits_movieLens")
plt.plot(iterations, hits_MF, color=color_MF)
plt.plot(iterations, hits_GMF, color=color_GMF, linestyle="dashed")
plt.plot(iterations, hits_MLP, color=color_MLP)
plt.plot(iterations, ndcgs_NeuMF, color=color_NeuMF)
plt.xlabel("Iteration")
plt.ylabel("HR@"+str(K))
plt.legend(("MF", "GMF","MLP","NeuMF",))
plt.title("MovieLens")
plt.grid()
plt.savefig(fname=path_to_save_figures+"experiment1_hr_movielens.png")

plt.figure("ndcgs_movielens")
plt.plot(iterations, ndcgs_MF, color=color_MF)
plt.plot(iterations, ndcgs_GMF, color=color_GMF, linestyle="dashed")
plt.plot(iterations, ndcgs_MLP, color=color_MLP)
plt.plot(iterations, hits_NeuMF, color=color_NeuMF)
plt.xlabel("Iteration")
plt.ylabel("NDCG@"+str(K))
plt.legend(("MF", "GMF","MLP","NeuMF",))
plt.title("MovieLens")
plt.grid()
plt.savefig(fname=path_to_save_figures+"experiment1_ndcg_movielens.png")

plt.show()
```

<!-- #region id="2pl0j7HNCFp1" -->
#### Dataset Pinterest
<!-- #endregion -->

```python id="o-OC2SPXCFp2" outputId="27399db2-6d80-494b-d693-1248db0b5641"
start = time()

MF_pinterest = get_model_MF(num_users=num_users_pinterest, num_items=num_items_pinterest)
GMF_pinterest = get_model_GMF(num_users=num_users_pinterest, num_items=num_items_pinterest)
MLP_pinterest = get_model_MLP(num_users=num_users_pinterest, num_items=num_items_pinterest)
NeuMF_pinterest = get_model_NeuMF(num_users=num_users_pinterest, num_items=num_items_pinterest)

models_pinterest = [MF_pinterest, GMF_pinterest, MLP_pinterest, NeuMF_pinterest]

hits, ndcgs, losses = train_models(models=models_pinterest, train_features=train_features_pinterest, train_labels=train_labels_pinterest, 
                                   test_set=test_set_pinterest, batch_size=batch_size, epochs=epochs, K=K)

loss_MF, loss_GMF, loss_MLP, loss_NeuMF = losses[:,0], losses[:,1], losses[:,2], losses[:,3]
hits_MF, hits_GMF, hits_MLP, hits_NeuMF = hits[:,0], hits[:,1], hits[:,2], hits[:,3]
ndcgs_MF, ndcgs_GMF, ndcgs_MLP, ndcgs_NeuMF = ndcgs[:,0], ndcgs[:,1], ndcgs[:,2], ndcgs[:,3]

run_time = time()-start
print("\nRunning time : {:.0f} min {:.0f} sec".format(run_time//60, run_time%60))
```

```python id="ULixCjNhCFp2" outputId="9594edc0-ae49-4107-8e33-2c99c5deefa1"
iterations = [e for e in range(epochs+1)]
color_MF, color_GMF, color_MLP, color_NeuMF = "green", "mediumblue", "rebeccapurple", "red"
path_to_save_figures = "Figures/"

plt.figure("loss_pinterest")
plt.plot(iterations, loss_MF, color=color_MF)
plt.plot(iterations, loss_GMF, color=color_GMF, linestyle="dashed")
plt.plot(iterations, loss_MLP, color=color_MLP)
plt.plot(iterations, loss_NeuMF, color=color_NeuMF)
plt.xlabel("Iteration")
plt.ylabel("Training loss")
plt.legend(("MF", "GMF","MLP","NeuMF",))
plt.title("Pinterest")
plt.grid()
plt.savefig(fname=path_to_save_figures+"experiment1_loss_pinterest.png")

plt.figure("hits_pinterest")
plt.plot(iterations, hits_MF, color=color_MF)
plt.plot(iterations, hits_GMF, color=color_GMF, linestyle="dashed")
plt.plot(iterations, hits_MLP, color=color_MLP)
plt.plot(iterations, ndcgs_NeuMF, color=color_NeuMF)
plt.xlabel("Iteration")
plt.ylabel("HR@"+str(K))
plt.legend(("MF", "GMF","MLP","NeuMF",))
plt.title("Pinterest")
plt.grid()
plt.savefig(fname=path_to_save_figures+"experiment1_hr_pinterest.png")

plt.figure("ndcgs_pinterest")
plt.plot(iterations, ndcgs_MF, color=color_MF)
plt.plot(iterations, ndcgs_GMF, color=color_GMF, linestyle="dashed")
plt.plot(iterations, ndcgs_MLP, color=color_MLP)
plt.plot(iterations, hits_NeuMF, color=color_NeuMF)
plt.xlabel("Iteration")
plt.ylabel("NDCG@"+str(K))
plt.legend(("MF", "GMF","MLP","NeuMF",))
plt.title("Pinterest")
plt.grid()
plt.savefig(fname=path_to_save_figures+"experiment1_ndcg_pinterest.png")

plt.show()
```

<!-- #region id="vd3UR0rECFp4" -->
#### Dataset MovieLens
<!-- #endregion -->

```python id="aaj9frCkCFp4" outputId="f40c11cb-d052-41d0-baf5-d3c9b06e4609"
start = time()

names_models = {0: "MF", 1: "GMF", 2: "MLP", 3: "NeuMF"}

max_num_negative = 10
hits_total, ndcgs_total = [], []
for i in range(1, max_num_negative+1):
    print("\nNumber of negative interactions per positive interaction in train set : {}".format(i))
    train_set = get_train_set(path=train_file_path_movielens, num_negative_instances=i, num_items=num_items_movielens)
    train_features = [train_set[:,0], train_set[:,1]]
    train_labels = train_set[:,2]
    # pour chaque nombre d'instances négatives il faut créer de nouveau modèles et les entrainer à partir de 0
    MF_movielens = get_model_MF(num_users=num_users_movielens, num_items=num_items_movielens, factors=16)
    GMF_movielens = get_model_GMF(num_users=num_users_movielens, num_items=num_items_movielens, factors=16)
    MLP_movielens = get_model_MLP(num_users=num_users_movielens, num_items=num_items_movielens, factors=16)
    NeuMF_movielens = get_model_NeuMF(num_users=num_users_movielens, num_items=num_items_movielens, factors=16)
    models_movielens = [MF_movielens, GMF_movielens, MLP_movielens, NeuMF_movielens]
    k=0
    for model in models_movielens:
        print("-- Training of model {}...".format(names_models[k]))
        history = model.fit(x=train_features, y=train_labels, batch_size=batch_size, epochs=3, verbose=0, shuffle=True)
        k+=1
    print("Evaluation...")
    hits, ndcgs = evaluate_models(models=models_movielens, test_set=test_set_movielens, K=K)
    hits_total.append(hits)
    ndcgs_total.append(ndcgs)

hits_total, ndcgs_total = np.array(hits_total), np.array(ndcgs_total)

hits_MF, hits_GMF, hits_MLP, hits_NeuMF = hits_total[:,0], hits_total[:,1], hits_total[:,2], hits_total[:,3]
ndcgs_MF, ndcgs_GMF, ndcgs_MLP, ndcgs_NeuMF = ndcgs_total[:,0], ndcgs_total[:,1], ndcgs_total[:,2], ndcgs_total[:,3]

run_time = time()-start
print("\nRunning time : {:.0f} min {:.0f} sec".format(run_time//60, run_time%60))
```

```python id="LabyKjwJCFp5" outputId="d66784c5-5ccb-4760-9b81-bdab19be9b65"
negatives = [i for i in range(1,max_num_negative+1)]
color_MF, color_GMF, color_MLP, color_NeuMF = "green", "mediumblue", "rebeccapurple", "red"
path_to_save_figures = "Figures/"

plt.figure("Negative")
plt.plot(negatives, hits_MF, color=color_MF, marker="^")
plt.plot(negatives, hits_GMF, color=color_GMF, linestyle="dashed", marker="o")
plt.plot(negatives, hits_MLP, color=color_MLP, marker="x")
plt.plot(negatives, hits_NeuMF, color=color_NeuMF, marker="d")
plt.xlabel("Number of negatives")
plt.ylabel("HR@"+str(K))
plt.legend(("MF","GMF","MLP","NeuMF",))
plt.title("MovieLens")
plt.grid()
#plt.savefig(fname=path_to_save_figures+"experiment2_hr_movielens.png")

plt.figure("Negative2")
plt.plot(negatives, hits_MF, color=color_MF, marker="^")
plt.plot(negatives, ndcgs_GMF, color=color_GMF, linestyle="dashed", marker="o")
plt.plot(negatives, ndcgs_MLP, color=color_MLP, marker="x")
plt.plot(negatives, ndcgs_NeuMF, color=color_NeuMF, marker="d")
plt.xlabel("Number of negatives")
plt.ylabel("NDCG@"+str(K))
plt.legend(("MF","GMF","MLP","NeuMF",))
plt.title("MovieLens")
plt.grid()
#plt.savefig(fname=path_to_save_figures+"experiment2_ndcg_movielens.png")

plt.show()
```

<!-- #region id="dfEL4SNsCFp5" -->
#### Dataset Pinterest
<!-- #endregion -->

```python id="EyIFAhXuCFp6" outputId="c03be06e-f039-4a0a-eacc-5d93fdd55762"
start = time()

names_models = {0: "MF", 1: "GMF", 2: "MLP", 3: "NeuMF"}

max_num_negative = 10
hits_total, ndcgs_total = [], []
for i in range(1, max_num_negative+1):
    print("\nNumber of negative interactions per positive interaction in train set : {}".format(i))
    train_set = get_train_set(path=train_file_path_pinterest, num_negative_instances=i, num_items=num_items_pinterest)
    train_features = [train_set[:,0], train_set[:,1]]
    train_labels = train_set[:,2]
    # pour chaque nombre d'instances négatives il faut entrainer de nouveaux modèles
    MF_pinterest = get_model_MF(num_users=num_users_pinterest, num_items=num_items_pinterest, factors=16)
    GMF_pinterest = get_model_GMF(num_users=num_users_pinterest, num_items=num_items_pinterest, factors=16)
    MLP_pinterest = get_model_MLP(num_users=num_users_pinterest, num_items=num_items_pinterest, factors=16)
    NeuMF_pinterest = get_model_NeuMF(num_users=num_users_pinterest, num_items=num_items_pinterest, factors=16)
    models_pinterest = [MF_pinterest, GMF_pinterest, MLP_pinterest, NeuMF_pinterest]
    k=0
    for model in models_pinterest:
        print("-- Training of model {}...".format(names_models[k]))
        history = model.fit(x=train_features, y=train_labels, batch_size=batch_size, epochs=3, verbose=0, shuffle=True)
        k+=1
    print("Evaluation...")
    hits, ndcgs = evaluate_models(models=models_pinterest, test_set=test_set_pinterest, K=K)
    hits_total.append(hits)
    ndcgs_total.append(ndcgs)

hits_total, ndcgs_total= np.array(hits_total), np.array(ndcgs_total)

hits_MF, hits_GMF, hits_MLP, hits_NeuMF = hits_total[:,0], hits_total[:,1], hits_total[:,2], hits_total[:,3]
ndcgs_MF, ndcgs_GMF, ndcgs_MLP, ndcgs_NeuMF = ndcgs_total[:,0], ndcgs_total[:,1], ndcgs_total[:,2], ndcgs_total[:,3]

run_time = time()-start
print("\nRunning time : {:.0f} min {:.0f} sec".format(run_time//60, run_time%60))
```

```python id="W3UAyMVoCFp7" outputId="a2e35cbe-2da7-41d5-ae0c-e88178e20a3a"
negatives = [i for i in range(1,max_num_negative+1)]
color_MF, color_GMF, color_MLP, color_NeuMF = "green", "mediumblue", "rebeccapurple", "red"
path_to_save_figures = "Figures/"

plt.figure("Negative")
plt.plot(negatives, hits_MF, color=color_MF, marker="^")
plt.plot(negatives, hits_GMF, color=color_GMF, linestyle="dashed", marker="o")
plt.plot(negatives, hits_MLP, color=color_MLP, marker="x")
plt.plot(negatives, hits_NeuMF, color=color_NeuMF, marker="d")
plt.xlabel("Number of negatives")
plt.ylabel("HR@"+str(K))
plt.legend(("MF","GMF","MLP","NeuMF",))
plt.title("Pinterest")
plt.grid()
plt.savefig(fname=path_to_save_figures+"experiment2_hr_pinterest.png")

plt.figure("Negative2")
plt.plot(negatives, hits_MF, color=color_MF, marker="^")
plt.plot(negatives, ndcgs_GMF, color=color_GMF, linestyle="dashed", marker="o")
plt.plot(negatives, ndcgs_MLP, color=color_MLP, marker="x")
plt.plot(negatives, ndcgs_NeuMF, color=color_NeuMF, marker="d")
plt.xlabel("Number of negatives")
plt.ylabel("NDCG@"+str(K))
plt.legend(("MF","GMF","MLP","NeuMF",))
plt.title("Pinterest")
plt.grid()
plt.savefig(fname=path_to_save_figures+"experiment2_ndcg_pinterest.png")

plt.show()
```

<!-- #region id="HfxZq-tICFp9" -->
#### Dataset MovieLens
<!-- #endregion -->

```python id="EiC-bGBwCFp-" outputId="0079dedb-a79f-45f6-b5a5-cd4ee8a06660"
start = time()

factors = [8,16,32,64]
hits, ndcgs = [], []

for f in factors:
    print("Training for {:.0f} predictive factors...".format(f))
    #layers_MLP, layers_NeuMF = [4*f, 2*f, f], [2*f, f, int(f/2)]
    MF = get_model_MF(num_users=num_users_movielens, num_items=num_items_movielens, factors=f)
    GMF = get_model_MF(num_users=num_users_movielens, num_items=num_items_movielens, factors=f)
    MLP = get_model_MLP(num_users=num_users_movielens, num_items=num_items_movielens, factors=f)
    NeuMF = get_model_NeuMF(num_users=num_users_movielens, num_items=num_items_movielens, factors=f)
    models = [MF, GMF, MLP, NeuMF]
    for model in models:
        model.fit(x=train_features_movielens, y=train_labels_movielens, batch_size=batch_size, epochs=3, verbose=0, shuffle=True)
    print("Evaluation...\n")
    hit, ndcg = evaluate_models(models=models, test_set=test_set_movielens, K=K)
    hits.append(hit)
    ndcgs.append(ndcg)

hits, ndcgs = np.array(hits), np.array(ndcgs)

hits_MF, hits_GMF, hits_MLP, hits_NeuMF = hits[:,0], hits[:,1], hits[:,2], hits[:,3]
ndcgs_MF, ndcgs_GMF, ndcgs_MLP, ndcgs_NeuMF = ndcgs[:,0], ndcgs[:,1], ndcgs[:,2], ndcgs[:,3]

run_time = time()-start
print("\nRunning time : {:.0f} min {:.0f} sec".format(run_time//60, run_time%60))
```

```python id="XlnlnN8jCFp_" outputId="1273022a-2792-4e28-a44e-ad04519c3390"
color_MF, color_GMF, color_MLP, color_NeuMF = "green", "mediumblue", "rebeccapurple", "red"
path_to_save_figures = "Figures/"

plt.figure("Factors")
plt.plot(factors, hits_MF, color=color_MF, marker="^")
plt.plot(factors, hits_GMF, color=color_GMF, marker="o", linestyle="dashed")
plt.plot(factors, hits_MLP, color=color_MLP, marker="x")
plt.plot(factors, hits_NeuMF, color=color_NeuMF, marker="d")
plt.xlabel("Number of predictive factors")
plt.ylabel("HR@"+str(K))
plt.legend(("MF","GMF","MLP","NeuMF",))
plt.title("MovieLens")
plt.xticks(factors)
plt.grid()
#plt.savefig(fname=path_to_save_figures+"experiment3_hr_movielens.png")

plt.figure("Factors2")
plt.plot(factors, ndcgs_MF, color=color_MF, marker="^")
plt.plot(factors, ndcgs_GMF, color=color_GMF, marker="o", linestyle="dashed")
plt.plot(factors, hits_MLP, color=color_MLP, marker="x")
plt.plot(factors, hits_NeuMF, color=color_NeuMF, marker="d")
plt.xlabel("Number of predictive factors")
plt.ylabel("NDCG@"+str(K))
plt.legend(("MF","GMF","MLP","NeuMF",))
plt.title("MovieLens")
plt.xticks(factors)
plt.grid()
#plt.savefig(fname=path_to_save_figures+"experiment3_ndcg_movielens.png")

plt.show()
```

<!-- #region id="1pPz-qxZCFqA" -->
#### Dataset Pinterest
<!-- #endregion -->

<!-- #region id="woV2AcSkCFqA" -->
On fait de même sur le dataset Pinterest.
<!-- #endregion -->

```python id="hzmRzES1CFqB" outputId="a1ae8cf0-01c6-4100-a975-4a668e02a961"
start = time()

factors = [8,16,32,64]
hits, ndcgs = [], []

for f in factors:
    print("Training for {:.0f} predictive factors...".format(f))
    MF = get_model_MF(num_users=num_users_pinterest, num_items=num_items_pinterest, factors=f)
    GMF = get_model_MF(num_users=num_users_pinterest, num_items=num_items_pinterest, factors=f)
    MLP = get_model_MLP(num_users=num_users_pinterest, num_items=num_items_pinterest, factors=f)
    NeuMF = get_model_NeuMF(num_users=num_users_pinterest, num_items=num_items_pinterest, factors=f)
    models = [MF, GMF, MLP, NeuMF]
    for model in models:
        model.fit(x=train_features_pinterest, y=train_labels_pinterest, batch_size=batch_size, epochs=3, verbose=0, shuffle=True)
    print("Evaluation...\n")
    hit, ndcg = evaluate_models(models=models, test_set=test_set_pinterest, K=K)
    hits.append(hit)
    ndcgs.append(ndcg)

hits, ndcgs = np.array(hits), np.array(ndcgs)

hits_MF, hits_GMF, hits_MLP, hits_NeuMF = hits[:,0], hits[:,1], hits[:,2], hits[:,3]
ndcgs_MF, ndcgs_GMF, ndcgs_MLP, ndcgs_NeuMF = ndcgs[:,0], ndcgs[:,1], ndcgs[:,2], ndcgs[:,3]

run_time = time()-start
print("\nRunning time : {:.0f} min {:.0f} sec".format(run_time//60, run_time%60))
```

```python id="Wd4d4q4yCFqB" outputId="eea627cf-426f-4795-9c9a-d47d8bc79a8b"
color_MF, color_GMF, color_MLP, color_NeuMF = "green", "mediumblue", "rebeccapurple", "red"
path_to_save_figures = "Figures/"

plt.figure("Factors")
plt.plot(factors, hits_MF, color=color_MF, marker="^")
plt.plot(factors, hits_GMF, color=color_GMF, marker="o", linestyle="dashed")
plt.plot(factors, hits_MLP, color=color_MLP, marker="x")
plt.plot(factors, hits_NeuMF, color=color_NeuMF, marker="d")
plt.xlabel("Number of predictive factors")
plt.ylabel("HR@"+str(K))
plt.legend(("MF","GMF","MLP","NeuMF",))
plt.title("Pinterest")
plt.xticks(factors)
plt.grid()
plt.savefig(fname=path_to_save_figures+"experiment3_hr_pinterest.png")

plt.figure("Factors2")
plt.plot(factors, ndcgs_MF, color=color_MF, marker="^")
plt.plot(factors, ndcgs_GMF, color=color_GMF, marker="o", linestyle="dashed")
plt.plot(factors, hits_MLP, color=color_MLP, marker="x")
plt.plot(factors, hits_NeuMF, color=color_NeuMF, marker="d")
plt.xlabel("Number of predictive factors")
plt.ylabel("NDCG@"+str(K))
plt.legend(("MF","GMF","MLP","NeuMF",))
plt.title("Pinterest")
plt.xticks(factors)
plt.grid()
plt.savefig(fname=path_to_save_figures+"experiment3_ndcg_pinterest.png")

plt.show()
```

<!-- #region id="V5x3C3nnCFqD" -->
#### Dataset MovieLens
<!-- #endregion -->

```python id="LLKzGOYbCFqD" outputId="e38a0a49-38f5-47ed-b230-9a28abb2142f"
MF = get_model_MF(num_users=num_users_movielens, num_items=num_items_movielens)
GMF = get_model_GMF(num_users=num_users_movielens, num_items=num_items_movielens)
MLP = get_model_MLP(num_users=num_users_movielens, num_items=num_items_movielens)
NeuMF = get_model_NeuMF(num_users=num_users_movielens, num_items=num_items_movielens)

print("Training of every model over 3 epochs...\n")

MF.fit(x=train_features_movielens, y=train_labels_movielens, batch_size=batch_size, epochs=3, verbose=0, shuffle=True)
GMF.fit(x=train_features_movielens, y=train_labels_movielens, batch_size=batch_size, epochs=3, verbose=0, shuffle=True)
MLP.fit(x=train_features_movielens, y=train_labels_movielens, batch_size=batch_size, epochs=3, verbose=0, shuffle=True)
NeuMF.fit(x=train_features_movielens, y=train_labels_movielens, batch_size=batch_size, epochs=3, verbose=0, shuffle=True)

maxK = 10
hits, ndcgs = [], []

start = time()

for K in range(1, maxK+1):
    print("Evaluation of Top-K item recommendation for K={:.0f}".format(K))
    hit, ndcg = evaluate_models(models=[MF, GMF, MLP, NeuMF], test_set=test_set_movielens, K=K)
    hits.append(hit)
    ndcgs.append(ndcg)

hits, ndcgs = np.array(hits), np.array(ndcgs)

hits_MF, hits_GMF, hits_MLP, hits_NeuMF = hits[:,0], hits[:,1], hits[:,2], hits[:,3]
ndcgs_MF, ndcgs_GMF, ndcgs_MLP, ndcgs_NeuMF = ndcgs[:,0], ndcgs[:,1], ndcgs[:,2], ndcgs[:,3]

run_time = time()-start
print("\nRunning time : {:.0f} min {:.0f} sec".format(run_time//60, run_time%60))
```

```python id="fSkXN59nCFqD" outputId="436374e4-c382-4523-db09-9d4c2851179f"
color_MF, color_GMF, color_MLP, color_NeuMF = "green", "mediumblue", "rebeccapurple", "red"
K_values = [i for i in range(1,maxK+1)]
path_to_save_figures = "Figures/"

plt.figure("K")
plt.plot(K_values, hits_MF, color=color_MF, marker="^")
plt.plot(K_values, hits_GMF, color=color_GMF, marker="o")
plt.plot(K_values, hits_MLP, color=color_MLP, marker="x")
plt.plot(K_values, hits_NeuMF, color=color_NeuMF, marker="d")
plt.xlabel("K")
plt.ylabel("HR@K")
plt.legend(("MF","GMF","MLP","NeuMF",))
plt.title("Movielens")
plt.xticks(K_values)
plt.grid()
plt.savefig(fname=path_to_save_figures+"experiment4_hr_movielens.png")

plt.figure("K2")
plt.plot(K_values, ndcgs_MF, color=color_MF, marker="^")
plt.plot(K_values, ndcgs_GMF, color=color_GMF, marker="o")
plt.plot(K_values, ndcgs_MLP, color=color_MLP, marker="x")
plt.plot(K_values, ndcgs_NeuMF, color=color_NeuMF, marker="d")
plt.xlabel("K")
plt.ylabel("NDCG@K")
plt.legend(("MF","GMF","MLP","NeuMF",))
plt.title("Movielens")
plt.xticks(K_values)
plt.grid()
plt.savefig(fname=path_to_save_figures+"experiment4_ndcg_movielens.png")

plt.show()
```

<!-- #region id="6ulYCyNZCFqE" -->
#### Dataset Pinterest
<!-- #endregion -->

```python id="VfadOnH2CFqE" outputId="9d4af5e1-32f0-42a0-cf9a-4cec26cac3f2"
MF = get_model_MF(num_users=num_users_pinterest, num_items=num_items_pinterest)
GMF = get_model_GMF(num_users=num_users_pinterest, num_items=num_items_pinterest)
MLP = get_model_MLP(num_users=num_users_pinterest, num_items=num_items_pinterest)
NeuMF = get_model_NeuMF(num_users=num_users_pinterest, num_items=num_items_pinterest)

print("Training of every model over 3 epochs...\n")

MF.fit(x=train_features_pinterest, y=train_labels_pinterest, batch_size=batch_size, epochs=3, verbose=0, shuffle=True)
GMF.fit(x=train_features_pinterest, y=train_labels_pinterest, batch_size=batch_size, epochs=3, verbose=0, shuffle=True)
MLP.fit(x=train_features_pinterest, y=train_labels_pinterest, batch_size=batch_size, epochs=3, verbose=0, shuffle=True)
NeuMF.fit(x=train_features_pinterest, y=train_labels_pinterest, batch_size=batch_size, epochs=3, verbose=0, shuffle=True)

maxK = 10
hits, ndcgs = [], []

start = time()

for K in range(1, maxK+1):
    print("Evaluation of Top-K item recommendation for K={:.0f}".format(K))
    hit, ndcg = evaluate_models(models=[MF, GMF, MLP, NeuMF], test_set=test_set_pinterest, K=K)
    hits.append(hit)
    ndcgs.append(ndcg)

hits, ndcgs = np.array(hits), np.array(ndcgs)

hits_MF, hits_GMF, hits_MLP, hits_NeuMF = hits[:,0], hits[:,1], hits[:,2], hits[:,3]
ndcgs_MF, ndcgs_GMF, ndcgs_MLP, ndcgs_NeuMF = ndcgs[:,0], ndcgs[:,1], ndcgs[:,2], ndcgs[:,3]

run_time = time()-start
print("\nRunning time : {:.0f} min {:.0f} sec".format(run_time//60, run_time%60))
```

```python id="DaAXI7qZCFqF" outputId="fb19602e-34d1-4433-b59c-660d41014677"
color_MF, color_GMF, color_MLP, color_NeuMF = "green", "mediumblue", "rebeccapurple", "red"
K_values = [i for i in range(1,maxK+1)]
path_to_save_figures = "Figures/"

plt.figure("K")
plt.plot(K_values, hits_MF, color=color_MF, marker="^")
plt.plot(K_values, hits_GMF, color=color_GMF, marker="o")
plt.plot(K_values, hits_MLP, color=color_MLP, marker="x")
plt.plot(K_values, hits_NeuMF, color=color_NeuMF, marker="d")
plt.xlabel("K")
plt.ylabel("HR@K")
plt.legend(("MF","GMF","MLP","NeuMF",))
plt.title("Pinterest")
plt.xticks(K_values)
plt.grid()
plt.savefig(fname=path_to_save_figures+"experiment4_hr_pinterest.png")

plt.figure("K2")
plt.plot(K_values, ndcgs_MF, color=color_MF, marker="^")
plt.plot(K_values, ndcgs_GMF, color=color_GMF, marker="o")
plt.plot(K_values, ndcgs_MLP, color=color_MLP, marker="x")
plt.plot(K_values, ndcgs_NeuMF, color=color_NeuMF, marker="d")
plt.xlabel("K")
plt.ylabel("NDCG@K")
plt.legend(("MF","GMF","MLP","NeuMF",))
plt.title("Pinterest")
plt.xticks(K_values)
plt.grid()
plt.savefig(fname=path_to_save_figures+"experiment4_ndcg_pinterest.png")

plt.show()
```

<!-- #region id="eRQ5fghzCFqG" -->
Ici nous allons faire varier le nombre de couches et le nombre de neurones par couches dans le MLP.
<!-- #endregion -->

<!-- #region id="46RROeJfCFqG" -->
#### Dataset MovieLens
<!-- #endregion -->

```python id="Xy_Md-00CFqG" outputId="0abf1a5a-6b90-4fcb-8f28-bea0100096e8"
start = time()

factors = [8,16,32,64]
hits, ndcgs = [], []
max_num_hidden_layers = 4

for num_hidden_layers in range(max_num_hidden_layers+1):
    print()
    print("Number of hidden layers:", num_hidden_layers)
    models_MLP = []
    for f in factors:
        print("Number of predictive factors : {}".format(f))
        MLP = get_model_MLP(num_users=num_users_movielens, num_items=num_items_movielens, num_layers=num_hidden_layers, factors=f)
        MLP.fit(x=train_features_movielens, y=train_labels_movielens, epochs=3, batch_size=batch_size, shuffle=True, verbose=0)
        models_MLP.append(MLP)
    hits_hidden_layers, ndcgs_hidden_layers = evaluate_models(models=models_MLP, test_set=test_set_movielens, K=K)
    hits.append(hits_hidden_layers)
    ndcgs.append(ndcgs_hidden_layers)

hits, ndcgs = np.transpose(np.array(hits)), np.transpose(np.array(ndcgs))

print("\nHits:")
print(hits)
print("\nNDCGs")
print(ndcgs)

run_time = time()-start
print("\nRunning time : {:.0f} min {:.0f} sec".format(run_time//60, run_time%60))
```

<!-- #region id="aLU-gQBzCFqH" -->
#### Dataset Pinterest
<!-- #endregion -->

```python id="XqVlk66CCFqH" outputId="574bca18-f39e-4d4a-f5f1-1c79c169f346"
start = time()

factors = [8,16,32,64]
hits, ndcgs = [], []
max_num_hidden_layers = 4

for num_hidden_layers in range(max_num_hidden_layers+1):
    print()
    print("Number of hidden layers:", num_hidden_layers)
    models_MLP = []
    for f in factors:
        print("Number of predictive factors : {}".format(f))
        MLP = get_model_MLP(num_users=num_users_pinterest, num_items=num_items_pinterest, num_layers=num_hidden_layers, factors=f)
        MLP.fit(x=train_features_pinterest, y=train_labels_pinterest, epochs=3, batch_size=batch_size, shuffle=True, verbose=0)
        models_MLP.append(MLP)
    hits_hidden_layers, ndcgs_hidden_layers = evaluate_models(models=models_MLP, test_set=test_set_pinterest, K=K)
    hits.append(hits_hidden_layers)
    ndcgs.append(ndcgs_hidden_layers)

hits, ndcgs = np.transpose(np.array(hits)), np.transpose(np.array(ndcgs))

print("\nHits:")
print(hits)
print("\nNDCGs")
print(ndcgs)

run_time = time()-start
print("\nRunning time : {:.0f} min {:.0f} sec".format(run_time//60, run_time%60))
```

```python id="SV_iZOAeCFqJ" outputId="1739d195-3c84-4dd4-b6f4-da12ee581845"
travel_filepath = "Data/Travel_28K.csv"
df = pd.read_csv(filepath_or_buffer=travel_filepath)
df.head()
```

```python id="Ar0Y7kBBCFqJ"
def get_dataset_airline(travel_filepath, num_negative_instances, num_items):
    df = pd.read_csv(filepath_or_buffer=travel_filepath)
    ones = np.ones(shape=(len(df.values),1))
    positive_interactions = np.concatenate((df.values, ones), axis=1).astype(dtype="int64")
    train_set, test_set = [], []
    current_user = -1
    interactions_for_this_user = []
    for interaction in positive_interactions:
        user, item = interaction[0], interaction[1]
        if user==current_user:
            train_set.append(interaction)
            interactions_for_this_user.append(item)
        else:
            num_interactions = len(interactions_for_this_user)
            for i in range(num_negative_instances * num_interactions):
                random_item = np.random.randint(num_items)
                while (random_item in interactions_for_this_user):
                    random_item = np.random.randint(num_items)
                train_set.append([current_user, random_item, 0])
                interactions_for_this_user.append([current_user, random_item, 0])
            current_user = user
            positive_item = item
            items_test_for_this_user = [positive_item]
            interactions_for_this_user = []
            for i in range(99):
                random_item = np.random.randint(num_items)
                while random_item in items_test_for_this_user:
                    random_item = np.random.randint(num_items)
                items_test_for_this_user.append(random_item)
            test_set.append(items_test_for_this_user)
    for i in range(num_negative_instances * num_interactions):
        random_item = np.random.randint(num_items+1)
        while (random_item in interactions_for_this_user):
            random_item = np.random.randint(num_items+1)
        train_set.append([current_user, random_item, 0])
    return np.array(train_set), np.array(test_set)
```

```python id="gN9-ijpICFqK" outputId="e24213a0-0d4d-4dde-b00b-521d50a3d793"
print("\n*** Preprocessing of data ***")

num_users_airline, num_items_airline = number_users_and_items(path=travel_filepath, sep=",", title_columns=True)
train_set_airline, test_set_airline = get_dataset_airline(travel_filepath=travel_filepath, num_negative_instances=2, num_items=num_items_airline)

print("\nAirline Travel Dataset")
print("Number of users : {}".format(num_users_airline))
print("Number of items : {}".format(num_items_airline))
print("Number of training interactions : {}".format(len(train_set_airline)))
train_features_airline = [train_set_airline[:,0], train_set_airline[:,1]]
train_labels_airline = train_set_airline[:,2]

print("\nDataset loaded successfully !")
```

```python id="CzEpLZ_5CFqK" outputId="6e2b901b-1323-4916-9798-ee1d8319fae0"
start = time()

MF_airline = get_model_MF(num_users=num_users_airline, num_items=num_items_airline)
GMF_airline = get_model_GMF(num_users=num_users_airline, num_items=num_items_airline)
MLP_airline = get_model_MLP(num_users=num_users_airline, num_items=num_items_airline)
NeuMF_airline = get_model_NeuMF(num_users=num_users_airline, num_items=num_items_airline)

models_airline = [MF_airline, GMF_airline, MLP_airline, NeuMF_airline]

hits, ndcgs, losses = train_models(models=models_airline, train_features=train_features_airline, train_labels=train_labels_airline, 
                                   test_set=test_set_airline, batch_size=256, epochs=epochs, K=K)

loss_MF, loss_GMF, loss_MLP, loss_NeuMF = losses[:,0], losses[:,1], losses[:,2], losses[:,3]
hits_MF, hits_GMF, hits_MLP, hits_NeuMF = hits[:,0], hits[:,1], hits[:,2], hits[:,3]
ndcgs_MF, ndcgs_GMF, ndcgs_MLP, ndcgs_NeuMF = ndcgs[:,0], ndcgs[:,1], ndcgs[:,2], ndcgs[:,3]

run_time = time()-start
print("\nRunning time : {:.0f} min {:.0f} sec".format(run_time//60, run_time%60))
```

```python id="37nUlpNpCFqL" outputId="ae406a90-fe24-4416-9ed8-7a2ba3fdbecd"
iterations = [e for e in range(epochs+1)]
color_MF, color_GMF, color_MLP, color_NeuMF = "green", "mediumblue", "rebeccapurple", "red"
path_to_save_figures = "Figures/"

plt.figure("loss_airline")
plt.plot(iterations, loss_MF, color=color_MF)
plt.plot(iterations, loss_GMF, color=color_GMF, linestyle="dashed")
plt.plot(iterations, loss_MLP, color=color_MLP)
plt.plot(iterations, loss_NeuMF, color=color_NeuMF)
plt.xlabel("Iteration")
plt.ylabel("Training loss")
plt.legend(("MF", "GMF","MLP","NeuMF",))
plt.title("Airline Travel")
plt.grid()
plt.savefig(fname=path_to_save_figures+"experiment6_loss_airline.png")

plt.figure("hits_airline")
plt.plot(iterations, hits_MF, color=color_MF)
plt.plot(iterations, hits_GMF, color=color_GMF, linestyle="dashed")
plt.plot(iterations, hits_MLP, color=color_MLP)
plt.plot(iterations, ndcgs_NeuMF, color=color_NeuMF)
plt.xlabel("Iteration")
plt.ylabel("HR@"+str(K))
plt.legend(("MF", "GMF","MLP","NeuMF",))
plt.title("Airline Travel")
plt.grid()
plt.savefig(fname=path_to_save_figures+"experiment6_hr_airline.png")

plt.figure("ndcgs_airline")
plt.plot(iterations, ndcgs_MF, color=color_MF)
plt.plot(iterations, ndcgs_GMF, color=color_GMF, linestyle="dashed")
plt.plot(iterations, ndcgs_MLP, color=color_MLP)
plt.plot(iterations, hits_NeuMF, color=color_NeuMF)
plt.xlabel("Iteration")
plt.ylabel("NDCG@"+str(K))
plt.legend(("MF", "GMF","MLP","NeuMF",))
plt.title("Airline Travel")
plt.grid()
plt.savefig(fname=path_to_save_figures+"experiment6_ndcg_airline.png")

plt.show()
```

```python id="159d4jefCFqM"

```
