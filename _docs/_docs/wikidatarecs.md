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

```python colab={"base_uri": "https://localhost:8080/"} id="sP1UYiJA28vr" executionInfo={"status": "ok", "timestamp": 1626515193914, "user_tz": -330, "elapsed": 3329, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="900376e0-207b-433e-cc91-ecda337b1ef9"
!wget https://github.com/WikidataRec-developer/Wikidata_Recommender/raw/main/Wikidata_Active_Editor.csv
```

```python colab={"base_uri": "https://localhost:8080/"} id="4yreVynj2-hz" executionInfo={"status": "ok", "timestamp": 1626515198784, "user_tz": -330, "elapsed": 1382, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="7547a920-399d-4e86-d1dc-afd0c6ea4bc2"
!wget https://github.com/WikidataRec-developer/Wikidata_Recommender/raw/main/Items_content_of_Wikidata_Active_Editor.csv
```

```python id="DmFzyKr83EOX" executionInfo={"status": "ok", "timestamp": 1626515202113, "user_tz": -330, "elapsed": 387, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import pandas as pd
```

```python colab={"base_uri": "https://localhost:8080/", "height": 419} id="b2ubsqL_3FrI" executionInfo={"status": "ok", "timestamp": 1626515222737, "user_tz": -330, "elapsed": 1954, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="85654416-eaaa-446f-b94c-4cbb401b311d"
df = pd.read_csv('Wikidata_Active_Editor.csv')
df
```

```python colab={"base_uri": "https://localhost:8080/", "height": 419} id="vZVZUc7u3KKg" executionInfo={"status": "ok", "timestamp": 1626515251487, "user_tz": -330, "elapsed": 998, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="252e4e08-8ce9-4419-f433-dfe5b27b6707"
df2 = pd.read_csv('Items_content_of_Wikidata_Active_Editor.csv')
df2
```

```python colab={"base_uri": "https://localhost:8080/"} id="HOMiHxRR3Rq_" executionInfo={"status": "ok", "timestamp": 1626515363936, "user_tz": -330, "elapsed": 1044, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="480f165c-d513-4ac5-c937-9d9fbe3153d7"
!wget https://github.com/WikidataRec-developer/Wikidata_Recommender/raw/cefd891b3f70f21a970681ee4b2e205f88864cd5/Scripts/NMoE.py
```

```python id="1ZPyVZwY4Smn" executionInfo={"status": "ok", "timestamp": 1626515552891, "user_tz": -330, "elapsed": 2887, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import random
import tensorflow.keras
from tensorflow.keras import backend as K
import pandas as pd
import scipy.sparse as sparse
import numpy as np

from sklearn.model_selection import train_test_split

import ast

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import scipy.sparse as sp
import random
import csv
from scipy import sparse
from collections import defaultdict

import tensorflow.keras
from tensorflow.keras import backend as K
```

```python id="NHNKGSIg3jIk" executionInfo={"status": "ok", "timestamp": 1626515557938, "user_tz": -330, "elapsed": 963, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
class Dataset(object):
    
    def __init__(self, interactions, item_content, item_relations):

        ''' Constructor'''
    
        ##Loading the interactions data    
        self.interactions_df = interactions
        
        ##Loading the metadata of items
        self.item_content = item_content
        self.item_relations = item_relations
        
        self.interactions_with_metadata=pd.merge(self.interactions_df, self.item_content, on='item_id')
        self.interactions_with_metadata=pd.merge(self.interactions_with_metadata, self.item_relations, on='item_id')
        
        self.interactions_with_metadata['content_features'] = self.interactions_with_metadata['content_features'].apply(lambda x: ast.literal_eval(x)) 
        self.interactions_with_metadata['relations_features'] = self.interactions_with_metadata['relations_features'].apply(lambda x: ast.literal_eval(x)) 
        
        ## Converting the numbers to categories to be used for creating the categorical codes to avoid using long hash keys 
        self.interactions_with_metadata['user_id'] = self.interactions_with_metadata['user_id'].astype("category")
        self.interactions_with_metadata['item_id'] = self.interactions_with_metadata['item_id'].astype("category")
        
        ## cat.codes creates a categorical id for the users and items
        self.interactions_with_metadata['user_id'] = self.interactions_with_metadata['user_id'].cat.codes
        self.interactions_with_metadata['item_id'] = self.interactions_with_metadata['item_id'].cat.codes
        #print(interactions_with_metadata)
        
        self.interactions_df = self.interactions_with_metadata.drop(['content_features', 'relations_features'], axis=1)
        
        ##Preparing the train and test sets 
        self.train, self.test = self.get_train_test_files(self.interactions_df)        
        self.train_matrix= self.get_data_matrix(self.train, self.interactions_df)
        self.test_matrix= self.get_data_matrix(self.test, self.interactions_df)
        
        self.train_with_metadata, self.test_with_metadata = self.get_train_test_files(self.interactions_with_metadata)
    
    
    
    def get_train_test_files(self, interactions):
        '''Split the interactions into train/test sets 80%/20% in random way'''
        train, test = train_test_split(interactions, test_size=0.20, random_state=42, shuffle=False)
        
        return train, test
    
        
    def get_data_matrix(self, data_set, full_data):
        '''Convert train dataframe into matrix'''
        mat = np.zeros((max(full_data.user_id)+1, max(full_data.item_id)+1))
        for index, line in data_set.iterrows():
            user, item, frequence = line['user_id'], line['item_id'], line['frequence']
            mat[user, item] = frequence
        return mat
```

```python id="_RL9p4yM3xRf" executionInfo={"status": "ok", "timestamp": 1626515558567, "user_tz": -330, "elapsed": 637, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
class Matrix_Factorization(object):
    ##Step1: Loading and preparing the data
    def __init__(self, interactions, data, data_matrix, process):
    
        print('The interaction data') 
        
         
        self.interactions_df = interactions
        self.all_items = self.interactions_df['item_id'].tolist()

        self.n_users = len(self.interactions_df.user_id.unique())
        self.n_items = len(self.interactions_df.item_id.unique())
        print('full n_users')
        print(self.n_users)
        print('full n_items')
        print(self.n_items)

        
        self.data, self.data_matrix= data, data_matrix

        self.data_n_users = len(self.data.user_id.unique())
        self.data_n_items = len(self.data.item_id.unique())

        print('data_set')
        print(self.data.shape)
        print(self.data_n_users)
        print(self.data_n_items)

        print('data_matrix')
        print(self.data_matrix.shape)
        print(self.data_matrix)   

        self.user_interactions = K.constant(data_matrix)
        self.item_interactions = K.constant(data_matrix.T)

        print('The constants')
        print(self.user_interactions.shape)
        print(self.item_interactions.shape)
        
        self.process_name = process
        
        self.MF_model = self.main(self.user_interactions, self.item_interactions, self.data, self.data_n_users, self.data_n_items, self.all_items, self.process_name)


    ##Step2: Build the model
    def identity_loss(y_true, y_pred):
        """Ignore y_true and return the mean of y_pred.
        This is a hack to work-around the design of the Keras API that is
        not really suited to train networks with a triplet loss by default.
        """
        return K.mean(y_pred - 0 * y_true)


    class BPR_triplet_loss(tensorflow.keras.layers.Layer):
        """
            Layer object to minimise the triplet loss.
            We implement the Bayesian Personal Ranking triplet loss.
        """
        
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
         
        def call(self, inputs):
            user_latent, positive_item_latent, negative_item_latent = inputs
        
            # BPR loss
            pos_similarity = K.dot(user_latent,K.transpose(positive_item_latent))
            neg_similarity = K.dot(user_latent,K.transpose(negative_item_latent))
            
            loss = 1.0 - K.sigmoid(pos_similarity - neg_similarity)

            return loss


    class ScoreLayer(tensorflow.keras.layers.Layer):
        """
            Layer object to predict positive matches.
        """
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
        
        def rec_similarity(self, inputs):
            """
                rec_similarity function
            """
            user, item = inputs
            score = K.dot(user,K.transpose(item))
            return score
        
        def call(self, inputs):
            pred = self.rec_similarity(inputs)
            return pred


    def get_model(self, n_users, n_items, latent_dim, u_interactions, i_interactions):
        #Input variables
        user_input = tensorflow.keras.layers.Input(shape=(1,), dtype='int32', name = 'user_input')
        item_input_pos = tensorflow.keras.layers.Input(shape=(1,), dtype='int32', name = 'item_input_positive')
        item_input_neg = tensorflow.keras.layers.Input(shape=(1,), dtype='int32', name = 'item_input_negative')
        
        user_editing_input = tensorflow.keras.layers.Lambda(lambda x: K.gather(u_interactions, x))(user_input)
        user_editing_vector = tensorflow.keras.layers.Flatten()(user_editing_input)
        
        pos_item_editing_input = tensorflow.keras.layers.Lambda(lambda x: K.gather(i_interactions, x))(item_input_pos)
        pos_item_editing_vector = tensorflow.keras.layers.Flatten()(pos_item_editing_input)
        neg_item_editing_input = tensorflow.keras.layers.Lambda(lambda x: K.gather(i_interactions, x))(item_input_neg)
        neg_item_editing_vector = tensorflow.keras.layers.Flatten()(neg_item_editing_input)

        user_layer = tensorflow.keras.layers.Dense(units=latent_dim, name='user_layer')
        
        #Shared layer for positive and negative items
        item_layer = tensorflow.keras.layers.Dense(units=latent_dim, name='item_layer')
            
        #User
        user_latent = user_layer(user_editing_vector)
        
        #Positive
        item_latent_pos = item_layer(pos_item_editing_vector)
        
        #Negative
        item_latent_neg = item_layer(neg_item_editing_vector)
        
        #TripletLoss Layer
        self.loss_layer = self.BPR_triplet_loss(name='triplet_loss_layer')([user_latent, item_latent_pos, item_latent_neg])
        
        ##Model
        network_train = tensorflow.keras.models.Model(inputs=[user_input, item_input_pos, item_input_neg], outputs=self.loss_layer, name = 'training_model')

            
        return network_train
        
        
        
    def main(self, user_interactions, item_interactions, data, n_users, n_items, all_items, process_name):

        latent_dim = 1024

        MF_train = self.get_model(n_users, n_items, latent_dim, user_interactions, item_interactions)

        ##config the model with losses and metrics
        MF_train.compile(loss=self.identity_loss, optimizer=tensorflow.keras.optimizers.Adam(lr=0.001))
        print(MF_train.summary())
        '''print(MF_train.layers)
        print(MF_predict.layers)'''



        ##Step4: train the model and check the performance
        def get_triplets(data):
            user_input,item_pos,item_neg = [],[],[]
            for ind in data.index:
                # Positive instance
                user_input.append(data['user_id'][ind])
                ##Pick one of the positve ids
                item_pos.append(data['item_id'][ind])
                ##Pick one of the negative ids
                nni = random.choice(all_items)
                item_neg.append(nni)
                                   
            return user_input,item_pos,item_neg



        ### Hyper parameters
        batch_size = 32
        epochs = 100

        ### Training
        print("Fit model on training data")
        user_input, item_input_pos, item_input_neg = get_triplets(data, all_items)
        loss = MF_train.fit([np.array(user_input), np.array(item_input_pos), np.array(item_input_neg)], #Triplet input
                                 np.ones(len(user_input)), #Labels
                                 batch_size=batch_size, verbose=1, shuffle=True, validation_split = 0.10, epochs=epochs)


        train_model_out_file = 'Pretrain_MF_Wikidata_14M_{}.tf'.format(process_name) ## The process is train or test 

        MF_train.save_weights(train_model_out_file, overwrite=True)
        print('The train and predict models have saved successfully')
        
        
        return MF_train
```

```python id="d53L3zmE30e6"
#Step1: Loading and reading the data 

##Loading the interactions data 
print('Loading interaction data and metadata data')    
dataset_interactions_file = pd.read_csv('Wikidata-14M.csv', sep=',', encoding= 'utf-8', header=0)
dataset_item_content = pd.read_csv('Wikidata-14M_ELMo_embeddings.csv', sep=',', encoding= 'utf-8', header=0)
dataset_item_relations = pd.read_csv('Wikidata-14M_TransR_ent_embeddings.csv', sep=',', encoding= 'utf-8', header=0)

dataset = Dataset(dataset_interactions_file, dataset_item_content, dataset_item_relations)

print('The full interactions data')
interactions_with_metadata_df = dataset.interactions_with_metadata
print(interactions_with_metadata_df.head())

interactions_df = dataset.interactions_df

all_items = interactions_df['item_id'].tolist()

n_users = len(interactions_df.user_id.unique())
n_items = len(interactions_df.item_id.unique())
print('n_users')
print(n_users)
print('n_items')
print(n_items)


##Loading the train data 
train, train_matrix = dataset.train, dataset.train_matrix
train_with_metadata = dataset.train_with_metadata

train_n_users = len(train.user_id.unique())
train_n_items = len(train.item_id.unique())

print('train_set')
print(train.shape)
print(train_n_users)
print(train_n_items)

print(train_matrix.shape)
#print(train_matrix)   

user_interactions = K.constant(train_matrix)
item_interactions = K.constant(train_matrix.T)

print('The constants')
print(user_interactions.shape)
print(item_interactions.shape)



##Step2: Build the model

def get_model(MF_model, latent_dim):
        
    #Input variables
    user_input = tensorflow.keras.layers.Input(shape=(1,), dtype='int32', name = 'user_input')
    item_input = tensorflow.keras.layers.Input(shape=(1,), dtype='int32', name = 'item_input')
    
    item_contents_input = tensorflow.keras.layers.Input(shape=(1024,), name = 'item_contents_input') 
    item_relations_input = tensorflow.keras.layers.Input(shape=(1024,), name = 'item_relations_input') 

    user_editing_input = tensorflow.keras.layers.Lambda(lambda x: K.gather(user_interactions, x))(user_input)
    user_editing_vector = tensorflow.keras.layers.Flatten()(user_editing_input)
    
    item_editing_input = tensorflow.keras.layers.Lambda(lambda x: K.gather(item_interactions, x))(item_input)
    item_editing_vector = tensorflow.keras.layers.Flatten()(item_editing_input)
    
    #User
    user_latent = MF_model.get_layer('user_layer').get_weights()
    user_latent = tensorflow.keras.layers.Dense(units=latent_dim, name='user_layer', weights=user_latent, trainable=False)(user_editing_vector)
   
    #Item
    item_latent = MF_model.get_layer('item_layer').get_weights()
    item_latent = tensorflow.keras.layers.Dense(units=latent_dim, name='item_layer', weights=item_latent, trainable=False)(item_editing_vector)
    
    #Item metadata
    item_contents = tensorflow.keras.layers.Dense(units=latent_dim, name='item_contents_features')(item_contents_input) 
    item_relations = tensorflow.keras.layers.Dense(units=latent_dim, name='item_relations_features')(item_relations_input) 
        
    #Reshape
    item_latent = tensorflow.keras.layers.Reshape((1,1024))(item_latent)
    item_contents = tensorflow.keras.layers.Reshape((1,1024))(item_contents)
    item_relations = tensorflow.keras.layers.Reshape((1,1024))(item_relations)
    
    #Concatenate layer
    concat_items = tensorflow.keras.layers.Concatenate(axis=1)([item_latent, item_contents, item_relations])
    
    #MoE layer - as CNN layers
    x = tensorflow.keras.layers.Conv1D(1024, 1, activation='relu')(concat_items)
    x = tensorflow.keras.layers.Conv1D(1024, 1, activation='relu')(x)
    ##Soft Gating
    weights = tensorflow.keras.layers.Conv1D(1024, 1, activation='softmax')(x)
    
    ##Adding weights to the features 
    weighted_items = tensorflow.keras.layers.multiply([weights, concat_items])
    
    #Element-wise product of user and item 
    input_vector = tensorflow.keras.layers.multiply([user_latent, weighted_items])
    input_vector = tensorflow.keras.layers.Flatten()(input_vector)
    
    prediction = tensorflow.keras.layers.Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name = 'prediction')(input_vector)
    
    model = tensorflow.keras.models.Model(inputs=[user_input, item_input, item_contents_input, item_relations_input], outputs=prediction)
    
    return model

  
latent_dim = 1024   

MF = Matrix_Factorization(interactions_df, train, train_matrix, 'train')
MF_model = MF.MF_model
MF_model.load_weights('Pretrain_MF_BPR_s8.tf')

Hybrid_model = get_model(MF_model, latent_dim)


##config the model with losses and metrics
Hybrid_model.compile(optimizer=tensorflow.keras.optimizers.Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy', 'AUC', 'Precision', 'Recall'])
print(Hybrid_model.summary())


##Step4: train the model and check the performance
def get_train_instances(data, num_negatives):
    user_input, item_input, item_content, item_relations, labels = [],[],[],[],[]
    count=0
    for ind in data.index:
        count= count + 1
        user_input.append(data['user_id'][ind])
        item_input.append(data['item_id'][ind])
        content = data.loc[data['item_id'] == data['item_id'][ind], 'content_features'].iloc[0]
        relations = data.loc[data['item_id'] == data['item_id'][ind], 'relations_features'].iloc[0]
        item_content.append(content)
        item_relations.append(relations)
        labels.append(1)
        # negative instances
        for t in range(num_negatives):
            #Randomly return one item
            ni = random.choice(all_items)
            user_input.append(data['user_id'][ind])
            item_input.append(ni)
            content = interactions_df.loc[interactions_df['item_id'] == ni, 'content_features'].iloc[0]
            relations = interactions_df.loc[interactions_df['item_id'] == ni, 'relations_features'].iloc[0]
            item_content.append(content)
            item_relations.append(relations)
            labels.append(0)
    return user_input, item_input, item_content, item_relations, labels


user_input, item_input, item_content, item_relations, labels = get_train_instances(train_with_metadata, 1) 
user_input = np.array(user_input)
item_input = np.array(item_input)
item_content = np.array(item_content)
item_relations = np.array(item_relations)
labels = np.array(labels)

print("Fit model on training data")
hist = Hybrid_model.fit([user_input, item_input, item_content, item_relations], #input
                     labels, # labels 
                     batch_size=32, epochs=100, verbose=1, validation_split=0.10, shuffle=True)


model_out_file = 'Pretrain_NMoE_Wikidata-14M.h5'

Hybrid_model.save(model_out_file, overwrite=True)
print('The NMoE model has saved successfully')
```
