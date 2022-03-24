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

<!-- #region id="dEuxTLeNBu5e" -->
# Medical Drug Recommender
> Recommending top-k drugs for the given medical condition based on historical feedbacks, using Neural Collaborative Filtering model

- toc: true
- badges: true
- comments: true
- categories: [NCF, Keras, Healthcare, Kaggle]
- image:
<!-- #endregion -->

<!-- #region id="OjKDQXUfBpph" -->
<!-- #endregion -->

<!-- #region id="SgIhTvKRlGAS" -->
### Introduction

Machine learning can be used in all aspects of human life, New Drug discovery can be very expensive and can take up to a year, Hence it is very beneficial to find the drug that can be used to treat the specific medical condition from the existing drugs.

The development of a new drug is a tedious, costly, and time-consuming process. Drug repurposing is intended to find alternative uses for a pioneering drug . AI-empowered drug repurposing is a cheaper, faster, and effective approach and can reduce the failures in clinical trials.AI-based Deep learning models can predict drug structures that could potentially treat COVID-19.

With an increase in technology, the information/data about drugs is increasing Data is available for the research  ( eg Drug bank ), Medical researchers around the globe are using this data for various purposes.

With the help of AI and ML techniques, we can find the correlation between different drugs and medical conditions(Recommendation systems), and using this relation we can suggest few drugs that can help in the treatment of the specific medical condition thereby helping the medical researchers and institutions to get the results more quickly and efficiently.

This idea is similar to the idea of recommendation systems where the item is recommended to the user based on certain factors

Similarly, we can recommend the drug for the treatment of certain medical conditions based on certain factors.

To do so, here we are using the Nural collaborative filtering approach to predict the drugs for a specific medical condition. We will be using the Multilayer perceptron model.
<!-- #endregion -->

<!-- #region id="qssvwc1P2_Bm" -->
### Environment setup
<!-- #endregion -->

```python id="jmHibzRajoIq"
# !pip install -q -U kaggle
# !pip install --upgrade --force-reinstall --no-deps kaggle
# !mkdir ~/.kaggle
# !cp /content/drive/MyDrive/kaggle.json ~/.kaggle/
# !chmod 600 ~/.kaggle/kaggle.json
# !kaggle datasets download -d jessicali9530/kuc-hackathon-winter-2018
# !unzip kuc-hackathon-winter-2018.zip
```

```python id="uH8Naer_jioV"
import sys 
import multiprocessing
from time import time

import numpy as np
import pandas as pd
import random
import math
import argparse
import heapq
import scipy.sparse as sp
from sklearn.preprocessing import LabelEncoder

import theano
import theano.tensor as T
import tensorflow as tf
import keras

from keras import layers
from keras.models import Sequential,Model
from keras import backend as K
from keras import initializers
from keras.regularizers import l1, l2 
from keras.layers import Dense, Lambda, Activation
from keras.layers import Embedding, Input, Dense, merge ,Reshape, Flatten, Dropout
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
```

<!-- #region id="Z05qZcoK28-e" -->
### Data loading
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 289} id="KCXvzEHMjmbm" outputId="9e459b55-b6cc-470b-e479-3ec50ce9cbae"
df = pd.read_csv('/content/drugsComTrain_raw.csv')
df.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="TKRAjrwIkEyt" outputId="4b29f509-5665-4f9c-f4a6-43d2e9cbc0d6"
df.info()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 266} id="NhOJdCm_mLeR" outputId="0c0b6b08-564e-41f9-e762-20973885450e"
df.describe(include='all').T
```

<!-- #region id="1lPO4y0C244T" -->
### Data pre-processing
<!-- #endregion -->

```python id="a4VQPNJqyMlc"
le_drug = LabelEncoder()
le_condition = LabelEncoder()

n_condition = 0 # number of conditions
n_drugs = 0 # number of drugs
```

```python id="XNs6gtGUmR2n"
def preprocessing(data, test=False):
    # selecting only usefule columns
    data = data[['condition','drugName','usefulCount']]

    # condition column has some missing values - filling with 'NA'
    data['condition'] = data['condition'].fillna('NA')

    # label encoding
    if test:
        data['condition'] = le_condition.transform(data['condition'])
        data['drugName'] = le_drug.transform(data['drugName'])
    else:
        data['condition'] = le_condition.fit_transform(data['condition'])
        data['drugName'] = le_drug.fit_transform(data['drugName'])
    
    # train set -> sparse matrix
    global n_condition
    global n_drugs
    n_condition = data.condition.nunique()
    n_drugs = data.drugName.nunique()
    train = sp.dok_matrix((n_condition, n_drugs), dtype=np.float32)
    for i in range(len(data['condition'])):
        ls = list(data.iloc[i])
        train[ls[0],ls[1]] = 1.0

    # test set - taking 200 random interactions
    test = []
    for j in range(200):
        i = random.randint(0, len(data))
        ls = list(data.iloc[i])
        test.append([ls[0],ls[1]])

    return train, test
```

```python colab={"base_uri": "https://localhost:8080/"} id="L6jCCQb8yJ15" outputId="7cdfdaa9-d8e8-45d2-9881-cdcac94f73e7"
train, test = preprocessing(df)
train
```

<!-- #region id="lCm-Op3dva7K" -->
### Evaluation methods
To evaluate the performance of drug recommendation, we adopted the leave-one-out evaluation.

HR intuitively measures whether the test item is present on the top-10 list, and the NDCG accounts for the position of the hit by assigning higher scores to hits at top ranks. We will calculate both metrics for each test user and asssine the average score.
<!-- #endregion -->

```python id="6nNNUVulvaCn"
def getHitRatio(ranklist, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1
    return 0

def getNDCG(ranklist, gtItem):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i+2)
    return 0
```

```python id="5_4syID6vwAl"
def evaluate(train, test, model, K):

  HR, NDCG = [],[]

  for i in range(len(test)):                                              
    rating = test[i]                                                     
    u = rating[0]  

    # taking 99 random untested conditions by that drug
    count = 0
    drugs = [] 
    while(count != 99):
      j = random.randint(0, n_drugs-1)     
      if (u,j) in train.keys():
        continue
      drugs.append(j)
      count+=1                                                                      
    gtdrug = rating[1]                                                     
    drugs.append(gtdrug)                                                   
    
    # Get prediction scores
    map_drug_score = {}
    medical_conditions = np.full(len(drugs), u, dtype = 'int32')
    predictions = model.predict([medical_conditions, np.array(drugs)], 
                                 batch_size=64, verbose=0)               
    
    for i in range(len(drugs)):                                            
        drug = drugs[i]
        map_drug_score[drug] = predictions[i]
    drugs.pop()                                                             
    
    ranklist = heapq.nlargest(K, map_drug_score, key=map_drug_score.get)   
    hr = getHitRatio(ranklist, gtdrug)                                       
    ndcg = getNDCG(ranklist, gtdrug)

    HR.append(hr)
    NDCG.append(ndcg) 

  return (HR, NDCG)
```

<!-- #region id="EVvb5YTP2y6K" -->
### Model build
<!-- #endregion -->

<!-- #region id="R3g9J-MzAnFp" -->
<!-- #endregion -->

```python id="-1SYKS8nwrJz"
def get_model(num_medical_conditions, num_drugs, layers = [16,8], reg_layers=[0,0]):

    assert len(layers) == len(reg_layers)
    num_layer = len(layers) # Number of layers in the MLP
    medical_condition_input = Input(shape=(1,), dtype='int32', name = 'user_input')
    drug_input = Input(shape=(1,), dtype='int32', name = 'item_input')

    MLP_Embedding_Medical_Conditions = Embedding(input_dim = num_medical_conditions, output_dim = int(layers[0]/2), 
                                   name = 'medical_condition_embedding', input_length=1)
    MLP_Embedding_Drugs = Embedding(input_dim = num_drugs, output_dim = int(layers[0]/2), 
                                   name = 'drug_embedding', input_length=1)   
    
    medical_condition_latent = Flatten()(MLP_Embedding_Medical_Conditions(medical_condition_input)) # flattening embedding for user 
    drug_latent = Flatten()(MLP_Embedding_Drugs(drug_input)) # flattening embedding for items
    vector = keras.layers.concatenate([medical_condition_latent,drug_latent]) # forming the 0th layer of NN by concatinating the user and items flatten layer 
    
    # MLP layers
    for idx in range(1, num_layer):
        layer = Dense(layers[idx], kernel_regularizer= l2(reg_layers[idx]), activation='relu', name = 'layer%d' %idx)
        vector = layer(vector)
        #layer1 = Dropout(0.25)
        #vector = layer1(vector)
        
    # Final prediction layer
    prediction = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name = 'prediction')(vector)
    model = Model(inputs=[medical_condition_input, drug_input],
                  outputs=prediction)
    
    return model
```

```python id="FVGxo6NCxRk6"
def get_train_instances(train, num_negatives):
  
    medical_condition_input, drug_input, labels = [],[],[]
    num_medical_conditions = train.shape[0]

    for (u, i) in train.keys():

        # positive instance
        medical_condition_input.append(u)
        drug_input.append(i)
        labels.append(1) # 1 for positive instance

        # negative instances
        for t in range(num_negatives):
            j = np.random.randint(num_drugs)
            while ( (u,j) in train.keys() ) :
                j = np.random.randint(num_drugs)
            medical_condition_input.append(u)
            drug_input.append(j)
            labels.append(0) # 0 for negative instance
        
    return medical_condition_input, drug_input, labels
```

<!-- #region id="wgKVD8MO3L5_" -->
### Model training and evaluation
<!-- #endregion -->

```python id="wwDbBw8xyn8M"
path = '/content'
layers = [256,128,64,32,16,8]
reg_layers = [0,0,0,0,0,0]
num_negatives =  6
learner =  'adam'
learning_rate = 0.001
batch_size = 256
epochs = 5
verbose = 1

topK = 3
model_out_file = 'Pretrain_new.h5'  
```

```python colab={"base_uri": "https://localhost:8080/"} id="LdT5ZRLZyt9O" outputId="d8d43fda-67b3-4d28-c707-821475b64c9d"
num_medical_conditions, num_drugs = train.shape

# Build model
model = get_model(num_medical_conditions, num_drugs, layers, reg_layers)

#compile model
model.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy',metrics=['accuracy'])

# Check Init performance
t1 = time()
(hr, ndcg) = evaluate(train, test, model,topK)
HR, NDCG = np.array(hr).mean(), np.array(ndcg).mean()
print('Init: HR = %.4f, NDCG = %.4f [%.1f]' %(HR, NDCG, time()-t1))
```

```python colab={"base_uri": "https://localhost:8080/"} id="M4Pcha_Xyd3J" outputId="07f9a327-1098-4b6b-cb54-5629b8e9ff22"
# Train model
best_hr, best_ndcg, best_iter = HR, NDCG, -1

for epoch in range(epochs):
    t1 = time()

    # Generate training instances
    medical_condition_input, drug_input, labels = get_train_instances(train, num_negatives)

    # Training        
    hist = model.fit([np.array(medical_condition_input), np.array(drug_input)],
                        np.array(labels),batch_size=batch_size, epochs=20, verbose=0, shuffle=True)
    
    t2 = time()

    # Evaluation
    if epoch %verbose == 0:

        (hr, ndcg) = evaluate(train, test, model, topK)
        HR, NDCG, loss = np.array(hr).mean(), np.array(ndcg).mean(), hist.history['loss'][0]
        print('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]' 
                % (epoch,  t2-t1, HR, NDCG, loss, time()-t2))
        
        if HR >= best_hr and NDCG >= best_ndcg:

            best_hr, best_ndcg, best_iter = HR, NDCG, epoch
            model.save(model_out_file)

print("End. Best Iteration %d:  HR = %.4f, NDCG = %.4f. " %(best_iter, best_hr, best_ndcg))
print("The best MLP model is saved to %s" %(model_out_file))
```

<!-- #region id="WnOcs_L-3bkB" -->
### Inference
<!-- #endregion -->

```python id="c58Iehyz4A_B"
Best_model = tf.keras.models.load_model('/content/Pretrain_new.h5', compile = True)
```

```python id="dhGCYRoI4KDx"
def recommend(condition_name='Influenza Prophylaxis', topk=3):

    drugs = [i for i in range(n_drugs)]
    condition_id = le_condition.transform([condition_name])[0]
    medical_conditions = np.full(len(drugs), condition_id, dtype = 'int32')
    predictions = Best_model.predict([medical_conditions, np.array(drugs)], 
                                    batch_size=100, verbose=0)

    map_drug_score ={}
    for i in range(len(drugs)): # creating the{ item : chance } dict 
            drug = drugs[i]
            map_drug_score[drug] = predictions[i]

    ranklist = heapq.nlargest(topk, map_drug_score, key=map_drug_score.get)
    print("{} can be treated by: ".format(condition_name))
    for i in ranklist:
        print("\n\t"+le_drug.inverse_transform([i])[0])
```

<!-- #region id="uq70M4fC3eQj" -->
Let’s select some random medical conditions and recommendations drugs to cure these conditions.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="6pozQ12_46bx" outputId="af7a0c97-5507-42bc-f923-9ce3957f9aa0"
conditions_random3 = np.random.choice(le_condition.classes_, 3)
conditions_random3
```

<!-- #region id="CyiNmUS450rL" -->
**Fibromyalgia**

> Note: I think first letter is 'F' and it is missing from the condition name. We will continue our inference journey by assuming it is `Fibromyalgia`.

*Fibromyalgia is a disorder characterized by widespread musculoskeletal pain accompanied by fatigue, sleep, memory and mood issues. Researchers believe that fibromyalgia amplifies painful sensations by affecting the way your brain and spinal cord process painful and nonpainful signals.*
<!-- #endregion -->

<!-- #region id="JnZawAn86v5h" -->
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="pU77Ch4s_qcB" outputId="e23a2977-c1fd-45af-ef82-e2e1cac8f94d"
recommend(conditions_random3[0])
```

<!-- #region id="VF2yR6jL-UzT" -->
**Influenza**

*Influenza is a viral infection that attacks your respiratory system — your nose, throat and lungs. Influenza is commonly called the flu, but it's not the same as stomach "flu" viruses that cause diarrhea and vomiting. For most people, the flu resolves on its own.*
<!-- #endregion -->

<!-- #region id="gfUcfLRp-zXL" -->
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="P4DUXyV0_ylz" outputId="875c258d-bca4-4c03-d688-f87eb7f712ec"
recommend(conditions_random3[1])
```

<!-- #region id="MBe1-r8M-4zk" -->
**Benign Prostatic Hyperplasia**

*Age-associated prostate gland enlargement that can cause urination difficulty.
This type of prostate enlargement isn't thought to be a precursor to prostate cancer. With this condition, the urinary stream may be weak or stop and start. In some cases, it can lead to infection, bladder stones and reduced kidney function. Treatments include medication that relaxes or shrinks the prostate, surgery and minimally invasive surgery.*
<!-- #endregion -->

<!-- #region id="_xiAKYlo_Qqg" -->
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="G3buc9z25lkC" outputId="6fed3e7d-7f44-4e17-81a4-d51187997d2f"
recommend(conditions_random3[2])
```

<!-- #region id="sf8A3Sj3AAMu" -->
### References
1. https://github.com/mankar1257/Drug-Prediction-using-neural-collaborative-filtering `repo`
2. https://medium.com/analytics-vidhya/neural-collaborative-filtering-for-drug-prediction-e8d0c552317b `blog`
<!-- #endregion -->

<!-- #region id="ujgknp8rAz2t" -->
### Further improvements

This was the basic model for drug prediction. Modification can be done to learn some more complex relations:
1. We can consider the features of the drug ( like protein structure, activity in different environments… )
2. the large data set can also help to increase the radius of the possibilities and can lead to better predictions
3. using more deep NN structure
4. fine-tuning the hyperparameters
<!-- #endregion -->
