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

<!-- #region id="o5AaP597b2Y3" -->
# Multi-Task Learning

> Training 5 models on census data taking salary and marital status as proxies for CTR and CTCVR proxy. Objective is to learn about different models and how they work.
<!-- #endregion -->

<!-- #region id="jTBvw75NcuQp" -->
## Setup
<!-- #endregion -->

```python id="IZjIdtVyPqK-"
!pip install deepctr[cpu]
```

```python id="_2NnJXRUQDY8"
# !wget -q --show-progress http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
# !wget -q --show-progress http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test
```

```python id="WV5ZuOGTLGaD" executionInfo={"status": "ok", "timestamp": 1635311782621, "user_tz": -330, "elapsed": 1687, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
!git clone -q https://github.com/sparsh-ai/multiobjective-optimizations.git
!cp multiobjective-optimizations/data/adult.data .
!cp multiobjective-optimizations/data/adult.test .
```

```python id="4KwQhlFbVLtv" executionInfo={"status": "ok", "timestamp": 1635311815541, "user_tz": -330, "elapsed": 2484, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import roc_auc_score

import tensorflow as tf

from deepctr.feature_column import build_input_features, input_from_feature_columns
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names

from deepctr.layers.core import PredictionLayer, DNN
from deepctr.layers.utils import combined_dnn_input 

import warnings
warnings.filterwarnings('ignore')
```

<!-- #region id="hrtoxsdZZvTc" -->
## Tasks
<!-- #endregion -->

<!-- #region id="Z_8ZpKvGZw4q" -->
| Task | Goal | Proxy |
| - | -:| ---:|
| Task 1 | Predict whether the income exceeds 50K | ctr |
| Task 2 | Predict whether this person’s marital status is never married | ctcvr |
<!-- #endregion -->

<!-- #region id="e3fReRLqZ8uX" -->
## Data
<!-- #endregion -->

```python id="s8eRPP47Z-Te" executionInfo={"status": "ok", "timestamp": 1635311815546, "user_tz": -330, "elapsed": 11, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
CENSUS_COLUMNS = ['age','workclass','fnlwgt','education','education_num',
                  'marital_status','occupation','relationship','race','gender',
                  'capital_gain','capital_loss','hours_per_week','native_country',
                  'income_bracket']

df_train = pd.read_csv('adult.data',header=None,names=CENSUS_COLUMNS)
df_test = pd.read_csv('adult.test',header=None,names=CENSUS_COLUMNS)

data = pd.concat([df_train, df_test], axis=0)

#take task1 as ctr task, take task2 as ctcvr task.
data['ctr_label'] = data['income_bracket'].map({' >50K.':1, ' >50K':1, ' <=50K.':0, ' <=50K':0})
data['ctcvr_label'] = data['marital_status'].apply(lambda x: 1 if x==' Never-married' else 0)
data.drop(labels=['marital_status', 'income_bracket'], axis=1, inplace=True)
```

<!-- #region id="GkJkFfoXawZs" -->
## Features
<!-- #endregion -->

```python id="hZEZteUgayC9" executionInfo={"status": "ok", "timestamp": 1635312052503, "user_tz": -330, "elapsed": 645, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
#define dense and sparse features
columns = data.columns.values.tolist()
dense_features = ['fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
sparse_features = [col for col in columns if col not in dense_features and col not in ['ctr_label', 'ctcvr_label']]

data[sparse_features] = data[sparse_features].fillna('-1', )
data[dense_features] = data[dense_features].fillna(0, )
mms = MinMaxScaler(feature_range=(0, 1))
data[dense_features] = mms.fit_transform(data[dense_features])
    
for feat in sparse_features:
    lbe = LabelEncoder()
    data[feat] = data[feat].astype('str')
    data[feat] = lbe.fit_transform(data[feat])
    
fixlen_feature_columns = [SparseFeat(feat, data[feat].max()+1, embedding_dim=16)for feat in sparse_features] \
+ [DenseFeat(feat, 1,) for feat in dense_features]

dnn_feature_columns = fixlen_feature_columns

feature_names = get_feature_names(dnn_feature_columns)
```

<!-- #region id="WCt1FpDLa4kL" -->
## Train & Test Data
<!-- #endregion -->

```python id="o-oz915Fa7f6" executionInfo={"status": "ok", "timestamp": 1635312057210, "user_tz": -330, "elapsed": 500, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
#train test split
n_train = df_train.shape[0]
train = data[:n_train]
test = data[n_train:]
train_model_input = {name: train[name] for name in feature_names}
test_model_input = {name: test[name] for name in feature_names}
```

<!-- #region id="nPWtKmlVQcys" -->
## Shared Bottom
<!-- #endregion -->

```python id="o4dK-MS4QR9A" executionInfo={"status": "ok", "timestamp": 1635312060969, "user_tz": -330, "elapsed": 471, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
# import tensorflow as tf

# from deepctr.feature_column import build_input_features, input_from_feature_columns
# from deepctr.layers.core import PredictionLayer, DNN
# from deepctr.layers.utils import combined_dnn_input 

def Shared_Bottom(dnn_feature_columns, num_tasks, task_types, task_names,
                  bottom_dnn_units=[128, 128], tower_dnn_units_lists=[[64,32], [64,32]],
                  l2_reg_embedding=0.00001, l2_reg_dnn=0, seed=1024, dnn_dropout=0,
                 dnn_activation='relu', dnn_use_bn=False):
    """Instantiates the Shared_Bottom multi-task learning Network architecture.
    
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param num_tasks:  integer, number of tasks, equal to number of outputs, must be greater than 1.
    :param task_types: list of str, indicating the loss of each tasks, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss. e.g. ['binary', 'regression']
    :param task_names: list of str, indicating the predict target of each tasks
    :param bottom_dnn_units: list,list of positive integer or empty list, the layer number and units in each layer of shared-bottom DNN
    :param tower_dnn_units_lists: list, list of positive integer list, its length must be euqal to num_tasks, the layer number and units in each layer of task-specific DNN
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in DNN
    :return: A Keras model instance.
    """
    if num_tasks <= 1:
        raise ValueError("num_tasks must be greater than 1")
    if len(task_types) != num_tasks:
        raise ValueError("num_tasks must be equal to the length of task_types")
        
    for task_type in task_types:
        if task_type not in ['binary', 'regression']:
            raise ValueError("task must be binary or regression, {} is illegal".format(task_type))
            
    if num_tasks != len(tower_dnn_units_lists):
        raise ValueError("the length of tower_dnn_units_lists must be euqal to num_tasks")

    features = build_input_features(dnn_feature_columns)
    inputs_list = list(features.values())
    
    sparse_embedding_list, dense_value_list = input_from_feature_columns(features, dnn_feature_columns, l2_reg_embedding,seed)

    dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
    shared_bottom_output = DNN(bottom_dnn_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed)(dnn_input)

    tasks_output = []
    for task_type, task_name, tower_dnn in zip(task_types, task_names, tower_dnn_units_lists):
        tower_output = DNN(tower_dnn, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed, name='tower_'+task_name)(shared_bottom_output)
        
        logit = tf.keras.layers.Dense(1, use_bias=False, activation=None)(tower_output)
        output = PredictionLayer(task_type, name=task_name)(logit) #regression->keep, binary classification->sigmoid
        tasks_output.append(output)

    model = tf.keras.models.Model(inputs=inputs_list, outputs=tasks_output)
    return model
```

```python colab={"base_uri": "https://localhost:8080/"} id="t4ysNDKebMdz" executionInfo={"status": "ok", "timestamp": 1635312425191, "user_tz": -330, "elapsed": 8743, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="e9764e29-6613-451a-82ac-10c9089b7fff"
task_names = ['income', 'marital']
model = Shared_Bottom(dnn_feature_columns, num_tasks=2, task_types= ['binary', 'binary'], task_names=task_names, bottom_dnn_units=[128, 128], tower_dnn_units_lists=[[64,32], [64,32]])

model.compile("adam", loss=["binary_crossentropy", "binary_crossentropy"], metrics=['AUC'])
history = model.fit(train_model_input, [train['ctr_label'].values, train['ctcvr_label'].values],batch_size=256, epochs=5, verbose=2, validation_split=0.0 )

pred_ans = model.predict(test_model_input, batch_size=256)
```

<!-- #region id="4LZ8mzYeVVUc" -->
## ESSM
<!-- #endregion -->

```python id="rJd4euhoVVUx" executionInfo={"status": "ok", "timestamp": 1635312450778, "user_tz": -330, "elapsed": 776, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
# import tensorflow as tf

# from deepctr.feature_column import build_input_features, input_from_feature_columns
# from deepctr.layers.core import PredictionLayer, DNN
# from deepctr.layers.utils import combined_dnn_input 


def ESSM(dnn_feature_columns, task_type='binary', task_names=['ctr', 'ctcvr'],
        tower_dnn_units_lists=[[128, 128],[128, 128]], l2_reg_embedding=0.00001, l2_reg_dnn=0, 
         seed=1024, dnn_dropout=0,dnn_activation='relu', dnn_use_bn=False):
    """Instantiates the Entire Space Multi-Task Model architecture.
    
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param task_type:  str, indicating the loss of each tasks, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss.
    :param task_names: list of str, indicating the predict target of each tasks. default value is ['ctr', 'ctcvr']
    :param tower_dnn_units_lists: list, list of positive integer, the length must be equal to 2, the layer number and units in each layer of task-specific DNN
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in DNN
    :return: A Keras model instance.
    """
    if len(task_names)!=2:
        raise ValueError("the length of task_names must be equal to 2")
    
    if len(tower_dnn_units_lists)!=2:
        raise ValueError("the length of tower_dnn_units_lists must be equal to 2")

    features = build_input_features(dnn_feature_columns)
    inputs_list = list(features.values())
    
    sparse_embedding_list, dense_value_list = input_from_feature_columns(features, dnn_feature_columns, l2_reg_embedding,seed)

    dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
    
    ctr_output = DNN(tower_dnn_units_lists[0], dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed)(dnn_input)
    cvr_output = DNN(tower_dnn_units_lists[1], dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed)(dnn_input)
    
    ctr_logit = tf.keras.layers.Dense(1, use_bias=False, activation=None)(ctr_output)
    cvr_logit = tf.keras.layers.Dense(1, use_bias=False, activation=None)(cvr_output)
    
    ctr_pred = PredictionLayer(task_type, name=task_names[0])(ctr_logit) 
    cvr_pred = PredictionLayer(task_type)(cvr_logit) 
    
    ctcvr_pred = tf.keras.layers.Multiply(name=task_names[1])([ctr_pred, cvr_pred]) #CTCVR = CTR * CVR

    model = tf.keras.models.Model(inputs=inputs_list, outputs=[ctr_pred, ctcvr_pred])    
    return model
```

```python colab={"base_uri": "https://localhost:8080/"} id="IX6jiuBLZf56" executionInfo={"status": "ok", "timestamp": 1635312457674, "user_tz": -330, "elapsed": 6105, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="fa967eca-c267-456e-eb9b-952ebe12aaf0"
model = ESSM(dnn_feature_columns, task_type='binary', task_names=['ctr', 'ctcvr'],
        tower_dnn_units_lists=[[64, 64],[64, 64]])
model.compile("adam", loss=["binary_crossentropy", "binary_crossentropy"],
              metrics=['AUC'])

history = model.fit(train_model_input, [train['ctr_label'].values, train['ctcvr_label'].values],batch_size=256, epochs=5, verbose=2, validation_split=0.0 )

pred_ans = model.predict(test_model_input, batch_size=256)
```

<!-- #region id="xVOqqp7JXG8Y" -->
## MMoE
<!-- #endregion -->

```python id="6p2MP7eaXG8i" executionInfo={"status": "ok", "timestamp": 1635312468060, "user_tz": -330, "elapsed": 820, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
# import tensorflow as tf

# from deepctr.feature_column import build_input_features, input_from_feature_columns
# from deepctr.layers.core import PredictionLayer, DNN
# from deepctr.layers.utils import combined_dnn_input

def MMOE(dnn_feature_columns, num_tasks, task_types, task_names, num_experts=4, 
          expert_dnn_units=[32,32],  gate_dnn_units=[16,16], tower_dnn_units_lists=[[16,8],[16,8]],
          l2_reg_embedding=1e-5, l2_reg_dnn=0, seed=1024, dnn_dropout=0, dnn_activation='relu', dnn_use_bn=False):
    """Instantiates the Multi-gate Mixture-of-Experts multi-task learning architecture.
    
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param num_tasks: integer, number of tasks, equal to number of outputs, must be greater than 1.
    :param task_types: list of str, indicating the loss of each tasks, ``"binary"`` for  binary logloss, ``"regression"`` for regression loss. e.g. ['binary', 'regression']
    :param task_names: list of str, indicating the predict target of each tasks
    
    :param num_experts: integer, number of experts.
    :param expert_dnn_units: list, list of positive integer, its length must be greater than 1, the layer number and units in each layer of expert DNN
    :param gate_dnn_units: list, list of positive integer, its length must be greater than 1, the layer number and units in each layer of gate DNN
    :param tower_dnn_units_lists: list, list of positive integer list, its length must be euqal to num_tasks, the layer number and units in each layer of task-specific DNN
    
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in DNN
    :return: a Keras model instance
    """
    
    if num_tasks <= 1:
        raise ValueError("num_tasks must be greater than 1")
    if len(task_types) != num_tasks:
        raise ValueError("num_tasks must be equal to the length of task_types")
        
    for task_type in task_types:
        if task_type not in ['binary', 'regression']:
            raise ValueError("task must be binary or regression, {} is illegal".format(task_type))
            
    if num_tasks != len(tower_dnn_units_lists):
        raise ValueError("the length of tower_dnn_units_lists must be euqal to num_tasks")

    features = build_input_features(dnn_feature_columns)

    inputs_list = list(features.values())

    sparse_embedding_list, dense_value_list = input_from_feature_columns(features, dnn_feature_columns,
                                                                         l2_reg_embedding, seed)
    dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
    
    #build expert layer
    expert_outs = []
    for i in range(num_experts):
        expert_network = DNN(expert_dnn_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed, name='expert_'+str(i))(dnn_input)
        expert_outs.append(expert_network)
    expert_concat = tf.keras.layers.concatenate(expert_outs, axis=1, name='expert_concat')
    expert_concat = tf.keras.layers.Reshape([num_experts, expert_dnn_units[-1]], name='expert_reshape')(expert_concat) #(num_experts, output dim of expert_network)

    mmoe_outs = []
    for i in range(num_tasks): #one mmoe layer: nums_tasks = num_gates
        #build gate layers
        gate_network = DNN(gate_dnn_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed, name='gate_'+task_names[i])(dnn_input)
        gate_out = tf.keras.layers.Dense(num_experts, use_bias=False, activation='softmax', name='gate_softmax_'+task_names[i])(gate_network)
        gate_out = tf.tile(tf.expand_dims(gate_out, axis=-1), [1, 1, expert_dnn_units[-1]]) #let the shape of gate_out be (num_experts, output dim of expert_network)

        #gate multiply the expert
        gate_mul_expert = tf.keras.layers.Multiply(name='gate_mul_expert_'+task_names[i])([expert_concat, gate_out]) 
        gate_mul_expert = tf.math.reduce_sum(gate_mul_expert, axis=1) #sum pooling in the expert ndim
        mmoe_outs.append(gate_mul_expert)
    
    task_outs = []
    for task_type, task_name, tower_dnn, mmoe_out in zip(task_types, task_names, tower_dnn_units_lists, mmoe_outs):
        #build tower layer
        tower_output = DNN(tower_dnn, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed, name='tower_'+task_name)(mmoe_out)
        
        logit = tf.keras.layers.Dense(1, use_bias=False, activation=None)(tower_output)
        output = PredictionLayer(task_type, name=task_name)(logit) 
        task_outs.append(output)
        
    model = tf.keras.models.Model(inputs=inputs_list, outputs=task_outs)
    return model
```

```python colab={"base_uri": "https://localhost:8080/"} id="37XlaPF2bQ3t" executionInfo={"status": "ok", "timestamp": 1635312486435, "user_tz": -330, "elapsed": 17542, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="28e1d482-add5-4baa-ba7a-fd4ae5c9f477"
model = MMOE(dnn_feature_columns, num_tasks=2, task_types=['binary', 'binary'], task_names=task_names, 
num_experts=8, expert_dnn_units=[64,64], gate_dnn_units=[32,32], tower_dnn_units_lists=[[32,32],[32,32]])
model.compile("adam", loss=["binary_crossentropy", "binary_crossentropy"], metrics=['AUC'])

history = model.fit(train_model_input, [train['ctr_label'].values, train['ctcvr_label'].values], batch_size=256, epochs=5, verbose=2, validation_split=0.0 )

pred_ans = model.predict(test_model_input, batch_size=256)
```

<!-- #region id="YGUyMNPgVV3e" -->
## CGC
<!-- #endregion -->

```python id="Roq2o3zzVV3w" executionInfo={"status": "ok", "timestamp": 1635312486436, "user_tz": -330, "elapsed": 13, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
# import tensorflow as tf

# from deepctr.feature_column import build_input_features, input_from_feature_columns
# from deepctr.layers.core import PredictionLayer, DNN
# from deepctr.layers.utils import combined_dnn_input 

def PLE_CGC(dnn_feature_columns, num_tasks, task_types, task_names, num_experts_specific=8, num_experts_shared=4,
          expert_dnn_units=[64,64],  gate_dnn_units=[16,16], tower_dnn_units_lists=[[16,16],[16,16]],
          l2_reg_embedding=1e-5, l2_reg_dnn=0, seed=1024, dnn_dropout=0, dnn_activation='relu', dnn_use_bn=False):
    """Instantiates the Customized Gate Control block of Progressive Layered Extraction architecture.
    
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param num_tasks: integer, number of tasks, equal to number of outputs, must be greater than 1.
    :param task_types: list of str, indicating the loss of each tasks, ``"binary"`` for  binary logloss, ``"regression"`` for regression loss. e.g. ['binary', 'regression']
    :param task_names: list of str, indicating the predict target of each tasks
    
    :param num_experts_specific: integer, number of task-specific experts.
    :param num_experts_shared: integer, number of task-shared experts.
    :param expert_dnn_units: list, list of positive integer, its length must be greater than 1, the layer number and units in each layer of expert DNN
    :param gate_dnn_units: list, list of positive integer, its length must be greater than 1, the layer number and units in each layer of gate DNN
    :param tower_dnn_units_lists: list, list of positive integer list, its length must be euqal to num_tasks, the layer number and units in each layer of task-specific DNN
    
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in DNN
    :return: a Keras model instance
    """
    
    if num_tasks <= 1:
        raise ValueError("num_tasks must be greater than 1")
    if len(task_types) != num_tasks:
        raise ValueError("num_tasks must be equal to the length of task_types")
        
    for task_type in task_types:
        if task_type not in ['binary', 'regression']:
            raise ValueError("task must be binary or regression, {} is illegal".format(task_type))
            
    if num_tasks != len(tower_dnn_units_lists):
        raise ValueError("the length of tower_dnn_units_lists must be euqal to num_tasks")

    features = build_input_features(dnn_feature_columns)

    inputs_list = list(features.values())

    sparse_embedding_list, dense_value_list = input_from_feature_columns(features, dnn_feature_columns,
                                                                         l2_reg_embedding, seed)
    dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
    
    expert_outputs = []
    #build task-specific expert layer
    for i in range(num_tasks):
        for j in range(num_experts_specific):
            expert_network = DNN(expert_dnn_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed, name='task_'+task_names[i]+'_expert_specific_'+str(j))(dnn_input)
            expert_outputs.append(expert_network)

    #build task-shared expert layer
    for i in range(num_experts_shared):
        expert_network = DNN(expert_dnn_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed, name='expert_shared_'+str(i))(dnn_input)
        expert_outputs.append(expert_network)
        


    cgc_outs = []
    for i in range(num_tasks): 
        #concat task-specific expert and task-shared expert
        cur_expert_num = num_experts_specific + num_experts_shared
        cur_experts = expert_outputs[i * num_experts_specific:(i + 1) * num_experts_specific] + expert_outputs[-int(num_experts_shared):] #task_specific + task_shared
        expert_concat = tf.keras.layers.concatenate(cur_experts, axis=1, name='expert_concat_'+task_names[i])
        expert_concat = tf.keras.layers.Reshape([cur_expert_num, expert_dnn_units[-1]], name='expert_reshape_'+task_names[i])(expert_concat)
        
        #build gate layers
        gate_network = DNN(gate_dnn_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed, name='gate_'+task_names[i])(dnn_input)
        gate_out = tf.keras.layers.Dense(cur_expert_num, use_bias=False, activation='softmax', name='gate_softmax_'+task_names[i])(gate_network)
        gate_out = tf.tile(tf.expand_dims(gate_out, axis=-1), [1, 1, expert_dnn_units[-1]]) 
        
        #gate multiply the expert
        gate_mul_expert = tf.keras.layers.Multiply(name='gate_mul_expert_'+task_names[i])([expert_concat, gate_out]) 
        gate_mul_expert = tf.math.reduce_sum(gate_mul_expert, axis=1) #sum pooling in the expert ndim
        cgc_outs.append(gate_mul_expert)
    
    task_outs = []
    for task_type, task_name, tower_dnn, cgc_out in zip(task_types, task_names, tower_dnn_units_lists, cgc_outs):
        #build tower layer
        tower_output = DNN(tower_dnn, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed, name='tower_'+task_name)(cgc_out)
        logit = tf.keras.layers.Dense(1, use_bias=False, activation=None)(tower_output)
        output = PredictionLayer(task_type, name=task_name)(logit) 
        task_outs.append(output)
        
    model = tf.keras.models.Model(inputs=inputs_list, outputs=task_outs)
    return model
```

```python colab={"base_uri": "https://localhost:8080/"} id="PGDbvHTMbWXd" executionInfo={"status": "ok", "timestamp": 1635312517472, "user_tz": -330, "elapsed": 31047, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="79dab04c-55f4-4680-cc8a-8f7fae0d5dbc"
model = PLE_CGC(dnn_feature_columns, num_tasks=2, task_types=['binary', 'binary'], task_names=task_names, 
num_experts_specific=8, num_experts_shared=4, expert_dnn_units=[64,64],  gate_dnn_units=[16,16], tower_dnn_units_lists=[[32,32],[32,32]])
model.compile("adam", loss=["binary_crossentropy", "binary_crossentropy"], metrics=['AUC'])

history = model.fit(train_model_input, [train['ctr_label'].values, train['ctcvr_label'].values], batch_size=256, epochs=5, verbose=2, validation_split=0.0 )

pred_ans = model.predict(test_model_input, batch_size=256)
```

<!-- #region id="oNCTMODkVWGg" -->
## PLE
<!-- #endregion -->

```python id="58llgjkxVWGs" executionInfo={"status": "ok", "timestamp": 1635312519416, "user_tz": -330, "elapsed": 738, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
# import tensorflow as tf

# from deepctr.feature_column import build_input_features, input_from_feature_columns
# from deepctr.layers.core import PredictionLayer, DNN
# from deepctr.layers.utils import combined_dnn_input 

def PLE(dnn_feature_columns, num_tasks, task_types, task_names, num_levels=1, num_experts_specific=8, num_experts_shared=4,
          expert_dnn_units=[64,64],  gate_dnn_units=[16,16], tower_dnn_units_lists=[[16,16],[16,16]],
          l2_reg_embedding=1e-5, l2_reg_dnn=0, seed=1024, dnn_dropout=0, dnn_activation='relu', dnn_use_bn=False):
    """Instantiates the multi level of Customized Gate Control of Progressive Layered Extraction architecture.
    
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param num_tasks: integer, number of tasks, equal to number of outputs, must be greater than 1.
    :param task_types: list of str, indicating the loss of each tasks, ``"binary"`` for  binary logloss, ``"regression"`` for regression loss. e.g. ['binary', 'regression']
    :param task_names: list of str, indicating the predict target of each tasks
    
    :param num_levels: integer, number of CGC levels.
    :param num_experts_specific: integer, number of task-specific experts.
    :param num_experts_shared: integer, number of task-shared experts.
    :param expert_dnn_units: list, list of positive integer, its length must be greater than 1, the layer number and units in each layer of expert DNN
    :param gate_dnn_units: list, list of positive integer, its length must be greater than 1, the layer number and units in each layer of gate DNN
    :param tower_dnn_units_lists: list, list of positive integer list, its length must be euqal to num_tasks, the layer number and units in each layer of task-specific DNN
    
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in DNN
    :return: a Keras model instance
    """
    
    if num_tasks <= 1:
        raise ValueError("num_tasks must be greater than 1")
    if len(task_types) != num_tasks:
        raise ValueError("num_tasks must be equal to the length of task_types")
        
    for task_type in task_types:
        if task_type not in ['binary', 'regression']:
            raise ValueError("task must be binary or regression, {} is illegal".format(task_type))
            
    if num_tasks != len(tower_dnn_units_lists):
        raise ValueError("the length of tower_dnn_units_lists must be euqal to num_tasks")

    features = build_input_features(dnn_feature_columns)

    inputs_list = list(features.values())

    sparse_embedding_list, dense_value_list = input_from_feature_columns(features, dnn_feature_columns,
                                                                         l2_reg_embedding, seed)
    dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
 
    #single cgc layer
    def cgc_net(inputs, level_name, is_last=False):
        #inputs: [task1, task2, ... taskn, shared task]
        
        expert_outputs = []
        #build task-specific expert layer
        for i in range(num_tasks):
            for j in range(num_experts_specific):
                expert_network = DNN(expert_dnn_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed, name=level_name+'task_'+task_names[i]+'_expert_specific_'+str(j))(inputs[i])
                expert_outputs.append(expert_network)

        #build task-shared expert layer
        for i in range(num_experts_shared):
            expert_network = DNN(expert_dnn_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed, name=level_name+'expert_shared_'+str(i))(inputs[-1]) 
            expert_outputs.append(expert_network)

        #task_specific gate (count = num_tasks)
        cgc_outs = []
        for i in range(num_tasks): 
            #concat task-specific expert and task-shared expert
            cur_expert_num = num_experts_specific + num_experts_shared
            cur_experts = expert_outputs[i * num_experts_specific:(i + 1) * num_experts_specific] + expert_outputs[-int(num_experts_shared):] #task_specific + task_shared
            
            expert_concat = tf.keras.layers.concatenate(cur_experts, axis=1, name=level_name+'expert_concat_specific_'+task_names[i])
            expert_concat = tf.keras.layers.Reshape([cur_expert_num, expert_dnn_units[-1]], name=level_name+'expert_reshape_specific_'+task_names[i])(expert_concat)

            #build gate layers
            gate_network = DNN(gate_dnn_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed, name=level_name+'gate_specific_'+task_names[i])(inputs[i]) #gate[i] for task input[i]
            gate_out = tf.keras.layers.Dense(cur_expert_num, use_bias=False, activation='softmax', name=level_name+'gate_softmax_specific_'+task_names[i])(gate_network)
            gate_out = tf.tile(tf.expand_dims(gate_out, axis=-1), [1, 1, expert_dnn_units[-1]]) 

            #gate multiply the expert
            gate_mul_expert = tf.keras.layers.Multiply(name=level_name+'gate_mul_expert_specific_'+task_names[i])([expert_concat, gate_out]) 
            gate_mul_expert = tf.math.reduce_sum(gate_mul_expert, axis=1) #sum pooling in the expert ndim
            cgc_outs.append(gate_mul_expert)
        
        #task_shared gate, if the level not in last, add one shared gate
        if not is_last:
            cur_expert_num = num_tasks * num_experts_specific + num_experts_shared
            cur_experts = expert_outputs #all the expert include task-specific expert and task-shared expert
            
            expert_concat = tf.keras.layers.concatenate(cur_experts, axis=1, name=level_name+'expert_concat_shared_'+task_names[i])
            expert_concat = tf.keras.layers.Reshape([cur_expert_num, expert_dnn_units[-1]], name=level_name+'expert_reshape_shared_'+task_names[i])(expert_concat)
            
            #build gate layers
            gate_network = DNN(gate_dnn_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed, name=level_name+'gate_shared_'+str(i))(inputs[-1]) #gate for shared task input
            gate_out = tf.keras.layers.Dense(cur_expert_num, use_bias=False, activation='softmax', name=level_name+'gate_softmax_shared_'+str(i))(gate_network)
            gate_out = tf.tile(tf.expand_dims(gate_out, axis=-1), [1, 1, expert_dnn_units[-1]]) 

            #gate multiply the expert
            gate_mul_expert = tf.keras.layers.Multiply(name=level_name+'gate_mul_expert_shared_'+task_names[i])([expert_concat, gate_out]) 
            gate_mul_expert = tf.math.reduce_sum(gate_mul_expert, axis=1) #sum pooling in the expert ndim
            cgc_outs.append(gate_mul_expert)
        return cgc_outs
    
    ple_inputs = [dnn_input]*(num_tasks+1) #[task1, task2, ... taskn, shared task]
    ple_outputs = []
    for i in range(num_levels):
        if i == num_levels-1: #the last level
            ple_outputs = cgc_net(inputs=ple_inputs, level_name='level_'+str(i)+'_', is_last=True)
            break
        else:
            ple_outputs = cgc_net(inputs=ple_inputs, level_name='level_'+str(i)+'_', is_last=False)
            ple_inputs = ple_outputs
            
    task_outs = []
    for task_type, task_name, tower_dnn, ple_out in zip(task_types, task_names, tower_dnn_units_lists, ple_outputs):
        #build tower layer
        tower_output = DNN(tower_dnn, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed, name='tower_'+task_name)(ple_out)
        logit = tf.keras.layers.Dense(1, use_bias=False, activation=None)(tower_output)
        output = PredictionLayer(task_type, name=task_name)(logit) 
        task_outs.append(output)
        
    model = tf.keras.models.Model(inputs=inputs_list, outputs=task_outs)
    return model
```

```python colab={"base_uri": "https://localhost:8080/"} id="Gx0FBpalbcvA" executionInfo={"status": "ok", "timestamp": 1635312574946, "user_tz": -330, "elapsed": 55538, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="251c116a-82b9-4b19-ef23-de0dc2de2aac"
#Test PLE Model

model = PLE(dnn_feature_columns, num_tasks=2, task_types=['binary', 'binary'], task_names=task_names, 
num_levels=2, num_experts_specific=8, num_experts_shared=4, expert_dnn_units=[64,64],  gate_dnn_units=[16,16], tower_dnn_units_lists=[[32,32],[32,32]])
model.compile("adam", loss=["binary_crossentropy", "binary_crossentropy"], metrics=['AUC'])

history = model.fit(train_model_input, [train['ctr_label'].values, train['ctcvr_label'].values], batch_size=256, epochs=5, verbose=2, validation_split=0.0 )

pred_ans = model.predict(test_model_input, batch_size=256)
```

<!-- #region id="dby3KYwfOatU" -->
## Watermark
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="1OXidUrYOc3i" executionInfo={"status": "ok", "timestamp": 1635312610609, "user_tz": -330, "elapsed": 4202, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="835a3c4f-c399-4a33-c83c-51cdaebd8064"
!pip install -q watermark
%reload_ext watermark
%watermark -m -iv -u -t -d
```
