---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.7
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

<!-- #region id="eIV0pbarAxkG" -->
# Transformers4Rec XLNet on Synthetic data 
<!-- #endregion -->

<!-- #region id="f0f8b4e3" -->
In this tutorial we introduce the [Transformers4Rec](https://github.com/NVIDIA-Merlin/Transformers4Rec) library for sequential and session-based recommendation. This tutorial uses the PyTorch API, but a TensorFlow API is also available. Transformers4Rec integrates with the popular [HuggingFace’s Transformers](https://github.com/huggingface/transformers) and make it possible to experiment with cutting-edge implementation of the latest NLP Transformer architectures.  

We demonstrate how to build a session-based recommendation model with the [XLNET](https://arxiv.org/abs/1906.08237) Transformer architecture. The XLNet architecture was designed to leverage the best of both auto-regressive language modeling and auto-encoding with its Permutation Language Modeling training method. In this example we will use XLNET with masked language modeling (MLM) training method, which showed very promising results.
<!-- #endregion -->

<!-- #region id="ab6085c0" -->
First, we are going to generate synthetic data and then create sequential features with [NVTabular](https://github.com/NVIDIA-Merlin/NVTabular). Such data will be used to train a session-based recommendation model.

NVTabular is a feature engineering and preprocessing library for tabular data designed to quickly and easily manipulate terabyte scale datasets used to train deep learning based recommender systems. It provides a high level abstraction to simplify code and accelerates computation on the GPU using the RAPIDS cuDF library.
<!-- #endregion -->

<!-- #region id="add26d16" -->
## Import required libraries
<!-- #endregion -->

```python id="1e8dae24"
import os
import glob

import torch 
import numpy as np
import pandas as pd

import cudf
import cupy as cp
import nvtabular as nvt

from transformers4rec import torch as tr
from transformers4rec.torch.ranking_metric import NDCGAt, AvgPrecisionAt, RecallAt
```

<!-- #region id="c3206b3f" -->
## Define Input/Output Path
<!-- #endregion -->

```python id="105dd71c"
INPUT_DATA_DIR = os.environ.get("INPUT_DATA_DIR", "/workspace/data/")
```

<!-- #region id="36498a01" -->
## Create a Synthetic Input Data
<!-- #endregion -->

```python id="929036ae"
NUM_ROWS = 100000
long_tailed_item_distribution = np.clip(np.random.lognormal(3., 1., NUM_ROWS).astype(np.int32), 1, 50000)

# generate random item interaction features 
df = pd.DataFrame(np.random.randint(70000, 80000, NUM_ROWS), columns=['session_id'])
df['item_id'] = long_tailed_item_distribution

# generate category mapping for each item-id
df['category'] = pd.cut(df['item_id'], bins=334, labels=np.arange(1, 335)).astype(np.int32)
df['timestamp/age_days'] = np.random.uniform(0, 1, NUM_ROWS)
df['timestamp/weekday/sin']= np.random.uniform(0, 1, NUM_ROWS)

# generate day mapping for each session 
map_day = dict(zip(df.session_id.unique(), np.random.randint(1, 10, size=(df.session_id.nunique()))))
df['day'] =  df.session_id.map(map_day)
```

<!-- #region id="cd861fcd" -->
- Visualize couple of rows of the synthetic dataset
<!-- #endregion -->

```python id="9617e30c" outputId="f190c609-8e89-40fa-c4df-176ada3d9454"
df.head()
```

<!-- #region id="fae36e04" -->
## Feature Engineering with NVTabular
<!-- #endregion -->

<!-- #region id="139de226" -->
Deep Learning models require dense input features. Categorical features are sparse, and need to be represented by dense embeddings in the model. To allow for that, categorical features need first to be encoded as contiguous integers `(0, ..., |C|)`, where `|C|` is the feature cardinality (number of unique values), so that their embeddings can be efficiently stored in embedding layers.  We will use NVTabular to preprocess the categorical features, so that all categorical columns are encoded as contiguous integers.  Note that in the `Categorify` op we set `start_index=1`, the reason for that we want the encoded null values to start from `1` instead of `0` because we reserve `0` for padding the sequence features.
<!-- #endregion -->

<!-- #region id="55b3bb9c" -->
Here our goal is to create sequential features.  In this cell, we are creating temporal features and grouping them together at the session level, sorting the interactions by time. Note that we also trim each feature sequence in a  session to a certain length. Here, we use the NVTabular library so that we can easily preprocess and create features on GPU with a few lines.
<!-- #endregion -->

```python id="a256f195"
# Categorify categorical features
categ_feats = ['session_id', 'item_id', 'category'] >> nvt.ops.Categorify(start_index=1)

# Define Groupby Workflow
groupby_feats = categ_feats + ['day', 'timestamp/age_days', 'timestamp/weekday/sin']

# Groups interaction features by session and sorted by timestamp
groupby_features = groupby_feats >> nvt.ops.Groupby(
    groupby_cols=["session_id"], 
    aggs={
        "item_id": ["list", "count"],
        "category": ["list"],     
        "day": ["first"],
        "timestamp/age_days": ["list"],
        'timestamp/weekday/sin': ["list"],
        },
    name_sep="-")

# Select and truncate the sequential features
sequence_features_truncated = (groupby_features['category-list', 'item_id-list', 
                                          'timestamp/age_days-list', 'timestamp/weekday/sin-list']) >> \
                            nvt.ops.ListSlice(0,20) >> nvt.ops.Rename(postfix = '_trim')

# Filter out sessions with length 1 (not valid for next-item prediction training and evaluation)
MINIMUM_SESSION_LENGTH = 2
selected_features = groupby_features['item_id-count', 'day-first', 'session_id'] + sequence_features_truncated
filtered_sessions = selected_features >> nvt.ops.Filter(f=lambda df: df["item_id-count"] >= MINIMUM_SESSION_LENGTH)


workflow = nvt.Workflow(filtered_sessions)
dataset = nvt.Dataset(df, cpu=False)
# Generating statistics for the features
workflow.fit(dataset)
# Applying the preprocessing and returning an NVTabular dataset
sessions_ds = workflow.transform(dataset)
# Converting the NVTabular dataset to a Dask cuDF dataframe (`to_ddf()`) and then to cuDF dataframe (`.compute()`)
sessions_gdf = sessions_ds.to_ddf().compute()
```

```python id="4dcbca33" outputId="3d3c969c-76e7-4106-f353-70496ccf6480"
sessions_gdf.head(3)
```

<!-- #region id="2458c28f" -->
It is possible to save the preprocessing workflow. That is useful to apply the same preprocessing to other data (with the same schema) and also to deploy the session-based recommendation pipeline to Triton Inference Server.
<!-- #endregion -->

```python id="ff88e98f"
workflow.save('workflow_etl')
```

<!-- #region id="02a41961" -->
## Export pre-processed data by day
<!-- #endregion -->

<!-- #region id="f9cedca3" -->
In this example we are going to split the preprocessed parquet files by days, to allow for temporal training and evaluation. There will be a folder for each day and three parquet files within each day folder: train.parquet, validation.parquet and test.parquet
<!-- #endregion -->

```python id="12d3e59b"
OUTPUT_FOLDER = os.environ.get("OUTPUT_FOLDER",os.path.join(INPUT_DATA_DIR, "sessions_by_day"))
!mkdir -p $OUTPUT_FOLDER
```

```python id="6c67a92b" outputId="20667850-4128-49bd-96b2-53cbda889b6c"
from transformers4rec.data.preprocessing import save_time_based_splits
save_time_based_splits(data=nvt.Dataset(sessions_gdf),
                       output_dir= OUTPUT_FOLDER,
                       partition_col='day-first',
                       timestamp_col='session_id', 
                      )
```

<!-- #region id="0b72337b" -->
## Checking the preprocessed outputs
<!-- #endregion -->

```python id="dd04ec82"
TRAIN_PATHS = sorted(glob.glob(os.path.join(OUTPUT_FOLDER, "1", "train.parquet")))
```

```python id="8e5e6358" outputId="3316ae01-4d0d-4449-d6e0-5e7c0ef21039"
gdf = cudf.read_parquet(TRAIN_PATHS[0])
gdf.head()
```

<!-- #region id="a6461a96" -->
We have created session-level features to train a session-based recommendation model using NVTabular. Now we will train a session-based recommendation model using [XLNet](https://arxiv.org/abs/1906.08237), one of the state-of-the-art NLP model.
<!-- #endregion -->

<!-- #region id="ab881baf" -->
Next, we will learn:

- Accelerating data loading of parquet files with multiple features on PyTorch using NVTabular library
- Training and evaluating a Transformer-based (XLNET-MLM) session-based recommendation model with multiple features
<!-- #endregion -->

<!-- #region id="3b80af76" -->
Transformers4Rec library relies on a schema object to automatically build all necessary layers to represent, normalize and aggregate input features. As you can see below, `schema.pb` is a protobuf file that contains metadata including statistics about features such as cardinality, min and max values and also tags features based on their characteristics and dtypes (e.g., categorical, continuous, list, integer).
<!-- #endregion -->

<!-- #region id="24ca5c73" -->
## Manually set the schema 
<!-- #endregion -->

```python id="6b6a1f17" outputId="0010ec27-8dff-4194-c154-0d959526acaf"
from merlin_standard_lib import Schema
SCHEMA_PATH = "schema.pb"
schema = Schema().from_proto_text(SCHEMA_PATH)
!cat $SCHEMA_PATH
```

```python id="5fcf0717"
# You can select a subset of features for training
schema = schema.select_by_name(['item_id-list_trim', 
                                'category-list_trim', 
                                'timestamp/weekday/sin-list_trim',
                                'timestamp/age_days-list_trim'])
```

<!-- #region id="18148079" -->
## Define the sequential input module
<!-- #endregion -->

<!-- #region id="985341a3" -->
Below we define our `input` block using the `TabularSequenceFeatures` [class](https://github.com/NVIDIA-Merlin/Transformers4Rec/blob/main/transformers4rec/torch/features/sequence.py#L91). The `from_schema()` method processes the schema and creates the necessary layers to represent features and aggregate them. It keeps only features tagged as `categorical` and `continuous` and supports data aggregation methods like `concat` and `elementwise-sum` techniques. It also support data augmentation techniques like stochastic swap noise. It outputs an interaction representation after combining all features and also the input mask according to the training task (more on this later).

<!-- #endregion -->

<!-- #region id="8d0532bc" -->
The `max_sequence_length` argument defines the maximum sequence length of our sequential input, and if `continuous_projection` argument is set, all numerical features are concatenated and projected by an MLP block so that continuous features are represented by a vector of size defined by user, which is `64` in this example.
<!-- #endregion -->

```python id="0b2d13d0"
inputs = tr.TabularSequenceFeatures.from_schema(
        schema,
        max_sequence_length=20,
        continuous_projection=64,
        d_output=100,
        masking="mlm",
)
```

<!-- #region id="54260cd3" -->
The output of the `TabularSequenceFeatures` module is the sequence of interactions embeddings vectors defined in the following steps:
- 1. Create sequence inputs: If the schema contains non sequential features, expand each feature to a sequence by repeating the value as many as the `max_sequence_length` value.  
- 2. Get a representation vector of categorical features: Project each sequential categorical feature using the related embedding table. The resulting tensor is of shape (bs, max_sequence_length, embed_dim).
- 3. Project scalar values if `continuous_projection` is set : Apply an MLP layer with hidden size equal to `continuous_projection` vector size value. The resulting tensor is of shape (batch_size, max_sequence_length, continuous_projection).
- 4. Aggregate the list of features vectors to represent each interaction in the sequence with one vector: For example, `concat` will concat all vectors based on the last dimension `-1` and the resulting tensor will be of shape (batch_size, max_sequence_length, D) where D is the sum over all embedding dimensions and the value of continuous_projection. 
- 5. If masking schema is set (needed only for the NextItemPredictionTask training), the masked labels are derived from the sequence of raw item-ids and the sequence of interactions embeddings are processed to mask information about the masked positions.
<!-- #endregion -->

<!-- #region id="03933343" -->
## Define the Transformer Block
<!-- #endregion -->

<!-- #region id="3b661085" -->
In the next cell, the whole model is build with a few lines of code. 
Here is a brief explanation of the main classes:  
- [XLNetConfig](https://github.com/NVIDIA-Merlin/Transformers4Rec/blob/main/transformers4rec/config/transformer.py#L261) - We have injected in the HF transformers config classes like `XLNetConfig`the `build()` method, that provides default configuration to Transformer architectures for session-based recommendation. Here we use it to instantiate and configure an XLNET architecture.  
- [TransformerBlock](https://github.com/NVIDIA-Merlin/Transformers4Rec/blob/main/transformers4rec/torch/block/transformer.py#L37) class integrates with HF Transformers, which are made available as a sequence processing module for session-based and sequential-based recommendation models.  
- [NextItemPredictionTask](https://github.com/NVIDIA-Merlin/Transformers4Rec/blob/main/transformers4rec/torch/model/head.py#L238) supports the next-item prediction task. We also support other predictions [tasks](https://github.com/NVIDIA-Merlin/Transformers4Rec/blob/main/transformers4rec/torch/model/head.py), like classification and regression for the whole sequence. 
<!-- #endregion -->

```python id="e0b19550"
# Define XLNetConfig class and set default parameters for HF XLNet config  
transformer_config = tr.XLNetConfig.build(
    d_model=64, n_head=4, n_layer=2, total_seq_length=20
)
# Define the model block including: inputs, masking, projection and transformer block.
body = tr.SequentialBlock(
    inputs, tr.MLPBlock([64]), tr.TransformerBlock(transformer_config, masking=inputs.masking)
)

# Defines the evaluation top-N metrics and the cut-offs
metrics = [NDCGAt(top_ks=[20, 40], labels_onehot=True),  
           RecallAt(top_ks=[20, 40], labels_onehot=True)]

# Define a head related to next item prediction task 
head = tr.Head(
    body,
    tr.NextItemPredictionTask(weight_tying=True, hf_format=True, 
                              metrics=metrics),
    inputs=inputs,
)

# Get the end-to-end Model class 
model = tr.Model(head)
```

<!-- #region id="6ea528f2" -->
Note that we can easily define an RNN-based model inside the `SequentialBlock` instead of a Transformer-based model. You can explore this [tutorial](https://github.com/NVIDIA-Merlin/Transformers4Rec/tree/main/examples/tutorial) for a GRU-based model example.
<!-- #endregion -->

<!-- #region id="ff2ceded" -->
## Train the model 
<!-- #endregion -->

<!-- #region id="f43212e3" -->
We use the NVTabular PyTorch Dataloader for optimized loading of multiple features from input parquet files. You can learn more about this data loader [here](https://nvidia-merlin.github.io/NVTabular/main/training/pytorch.html).
<!-- #endregion -->

<!-- #region id="4c714512" -->
## Set Training arguments
<!-- #endregion -->

```python id="88cfd979"
from transformers4rec.config.trainer import T4RecTrainingArguments
from transformers4rec.torch import Trainer
# Set hyperparameters for training 
train_args = T4RecTrainingArguments(data_loader_engine='nvtabular', 
                                    dataloader_drop_last = True,
                                    report_to = [], 
                                    gradient_accumulation_steps = 1,
                                    per_device_train_batch_size = 256, 
                                    per_device_eval_batch_size = 32,
                                    output_dir = "./tmp", 
                                    learning_rate=0.0005,
                                    lr_scheduler_type='cosine', 
                                    learning_rate_num_cosine_cycles_by_epoch=1.5,
                                    num_train_epochs=5,
                                    max_sequence_length=20, 
                                    no_cuda=False)
```

<!-- #region id="3b20ca71" -->
Note that we add an argument `data_loader_engine='nvtabular'` to automatically load the features needed for training using the schema. The default value is nvtabular for optimized GPU-based data-loading. Optionally a PyarrowDataLoader (pyarrow) can also be used as a basic option, but it is slower and works only for small datasets, as the full data is loaded to CPU memory.
<!-- #endregion -->

<!-- #region id="41277491" -->
## Daily Fine-Tuning: Training over a time window
<!-- #endregion -->

<!-- #region id="061a162b" -->
Here we do daily fine-tuning meaning that we use the first day to train and second day to evaluate, then we use the second day data to train the model by resuming from the first step, and evaluate on the third day, so on so forth.
<!-- #endregion -->

<!-- #region id="612a5034" -->
We have extended the HuggingFace transformers `Trainer` class (PyTorch only) to support evaluation of RecSys metrics. In this example, the evaluation of the session-based recommendation model is performed using traditional Top-N ranking metrics such as Normalized Discounted Cumulative Gain (NDCG@20) and Hit Rate (HR@20). NDCG accounts for rank of the relevant item in the recommendation list and is a more fine-grained metric than HR, which only verifies whether the relevant item is among the top-n items. HR@n is equivalent to Recall@n when there is only one relevant item in the recommendation list.
<!-- #endregion -->

```python id="5880612b"
# Instantiate the T4Rec Trainer, which manages training and evaluation for the PyTorch API
trainer = Trainer(
    model=model,
    args=train_args,
    schema=schema,
    compute_metrics=True,
)
```

<!-- #region id="5e839647" -->
- Define the output folder of the processed parquet files
<!-- #endregion -->

```python id="9ffbc397"
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/workspace/data/sessions_by_day")
```

```python id="1d5e98ae" outputId="cf50855f-b7b5-4e99-d0c8-5fb501f13212"
start_time_window_index = 1
final_time_window_index = 7
#Iterating over days of one week
for time_index in range(start_time_window_index, final_time_window_index):
    # Set data 
    time_index_train = time_index
    time_index_eval = time_index + 1
    train_paths = glob.glob(os.path.join(OUTPUT_DIR, f"{time_index_train}/train.parquet"))
    eval_paths = glob.glob(os.path.join(OUTPUT_DIR, f"{time_index_eval}/valid.parquet"))
    print(train_paths)
    
    # Train on day related to time_index 
    print('*'*20)
    print("Launch training for day %s are:" %time_index)
    print('*'*20 + '\n')
    trainer.train_dataset_or_path = train_paths
    trainer.reset_lr_scheduler()
    trainer.train()
    trainer.state.global_step +=1
    print('finished')
    
    # Evaluate on the following day
    trainer.eval_dataset_or_path = eval_paths
    train_metrics = trainer.evaluate(metric_key_prefix='eval')
    print('*'*20)
    print("Eval results for day %s are:\t" %time_index_eval)
    print('\n' + '*'*20 + '\n')
    for key in sorted(train_metrics.keys()):
        print(" %s = %s" % (key, str(train_metrics[key]))) 
    trainer.wipe_memory()
```

<!-- #region id="8af6146d" -->
## Saves the model
<!-- #endregion -->

```python id="3f03cd91" outputId="e604fd31-ae4a-45c3-d931-d60ebe7e609d"
trainer._save_model_and_checkpoint(save_model_class=True)
```

<!-- #region id="12ba5fc4" -->
## Reloads the model
<!-- #endregion -->

```python id="562ad5a2"
trainer.load_model_trainer_states_from_checkpoint('./tmp/checkpoint-%s'%trainer.state.global_step)
```

<!-- #region id="cbd0c5fd" -->
## Re-compute eval metrics of validation data
<!-- #endregion -->

```python id="dc39a93a"
eval_data_paths = glob.glob(os.path.join(OUTPUT_DIR, f"{time_index_eval}/valid.parquet"))
```

```python id="70b9ba80" outputId="121af87e-91bb-481d-b16d-ef5493b2e9c7"
# set new data from day 7
eval_metrics = trainer.evaluate(eval_dataset=eval_data_paths, metric_key_prefix='eval')
for key in sorted(eval_metrics.keys()):
    print("  %s = %s" % (key, str(eval_metrics[key])))
```

<!-- #region id="d8826ea0" -->
That's it!  
You have just trained your session-based recommendation model using Transformers4Rec.
<!-- #endregion -->

<!-- #region id="0c1e2394" -->
Tip: We can easily log and visualize model training and evaluation on [Weights & Biases (W&B)](https://wandb.ai/home), [Tensorboard](https://www.tensorflow.org/tensorboard) and [NVIDIA DLLogger](https://github.com/NVIDIA/dllogger). By default, the HuggingFace transformers `Trainer` (which we extend) uses Weights & Biases (W&B) to log training and evaluation metrics, which provides nice results visualization and comparison between different runs.
<!-- #endregion -->
