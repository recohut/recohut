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

<!-- #region tags=[] id="76421714" -->
# Transformers4Rec Session-based Recommender on Yoochoose
<!-- #endregion -->

<!-- #region id="916719d3" -->
In recent years, several deep learning-based algorithms have been proposed for recommendation systems while its adoption in industry deployments have been steeply growing. In particular, NLP inspired approaches have been successfully adapted for sequential and session-based recommendation problems, which are important for many domains like e-commerce, news and streaming media. Session-Based Recommender Systems (SBRS) have been proposed to model the sequence of interactions within the current user session, where a session is a short sequence of user interactions typically bounded by user inactivity. They have recently gained popularity due to their ability to capture short-term or contextual user preferences towards items. 

The field of NLP has evolved significantly within the last decade, particularly due to the increased usage of deep learning. As a result, state of the art NLP approaches have inspired RecSys practitioners and researchers to adapt those architectures, especially for sequential and session-based recommendation problems. Here, we leverage one of the state-of-the-art Transformer-based architecture, [XLNet](https://arxiv.org/abs/1906.08237) with Masked Language Modeling (MLM) training technique (see our [tutorial](https://github.com/NVIDIA-Merlin/Transformers4Rec/tree/main/examples/tutorial) for details) for training a session-based model.

In this end-to-end-session-based recommnender model example, we use `Transformers4Rec` library, which leverages the popular [HuggingFace’s Transformers](https://github.com/huggingface/transformers) NLP library and make it possible to experiment with cutting-edge implementation of such architectures for sequential and session-based recommendation problems. For detailed explanations of the building blocks of Transformers4Rec meta-architecture visit [getting-started-session-based](https://github.com/NVIDIA-Merlin/Transformers4Rec/tree/main/examples/getting-started-session-based) and [tutorial](https://github.com/NVIDIA-Merlin/Transformers4Rec/tree/main/examples/tutorial) example notebooks.
<!-- #endregion -->

<!-- #region id="b24dd14d" -->
## 1. Setup
<!-- #endregion -->

<!-- #region id="1360ccb2" -->
### 1.1. Import Libraries and Define Data Input and Output Paths
<!-- #endregion -->

```python id="790dfa63"
import os
import glob
import numpy as np
import gc

import cudf
import cupy
import nvtabular as nvt
```

```python id="880e5250"
DATA_FOLDER = "/workspace/data/"
FILENAME_PATTERN = 'yoochoose-clicks.dat'
DATA_PATH = os.path.join(DATA_FOLDER, FILENAME_PATTERN)

OUTPUT_FOLDER = "./yoochoose_transformed"
OVERWRITE = False
```

<!-- #region id="d07dbd85" -->
### 1.2. Download the data
<!-- #endregion -->

<!-- #region id="9a55f498" -->
In this notebook we are using the `YOOCHOOSE` dataset which contains a collection of sessions from a retailer. Each session  encapsulates the click events that the user performed in that session.

The dataset is available on [Kaggle](https://www.kaggle.com/chadgostopp/recsys-challenge-2015). You need to download it and copy to the `DATA_FOLDER` path. Note that we are only using the `yoochoose-clicks.dat` file.

<!-- #endregion -->

<!-- #region id="060f2781" -->
### 1.3. Load and clean raw data
<!-- #endregion -->

```python id="9741bf12"
interactions_df = cudf.read_csv(DATA_PATH, sep=',', 
                                names=['session_id','timestamp', 'item_id', 'category'], 
                                dtype=['int', 'datetime64[s]', 'int', 'int'])
```

<!-- #region id="2f35855b" -->
#### Remove repeated interactions within the same session
<!-- #endregion -->

```python tags=[] id="2e476761" outputId="846a1e97-9755-4040-ef28-9dbfd96464f6"
print("Count with in-session repeated interactions: {}".format(len(interactions_df)))
# Sorts the dataframe by session and timestamp, to remove consecutive repetitions
interactions_df.timestamp = interactions_df.timestamp.astype(int)
interactions_df = interactions_df.sort_values(['session_id', 'timestamp'])
past_ids = interactions_df['item_id'].shift(1).fillna()
session_past_ids = interactions_df['session_id'].shift(1).fillna()
# Keeping only no consectutive repeated in session interactions
interactions_df = interactions_df[~((interactions_df['session_id'] == session_past_ids) & (interactions_df['item_id'] == past_ids))]
print("Count after removed in-session repeated interactions: {}".format(len(interactions_df)))
```

<!-- #region id="0328cd64" -->
#### Creates new feature with the timestamp when the item was first seen
<!-- #endregion -->

```python tags=[] id="a0848f57" outputId="208aa81f-72b2-4d9d-cff0-6dc6ac10ce14"
items_first_ts_df = interactions_df.groupby('item_id').agg({'timestamp': 'min'}).reset_index().rename(columns={'timestamp': 'itemid_ts_first'})
interactions_merged_df = interactions_df.merge(items_first_ts_df, on=['item_id'], how='left')
interactions_merged_df.head()
```

```python tags=[] id="b116d046" outputId="5d572048-09bf-4de5-a153-449bf2ba1e6d"
# free gpu memory
del interactions_df, session_past_ids, items_first_ts_df
gc.collect()
```

<!-- #region id="abbd5ef7" -->
## 2. Define a preprocessing workflow with NVTabular
<!-- #endregion -->

<!-- #region id="a3d83c8d" -->
NVTabular is a feature engineering and preprocessing library for tabular data designed to quickly and easily manipulate terabyte scale datasets used to train deep learning based recommender systems. It provides a high level abstraction to simplify code and accelerates computation on the GPU using the RAPIDS cuDF library.

NVTabular supports different feature engineering transformations required by deep learning (DL) models such as Categorical encoding and numerical feature normalization. It also supports feature engineering and generating sequential features. 

More information about the supported features can be found <a href=https://nvidia.github.io/NVTabular/main/index.html> here. </a>
<!-- #endregion -->

<!-- #region id="c21f3dac" -->
### 2.1 Feature engineering: Create and Transform items features
<!-- #endregion -->

<!-- #region id="47ece553" -->
In this cell, we are defining three transformations ops: 

- 1. Encoding categorical variables using `Categorify()` op. We set `start_index` to 1, so that encoded null values start from `1` instead of `0` because we reserve `0` for padding the sequence features.
- 2. Deriving temporal features from timestamp and computing their cyclical representation using a custom lambda function. 
- 3. Computing the item recency in days using a custom Op. Note that item recency is defined as the difference between the first occurence of the item in dataset and the actual date of item interaction. 

For more ETL workflow examples, visit NVTabular [example notebooks](https://github.com/NVIDIA/NVTabular/tree/main/examples).
<!-- #endregion -->

```python tags=[] id="2c1620e3"
# Encodes categorical features as contiguous integers
cat_feats = nvt.ColumnSelector(['session_id', 'category', 'item_id']) >> nvt.ops.Categorify(start_index=1)

# create time features
session_ts = nvt.ColumnSelector(['timestamp'])
session_time = (
    session_ts >> 
    nvt.ops.LambdaOp(lambda col: cudf.to_datetime(col, unit='s')) >> 
    nvt.ops.Rename(name = 'event_time_dt')
)
sessiontime_weekday = (
    session_time >> 
    nvt.ops.LambdaOp(lambda col: col.dt.weekday) >> 
    nvt.ops.Rename(name ='et_dayofweek')
)

# Derive cyclical features: Defines a custom lambda function 
def get_cycled_feature_value_sin(col, max_value):
    value_scaled = (col + 0.000001) / max_value
    value_sin = np.sin(2*np.pi*value_scaled)
    return value_sin

weekday_sin = sessiontime_weekday >> (lambda col: get_cycled_feature_value_sin(col+1, 7)) >> nvt.ops.Rename(name = 'et_dayofweek_sin')

# Compute Item recency: Define a custom Op 
class ItemRecency(nvt.ops.Operator):
    def transform(self, columns, gdf):
        for column in columns.names:
            col = gdf[column]
            item_first_timestamp = gdf['itemid_ts_first']
            delta_days = (col - item_first_timestamp) / (60*60*24)
            gdf[column + "_age_days"] = delta_days * (delta_days >=0)
        return gdf
           
    def output_column_names(self, columns):
        return nvt.ColumnSelector([column + "_age_days" for column in columns.names])

    def dependencies(self):
        return ["itemid_ts_first"]
    
recency_features = session_ts >> ItemRecency() 
# Apply standardization to this continuous feature
recency_features_norm = recency_features >> nvt.ops.LogOp() >> nvt.ops.Normalize() >> nvt.ops.Rename(name='product_recency_days_log_norm')

time_features = (
    session_time +
    sessiontime_weekday +
    weekday_sin + 
    recency_features_norm
)

features = nvt.ColumnSelector(['timestamp', 'session_id']) + cat_feats + time_features 
```

<!-- #region id="343ce466" -->
### 2.2 Defines the preprocessing of sequential features
<!-- #endregion -->

<!-- #region id="9b397a1c" -->
Once the item features are generated, the objective of this cell is grouping interactions at the session level, sorting the interactions by time. We additionally truncate all sessions to first 20 interactions and filter out sessions with less than 2 interactions.
<!-- #endregion -->

```python id="c15cae90"
# Define Groupby Operator
groupby_features = features >> nvt.ops.Groupby(
    groupby_cols=["session_id"], 
    sort_cols=["timestamp"],
    aggs={
        'item_id': ["list", "count"],
        'category': ["list"],  
        'timestamp': ["first"],
        'event_time_dt': ["first"],
        'et_dayofweek_sin': ["list"],
        'product_recency_days_log_norm': ["list"]
        },
    name_sep="-")


# Truncate sequence features to first interacted 20 items 
SESSIONS_MAX_LENGTH = 20 

groupby_features_list = groupby_features['item_id-list', 'category-list', 'et_dayofweek_sin-list', 'product_recency_days_log_norm-list']
groupby_features_truncated = groupby_features_list >> nvt.ops.ListSlice(0, SESSIONS_MAX_LENGTH) >> nvt.ops.Rename(postfix = '_seq')

# Calculate session day index based on 'event_time_dt-first' column
day_index = ((groupby_features['event_time_dt-first'])  >> 
    nvt.ops.LambdaOp(lambda col: (col - col.min()).dt.days +1) >> 
    nvt.ops.Rename(f = lambda col: "day_index")
)

# Select features for training 
selected_features = groupby_features['session_id', 'item_id-count'] + groupby_features_truncated + day_index

# Filter out sessions with less than 2 interactions 
MINIMUM_SESSION_LENGTH = 2
filtered_sessions = selected_features >> nvt.ops.Filter(f=lambda df: df["item_id-count"] >= MINIMUM_SESSION_LENGTH) 
```

<!-- #region id="9c902ae1" -->
- Avoid Numba low occupancy warnings
<!-- #endregion -->

```python id="82c0fb47"
from numba import config
config.CUDA_LOW_OCCUPANCY_WARNINGS = 0
```

<!-- #region id="872492c4" -->
### 2.3 Execute NVTabular workflow
<!-- #endregion -->

<!-- #region id="2c5390d0" -->
Once we have defined the general workflow (`filtered_sessions`), we provide our cudf dataset to nvt.Dataset class which is optimized to split data into chunks that can fit in device memory and to handle the calculation of complex global statistics. Then, we execute the pipeline that fits and transforms data to get the desired output features.
<!-- #endregion -->

```python tags=[] id="185f2759"
dataset = nvt.Dataset(interactions_merged_df)
workflow = nvt.Workflow(filtered_sessions)
# Learns features statistics necessary of the preprocessing workflow
workflow.fit(dataset)
# Apply the preprocessing workflow in the dataset and converts the resulting Dask cudf dataframe to a cudf dataframe
sessions_gdf = workflow.transform(dataset).compute()
```

<!-- #region id="88627f9d" -->
Let's print the head of our preprocessed dataset. You can notice that now each example (row) is a session and the sequential features with respect to user interactions were converted to lists with matching length.
<!-- #endregion -->

```python tags=[] id="acd67e9f" outputId="76f3f081-f868-40a8-88bf-1df141bef8dd"
sessions_gdf.head()
```

<!-- #region id="3a08cb37" -->
#### Saves the preprocesing workflow
<!-- #endregion -->

```python id="aec9d6c3"
workflow.save('workflow_etl')
```

<!-- #region id="9185bb30" -->
### 2.4 Export pre-processed data by day
<!-- #endregion -->

<!-- #region id="7aef05ce" -->
In this example we are going to split the preprocessed parquet files by days, to allow for temporal training and evaluation. There will be a folder for each day and three parquet files within each day: `train.parquet`, `validation.parquet` and `test.parquet`
  
P.s. It is worthwhile a note that the dataset have a single categorical feature (category), but it is inconsistent over time in the dataset. All interactions before day 84 (2014-06-23) have the same value for that feature, whereas many other categories are introduced afterwards. Thus for the demo we save only the last five days.
<!-- #endregion -->

```python id="04176356"
sessions_gdf = sessions_gdf[sessions_gdf.day_index>=178]
```

```python tags=[] id="07067c7e" outputId="3ee75254-68ca-4a92-ec4a-4256ac260354"
from transformers4rec.data.preprocessing import save_time_based_splits
save_time_based_splits(data=nvt.Dataset(sessions_gdf),
                       output_dir= "./preproc_sessions_by_day",
                       partition_col='day_index',
                       timestamp_col='session_id', 
                      )
```

```python tags=[] id="e058a369" outputId="6ec4fc1e-aaaf-40f9-c384-77a2fb4005a9"
from transformers4rec.torch.utils.examples_utils import list_files
list_files('./preproc_sessions_by_day')
```

```python tags=[] id="55b58f39" outputId="42e861fd-82c4-4adf-99dc-0dafe5f5989f"
# free gpu memory
del  sessions_gdf
gc.collect()
```

<!-- #region id="015d14a5" -->
## 3. Model definition using Transformers4Rec
<!-- #endregion -->

<!-- #region id="9c1b9492" -->
### 3.1 Get the schema 
<!-- #endregion -->

<!-- #region id="6e2aa3f0" -->
The library uses a schema format to configure the input features and automatically creates the necessary layers. This *protobuf* text file contains the description of each input feature by defining: the name, the type, the number of elements of a list column,  the cardinality of a categorical feature and the min and max values of each feature. In addition, the annotation field contains the tags such as specifying `continuous` and `categorical` features, the `target` column or the `item_id` feature, among others.
<!-- #endregion -->

```python tags=[] id="0b98e462" outputId="cb8103d1-73f6-49ed-aed5-2d499485e364"
from merlin_standard_lib import Schema
SCHEMA_PATH = "schema_demo.pb"
schema = Schema().from_proto_text(SCHEMA_PATH)
!cat $SCHEMA_PATH
```

<!-- #region id="7c88641c" -->
We can select the subset of features we want to use for training the model by their tags or their names.
<!-- #endregion -->

```python id="32c0ede1"
schema = schema.select_by_name(
   ['item_id-list_seq', 'category-list_seq', 'product_recency_days_log_norm-list_seq', 'et_dayofweek_sin-list_seq']
)
```

<!-- #region id="0f668c2d" -->
### 3.2 Define the end-to-end Session-based Transformer-based recommendation model
<!-- #endregion -->

<!-- #region tags=[] id="42aa9ee9" -->
For session-based recommendation model definition, the end-to-end model definition requires four steps:

1. Instantiate [TabularSequenceFeatures](https://nvidia-merlin.github.io/Transformers4Rec/main/api/transformers4rec.tf.features.html?highlight=tabularsequence#transformers4rec.tf.features.sequence.TabularSequenceFeatures) input-module from schema to prepare the embedding tables of categorical variables and project continuous features, if specified. In addition, the module provides different aggregation methods (e.g. 'concat', 'elementwise-sum') to merge input features and generate the sequence of interactions embeddings. The module also supports language modeling tasks to prepare masked labels for training and evaluation (e.g: 'mlm' for masked language modeling) 

2. Next, we need to define one or multiple prediction tasks. For this demo, we are going to use [NextItemPredictionTask](https://nvidia-merlin.github.io/Transformers4Rec/main/api/transformers4rec.tf.model.html?highlight=nextitem#transformers4rec.tf.model.prediction_task.NextItemPredictionTask) with `Masked Language modeling`: during training randomly selected items are masked and predicted using the unmasked sequence items. For inference it is meant to always predict the next item to be interacted with.

3. Then we construct a `transformer_config` based on the architectures provided by [Hugging Face Transformers](https://github.com/huggingface/transformers) framework. </a>

4. Finally we link the transformer-body to the inputs and the prediction tasks to get the final pytorch `Model` class.
    
For more details about the features supported by each sub-module, please check out the library [documentation](https://nvidia-merlin.github.io/Transformers4Rec/main/index.html) page.
<!-- #endregion -->

```python tags=[] id="5f48ba32" outputId="836a3a04-566b-43bd-ddf5-ceaf6f1e0ca7"
from transformers4rec import torch as tr

max_sequence_length, d_model = 20, 320
# Define input module to process tabular input-features and to prepare masked inputs
input_module = tr.TabularSequenceFeatures.from_schema(
    schema,
    max_sequence_length=max_sequence_length,
    continuous_projection=64,
    aggregation="concat",
    d_output=d_model,
    masking="mlm",
)

# Define Next item prediction-task 
prediction_task = tr.NextItemPredictionTask(hf_format=True, weight_tying=True)

# Define the config of the XLNet Transformer architecture
transformer_config = tr.XLNetConfig.build(
    d_model=d_model, n_head=8, n_layer=2, total_seq_length=max_sequence_length
)

#Get the end-to-end model 
model = transformer_config.to_torch_model(input_module, prediction_task)
```

```python tags=[] id="e90f69c6" outputId="05da13ec-2d43-43f8-ad42-7ccae82f829f"
model
```

<!-- #region id="cb3e5a39" -->
### 3.3. Daily Fine-Tuning: Training over a time window¶
<!-- #endregion -->

<!-- #region id="cb1e05a6" -->
Now that the model is defined, we are going to launch training. For that, Transfromers4rec extends HF Transformers Trainer class to adapt the evaluation loop for session-based recommendation task and the calculation of ranking metrics. The original `train()` method is not modified meaning that we leverage the efficient training implementation from that library, which manages for example half-precision (FP16) training.
<!-- #endregion -->

<!-- #region id="1e55dbd4" -->
#### Sets Training arguments
<!-- #endregion -->

<!-- #region id="172b6d9a" -->
An additional argument `data_loader_engine` is defined to automatically load the features needed for training using the schema. The default value is `nvtabular` for optimized GPU-based data-loading.  Optionally a `PyarrowDataLoader` (`pyarrow`) can also be used as a basic option, but it is slower and works only for small datasets, as the full data is loaded to CPU memory.
<!-- #endregion -->

```python id="23bfa8d9"
training_args = tr.trainer.T4RecTrainingArguments(
            output_dir="./tmp",
            max_sequence_length=20,
            data_loader_engine='nvtabular',
            num_train_epochs=10, 
            dataloader_drop_last=False,
            per_device_train_batch_size = 384,
            per_device_eval_batch_size = 512,
            learning_rate=0.0005,
            fp16=True,
            report_to = [],
            logging_steps=200
        )
```

<!-- #region id="845e8cac" -->
#### Instantiate the trainer
<!-- #endregion -->

```python tags=[] id="4d8ba34d" outputId="e5cf6c24-86da-4095-f62c-70660c65f0ba"
recsys_trainer = tr.Trainer(
    model=model,
    args=training_args,
    schema=schema,
    compute_metrics=True)
```

<!-- #region id="2189f1a3" -->
#### Launches daily Training and Evaluation
<!-- #endregion -->

<!-- #region id="06c017e1" -->
In this demo, we will use the `fit_and_evaluate` method that allows us to conduct a time-based finetuning by iteratively training and evaluating using a sliding time window: At each iteration, we use training data of a specific time index $t$ to train the model then we evaluate on the validation data of next index $t + 1$. Particularly, the start time is set to 178 and end time to 180.
<!-- #endregion -->

```python tags=[] id="ffb7421b" outputId="a635a465-289b-47a8-d22a-1e8934c10002"
from transformers4rec.torch.utils.examples_utils import fit_and_evaluate
aot_results = fit_and_evaluate(recsys_trainer, start_time_index=178, end_time_index=178, input_dir='./preproc_sessions_by_day')
```

<!-- #region tags=[] id="8b555d34" -->
#### Visualize the average over time metrics
<!-- #endregion -->

```python tags=[] id="a3759c15" outputId="934eb01d-97bd-44f8-c2e7-959ee2c5a588"
mean_results = {k: np.mean(v) for k,v in aot_results.items()}
for key in sorted(mean_results.keys()): 
    print(" %s = %s" % (key, str(mean_results[key]))) 
```

<!-- #region id="61de28fc" -->
#### Saves the model
<!-- #endregion -->

```python tags=[] id="aca62d57" outputId="fd045bb5-94c8-4679-ca0e-f90c2edb66df"
recsys_trainer._save_model_and_checkpoint(save_model_class=True)
```

<!-- #region id="867de4a0" -->
#### Exports the preprocessing workflow and model in the format required by Triton server:** 

NVTabular’s `export_pytorch_ensemble()` function enables us to create model files and config files to be served to Triton Inference Server. 
<!-- #endregion -->

```python id="900d855f"
from nvtabular.inference.triton import export_pytorch_ensemble
export_pytorch_ensemble(
    model,
    workflow,
    sparse_max=recsys_trainer.get_train_dataloader().dataset.sparse_max,
    name= "t4r_pytorch",
    model_path= "/workspace/TF4Rec/models/",
    label_columns =[],
)
```

<!-- #region id="abcbf3c2" -->
## 4. Serving Ensemble Model to the Triton Inference Server
<!-- #endregion -->

<!-- #region id="d84718d0" -->
NVIDIA [Triton Inference Server (TIS)](https://github.com/triton-inference-server/server) simplifies the deployment of AI models at scale in production. TIS provides a cloud and edge inferencing solution optimized for both CPUs and GPUs. It supports a number of different machine learning frameworks such as TensorFlow and PyTorch.

The last step of machine learning (ML)/deep learning (DL) pipeline is to deploy the ETL workflow and saved model to production. In the production setting, we want to transform the input data as done during training (ETL). We need to apply the same mean/std for continuous features and use the same categorical mapping to convert the categories to continuous integer before we use the DL model for a prediction. Therefore, we deploy the NVTabular workflow with the PyTorch model as an ensemble model to Triton Inference. The ensemble model guarantees that the same transformation is applied to the raw inputs.


In this section, you will learn how to
- to deploy saved NVTabular and PyTorch models to Triton Inference Server 
- send requests for predictions and get responses.
<!-- #endregion -->

<!-- #region id="89ae4876" -->
### 4.1. Pull and Start Inference Container

At this point, before connecing to the Triton Server, we launch the inference docker container and then load the ensemble `t4r_pytorch` to the inference server. This is done with the scripts below:

**Launch the docker container**
```
docker run -it --gpus device=0 -p 8000:8000 -p 8001:8001 -p 8002:8002 -v <path_to_saved_models>:/workspace/models/ nvcr.io/nvidia/merlin/merlin-inference:21.09
```
This script will mount your local model-repository folder that includes your saved models from the previous cell to `/workspace/models` directory in the merlin-inference docker container.

**Start triton server**<br>
After you started the merlin-inference container, you can start triton server with the command below. You need to provide correct path of the models folder.


```
tritonserver --model-repository=<path_to_models> --model-control-mode=explicit
```
Note: The model-repository path for our example is `/workspace/models`. The models haven't been loaded, yet. Below, we will request the Triton server to load the saved ensemble model below.
<!-- #endregion -->

<!-- #region id="9fbd5275" -->
### Connect to the Triton Inference Server and check if the server is alive
<!-- #endregion -->

```python tags=[] id="38494a8c" outputId="00ab1c3a-7d44-4e0a-db88-ba83ade37708"
import tritonhttpclient
try:
    triton_client = tritonhttpclient.InferenceServerClient(url="localhost:8000", verbose=True)
    print("client created.")
except Exception as e:
    print("channel creation failed: " + str(e))
triton_client.is_server_live()
```

<!-- #region id="df34e3e3" -->
### Load raw data for inference
We select the last 50 interactions and filter out sessions with less than 2 interactions. 
<!-- #endregion -->

```python id="273df114"
interactions_merged_df=interactions_merged_df.sort_values('timestamp')
batch = interactions_merged_df[-50:]
sessions_to_use = batch.session_id.value_counts()
filtered_batch = batch[batch.session_id.isin(sessions_to_use[sessions_to_use.values>1].index.values)]
```

<!-- #region id="7251a7cf" -->
### Send the request to triton server
<!-- #endregion -->

```python id="1a89d90e" outputId="22015800-427b-40e1-cee8-79ce10e96e79"
triton_client.get_model_repository_index()
```

<!-- #region id="b779a53b" -->
### Load the ensemble model to triton
If all models are loaded succesfully, you should be seeing `successfully loaded` status next to each model name on your terminal.
<!-- #endregion -->

```python id="d3480e0b" outputId="f0d96461-f9c2-49c4-b81d-9d2e14da2593"
triton_client.load_model(model_name="t4r_pytorch")
```

```python tags=[] id="b3c711da" outputId="20c57716-00ef-485f-e3b2-12fe00bbadb1"
import nvtabular.inference.triton as nvt_triton
import tritonclient.grpc as grpcclient

inputs = nvt_triton.convert_df_to_triton_input(filtered_batch.columns, filtered_batch, grpcclient.InferInput)

output_names = ["output"]

outputs = []
for col in output_names:
    outputs.append(grpcclient.InferRequestedOutput(col))
    
MODEL_NAME_NVT = "t4r_pytorch"

with grpcclient.InferenceServerClient("localhost:8001") as client:
    response = client.infer(MODEL_NAME_NVT, inputs)
    print(col, ':\n', response.as_numpy(col))
```

<!-- #region id="9ae5632d" -->
- Visualise top-k predictions
<!-- #endregion -->

```python tags=[] id="02c0b5e7" outputId="1b84c4fa-b9e8-4261-f68a-44653f2014fd"
from transformers4rec.torch.utils.examples_utils import visualize_response
visualize_response(filtered_batch, response, top_k=5, session_col='session_id')
```

<!-- #region id="844c949d" -->
As you noticed, we first got prediction results (logits) from the trained model head, and then by using a handy util function `visualize_response` we extracted top-k encoded item-ids from logits. Basically, we generated recommended items for a given session.

This is the end of the tutorial. You successfully

- performed feature engineering with NVTabular
- trained transformer architecture based session-based recommendation models with Transformers4Rec
- deployed a trained model to Triton Inference Server, sent request and got responses from the server.
<!-- #endregion -->

<!-- #region id="5f6274a5" -->
**Unload models**
<!-- #endregion -->

```python id="57718050"
triton_client.unload_model(model_name="t4r_pytorch")
triton_client.unload_model(model_name="t4r_pytorch_nvt")
triton_client.unload_model(model_name="t4r_pytorch_pt")
```

<!-- #region id="042f81e3" -->
## References
<!-- #endregion -->

<!-- #region id="e983a739" -->
- Merlin Transformers4rec: https://github.com/NVIDIA-Merlin/Transformers4Rec

- Merlin NVTabular: https://github.com/NVIDIA/NVTabular/tree/main/nvtabular

- Triton inference server: https://github.com/triton-inference-server
<!-- #endregion -->
