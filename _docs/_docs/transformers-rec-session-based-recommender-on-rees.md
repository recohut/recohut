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

<!-- #region id="wOOL7LCXEWQz" -->
# Transformers4Rec session-based recommender on REES46
<!-- #endregion -->

<!-- #region id="Z3Ea3xXkEgCX" -->
Session-based recommendation, a sub-area of sequential recommendation, has been an important task in online services like e-commerce and news portals. Session-based recommenders provide relevant and personalized recommendations even when prior user history is not available or their tastes change over time. They recently gained popularity due to their ability to capture short-term or contextual user preferences towards items.
<!-- #endregion -->

<!-- #region id="Qx_02_Z3EgTU" -->
## Learning Objectives
In this tutorial, we will learn:
- the main concepts and algorithms for session-based recommendation
- implementation of preprocessing and feature engineering techniques for session-based recommendation model on GPU with NVTabular
- how to build, train and evaluate a session-based recommendation model based on RNN and Transformer architectures with Transformers4Rec library
- how to deploy a trained model to the Triton Inference Server
- Preprocessing with cuDF and NVTabular
- Feature engineering with NVTabular
- Introduction to Transformers4Rec
- Introduction to session-based recommendation
- Accelerated dataloaders for PyTorch
- Traning and evaluating an RNN-based session based recommendation model for next item prediction task
- Traning and evaluating Transformer architecture based session-based recommendation model next item prediction task
- Using side information (additional features) to improve the accuracy of a model
- Deploying to inference with Triton
<!-- #endregion -->

<!-- #region id="WyqP2RTYE-bf" -->
## Import the required libraries
<!-- #endregion -->

```python id="NVYfsPmqE-bg"
import os
import numpy as np 
import gc
import shutil
import glob

import cudf
import cupy as cp
import nvtabular as nvt
from nvtabular import ColumnSelector

import transformers4rec.torch as tr
from transformers4rec.torch.ranking_metric import NDCGAt, RecallAt
```

<!-- #region id="QBgPdlr1E-bX" -->
## Read and Process E-Commerce data
<!-- #endregion -->

<!-- #region id="jIHSWbBjE-bb" -->
In this section, we are going to use a subset of a publicly available [eCommerce dataset](https://www.kaggle.com/mkechinov/ecommerce-behavior-data-from-multi-category-store). The full dataset contains 7 months data (from October 2019 to April 2020) from a large multi-category online store. Each row in the file represents an event. All events are related to products and users. Each event is like many-to-many relation between products and users.
Data collected by Open CDP project and the source of the dataset is [REES46 Marketing Platform](https://rees46.com/).
<!-- #endregion -->

<!-- #region id="LPHsMoXxE-bd" -->
We use only `2019-Oct.csv` file for training our models, so you can visit this site and download the csv file: https://www.kaggle.com/mkechinov/ecommerce-behavior-data-from-multi-category-store.
<!-- #endregion -->

<!-- #region id="AojjHH9NE-bi" -->
## Read Data via cuDF from CSV
<!-- #endregion -->

<!-- #region id="P_uW4DuYE-bk" -->
At this point we expect that you have already downloaded the `2019-Oct.csv` dataset and stored it in the `INPUT_DATA_DIR` as defined below. It is worth mentioning that the raw dataset is ~ 6 GB, therefore a single GPU with 16 GB or less memory might run out of memory.
<!-- #endregion -->

```python id="DB3xflW4E-bm"
# define some information about where to get our data
INPUT_DATA_DIR = os.environ.get("INPUT_DATA_DIR", "/workspace/data/")
```

```python id="uYHVpXf2E-bo" outputId="6ba40ec4-72d0-4019-b86e-3f0be4e8bcd5"
%%time
raw_df = cudf.read_csv(os.path.join(INPUT_DATA_DIR, '2019-Oct.csv')) 
raw_df.head()
```

```python id="Lz998es4E-br" outputId="2f78ee1e-b7d2-4712-c7b0-d922fc00241e"
raw_df.shape
```

<!-- #region id="6Si2x7muE-bs" -->
## Convert timestamp from datetime
<!-- #endregion -->

```python id="Rx30B89IE-bt" outputId="1bdab8cc-24ae-494c-c005-6d54b5d221a2"
raw_df['event_time_dt'] = raw_df['event_time'].astype('datetime64[s]')
raw_df['event_time_ts']= raw_df['event_time_dt'].astype('int')
raw_df.head()
```

```python id="PThaWmIiE-bv" outputId="afaa993d-e789-4cfc-8593-d78267160131"
# check out the columns with nulls
raw_df.isnull().any()
```

```python id="z2yjt_nCE-bx" outputId="0130b474-adba-401e-ab88-0eb16ba5008c"
# Remove rows where `user_session` is null.
raw_df = raw_df[raw_df['user_session'].isnull()==False]
len(raw_df)
```

<!-- #region id="--qT8qvmE-by" -->
We no longer need `event_time` column.
<!-- #endregion -->

```python id="qaUO-tRUE-bz"
raw_df = raw_df.drop(['event_time'],  axis=1)
```

<!-- #region id="qDsU-44pE-bz" -->
## Categorify `user_session` column
Although `user_session` is not used as an input feature for the model, it is useful to convert those raw long string to int values to avoid potential failures when grouping interactions by `user_session` in the next section.
<!-- #endregion -->

```python id="u-5hxU0XE-b0" outputId="ad46a69b-b288-40e3-cbf6-a726f4254795"
cols = list(raw_df.columns)
cols.remove('user_session')
cols
```

```python id="qX_yH9nvE-b1"
# load data 
df_event = nvt.Dataset(raw_df) 

# categorify user_session 
cat_feats = ['user_session'] >> nvt.ops.Categorify()

workflow = nvt.Workflow(cols + cat_feats)
workflow.fit(df_event)
df = workflow.transform(df_event).to_ddf().compute()
```

```python id="d582q3NXE-b2" outputId="7468d8e9-acfa-43bd-ac3a-0b44a4385b7d"
df.head()
```

```python id="mkvU4w_mE-b4"
raw_df = None
del(raw_df)
```

```python id="tjTswg3QE-b5" outputId="0cc8bf62-247a-47c9-fd10-98bfbf15dd81"
gc.collect()
```

<!-- #region id="1Ufoo58oE-b7" -->
## Removing consecutive repeated (user, item) interactions
<!-- #endregion -->

<!-- #region id="C159Al0wE-b7" -->
We keep repeated interactions on the same items, removing only consecutive interactions, because it might be due to browser tab refreshes or different interaction types (e.g. click, add-to-card, purchase)
<!-- #endregion -->

```python id="KvJPNGpbE-b8" outputId="040ca1bb-82ac-4d94-d047-5ce9abe0473f"
%%time
df = df.sort_values(['user_session', 'event_time_ts']).reset_index(drop=True)

print("Count with in-session repeated interactions: {}".format(len(df)))
# Sorts the dataframe by session and timestamp, to remove consecutive repetitions
df['product_id_past'] = df['product_id'].shift(1).fillna(0)
df['session_id_past'] = df['user_session'].shift(1).fillna(0)
#Keeping only no consecutive repeated in session interactions
df = df[~((df['user_session'] == df['session_id_past']) & \
             (df['product_id'] == df['product_id_past']))]
print("Count after removed in-session repeated interactions: {}".format(len(df)))
del(df['product_id_past'])
del(df['session_id_past'])

gc.collect()
```

<!-- #region id="Yho5LkXrE-b9" -->
## Include the item first time seen feature (for recency calculation)
<!-- #endregion -->

<!-- #region id="HN0elKgCE-b-" -->
We create `prod_first_event_time_ts` column which indicates the timestamp that an item was seen first time.
<!-- #endregion -->

```python id="Rho7eH5dE-b_" outputId="cedd5778-c4a7-46f7-c65c-1424fc28215f"
item_first_interaction_df = df.groupby('product_id').agg({'event_time_ts': 'min'}) \
            .reset_index().rename(columns={'event_time_ts': 'prod_first_event_time_ts'})
item_first_interaction_df.head()
gc.collect()
```

```python id="xBfuAk3vE-b_"
df = df.merge(item_first_interaction_df, on=['product_id'], how='left').reset_index(drop=True)
```

```python id="KGRPz0pqE-cA" outputId="cb01ffc8-ebdd-47c7-a2f7-2a33c1d3beda"
df.head()
```

```python id="tzLFbQG1E-cB" outputId="24cde00a-fabf-42eb-e0a0-b91e80b862f0"
del(item_first_interaction_df)
item_first_interaction_df=None
gc.collect()
```

<!-- #region id="WiE2JuncE-cC" -->
In this tutorial, we only use one week of data from Oct 2019 dataset.
<!-- #endregion -->

```python id="SoV0lsUTE-cC" outputId="95fc4898-cf68-4ed3-87cf-45cb5e09eec6"
# check the min date
df['event_time_dt'].min()
```

```python id="1PdlgOiUE-cD"
# Filters only the first week of the data.
df = df[df['event_time_dt'] < np.datetime64('2019-10-08')].reset_index(drop=True)
```

<!-- #region id="9ePNRj5fE-cD" -->
We verify that we only have the first week of Oct-2019 dataset.
<!-- #endregion -->

```python id="WRKAYLk8E-cE" outputId="a9752c5d-2ac7-4397-e89a-31c75f507269"
df['event_time_dt'].max()
```

<!-- #region id="9p8s8iRaE-cE" -->
We drop `event_time_dt` column as it will not be used anymore.
<!-- #endregion -->

```python id="UGjYRVEbE-cF"
df = df.drop(['event_time_dt'],  axis=1)
```

```python id="QIUUcjyaE-cF" outputId="e68ecd23-e613-450a-d3f1-450d725c7cd6"
df.head()
```

<!-- #region id="obYUFiImE-cG" -->
Save the data as a single parquet file to be used in the ETL section.
<!-- #endregion -->

```python id="Hn0AAqHoE-cG"
# save df as parquet files on disk
df.to_parquet(os.path.join(INPUT_DATA_DIR, 'Oct-2019.parquet'))
```

<!-- #region id="YlU41Xs6FgSy" -->
## ETL with NVTabular
<!-- #endregion -->

<!-- #region id="rdbomPTRFgS0" -->
In this section, we will create a preprocessing and feature engineering pipeline with [Rapids cuDF](https://github.com/rapidsai/cudf) and [Merlin NVTabular](https://github.com/NVIDIA/NVTabular) libraries to prepare our dataset for session-based recommendation model training. 

NVTabular is a feature engineering and preprocessing library for tabular data that is designed to easily manipulate terabyte scale datasets and train deep learning (DL) based recommender systems. It provides high-level abstraction to simplify code and accelerates computation on the GPU using the RAPIDS Dask-cuDF library, and is designed to be interoperable with both PyTorch and TensorFlow using dataloaders that have been developed as extensions of native framework code.

Our main goal is to create sequential features. In order to do that, we are going to perform the following:

- Categorify categorical features with `Categorify()` op
- Create temporal features with a `user-defined custom` op and `Lambda` op
- Transform continuous features using `Log` and `Normalize` ops
- Group all these features together at the session level sorting the interactions by time with `Groupby`
- Finally export the preprocessed datasets to parquet files by hive-partitioning.
<!-- #endregion -->

```python id="XTEGEg3bFgS8"
# define data path about where to get our data
INPUT_DATA_DIR = os.environ.get("INPUT_DATA_DIR", "/workspace/data/")
```

<!-- #region id="mzHE1FEDFgS-" -->
## Read the Input Parquet file
<!-- #endregion -->

<!-- #region id="xqq-E0sVFgS-" -->
We already performed certain preprocessing steps on the first month (Oct-2019) of the raw dataset : <br>

- we created `event_time_ts` column from `event_time` column which shows the time when event happened at (in UTC).
- we created `prod_first_event_time_ts` column which indicates the timestamp that an item was seen first time.
- we removed the rows where the `user_session` is Null. As a result, 2 rows were removed.
- we categorified the `user_session` column, so that it now has only integer values.
- we removed consequetively repeated (user, item) interactions. For example, an original session with `[1, 2, 4, 1, 2, 2, 3, 3, 3]` product interactions has become `[1, 2, 4, 1, 2, 3]` after removing the repeated interactions on the same item within the same session.
<!-- #endregion -->

<!-- #region id="hsav2LH9FgS_" -->
Even though the original dataset contains 7 months data files, we are going to use the first seven days of the `Oct-2019.csv` ecommerce dataset. We use cuDF to read the parquet file. 
<!-- #endregion -->

```python id="ZP_DOjBbFgTA" outputId="86c5792d-339d-4fbc-f70d-15ba398d1652"
%%time
df = cudf.read_parquet(os.path.join(INPUT_DATA_DIR, 'Oct-2019.parquet'))  
df.head(5)
```

```python id="PGPAp5YxFgTB" outputId="0192cff4-27b9-4a71-aa40-1a50701b90ab"
df.shape
```

<!-- #region id="lcn9L1PDFgTC" -->
Let's check if there is any column with nulls.
<!-- #endregion -->

```python id="-xLmJ8LGFgTD" outputId="84a5c0e4-a11e-4a6a-8d27-f3fc0d519439"
df.isnull().any()
```

<!-- #region tags=[] id="9wXtQcIWFgTD" -->
We see that `'category_code'` and `'brand'` columns have null values, and in the following cell we are going to fill these nulls with via categorify op, and then all categorical columns will be encoded to continuous integers. Note that we add `start_index=1` in the `Categorify op` for the categorical columns, the reason for that we want the encoded null values to start from `1` instead of `0` because we reserve `0` for padding the sequence features.
<!-- #endregion -->

<!-- #region id="pq-ivc1PFgTE" -->
## Initialize NVTabular Workflow

### Categorical Features Encoding
<!-- #endregion -->

```python id="rJSahDvGFgTE"
# categorify features 
cat_feats = ['user_session', 'category_code', 'brand', 'user_id', 'product_id', 'category_id', 'event_type'] >> nvt.ops.Categorify(start_index=1)
```

<!-- #region id="qhLlRxTSFgTF" -->
### Extract Temporal Features
<!-- #endregion -->

```python id="bU4fd7EwFgTF"
# create time features
session_ts = ['event_time_ts']

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
```

<!-- #region id="vy8iqoiDFgTF" -->
Now let's create cycling features from the `sessiontime_weekday` column. We would like to use the temporal features (hour, day of week, month, etc.) that have inherently cyclical characteristic. We represent the day of week as a cycling feature (sine and cosine), so that it can be represented in a continuous space. That way, the difference between the representation of two different days is the same, in other words, with cyclical features we can convey closeness between data. You can read more about it [here](https://ianlondon.github.io/blog/encoding-cyclical-features-24hour-time/).
<!-- #endregion -->

```python id="Afiq11p_FgTG"
def get_cycled_feature_value_sin(col, max_value):
    value_scaled = (col + 0.000001) / max_value
    value_sin = np.sin(2*np.pi*value_scaled)
    return value_sin

def get_cycled_feature_value_cos(col, max_value):
    value_scaled = (col + 0.000001) / max_value
    value_cos = np.cos(2*np.pi*value_scaled)
    return value_cos
```

```python id="B4gqz2X7FgTG"
weekday_sin = sessiontime_weekday >> (lambda col: get_cycled_feature_value_sin(col+1, 7)) >> nvt.ops.Rename(name = 'et_dayofweek_sin')
weekday_cos= sessiontime_weekday >> (lambda col: get_cycled_feature_value_cos(col+1, 7)) >> nvt.ops.Rename(name = 'et_dayofweek_cos')
```

<!-- #region id="BcirHkJgFgTH" -->
**Add Product Recency feature**
<!-- #endregion -->

<!-- #region id="BEUI-kTwFgTH" -->
- Let's define a custom op to calculate product recency in days
<!-- #endregion -->

```python id="rMTSXhwFFgTH"
from nvtabular.ops import Operator

class ItemRecency(Operator):
    def transform(self, columns, gdf):
        for column in columns.names:
            col = gdf[column]
            item_first_timestamp = gdf['prod_first_event_time_ts']
            delta_days = (col - item_first_timestamp) / (60*60*24)
            gdf[column + "_age_days"] = delta_days * (delta_days >=0)
        return gdf
            
    def output_column_names(self, columns):
        return ColumnSelector([column + "_age_days" for column in columns.names])

    def dependencies(self):
        return ["prod_first_event_time_ts"]
    
    
recency_features = ['event_time_ts'] >> ItemRecency() 
recency_features_norm = recency_features >> nvt.ops.LogOp() >> nvt.ops.Normalize() >> nvt.ops.Rename(name='product_recency_days_log_norm')
```

```python id="zvemsATWFgTI"
time_features = (
    session_time +
    sessiontime_weekday +
    weekday_sin +
    weekday_cos +
    recency_features_norm
)
```

<!-- #region id="br8Yv6cMFgTI" -->
### Normalize Continuous Features
<!-- #endregion -->

```python id="-ygRxgYuFgTI"
# Smoothing price long-tailed distribution and applying standardization
price_log = ['price'] >> nvt.ops.LogOp() >> nvt.ops.Normalize() >> nvt.ops.Rename(name='price_log_norm')
```

```python id="EyoOLlqkFgTJ"
# Relative price to the average price for the category_id
def relative_price_to_avg_categ(col, gdf):
    epsilon = 1e-5
    col = ((gdf['price'] - col) / (col + epsilon)) * (col > 0).astype(int)
    return col
    
avg_category_id_pr = ['category_id'] >> nvt.ops.JoinGroupby(cont_cols =['price'], stats=["mean"]) >> nvt.ops.Rename(name='avg_category_id_price')
relative_price_to_avg_category = avg_category_id_pr >> nvt.ops.LambdaOp(relative_price_to_avg_categ, dependency=['price']) >> nvt.ops.Rename(name="relative_price_to_avg_categ_id")
```

<!-- #region id="bgQ675fqFgTJ" -->
### Grouping interactions into sessions
<!-- #endregion -->

<!-- #region id="rZ4dLmgUFgTJ" -->
**Aggregate by session id and creates the sequential features**
<!-- #endregion -->

```python id="J_G1YzfqFgTK"
groupby_feats = ['event_time_ts', 'user_session'] + cat_feats + time_features + price_log + relative_price_to_avg_category
```

```python id="xtSyScspFgTK"
# Define Groupby Workflow
groupby_features = groupby_feats >> nvt.ops.Groupby(
    groupby_cols=["user_session"], 
    sort_cols=["event_time_ts"],
    aggs={
        'user_id': ['first'],
        'product_id': ["list", "count"],
        'category_code': ["list"],  
        'event_type': ["list"], 
        'brand': ["list"], 
        'category_id': ["list"], 
        'event_time_ts': ["first"],
        'event_time_dt': ["first"],
        'et_dayofweek_sin': ["list"],
        'et_dayofweek_cos': ["list"],
        'price_log_norm': ["list"],
        'relative_price_to_avg_categ_id': ["list"],
        'product_recency_days_log_norm': ["list"]
        },
    name_sep="-")
```

<!-- #region id="KVtitxs7FgTK" -->
- Select columns which are list
<!-- #endregion -->

```python id="8jkZsNNvFgTL"
groupby_features_list = groupby_features['product_id-list',
        'category_code-list',  
        'event_type-list', 
        'brand-list', 
        'category_id-list', 
        'et_dayofweek_sin-list',
        'et_dayofweek_cos-list',
        'price_log_norm-list',
        'relative_price_to_avg_categ_id-list',
        'product_recency_days_log_norm-list']
```

```python id="5zz4wa_TFgTL"
SESSIONS_MAX_LENGTH = 20 
MINIMUM_SESSION_LENGTH = 2
```

<!-- #region id="Kk9up4feFgTM" -->
We truncate the sequence features in length according to sessions_max_length param, which is set as 20 in our example.
<!-- #endregion -->

```python id="NYwTLjSyFgTM"
groupby_features_trim = groupby_features_list >> nvt.ops.ListSlice(0,SESSIONS_MAX_LENGTH) >> nvt.ops.Rename(postfix = '_seq')
```

<!-- #region id="vrTG8oPOFgTM" -->
- Create a `day_index` column in order to partition sessions by day when saving the parquet files.
<!-- #endregion -->

```python id="RdeUzUyiFgTN"
# calculate session day index based on 'timestamp-first' column
day_index = ((groupby_features['event_time_dt-first'])  >> 
    nvt.ops.LambdaOp(lambda col: (col - col.min()).dt.days +1) >> 
    nvt.ops.Rename(f = lambda col: "day_index")
)
```

<!-- #region id="1ZhdHSbkFgTN" -->
- Select certain columns to be used in model training
<!-- #endregion -->

```python id="YpMnwY3lFgTO"
selected_features = groupby_features['user_session', 'product_id-count'] + groupby_features_trim + day_index
```

<!-- #region id="nvOhA3FPFgTO" -->
- Filter out the session that have less than 2 interactions.
<!-- #endregion -->

```python id="4IMUckPnFgTO"
filtered_sessions = selected_features >> nvt.ops.Filter(f=lambda df: df["product_id-count"] >= MINIMUM_SESSION_LENGTH)
```

```python id="XaF2xzSlFgTO"
# avoid numba warnings
from numba import config
config.CUDA_LOW_OCCUPANCY_WARNINGS = 0
```

<!-- #region id="vB-GJ2SSFgTP" -->
- Initialize the NVTabular dataset object and workflow graph.
<!-- #endregion -->

<!-- #region id="q4ioP3jDFgTP" -->
NVTabular's preprocessing and feature engineering workflows are directed graphs of operators. When we initialize a Workflow with our pipeline, workflow organizes the input and output columns.
<!-- #endregion -->

```python id="4C61VMBNFgTP"
dataset = nvt.Dataset(df)

workflow = nvt.Workflow(filtered_sessions)
workflow.fit(dataset)
sessions_gdf = workflow.transform(dataset).to_ddf()
```

<!-- #region id="TFSyYeV5FgTQ" -->
Above, we created an NVTabular Dataset object using our input dataset. Then, we calculate statistics for this workflow on the input dataset, i.e. on our training set, using the `workflow.fit()` method so that our Workflow can use these stats to transform any given input.
<!-- #endregion -->

<!-- #region id="_r_nHCIUFgTQ" -->
Let's print the head of our preprocessed dataset. You can notice that now each example (row) is a session and the sequential features with respect to user interactions were converted to lists with matching length.
<!-- #endregion -->

```python id="277HZkLWFgTQ" outputId="df51cf64-5032-4b2a-95e5-780212988b70"
sessions_gdf.head(3)
```

```python id="vnFFkk8XFgTQ" outputId="ba1aa575-7857-4159-9e70-a33a23eb1399"
workflow.output_schema.column_names
```

<!-- #region id="bJcnPhNBFgTR" -->
- Save NVTabular workflow to load at the inference step.
<!-- #endregion -->

```python id="VCsSnQXMFgTR"
workflow_path = os.path.join(INPUT_DATA_DIR, 'workflow_etl')
workflow.save(workflow_path)
```

<!-- #region id="jbHfTea4FgTR" -->
## Exporting data
<!-- #endregion -->

<!-- #region id="6mKC6NmQFgTR" -->
We export dataset to parquet partioned by the session `day_index` column.
<!-- #endregion -->

```python id="lFzVEYsKFgTS"
# define partition column
PARTITION_COL = 'day_index'

# define output_folder to store the partitioned parquet files
OUTPUT_FOLDER = os.environ.get("OUTPUT_FOLDER", INPUT_DATA_DIR + "sessions_by_day")
!mkdir -p $OUTPUT_FOLDER
```

<!-- #region id="bJmTyfkzFgTS" -->
In this section we are going to create a folder structure as shown below. As we explained above, this is just to structure parquet files so that it would be easier to do incremental training and evaluation.
<!-- #endregion -->

<!-- #region id="Zok2k9DWFgTS" -->
```
/sessions_by_day/
|-- 1
|   |-- train.parquet
|   |-- valid.parquet
|   |-- test.parquet

|-- 2
|   |-- train.parquet
|   |-- valid.parquet
|   |-- test.parquet
```
<!-- #endregion -->

<!-- #region id="K-fCUKlrFgTS" -->
`gpu_preprocessing` function converts the process df to a Dataset object and write out hive-partitioned data to disk.
<!-- #endregion -->

```python id="JV6k0dsMFgTS" outputId="1cdd4770-8c22-4021-faed-9a2194eae823"
from transformers4rec.data.preprocessing import save_time_based_splits
save_time_based_splits(data=nvt.Dataset(sessions_gdf),
                       output_dir= OUTPUT_FOLDER,
                       partition_col=PARTITION_COL,
                       timestamp_col='user_session', 
                      )
```

```python id="txYDB2f5FgTT" outputId="ae9db4f2-747d-484b-c247-4ad072f57585"
# check out the OUTPUT_FOLDER
!ls $OUTPUT_FOLDER
```

```python id="f2d1e30c"
# Copyright 2021 NVIDIA Corporation. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
```

<!-- #region id="85BpF7YjGkUa" -->
## Session-based recommendation with Transformers4Rec
<!-- #endregion -->

<!-- #region id="b7ec3b1b" -->
In the previous section we went through our ETL pipeline with NVTabular library, and created sequential features to be used for training a session-based recommendation model. In this section we will learn:

- Accelerating data loading of parquet files multiple features on PyTorch using NVTabular library
- Training and evaluating an RNN-based (GRU) session-based recommendation model 
- Training and evaluating a Transformer architecture (XLNET) for session-based recommendation model
- Integrate side information (additional features) into transformer architectures in order to improve recommendation accuracy
<!-- #endregion -->

<!-- #region id="a9193ebc" -->
Session-based recommendation, a sub-area of sequential recommendation, has been an important task in online services like e-commerce and news portals, where most users either browse anonymously or may have very distinct interests for different sessions. Session-Based Recommender Systems (SBRS) have
been proposed to model the sequence of interactions within the current user session, where a session is a short sequence of user interactions typically bounded by user inactivity. They have recently gained popularity due to their ability to capture short-term and contextual user preferences towards items.


Many methods have been proposed to leverage the sequence of interactions that occur during a session, including session-based k-NN algorithms like V-SkNN [1] and neural approaches like GRU4Rec [2]. In addition,  state of the art NLP approaches have inspired RecSys practitioners and researchers to leverage the self-attention mechanism and the Transformer-based architectures for sequential [3] and session-based recommendation [4].
<!-- #endregion -->

<!-- #region id="0147de64" -->
In this tutorial, we introduce the [Transformers4Rec](https://github.com/NVIDIA-Merlin/Transformers4Rec) open-source library for sequential and session-based recommendation task.

With Transformers4Rec we import from the HF Transformers NLP library the transformer architectures and their configuration classes. 

In addition, Transformers4Rec provides additional blocks necessary for recommendation, e.g., input features normalization and aggregation, and heads for recommendation and sequence classification/prediction. We also extend their Trainer class to allow for the evaluation with RecSys metrics.

Here are some of the most important modules:

- [TabularSequenceFeatures](https://nvidia-merlin.github.io/Transformers4Rec/main/api/transformers4rec.torch.html#transformers4rec.torch.TabularSequenceFeatures) is the input block for sequential features. Based on a `Schema` and options set by the user, it dynamically creates all the necessary layers (e.g. embedding layers) to encode, normalize, and aggregate categorical and continuous features. It also allows to set the `masking` training approach (e.g. Causal LM, Masked LM).
- [TransformerBlock](https://nvidia-merlin.github.io/Transformers4Rec/main/api/transformers4rec.torch.html#transformers4rec.torch.TransformerBlock) class is the bridge that adapts HuggingFace Transformers for session-based and sequential-based recommendation models.
- [SequentialBlock](https://nvidia-merlin.github.io/Transformers4Rec/main/api/transformers4rec.torch.html#transformers4rec.torch.SequentialBlock) allows the definition of a model body as as sequence of layer (similarly to [torch.nn.sequential](https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html)). It is designed to define our model as a sequence of layers and automatically setting the input shape of a layer from the output shape of the previous one.
- [Head](https://nvidia-merlin.github.io/Transformers4Rec/main/api/transformers4rec.torch.html#transformers4rec.torch.Head) class defines the head of a model.
- [NextItemPredictionTask](https://nvidia-merlin.github.io/Transformers4Rec/main/api/transformers4rec.torch.html#transformers4rec.torch.NextItemPredictionTask) is the class to support next item prediction task, combining a model body with a head.
- [Trainer](https://nvidia-merlin.github.io/Transformers4Rec/main/api/transformers4rec.torch.html#transformers4rec.torch.Trainer) extends the `Trainer` class from HF transformers and manages the model training and evaluation.

You can check the [full documentation](https://nvidia-merlin.github.io/Transformers4Rec/main/index.html) of Transformers4Rec if needed.
<!-- #endregion -->

<!-- #region id="c5a9886d" -->
In the following Figure, we present a reference architecture that we are going to build with Transformers4Rec PyTorch API in this section. We are going to start using only `product-id` as input feature, but as you can notice in the figure, we can add additional categorical and numerical features later to improve recommendation accuracy.
<!-- #endregion -->

<!-- #region id="16e6d8c7" -->
<img src='https://s3.us-west-2.amazonaws.com/secure.notion-static.com/360a24b7-5d3f-41bc-bafc-e6f7a50b8a90/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20211011%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20211011T112731Z&X-Amz-Expires=86400&X-Amz-Signature=2127778c75a954ad2b20a01cbb06aa9774a935b18af8875d784d0250dc999f93&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22'>
<p><center>Figure 1. Transformers4Rec meta-architecture.</center></p>
<!-- #endregion -->

<!-- #region id="d08dfc19" -->
## Training an RNN-based Session-based Recommendation Model
<!-- #endregion -->

<!-- #region id="6fddb668" -->
In this section, we use a type of Recurrent Neural Networks (RNN) - the Gated Recurrent Unit (GRU)[5] - to do next-item prediction using a sequence of events (e.g., click, view, or purchase) per user in a given session. There is obviously some sequential patterns that we want to capture to provide more relevant recommendations. In our case, the input of the GRU layer is a representation of the user interaction, the internal GRU hidden state encodes a representation of the session based on past interactions and the outputs are the next-item predictions. Basically, for each item in a given session, we generate the output as the predicted preference of the items, i.e. the likelihood of being the next.
<!-- #endregion -->

<!-- #region id="49750806" -->
Figure 2 illustrates the logic of predicting next item in a given session. First, the product ids are embedded and fed as a sequence to a GRU layer, which outputs a representation than can be used to predict the next item. For the sake of simplicity, we treat the recommendation as a multi-class classification problem and use cross-entropy loss. In our first example, we use a GRU block instead of `Transformer block` (shown in the Figure 1).
<!-- #endregion -->

<!-- #region id="bfda5ba7" -->
<p><center><img src='https://s3.us-west-2.amazonaws.com/secure.notion-static.com/72275c7e-3e6c-4399-adb0-406c1bd36863/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20211011%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20211011T112939Z&X-Amz-Expires=86400&X-Amz-Signature=19668197b543d096cd1b1e9ed1579dfca46c9329d70edd4b4b937902209ef0e4&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22'><br>Figure 2. Next item prediction with RNN.</center></p>
<!-- #endregion -->

<!-- #region id="a8e7be36" -->
### Instantiates Schema object from a `schema` file.
<!-- #endregion -->

```python id="a64c3082"
from merlin_standard_lib import Schema
# Define schema object to pass it to the TabularSequenceFeatures class
SCHEMA_PATH = 'schema_tutorial.pb'
schema = Schema().from_proto_text(SCHEMA_PATH)
schema = schema.select_by_name(['product_id-list_seq'])
```

<!-- #region id="9a4493de" -->
Transformers4Rec library relies on `Schema` object in `TabularSequenceFeatures` that takes the input features as input and create all the necessary layers to process and aggregate them. As you can see below, the `schema.pb` is a protobuf text file contains features metadata, including statistics about features such as cardinality, min and max values and also tags based on their characteristics and dtypes (e.g., `categorical`, `continuous`, `list`, `item_id`). We can tag our target column and even add the prediction task such as `binary`, `regression` or `multiclass` as tags for the target column in the `schema.pb` file. The `Schema` provides a standard representation for metadata that is useful when training machine learning or deep learning models.

The metadata information loaded from `Schema` and their tags are used to automatically set the parameters of Transformers4rec models. Certain Transformers4rec modules have a `from_schema()` method to instantiate their parameters and layers from protobuf text file respectively. 

Although in this tutorial we are defining the `Schema` manually, the next NVTabular release is going to generate the schema with appropriate types and tags automatically from the preprocessing workflow, allowing the user to set additional feaure tags if needed.
<!-- #endregion -->

<!-- #region id="81225a6d" -->
Let's inspect the first lines of `schema.pb`
<!-- #endregion -->

```python id="33ab76ed" outputId="f39ee485-acf9-4a6c-e056-c86f06914ef0"
!head -30 $SCHEMA_PATH
```

<!-- #region id="68b323f3" -->
### Defining the input block: `TabularSequenceFeatures`
<!-- #endregion -->

<!-- #region id="14f277f5" -->
We define our input block using `TabularSequenceFeatures` class. The `from_schema()` method directly parses the schema and accepts sequential and non-sequential features. Based on the `Schema` and some user-defined options, the categorical features are represented by embeddings and numerical features can be represented as continuous scalars or by a technique named Soft One-Hot embeddings (more info in our [paper's online appendix](https://github.com/NVIDIA-Merlin/publications/blob/main/2021_acm_recsys_transformers4rec/Appendices/Appendix_A-Techniques_used_in_Transformers4Rec_Meta-Architecture.md)). 

The embedding features can optionally be normalized (`layer_norm=True`). Data augmentation methods like "Stochastic Swap Noise" (`pre="stochastic-swap-noise"`) and `aggregation` opptions (like `concat` and `elementwise-sum`) are also available. The continuous features can also be combined and projected by MLP layers by setting `continuous_projection=[dim]`. Finally, the `max_sequence_length` argument defines the maximum sequence length of our sequential input.

Another important argument is the `masking` method, which sets the training approach. See Section 3.2.2 for details on this.
<!-- #endregion -->

```python id="1d10f641"
sequence_length = 20
inputs = tr.TabularSequenceFeatures.from_schema(
        schema,
        max_sequence_length= sequence_length,
        masking = 'causal',
    )
```

<!-- #region id="dbd88fe6" -->
### Connecting the blocks with `SequentialBlock`

The `SequentialBlock` creates a pipeline by connecting the building blocks in a serial way, so that the input shape of one block is inferred from the output of the previus block. In this example, the `TabularSequenceFeatures` object is followed by an MLP projection layer, which feeds data to a GRU block.
<!-- #endregion -->

```python id="31aafb08"
d_model = 128
body = tr.SequentialBlock(
        inputs,
        tr.MLPBlock([d_model]),
        tr.Block(torch.nn.GRU(input_size=d_model, hidden_size=d_model, num_layers=1), [None, 20, d_model])
)
```

<!-- #region id="d8ca3850" -->
### Item Prediction head and tying embeddings
<!-- #endregion -->

<!-- #region id="8bdf2e4a" -->
In our experiments published in our [ACM RecSys'21 paper](https://github.com/NVIDIA-Merlin/publications/blob/main/2021_acm_recsys_transformers4rec/recsys21_transformers4rec_paper.pdf) [8], we used the next item prediction head, which projects the output of the RNN/Transformer block to the items space, followed by a softmax layer to produce the relevance scores over all items. For the output layer we provide the `Tying Embeddings` technique (`weight_tying`). It was proposed originally by the NLP community to tie the weights of the input (item id) embedding matrix with the output projection layer, showed to be a very effective technique in extensive experimentation for competitions and empirical analysis (for more details see our [paper](https://github.com/NVIDIA-Merlin/publications/blob/main/2021_acm_recsys_transformers4rec/recsys21_transformers4rec_paper.pdf) and its online [appendix](https://github.com/NVIDIA-Merlin/publications/blob/main/2021_acm_recsys_transformers4rec/Appendices/Appendix_A-Techniques_used_in_Transformers4Rec_Meta-Architecture.md)). In practice, such technique helps the network to learn faster item embeddings even for rare items, reduces the number of parameters for large item cardinalities and enables Approximate Nearest Neighbours (ANN) search on inference, as the predictions can be obtained by a dot product between the model output and the item embeddings.
<!-- #endregion -->

<!-- #region id="c7116b15" -->
Next, we link the transformer-body to the inputs and the prediction tasks to get the final PyTorch `Model` class.
<!-- #endregion -->

```python id="89060b5d" outputId="adcaaec1-bf9b-412d-fa98-e2a3c6d4aeb0"
head = tr.Head(
    body,
    tr.NextItemPredictionTask(weight_tying=True, hf_format=True, 
                              metrics=[NDCGAt(top_ks=[10, 20], labels_onehot=True),  
                                       RecallAt(top_ks=[10, 20], labels_onehot=True)]),
)
model = tr.Model(head)
```

<!-- #region id="b7a1426f" -->
### Define a Dataloader function from schema
<!-- #endregion -->

<!-- #region id="29075a9e" -->
We use optimized NVTabular PyTorch Dataloader which has the following benefits:
- removing bottlenecks from dataloading by processing large chunks of data at a time instead iterating by row
- processing datasets that don’t fit within the GPU or CPU memory by streaming from the disk
- reading data directly into the GPU memory and removing CPU-GPU communication
- preparing batch asynchronously into the GPU to avoid CPU-GPU communication
- supporting commonly used formats such as parquet
- having native support to sparse sequential features
<!-- #endregion -->

```python id="a1baaf30"
# import NVTabular dependencies
from transformers4rec.torch.utils.data_utils import NVTabularDataLoader

x_cat_names, x_cont_names = ['product_id-list_seq'], []

# dictionary representing max sequence length for column
sparse_features_max = {
    fname: sequence_length
    for fname in x_cat_names + x_cont_names
}

# Define a `get_dataloader` function to call in the training loop
def get_dataloader(path, batch_size=32):

    return NVTabularDataLoader.from_schema(
        schema,
        path, 
        batch_size,
        max_sequence_length=sequence_length,
        sparse_names=x_cat_names + x_cont_names,
        sparse_max=sparse_features_max,
)
```

<!-- #region id="9ff98ee9" -->
### Daily Fine-Tuning: Training over a time window
<!-- #endregion -->

<!-- #region id="825efe4a" -->
Now that the model is defined, we are going to launch training. For that, Transfromers4rec extends the HF Transformers `Trainer` class to adapt the evaluation loop for session-based recommendation task and the calculation of ranking metrics. 
The original HF `Trainer.train()` method is not overloaded, meaning that we leverage the efficient training implementation from HF transformers library, which manages for example half-precision (FP16) training.
<!-- #endregion -->

<!-- #region id="3f217071" -->
### Set training arguments
<!-- #endregion -->

```python id="a112d8ed"
from transformers4rec.config.trainer import T4RecTrainingArguments
from transformers4rec.torch import Trainer

#Set arguments for training 
train_args = T4RecTrainingArguments(local_rank = -1, 
                                    dataloader_drop_last = False,
                                    report_to = [],   #set empy list to avoig logging metrics to Weights&Biases
                                    gradient_accumulation_steps = 1,
                                    per_device_train_batch_size = 256, 
                                    per_device_eval_batch_size = 32,
                                    output_dir = "./tmp", 
                                    max_sequence_length=sequence_length,
                                    learning_rate=0.00071,
                                    num_train_epochs=3,
                                    logging_steps=200,
                                   )
```

<!-- #region id="85047d3a" -->
### Instantiate the Trainer
<!-- #endregion -->

```python id="5aca4e84"
# Instantiate the T4Rec Trainer, which manages training and evaluation
trainer = Trainer(
    model=model,
    args=train_args,
    schema=schema,
    compute_metrics=True,
)
```

<!-- #region id="fcf86a74" -->
Define the output folder of the processed parquet files
<!-- #endregion -->

```python id="fc64a1f4"
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/workspace/data/sessions_by_day")
```

<!-- #region id="f5293026" -->
### Model finetuning and incremental evaluation
Training models incrementally, e.g. fine-tuning pre-trained models with new data over time is a common practice in industry to scale to the large streaming data been generated every data. Furthermore, it is common to evaluate recommendation models on data that came after the one used to train the models, for a more realistic evaluation.

Here, we use a loop that to conduct a time-based finetuning, by iteratively training and evaluating using a sliding time window as follows: At each iteration, we use training data of a specific time index <i>t</i> to train the model then we evaluate on the validation data of next index <i>t + 1</i>. We set the start time to 1 and end time to 4.
<!-- #endregion -->

```python id="3e48f16f" outputId="f0763043-ea27-423e-b647-2d1c5853562d"
%%time
start_time_window_index = 1
final_time_window_index = 4
for time_index in range(start_time_window_index, final_time_window_index):
    # Set data 
    time_index_train = time_index
    time_index_eval = time_index + 1
    train_paths = glob.glob(os.path.join(OUTPUT_DIR, f"{time_index_train}/train.parquet"))
    eval_paths = glob.glob(os.path.join(OUTPUT_DIR, f"{time_index_eval}/valid.parquet"))
    
    # Initialize dataloaders
    trainer.train_dataloader = get_dataloader(train_paths, train_args.per_device_train_batch_size)
    trainer.eval_dataloader = get_dataloader(eval_paths, train_args.per_device_eval_batch_size)
    
    # Train on day related to time_index 
    print('*'*20)
    print("Launch training for day %s are:" %time_index)
    print('*'*20 + '\n')
    trainer.reset_lr_scheduler()
    trainer.train()
    trainer.state.global_step +=1
    
    # Evaluate on the following day
    train_metrics = trainer.evaluate(metric_key_prefix='eval')
    print('*'*20)
    print("Eval results for day %s are:\t" %time_index_eval)
    print('\n' + '*'*20 + '\n')
    for key in sorted(train_metrics.keys()):
        print(" %s = %s" % (key, str(train_metrics[key]))) 
    trainer.wipe_memory()
```

<!-- #region id="ee3a0dab" -->
Let's write out model evaluation accuracy results to a text file to compare model at the end
<!-- #endregion -->

```python id="bc41e5c5"
with open("results.txt", 'w') as f: 
    f.write('GRU accuracy results:')
    f.write('\n')
    for key, value in  model.compute_metrics().items(): 
        f.write('%s:%s\n' % (key, value.item()))
```

<!-- #region tags=[] id="2c63913c" -->
### Metrics
<!-- #endregion -->

<!-- #region id="d062ec1a" -->
We have extended the HuggingFace transformers Trainer class (PyTorch only) to support evaluation of RecSys metrics. The following information
retrieval metrics are used to compute the Top-20 accuracy of recommendation lists containing all items: <br> 
- **Normalized Discounted Cumulative Gain (NDCG@20):** NDCG accounts for rank of the relevant item in the recommendation list and is a more fine-grained metric than HR, which only verifies whether the relevant item is among the top-k items.

- **Hit Rate (HR@20)**: Also known as `Recall@n` when there is only one relevant item in the recommendation list. HR just verifies whether the relevant item is among the top-n items.
<!-- #endregion -->

<!-- #region id="8dab45a7" -->
### Restart the kernel to free our GPU memory
<!-- #endregion -->

```python id="e2138cba" outputId="7d63ad79-8255-4623-9eb2-f260c45437c2"
import IPython
app = IPython.Application.instance()
app.kernel.do_shutdown(True)
```

<!-- #region id="6b21fd3b" -->
At this stage if the kernel does not restart automatically, we expect you to manually restart the kernel to free GPU memory so that you can move on to the next session-based model training with a SOTA deep learning Transformer-based model, [XLNet](https://arxiv.org/pdf/1906.08237.pdf).
<!-- #endregion -->

<!-- #region id="08de0ba3" -->
## Training a Transformer-based Session-based Recommendation Model
<!-- #endregion -->

<!-- #region id="26cdec77" -->
### What's Transformers?
<!-- #endregion -->

<!-- #region id="cbac868e" -->
The Transformer is a competitive alternative to the models using Recurrent Neural Networks (RNNs) for a range of sequence modeling tasks. The Transformer architecture [6] was introduced as a novel architecture in NLP domain that aims to solve sequence-to-sequence tasks relying entirely on self-attention mechanism to compute representations of its input and output. Hence, the Transformer overperforms RNNs with their three mechanisms: 

- Non-sequential: Transformers network is parallelized where as RNN computations are inherently sequential. That resulted in significant speed-up in the training time.
- Self-attention mechanisms: Transformers rely entirely on self-attention mechanisms that directly model relationships between all item-ids in a sequence.  
- Positional encodings: A representation of the location or “position” of items in a sequence which is used to give the order context to the model architecture.
<!-- #endregion -->

<!-- #region id="06235f16" -->
<p><center><img src='https://s3.us-west-2.amazonaws.com/secure.notion-static.com/5a89728b-f767-43f9-a04b-249ac5148c2a/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20211011%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20211011T113310Z&X-Amz-Expires=86400&X-Amz-Signature=ae0e7ad7c947f6e916c5e809d2779079c81ebd16d9dc2eca17604c5fcca04d02&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22'><br>Figure 3. Transformer vs vanilla RNN.</center></p>
<!-- #endregion -->

<!-- #region id="3988fdc6" -->
Figure 4 illustrates the differences of Transformer (self-attention based) and a vanilla RNN architecture. As we see, RNN cannot be parallelized because it uses sequential processing over time (notice the sequential path from previous cells to the current one). On the other hand, the Transformer is a more powerful architecture because the self-attention mechanism is capable of representing dependencies within the sequence of tokens, favors parallel processing and handle longer sequences.

As illustrated in the [Attention is All You Need](https://arxiv.org/pdf/1706.03762.pdf) paper, the original transformer model is made up of an encoder and decoder where each is a stack we can call a transformer block. In Transformers4Rec architectures we use the encoder block of transformer architecture.
<!-- #endregion -->

<!-- #region id="6727e272" -->
<p><center><img src='https://s3.us-west-2.amazonaws.com/secure.notion-static.com/f0b9e980-ad4e-452b-8936-68fbac8fccd7/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20211011%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20211011T113407Z&X-Amz-Expires=86400&X-Amz-Signature=263317ed25284a919e276c6d552bc032ab9100106730663b6d2d8dc5fcc77c95&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22'><br>Figure 4. Encoder block of the Transformer Architecture.</center></p>
<!-- #endregion -->

<!-- #region id="184fca87" -->
### XLNet
<!-- #endregion -->

<!-- #region id="da6da733" -->
Here, we use XLNet [10] as the Transformer block in our architecture. It was originally proposed to be trained with the *Permutation Language Modeling (PLM)*  technique, that combines the advantages of autoregressive (Causal LM) and autoencoding (Masked LM). Although, we found out in our [paper](https://github.com/NVIDIA-Merlin/publications/blob/main/2021_acm_recsys_transformers4rec/recsys21_transformers4rec_paper.pdf) [8] that the *Masked Language Model (MLM)* approach worked better than PLM for the small sequences in session-based recommendation, thus we use MLM for this example. MLM was introduced in *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding* paper [8]. 

Figure 5 illustrates the causal language modeling (LM) and masked LM. In this example, we use in causal LM for RNN masked LM for XLNet. Causal LM is the task of predicting the token following a sequence of tokens, where the model only attends to the left context, i.e. models the probability of a token given the previous tokens in a sentence [7]. On the other hand, the MLM randomly masks some of the tokens from the input sequence, and the objective is to predict the original vocabulary id of the masked word based only on its bi-directional context. When we train with MLM, the Transformer layer is also allowed to use positions on the right (future information) during training. During inference, all past items are visible for the Transformer layer, which tries to predict the next item. It performs a type of data augmentation, by masking different positions of the sequences in each training epoch.
<!-- #endregion -->

<!-- #region id="26638655" -->
<p><center><img src='https://s3.us-west-2.amazonaws.com/secure.notion-static.com/35f985e3-b874-4aba-9420-7ee6fa0b8e7e/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20211011%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20211011T113457Z&X-Amz-Expires=86400&X-Amz-Signature=4e1a1639f829054317666ff74fc10dbff4411c150405bec184f5b3cbd2127977&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22'><br>Figure 5. Causal and Masked Language Model masking methods.</center></p>
<!-- #endregion -->

<!-- #region id="a512b632" -->
### Train XLNET for Next Item Prediction
<!-- #endregion -->

<!-- #region id="b7b45ea4" -->
Now we are going to define an architecture for next-item prediction using the XLNET architecture.
<!-- #endregion -->

```python id="4a12d605"
import os
import glob

import torch 
import transformers4rec.torch as tr

from transformers4rec.torch.ranking_metric import NDCGAt, RecallAt
```

<!-- #region id="69c6542b" -->
As we did above, we start with defining our schema object and selecting only the `product_id` feature for training.
<!-- #endregion -->

```python id="e04d30d5"
from merlin_standard_lib import Schema

# Define schema object to pass it to the TabularSequenceFeatures class
SCHEMA_PATH = 'schema_tutorial.pb'
schema = Schema().from_proto_text(SCHEMA_PATH)

# Create a sub-schema only with the selected features
schema = schema.select_by_name(['product_id-list_seq'])
```

<!-- #region id="04b7b077" -->
### Define Input block
Here we instantiate `TabularSequenceFeatures` from the feature schema and set `masking="mlm"` to use MLM as training method.
<!-- #endregion -->

```python id="db863122"
#Input 
sequence_length, d_model = 20, 192
# Define input module to process tabular input-features and to prepare masked inputs
inputs= tr.TabularSequenceFeatures.from_schema(
    schema,
    max_sequence_length=sequence_length,
    d_output=d_model,
    masking="mlm",
)
```

<!-- #region id="8a611eed" -->
We have inherited the original `XLNetConfig` class of HF transformers with some default arguments in the `build()` method. Here we use it to instantiate an XLNET model according to the arguments (`d_model`, `n_head`, etc.), defining the model architecture.

The `TransformerBlock` class supports HF Transformers for session-based and sequential-based recommendation models. `NextItemPredictionTask` is the class to support next item prediction task, encapsulating the corresponding heads and loss.
<!-- #endregion -->

```python id="d589d961" outputId="e676e74b-4659-413e-8253-d03a3e588435"
# Define XLNetConfig class and set default parameters for HF XLNet config  
transformer_config = tr.XLNetConfig.build(
    d_model=d_model, n_head=4, n_layer=2, total_seq_length=sequence_length
)
# Define the model block including: inputs, masking, projection and transformer block.
body = tr.SequentialBlock(
    inputs, tr.MLPBlock([192]), tr.TransformerBlock(transformer_config, masking=inputs.masking)
)

# Define the head for to next item prediction task 
head = tr.Head(
    body,
    tr.NextItemPredictionTask(weight_tying=True, hf_format=True, 
                              metrics=[NDCGAt(top_ks=[10, 20], labels_onehot=True),  
                                       RecallAt(top_ks=[10, 20], labels_onehot=True)]),
)

# Get the end-to-end Model class 
model = tr.Model(head)
```

<!-- #region id="f55e1bd0" -->
**Set training arguments**
<!-- #endregion -->

<!-- #region id="1cae8fe6" -->
Among the training arguments you can set the `data_loader_engine` to automatically instantiate the dataloader based on the schema, rather than instantiating the data loader manually like we did for the RNN example. The default value is `"nvtabular"` for optimized GPU-based data-loading. Optionally the PyarrowDataLoader (`"pyarrow"`) can also be used as a basic option, but it is slower and works only for small datasets, as the full data is loaded into CPU memory.
<!-- #endregion -->

```python id="848baadc" outputId="4b5b29b8-2358-47f5-df68-e1bb98ca090d"
from transformers4rec.config.trainer import T4RecTrainingArguments
from transformers4rec.torch import Trainer

#Set arguments for training 
training_args = T4RecTrainingArguments(
            output_dir="./tmp",
            max_sequence_length=20,
            data_loader_engine='nvtabular',
            num_train_epochs=3, 
            dataloader_drop_last=False,
            per_device_train_batch_size = 256,
            per_device_eval_batch_size = 32,
            gradient_accumulation_steps = 1,
            learning_rate=0.000666,
            report_to = [],
            logging_steps=200,
        )
```

<!-- #region id="fc4bd782" -->
**Instantiate the trainer**
<!-- #endregion -->

```python id="3192fa33"
# Instantiate the T4Rec Trainer, which manages training and evaluation
trainer = Trainer(
    model=model,
    args=training_args,
    schema=schema,
    compute_metrics=True,
)
```

<!-- #region id="b8b383d2" -->
Define the output folder of the processed parquet files
<!-- #endregion -->

```python id="c3b6ae2e"
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/workspace/data/sessions_by_day")
```

<!-- #region id="99663319" -->
Now, we do time-based fine-tuning the model by iteratively training and evaluating using a sliding time window, like we did for the RNN example.
<!-- #endregion -->

```python id="813c00c4" outputId="65b29789-087f-43cf-ddd2-19a3069e12a5"
%%time
start_time_window_index = 1
final_time_window_index = 4
for time_index in range(start_time_window_index, final_time_window_index):
    # Set data 
    time_index_train = time_index
    time_index_eval = time_index + 1
    train_paths = glob.glob(os.path.join(OUTPUT_DIR, f"{time_index_train}/train.parquet"))
    eval_paths = glob.glob(os.path.join(OUTPUT_DIR, f"{time_index_eval}/valid.parquet"))
    # Train on day related to time_index 
    print('*'*20)
    print("Launch training for day %s are:" %time_index)
    print('*'*20 + '\n')
    trainer.train_dataset_or_path = train_paths
    trainer.reset_lr_scheduler()
    trainer.train()
    trainer.state.global_step +=1
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

<!-- #region id="1cda4321" -->
Add eval accuracy metric results to the existing resuls.txt file.
<!-- #endregion -->

```python id="cc3203b5"
with open("results.txt", 'a') as f:
    f.write('\n')
    f.write('XLNet-MLM accuracy results:')
    f.write('\n')
    for key, value in  model.compute_metrics().items(): 
        f.write('%s:%s\n' % (key, value.item()))
```

<!-- #region id="9fbf1aff" -->
**Restart the kernel to free our GPU memory**
<!-- #endregion -->

```python id="b3e2ceae" outputId="98a348c2-0e95-4d04-9130-8f2bdd84fd2d"
import IPython
app = IPython.Application.instance()
app.kernel.do_shutdown(True)
```

<!-- #region id="54e6c70f" -->
At this stage if the kernel does not restart automatically, we expect you to manually restart the kernel to free GPU memory so that you can move on to the next session-based model training with XLNet using side information.
<!-- #endregion -->

<!-- #region id="3ee17c91" -->
### Train XLNET with Side Information for Next Item Prediction
<!-- #endregion -->

<!-- #region id="4f82f8a9" -->
It is a common practice in RecSys to leverage additional tabular features of item (product) metadata and user context, providing the model more
information for meaningful predictions. With that motivation, in this section, we will use additional features to train our XLNET architecture. We already checked our `schema.pb`, saw that it includes features and their tags. Now it is time to use these additional features that we created earlier.
<!-- #endregion -->

```python id="3aefd29c"
import os
import glob
import nvtabular as nvt

import torch 
import transformers4rec.torch as tr

from transformers4rec.torch.ranking_metric import NDCGAt, RecallAt
```

```python id="1389b745"
# Define categorical and continuous columns to fed to training model
x_cat_names = ['product_id-list_seq', 'category_id-list_seq', 'brand-list_seq']
x_cont_names = ['product_recency_days_log_norm-list_seq', 'et_dayofweek_sin-list_seq', 'et_dayofweek_cos-list_seq', 
                'price_log_norm-list_seq', 'relative_price_to_avg_categ_id-list_seq']

from merlin_standard_lib import Schema

# Define schema object to pass it to the TabularSequenceFeatures class
SCHEMA_PATH ='schema_tutorial.pb'
schema = Schema().from_proto_text(SCHEMA_PATH)
schema = schema.select_by_name(x_cat_names + x_cont_names)
```

<!-- #region id="e9c1bcff" -->
Here we set `aggregation="concat"`, so that all categorical and continuous features are concatenated to form an interaction representation.
<!-- #endregion -->

```python id="1929fc23" outputId="7ff55bf1-cbb6-4758-ccc7-11dfc4f41886"
# Define input block
sequence_length, d_model = 20, 192
# Define input module to process tabular input-features and to prepare masked inputs
inputs= tr.TabularSequenceFeatures.from_schema(
    schema,
    max_sequence_length=sequence_length,
    aggregation="concat",
    d_output=d_model,
    masking="mlm",
)

# Define XLNetConfig class and set default parameters for HF XLNet config  
transformer_config = tr.XLNetConfig.build(
    d_model=d_model, n_head=4, n_layer=2, total_seq_length=sequence_length
)
# Define the model block including: inputs, masking, projection and transformer block.
body = tr.SequentialBlock(
    inputs, tr.MLPBlock([192]), tr.TransformerBlock(transformer_config, masking=inputs.masking)
)

# Define the head related to next item prediction task 
head = tr.Head(
    body,
    tr.NextItemPredictionTask(weight_tying=True, hf_format=True, 
                                     metrics=[NDCGAt(top_ks=[10, 20], labels_onehot=True),  
                                              RecallAt(top_ks=[10, 20], labels_onehot=True)]),
)

# Get the end-to-end Model class 
model = tr.Model(head)
```

<!-- #region id="b72e5aef" -->
### Training and Evaluation
<!-- #endregion -->

```python id="103b3a99"
from transformers4rec.config.trainer import T4RecTrainingArguments
from transformers4rec.torch import Trainer

#Set arguments for training 
training_args = T4RecTrainingArguments(
            output_dir="./tmp",
            max_sequence_length=20,
            data_loader_engine='nvtabular',
            num_train_epochs=3, 
            dataloader_drop_last=False,
            per_device_train_batch_size = 256,
            per_device_eval_batch_size = 32,
            gradient_accumulation_steps = 1,
            learning_rate=0.000666,
            report_to = [],
            logging_steps=200,
)
```

```python id="35398982"
# Instantiate the T4Rec Trainer, which manages training and evaluation
trainer = Trainer(
    model=model,
    args=training_args,
    schema=schema,
    compute_metrics=True,
)
```

<!-- #region id="d72bd497" -->
Define the output folder of the processed parquet files
<!-- #endregion -->

```python id="254c744a"
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/workspace/data/sessions_by_day")
```

```python id="ec86fada" outputId="05b466fa-f091-409b-c790-5d805b22d10c"
%%time
start_time_window_index = 1
final_time_window_index = 4
for time_index in range(start_time_window_index, final_time_window_index):
    # Set data 
    time_index_train = time_index
    time_index_eval = time_index + 1
    train_paths = glob.glob(os.path.join(OUTPUT_DIR, f"{time_index_train}/train.parquet"))
    eval_paths = glob.glob(os.path.join(OUTPUT_DIR, f"{time_index_eval}/valid.parquet"))
    # Train on day related to time_index 
    print('*'*20)
    print("Launch training for day %s are:" %time_index)
    print('*'*20 + '\n')
    trainer.train_dataset_or_path = train_paths
    trainer.reset_lr_scheduler()
    trainer.train()
    trainer.state.global_step +=1
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

<!-- #region id="e6978124" -->
Add XLNet-MLM with side information accuracy results to the `results.txt`
<!-- #endregion -->

```python id="d7998f38"
with open("results.txt", 'a') as f:
    f.write('\n')
    f.write('XLNet-MLM with side information accuracy results:')
    f.write('\n')
    for key, value in  model.compute_metrics().items(): 
        f.write('%s %s\n' % (key, value.item()))
```

<!-- #region id="0657768a" -->
After model training and evaluation is completed we can save our trained model in the next section. 
<!-- #endregion -->

<!-- #region id="43341500" -->
## Exporting the preprocessing workflow and model for deployment to Triton server
<!-- #endregion -->

<!-- #region id="70927a43" -->
Load the preproc workflow that we saved in the ETL section.
<!-- #endregion -->

```python id="8fe450c4"
import nvtabular as nvt

# define data path about where to get our data
INPUT_DATA_DIR = os.environ.get("INPUT_DATA_DIR", "/workspace/data/")
workflow_path = os.path.join(INPUT_DATA_DIR, 'workflow_etl')
workflow = nvt.Workflow.load(workflow_path)
```

```python id="4af4431c" outputId="6041333c-6693-4bc2-c57f-dec5e47b0045"
# dictionary representing max sequence length for the sequential (list) columns
sparse_features_max = {
    fname: sequence_length
    for fname in x_cat_names + x_cont_names
}

sparse_features_max
```

<!-- #region id="e6839f8c" -->
It is time to export the proc workflow and model in the format required by Triton Inference Server, by using the NVTabular’s `export_pytorch_ensemble()` function.
<!-- #endregion -->

```python id="d2b9e84c"
from nvtabular.inference.triton import export_pytorch_ensemble
export_pytorch_ensemble(
    model,
    workflow,
    sparse_max=sparse_features_max,
    name= "t4r_pytorch",
    model_path= "/workspace/models",
    label_columns =[],
)
```

<!-- #region id="cbfb8213" -->
Before we move on to the next section, let's print out our results.txt file. 
<!-- #endregion -->

```python id="54a92609" outputId="ed0a674a-eddb-4ee2-a2ef-9badac3ccd86"
!cat results.txt
```

<!-- #region id="146af29e" -->
Let's plot bar charts to visualize and compare the accuracy results using `visuals` util function.
<!-- #endregion -->

```python id="c66e3f39" outputId="3c244ec6-52fb-4ab4-dc49-3ba01794f025"
from visuals import create_bar_chart
create_bar_chart('results.txt')
```

<!-- #region id="77d36393" -->
## Triton for Recommender Systems
<!-- #endregion -->

<!-- #region id="172f9e15" -->
NVIDIA [Triton Inference Server (TIS)](https://github.com/triton-inference-server/server) simplifies the deployment of AI models at scale in production. The Triton Inference Server allows us to deploy and serve our model for inference. It supports a number of different machine learning frameworks such as TensorFlow and PyTorch.

The last step of machine learning (ML)/deep learning (DL) pipeline is to deploy the ETL workflow and saved model to production. In the production setting, we want to transform the input data as done during training (ETL). We need to apply the same mean/std for continuous features and use the same categorical mapping to convert the categories to continuous integer before we use the DL model for a prediction. Therefore, we deploy the NVTabular workflow with the PyTorch model as an ensemble model to Triton Inference. The ensemble model guarantees that the same transformation is applied to the raw inputs.
<!-- #endregion -->

<!-- #region id="ae645daa" -->
<img src='https://s3.us-west-2.amazonaws.com/secure.notion-static.com/55f95680-f556-45b4-93b3-a3f5eaf715f7/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20211011%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20211011T114013Z&X-Amz-Expires=86400&X-Amz-Signature=3229c6eb93dda4a077bae72a876dcaa5da46602d1cdc28193ae42c540ff67944&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22'>
<!-- #endregion -->

<!-- #region id="6f85f45d" -->
**Objectives:**

Learn how to deploy a model to Triton
1. Deploy saved NVTabular and PyTorch models to Triton Inference Server
2. Sent requests for predictions
<!-- #endregion -->

<!-- #region id="43dc14a8" -->
## Pull and start Inference docker container
<!-- #endregion -->

<!-- #region id="f22667d0" -->
At this point, before connecing to the Triton Server, we launch the inference docker container and then load the exported ensemble `t4r_pytorch` to the inference server. This is done with the scripts below:

**Launch the docker container:**
```
docker run -it --gpus device=0 -p 8000:8000 -p 8001:8001 -p 8002:8002 -v <path_to_saved_models>:/workspace/models/ nvcr.io/nvidia/merlin/merlin-inference:21.09
```

This script will mount your local model-repository folder that includes your saved models from the previous cell to `/workspace/models` directory in the merlin-inference docker container.

**Start triton server**
After you started the merlin-inference container, you can start triton server with the command below. You need to provide correct path of the models folder.
```
tritonserver --model-repository=<path_to_models> --model-control-mode=explicit
```
Note: The model-repository path for our example is `/workspace/models`. The models haven't been loaded, yet. Below, we will request the Triton server to load the saved ensemble model.
<!-- #endregion -->

<!-- #region id="07499907" -->
## Deploy PyTorch and NVTabular Model to Triton Inference Server
<!-- #endregion -->

<!-- #region id="6b61ed1a" -->
Our Triton server has already been launched and is ready to make requests. Remember we already exported the saved PyTorch model in the previous section, and generated the config files for Triton Inference Server.
<!-- #endregion -->

```python id="6645e40e"
# Import dependencies
import os
from time import time

import argparse
import numpy as np
import pandas as pd
import sys
import cudf
```

<!-- #region id="72c90e93" -->
## Review exported files
<!-- #endregion -->

<!-- #region id="6b8b7a4c" -->
Triton expects a specific directory structure for our models as the following format:
<!-- #endregion -->

<!-- #region id="d34dcb28" -->
```
<model-name>/
[config.pbtxt]
<version-name>/
  [model.savedmodel]/
    <pytorch_saved_model_files>/
      ...
```
<!-- #endregion -->

<!-- #region id="9d7d3156" -->
Let's check out our model repository layout. You can install tree library with `apt-get install tree`, and then run `!tree /workspace/models/` to print out the model repository layout as below:

```
├── t4r_pytorch
│   ├── 1
│   └── config.pbtxt
├── t4r_pytorch_nvt
│   ├── 1
│   │   ├── model.py
│   │   ├── __pycache__
│   │   │   └── model.cpython-38.pyc
│   │   └── workflow
│   │       ├── categories
│   │       │   ├── cat_stats.category_id.parquet
│   │       │   ├── unique.brand.parquet
│   │       │   ├── unique.category_code.parquet
│   │       │   ├── unique.category_id.parquet
│   │       │   ├── unique.event_type.parquet
│   │       │   ├── unique.product_id.parquet
│   │       │   ├── unique.user_id.parquet
│   │       │   └── unique.user_session.parquet
│   │       ├── metadata.json
│   │       └── workflow.pkl
│   └── config.pbtxt
└── t4r_pytorch_pt
    ├── 1
    │   ├── model_info.json
    │   ├── model.pkl
    │   ├── model.pth
    │   ├── model.py
    │   └── __pycache__
    │       └── model.cpython-38.pyc
    └── config.pbtxt
```
<!-- #endregion -->

<!-- #region id="79b1036b" -->
Triton needs a [config file](https://github.com/triton-inference-server/server/blob/main/docs/model_configuration.md) to understand how to interpret the model. Let's look at the generated config file. It defines the input columns with datatype and dimensions and the output layer. Manually creating this config file can be complicated and NVTabular generates it with the `export_pytorch_ensemble()` function, which we used in the previous section.

The [config file](https://github.com/triton-inference-server/server/blob/main/docs/model_configuration.md) needs the following information:
* `name`: The name of our model. Must be the same name as the parent folder.
* `platform`: The type of framework serving the model.
* `input`: The input our model expects.
  * `name`: Should correspond with the model input name.
  * `data_type`: Should correspond to the input's data type.
  * `dims`: The dimensions of the *request* for the input. For models that support input and output tensors with variable-size dimensions, those dimensions can be listed as -1 in the input and output configuration.
* `output`: The output parameters of our model.
  * `name`: Should correspond with the model output name.
  * `data_type`: Should correspond to the output's data type.
  * `dims`: The dimensions of the output.
<!-- #endregion -->

<!-- #region id="0adbbbf2" -->
## Loading Model
<!-- #endregion -->

<!-- #region id="276a0e31" -->
Next, let's build a client to connect to our server. The `InferenceServerClient` object is what we'll be using to talk to Triton.
<!-- #endregion -->

```python id="b1e8ac0f" outputId="d860e2b2-893c-4209-c1cd-3cee1a4f3164"
import tritonhttpclient

try:
    triton_client = tritonhttpclient.InferenceServerClient(url="localhost:8000", verbose=True)
    print("client created.")
except Exception as e:
    print("channel creation failed: " + str(e))
triton_client.is_server_live()
```

```python id="f61231d8" outputId="c3489c5e-d903-4fc0-f011-4e1d8bb92bad"
triton_client.get_model_repository_index()
```

<!-- #region id="3d091905" -->
We load the ensemble model
<!-- #endregion -->

```python id="260d063d" outputId="ce8e8aa4-3388-4bd7-a588-c4ff17203289"
model_name = "t4r_pytorch"
triton_client.load_model(model_name=model_name)
```

<!-- #region id="26345f7d" -->
If all models are loaded succesfully, you should be seeing successfully loaded status next to each model name on your terminal.
<!-- #endregion -->

<!-- #region id="fe1debc7" -->
## Sent Requests for Predictions
<!-- #endregion -->

<!-- #region id="2b2cc71a" -->
Load raw data for inference: We select the first 50 interactions and filter out sessions with less than 2 interactions. For this tutorial, just as an example we use the `Oct-2019` dataset that we used for model training.
<!-- #endregion -->

```python id="5309a22e"
INPUT_DATA_DIR = os.environ.get("INPUT_DATA_DIR", "/workspace/data/")
df= cudf.read_parquet(os.path.join(INPUT_DATA_DIR, 'Oct-2019.parquet'))
df=df.sort_values('event_time_ts')
batch = df.iloc[:50,:]
```

```python id="592aad96"
sessions_to_use = batch.user_session.value_counts()
filtered_batch = batch[batch.user_session.isin(sessions_to_use[sessions_to_use.values>1].index.values)]
```

```python id="b860b31c" outputId="d5d6d2af-a47d-472f-8a1e-7a53b6e8cf3e"
filtered_batch.head()
```

```python id="b40c3922" outputId="97faa931-c7bb-485d-c8cc-d24e283032d9"
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

<!-- #region id="2f2e07dc" -->
## Visualise top-k predictions
<!-- #endregion -->

```python id="45c64075" outputId="9fad59ae-4234-41f9-c15d-fecdb3b5b565"
from transformers4rec.torch.utils.examples_utils import visualize_response
visualize_response(filtered_batch, response, top_k=5, session_col='user_session')
```

<!-- #region id="1619777f" -->
As you see we first got prediction results (logits) from the trained model head, and then by using a handy util function `visualize_response` we extracted top-k encoded item-ids from logits. Basically, we  generated recommended items for a given session.

This is the end of the tutorial. You successfully ...
1. performed feature engineering with NVTabular
2. trained transformer architecture based session-based recommendation models with Transformers4Rec 
3. deployed a trained model to Triton Inference Server, sent request and got responses from the server.
<!-- #endregion -->

<!-- #region id="6224a7fe" -->
## Unload models and shut down the kernel
<!-- #endregion -->

```python id="f481d47f" outputId="f86accf1-6809-4b7d-8719-55149a125f16"
triton_client.unload_model(model_name="t4r_pytorch")
triton_client.unload_model(model_name="t4r_pytorch_nvt")
triton_client.unload_model(model_name="t4r_pytorch_pt")
```

<!-- #region id="496bdf04" -->
## References
<!-- #endregion -->

<!-- #region id="66f2a8e5" -->
[1] Malte Ludewig and Dietmar Jannach. 2018. Evaluation of session-based recommendation algorithms. User Modeling and User-Adapted Interaction 28, 4-5 (2018), 331–390.<br>
[2] Balázs Hidasi and Alexandros Karatzoglou. 2018. Recurrent neural networks with top-k gains for session-based recommendations. In Proceedings of the 27th ACMinternational conference on information and knowledge management. 843–852.<br>
[3] Fei Sun, Jun Liu, Jian Wu, Changhua Pei, Xiao Lin, Wenwu Ou, and Peng Jiang. 2019. BERT4Rec: Sequential recommendation with bidirectional encoder representations from transformer. In Proceedings of the 28th ACM international conference on information and knowledge management. 1441–1450.
[4] Shiming Sun, Yuanhe Tang, Zemei Dai, and Fu Zhou. 2019. Self-attention network for session-based recommendation with streaming data input. IEEE Access 7 (2019), 110499–110509.  
[5] Kyunghyun Cho, Bart Van Merriënboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. 2014. Learning phrase representations using RNN encoder-decoder for statistical machine translation. arXiv preprint arXiv:1406.1078 (2014).  
[6] Vaswani, A., et al. (2017). Attention is all you need. In Advances in neural information processing systems (pp. 5998-6008).  
[7] Lample, Guillaume, and Alexis Conneau. "Cross-lingual language model pretraining." arXiv preprint arXiv:1901.07291  
[8] Gabriel De Souza P. Moreira, et al. (2021). Transformers4Rec: Bridging the Gap between NLP and Sequential / Session-Based Recommendation. RecSys'21.  
[9] Understanding XLNet, BorealisAI. Online available: https://www.borealisai.com/en/blog/understanding-xlnet/  
[10] Yang, Zhilin, et al. "Xlnet: Generalized autoregressive pretraining for language understanding." Advances in neural information processing systems 32 (2019).
<!-- #endregion -->
