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

```python colab={"base_uri": "https://localhost:8080/"} id="Dc9-z1zdmo58" executionInfo={"status": "ok", "timestamp": 1621259731358, "user_tz": -330, "elapsed": 1671, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="6eb57058-b089-4414-9eab-70b86e0ac3a6"
%%writefile requirements.txt

appnope==0.1.0
backcall==0.1.0
boto==2.49.0
boto3==1.10.26
botocore==1.13.26
certifi==2019.9.11
chardet==3.0.4
cPython==0.0.5
decorator==4.4.1
docutils==0.15.2
gensim==3.8.1
idna==2.8
ipykernel==5.1.3
ipython==7.9.0
ipython-genutils==0.2.0
jedi==0.15.1
jmespath==0.9.4
joblib==0.14.0
jupyter-client==5.3.4
jupyter-core==4.6.1
networkx==2.4
node2vec==0.3.1
numpy==1.17.3
pandas==0.25.3
parso==0.5.1
pexpect==4.7.0
pickleshare==0.7.5
prompt-toolkit==2.0.10
ptyprocess==0.6.0
Pygments==2.7.4
pymongo==3.9.0
python-dateutil==2.8.0
pytz==2019.3
pyzmq==18.1.0
requests==2.22.0
s3transfer==0.2.1
scikit-learn==0.21.3
scipy==1.3.1
six==1.13.0
smart-open==1.9.0
tornado==6.0.3
tqdm==4.39.0
traitlets==4.3.3
urllib3==1.25.8
wcwidth==0.1.7
```

```python id="jciwxPd1nBjs" executionInfo={"status": "ok", "timestamp": 1621260529850, "user_tz": -330, "elapsed": 1396, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
# !pip install -r requirements.txt
```

```python colab={"base_uri": "https://localhost:8080/"} id="NXLEPpZ0nEPa" executionInfo={"status": "ok", "timestamp": 1621260901458, "user_tz": -330, "elapsed": 122716, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="bb9f6913-acad-42f4-95c7-c8b7ed845ad5"
# !wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Books.json.gz
!wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Electronics.json.gz
```

```python id="XReOSIxcrDxa" executionInfo={"status": "ok", "timestamp": 1621260901461, "user_tz": -330, "elapsed": 5240, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(message)s')

# create console handler and set level to info
ch = logging.StreamHandler()
ch.setFormatter(formatter)
ch.setLevel(logging.INFO)

# add ch to logger
logger.addHandler(ch)
```

```python id="jdT1qs3ECzIH" executionInfo={"status": "ok", "timestamp": 1621267004294, "user_tz": -330, "elapsed": 1063, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
DATA_PATH = './'
MODEL_PATH = './'
```

```python id="NsD_2NoxrgdT" executionInfo={"status": "ok", "timestamp": 1621260963156, "user_tz": -330, "elapsed": 1608, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
"""
Parses the raw json data into csv file for faster loading into pd.DataFrame.
"""
import argparse
import csv
import gzip
from typing import List

import numpy as np
import pandas as pd
from pandas.api.types import is_object_dtype


def parse(path: str):
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)


def parse_json_to_df(path: str) -> pd.DataFrame:
    i = 0
    df_dict = {}
    for d in parse(path):
        df_dict[i] = d
        i += 1
        if i % 10000 == 0:
            logger.info('Rows processed: {:,}'.format(i))

    df = pd.DataFrame.from_dict(df_dict, orient='index')

    # Lowercase
    df['related'] = df['related'].astype(str)
    df['categories'] = df['categories'].astype(str)
    df['salesRank'] = df['salesRank'].astype(str)
    df = lowercase_df(df)

    return df


# Lowercase Functions
def lowercase_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lowercase characters from all columns in a dataframe.

    Args:
        df: Pandas dataframe

    Returns:
        Lowercased dataframe
    """
    df = df.copy()
    for col in df.columns:
        if is_object_dtype(df[col]):
            df = lowercase_cols(df, [col])
    return df


def lowercase_cols(df: pd.DataFrame, colnames: List[str]) -> pd.DataFrame:
    """
    Lowercase characters from specified columns in a dataframe

    Args:
        df: Pandas dataframe
        colnames (List): Names of columns to be lowercased

    Returns: Lowercased dataframe

    """
    df = df.copy()
    for col in colnames:
        assert df[col].dtype != np.float64 and df[col].dtype != np.int64, \
            'Trying to lowercase a non-string column: {}'.format(col)
        df[col] = df[col].str.lower()
    return df


def parse_json_to_csv(read_path: str, write_path: str) -> None:
    """
    Note: This assumes that the first json in the path has all the keys, which could be WRONG

    Args:
        read_path:
        write_path:

    Returns:

    """
    csv_writer = csv.writer(open(write_path, 'w'))
    i = 0
    for d in parse(read_path):
        if i == 0:
            header = d.keys()
            csv_writer.writerow(header)

        csv_writer.writerow(d.values().lower())
        i += 1
        if i % 10000 == 0:
            logger.info('Rows processed: {:,}'.format(i))

    logger.info('Csv saved to {}'.format(write_path))
```

```python id="esr-okAqrwaV"
read_path = 'meta_Electronics.json.gz'
write_path = 'electronics.csv'

df = parse_json_to_df(read_path)
df.to_csv(write_path, index=False)
logger.info('Csv saved to {}'.format(write_path))
```

```python id="iLJs0G_ssPvH" executionInfo={"status": "ok", "timestamp": 1621261352645, "user_tz": -330, "elapsed": 1832, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
"""
Parses item to item relationships in 'related' field and explodes it such that each relationship is a single row.
"""
import argparse

import numpy as np
import pandas as pd


def get_also_bought_count(related):
    try:
        return len(related['also_bought'])
    except KeyError:
        return -1


def explode_on_related(df: pd.DataFrame, relationship: str) -> pd.DataFrame:
    # Filter on relationship
    df = df[df['related'].apply(lambda x: relationship in x.keys())].copy()

    # Get value (list) from relationship dict
    df['related'] = df['related'].apply(lambda x: x[relationship])

    # Explode efficiently using numpy
    vals = df['related'].values.tolist()
    lens = [len(val_list) for val_list in vals]
    vals_array = np.repeat(df['asin'], lens)
    exploded_df = pd.DataFrame(np.column_stack((vals_array, np.concatenate(vals))), columns=df.columns)

    # Add relationship
    exploded_df['relationship'] = relationship
    logger.info('Exploded for relationship: {}'.format(relationship))

    return exploded_df


def get_node_relationship(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a dataframe of products and their relationships (e.g., bought together, also bought, also viewed)
    """
    # Keep only rows with related data
    df = df[~df['related'].isnull()].copy()
    logger.info('DF shape after dropping empty related: {}'.format(df.shape))

    df = df[~df['title'].isnull()].copy()
    logger.info('DF shape after dropping empty title: {}'.format(df.shape))
    df = df[['asin', 'related']].copy()

    # Evaluate related str into dict
    df['related'] = df['related'].apply(eval)
    logger.info('Completed eval on "related" string')

    # Exclude products where also bought relationships less than 2
    df['also_bought_count'] = df['related'].apply(get_also_bought_count)
    df = df[df['also_bought_count'] >= 2].copy()
    logger.info('DF shape after dropping products with <2 edges: {}'.format(df.shape))
    df.drop(columns='also_bought_count', inplace=True)

    # Explode columns
    bought_together_df = explode_on_related(df, relationship='bought_together')
    also_bought_df = explode_on_related(df, relationship='also_bought')
    also_viewed_df = explode_on_related(df, relationship='also_viewed')

    # Concatenate df
    combined_df = pd.concat([bought_together_df, also_bought_df, also_viewed_df], axis=0)
    logger.info('Distribution of relationships: \n{}'.format(combined_df['relationship'].value_counts()))

    return combined_df
```

```python colab={"base_uri": "https://localhost:8080/"} id="9D3VjMvttPcw" executionInfo={"status": "ok", "timestamp": 1621261499103, "user_tz": -330, "elapsed": 45578, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="c13f176a-cc6c-4210-b16d-acb1f959101c"
read_path = 'electronics.csv'
write_path = 'electronics_relationships.csv'

df = pd.read_csv(read_path, error_bad_lines=False, warn_bad_lines=True,
                  dtype={'asin': 'str', 'title': 'str', 'brand': 'str'})
logger.info('DF shape: {}'.format(df.shape))

exploded_df = get_node_relationship(df)

exploded_df.to_csv(write_path, index=False)
logger.info('Csv saved to {}'.format(write_path))
```

```python id="oFGzDh8AtohS" executionInfo={"status": "ok", "timestamp": 1621264879182, "user_tz": -330, "elapsed": 1307, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
"""
Parses out the metadata from the original csv.
"""
import argparse

import numpy as np
import pandas as pd


def get_category_lvl(category_list: list, lvl=0) -> str:
    try:
        return category_list[lvl]
    except IndexError:
        return 'NA_VALUE'


def get_categories(df: pd.DataFrame) -> pd.DataFrame:
    df['category_lvl_1'] = df['categories'].apply(get_category_lvl, args=(0,))
    df['category_lvl_2'] = df['categories'].apply(get_category_lvl, args=(1,))
    df['category_lvl_3'] = df['categories'].apply(get_category_lvl, args=(2,))
    df['category_lvl_4'] = df['categories'].apply(get_category_lvl, args=(3,))
    logger.info('Categories lvl 1 - 4 prepared')

    return df


def get_meta(df: pd.DataFrame) -> pd.DataFrame:
    # Update to reflect if relationship exist
    df['related'] = np.where(df['related'].isnull(), 0, 1)

    # Prep categories
    df['categories'] = df['categories'].apply(eval)
    df['categories'] = df['categories'].apply(lambda x: x[0])  # Get first category only
    df = get_categories(df)

    # Prep title and description
    # TODO: Add cleaning of title and description

    return df
```

```python colab={"base_uri": "https://localhost:8080/"} id="phfhqFwu6qWy" executionInfo={"status": "ok", "timestamp": 1621264952956, "user_tz": -330, "elapsed": 35841, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="5fe03cac-f026-46d0-929b-71f8c6f4ebdd"
read_path = 'electronics.csv'
write_path = 'electronics_meta.csv'

META_COLS = ['asin', 'categories', 'title', 'description', 'price', 'brand', 'related']
df = pd.read_csv(read_path, error_bad_lines=False, warn_bad_lines=True,
                  dtype={'asin': 'str', 'title': 'str', 'brand': 'str'},
                  usecols=META_COLS)
logger.info('DF shape: {}'.format(df.shape))

meta_df = get_meta(df)

meta_df.to_csv(write_path, index=False)
logger.info('Csv saved to {}'.format(write_path))
```

```python id="g5w7gOJs62Hz" executionInfo={"status": "ok", "timestamp": 1621264971134, "user_tz": -330, "elapsed": 1151, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
"""
Converts edge relationships (e.g., bought together, also bought) to numeric weights between two nodes.
"""
import argparse

import numpy as np
import pandas as pd

relationship_weights = {'bought_together': 1.2,
                        'also_bought': 1.0,
                        'also_viewed': 0.5}


def create_product_pair(df, col_list):
    pairs = df[col_list].values
    pairs.sort(axis=1)
    df['product_pair'] = ['|'.join(arr) for arr in pairs]

    return df


def split_product_pair(product_pair):
    result = product_pair.split('|')
    return result[0], result[1]


def get_relationship_weights(df, relationship_weights):
    df['weight'] = 0
    for relationship, weight in relationship_weights.items():
        df.loc[df['relationship'] == relationship, 'weight'] += weight

    return df


def get_edges(df):
    """
    Returns a dataframe of products and the weights of the edges between them.

    Args:
        df:

    Returns:

    """
    logger.info('Relationship distribution: \n{}'.format(df['relationship'].value_counts()))

    df = create_product_pair(df, col_list=['asin', 'related'])
    logger.info('Product pairs created')

    df = get_relationship_weights(df, relationship_weights)
    logger.info('Relationship weights updated')

    # Aggregate to remove duplicates
    logger.info('Original no. of edges: {:,}'.format(df.shape[0]))
    df = df.groupby('product_pair').agg({'weight': 'sum'}).reset_index()
    logger.info('Deduplicated no. of edges: {:,}'.format(df.shape[0]))

    # Save edge list
    df['product1'], df['product2'] = zip(*df['product_pair'].apply(split_product_pair))

    df = df[['product1', 'product2', 'weight', 'product_pair']]
    return df
```

```python colab={"base_uri": "https://localhost:8080/"} id="Ni9BF1-47AO1" executionInfo={"status": "ok", "timestamp": 1621265121099, "user_tz": -330, "elapsed": 54194, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="d97bc3ed-be27-4a73-bce6-172889e7af88"
read_path = 'electronics_relationships.csv'
write_path = 'electronics_edges.csv'
sample_size = None

df = pd.read_csv(read_path, error_bad_lines=False, warn_bad_lines=True,
                  dtype={'asin': 'str', 'related': 'str'})
logger.info('DF shape: {}'.format(df.shape))

# Sample for development efficiency
if sample_size:
    sample_idx = np.random.choice(df.shape[0], size=sample_size, replace=False)
    df = df.iloc[sample_idx]

df = get_edges(df)

df.to_csv(write_path, index=False)
logger.info('Csv saved to {}'.format(write_path))
```

```python id="eJt4SWNS7arz" executionInfo={"status": "ok", "timestamp": 1621265270704, "user_tz": -330, "elapsed": 1761, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
"""
Splits all ground truth edges into train and validation set, with some constraints
- The validation set should only contain edges where both products are in the train set

For the validation set, negative samples are created by randomly selecting a pair of nodes and creating a negative edge.
- From these samples, we exclude valid edges from either the train or validation set.
"""
import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def train_val_split(df, n_val_samples: int, filter_out_unseen: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if filter_out_unseen:
        # First split to get some test samples
        train, val = train_test_split(df, test_size=int(1.1 * n_val_samples), random_state=42)  # Need slightly more
        logger.info('Train shape: {}, val shape: {}'.format(train.shape, val.shape))

        # Get set of products in train
        train_product_set = set(train['product1']).union(set(train['product2']))
        logger.info('No. of unique products in train: {:,}'.format(len(train_product_set)))

        # Only keep val where both products are in train product set
        val = val[(val['product1'].isin(train_product_set)) & (val['product2'].isin(train_product_set))]
        logger.info('Updated val shape: {}'.format(val.shape))

        # Split again to only get n_val_samples
        val = val.iloc[:n_val_samples].copy()
        logger.info('Final val shape: {}'.format(val.shape))

        # Get train set
        train = df[~df.index.isin(set(val.index))].copy()
        logger.info('Final train shape: {}'.format(train.shape))

    else:
        # First split to get some test samples
        train, val = train_test_split(df, test_size=int(n_val_samples), random_state=42)
        logger.info('Train shape: {}, val shape: {}'.format(train.shape, val.shape))

    return train, val


def get_sample(item_array, n_iter=None, sample_size=2):
    np.random.seed(42)
    n = len(item_array)

    # find the index we last sampled from
    start_idx = (n_iter * sample_size) % n
    if (start_idx + sample_size >= n) or (start_idx <= sample_size):
        # shuffle array if we have reached the end and repeat again
        np.random.shuffle(item_array)

    return item_array[start_idx:start_idx + sample_size]


def collect_samples(item_array, sample_size, n_samples):
    samples = []

    for i in range(0, n_samples):
        if i % 1000000 == 0:
            logger.info('Neg sample: {:,}'.format(i))

        sample = get_sample(item_array, n_iter=i, sample_size=sample_size)
        samples.append(sample)

    return samples


def create_negative_edges(df, val, n_val_samples):
    # Get set of valid product edges (across both train and val)
    valid_product_pairs = set(df['product_pair'])
    logger.info('No. of valid product pairs: {:,}'.format(len(valid_product_pairs)))

    # Get set of products in val (to generate edges)
    val_product_arr = np.array(list(set(val['product1']).union(set(val['product2']))))
    logger.info('No. of unique products in val: {:,}'.format(len(val_product_arr)))

    # Create negative samples
    neg_samples = collect_samples(val_product_arr, sample_size=2, n_samples=int(1.1 * n_val_samples))
    neg_samples_df = pd.DataFrame(neg_samples, columns=['product1', 'product2'])
    neg_samples_df.dropna(inplace=True)
    neg_samples_df = create_product_pair(neg_samples_df, col_list=['product1', 'product2'])
    logger.info('No. of negative samples: {:,}'.format(neg_samples_df.shape[0]))

    # Exclude neg samples that are valid pairs
    neg_samples_df = neg_samples_df[~neg_samples_df['product_pair'].isin(valid_product_pairs)].copy()
    logger.info('Updated no. of negative samples: {:,}'.format(neg_samples_df.shape[0]))

    # Only keep no. of val samples required
    neg_samples_df = neg_samples_df.iloc[:n_val_samples].copy()
    logger.info('Final no. of negative samples: {:,}'.format(neg_samples_df.shape[0]))

    return neg_samples_df


def combine_val_and_neg_edges(val, neg_samples):
    neg_samples['edge'] = 0
    val['edge'] = 1

    VAL_COLS = ['product1', 'product2', 'edge']
    neg = neg_samples[VAL_COLS].copy()
    val = val[VAL_COLS].copy()
    logger.info('Val shape: {}, Neg edges shape: {}, Ratio: {}'.format(val.shape, neg.shape,
                                                                       val.shape[0] / (val.shape[0] + neg.shape[0])))

    val = pd.concat([val, neg])
    logger.info('Final val shape: {}'.format(val.shape))

    return val


def get_train_and_val(df, val_prop: float):
    """
    Splits into training and validation set, where validation set has 50% negative edges

    Args:
        df:
        val_prop:

    Returns:

    """
    n_val_samples = int(val_prop * df.shape[0])
    logger.info('Eventual required val samples (proportion: {}): {:,}'.format(val_prop, n_val_samples))

    train, val = train_val_split(df, n_val_samples)
    logger.info('Ratio of train to val: {:,}:{:,} ({:.2f})'.format(train.shape[0], val.shape[0],
                                                                   val.shape[0] / (train.shape[0] + val.shape[0])))

    neg_samples = create_negative_edges(df, val, n_val_samples)

    val = combine_val_and_neg_edges(val, neg_samples)
    train = train[['product1', 'product2', 'weight']].copy()

    return train, val
```

```python colab={"base_uri": "https://localhost:8080/"} id="AeTMzQpL8MBJ" executionInfo={"status": "ok", "timestamp": 1621265517058, "user_tz": -330, "elapsed": 51707, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="2cfbe8f9-b831-4418-8572-a89bd4014ca5"
read_path = 'electronics_edges.csv'
val_prop = 0.33
DATA_PATH = './'

df = pd.read_csv(read_path, error_bad_lines=False, warn_bad_lines=True,
                  dtype={'product1': 'str', 'product2': 'str'})
logger.info('DF shape: {}'.format(df.shape))

train, val = get_train_and_val(df, val_prop=val_prop)

# Save to train, val, and train edgelist
input_filename = Path(read_path).resolve().stem
train.to_csv('{}/{}_train.csv'.format(DATA_PATH, input_filename), index=False)
logger.info('Train saved as: {}/{}_train.csv'.format(DATA_PATH, input_filename))
val.to_csv('{}/{}_val.csv'.format(DATA_PATH, input_filename), index=False)
logger.info('Val saved as: {}/{}_val.csv'.format(DATA_PATH, input_filename))

train.to_csv('{}/{}_train.edgelist'.format(DATA_PATH, input_filename), sep=' ', index=False, header=False)
logger.info('Train edgelist saved as: {}/{}_train.edgelist'.format(DATA_PATH, input_filename))
```

```python id="43kCApVM9Tp-" executionInfo={"status": "ok", "timestamp": 1621265570443, "user_tz": -330, "elapsed": 1615, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import gzip
import pickle
from typing import Any


def save_model(model: Any, model_path: str) -> None:
    """
    Saves model in gzip format

    Args:
        model: Model to be saved
        model_path: Path to save model to

    Returns:
        (None)
    """
    with gzip.open(model_path, "wb") as f:
        pickle.dump(model, f)

    logger.info('Model saved to {}'.format(model_path))


def load_model(model_path: str) -> Any:
    """
    Loads model from gzip format

    Args:
        model_path: Path to load model from

    Returns:

    """
    with gzip.open(model_path, 'rb') as f:
        model = pickle.load(f)

    logger.info('Model loaded from: {}'.format(model_path))
    return model
```

```python id="diyc1J3E8jic" executionInfo={"status": "ok", "timestamp": 1621265594519, "user_tz": -330, "elapsed": 1763, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
"""
Builds a graph from the edges (training set) and performs random walk sampling from the graph
- Currently returns 10 samples of sequence length 10 for each node (this is a parameter in create_random_walk_samples)
"""
import argparse
import random

import networkx
import numpy as np
import scipy as sp


def load_network(edgelist_path):
    graph = networkx.read_weighted_edgelist(edgelist_path)
    logger.info('No of nodes ({:,}) and edges ({:,})'.format(graph.number_of_nodes(), graph.number_of_edges()))

    # Get dictionary mapping of integer to nodes
    node_dict = {i: key for i, key in enumerate(graph.nodes.keys())}

    return graph, node_dict


def create_transition_matrix(graph):
    """
    https://stackoverflow.com/questions/37311651/get-node-list-from-random-walk-in-networkx
    https://stackoverflow.com/questions/15330380/probability-to-visit-nodes-in-a-random-walk-on-graph

    Args:
        graph:

    Returns:

    """
    adjacency_mat = networkx.adj_matrix(graph)
    logger.info('Adjacency matrix shape: {}'.format(adjacency_mat.shape))
    graph = None

    degree_vector = sp.sparse.csr_matrix(1 / np.sum(adjacency_mat, axis=0))

    transition_matrix = adjacency_mat.multiply(degree_vector).T  # Need to transpose so each row probability sum to 1
    logger.info('Transition matrix shape: {}'.format(transition_matrix.shape))

    return transition_matrix


def create_transition_dict(transition_matrix):
    transition_dict = {}
    rows, cols = transition_matrix.nonzero()

    # Create dictionary of transition product and probabilities for each product
    prev_row = -1
    for row, col in zip(rows, cols):
        if row != prev_row:
            transition_dict.setdefault(row, {})
            transition_dict[row].setdefault('product', [])
            transition_dict[row].setdefault('probability', [])

        transition_dict[row]['product'].append(col)
        transition_dict[row]['probability'].append(transition_matrix[row, col])
        prev_row = row

    return transition_dict


def create_random_walk_samples(node_dict, transition_dict, samples_per_node=10, sequence_len=10):
    random.seed(42)
    n_nodes = len(node_dict)

    sample_array = np.zeros((n_nodes * samples_per_node, sequence_len), dtype=int)
    logger.info('Sample array shape: {}'.format(sample_array.shape))

    # For each node
    for node_idx in range(n_nodes):

        if node_idx % 100000 == 0:
            logger.info('Getting samples for node: {:,}/{:,}'.format(node_idx, n_nodes))

        # For each sample
        for sample_idx in range(samples_per_node):
            node = node_idx

            # For each event in sequence
            for seq_idx in range(sequence_len):
                sample_array[node_idx * samples_per_node + sample_idx, seq_idx] = node
                node = random.choices(population=transition_dict[node]['product'],
                                      weights=transition_dict[node]['probability'], k=1)[0]

    return sample_array


def get_samples(edgelist_path):
    graph, node_dict = load_network(edgelist_path)
    logger.info('Network loaded')

    transition_matrix = create_transition_matrix(graph)
    logger.info('Transition matrix created')
    graph = None

    transition_dict = create_transition_dict(transition_matrix)
    logger.info('Transition dict created')
    transition_matrix = None

    sample_array = create_random_walk_samples(node_dict, transition_dict)
    logger.info('Random walk samples created')

    # Convert array of nodeIDs back to product IDs
    sample_array = np.vectorize(node_dict.get)(sample_array)
    logger.info('Converted back to product IDs')

    return sample_array, node_dict, transition_dict
```

```python colab={"base_uri": "https://localhost:8080/"} id="pY90t-yL9Lvs" executionInfo={"status": "ok", "timestamp": 1621266755833, "user_tz": -330, "elapsed": 1050906, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="cfbb8ec0-9c61-490f-9d83-da696193cff7"
read_path = 'electronics_edges_train.edgelist'
write_path = 'electronics_sequences.npy'
DATA_PATH = './'
graph_name = 'electronics'

sample_array, node_dict, transition_dict = get_samples(read_path)

np.save(write_path, sample_array)
logger.info('Sample array saved to {}'.format(write_path))
sample_array = None

save_model(node_dict, '{}/{}_node_dict.tar.gz'.format(DATA_PATH, graph_name))
node_dict = None

save_model(transition_dict, '{}/{}_transition_dict.tar.gz'.format(DATA_PATH, graph_name))
transition_dict = None
```

```python id="BkV2-T_m-0oh" executionInfo={"status": "ok", "timestamp": 1621266758545, "user_tz": -330, "elapsed": 1326, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
# import networkx as nx
# from node2vec import Node2Vec


# def train_embeddings(edgelist_path, embedding_path):
#     # Create path
#     graph = nx.read_weighted_edgelist(edgelist_path)
#     logger.info('Graph created!')
#     assert graph.get_edge_data('0000013714', '0005064295')['weight'] == 3.2, 'Expected edge weight of 3.2'

#     # Precomput probabilities and generate walks
#     node2vec = Node2Vec(graph, dimensions=128, walk_length=30, num_walks=10, workers=10, temp_folder=DATA_PATH)
#     logger.info('Computed probabilities and generated walks')
#     graph = None  # We don't need graph anymore since probabilities have been precomputed

#     # Embed nodes
#     model = node2vec.fit(window=5, min_count=1, batch_words=128)
#     logger.info('Nodes embedded')

#     # Save embeddings for later use
#     model.wv.save_word2vec_format(embedding_path)
#     logger.info('Embedding saved')
```

```python id="8szSCTe792dh" executionInfo={"status": "ok", "timestamp": 1621266791034, "user_tz": -330, "elapsed": 1126, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
# # Slow and requires a lot of ram
# read_path = 'electronics_edges_train.edgelist'
# write_path = 'electronics_embeddings.kv'

# train_embeddings(read_path, write_path)
```

```python id="6awlYhfj_fH5" executionInfo={"status": "ok", "timestamp": 1621267209725, "user_tz": -330, "elapsed": 1081, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import argparse
import datetime

import numpy as np
from gensim.models import Word2Vec


def load_sequences(sequence_path):
    """
    Expects a numpy array at sequence_path

    Args:
        sequence_path:

    Returns:

    """
    sequences = np.load(sequence_path)
    logger.info('Sequences shape: {}'.format(sequences.shape))

    # Convert sequences to string and list of list
    sequences = sequences.astype(str).tolist()

    return sequences


def train_embeddings(sequences, workers, dimension=128, window=5, min_count=1, negative=5, epochs=3, seed=42):
    # Logging specific to gensim training
    import logging
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    # Initialize model
    model = Word2Vec(sequences, workers=workers,
                     size=dimension, window=window, min_count=min_count, negative=negative, seed=seed)
    logger.info('Model initialized')

    # Train model (No need to retrain model as initialization includes training)
    # model.train(sequences, total_examples=len(sequences), epochs=epochs)
    # logger.info('Model trained!')

    return model


def save_model(model):
    # Save model and keyedvectors
    current_datetime = datetime.datetime.now().strftime('%Y-%m-%d-%H%M')
    model.save('{}/gensim-w2v-{}.model'.format(MODEL_PATH, current_datetime))
    model.wv.save('{}/gensim-w2v-{}.kv'.format(MODEL_PATH, current_datetime))
```

```python colab={"base_uri": "https://localhost:8080/"} id="DEST3-U0_D31" executionInfo={"status": "ok", "timestamp": 1621267850183, "user_tz": -330, "elapsed": 624219, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="9a3b9923-c226-4224-89af-a51ae5880747"
# Works fine with multiprocess
read_path = 'electronics_sequences.npy'
n_workers = 6

sequences = load_sequences(read_path)

start_time = datetime.datetime.now()
model = train_embeddings(sequences, workers=n_workers)
end_time = datetime.datetime.now()
time_diff = round((end_time - start_time).total_seconds() / 60, 2)
logger.info('Total time taken: {:,} minutes'.format(time_diff))
save_model(model)
```

```python id="VXCA9dNuC5Rr" executionInfo={"status": "ok", "timestamp": 1621267882087, "user_tz": -330, "elapsed": 999, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def get_product_id(mapping):
    def func(x):
        return mapping.get(x, -1)
    return func
```

```python id="n2E6qFaGC5L8" executionInfo={"status": "ok", "timestamp": 1621267893037, "user_tz": -330, "elapsed": 1214, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
from collections import OrderedDict

import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve


def plot_auc(label, score, title):
    precision, recall, thresholds = precision_recall_curve(label, score)
    plt.figure(figsize=(15, 5))
    plt.grid()
    plt.plot(thresholds, precision[1:], color='r', label='Precision')
    plt.plot(thresholds, recall[1:], color='b', label='Recall')
    plt.gca().invert_xaxis()
    plt.legend(loc='lower right')

    plt.xlabel('Threshold (0.00 - 1.00)')
    plt.ylabel('Precision / Recall')
    _ = plt.title(title)


def plot_roc(label, score, title):
    fpr, tpr, roc_thresholds = roc_curve(label, score)
    plt.figure(figsize=(5, 5))
    plt.grid()
    plt.plot(fpr, tpr, color='b')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    _ = plt.title(title)


def plot_tradeoff(label, score, title):
    precision, recall, thresholds = precision_recall_curve(label, score)
    plt.figure(figsize=(5, 5))
    plt.grid()
    plt.step(recall, precision, color='b', label='Precision-Recall Trade-off')
    plt.fill_between(recall, precision, alpha=0.1, color='b')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    _ = plt.title(title)


def plot_metrics(df, ylim=None):
    plt.figure(figsize=(15, 5))
    plt.grid()
    plt.plot(df.index, df['auc'], label='AUC-ROC', color='black')

    # Plot learning rate resets
    lr_reset_batch = df[df['batches'] == df['batches'].max()]
    for idx in lr_reset_batch.index:
        plt.vlines(idx, df['auc'].min(), 1, label='LR reset (per epoch)',
                   linestyles='--', colors='grey')

    # PLot legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    _ = plt.legend(by_label.values(), by_label.keys(), loc='lower right')

    # Tidy axis
    if ylim:
        plt.ylim(ylim)
    else:
        plt.ylim(df['auc'].min() * 1.2, 0.96)
    plt.xlim(0, df.index.max())
    plt.ylabel('AUC-ROC', size=12)
    plt.xlabel('Batches (over 5 epochs)', size=12)
    _ = plt.title('AUC-ROC on sample val set over 5 epochs', size=15)
```

<!-- #region id="AZa4_7QyJVOG" -->
https://github.com/eugeneyan/recsys-nlp-graph
<!-- #endregion -->
