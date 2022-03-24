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

```python id="5bWKlMcWimzN"
project_name = "coveo"; branch = "master"; account = "sparsh-ai"
```

```python colab={"base_uri": "https://localhost:8080/"} id="dUob1Ec1imzO" executionInfo={"status": "ok", "timestamp": 1626548046184, "user_tz": -330, "elapsed": 1506, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="83e1cd4c-8726-4c12-fe1c-ffea19698b8b"
!cp /content/drive/MyDrive/mykeys.py /content
import mykeys
!rm /content/mykeys.py
path = "/content/" + project_name; 
!mkdir "{path}"
%cd "{path}"
import sys; sys.path.append(path)
!git config --global user.email "nb@recohut.com"
!git config --global user.name  "colab-sparsh"
!git init
!git remote add origin https://"{mykeys.git_token}":x-oauth-basic@github.com/"{account}"/"{project_name}".git
!git pull origin "{branch}"
```

```python id="4gYTGO_0CDro"
!rm -r /content/coveo/SIGIR-ecom-data-challenge
```

```python id="dtEMhgJVimzP"
!git status
!git add . && git commit -m 'commit' && git push origin "{branch}"
```

```python id="p2hPMGIzjIpt"
DATA_DIR = "/content/coveo/data/sample"
```

```python id="ElgoxEJs3WQa"
import os
import pandas as pd
import json
from pathlib import Path
import pickle
import csv
from datetime import datetime
```

```python id="zda9Ey9s35r3"
%reload_ext google.colab.data_table
```

```python id="LoPw8Ec08i5h"
BROWSING_FILE_PATH = os.path.join(DATA_DIR, 'train', 'browsing_train.csv')
SEARCH_TRAIN_PATH = os.path.join(DATA_DIR, 'train', 'search_train.csv')
SKU_2_CONTENT_PATH = os.path.join(DATA_DIR, 'train', 'sku_to_content.csv')
```

```python id="035hgcVE4fDV"
# This script uses a subset of session IDs to sub-sample the training data.
TRAIN_RATIO = 1 #0.01

TRAIN_PATH = Path(os.path.join(DATA_DIR,'train/browsing_train.csv'))
TEST_PATH = Path(os.path.join(DATA_DIR,'test/rec_test_sample.json'))

PREPARED_FOLDER = Path(os.path.join(DATA_DIR,'prepared'))
PREPARED_FOLDER.mkdir(parents=True, exist_ok=True)

PREPARED_TRAIN_PATH = PREPARED_FOLDER / 'sigir_train_full.txt'
PREPARED_TEST_PATH = PREPARED_FOLDER / 'sigir_test.txt'
ITEM_LABEL_ENCODING_MAP_PATH = PREPARED_FOLDER / 'item_label_encoding.p'

SessionId = 'SessionId'
ItemId = 'ItemId'
Time = 'Time'
```

```python id="H-Ab46-f8lI3"
def get_rows(file_path: str, print_limit: int = 2):
    """
    Util function reading the csv file and printing the first few lines out for visual debugging.

    :param file_path: local path to the csv file
    :param print_limit: specifies how many rows to print out in the console for debug
    :return: list of dictionaries, one per each row in the file
    """
    rows = []
    print("\n============== {}".format(file_path))
    with open(file_path) as csvfile:
        reader = csv.DictReader(csvfile)
        for idx, row in enumerate(reader):
            # print out first few lines
            if idx < print_limit:
                print(row)
            rows.append(row)

    return rows
```

```python id="s6u_ZfTp8IIX"
def get_descriptive_stats(
        browsing_train_path : str,
        search_train_path: str,
        sku_2_content_path: str
):
    """
    Simple function showing how to read the main training files, print out some
    example rows, and producing the counts found in the Data Challenge paper.

    We use basic python library commands, optimizing for clarity, not performance.

    :param browsing_train_path: path to the file containing the browsing interactions
    :param search_train_path: path to the file containing the search interactions
    :param sku_2_content_path: path to the file containing the product meta-data
    :return:
    """
    print("Starting our counts at {}".format(datetime.utcnow()))
    # first, just read in the csv files and display some rows
    browsing_events = get_rows(browsing_train_path)
    print("# {} browsing events".format(len(browsing_events)))
    search_events = get_rows(search_train_path)
    print("# {} search events".format(len(search_events)))
    sku_mapping = get_rows(sku_2_content_path)
    print("# {} products".format(len(sku_mapping)))
    # now do some counts
    print("\n\n=============== COUNTS ===============")
    print("# {} of distinct SKUs with interactions".format(
        len(set([r['product_sku_hash'] for r in browsing_events if r['product_sku_hash']]))))
    print("# {} of add-to-cart events".format(sum(1 for r in browsing_events if r['product_action'] == 'add')))
    print("# {} of purchase events".format(sum(1 for r in browsing_events if r['product_action'] == 'purchase')))
    print("# {} of total interactions".format(sum(1 for r in browsing_events if r['product_action'])))
    print("# {} of distinct sessions".format(
        len(set([r['session_id_hash'] for r in browsing_events if r['session_id_hash']]))))
    # now run some tests
    print("\n\n*************** TESTS ***************")
    for r in browsing_events:
        assert len(r['session_id_hash']) == 64
        assert not r['product_sku_hash'] or len(r['product_sku_hash']) == 64
    for p in sku_mapping:
        assert not p['price_bucket'] or float(p['price_bucket']) <= 10
    # say goodbye
    print("All done at {}: see you, space cowboy!".format(datetime.utcnow()))

    return
```

```python colab={"base_uri": "https://localhost:8080/"} id="CXJBc9sW8ggZ" executionInfo={"status": "ok", "timestamp": 1626550229710, "user_tz": -330, "elapsed": 14, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="f1040960-43e0-4a0b-9143-260efbd06cd3"
get_descriptive_stats(BROWSING_FILE_PATH, SEARCH_TRAIN_PATH, SKU_2_CONTENT_PATH)
```

<!-- #region id="53dtgqgA3mYA" -->
## Data Preparation
<!-- #endregion -->

<!-- #region id="MKTPswdp3qoG" -->
Load the recommendation train and test data files, and convert them into "prepared" data that can be directly consumed by session_rec repo to run baselines for the recommendation task.
<!-- #endregion -->

```python id="LEt3RSBl4pPI"
def label_encode_series(series: pd.Series):
    """
    Applies label encoding to a Pandas series and returns the encoded series,
    together with the label to index and index to label mappings.
    :param series: input Pandas series
    :return: Pandas series with label encoding, label-integer mapping and integer-label mapping.
    """
    labels = set(series.unique())
    label_to_index = {l: idx for idx, l in enumerate(labels)}
    index_to_label = {v: k for k, v in label_to_index.items()}
    return series.map(label_to_index), label_to_index, index_to_label
```

<!-- #region id="ilIlVVkN44Q7" -->
### Generate 'prepared' train
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 640} id="6RnKTsWY49j3" executionInfo={"status": "ok", "timestamp": 1626550547589, "user_tz": -330, "elapsed": 505, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="6340510c-9f51-45bb-a72c-056f90f81b99"
train_data_df = pd.read_csv(TRAIN_PATH)
train_data_df
```

```python id="alOIM9Dw4swx"
session_ids = set(train_data_df['session_id_hash'].unique())
train_cutoff = int(len(session_ids) * TRAIN_RATIO)
train_session_ids = list(session_ids)[:train_cutoff]
train_data_df = train_data_df[train_data_df['session_id_hash'].isin(train_session_ids)]
```

<!-- #region id="aPEMkhta51FY" -->
Filter out
* `remove from cart` events to avoid feeding them to session_rec as positive signals
* rows with null product_sku_hash
* sessions with only one action
<!-- #endregion -->

```python id="2TOANGtZ5-d1"
train_data_df = train_data_df[train_data_df['product_action'] != 'remove']
train_data_df = train_data_df.dropna(subset=['product_sku_hash'])
train_data_df['session_len_count'] = train_data_df.groupby('session_id_hash')['session_id_hash'].transform('count')
train_data_df = train_data_df[train_data_df['session_len_count'] >= 2]
```

```python colab={"base_uri": "https://localhost:8080/", "height": 299} id="1Rs6n5wH6geg" executionInfo={"status": "ok", "timestamp": 1626550551706, "user_tz": -330, "elapsed": 9, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="39a5422c-adbb-4f31-f43e-cc866f670fbb"
# sort by session, then timestamp
train_data_df = train_data_df.sort_values(['session_id_hash', 'server_timestamp_epoch_ms'], ascending=True)

# Encode labels with integers
item_id_int_series, item_label_to_index, item_index_to_label = label_encode_series(train_data_df.product_sku_hash)
item_string_set = set(item_label_to_index.keys())

# Add tokenized session ID, tokenized item ID, and seconds since epoch time.
train_data_df[SessionId] = train_data_df.groupby([train_data_df.session_id_hash]).grouper.group_info[0]
train_data_df[Time] = train_data_df.server_timestamp_epoch_ms / 1000
train_data_df[ItemId] = item_id_int_series

# Get final dataframe
final_train_df = train_data_df[[SessionId, ItemId, Time]]

final_train_df
```

```python colab={"base_uri": "https://localhost:8080/"} id="AsvwDyQf6uro" executionInfo={"status": "ok", "timestamp": 1626550553471, "user_tz": -330, "elapsed": 7, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="5fa96b44-5020-47be-b4e3-28c325641531"
# Generate CSV and label encoder
final_train_df.to_csv(PREPARED_TRAIN_PATH, sep='\t', index=False)
pickle.dump(item_index_to_label, ITEM_LABEL_ENCODING_MAP_PATH.open(mode='wb'))
print("Done generating 'prepared' for training")
```

<!-- #region id="1XiOQhiz64bA" -->
### Generate 'prepared' test
<!-- #endregion -->

```python id="RFtfrUPw7L7K"
with TEST_PATH.open() as f:
    test_data = json.load(f)
```

```python id="3epbZmP87Kta"
test_output = []
dummy_label_item = next(iter(item_string_set))
for idx, query_label in enumerate(test_data):
    query = query_label['query']

    # Here, since we must give a label in order to use session_rec repo, we use a dummy item
    # from train data to fill the label for testing data generation;
    # it is fine since we do not need to obtain the real metrics from the session_rec run,
    # the final metrics will be generated using our own scripts)
    first_label_item = dummy_label_item

    cleaned_query_events = []
    for q in query:
        # Skip if it is an action with no item id or it is remove-from-cart or it is unseen from train data
        if q['product_sku_hash'] is None \
                or q['product_action'] == 'remove' \
                or q['product_sku_hash'] not in item_string_set:
            continue
        q_session_id = idx
        q_session_item = q['product_sku_hash']
        q_session_time = q['server_timestamp_epoch_ms'] / 1000
        q_event = {SessionId: q_session_id,
                   ItemId: q_session_item,
                   Time: q_session_time}
        cleaned_query_events.append(q_event)
    if len(cleaned_query_events) > 0:
        # sessionId is mapped to idx integer
        l_event = {SessionId: idx,
                   ItemId: first_label_item,
                   Time: cleaned_query_events[-1][Time] + 1}
        test_output = test_output + cleaned_query_events
        test_output = test_output + [l_event]
```

```python colab={"base_uri": "https://localhost:8080/", "height": 89} id="uuDsjw843YED" executionInfo={"status": "ok", "timestamp": 1626549962858, "user_tz": -330, "elapsed": 397, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="5824cd41-1141-4e23-ba78-f48b3655485b"
# Create the final dataframe and apply label encoding.
test_output_df = pd.DataFrame(test_output)
test_output_df['ItemId'] = test_output_df.ItemId.map(item_label_to_index)
test_output_df
```

```python id="BtkdmZrS7T67"
test_output_df.to_csv(PREPARED_TEST_PATH, sep='\t', index=False)
print("Done generating 'prepared' for testing")
```

```python colab={"base_uri": "https://localhost:8080/"} id="wE8Ej97C_FaA" executionInfo={"status": "ok", "timestamp": 1626551056225, "user_tz": -330, "elapsed": 425, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="b7a554f7-db13-4e58-bb6a-56560794ad74"
%%writefile config.yml
type: single # single|window, maybe add opt
key: models #added to the csv names
evaluation: evaluation_last #evaluation|evaluation_last|evaluation_multiple
data:
  name: sigir #added in the end of the csv names
  folder: data/prepared/
  prefix: sigir

results:
  folder: results/last/sigir/

metrics:
- class: accuracy.HitRate
  length: [3,5,10,15,20]
- class: accuracy.MRR
  length: [3,5,10,15,20]
- class: accuracy_multiple.NDCG
  length: [3,5,10,15,20]
- class: coverage.Coverage
  length: [20]
- class: popularity.Popularity
  length: [20]
- class: saver.Saver
  length: [20]

algorithms:
- class: gru4rec.gru4rec.GRU4Rec
  params: { loss: 'top1-max', final_act: 'linear', dropout_p_hidden: 0.1, learning_rate: 0.08, momentum: 0.1, constrained_embedding: False }
  key: gru4rec
- class: STAMP.model.STAMP.Seq2SeqAttNN
  params: { init_lr: 0.003, n_epochs: 10, decay_rate: 0.4}
  key: stamp
- class: narm.narm.NARM
  params: { epochs: 20, lr: 0.007, hidden_units: 100, factors: 100 }
  key: narm
```

```python colab={"base_uri": "https://localhost:8080/"} id="uad_l0me9CNh" executionInfo={"status": "ok", "timestamp": 1626550839933, "user_tz": -330, "elapsed": 1184, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="7dfdf532-7965-41fd-8195-958c29720d5f"
!git clone https://github.com/rn5l/session-rec.git
```

```python id="IXNwtWm3_CNP"
!cp -r $PREPARED_FOLDER /content/coveo/session-rec/data
```

```python id="fJW2gu0h_toq"
!cp /content/coveo/config.yml /content/coveo/session-rec/conf
```

```python id="raBHuoxEAduU"
!pip install scikit-optimize
!pip install python-telegram-bot
```

```python colab={"base_uri": "https://localhost:8080/"} id="AlRTlcU7AGF9" executionInfo={"status": "ok", "timestamp": 1626551507757, "user_tz": -330, "elapsed": 1849, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="52e4142e-28cd-4797-89c0-6380f7c5c6b4"
%cd session-rec
!python run_config.py conf/example_sigir.yml
```

```python id="VQeSH6NYAmgL"
!mkdir -p /content/coveo/session-rec/results/last/sigir
```

```python id="EK_fv-grBP8k"

```
