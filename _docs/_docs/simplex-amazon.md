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

<!-- #region id="TdOWFerSmTjg" -->
# SimpleX MF Model on Amazon Books Dataset
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="eifpuuZSML3n" executionInfo={"status": "ok", "timestamp": 1633111763888, "user_tz": -330, "elapsed": 3689, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="c58f0ae8-8260-4a4e-9ed7-684be81d1d9b"
# !wget -q --show-progress https://github.com/kuandeng/LightGCN/raw/master/Data/amazon-book/user_list.txt
# !wget -q --show-progress https://github.com/kuandeng/LightGCN/raw/master/Data/amazon-book/item_list.txt
!wget -q --show-progress https://github.com/kuandeng/LightGCN/raw/master/Data/amazon-book/train.txt
!wget -q --show-progress https://github.com/kuandeng/LightGCN/raw/master/Data/amazon-book/test.txt
```

```python colab={"base_uri": "https://localhost:8080/"} id="lc-wtoK6M3ls" executionInfo={"status": "ok", "timestamp": 1633111763889, "user_tz": -330, "elapsed": 19, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="f5ff47eb-d61f-47b9-912b-6e04ab07ab00"
!head train.txt
```

```python colab={"base_uri": "https://localhost:8080/"} id="kVW7oiy7Lz0-" executionInfo={"status": "ok", "timestamp": 1633111868181, "user_tz": -330, "elapsed": 104297, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="83eec655-e121-44c2-edd1-01007242cf1a"
import pandas as pd

user_history_dict = dict()
train_data = []
item_corpus = []
corpus_index = dict()

with open("train.txt", "r") as fid:
    for line in fid:
        splits = line.strip().split()
        user_id = splits[0]
        items = splits[1:]
        user_history_dict[user_id] = items
        for item in items:
            if item not in corpus_index:
                corpus_index[item] = len(corpus_index)
                item_corpus.append([corpus_index[item], item])
            history = user_history_dict[user_id].copy()
            history.remove(item)
            train_data.append([user_id, corpus_index[item], 1, user_id, "^".join(history)])
train = pd.DataFrame(train_data, columns=["query_index", "corpus_index", "label", "user_id", "user_history"])
print("train samples:", len(train))
train.to_csv("train.csv", index=False)

test_data = []
with open("test.txt", "r") as fid:
    for line in fid:
        splits = line.strip().split()
        user_id = splits[0]
        items = splits[1:]
        for item in items:
            if item not in corpus_index:
                corpus_index[item] = len(corpus_index)
                item_corpus.append([corpus_index[item], item])
            history = user_history_dict[user_id].copy()
            test_data.append([user_id, corpus_index[item], 1, user_id, "^".join(history)])
test = pd.DataFrame(test_data, columns=["query_index", "corpus_index", "label", "user_id", "user_history"])
print("test samples:", len(test))
test.to_csv("test.csv", index=False)

corpus = pd.DataFrame(item_corpus, columns=["corpus_index", "item_id"])
print("number of items:", len(item_corpus))
corpus = corpus.set_index("corpus_index")
corpus.to_csv("item_corpus.csv", index=False)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 424} id="g33jjCqRM_0c" executionInfo={"status": "ok", "timestamp": 1633111912375, "user_tz": -330, "elapsed": 44213, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="aeb79b5d-ebcb-4f10-f8ca-bd8ee40b7f6c"
pd.read_csv('train.csv')
```

<!-- #region id="OUPTOzEvb5Oz" -->
## Imports
<!-- #endregion -->

```python id="81z_rTuiOG_i"
import os
import logging
import logging.config
import yaml
import glob
import json
from collections import OrderedDict
import itertools
import subprocess
import yaml
import os
import numpy as np
import time
import glob
import hashlib
import sys
import os
import numpy as np
import torch
from torch import nn
import random
import h5py
import os
import logging
import numpy as np
import multiprocessing as mp
import gc
import glob
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
import torch
import logging
from collections import defaultdict
import multiprocessing as mp
import os
import pickle
import shutil
import numpy as np
import logging
import heapq
import itertools
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
```

```python id="Aiv2nz0xbdxh"
# add this line to avoid weird characters in yaml files
yaml.Dumper.ignore_aliases = lambda *args : True
```

<!-- #region id="of-8kWisb6yt" -->
## Utils
<!-- #endregion -->

```python id="5Xy_Bp00bHE5"
def load_config(config_dir, experiment_id):
    params = dict()
    model_configs = glob.glob(os.path.join(config_dir, 'model_config.yaml'))
    if not model_configs:
        model_configs = glob.glob(os.path.join(config_dir, 'model_config/*.yaml'))
    if not model_configs:
        raise RuntimeError('config_dir={} is not valid!'.format(config_dir))
    found_params = dict()
    for config in model_configs:
        with open(config, 'r') as cfg:
            config_dict = yaml.load(cfg)
            if 'Base' in config_dict:
                found_params['Base'] = config_dict['Base']
            if experiment_id in config_dict:
                found_params[experiment_id] = config_dict[experiment_id]
        if len(found_params) == 2:
            break
    # Update base setting first so that values can be overrided when conflict 
    # with experiment_id settings
    params.update(found_params.get('Base', {}))
    params.update(found_params.get(experiment_id))
    if 'dataset_id' not in params:
        raise RuntimeError('experiment_id={} is not valid in config.'.format(experiment_id))
    params['model_id'] = experiment_id
    dataset_id = params['dataset_id']
    dataset_configs = glob.glob(os.path.join(config_dir, 'dataset_config.yaml'))
    if not dataset_configs:
        dataset_configs = glob.glob(os.path.join(config_dir, 'dataset_config/*.yaml'))
    for config in dataset_configs:
        with open(config, 'r') as cfg:
            config_dict = yaml.load(cfg)
            if dataset_id in config_dict:
                params.update(config_dict[dataset_id])
                break
    return params
```

```python id="PRXvxD4ibMgP"
def set_logger(params):
    dataset_id = params['dataset_id']
    model_id = params['model_id']
    log_dir = os.path.join(params['model_root'], dataset_id)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, model_id + '.log')

    # logs will not show in the file without the two lines.
    for handler in logging.root.handlers[:]: 
        logging.root.removeHandler(handler)
        
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s P%(process)d %(levelname)s %(message)s',
                        handlers=[logging.FileHandler(log_file, mode='w'),
                                  logging.StreamHandler()])
```

```python id="p-gYWbvWbMdT"
def print_to_json(data, sort_keys=True):
    new_data = dict((k, str(v)) for k, v in data.items())
    if sort_keys:
        new_data = OrderedDict(sorted(new_data.items(), key=lambda x: x[0]))
    return json.dumps(new_data, indent=4)
```

```python id="6fwIkK3-bQ-p"
def print_to_list(data):
    return ' - '.join('{}: {:.6f}'.format(k, v) for k, v in data.items())
```

```python id="fQkxgzYsbRl3"
class Monitor(object):
    def __init__(self, kv):
        if isinstance(kv, str):
            kv = {kv: 1}
        self.kv_pairs = kv

    def get_value(self, logs):
        value = 0
        for k, v in self.kv_pairs.items():
            value += logs[k] * v
        return value
```

```python id="VWOeOCAjbkWR"
def seed_everything(seed=1029):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def set_device(gpu=-1):
    if gpu >= 0 and torch.cuda.is_available():
        device = torch.device("cuda: " + str(gpu))
    else:
        device = torch.device("cpu")   
    return device

def set_optimizer(optimizer):
    if isinstance(optimizer, str):
        if optimizer.lower() == "adam":
            optimizer = "Adam"
        elif optimizer.lower() == "rmsprop":
            optimizer = "RMSprop"
        elif optimizer.lower() == "sgd":
            optimizer = "SGD"
        return getattr(torch.optim, optimizer)

def set_loss(loss):
    if isinstance(loss, str):
        if loss in ["bce", "binary_crossentropy", "binary_cross_entropy"]:
            loss = "binary_cross_entropy"
        else:
            raise NotImplementedError("loss={} is not supported.".format(loss))
    return loss

def set_regularizer(reg):
    reg_pair = [] # of tuples (p_norm, weight)
    if isinstance(reg, float):
        reg_pair.append((2, reg))
    elif isinstance(reg, str):
        try:
            if reg.startswith("l1(") or reg.startswith("l2("):
                reg_pair.append((int(reg[1]), float(reg.rstrip(")").split("(")[-1])))
            elif reg.startswith("l1_l2"):
                l1_reg, l2_reg = reg.rstrip(")").split("(")[-1].split(",")
                reg_pair.append((1, float(l1_reg)))
                reg_pair.append((2, float(l2_reg)))
            else:
                raise NotImplementedError
        except:
            raise NotImplementedError("regularizer={} is not supported.".format(reg))
    return reg_pair

def set_activation(activation):
    if isinstance(activation, str):
        if activation.lower() == "relu":
            return nn.ReLU()
        elif activation.lower() == "sigmoid":
            return nn.Sigmoid()
        elif activation.lower() == "tanh":
            return nn.Tanh()
        else:
            return getattr(nn, activation)()
    else:
        return activation

def pad_sequences(sequences, maxlen=None, dtype='int32',
                  padding='pre', truncating='pre', value=0.):
    """ Pads sequences (list of list) to the ndarray of same length 
        This is an equivalent implementation of tf.keras.preprocessing.sequence.pad_sequences
        for Pytorch
    """

    assert padding in ["pre", "post"], "Invalid padding={}.".format(padding)
    assert truncating in ["pre", "post"], "Invalid truncating={}.".format(truncating)
    
    if maxlen is None:
        maxlen = max(len(x) for x in sequences)
    arr = np.full((len(sequences), maxlen), value, dtype=dtype)
    for idx, x in enumerate(sequences):
        if len(x) == 0:
            continue  # empty list
        if truncating == 'pre':
            trunc = x[-maxlen:]
        else:
            trunc = x[:maxlen]
        trunc = np.asarray(trunc, dtype=dtype)

        if padding == 'pre':
            arr[idx, -len(trunc):] = trunc
        else:
            arr[idx, :len(trunc)] = trunc
    return arr
```

<!-- #region id="4TRMkZBcdtnQ" -->
## Preprocess
<!-- #endregion -->

```python id="1y4TLLdWdu4K"
from collections import Counter
import itertools
import numpy as np
import pandas as pd
import h5py
import pickle
import os
import sklearn.preprocessing as sklearn_preprocess
from tensorflow.keras.preprocessing.sequence import pad_sequences
```

```python id="daAMtV0Wd3hO"
class Tokenizer(object):
    def __init__(self, topk_words=None, na_value=None, min_freq=1, splitter=None, 
                 lower=False, oov_token=0, max_len=0, padding="pre"):
        self._topk_words = topk_words
        self._na_value = na_value
        self._min_freq = min_freq
        self._lower = lower
        self._splitter = splitter
        self.oov_token = oov_token # use 0 for __OOV__
        self.vocab = dict()
        self.vocab_size = 0 # include oov and padding
        self.max_len = max_len
        self.padding = padding
        self.use_padding = None

    def fit(self, texts, use_padding=False):
        self.use_padding = use_padding
        word_counts = Counter()
        if self._splitter is not None: # for sequence
            max_len = 0
            for text in texts:
                if not pd.isnull(text):
                    text_split = text.split(self._splitter)
                    max_len = max(max_len, len(text_split))
                    for text in text_split:
                        word_counts[text] += 1
            if self.max_len == 0:
                self.max_len = max_len # use pre-set max_len otherwise
        else:
            tokens = list(texts)
            word_counts = Counter(tokens)
        self.build_vocab(word_counts)

    def build_vocab(self, word_counts):
        # sort to guarantee the determinism of index order
        word_counts = sorted(word_counts.items(), key=lambda x: (-x[1], x[0]))
        words = []
        for token, count in word_counts:
            if count >= self._min_freq:
                if self._na_value is None or token != self._na_value:
                    words.append(token.lower() if self._lower else token)
        if self._topk_words:
            words = words[0:self._topk_words]
        self.vocab = dict((token, idx) for idx, token in enumerate(words, 1 + self.oov_token))
        self.vocab["__OOV__"] = self.oov_token
        if self.use_padding:
            self.vocab["__PAD__"] = len(words) + self.oov_token + 1 # use the last index for __PAD__
        self.vocab_size = len(self.vocab) + self.oov_token

    def encode_category(self, categories):
        category_indices = [self.vocab.get(x, self.oov_token) for x in categories]
        return np.array(category_indices)

    def encode_sequence(self, texts):
        sequence_list = []
        for text in texts:
            if pd.isnull(text) or text == '':
                sequence_list.append([])
            else:
                sequence_list.append([self.vocab.get(x, self.oov_token) for x in text.split(self._splitter)])
        sequence_list = pad_sequences(sequence_list, maxlen=self.max_len, value=self.vocab_size - 1,
                                      padding=self.padding, truncating=self.padding)
        return np.array(sequence_list)
    
    def load_pretrained_embedding(self, feature_name, key_dtype, pretrain_path, embedding_dim, output_path):
        with h5py.File(pretrain_path, 'r') as hf:
            keys = hf["key"][:]
            if issubclass(keys.dtype.type, key_dtype): # in case mismatch between int and str
                keys = keys.astype(key_dtype)
            pretrained_vocab = dict(zip(keys, range(len(keys))))
            pretrained_emb = hf["value"][:]
        # update vocab with pretrained keys, in case new token ids appear in validation or test set
        num_new_words = 0
        for word in pretrained_vocab.keys():
            if word not in self.vocab:
                self.vocab[word] = self.vocab.get("__PAD__", self.vocab_size) + num_new_words
                num_new_words += 1
        self.vocab_size += num_new_words
        embedding_matrix = np.random.normal(loc=0, scale=1.e-4, size=(self.vocab_size, embedding_dim))
        if "__PAD__" in self.vocab:
            self.vocab["__PAD__"] = self.vocab_size - 1
            embedding_matrix[-1, :] = 0 # set as zero vector for PAD
        for word in pretrained_vocab.keys():
            embedding_matrix[self.vocab[word]] = pretrained_emb[pretrained_vocab[word]]
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with h5py.File(output_path, 'a') as hf:
            hf.create_dataset(feature_name, data=embedding_matrix)

    def load_vocab_from_file(self, vocab_file):
        with open(vocab_file, 'r') as fid:
            word_counts = json.load(fid)
        self.build_vocab(word_counts)

    def set_vocab(self, vocab):
        self.vocab = vocab
        self.vocab_size = len(self.vocab) + self.oov_token
            
        
class Normalizer(object):
    def __init__(self, normalizer_name):
        if normalizer_name in ['StandardScaler', 'MinMaxScaler']:
            self.normalizer = getattr(sklearn_preprocess, normalizer_name)()
        else:
            raise NotImplementedError('normalizer={}'.format(normalizer_name))

    def fit(self, X):
        null_index = np.isnan(X)
        self.normalizer.fit(X[~null_index].reshape(-1, 1))

    def transform(self, X):
        return self.normalizer.transform(X.reshape(-1, 1)).flatten()
```

<!-- #region id="FPX5MeVveLXX" -->
## Features
<!-- #endregion -->

```python id="WfxAT_b0eMwH"
import numpy as np
from collections import Counter, OrderedDict
import pandas as pd
import pickle
import os
import logging
import json
from collections import defaultdict
```

```python id="PV79388eeNuN"
class FeatureMap(object):
    def __init__(self, dataset_id, data_dir, query_index, corpus_index, label_name, version="pytorch"):
        self.data_dir = data_dir
        self.dataset_id = dataset_id
        self.version = version
        self.num_fields = 0
        self.num_features = 0
        self.num_items = 0
        self.query_index = query_index
        self.corpus_index = corpus_index
        self.label_name = label_name
        self.feature_specs = OrderedDict()

    def load(self, json_file):
        logging.info("Load feature_map from json: " + json_file)
        with open(json_file, "r", encoding="utf-8") as fd:
            feature_map = json.load(fd, object_pairs_hook=OrderedDict)
        if feature_map["dataset_id"] != self.dataset_id:
            raise RuntimeError("dataset_id={} does not match to feature_map!".format(self.dataset_id))
        self.num_fields = feature_map["num_fields"]
        self.num_features = feature_map.get("num_features", None)
        self.label_name = feature_map.get("label_name", None)
        self.feature_specs = OrderedDict(feature_map["feature_specs"])

    def save(self, json_file):
        logging.info("Save feature_map to json: " + json_file)
        os.makedirs(os.path.dirname(json_file), exist_ok=True)
        feature_map = OrderedDict()
        feature_map["dataset_id"] = self.dataset_id
        feature_map["num_fields"] = self.num_fields
        feature_map["num_features"] = self.num_features
        feature_map["num_items"] = self.num_items
        feature_map["query_index"] = self.query_index
        feature_map["corpus_index"] = self.corpus_index
        feature_map["label_name"] = self.label_name
        feature_map["feature_specs"] = self.feature_specs
        with open(json_file, "w", encoding="utf-8") as fd:
            json.dump(feature_map, fd, indent=4)

    def get_num_fields(self, feature_source=[]):
        if type(feature_source) != list:
            feature_source = [feature_source]
        num_fields = 0
        for feature, feature_spec in self.feature_specs.items():
            if not feature_source or feature_spec["source"] in feature_source:
                num_fields += 1
        return num_fields


```

```python id="QkDR6vbeBpQt"
class FeatureEncoder(object):
    def __init__(self, 
                 feature_cols=[], 
                 label_col={}, 
                 dataset_id=None, 
                 data_root="../data/", 
                 version="pytorch", 
                 **kwargs):
        logging.info("Set up feature encoder...")
        self.data_dir = os.path.join(data_root, dataset_id)
        self.pickle_file = os.path.join(self.data_dir, "feature_encoder.pkl")
        self.json_file = os.path.join(self.data_dir, "feature_map.json")
        self.feature_cols = self._complete_feature_cols(feature_cols)
        self.label_col = label_col
        self.version = version
        self.feature_map = FeatureMap(dataset_id, self.data_dir, kwargs["query_index"], 
                                      kwargs["corpus_index"], self.label_col["name"], version)
        self.dtype_dict = dict((feat["name"], eval(feat["dtype"]) if type(feat["dtype"]) == str else feat["dtype"]) 
                               for feat in self.feature_cols + [self.label_col])
        self.encoders = dict()

    def _complete_feature_cols(self, feature_cols):
        full_feature_cols = []
        for col in feature_cols:
            name_or_namelist = col["name"]
            if isinstance(name_or_namelist, list):
                for _name in name_or_namelist:
                    _col = col.copy()
                    _col["name"] = _name
                    full_feature_cols.append(_col)
            else:
                full_feature_cols.append(col)
        return full_feature_cols

    def read_csv(self, data_path, sep=",", nrows=None, **kwargs):
        if data_path is not None:
            logging.info("Reading file: " + data_path)
            usecols_fn = lambda x: x in self.dtype_dict
            ddf = pd.read_csv(data_path, sep=sep, usecols=usecols_fn, 
                              dtype=object, memory_map=True, nrows=nrows)
            return ddf
        else:
            return None

    def preprocess(self, ddf):
        logging.info("Preprocess feature columns...")
        if self.feature_map.query_index in ddf.columns: # for train/val/test ddf
            all_cols = [self.label_col] + [col for col in self.feature_cols[::-1] if col.get("source") != "item"]
        else: # for item_corpus ddf
            all_cols = [col for col in self.feature_cols[::-1] if col.get("source") == "item"]
        for col in all_cols:
            name = col["name"]
            if name in ddf.columns and ddf[name].isnull().values.any():
                ddf[name] = self._fill_na_(col, ddf[name])
            if "preprocess" in col and col["preprocess"] != "":
                preprocess_fn = getattr(self, col["preprocess"])
                ddf[name] = preprocess_fn(ddf, name)
            ddf[name] = ddf[name].astype(self.dtype_dict[name])
        active_cols = [col["name"] for col in all_cols if col.get("active") != False]
        ddf = ddf.loc[:, active_cols]
        return ddf

    def _fill_na_(self, col, series):
        na_value = col.get("na_value")
        if na_value is not None:
            return series.fillna(na_value)
        elif col["dtype"] in ["str", str]:
            return series.fillna("")
        else:
            raise RuntimeError("Feature column={} requires to assign na_value!".format(col["name"]))

    def fit(self, train_ddf, corpus_ddf, min_categr_count=1, num_buckets=10, **kwargs):      
        logging.info("Fit feature encoder...") 
        self.feature_map.num_items = len(corpus_ddf)
        train_ddf = train_ddf.join(corpus_ddf, on=self.feature_map.corpus_index)
        for col in self.feature_cols:
            name = col["name"]
            if col["active"]:
                self.feature_map.num_fields += 1
                logging.info("Processing column: {}".format(col))
                if col["type"] == "index":
                    self.fit_index_col(col)
                elif col["type"] == "numeric":
                    self.fit_numeric_col(col, train_ddf[name].values)
                elif col["type"] == "categorical":
                    self.fit_categorical_col(col, train_ddf[name].values, 
                                             min_categr_count=min_categr_count,
                                             num_buckets=num_buckets)
                elif col["type"] == "sequence":
                    self.fit_sequence_col(col, train_ddf[name].values, 
                                          min_categr_count=min_categr_count)
                else:
                    raise NotImplementedError("feature_col={}".format(feature_col))
        self.save_pickle(self.pickle_file)
        self.feature_map.save(self.json_file)
        logging.info("Set feature encoder done.")

    def fit_index_col(self, feature_col):
        name = feature_col["name"]
        feature_type = feature_col["type"]
        feature_source = feature_col.get("source", "")
        self.feature_map.feature_specs[name] = {"source": feature_source,
                                                "type": feature_type}  

    def fit_numeric_col(self, feature_col, data_vector):
        name = feature_col["name"]
        feature_type = feature_col["type"]
        feature_source = feature_col.get("source", "")
        self.feature_map.feature_specs[name] = {"source": feature_source,
                                                "type": feature_type}
        if "embedding_callback" in feature_col:
            self.feature_map.feature_specs[name]["embedding_callback"] = feature_col["embedding_callback"]
        if "normalizer" in feature_col:
            normalizer = Normalizer(feature_col["normalizer"])
            normalizer.fit(data_vector)
            self.encoders[name + "_normalizer"] = normalizer
        self.feature_map.num_features += 1
        
    def fit_categorical_col(self, feature_col, data_vector, min_categr_count=1, num_buckets=10):
        name = feature_col["name"]
        feature_type = feature_col["type"]
        feature_source = feature_col.get("source", "")
        min_categr_count = feature_col.get("min_categr_count", min_categr_count)
        self.feature_map.feature_specs[name] = {"source": feature_source,
                                                "type": feature_type,
                                                "min_categr_count": min_categr_count}
        if "embedding_callback" in feature_col:
            self.feature_map.feature_specs[name]["embedding_callback"] = feature_col["embedding_callback"]
        if "embedding_dim" in feature_col:
            self.feature_map.feature_specs[name]["embedding_dim"] = feature_col["embedding_dim"]
        if "category_encoder" not in feature_col:
            tokenizer = Tokenizer(min_freq=min_categr_count, 
                                  na_value=feature_col.get("na_value", ""))
            if "share_embedding" in feature_col:
                self.feature_map.feature_specs[name]["share_embedding"] = feature_col["share_embedding"]
                tokenizer.set_vocab(self.encoders["{}_tokenizer".format(feature_col["share_embedding"])].vocab)
            else:
                if self._whether_share_emb_with_sequence(name):
                    tokenizer.fit(data_vector, use_padding=True)
                    if "pretrained_emb" not in feature_col:
                        self.feature_map.feature_specs[name]["padding_idx"] = tokenizer.vocab_size - 1
                else:
                    tokenizer.fit(data_vector, use_padding=False)
            if "pretrained_emb" in feature_col:
                logging.info("Loading pretrained embedding: " + name)
                self.feature_map.feature_specs[name]["pretrained_emb"] = "pretrained_{}.h5".format(name)
                self.feature_map.feature_specs[name]["freeze_emb"] = feature_col.get("freeze_emb", True)
                tokenizer.load_pretrained_embedding(name,
                                                    self.dtype_dict[name],
                                                    feature_col["pretrained_emb"], 
                                                    feature_col["embedding_dim"],
                                                    os.path.join(self.data_dir, "pretrained_{}.h5".format(name)))
                if tokenizer.use_padding: # update to account pretrained keys
                    self.feature_map.feature_specs[name]["padding_idx"] = tokenizer.vocab_size - 1
            self.encoders[name + "_tokenizer"] = tokenizer
            self.feature_map.feature_specs[name]["vocab_size"] = tokenizer.vocab_size
            self.feature_map.num_features += tokenizer.vocab_size
        else:
            category_encoder = feature_col["category_encoder"]
            self.feature_map.feature_specs[name]["category_encoder"] = category_encoder
            if category_encoder == "quantile_bucket": # transform numeric value to bucket
                num_buckets = feature_col.get("num_buckets", num_buckets)
                qtf = sklearn_preprocess.QuantileTransformer(n_quantiles=num_buckets + 1)
                qtf.fit(data_vector)
                boundaries = qtf.quantiles_[1:-1]
                self.feature_map.feature_specs[name]["vocab_size"] = num_buckets
                self.feature_map.num_features += num_buckets
                self.encoders[name + "_boundaries"] = boundaries
            elif category_encoder == "hash_bucket":
                num_buckets = feature_col.get("num_buckets", num_buckets)
                uniques = Counter(data_vector)
                num_buckets = min(num_buckets, len(uniques))
                self.feature_map.feature_specs[name]["vocab_size"] = num_buckets
                self.encoders[name + "_num_buckets"] = num_buckets
                self.feature_map.num_features += num_buckets
            else:
                raise NotImplementedError("category_encoder={} not supported.".format(category_encoder))

    def fit_sequence_col(self, feature_col, data_vector, min_categr_count=1):
        name = feature_col["name"]
        feature_type = feature_col["type"]
        feature_source = feature_col.get("source", "")
        min_categr_count = feature_col.get("min_categr_count", min_categr_count)
        self.feature_map.feature_specs[name] = {"source": feature_source,
                                                "type": feature_type,
                                                "min_categr_count": min_categr_count}
        embedding_callback = feature_col.get("embedding_callback", "layers.MaskedAveragePooling()")
        if embedding_callback not in [None, "null", "None", "none"]:
            self.feature_map.feature_specs[name]["embedding_callback"] = embedding_callback
        splitter = feature_col.get("splitter", " ")
        na_value = feature_col.get("na_value", "")
        max_len = feature_col.get("max_len", 0)
        padding = feature_col.get("padding", "post") # "post" or "pre"
        tokenizer = Tokenizer(min_freq=min_categr_count, splitter=splitter, 
                              na_value=na_value, max_len=max_len, padding=padding)
        if "share_embedding" in feature_col:
            self.feature_map.feature_specs[name]["share_embedding"] = feature_col["share_embedding"]
            tokenizer.set_vocab(self.encoders["{}_tokenizer".format(feature_col["share_embedding"])].vocab)
        else:
            tokenizer.fit(data_vector, use_padding=True)
        if "pretrained_emb" in feature_col:
            logging.info("Loading pretrained embedding: " + name)
            self.feature_map.feature_specs[name]["pretrained_emb"] = "pretrained_{}.h5".format(name)
            self.feature_map.feature_specs[name]["freeze_emb"] = feature_col.get("freeze_emb", True)
            tokenizer.load_pretrained_embedding(name,
                                                self.dtype_dict[name],
                                                feature_col["pretrained_emb"], 
                                                feature_col["embedding_dim"],
                                                os.path.join(self.data_dir, "pretrained_{}.h5".format(name)))
        self.encoders[name + "_tokenizer"] = tokenizer
        self.feature_map.feature_specs[name].update({"padding_idx": tokenizer.vocab_size - 1,
                                                     "vocab_size": tokenizer.vocab_size,
                                                     "max_len": tokenizer.max_len})
        self.feature_map.num_features += tokenizer.vocab_size

    def transform(self, ddf):
        logging.info("Transform feature columns...")
        data_dict = dict()
        for feature, feature_spec in self.feature_map.feature_specs.items():
            if feature in ddf.columns:
                feature_type = feature_spec["type"]
                data_vector = ddf.loc[:, feature].values
                if feature_type == "index":
                    data_dict[feature] = data_vector
                elif feature_type == "numeric":
                    data_vector = data_vector.astype(float)
                    normalizer = self.encoders.get(feature + "_normalizer")
                    if normalizer:
                         data_vector = normalizer.transform(data_vector)
                    data_dict[feature] = data_vector
                elif feature_type == "categorical":
                    category_encoder = feature_spec.get("category_encoder")
                    if category_encoder is None:
                        data_dict[feature] = self.encoders.get(feature + "_tokenizer").encode_category(data_vector)
                    elif encoder == "numeric_bucket":
                        raise NotImplementedError
                    elif encoder == "hash_bucket":
                        raise NotImplementedError
                elif feature_type == "sequence":
                    data_dict[feature] = self.encoders.get(feature + "_tokenizer").encode_sequence(data_vector)
        label = self.label_col["name"]
        if label in ddf.columns:
            data_dict[label] = ddf.loc[:, label].values.astype(float)
        return data_dict

    def _whether_share_emb_with_sequence(self, feature):
        for col in self.feature_cols:
            if col.get("share_embedding", None) == feature and col["type"] == "sequence":
                return True
        return False

    def load_pickle(self, pickle_file=None):
        """ Load feature encoder from cache """
        if pickle_file is None:
            pickle_file = self.pickle_file
        logging.info("Load feature_encoder from pickle: " + pickle_file)
        if os.path.exists(pickle_file):
            pickled_feature_encoder = pickle.load(open(pickle_file, "rb"))
            if pickled_feature_encoder.feature_map.dataset_id == self.feature_map.dataset_id:
                pickled_feature_encoder.version = self.version
                return pickled_feature_encoder
        raise IOError("pickle_file={} not valid.".format(pickle_file))

    def save_pickle(self, pickle_file):
        logging.info("Pickle feature_encode: " + pickle_file)
        if not os.path.exists(os.path.dirname(pickle_file)):
            os.makedirs(os.path.dirname(pickle_file))
        pickle.dump(self, open(pickle_file, "wb"))
```

<!-- #region id="54LTLjYPcBW6" -->
## Dataset
<!-- #endregion -->

```python id="rZuXyMBPcV00"
def save_h5(darray_dict, data_path):
    logging.info("Saving data to h5: " + data_path)
    if not os.path.exists(os.path.dirname(data_path)):
        try:
            os.makedirs(os.path.dirname(data_path))
        except:
            pass
    with h5py.File(data_path, 'w') as hf:
        hf.attrs["num_samples"] = len(list(darray_dict.values())[0])
        for key, arr in darray_dict.items():
            hf.create_dataset(key, data=arr)


def load_h5(data_path, verbose=True):
    if verbose:
        logging.info('Loading data from h5: ' + data_path)
    data_dict = dict()
    with h5py.File(data_path, 'r') as hf:
        num_samples = hf.attrs["num_samples"]
        for key in hf.keys():
            data_dict[key] = hf[key][:]
    return data_dict, num_samples


def split_train_test(train_ddf=None, valid_ddf=None, test_ddf=None, valid_size=0, 
                     test_size=0, split_type="sequential"):
    num_samples = len(train_ddf)
    train_size = num_samples
    instance_IDs = np.arange(num_samples)
    if split_type == "random":
        np.random.shuffle(instance_IDs)
    if test_size > 0:
        if test_size < 1:
            test_size = int(num_samples * test_size)
        train_size = train_size - test_size
        test_ddf = train_ddf.loc[instance_IDs[train_size:], :].reset_index()
        instance_IDs = instance_IDs[0:train_size]
    if valid_size > 0:
        if valid_size < 1:
            valid_size = int(num_samples * valid_size)
        train_size = train_size - valid_size
        valid_ddf = train_ddf.loc[instance_IDs[train_size:], :].reset_index()
        instance_IDs = instance_IDs[0:train_size]
    if valid_size > 0 or test_size > 0:
        train_ddf = train_ddf.loc[instance_IDs, :].reset_index()
    return train_ddf, valid_ddf, test_ddf


def transform_h5(feature_encoder, ddf, filename, preprocess=False, block_size=0):
    def _transform_block(feature_encoder, df_block, filename, preprocess):
        if preprocess:
            df_block = feature_encoder.preprocess(df_block)
        darray_dict = feature_encoder.transform(df_block)
        save_h5(darray_dict, os.path.join(feature_encoder.data_dir, filename))

    if block_size > 0:
        pool = mp.Pool(mp.cpu_count() // 2)
        block_id = 0
        for idx in range(0, len(ddf), block_size):
            df_block = ddf[idx: (idx + block_size)]
            pool.apply_async(_transform_block, args=(feature_encoder, 
                                                     df_block, 
                                                     filename.replace('.h5', '_part_{}.h5'.format(block_id)),
                                                     preprocess))
            block_id += 1
        pool.close()
        pool.join()
    else:
        _transform_block(feature_encoder, ddf, filename, preprocess)


def build_dataset(feature_encoder, item_corpus=None, train_data=None, valid_data=None, 
                  test_data=None, valid_size=0, test_size=0, split_type="sequential", **kwargs):
    """ Build feature_map and transform h5 data """
    
    # Load csv data
    train_ddf = feature_encoder.read_csv(train_data, **kwargs)
    valid_ddf = None
    test_ddf = None

    # Split data for train/validation/test
    if valid_size > 0 or test_size > 0:
        valid_ddf = feature_encoder.read_csv(valid_data, **kwargs)
        test_ddf = feature_encoder.read_csv(test_data, **kwargs)
        train_ddf, valid_ddf, test_ddf = split_train_test(train_ddf, valid_ddf, test_ddf, 
                                                          valid_size, test_size, split_type)

    # fit feature_encoder
    corpus_ddf = feature_encoder.read_csv(item_corpus, **kwargs)
    corpus_ddf = feature_encoder.preprocess(corpus_ddf)
    train_ddf = feature_encoder.preprocess(train_ddf)
    feature_encoder.fit(train_ddf, corpus_ddf, **kwargs)

    # transform corpus_ddf
    item_corpus_dict = feature_encoder.transform(corpus_ddf)
    save_h5(item_corpus_dict, os.path.join(feature_encoder.data_dir, 'item_corpus.h5'))
    del item_corpus_dict, corpus_ddf
    gc.collect()

    # transform train_ddf
    block_size = int(kwargs.get("data_block_size", 0)) # Num of samples in a data block
    transform_h5(feature_encoder, train_ddf, 'train.h5', preprocess=False, block_size=block_size)
    del train_ddf
    gc.collect()

    # Transfrom valid_ddf
    if valid_ddf is None and (valid_data is not None):
        valid_ddf = feature_encoder.read_csv(valid_data, **kwargs)
    if valid_ddf is not None:
        transform_h5(feature_encoder, valid_ddf, 'valid.h5', preprocess=True, block_size=block_size)
        del valid_ddf
        gc.collect()

    # Transfrom test_ddf
    if test_ddf is None and (test_data is not None):
        test_ddf = feature_encoder.read_csv(test_data, **kwargs)
    if test_ddf is not None:
        transform_h5(feature_encoder, test_ddf, 'test.h5', preprocess=True, block_size=block_size)
        del test_ddf
        gc.collect()
    logging.info("Transform csv data to h5 done.")


def h5_generator(feature_map, stage="both", train_data=None, valid_data=None, test_data=None,
                 item_corpus=None, batch_size=32, num_negs=10, shuffle=True, **kwargs):
    logging.info("Loading data...")
    # Choose different DataGenerator versions
    if feature_map.version == 'tensorflow':
        from ..tensorflow.data_generator import DataGenerator
    elif feature_map.version == 'pytorch':
        from ..pytorch.data_generator import TrainGenerator, TestGenerator
    train_gen = None
    valid_gen = None
    test_gen = None
    if stage in ["both", "train"]:
        train_blocks = glob.glob(train_data)
        valid_blocks = glob.glob(valid_data)
        assert len(train_blocks) > 0 and len(valid_blocks) > 0, "invalid data files or paths."
        train_gen = TrainGenerator(feature_map, train_blocks, item_corpus, batch_size=batch_size, 
                                   num_negs=num_negs, shuffle=shuffle, **kwargs)
        valid_gen = TestGenerator(feature_map, valid_blocks, item_corpus, batch_size=batch_size, 
                                  shuffle=False, **kwargs)
        logging.info("Train samples: total/{:d}, blocks/{:.0f}".format(train_gen.num_samples, train_gen.num_blocks))
        logging.info("Validation samples: total/{:d}, blocks/{:.0f}".format(valid_gen.num_samples, valid_gen.num_blocks))
        if stage == "train":
            logging.info("Loading train data done.")
            return train_gen, valid_gen

    if stage in ["both", "test"]:
        test_blocks = glob.glob(test_data)
        test_gen = TestGenerator(feature_map, test_blocks, item_corpus, batch_size=batch_size, 
                                 shuffle=False, **kwargs)
        logging.info("Test samples: total/{:d}, blocks/{:.0f}".format(test_gen.num_samples, test_gen.num_blocks))
        if stage == "test":
            logging.info("Loading test data done.")
            return test_gen

    logging.info("Loading data done.")
    return train_gen, valid_gen, test_gen
```

<!-- #region id="o7BLQSTOcgrS" -->
## Data Loader
<!-- #endregion -->

```python id="GuQ8Ia2ccgpW"
class TrainDataset(Dataset):
    def __init__(self, feature_map, data_path, item_corpus):
        self.data_dict, self.num_samples = load_h5(data_path)
        self.item_corpus_dict, self.num_items = load_h5(item_corpus)
        self.labels = self.data_dict[feature_map.label_name]
        self.pos_item_indexes = self.data_dict[feature_map.corpus_index]
        self.all_item_indexes = self.data_dict[feature_map.corpus_index]
        
    def __getitem__(self, index):
        user_dict = self.slice_array_dict(self.data_dict, index)
        item_indexes = self.all_item_indexes[index, :]
        item_dict = self.slice_array_dict(self.item_corpus_dict, item_indexes)
        label = self.labels[index]
        return user_dict, item_dict, label, item_indexes
    
    def __len__(self):
        return self.num_samples

    def slice_array_dict(self, array_dict, slice_index):
        return dict((k, v[slice_index]) for k, v in array_dict.items())


def get_user2items_dict(data_dict, feature_map):
    user2items_dict = defaultdict(list)
    for query_index, corpus_index in zip(data_dict[feature_map.query_index], 
                                         data_dict[feature_map.corpus_index]):
        user2items_dict[query_index].append(corpus_index)
    return user2items_dict


def collate_fn_unique(batch): 
    # TODO: check correctness
    user_dict, item_dict, labels, item_indexes = default_collate(batch)
    num_negs = item_indexes.size(1) - 1
    unique, inverse_indexes = torch.unique(item_indexes.flatten(), return_inverse=True, sorted=True)
    perm = torch.arange(inverse_indexes.size(0), dtype=inverse_indexes.dtype, device=inverse_indexes.device)
    inverse_indexes, perm = inverse_indexes.flip([0]), perm.flip([0])
    unique_indexes = inverse_indexes.new_empty(unique.size(0)).scatter_(0, inverse_indexes, perm) # obtain return_indicies in np.unique
    # reshape item data with (b*(num_neg + 1) x input_dim)
    for k, v in item_dict.items():
        item_dict[k] = v.flatten(end_dim=1)[unique_indexes]
    # add negative labels
    labels = torch.cat([labels.view(-1, 1).float(), torch.zeros((labels.size(0), num_negs))], dim=1)
    return user_dict, item_dict, labels, inverse_indexes


def collate_fn(batch):
    user_dict, item_dict, labels, item_indexes = default_collate(batch)
    num_negs = item_indexes.size(1) - 1
    # reshape item data with (b*(num_neg + 1) x input_dim)
    for k, v in item_dict.items():
        item_dict[k] = v.flatten(end_dim=1)
    # add negative labels
    labels = torch.cat([labels.view(-1, 1).float(), torch.zeros((labels.size(0), num_negs))], dim=1)
    return user_dict, item_dict, labels, None


def sampling_block(num_items, block_query_indexes, num_negs, user2items_dict, 
                   sampling_probs=None, ignore_pos_items=False, seed=None, dump_path=None):
    if seed is not None:
        np.random.seed(seed) # used in multiprocessing
    if sampling_probs is None:
        sampling_probs = np.ones(num_items) / num_items # uniform sampling
    if ignore_pos_items:
        sampled_items = []
        for query_index in block_query_indexes:
            pos_items = user2items_dict[query_index]
            probs = np.array(sampling_probs)
            probs[pos_items] = 0
            probs = probs / np.sum(probs) # renomalize to sum 1
            sampled_items.append(np.random.choice(num_items, size=num_negs, replace=True, p=probs))
        sampled_array = np.array(sampled_items)
    else:
        sampled_array = np.random.choice(num_items,
                                         size=(len(block_query_indexes), num_negs), 
                                         replace=True)
    if dump_path is not None:
        # To fix bug in multiprocessing: https://github.com/xue-pai/Open-CF-Benchmarks/issues/1
        pickle_array(sampled_array, dump_path)
    else:
        return sampled_array


def pickle_array(array, path):
    with open(path, "wb") as fout:
        pickle.dump(array, fout, pickle.HIGHEST_PROTOCOL)


def load_pickled_array(path):
    with open(path, "rb") as fin:
        return pickle.load(fin)


class TrainGenerator(DataLoader):
    # reference https://cloud.tencent.com/developer/article/1010247
    def __init__(self, feature_map, data_path, item_corpus, batch_size=32, shuffle=True, 
                 num_workers=1, num_negs=0, compress_duplicate_items=False, **kwargs):
        if type(data_path) == list:
            data_path = data_path[0]
            self.num_blocks = 1
        self.num_negs = num_negs
        self.dataset = TrainDataset(feature_map, data_path, item_corpus)
        super(TrainGenerator, self).__init__(dataset=self.dataset, batch_size=batch_size,
                                             shuffle=shuffle, num_workers=num_workers,
                                             collate_fn=collate_fn_unique if compress_duplicate_items else collate_fn)
        self.user2items_dict = get_user2items_dict(self.dataset.data_dict, feature_map)
        self.query_indexes = self.dataset.data_dict[feature_map.query_index]
        # delete some columns to speed up batch generator
        del self.dataset.data_dict[feature_map.query_index]
        del self.dataset.data_dict[feature_map.corpus_index]
        del self.dataset.data_dict[feature_map.label_name]
        self.num_samples = len(self.dataset)
        self.num_batches = int(np.ceil(self.num_samples * 1.0 / batch_size))
        self.sampling_num_process = kwargs.get("sampling_num_process", 1)
        self.ignore_pos_items = kwargs.get("ignore_pos_items", False)
        self.fix_sampling_seeds = kwargs.get("fix_sampling_seeds", True)

    def __iter__(self):
        self.negative_sampling()
        iter = super(TrainGenerator, self).__iter__()
        while True:
            yield next(iter) # a batch iterator

    def __len__(self):
        return self.num_batches

    def negative_sampling(self):
        if self.num_negs > 0:
            logging.info("Negative sampling num_negs={}".format(self.num_negs))
            sampling_probs = None # set it to item popularity when using importance sampling
            if self.sampling_num_process > 1:
                chunked_query_indexes = np.array_split(self.query_indexes, self.sampling_num_process)
                if self.fix_sampling_seeds:
                    seeds = np.random.randint(1000000, size=self.sampling_num_process)
                else:
                    seeds = [None] * self.sampling_num_process
                pool = mp.Pool(self.sampling_num_process)
                block_result = []
                os.makedirs("./tmp/pid_{}/".format(os.getpid()), exist_ok=True)
                dump_paths = ["./tmp/pid_{}/part_{}.pkl".format(os.getpid(), idx) for idx in range(len(chunked_query_indexes))]
                for idx, block_query_indexes in enumerate(chunked_query_indexes):
                    pool.apply_async(sampling_block, args=(self.dataset.num_items, 
                                                           block_query_indexes, 
                                                           self.num_negs, 
                                                           self.user2items_dict, 
                                                           sampling_probs, 
                                                           self.ignore_pos_items,
                                                           seeds[idx],
                                                           dump_paths[idx]))
                pool.close()
                pool.join()
                block_result = [load_pickled_array(dump_paths[idx]) for idx in range(len(chunked_query_indexes))]
                shutil.rmtree("./tmp/pid_{}/".format(os.getpid()))
                neg_item_indexes = np.vstack(block_result)
            else:
                neg_item_indexes = sampling_block(self.dataset.num_items, 
                                                  self.query_indexes, 
                                                  self.num_negs, 
                                                  self.user2items_dict, 
                                                  sampling_probs,
                                                  self.ignore_pos_items)
            self.dataset.all_item_indexes = np.hstack([self.dataset.pos_item_indexes.reshape(-1, 1), 
                                                       neg_item_indexes])
            logging.info("Negative sampling done")


class TestDataset(Dataset):
    def __init__(self, data_path):
        self.data_dict, self.num_samples = load_h5(data_path)

    def __getitem__(self, index):
        batch_dict = self.slice_array_dict(index)
        return batch_dict
    
    def __len__(self):
        return self.num_samples

    def slice_array_dict(self, slice_index):
        return dict((k, v[slice_index]) for k, v in self.data_dict.items())


class TestGenerator(object):
    def __init__(self, feature_map, data_path, item_corpus, batch_size=32, shuffle=False, 
                 num_workers=1, **kwargs):
        if type(data_path) == list:
            data_path = data_path[0]
            self.num_blocks = 1
        user_dataset = TestDataset(data_path)
        self.user2items_dict = get_user2items_dict(user_dataset.data_dict, feature_map)
        # pick users of unique query_index
        self.query_indexes, unique_rows = np.unique(user_dataset.data_dict[feature_map.query_index], 
                                                    return_index=True)
        user_dataset.num_samples = len(unique_rows)
        self.num_samples = len(user_dataset)
        # delete some columns to speed up batch generator
        del user_dataset.data_dict[feature_map.query_index]
        del user_dataset.data_dict[feature_map.corpus_index]
        del user_dataset.data_dict[feature_map.label_name]
        for k, v in user_dataset.data_dict.items():
            user_dataset.data_dict[k] = v[unique_rows]
        item_dataset = TestDataset(item_corpus)
        self.user_loader = DataLoader(dataset=user_dataset, batch_size=batch_size,
                                      shuffle=shuffle, num_workers=num_workers)
        self.item_loader = DataLoader(dataset=item_dataset, batch_size=batch_size,
                                      shuffle=shuffle, num_workers=num_workers)
```

<!-- #region id="_qS94t5_cgnq" -->
## Layers
<!-- #endregion -->

```python id="ux27A3yHcglo"
import torch
from torch import nn
import h5py
import os
import numpy as np
from collections import OrderedDict


class EmbeddingLayer(nn.Module):
    def __init__(self, 
                 feature_map,
                 embedding_dim,
                 disable_sharing_pretrain=False,
                 required_feature_columns=[],
                 not_required_feature_columns=[]):
        super(EmbeddingLayer, self).__init__()
        self.embedding_layer = EmbeddingDictLayer(feature_map, 
                                                  embedding_dim,
                                                  disable_sharing_pretrain=disable_sharing_pretrain,
                                                  required_feature_columns=required_feature_columns,
                                                  not_required_feature_columns=not_required_feature_columns)

    def forward(self, X, feature_source=None):
        feature_emb_dict = self.embedding_layer(X, feature_source=feature_source)
        feature_emb = self.embedding_layer.dict2tensor(feature_emb_dict)
        return feature_emb


class EmbeddingDictLayer(nn.Module):
    def __init__(self, 
                 feature_map, 
                 embedding_dim,
                 disable_sharing_pretrain=False,
                 required_feature_columns=None,
                 not_required_feature_columns=None):
        super(EmbeddingDictLayer, self).__init__()
        self._feature_map = feature_map
        self.required_feature_columns = required_feature_columns
        self.not_required_feature_columns = not_required_feature_columns
        self.embedding_layers = nn.ModuleDict()
        self.embedding_callbacks = nn.ModuleDict()
        for feature, feature_spec in self._feature_map.feature_specs.items():
            if self.is_required(feature):
                if disable_sharing_pretrain: # in case for LR
                    assert embedding_dim == 1
                    feat_emb_dim = embedding_dim
                else:
                    feat_emb_dim = feature_spec.get("embedding_dim", embedding_dim)
                if (not disable_sharing_pretrain) and "embedding_callback" in feature_spec:
                    self.embedding_callbacks[feature] = eval(feature_spec["embedding_callback"])
                # Set embedding_layer according to share_embedding
                if (not disable_sharing_pretrain) and "share_embedding" in feature_spec:
                    self.embedding_layers[feature] = self.embedding_layers[feature_spec["share_embedding"]]
                    continue
                    
                if feature_spec["type"] == "numeric":
                    self.embedding_layers[feature] = nn.Linear(1, feat_emb_dim, bias=False)
                elif feature_spec["type"] == "categorical":
                    padding_idx = feature_spec.get("padding_idx", None)
                    embedding_matrix = nn.Embedding(feature_spec["vocab_size"], 
                                                    feat_emb_dim,
                                                    padding_idx=padding_idx)
                    if (not disable_sharing_pretrain) and "pretrained_emb" in feature_spec:
                        embedding_matrix = self.load_pretrained_embedding(embedding_matrix,
                                                                          feature_map, 
                                                                          feature_name, 
                                                                          freeze=feature_spec["freeze_emb"],
                                                                          padding_idx=padding_idx)
                    self.embedding_layers[feature] = embedding_matrix
                elif feature_spec["type"] == "sequence":
                    padding_idx = feature_spec.get("padding_idx", None)
                    embedding_matrix = nn.Embedding(feature_spec["vocab_size"], 
                                                    feat_emb_dim, 
                                                    padding_idx=padding_idx)
                    if (not disable_sharing_pretrain) and "pretrained_emb" in feature_spec:
                        embedding_matrix = self.load_pretrained_embedding(embedding_matrix, 
                                                                          feature_map, 
                                                                          feature_name,
                                                                          freeze=feature_spec["freeze_emb"],
                                                                          padding_idx=padding_idx)
                    self.embedding_layers[feature] = embedding_matrix

    def is_required(self, feature):
        """ Check whether feature is required for embedding """
        feature_spec = self._feature_map.feature_specs[feature]
        if self.required_feature_columns and (feature not in self.required_feature_columns):
            return False
        if self.not_required_feature_columns and (feature in self.not_required_feature_columns):
            return False
        return True

    def get_pretrained_embedding(self, pretrained_path, feature_name):
        with h5py.File(pretrained_path, 'r') as hf:
            embeddings = hf[feature_name][:]
        return embeddings

    def load_pretrained_embedding(self, embedding_matrix, feature_map, feature_name, freeze=False, padding_idx=None):
        pretrained_path = os.path.join(feature_map.data_dir, feature_map.feature_specs[feature_name]["pretrained_emb"])
        embeddings = self.get_pretrained_embedding(pretrained_path, feature_name)
        if padding_idx is not None:
            embeddings[padding_idx] = np.zeros(embeddings.shape[-1])
        embeddings = torch.from_numpy(embeddings).float()
        embedding_matrix.weight = torch.nn.Parameter(embeddings)
        if freeze:
            embedding_matrix.weight.requires_grad = False
        return embedding_matrix

    def dict2tensor(self, embedding_dict):
        if len(embedding_dict) == 1:
            feature_emb = list(embedding_dict.values())[0]
        else:
            feature_emb = torch.stack(list(embedding_dict.values()), dim=1)
        return feature_emb

    def forward(self, inputs, feature_source=None, feature_type=None):
        feature_emb_dict = OrderedDict()
        for feature, feature_spec in self._feature_map.feature_specs.items():
            if feature_source and feature_spec["source"] != feature_source:
                continue
            if feature_type and feature_spec["type"] != feature_type:
                continue
            if feature in self.embedding_layers:
                if feature_spec["type"] == "numeric":
                    inp = inputs[feature].float().view(-1, 1)
                    embeddings = self.embedding_layers[feature](inp)
                elif feature_spec["type"] == "categorical":
                    inp = inputs[feature].long()
                    embeddings = self.embedding_layers[feature](inp)
                elif feature_spec["type"] == "sequence":
                    inp = inputs[feature].long()
                    embeddings = self.embedding_layers[feature](inp)
                else:
                    raise NotImplementedError
                if feature in self.embedding_callbacks:
                    embeddings = self.embedding_callbacks[feature](embeddings)     
                feature_emb_dict[feature] = embeddings
        return feature_emb_dict
```

```python id="1_eXWIOicgjO"
class MLP_Layer(nn.Module):
    def __init__(self, 
                 input_dim, 
                 output_dim=None, 
                 hidden_units=[], 
                 hidden_activations="ReLU",
                 final_activation=None, 
                 dropout_rates=[], 
                 batch_norm=False, 
                 use_bias=True):
        super(MLP_Layer, self).__init__()
        dense_layers = []
        if not isinstance(dropout_rates, list):
            dropout_rates = [dropout_rates] * len(hidden_units)
        if not isinstance(hidden_activations, list):
            hidden_activations = [hidden_activations] * len(hidden_units)
        hidden_activations = [set_activation(x) for x in hidden_activations]
        hidden_units = [input_dim] + hidden_units
        for idx in range(len(hidden_units) - 1):
            dense_layers.append(nn.Linear(hidden_units[idx], hidden_units[idx + 1], bias=use_bias))
            if batch_norm:
                dense_layers.append(nn.BatchNorm1d(hidden_units[idx + 1]))
            if hidden_activations[idx]:
                dense_layers.append(hidden_activations[idx])
            if dropout_rates[idx] > 0:
                dense_layers.append(nn.Dropout(p=dropout_rates[idx]))
        if output_dim is not None:
            dense_layers.append(nn.Linear(hidden_units[-1], output_dim, bias=use_bias))
        if final_activation is not None:
            dense_layers.append(set_activation(final_activation))
        self.mlp = nn.Sequential(*dense_layers) # * used to unpack list
    
    def forward(self, inputs):
        return self.mlp(inputs)
```

```python id="hbTVg0_HcggH"
class MaskedAveragePooling(nn.Module):
    def __init__(self):
        super(MaskedAveragePooling, self).__init__()

    def forward(self, embedding_matrix):
        sum_pooling_matrix = torch.sum(embedding_matrix, dim=1)
        non_padding_length = (embedding_matrix.sum(dim=-1) != 0).sum(dim=1, keepdim=True)
        embedding_vec = sum_pooling_matrix / (non_padding_length.float() + 1.e-12)
        return embedding_vec


class MaskedSumPooling(nn.Module):
    def __init__(self):
        super(MaskedSumPooling, self).__init__()

    def forward(self, embedding_matrix):
        return torch.sum(embedding_matrix, dim=1)
```

<!-- #region id="cXQ9kMz2cgc6" -->
## Losses
<!-- #endregion -->

```python id="3J3EOl6Ac7TT"
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
```

```python id="9qXVmVj8c8Fu"
class SoftmaxCrossEntropyLoss(nn.Module):
    def __init__(self):
        """
        :param num_negs: number of negative instances in bpr loss.
        """
        super(SoftmaxCrossEntropyLoss, self).__init__()

    def forward(self, y_pred, y_true):
        """
        :param y_true: Labels
        :param y_pred: Predicted result.
        """
        probs = F.softmax(y_pred, dim=1)
        hit_probs = probs[:, 0]
        loss = -torch.log(hit_probs).mean()
        return loss
```

```python id="rYWScszuc8Cy"
class CosineContrastiveLoss(nn.Module):
    def __init__(self, margin=0, negative_weight=None):
        """
        :param margin: float, margin in CosineContrastiveLoss
        :param num_negs: int, number of negative samples
        :param negative_weight:, float, the weight set to the negative samples. When negative_weight=None, it
            equals to num_negs
        """
        super(CosineContrastiveLoss, self).__init__()
        self._margin = margin
        self._negative_weight = negative_weight

    def forward(self, y_pred, y_true):
        """
        :param y_pred: prdicted values of shape (batch_size, 1 + num_negs) 
        :param y_true: true labels of shape (batch_size, 1 + num_negs)
        """
        pos_logits = y_pred[:, 0]
        pos_loss = torch.relu(1 - pos_logits)
        neg_logits = y_pred[:, 1:]
        neg_loss = torch.relu(neg_logits - self._margin)
        if self._negative_weight:
            loss = pos_loss + neg_loss.mean(dim=-1) * self._negative_weight
        else:
            loss = pos_loss + neg_loss.sum(dim=-1)
        return loss.mean()
```

```python id="yXDK8Ltxc7_c"
class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, y_pred, y_true):
        """
        :param y_pred: prdicted values of shape (batch_size, 1 + num_negs) 
        :param y_true: true labels of shape (batch_size, 1 + num_negs)
        """
        pos_logits = y_pred[:, 0]
        pos_loss = torch.pow(pos_logits - 1, 2) / 2
        neg_logits = y_pred[:, 1:]
        neg_loss = torch.pow(neg_logits, 2).sum(dim=-1) / 2
        loss = pos_loss + neg_loss
        return loss.mean()
```

```python id="_sQi7qWHc78P"
class PairwiseLogisticLoss(nn.Module):
    def __init__(self):
        super(PairwiseLogisticLoss, self).__init__()

    def forward(self, y_pred, y_true):
        """
        :param y_true: Labels
        :param y_pred: Predicted result.
        """
        pos_logits = y_pred[:, 0].unsqueeze(-1)
        neg_logits = y_pred[:, 1:]
        logits_diff = pos_logits - neg_logits
        loss = -torch.log(torch.sigmoid(logits_diff)).mean()
        return loss
```

```python id="RKcjkYscc75g"
class PairwiseMarginLoss(nn.Module):
    def __init__(self, margin=1.0):
        """
        :param num_negs: number of negative instances in bpr loss.
        """
        super(PairwiseMarginLoss, self).__init__()
        self._margin = margin

    def forward(self, y_pred, y_true):
        """
        :param y_true: Labels
        :param y_pred: Predicted result.
        """
        pos_logits = y_pred[:, 0].unsqueeze(-1)
        neg_logits = y_pred[:, 1:]
        loss = torch.relu(self._margin + neg_logits - pos_logits).mean()
        return loss
```

```python id="sqCjo0Irc72S"
class SigmoidCrossEntropyLoss(nn.Module):
    def __init__(self):
        """
        :param num_negs: number of negative instances in bpr loss.
        """
        super(SigmoidCrossEntropyLoss, self).__init__()

    def forward(self, y_pred, y_true):
        """
        :param y_true: Labels
        :param y_pred: Predicted result
        """
        logits = y_pred.flatten()
        labels = y_true.flatten()
        loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="sum")
        return loss
```

<!-- #region id="WTKBGfthdLkx" -->
## Models
<!-- #endregion -->

```python id="1g3FXaVSdLiK"
import torch.nn as nn
import numpy as np
import torch
import os
import sys
import logging
from tqdm import tqdm
```

```python id="7ajKAcC7dj_g"
class BaseModel(nn.Module):
    def __init__(self, 
                 feature_map, 
                 model_id="BaseModel", 
                 gpu=-1, 
                 monitor="AUC", 
                 save_best_only=True, 
                 monitor_mode="max", 
                 patience=2, 
                 eval_interval_epochs=1, 
                 embedding_regularizer=None, 
                 net_regularizer=None, 
                 reduce_lr_on_plateau=True, 
                 embedding_initializer="lambda w: nn.init.normal_(w, std=1e-4)", 
                 num_negs=0,
                 **kwargs):
        super(BaseModel, self).__init__()
        self.device = set_device(gpu)
        self.feature_map = feature_map
        self._monitor = Monitor(kv=monitor)
        self._monitor_mode = monitor_mode
        self._patience = patience
        self._eval_interval_epochs = eval_interval_epochs # float acceptable
        self._save_best_only = save_best_only
        self._embedding_regularizer = embedding_regularizer
        self._net_regularizer = net_regularizer
        self._reduce_lr_on_plateau = reduce_lr_on_plateau
        self._embedding_initializer = embedding_initializer
        self.model_id = model_id
        self.model_dir = os.path.join(kwargs["model_root"], feature_map.dataset_id)
        self.checkpoint = os.path.abspath(os.path.join(self.model_dir, self.model_id + ".model"))
        self._validation_metrics = kwargs["metrics"]
        self._verbose = kwargs["verbose"]
        self.num_negs = num_negs

    def compile(self, lr=1e-3, optimizer=None, loss=None, **kwargs):
        try:
            self.optimizer = set_optimizer(optimizer)(self.parameters(), lr=lr)
        except:
            raise NotImplementedError("optimizer={} is not supported.".format(optimizer))
        if loss == "SigmoidCrossEntropyLoss":
            self.loss_fn = losses.SigmoidCrossEntropyLoss()
        elif loss == "PairwiseLogisticLoss":
            self.loss_fn = losses.PairwiseLogisticLoss()
        elif loss == "SoftmaxCrossEntropyLoss":
            self.loss_fn = losses.SoftmaxCrossEntropyLoss()
        elif loss == "PairwiseMarginLoss":
            self.loss_fn = losses.PairwiseMarginLoss(margin=kwargs.get("margin", 1))
        elif loss == "MSELoss":
            self.loss_fn = losses.MSELoss()
        elif loss == "CosineContrastiveLoss":
            self.loss_fn = losses.CosineContrastiveLoss(margin=kwargs.get("margin", 0),
                                                        negative_weight=kwargs.get("negative_weight"))
        else:
            raise NotImplementedError("loss={} is not supported.".format(loss))
        self.apply(self.init_weights)
        self.to(device=self.device)

    def get_total_loss(self, y_pred, y_true):
        # y_pred: N x (1 + num_negs) 
        # y_true:  N x (1 + num_negs) 
        y_true = y_true.float().to(self.device)
        total_loss = self.loss_fn(y_pred, y_true)
        if self._embedding_regularizer or self._net_regularizer:
            emb_reg = set_regularizer(self._embedding_regularizer)
            net_reg = set_regularizer(self._net_regularizer)
            for name, param in self.named_parameters():
                if param.requires_grad:
                    if "embedding_layer" in name:
                        if self._embedding_regularizer:
                            for emb_p, emb_lambda in emb_reg:
                                total_loss += (emb_lambda / emb_p) * torch.norm(param, emb_p) ** emb_p
                    else:
                        if self._net_regularizer:
                            for net_p, net_lambda in net_reg:
                                total_loss += (net_lambda / net_p) * torch.norm(param, net_p) ** net_p
        return total_loss

    def init_weights(self, m):
        if type(m) == nn.ModuleDict:
            for k, v in m.items():
                if type(v) == nn.Embedding:
                    if "pretrained_emb" in self.feature_map.feature_specs[k]: # skip pretrained
                        continue
                    try:
                        initialize_emb = eval(self._embedding_initializer)
                        if v.padding_idx is not None:
                            # using the last index as padding_idx
                            initialize_emb(v.weight[0:-1, :])
                        else:
                            initialize_emb(v.weight)
                    except:
                        raise NotImplementedError("embedding_initializer={} is not supported."\
                                                  .format(self._embedding_initializer))
                elif type(v) == nn.Linear:
                    nn.init.xavier_normal_(v.weight)
                    if v.bias is not None:
                        v.bias.data.fill_(0)
        elif type(m) == nn.Linear:
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0)
        
    def to_device(self, inputs):
        self.batch_size = 0
        for k in inputs.keys():
            inputs[k] = inputs[k].to(self.device)
            if self.batch_size < 1:
                self.batch_size = inputs[k].size(0)
        return inputs

    def on_batch_end(self, train_generator, batch_index, logs={}):
        self._total_batches += 1
        if (batch_index + 1) % self._eval_interval_batches == 0 or (batch_index + 1) % self._batches_per_epoch == 0:
            val_logs = self.evaluate(train_generator, self.valid_gen)
            epoch = round(float(self._total_batches) / self._batches_per_epoch, 2)
            self.checkpoint_and_earlystop(epoch, val_logs)
            logging.info("--- {}/{} batches finished ---".format(batch_index + 1, self._batches_per_epoch))

    def reduce_learning_rate(self, factor=0.1, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            reduced_lr = max(param_group["lr"] * factor, min_lr)
            param_group["lr"] = reduced_lr
        return reduced_lr

    def checkpoint_and_earlystop(self, epoch, logs, min_delta=1e-6):
        monitor_value = self._monitor.get_value(logs)
        if (self._monitor_mode == "min" and monitor_value > self._best_metric - min_delta) or \
           (self._monitor_mode == "max" and monitor_value < self._best_metric + min_delta):
            self._stopping_steps += 1
            logging.info("Monitor({}) STOP: {:.6f} !".format(self._monitor_mode, monitor_value))
            if self._reduce_lr_on_plateau:
                current_lr = self.reduce_learning_rate()
                logging.info("Reduce learning rate on plateau: {:.6f}".format(current_lr))
                logging.info("Load best model: {}".format(self.checkpoint))
                self.load_weights(self.checkpoint)
        else:
            self._stopping_steps = 0
            self._best_metric = monitor_value
            if self._save_best_only:
                logging.info("Save best model: monitor({}): {:.6f}"\
                             .format(self._monitor_mode, monitor_value))
                self.save_weights(self.checkpoint)
        if self._stopping_steps * self._eval_interval_epochs >= self._patience:
            self._stop_training = True
            logging.info("Early stopping at epoch={:g}".format(epoch))
        if not self._save_best_only:
            self.save_weights(self.checkpoint)
     
    def fit(self, train_generator, epochs=1, valid_generator=None,
            verbose=0, max_gradient_norm=10., **kwargs):
        self.valid_gen = valid_generator
        self._max_gradient_norm = max_gradient_norm
        self._best_metric = np.Inf if self._monitor_mode == "min" else -np.Inf
        self._stopping_steps = 0
        self._total_batches = 0
        self._batches_per_epoch = len(train_generator)
        self._eval_interval_batches = int(np.ceil(self._eval_interval_epochs * self._batches_per_epoch))
        self._stop_training = False
        self._verbose = verbose
        
        logging.info("**** Start training: {} batches/epoch ****".format(self._batches_per_epoch))
        for epoch in range(epochs):
            epoch_loss = self.train_on_epoch(train_generator, epoch)
            logging.info("Train loss: {:.6f}".format(epoch_loss))
            if self._stop_training:
                break
            else:
                logging.info("************ Epoch={} end ************".format(epoch + 1))
        logging.info("Training finished.")
        logging.info("Load best model: {}".format(self.checkpoint))
        self.load_weights(self.checkpoint)

    def train_on_epoch(self, train_generator, epoch):
        epoch_loss = 0
        model = self.train()
        batch_generator = train_generator
        if self._verbose > 0:
            batch_generator = tqdm(train_generator, disable=False)#, file=sys.stdout)
        for batch_index, batch_data in enumerate(batch_generator):
            self.optimizer.zero_grad()
            return_dict = model.forward(batch_data)
            loss = return_dict["loss"]
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), self._max_gradient_norm)
            self.optimizer.step()
            epoch_loss += loss.item()
            self.on_batch_end(train_generator, batch_index)
            if self._stop_training:
                break
        return epoch_loss / self._batches_per_epoch

    def evaluate(self, train_generator, valid_generator):
        logging.info("--- Start evaluation ---")
        self.eval()  # set to evaluation mode
        with torch.no_grad():
            user_vecs = []
            item_vecs = []
            for user_batch in valid_generator.user_loader:
                user_vec = self.user_tower(user_batch)
                user_vecs.extend(user_vec.data.cpu().numpy())
            for item_batch in valid_generator.item_loader:
                item_vec = self.item_tower(item_batch)
                item_vecs.extend(item_vec.data.cpu().numpy())
            user_vecs = np.array(user_vecs, np.float64)
            item_vecs = np.array(item_vecs, np.float64)
            val_logs = evaluate_metrics(user_vecs,
                                        item_vecs,
                                        train_generator.user2items_dict,
                                        valid_generator.user2items_dict,
                                        valid_generator.query_indexes,
                                        self._validation_metrics)
            return val_logs
                
    def save_weights(self, checkpoint):
        torch.save(self.state_dict(), checkpoint)
    
    def load_weights(self, checkpoint):
        self.load_state_dict(torch.load(checkpoint, map_location=self.device))

    def count_parameters(self, count_embedding=True):
        total_params = 0
        for name, param in self.named_parameters(): 
            if not count_embedding and "embedding" in name:
                continue
            if param.requires_grad:
                total_params += param.numel()
        logging.info("Total number of parameters: {}.".format(total_params))
```

```python id="kr-dWPZJdW_o"
class MF(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="MF",
                 gpu=-1, 
                 learning_rate=1e-3, 
                 embedding_initializer="lambda w: nn.init.normal_(w, std=1e-4)", 
                 embedding_dim=10, 
                 num_negs=1,
                 regularizer=None,
                 embedding_dropout=0,
                 similarity_score="dot",
                 **kwargs):
        super(MF, self).__init__(feature_map, 
                                 model_id=model_id, 
                                 gpu=gpu, 
                                 embedding_regularizer=regularizer,
                                 num_negs=num_negs,
                                 embedding_initializer=embedding_initializer,
                                 **kwargs)
        self.similarity_score = similarity_score
        self.embedding_dim = embedding_dim
        self.embedding_layer = EmbeddingLayer(feature_map, embedding_dim)
        assert similarity_score in ["dot", "cosine", "sigmoid"]
        self.dropout = nn.Dropout(embedding_dropout)
        self.compile(lr=learning_rate, **kwargs)
        
    def forward(self, inputs):
        """
        Inputs: [user_dict, item_dict, label]
        """
        user_dict, item_dict, labels = inputs[0:3]
        user_vecs = self.user_tower(user_dict)
        user_vecs = self.dropout(user_vecs)
        item_vecs = self.item_tower(item_dict)
        y_pred = torch.bmm(item_vecs.view(user_vecs.size(0), self.num_negs + 1, -1), 
                           user_vecs.unsqueeze(-1)).squeeze(-1)
        if self.similarity_score == "sigmoid":
            y_pred = y_pred.sigmoid()
        loss = self.get_total_loss(y_pred, labels)
        return_dict = {"loss": loss, "y_pred": y_pred}
        return return_dict

    def user_tower(self, inputs):
        user_inputs = self.to_device(inputs)
        user_vec = self.embedding_layer(user_inputs, feature_source="user")
        if self.similarity_score == "cosine":
            user_vec = F.normalize(user_vec)
        return user_vec

    def item_tower(self, inputs):
        item_inputs = self.to_device(inputs)
        item_vec = self.embedding_layer(item_inputs, feature_source="item")
        if self.similarity_score == "cosine":
            item_vec = F.normalize(item_vec)
        return item_vec
```

```python id="aKlhpKe6dLee"
class BiasMF(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="BiasMF",
                 gpu=-1, 
                 learning_rate=1e-3, 
                 embedding_initializer="lambda w: nn.init.normal_(w, std=1e-4)", 
                 embedding_dim=10, 
                 user_id_field="user_id",
                 item_id_field="item_id",
                 enable_bias=False,
                 num_negs=1,
                 regularizer=None,
                 embedding_dropout=0,
                 similarity_score="dot",
                 **kwargs):
        super(BiasMF, self).__init__(feature_map, 
                                     model_id=model_id, 
                                     gpu=gpu, 
                                     embedding_regularizer=regularizer,
                                     num_negs=num_negs,
                                     embedding_initializer=embedding_initializer,
                                     **kwargs)
        self.similarity_score = similarity_score
        self.embedding_dim = embedding_dim
        self.embedding_layer = EmbeddingLayer(feature_map, embedding_dim)
        assert similarity_score in ["dot", "cosine", "sigmoid"]
        self.enable_bias = enable_bias
        if self.enable_bias:
            self.user_bias = EmbeddingLayer(feature_map, 1,
                                            disable_sharing_pretrain=True, 
                                            required_feature_columns=[user_id_field])
            self.item_bias = EmbeddingLayer(feature_map, 1, 
                                            disable_sharing_pretrain=True, 
                                            required_feature_columns=[item_id_field])
            self.global_bias = nn.Parameter(torch.zeros(1))
        self.dropout = nn.Dropout(embedding_dropout)
        self.compile(lr=learning_rate, **kwargs)
        
    def forward(self, inputs):
        """
        Inputs: [user_dict, item_dict, label]
        """
        user_dict, item_dict, labels = inputs[0:3]
        user_vecs = self.user_tower(user_dict)
        user_vecs = self.dropout(user_vecs)
        item_vecs = self.item_tower(item_dict)
        y_pred = torch.bmm(item_vecs.view(user_vecs.size(0), self.num_negs + 1, -1), 
                           user_vecs.unsqueeze(-1)).squeeze(-1)
        if self.enable_bias:
            # user_bias and global_bias only influence training, but not inference phase
            y_pred += self.user_bias(self.to_device(user_dict)) + self.global_bias
        if self.similarity_score == "sigmoid":
            y_pred = y_pred.sigmoid()
        loss = self.get_total_loss(y_pred, labels)
        return_dict = {"loss": loss, "y_pred": y_pred}
        return return_dict

    def user_tower(self, inputs):
        user_inputs = self.to_device(inputs)
        user_vec = self.embedding_layer(user_inputs, feature_source="user")
        if self.similarity_score == "cosine":
            user_vec = F.normalize(user_vec)
        if self.enable_bias: 
            user_vec = torch.cat([user_vec, torch.ones(user_vec.size(0), 1).to(self.device)], dim=-1)
        return user_vec

    def item_tower(self, inputs):
        item_inputs = self.to_device(inputs)
        item_vec = self.embedding_layer(item_inputs, feature_source="item")
        if self.similarity_score == "cosine":
            item_vec = F.normalize(item_vec)
        if self.enable_bias:
            item_vec = torch.cat([item_vec, self.item_bias(item_inputs)], dim=-1)
        return item_vec
```

```python id="uTkJJ-cjdLcJ"
class YoutubeDNN(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="YoutubeDNN", 
                 gpu=-1, 
                 learning_rate=1e-3, 
                 embedding_initializer="lambda w: nn.init.normal_(w, std=1e-4)", 
                 embedding_dim=10, 
                 output_dim=10,
                 user_hidden_units=[],
                 user_hidden_activations="ReLU",
                 user_final_activation=None,
                 num_negs=1,
                 embedding_dropout=0,
                 net_dropout=0,
                 batch_norm=False,
                 net_regularizer=None,
                 embedding_regularizer=None,
                 similarity_score="dot",
                 sample_weighting=False,
                 **kwargs):
        super(YoutubeDNN, self).__init__(feature_map, 
                                         model_id=model_id, 
                                         gpu=gpu, 
                                         embedding_regularizer=embedding_regularizer,
                                         net_regularizer=net_regularizer,
                                         num_negs=num_negs,
                                         sample_weighting=sample_weighting,
                                         embedding_initializer=embedding_initializer,
                                         **kwargs)
        self.similarity_score = similarity_score
        self.embedding_dim = embedding_dim
        self.embedding_layer = EmbeddingLayer(feature_map, embedding_dim)
        num_user_fields = feature_map.get_num_fields(feature_source="user")
        self.user_dnn_layer = MLP_Layer(input_dim=embedding_dim * num_user_fields,
                                        output_dim=output_dim, 
                                        hidden_units=user_hidden_units,
                                        hidden_activations=user_hidden_activations,
                                        final_activation=user_final_activation, 
                                        dropout_rates=net_dropout, 
                                        batch_norm=batch_norm) \
                              if user_hidden_units is not None else None
        self.dropout = nn.Dropout(embedding_dropout)
        self.compile(lr=learning_rate, **kwargs)
            
    def forward(self, inputs):
        """
        Inputs: [user_dict, item_dict, label]
        """
        user_dict, item_dict, labels = inputs[0:3]
        user_vecs = self.user_tower(user_dict)
        user_vecs = self.dropout(user_vecs)
        item_vecs = self.item_tower(item_dict)
        y_pred = torch.bmm(item_vecs.view(-1, self.num_negs + 1, self.embedding_dim), 
                           user_vecs.unsqueeze(-1)).squeeze(-1)
        loss = self.get_total_loss(y_pred, labels)
        return_dict = {"loss": loss, "y_pred": y_pred}
        return return_dict

    def user_tower(self, inputs):
        user_inputs = self.to_device(inputs)
        user_embedding = self.embedding_layer(user_inputs, feature_source="user")
        user_vec = user_embedding.flatten(start_dim=1)
        if self.user_dnn_layer is not None:
            user_vec = self.user_dnn_layer(user_vec)
        if self.similarity_score == "cosine":
            user_vec = F.normalize(user_vec)
        return user_vec

    def item_tower(self, inputs):
        item_inputs = self.to_device(inputs)
        item_vec = self.embedding_layer(item_inputs, feature_source="item")
        if self.similarity_score == "cosine":
            item_vec = F.normalize(item_vec)
        return item_vec
```

```python id="IJcmon-IdLZG"
class SimpleX(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="SimpleX", 
                 gpu=-1, 
                 learning_rate=1e-3, 
                 embedding_initializer="lambda w: nn.init.normal_(w, std=1e-4)", 
                 embedding_dim=10, 
                 user_id_field="user_id",
                 item_id_field="item_id",
                 user_history_field="user_history",
                 enable_bias=False,
                 num_negs=1,
                 net_dropout=0,
                 aggregator="mean",
                 gamma=0.5,
                 attention_dropout=0,
                 batch_norm=False,
                 net_regularizer=None,
                 embedding_regularizer=None,
                 similarity_score="dot",
                 **kwargs):
        super(SimpleX, self).__init__(feature_map, 
                                      model_id=model_id, 
                                      gpu=gpu, 
                                      embedding_regularizer=embedding_regularizer,
                                      net_regularizer=net_regularizer,
                                      num_negs=num_negs,
                                      embedding_initializer=embedding_initializer,
                                      **kwargs)
        self.similarity_score = similarity_score
        self.embedding_dim = embedding_dim
        self.user_id_field = user_id_field
        self.user_history_field = user_history_field
        self.embedding_layer = EmbeddingDictLayer(feature_map, embedding_dim)
        self.behavior_aggregation = BehaviorAggregator(embedding_dim, 
                                                       gamma=gamma,
                                                       aggregator=aggregator, 
                                                       dropout_rate=attention_dropout)
        self.enable_bias = enable_bias
        if self.enable_bias:
            self.user_bias = EmbeddingLayer(feature_map, 1,
                                            disable_sharing_pretrain=True, 
                                            required_feature_columns=[user_id_field])
            self.item_bias = EmbeddingLayer(feature_map, 1, 
                                            disable_sharing_pretrain=True, 
                                            required_feature_columns=[item_id_field])
            self.global_bias = nn.Parameter(torch.zeros(1))
        self.dropout = nn.Dropout(net_dropout)
        self.compile(lr=learning_rate, **kwargs)
            
    def forward(self, inputs):
        """
        Inputs: [user_dict, item_dict, label]
        """
        user_dict, item_dict, labels = inputs[0:3]
        user_vecs = self.user_tower(user_dict)
        user_vecs = self.dropout(user_vecs)
        item_vecs = self.item_tower(item_dict)
        y_pred = torch.bmm(item_vecs.view(user_vecs.size(0), self.num_negs + 1, -1), 
                           user_vecs.unsqueeze(-1)).squeeze(-1)
        if self.enable_bias: # user_bias and global_bias only influence training, but not inference for ranking
            y_pred += self.user_bias(self.to_device(user_dict)) + self.global_bias
        loss = self.get_total_loss(y_pred, labels)
        return_dict = {"loss": loss, "y_pred": y_pred}
        return return_dict

    def user_tower(self, inputs):
        user_inputs = self.to_device(inputs)
        user_emb_dict = self.embedding_layer(user_inputs, feature_source="user")
        user_id_emb = user_emb_dict[self.user_id_field]
        user_history_emb = user_emb_dict[self.user_history_field]
        user_vec = self.behavior_aggregation(user_id_emb, user_history_emb)
        if self.similarity_score == "cosine":
            user_vec = F.normalize(user_vec)
        if self.enable_bias: 
            user_vec = torch.cat([user_vec, torch.ones(user_vec.size(0), 1).to(self.device)], dim=-1)
        return user_vec

    def item_tower(self, inputs):
        item_inputs = self.to_device(inputs)
        item_vec_dict = self.embedding_layer(item_inputs, feature_source="item")
        item_vec = self.embedding_layer.dict2tensor(item_vec_dict)
        if self.similarity_score == "cosine":
            item_vec = F.normalize(item_vec)
        if self.enable_bias:
            item_vec = torch.cat([item_vec, self.item_bias(item_inputs)], dim=-1)
        return item_vec


class BehaviorAggregator(nn.Module):
    def __init__(self, embedding_dim, gamma=0.5, aggregator="mean", dropout_rate=0.):
        super(BehaviorAggregator, self).__init__()
        self.aggregator = aggregator
        self.gamma = gamma
        self.W_v = nn.Linear(embedding_dim, embedding_dim, bias=False)
        if self.aggregator in ["cross_attention", "self_attention"]:
            self.W_k = nn.Sequential(nn.Linear(embedding_dim, embedding_dim),
                                     nn.Tanh())
            self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
            if self.aggregator == "self_attention":
                self.W_q = nn.Parameter(torch.Tensor(embedding_dim, 1))
                nn.init.xavier_normal_(self.W_q)

    def forward(self, id_emb, sequence_emb):
        out = id_emb
        if self.aggregator == "mean":
            out = self.average_pooling(sequence_emb)
        elif self.aggregator == "cross_attention":
            out = self.cross_attention(id_emb, sequence_emb)
        elif self.aggregator == "self_attention":
            out = self.self_attention(sequence_emb)
        return self.gamma * id_emb + (1 - self.gamma) * out

    def cross_attention(self, id_emb, sequence_emb):
        key = self.W_k(sequence_emb) # b x seq_len x attention_dim
        mask = sequence_emb.sum(dim=-1) == 0
        attention = torch.bmm(key, id_emb.unsqueeze(-1)).squeeze(-1) # b x seq_len
        attention = self.masked_softmax(attention, mask)
        if self.dropout is not None:
            attention = self.dropout(attention)
        output = torch.bmm(attention.unsqueeze(1), sequence_emb).squeeze(1)
        return self.W_v(output)

    def self_attention(self, sequence_emb):
        key = self.W_k(sequence_emb) # b x seq_len x attention_dim
        mask = sequence_emb.sum(dim=-1) == 0
        attention = torch.matmul(key, self.W_q).squeeze(-1) # b x seq_len
        attention = self.masked_softmax(attention, mask)
        if self.dropout is not None:
            attention = self.dropout(attention)
        output = torch.bmm(attention.unsqueeze(1), sequence_emb).squeeze(1)
        return self.W_v(output)

    def average_pooling(self, sequence_emb):
        mask = sequence_emb.sum(dim=-1) != 0
        mean = sequence_emb.sum(dim=1) / (mask.float().sum(dim=-1, keepdim=True) + 1.e-12)
        return self.W_v(mean)

    def masked_softmax(self, X, mask):
        # use the following softmax to avoid nans when a sequence is entirely masked
        X = X.masked_fill_(mask, 0)
        e_X = torch.exp(X)
        return e_X / (e_X.sum(dim=1, keepdim=True) + 1.e-12)
```

<!-- #region id="p5sobOFddLWO" -->
## Metrics
<!-- #endregion -->

```python id="bC8AYTwUd79u"
def evaluate_metrics(user_embs, 
                     item_embs, 
                     train_user2items, 
                     valid_user2items, 
                     query_indexes,
                     metrics,
                     parallel=True):
    logging.info("Evaluating metrics for {:d} users...".format(len(user_embs)))
    metric_callers = []
    max_topk = 0
    for metric in metrics:
        try:
            metric_callers.append(eval(metric))
            max_topk = max(max_topk, int(metric.split("k=")[-1].strip(")")))
        except:
            raise NotImplementedError('metrics={} not implemented.'.format(metric))
    
    if parallel:
        num_workers = 2
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            chunk_size = int(np.ceil(len(user_embs) / float(num_workers)))
            tasks = []
            for idx in range(0, len(user_embs), chunk_size):
                chunk_user_embs = user_embs[idx: (idx + chunk_size), :]
                chunk_query_indexes = query_indexes[idx: (idx + chunk_size)]
                tasks.append(executor.submit(evaluate_block, chunk_user_embs, item_embs, chunk_query_indexes,
                                             train_user2items, valid_user2items, metric_callers, max_topk))
            results = [res for future in tqdm(as_completed(tasks), total=len(tasks)) for res in future.result()]
    else:
        results = evaluate_block(user_embs, item_embs, query_indexes, train_user2items, 
                                 valid_user2items, metric_callers, max_topk)
    average_result = np.average(np.array(results), axis=0).tolist()
    return_dict = dict(zip(metrics, average_result))
    logging.info('[Metrics] ' + ' - '.join('{}: {:.6f}'.format(k, v) for k, v in zip(metrics, average_result)))
    return return_dict


def evaluate_block(chunk_user_embs, item_embs, chunk_query_indexes, train_user2items, 
                   valid_user2items, metric_callers, max_topk):
    sim_matrix = np.dot(chunk_user_embs, item_embs.T)
    for i, query_index in enumerate(chunk_query_indexes):
        train_items = train_user2items[query_index]
        sim_matrix[i, train_items] = -np.inf # remove clicked items in train data
    item_indexes = np.argpartition(-sim_matrix, max_topk)[:, 0:max_topk]
    sim_matrix = sim_matrix[np.arange(item_indexes.shape[0])[:, None], item_indexes]
    sorted_idxs = np.argsort(-sim_matrix, axis=1)
    topk_items_chunk = item_indexes[np.arange(sorted_idxs.shape[0])[:, None], sorted_idxs]
    true_items_chunk = [valid_user2items[query_index] for query_index in chunk_query_indexes]
    chunk_results = [[fn(topk_items, true_items) for fn in metric_callers] \
                     for topk_items, true_items in zip(topk_items_chunk, true_items_chunk)]
    return chunk_results


class Recall(object):
    """Recall metric."""
    def __init__(self, k=1):
        self.topk = k

    def __call__(self, topk_items, true_items):
        topk_items = topk_items[:self.topk]
        hit_items = set(true_items) & set(topk_items)
        recall = len(hit_items) / (len(true_items) + 1e-12)
        return recall


class NormalizedRecall(object):
    """Recall metric normalized to max 1."""
    def __init__(self, k=1):
        self.topk = k

    def __call__(self, topk_items, true_items):
        topk_items = topk_items[:self.topk]
        hit_items = set(true_items) & set(topk_items)
        recall = len(hit_items) / min(self.topk, len(true_items) + 1e-12)
        return recall


class Precision(object):
    """Precision metric."""
    def __init__(self, k=1):
        self.topk = k

    def __call__(self, topk_items, true_items):
        topk_items = topk_items[:self.topk]
        hit_items = set(true_items) & set(topk_items)
        precision = len(hit_items) / (self.topk + 1e-12)
        return precision


class F1(object):
    def __init__(self, k=1):
        self.precision_k = Precision(k)
        self.recall_k = Recall(k)

    def __call__(self, topk_items, true_items):
        p = self.precision_k(topk_items, true_items)
        r = self.recall_k(topk_items, true_items)
        f1 = 2 * p * r / (p + r + 1e-12)
        return f1


class DCG(object):
    """ Calculate discounted cumulative gain
    """
    def __init__(self, k=1):
        self.topk = k

    def __call__(self, topk_items, true_items):
        topk_items = topk_items[:self.topk]
        true_items = set(true_items)
        dcg = 0
        for i, item in enumerate(topk_items):
            if item in true_items:
                dcg += 1 / np.log(2 + i)
        return dcg


class NDCG(object):
    """Normalized discounted cumulative gain metric."""
    def __init__(self, k=1):
        self.topk = k

    def __call__(self, topk_items, true_items):
        topk_items = topk_items[:self.topk]
        dcg_fn = DCG(k=self.topk)
        idcg = dcg_fn(true_items[:self.topk], true_items)
        dcg = dcg_fn(topk_items, true_items)
        return dcg / (idcg + 1e-12)


class MRR(object):
    """MRR metric"""
    def __init__(self, k=1):
        self.topk = k

    def __call__(self, topk_items, true_items):
        topk_items = topk_items[:self.topk]
        true_items = set(true_items)
        mrr = 0
        for i, item in enumerate(topk_items):
            if item in true_items:
                mrr += 1 / (i + 1.0)
        return mrr


class HitRate(object):
    def __init__(self, k=1):
        self.topk = k

    def __call__(self, topk_items, true_items):
        topk_items = topk_items[:self.topk]
        hit_items = set(true_items) & set(topk_items)
        hit_rate = 1 if len(hit_items) > 0 else 0
        return hit_rate


class MAP(object):
    """
    Calculate mean average precision.
    """
    def __init__(self, k=1):
        self.topk = k

    def __call__(self, topk_items, true_items):
        topk_items = topk_items[:self.topk]
        true_items = set(true_items)
        pos = 0
        precision = 0
        for i, item in enumerate(topk_items):
            if item in true_items:
                pos += 1
                precision += pos / (i + 1.0)
        return precision / (pos + 1e-12)
```

<!-- #region id="xkLnUk7XeC_5" -->
## Main
<!-- #endregion -->

```python id="ws5wotL5bMaV"
def load_model_config(config_dir, experiment_id):
    params = dict()
    model_configs = glob.glob(os.path.join(config_dir, "model_config.yaml"))
    if not model_configs:
        model_configs = glob.glob(os.path.join(config_dir, "model_config/*.yaml"))
    found_keys = []
    for config in model_configs:
        with open(config, "r") as cfg:
            config_dict = yaml.load(cfg)
            if "Base" in config_dict:
                params.update(config_dict["Base"])
                found_keys.append("Base")
            if experiment_id in config_dict:
                params.update(config_dict[experiment_id])
                found_keys.append(experiment_id)
        if len(found_keys) == 2:
            break
    if "dataset_id" not in params:
        raise RuntimeError("experiment_id={} is not valid in config.".format(experiment_id))
    params["model_id"] = experiment_id
    return params
```

```python id="SscSyoecbe_J"
def load_dataset_config(config_dir, dataset_id):
    params = dict()
    dataset_configs = glob.glob(os.path.join(config_dir, "dataset_config.yaml"))
    if not dataset_configs:
        dataset_configs = glob.glob(os.path.join(config_dir, "dataset_config/*.yaml"))
    for config in dataset_configs:
        with open(config, "r") as cfg:
            config_dict = yaml.load(cfg)
            if dataset_id in config_dict:
                params.update(config_dict[dataset_id])
                break
    return params
```

```python id="A2rNFFmObe7x"
def enumerate_params(config_file, exclude_expid=[]):
    with open(config_file, "r") as cfg:
        config_dict = yaml.load(cfg)
    # tuning space
    tune_dict = config_dict["tuner_space"]
    for k, v in tune_dict.items():
        if not isinstance(v, list):
            tune_dict[k] = [v]
    experiment_id = config_dict["base_expid"]
    if "model_config" in config_dict:
        model_dict = config_dict["model_config"][experiment_id]
    else:
        base_config_dir = config_dict.get("base_config", os.path.dirname(config_file))
        model_dict = load_model_config(base_config_dir, experiment_id)

    dataset_id = config_dict.get("dataset_id", model_dict["dataset_id"])
    if "dataset_config" in config_dict:
        dataset_dict = config_dict["dataset_config"][dataset_id]
    else:
        dataset_dict = load_dataset_config(base_config_dir, dataset_id)
        
    if model_dict["dataset_id"] == "TBD": # rename base expid
        model_dict["dataset_id"] = dataset_id
        experiment_id = model_dict["model"] + "_" + dataset_id
        
    # key checking
    tuner_keys = set(tune_dict.keys())
    base_keys = set(model_dict.keys()).union(set(dataset_dict.keys()))
    if len(tuner_keys - base_keys) > 0:
        raise RuntimeError("Invalid params in tuner config: {}".format(tuner_keys - base_keys))

    config_dir = config_file.replace(".yaml", "")
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)

    # enumerate dataset para combinations
    dataset_dict = {k: tune_dict[k] if k in tune_dict else [v] for k, v in dataset_dict.items()}
    dataset_para_keys = list(dataset_dict.keys())
    dataset_para_combs = dict()
    for idx, values in enumerate(itertools.product(*map(dataset_dict.get, dataset_para_keys))):
        dataset_params = dict(zip(dataset_para_keys, values))
        if dataset_params["data_format"] == "h5":
            dataset_para_combs[dataset_id] = dataset_params
        else:
            hash_id = hashlib.md5(print_to_json(dataset_params).encode("utf-8")).hexdigest()[0:8]
            dataset_para_combs[dataset_id + "_{}".format(hash_id)] = dataset_params

    # dump dataset para combinations to config file
    dataset_config = os.path.join(config_dir, "dataset_config.yaml")
    with open(dataset_config, "w") as fw:
        yaml.dump(dataset_para_combs, fw, default_flow_style=None, indent=4)

    # enumerate model para combinations
    model_dict = {k: tune_dict[k] if k in tune_dict else [v] for k, v in model_dict.items()}
    model_para_keys = list(model_dict.keys())
    model_param_combs = dict()
    for idx, values in enumerate(itertools.product(*map(model_dict.get, model_para_keys))):
        model_param_combs[idx + 1] = dict(zip(model_para_keys, values))
        
    # update dataset_id into model params
    merged_param_combs = dict()
    for idx, item in enumerate(itertools.product(model_param_combs.values(),
                                                 dataset_para_combs.keys())):
        para_dict = item[0]
        para_dict["dataset_id"] = item[1]
        random_number = ""
        if para_dict["debug_mode"]:
            random_number = str(np.random.randint(1e8)) # add a random number to avoid duplicate during debug
        hash_id = hashlib.md5((print_to_json(para_dict) + random_number).encode("utf-8")).hexdigest()[0:8]
        hash_expid = experiment_id + "_{:03d}_{}".format(idx + 1, hash_id)
        if hash_expid not in exclude_expid:
            merged_param_combs[hash_expid] = para_dict.copy()

    # dump model para combinations to config file
    model_config = os.path.join(config_dir, "model_config.yaml")
    with open(model_config, "w") as fw:
        yaml.dump(merged_param_combs, fw, default_flow_style=None, indent=4)
    print("Enumerate all tuner configurations done.")    
    return config_dir
```

<!-- #region id="xsn1fruVCUyh" -->
## Run
<!-- #endregion -->

```python id="-KDxMW0YLeJL"
class Args:
    dataset = 'amazonbooks_x0'
    data_root = '/content'
    data_format = 'csv'
    train_data = '/content/train.csv'
    valid_data = '/content/test.csv'
    item_corpus = '/content/item_corpus.csv'
    nrows = 'null'
    data_block_size = -1
    min_categr_count = 1
    query_index = 'query_index'
    corpus_index = 'corpus_index'
    feature_cols = [
        {'name': 'query_index', 'active': True, 'dtype': int, 'type': 'index'},
        {'name': 'corpus_index', 'active': True, 'dtype': int, 'type': 'index'},
        {'name': 'user_id', 'active': True, 'dtype': str, 'type': 'categorical', 'source': 'user'},
        {'name': 'item_id', 'active': True, 'dtype': str, 'type': 'categorical', 'source': 'item'},
    ]
    label_col = {'name': 'label', 'dtype': float}

    model_root = '../checkpoints/'
    num_workers = 3
    verbose = 1
    patience = 3
    save_best_only = True
    eval_interval_epochs = 1
    debug_mode = False
    model = 'SimpleX'
    dataset_id = 'TBD'
    metrics = ['Recall(k=20)', 'Recall(k=50)', 'NDCG(k=20)', 'NDCG(k=50)', 'HitRate(k=20)', 'HitRate(k=50)']
    optimizer = 'adam'
    learning_rate = 1.0e-3
    batch_size = 1024
    num_negs = 20
    embedding_dim = 64
    aggregator = 'mean'
    gamma = 0.5
    user_id_field = 'user_id'
    item_id_field = 'item_id'
    user_history_field = 'user_history'
    embedding_regularizer = 0
    net_regularizer = 0
    net_dropout = 0
    attention_dropout = 0
    enable_bias = False
    similarity_score = 'dot'
    loss = 'PairwiseLogisticLoss'
    margin = 0
    negative_weight = 'null'
    sampling_num_process = 1
    fix_sampling_seeds = True
    ignore_pos_items = False
    epochs = 100
    shuffle = True
    seed = 2019
    monitor = 'Recall(k=20)'
    monitor_mode = 'max'


args = Args()
```

```python id="o7NpwH6_FlQw"
!python -u benchmark.py --version {} --config {} --expid {} --gpu {}
```

```python id="zNidpNnseDfc"

```
