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

```python colab={"base_uri": "https://localhost:8080/"} id="BlhAa6WyUZKj" executionInfo={"status": "ok", "timestamp": 1641897135496, "user_tz": -330, "elapsed": 1303, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="d2f6f6b8-60a3-4969-f96e-74d392909db3"
!wget -q --show-progress https://github.com/RecoHut-Stanzas/S516304/raw/main/data/tiny_data/train_sample.csv
!wget -q --show-progress https://github.com/RecoHut-Stanzas/S516304/raw/main/data/tiny_data/valid_sample.csv
!wget -q --show-progress https://github.com/RecoHut-Stanzas/S516304/raw/main/data/tiny_data/test_sample.csv
```

```python id="-BVin3TuxI9S"
!pip install -q pytorch-lightning

!git clone --branch US632593 https://github.com/RecoHut-Projects/recohut.git
!pip install -U ./recohut
```

```python id="DihrHtzg7SP_"

```

```python id="fJptZHOChOrv"
import itertools
import numpy as np
import pandas as pd
import h5py
import six
import pickle
import sklearn.preprocessing as sklearn_preprocess
from collections import Counter, OrderedDict, defaultdict
import io
import os
import logging
import json
from datetime import datetime, date

import torch

from pytorch_lightning import LightningDataModule

from recohut.datasets.bases.common import Dataset as BaseDataset
from recohut.utils.common_utils import download_url
```

```python id="Ea_20yl9U1dH"
class Tokenizer(object):
    def __init__(self, topk_words=None, na_value=None, min_freq=1, splitter=None, 
                 lower=False, oov_token=0, max_len=0, padding_type="pre"):
        self._topk_words = topk_words
        self._na_value = na_value
        self._min_freq = min_freq
        self._lower = lower
        self._splitter = splitter
        self.oov_token = oov_token # use 0 for __OOV__
        self.word_counts = Counter()
        self.vocab = dict()
        self.vocab_size = 0 # include oov and padding
        self.max_len = max_len
        self.padding_type = padding_type

    def fit_on_texts(self, texts, use_padding=True):
        tokens = list(texts)
        if self._splitter is not None: # for sequence
            text_splits = [text.split(self._splitter) for text in texts if not pd.isnull(text)]
            if self.max_len == 0:
                self.max_len = max(len(x) for x in text_splits)
            tokens = list(itertools.chain(*text_splits))
        if self._lower:
            tokens = [tk.lower() for tk in tokens]
        if self._na_value is not None:
            tokens = [tk for tk in tokens if tk != self._na_value]
        self.word_counts = Counter(tokens)
        words = [token for token, count in self.word_counts.most_common() if count >= self._min_freq]
        self.word_counts.clear() # empty the dict to save memory
        if self._topk_words:
            words = words[0:self._topk_words]
        self.vocab = dict((token, idx) for idx, token in enumerate(words, 1 + self.oov_token))
        self.vocab["__OOV__"] = self.oov_token
        if use_padding:
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
        sequence_list = self.padding(sequence_list, maxlen=self.max_len, value=self.vocab_size - 1,
                                padding=self.padding_type, truncating=self.padding_type)
        return np.array(sequence_list)
    
    def load_pretrained_embedding(self, feature_name, pretrain_path, embedding_dim, output_path):
        with h5py.File(pretrain_path, 'r') as hf:
            keys = hf["key"][:]
            pretrained_vocab = dict(zip(keys, range(len(keys))))
            pretrained_emb = hf["value"][:]
        embedding_matrix = np.random.normal(loc=0, scale=1.e-4, size=(self.vocab_size, embedding_dim))
        for word, idx in self.vocab.items():
            if word in pretrained_vocab:
                embedding_matrix[idx] = pretrained_emb[pretrained_vocab[word]]
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with h5py.File(output_path, 'a') as hf:
            hf.create_dataset(feature_name, data=embedding_matrix)

    def set_vocab(self, vocab):
        self.vocab = vocab
        self.vocab_size = len(self.vocab) + self.oov_token

    @staticmethod
    def padding(sequences, maxlen=None, dtype='int32',
                padding='pre', truncating='pre', value=0.):
        """ Pads sequences (list of list) to the ndarray of same length """
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

```python id="xXhv2rEkcbb2"
class Normalizer(object):
    def __init__(self, normalizer):
        if not callable(normalizer):
            self.callable = False
            if normalizer in ['StandardScaler', 'MinMaxScaler']:
                self.normalizer = getattr(sklearn_preprocess, normalizer)()
            else:
                raise NotImplementedError('normalizer={}'.format(normalizer))
        else:
            # normalizer is a method
            self.normalizer = normalizer
            self.callable = True

    def fit(self, X):
        if not self.callable:
            null_index = np.isnan(X)
            self.normalizer.fit(X[~null_index].reshape(-1, 1))

    def normalize(self, X):
        if self.callable:
            return self.normalizer(X)
        else:
            return self.normalizer.transform(X.reshape(-1, 1)).flatten()
```

```python id="8W9veKeAUwBG"
class FeatureMap(object):
    def __init__(self, dataset_id='ctr'):
        self.dataset_id = dataset_id
        self.num_fields = 0
        self.num_features = 0
        self.feature_len = 0
        self.feature_specs = OrderedDict()
        
    def set_feature_index(self):
        logging.info("Set feature index...")
        idx = 0
        for feature, feature_spec in self.feature_specs.items():
            if feature_spec["type"] != "sequence":
                self.feature_specs[feature]["index"] = idx
                idx += 1
            else:
                seq_indexes = [i + idx for i in range(feature_spec["max_len"])]
                self.feature_specs[feature]["index"] = seq_indexes
                idx += feature_spec["max_len"]
        self.feature_len = idx

    def get_feature_index(self, feature_type=None):
        feature_indexes = []
        if feature_type is not None:
            if not isinstance(feature_type, list):
                feature_type = [feature_type]
            feature_indexes = [feature_spec["index"] for feature, feature_spec in self.feature_specs.items()
                               if feature_spec["type"] in feature_type]
        return feature_indexes

    def load(self, json_file):
        logging.info("Load feature_map from json: " + json_file)
        with io.open(json_file, "r", encoding="utf-8") as fd:
            feature_map = json.load(fd, object_pairs_hook=OrderedDict)
        if feature_map["dataset_id"] != self.dataset_id:
            raise RuntimeError("dataset_id={} does not match to feature_map!".format(self.dataset_id))
        self.num_fields = feature_map["num_fields"]
        self.num_features = feature_map.get("num_features", None)
        self.feature_len = feature_map.get("feature_len", None)
        self.feature_specs = OrderedDict(feature_map["feature_specs"])

    def save(self, json_file):
        logging.info("Save feature_map to json: " + json_file)
        if not os.path.exists(os.path.dirname(json_file)):
            os.makedirs(os.path.dirname(json_file))
        feature_map = OrderedDict()
        feature_map["dataset_id"] = self.dataset_id
        feature_map["num_fields"] = self.num_fields
        feature_map["num_features"] = self.num_features
        feature_map["feature_len"] = self.feature_len
        feature_map["feature_specs"] = self.feature_specs
        with open(json_file, "w") as fd:
            json.dump(feature_map, fd, indent=4)
```

```python id="Cx0lWbjVhZ65"
class CTRDataset(torch.utils.data.Dataset, BaseDataset):
    def __init__(self,
                 data_dir,
                 data_type=None,
                 *args,
                 **kwargs):
        super().__init__(data_dir)
        self.data_type = data_type
        self.pickle_file = os.path.join(self.processed_dir, "feature_encoder.pkl")
        self.json_file = os.path.join(self.processed_dir, "feature_map.json")
        self.feature_cols = self._complete_feature_cols(self.feature_cols)
        self.feature_map = FeatureMap()
        self.encoders = dict()

        if self.data_type == 'train':
            self.darray =  self.load_data(self.raw_paths[0])
            self.num_samples = len(self.darray)
        elif self.data_type == 'valid':
            self.darray = self.load_data(self.raw_paths[1])
            self.validation_samples = len(self.darray)
        elif self.data_type == 'test':
            self.darray = self.load_data(self.raw_paths[2])
            self.test_samples = len(self.darray)
        elif self.data_type is None:
            self._process()

    @property
    def raw_file_names(self):
        raise NotImplementedError

    @property
    def processed_file_names(self):
        return ['feature_encoder.pkl',
                'feature_map.json',
                'train_sample.h5',
                'valid_sample.h5',
                'test_sample.h5']

    def download(self):
        raise NotImplementedError

    def process(self):
        self.fit(self.raw_paths[0])

    @staticmethod
    def _complete_feature_cols(feature_cols):
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

    def read_csv(self, data_path):
        all_cols = self.feature_cols + [self.label_col]
        dtype_dict = dict((x["name"], eval(x["dtype"]) if isinstance(x["dtype"], str) else x["dtype"]) 
                          for x in all_cols)
        ddf = pd.read_csv(data_path, dtype=dtype_dict, memory_map=True) 
        return ddf

    def _preprocess(self, ddf):
        all_cols = [self.label_col] + self.feature_cols[::-1]
        for col in all_cols:
            name = col["name"]
            if name in ddf.columns and ddf[name].isnull().values.any():
                ddf[name] = self._fill_na(col, ddf[name])
            if "preprocess" in col and col["preprocess"] != "":
                preprocess_fn = getattr(self, col["preprocess"])
                ddf[name] = preprocess_fn(ddf, name)
        active_cols = [self.label_col["name"]] + [col["name"] for col in self.feature_cols if col["active"]]
        ddf = ddf.loc[:, active_cols]
        return ddf

    def _fill_na(self, col, series):
        na_value = col.get("na_value")
        if na_value is not None:
            return series.fillna(na_value)
        elif col["dtype"] == "str":
            return series.fillna("")
        else:
            raise RuntimeError("Feature column={} requires to assign na_value!".format(col["name"]))

    def fit(self, train_data, min_categr_count=1, num_buckets=10):           
        ddf = self.read_csv(train_data)
        ddf = self._preprocess(ddf)
        self.feature_map.num_fields = 0
        for col in self.feature_cols:
            if col["active"]:
                name = col["name"]
                self.fit_feature_col(col, ddf, 
                                     min_categr_count=min_categr_count,
                                     num_buckets=num_buckets)
                self.feature_map.num_fields += 1
        self.feature_map.set_feature_index()
        self.save_pickle(self.pickle_file)
        self.feature_map.save(self.json_file)
        
    def fit_feature_col(self, feature_column, ddf, min_categr_count=1, num_buckets=10):
        name = feature_column["name"]
        feature_type = feature_column["type"]
        feature_source = feature_column.get("source", "")
        self.feature_map.feature_specs[name] = {"source": feature_source,
                                                "type": feature_type}
        if "min_categr_count" in feature_column:
            min_categr_count = feature_column["min_categr_count"]
        self.feature_map.feature_specs[name]["min_categr_count"] = min_categr_count
        if "embedding_dim" in feature_column:
            self.feature_map.feature_specs[name]["embedding_dim"] = feature_column["embedding_dim"]
        feature_values = ddf[name].values
        if feature_type == "numeric":
            normalizer_name = feature_column.get("normalizer", None)
            if normalizer_name is not None:
                normalizer = Normalizer(normalizer_name)
                normalizer.fit(feature_values)
                self.encoders[name + "_normalizer"] = normalizer
            self.feature_map.num_features += 1
        elif feature_type == "categorical":
            encoder = feature_column.get("encoder", "")
            if encoder != "":
                self.feature_map.feature_specs[name]["encoder"] = encoder
            if encoder == "":
                tokenizer = Tokenizer(min_freq=min_categr_count, 
                                      na_value=feature_column.get("na_value", ""))
                if "share_embedding" in feature_column:
                    self.feature_map.feature_specs[name]["share_embedding"] = feature_column["share_embedding"]
                    tokenizer.set_vocab(self.encoders["{}_tokenizer".format(feature_column["share_embedding"])].vocab)
                else:
                    if self.is_share_embedding_with_sequence(name):
                        tokenizer.fit_on_texts(feature_values, use_padding=True)
                        self.feature_map.feature_specs[name]["padding_idx"] = tokenizer.vocab_size - 1
                    else:
                        tokenizer.fit_on_texts(feature_values, use_padding=False)
                self.encoders[name + "_tokenizer"] = tokenizer
                self.feature_map.num_features += tokenizer.vocab_size
                self.feature_map.feature_specs[name]["vocab_size"] = tokenizer.vocab_size
                if "pretrained_emb" in feature_column:
                    self.feature_map.feature_specs[name]["pretrained_emb"] = "pretrained_embedding.h5"
                    self.feature_map.feature_specs[name]["freeze_emb"] = feature_column.get("freeze_emb", True)
                    tokenizer.load_pretrained_embedding(name,
                                                        feature_column["pretrained_emb"], 
                                                        feature_column["embedding_dim"],
                                                        os.path.join(self.processed_dir, "pretrained_embedding.h5"))
            elif encoder == "numeric_bucket":
                num_buckets = feature_column.get("num_buckets", num_buckets)
                qtf = sklearn_preprocess.QuantileTransformer(n_quantiles=num_buckets + 1)
                qtf.fit(feature_values)
                boundaries = qtf.quantiles_[1:-1]
                self.feature_map.feature_specs[name]["vocab_size"] = num_buckets
                self.feature_map.num_features += num_buckets
                self.encoders[name + "_boundaries"] = boundaries
            elif encoder == "hash_bucket":
                num_buckets = feature_column.get("num_buckets", num_buckets)
                uniques = Counter(feature_values)
                num_buckets = min(num_buckets, len(uniques))
                self.feature_map.feature_specs[name]["vocab_size"] = num_buckets
                self.feature_map.num_features += num_buckets
                self.encoders[name + "_num_buckets"] = num_buckets
        elif feature_type == "sequence":
            encoder = feature_column.get("encoder", "MaskedAveragePooling")
            splitter = feature_column.get("splitter", " ")
            na_value = feature_column.get("na_value", "")
            max_len = feature_column.get("max_len", 0)
            padding = feature_column.get("padding", "post")
            tokenizer = Tokenizer(min_freq=min_categr_count, splitter=splitter, 
                                  na_value=na_value, max_len=max_len, padding=padding)
            if "share_embedding" in feature_column:
                self.feature_map.feature_specs[name]["share_embedding"] = feature_column["share_embedding"]
                tokenizer.set_vocab(self.encoders["{}_tokenizer".format(feature_column["share_embedding"])].vocab)
            else:
                tokenizer.fit_on_texts(feature_values, use_padding=True)
            self.encoders[name + "_tokenizer"] = tokenizer
            self.feature_map.num_features += tokenizer.vocab_size
            self.feature_map.feature_specs[name].update({"encoder": encoder,
                                                         "padding_idx": tokenizer.vocab_size - 1,
                                                         "vocab_size": tokenizer.vocab_size,
                                                         "max_len": tokenizer.max_len})
            if "pretrained_emb" in feature_column:
                self.feature_map.feature_specs[name]["pretrained_emb"] = "pretrained_embedding.h5"
                self.feature_map.feature_specs[name]["freeze_emb"] = feature_column.get("freeze_emb", True)
                tokenizer.load_pretrained_embedding(name,
                                                    feature_column["pretrained_emb"], 
                                                    feature_column["embedding_dim"],
                                                    os.path.join(self.processed_dir, "pretrained_embedding.h5"))
        else:
            raise NotImplementedError("feature_col={}".format(feature_column))

    def transform(self, ddf):
        ddf = self._preprocess(ddf)
        data_arrays = []
        for feature, feature_spec in self.feature_map.feature_specs.items():
            feature_type = feature_spec["type"]
            if feature_type == "numeric":
                numeric_array = ddf.loc[:, feature].fillna(0).apply(lambda x: float(x)).values
                normalizer = self.encoders.get(feature + "_normalizer")
                if normalizer:
                     numeric_array = normalizer.normalize(numeric_array)
                data_arrays.append(numeric_array) 
            elif feature_type == "categorical":
                encoder = feature_spec.get("encoder", "")
                if encoder == "":
                    data_arrays.append(self.encoders.get(feature + "_tokenizer") \
                                                    .encode_category(ddf.loc[:, feature].values))
                elif encoder == "numeric_bucket":
                    raise NotImplementedError
                elif encoder == "hash_bucket":
                    raise NotImplementedError
            elif feature_type == "sequence":
                data_arrays.append(self.encoders.get(feature + "_tokenizer") \
                                                .encode_sequence(ddf.loc[:, feature].values))
        label_name = self.label_col["name"]
        if ddf[label_name].dtype != np.float64:
            ddf.loc[:, label_name] = ddf.loc[:, label_name].apply(lambda x: float(x))
        data_arrays.append(ddf.loc[:, label_name].values) # add the label column at last
        data_arrays = [item.reshape(-1, 1) if item.ndim == 1 else item for item in data_arrays]
        data_array = np.hstack(data_arrays)
        return data_array

    def is_share_embedding_with_sequence(self, feature):
        for col in self.feature_cols:
            if col.get("share_embedding", None) == feature and col["type"] == "sequence":
                return True
        return False

    def load_pickle(self, pickle_file=None):
        return pickle.load(open(pickle_file, "rb"))

    def save_pickle(self, pickle_file):
        if not os.path.exists(os.path.dirname(pickle_file)):
            os.makedirs(os.path.dirname(pickle_file))
        pickle.dump(self.encoders, open(pickle_file, "wb"))
        
    def load_json(self, json_file):
        self.feature_map.load(json_file)

    def load_data(self, data_path, use_hdf5=True, data_format='csv'):
        self.load_json(self.json_file)
        self.encoders = self.load_pickle(self.pickle_file)
        if data_format == 'h5':
            data_array = self.load_hdf5(data_path)
            return data_array
        elif data_format == 'csv':
            hdf5_file = os.path.join(self.processed_dir, 
                                     os.path.splitext(os.path.basename(data_path))[0] + '.h5')
            if use_hdf5 and os.path.exists(hdf5_file):
                try:
                    data_array = self.load_hdf5(hdf5_file)
                    return data_array
                except:
                    print('Loading h5 file failed, reloading from {}'.format(data_path))
            ddf = self.read_csv(data_path)
            data_array = self.transform(ddf)
            if use_hdf5:
                self.save_hdf5(data_array, hdf5_file)
        return data_array

    def save_hdf5(self, data_array, data_path, key="data"):
        if not os.path.exists(os.path.dirname(data_path)):
            os.makedirs(os.path.dirname(data_path))
        with h5py.File(data_path, 'w') as hf:
            hf.create_dataset(key, data=data_array)

    def load_hdf5(self, data_path, key="data"):
        with h5py.File(data_path, 'r') as hf:
            data_array = hf[key][:]
        return data_array

    def __getitem__(self, index):
        X = self.darray[index, 0:-1]
        y = self.darray[index, -1]
        return X, y
    
    def __len__(self):
        return self.darray.shape[0]
```

```python id="6YK5FLS6h_qs"
class TaobaoDataset(CTRDataset):

    feature_cols = [{'name': ["userid","adgroup_id","pid","cate_id","campaign_id","customer","brand","cms_segid",
                                "cms_group_id","final_gender_code","age_level","pvalue_level","shopping_level","occupation"],
                        'active': True, 'dtype': 'str', 'type': 'categorical'}]
                        
    label_col = {'name': 'clk', 'dtype': float}

    train_url = "https://github.com/RecoHut-Datasets/sample_ctr/raw/v1/train_sample.csv"
    valid_url = "https://github.com/RecoHut-Datasets/sample_ctr/raw/v1/valid_sample.csv"
    test_url = "https://github.com/RecoHut-Datasets/sample_ctr/raw/v1/test_sample.csv"

    @property
    def raw_file_names(self):
        return ['train_sample.csv',
                'valid_sample.csv',
                'test_sample.csv']

    def download(self):
        download_url(self.train_url, self.raw_dir)
        download_url(self.valid_url, self.raw_dir)
        download_url(self.test_url, self.raw_dir)

    def convert_hour(self, df, col_name):
        return df['time_stamp'].apply(lambda ts: ts[11:13])

    def convert_weekday(self, df, col_name):
        def _convert_weekday(timestamp):
            dt = date(int(timestamp[0:4]), int(timestamp[5:7]), int(timestamp[8:10]))
            return dt.strftime('%w')
        return df['time_stamp'].apply(_convert_weekday)

    def convert_weekend(self, df, col_name):
        def _convert_weekend(timestamp):
            dt = date(int(timestamp[0:4]), int(timestamp[5:7]), int(timestamp[8:10]))
            return '1' if dt.strftime('%w') in ['6', '0'] else '0'
        return df['time_stamp'].apply(_convert_weekend)
```

```python id="FPp-PB-ow_Tv"
from typing import Any, Iterable, List, Optional, Tuple, Union, Callable
from torch.utils.data import DataLoader, Dataset


class CTRDataModule(LightningDataModule):

    dataset_cls: str = ""

    def __init__(self,
                 data_dir: Optional[str] = None,
                 num_workers: int = 0,
                 normalize: bool = False,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 pin_memory: bool = True,
                 drop_last: bool = False,
                 *args, 
                 **kwargs) -> None:
        """
        Args:
            data_dir: Where to save/load the data
            num_workers: How many workers to use for loading data
            normalize: If true applies rating normalize
            batch_size: How many samples per batch to load
            shuffle: If true shuffles the train data every epoch
            pin_memory: If true, the data loader will copy Tensors into CUDA pinned memory before
                        returning them
            drop_last: If true drops the last incomplete batch
        """
        super().__init__(data_dir)

        self.data_dir = data_dir if data_dir is not None else os.getcwd()
        self.num_workers = num_workers
        self.normalize = normalize
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.kwargs = kwargs

    def prepare_data(self, *args: Any, **kwargs: Any) -> None:
        """Saves files to data_dir."""
        self.dataset = self.dataset_cls(self.data_dir, **self.kwargs)

    def setup(self, stage: Optional[str] = None) -> None:
        """Creates train, val, and test dataset."""
        if stage == "fit" or stage is None:
            self.dataset_train = self.dataset_cls(self.data_dir, data_type='train', **self.kwargs)
            self.dataset_val = self.dataset_cls(self.data_dir, data_type='valid', **self.kwargs)
        if stage == "test" or stage is None:
            self.dataset_test = self.dataset_cls(self.data_dir, data_type='test', **self.kwargs)

    def train_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        """The train dataloader."""
        return self._data_loader(self.dataset_train, shuffle=self.shuffle)

    def val_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """The val dataloader."""
        return self._data_loader(self.dataset_val)

    def test_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """The test dataloader."""
        return self._data_loader(self.dataset_test)

    def _data_loader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )
```

```python id="o98gB9N_4dL_"
class TaobaoDataModule(CTRDataModule):
    dataset_cls = TaobaoDataset
```

```python id="egWI7tWgVMrX"
params = {'model_id': 'DCN_demo',
              'data_dir': '/content/data',
              'model_root': './checkpoints/',
              'dnn_hidden_units': [64, 64],
              'dnn_activations': "relu",
              'crossing_layers': 3,
              'learning_rate': 1e-3,
              'net_dropout': 0,
              'batch_norm': False,
              'optimizer': 'adamw',
              'task': 'binary_classification',
              'loss': 'binary_crossentropy',
              'metrics': ['logloss', 'AUC'],
              'embedding_dim': 10,
              'batch_size': 64,
              'epochs': 3,
              'shuffle': True,
              'seed': 2019,
              'use_hdf5': True,
              'workers': 1,
              'verbose': 0}
```

```python colab={"base_uri": "https://localhost:8080/"} id="Wm5J-aNOV3uC" executionInfo={"status": "ok", "timestamp": 1641901070196, "user_tz": -330, "elapsed": 1378, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="08c9a17f-be5e-44e5-b7f0-00f307eba439"
ds = TaobaoDataModule(**params)
ds.prepare_data()
ds.setup()

for batch in ds.train_dataloader():
    print(batch)
    break
```

```python id="m6bTZbgzW2PU"
from torch import nn
import torch

class MaskedAveragePooling(nn.Module):
    def __init__(self):
        super(MaskedAveragePooling, self).__init__()

    def forward(self, embedding_matrix):
        sum_pooling_matrix = torch.sum(embedding_matrix, dim=1)
        non_padding_length = (embedding_matrix != 0).sum(dim=1)
        embedding_vec = sum_pooling_matrix / (non_padding_length.float() + 1e-16)
        return embedding_vec


class MaskedSumPooling(nn.Module):
    def __init__(self):
        super(MaskedSumPooling, self).__init__()

    def forward(self, embedding_matrix):
        # mask by zeros
        return torch.sum(embedding_matrix, dim=1)
```

```python id="-6YE_P-qWlaG"
import torch
from torch import nn
import h5py
import os
import numpy as np
from collections import OrderedDict
# from . import sequence


class EmbeddingLayer_v3(nn.Module):
    def __init__(self, 
                 feature_map, 
                 embedding_dim,
                 embedding_dropout=0,
                 required_feature_columns=[],
                 not_required_feature_columns=[]):
        super(EmbeddingLayer_v3, self).__init__()
        self.embedding_layer = EmbeddingDictLayer(feature_map, 
                                                  embedding_dim,
                                                  required_feature_columns,
                                                  not_required_feature_columns)
        self.dropout = nn.Dropout2d(embedding_dropout) if embedding_dropout > 0 else None

    def forward(self, X):
        feature_emb_dict = self.embedding_layer(X)
        feature_emb = torch.stack(self.embedding_layer.dict2list(feature_emb_dict), dim=1)
        if self.dropout is not None:
            feature_emb = self.dropout(feature_emb)
        return feature_emb


class EmbeddingDictLayer(nn.Module):
    def __init__(self, 
                 feature_map, 
                 embedding_dim, 
                 required_feature_columns=[],
                 not_required_feature_columns=[]):
        super(EmbeddingDictLayer, self).__init__()
        self._feature_map = feature_map
        self.required_feature_columns = required_feature_columns
        self.not_required_feature_columns = not_required_feature_columns
        self.embedding_layer = nn.ModuleDict()
        self.seq_encoder_layer = nn.ModuleDict()
        for feature, feature_spec in self._feature_map.feature_specs.items():
            if self.is_required(feature):
                # Set embedding_layer according to share_embedding
                if "share_embedding" in feature_spec:
                    self.embedding_layer[feature] = self.embedding_layer[feature_spec["share_embedding"]]
                feat_emb_dim = feature_spec.get("embedding_dim", embedding_dim)
                if feature_spec["type"] == "numeric":
                    if feature not in self.embedding_layer:
                        self.embedding_layer[feature] = nn.Linear(1, feat_emb_dim, bias=False)
                elif feature_spec["type"] == "categorical":
                    if feature not in self.embedding_layer:
                        padding_idx = feature_spec.get("padding_idx", None)
                        embedding_matrix = nn.Embedding(feature_spec["vocab_size"], 
                                                        feat_emb_dim, 
                                                        padding_idx=padding_idx)
                        if "pretrained_emb" in feature_spec:
                            embeddings = self.get_pretrained_embedding(feature_map.data_dir, feature, feature_spec)
                            embedding_matrix = self.set_pretrained_embedding(embedding_matrix, embeddings, 
                                                                             freeze=feature_spec["freeze_emb"],
                                                                             padding_idx=padding_idx)
                        self.embedding_layer[feature] = embedding_matrix
                elif feature_spec["type"] == "sequence":
                    if feature not in self.embedding_layer:
                        padding_idx = feature_spec["vocab_size"] - 1
                        embedding_matrix = nn.Embedding(feature_spec["vocab_size"], 
                                                        feat_emb_dim, 
                                                        padding_idx=padding_idx)
                        if "pretrained_emb" in feature_spec:
                            embeddings = self.get_pretrained_embedding(feature_map.data_dir, feature, feature_spec)
                            embedding_matrix = self.set_pretrained_embedding(embedding_matrix, embeddings, 
                                                                             freeze=feature_spec["freeze_emb"],
                                                                             padding_idx=padding_idx)
                        self.embedding_layer[feature] = embedding_matrix
                    self.set_sequence_encoder(feature, feature_spec.get("encoder", None))

    def is_required(self, feature):
        """ Check whether feature is required for embedding """
        feature_spec = self._feature_map.feature_specs[feature]
        if len(self.required_feature_columns) > 0 and (feature not in self.required_feature_columns):
            return False
        elif feature in self.not_required_feature_columns:
            return False
        else:
            return True

    def set_sequence_encoder(self, feature, encoder):
        if encoder is None or encoder in ["none", "null"]:
            self.seq_encoder_layer.update({feature: None})
        elif encoder == "MaskedAveragePooling":
            self.seq_encoder_layer.update({feature: sequence.MaskedAveragePooling()})
        elif encoder == "MaskedSumPooling":
            self.seq_encoder_layer.update({feature: sequence.MaskedSumPooling()})
        else:
            raise RuntimeError("Sequence encoder={} is not supported.".format(encoder))

    def get_pretrained_embedding(self, data_dir, feature_name, feature_spec):
        pretrained_path = os.path.join(data_dir, feature_spec["pretrained_emb"])
        with h5py.File(pretrained_path, 'r') as hf:
            embeddings = hf[feature_name][:]
        return embeddings

    def set_pretrained_embedding(self, embedding_matrix, embeddings, freeze=False, padding_idx=None):
        if padding_idx is not None:
            embeddings[padding_idx] = np.zeros(embeddings.shape[-1])
        embeddings = torch.from_numpy(embeddings).float()
        embedding_matrix.weight = torch.nn.Parameter(embeddings)
        if freeze:
            embedding_matrix.weight.requires_grad = False
        return embedding_matrix

    def dict2list(self, embedding_dict):
        return list(embedding_dict.values())

    def dict2tensor(self, embedding_dict, feature_source=None, feature_type=None):
        if feature_source is not None:
            if not isinstance(feature_source, list):
                feature_source = [feature_source]
            feature_emb_list = []
            for feature, feature_spec in self._feature_map.feature_specs.items():
                if feature_spec["source"] in feature_source:
                    feature_emb_list.append(embedding_dict[feature])
            return torch.stack(feature_emb_list, dim=1)
        elif feature_type is not None:
            if not isinstance(feature_type, list):
                feature_type = [feature_type]
            feature_emb_list = []
            for feature, feature_spec in self._feature_map.feature_specs.items():
                if feature_spec["type"] in feature_type:
                    feature_emb_list.append(embedding_dict[feature])
            return torch.stack(feature_emb_list, dim=1)
        else:
            return torch.stack(list(embedding_dict.values()), dim=1)

    def forward(self, X):
        feature_emb_dict = OrderedDict()
        for feature, feature_spec in self._feature_map.feature_specs.items():
            if feature in self.embedding_layer:
                if feature_spec["type"] == "numeric":
                    inp = X[:, feature_spec["index"]].float().view(-1, 1)
                    embedding_vec = self.embedding_layer[feature](inp)
                elif feature_spec["type"] == "categorical":
                    inp = X[:, feature_spec["index"]].long()
                    embedding_vec = self.embedding_layer[feature](inp)
                elif feature_spec["type"] == "sequence":
                    inp = X[:, feature_spec["index"]].long()
                    seq_embed_matrix = self.embedding_layer[feature](inp)
                    if self.seq_encoder_layer[feature] is not None:
                        embedding_vec = self.seq_encoder_layer[feature](seq_embed_matrix)
                    else:
                        embedding_vec = seq_embed_matrix
                feature_emb_dict[feature] = embedding_vec
        return feature_emb_dict
```

```python id="SfFOjn2XW__7"
import numpy as np
from torch import nn
import torch


def get_activation(activation):
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


class DNN_Layer(nn.Module):
    def __init__(self, 
                 input_dim, 
                 output_dim=None, 
                 hidden_units=[], 
                 hidden_activations="ReLU",
                 final_activation=None, 
                 dropout_rates=[], 
                 batch_norm=False, 
                 use_bias=True):
        super(DNN_Layer, self).__init__()
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
            dense_layers.append(get_activation(final_activation))
        self.dnn = nn.Sequential(*dense_layers) # * used to unpack list
    
    def forward(self, inputs):
        return self.dnn(inputs)
```

```python id="YFVnf3TC-X5c"
class CrossNet(nn.Module):
    def __init__(self, input_dim, num_layers):
        super(CrossNet, self).__init__()
        self.num_layers = num_layers
        self.cross_net = nn.ModuleList(CrossInteractionLayer(input_dim)
                                       for _ in range(self.num_layers))

    def forward(self, X_0):
        X_i = X_0 # b x dim
        for i in range(self.num_layers):
            X_i = X_i + self.cross_net[i](X_0, X_i)
        return X_i


class CrossInteractionLayer(nn.Module):
    def __init__(self, input_dim):
        super(CrossInteractionLayer, self).__init__()
        self.weight = nn.Linear(input_dim, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(input_dim))

    def forward(self, X_0, X_i):
        interaction_out = self.weight(X_i) * X_0 + self.bias
        return interaction_out
```

```python id="RwU1b2_kXoNs"
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
import numpy as np
import logging


def evaluate_metrics(y_true, y_pred, metrics):
    result = dict()
    for metric in metrics:
        if metric in ['logloss', 'binary_crossentropy']:
            result[metric] = log_loss(y_true, y_pred, eps=1e-7)
        elif metric == 'AUC':
            result[metric] = roc_auc_score(y_true, y_pred)
        elif metric == "ACC":
            y_pred = np.argmax(y_pred, axis=1)
            result[metric] = accuracy_score(y_true, y_pred)
    logging.info('[Metrics] ' + ' - '.join('{}: {:.6f}'.format(k, v) for k, v in result.items()))
    return result
```

```python id="TRBB39Hb-Zrw"
from typing import Any, Iterable, List, Optional, Tuple, Union, Callable

from torch.nn import functional as F

from pytorch_lightning import LightningModule


class CTRModel(LightningModule):
    def __init__(self, 
                 feature_map, 
                 model_id="BaseModel",
                 optimizer='adamw',
                 learning_rate = 0.003,
                 **kwargs):
        super().__init__()
        self._feature_map = feature_map
        self.model_id = model_id
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.model_dir = os.path.join(kwargs["model_root"], feature_map.dataset_id)
        self.checkpoint = os.path.abspath(os.path.join(self.model_dir, self.model_id + "_model.ckpt"))
        self._validation_metrics = kwargs["metrics"]
        self._verbose = kwargs["verbose"]

    def forward(self, users, items):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        features, y_true = batch
        y_pred = self(features)

        y_pred = y_pred.view(-1,1).squeeze()
        y_true = y_true.float()

        loss = self.loss_fn(y_pred, y_true)

        return {
            "loss": loss,
            "y_pred": y_pred.detach(),
        }

    def training_epoch_end(self, outputs):
        # This function recevies as parameters the output from "training_step()"
        # Outputs is a list which contains a dictionary like:
        # [{'pred':x,'target':x,'loss':x}, {'pred':x,'target':x,'loss':x}, ...]
        pass

    def validation_step(self, batch, batch_idx):
        features, y_true = batch
        y_pred = self(features)

        y_pred = np.array(y_pred.cpu().numpy().reshape(-1), np.float64)
        y_true = np.array(y_true.cpu().numpy().reshape(-1), np.float64)
        val_logs = evaluate_metrics(y_true, y_pred, self._validation_metrics)
        self.log("Val Metrics", val_logs, prog_bar=True)

        return {
            "y_pred": y_pred,
        }

    def validation_epoch_end(self, outputs):
        pass

    def test_step(self, batch, batch_idx):
        features, y_true = batch
        y_pred = self(features)

        y_pred = np.array(y_pred.cpu().numpy().reshape(-1), np.float64)
        y_true = np.array(y_true.cpu().numpy().reshape(-1), np.float64)
        test_logs = evaluate_metrics(y_true, y_pred, self._validation_metrics)
        self.log("Test Metrics", test_logs, prog_bar=True)

        return {
            "y_pred": y_pred,
        }

    def test_epoch_end(self, outputs):
        pass

    def configure_optimizers(self):
        if self.optimizer == 'adamw':
            return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        elif self.optimizer == 'adam':
            return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        elif self.optimizer == 'sgd':
            return torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        else:
            raise ValueError(f'Invalid optimizer type: {self.optimizer}')

    def loss_fn(self, y_pred, y_true):
        return F.binary_cross_entropy(y_pred, y_true, reduction='mean')        

    def init_weights(self, embedding_initializer=None):
        def _initialize(m):
            if type(m) == nn.ModuleDict:
                for k, v in m.items():
                    if type(v) == nn.Embedding:
                        if "pretrained_emb" in self._feature_map.feature_specs[k]: # skip pretrained
                            continue
                        if embedding_initializer is not None:
                            try:
                                initializer = embedding_initializer.replace("(", "(v.weight,")
                                eval(initializer)
                            except:
                                raise NotImplementedError("embedding_initializer={} is not supported."\
                                                          .format(embedding_initializer))
                        else:
                            nn.init.xavier_normal_(v.weight)
            if type(m) == nn.Linear:
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0)
        self.apply(_initialize)

    def get_final_activation(self, task="binary_classification"):
        if task == "binary_classification":
            return nn.Sigmoid()
        elif task == "multi_classification":
            return nn.Softmax(dim=-1)
        elif task == "regression":
            return None
        else:
            raise NotImplementedError("task={} is not supported.".format(task))
```

```python id="bsVvu-uLV8DQ"
class DCN(CTRModel):
    def __init__(self, 
                 feature_map, 
                 model_id="DCN",
                 task="binary_classification",
                 learning_rate=1e-3, 
                 embedding_initializer="torch.nn.init.normal_(std=1e-4)",
                 embedding_dim=10, 
                 dnn_hidden_units=[], 
                 dnn_activations="ReLU",
                 crossing_layers=3, 
                 embedding_dropout=0,
                 net_dropout=0, 
                 batch_norm=False,
                 **kwargs):
        super(DCN, self).__init__(feature_map, 
                                  model_id=model_id,
                                  **kwargs)
        self.embedding_layer = EmbeddingLayer_v3(feature_map, embedding_dim)
        input_dim = feature_map.num_fields * embedding_dim
        self.dnn = DNN_Layer(input_dim=input_dim,
                             output_dim=None, # output hidden layer
                             hidden_units=dnn_hidden_units,
                             hidden_activations=dnn_activations,
                             final_activation=None, 
                             dropout_rates=net_dropout, 
                             batch_norm=batch_norm, 
                             use_bias=True) \
                   if dnn_hidden_units else None # in case of only crossing net used
        self.crossnet = CrossNet(input_dim, crossing_layers)
        final_dim = input_dim
        if isinstance(dnn_hidden_units, list) and len(dnn_hidden_units) > 0: # if use dnn
            final_dim += dnn_hidden_units[-1]
        self.fc = nn.Linear(final_dim, 1) # [cross_part, dnn_part] -> logit
        self.final_activation = self.get_final_activation(task)
        self.init_weights(embedding_initializer=embedding_initializer)

    def forward(self, inputs):
        feature_emb = self.embedding_layer(inputs)
        flat_feature_emb = feature_emb.flatten(start_dim=1)
        cross_out = self.crossnet(flat_feature_emb)
        if self.dnn is not None:
            dnn_out = self.dnn(flat_feature_emb)
            final_out = torch.cat([cross_out, dnn_out], dim=-1)
        else:
            final_out = cross_out
        y_pred = self.fc(final_out)
        if self.final_activation is not None:
            y_pred = self.final_activation(y_pred)
        return y_pred
```

```python id="OKIJr8n2Xzpk"
model = DCN(ds.dataset.feature_map, **params)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 622, "referenced_widgets": ["90c8d6ca4eab4a18826d9f17e6c0636a", "0ec7f324b42548408e92d5e46941457a", "b035f3678cc1405fbd6f7564cff1ed11", "7bd2575559c3495eb007f0f0cb977fe0", "54e7a8030d7249f5b17f49c5f8a3fbbc", "e872ca6772ac425d93cac83b155c5fc7", "f8eab32990684d6d958ab7d0d33245be", "9053c6b2a56b4cd186e75fc9b3c5062d", "4f3374782cd14df7af446143297b001f", "d32f1e4393d9438a9945bd212b171116", "b1af7ef3e7fa4319bbdf0df5edd916aa", "399ba0f7457b40049d8799902fe84a91", "bbf347d0912447aa82a179a4afb84fe1", "ab51107fe2c2487296db1c604995a759", "0066aea305524b8ab1e49e75ea706bc3", "da32380ac57e472cb9e23be86d72ef5d", "b0b15b2a71dc49d9ad3f33866efb64c9", "ef5473426dbb4d508d6c98fef517d5ab", "f7d4c60a11f141bd99c2be2477f7545d", "b00a382d630a47bd99d43f9aa522061c", "21d0bf55dbc6452d8a8941a729f36901", "6f91cad0cbd446b98b75d7eab4410549", "ae0c3dbb7dca40428364891d2254562c", "7a916ae44a904a7898a052075fbcf171", "9803d4cbd47141eaa472448302796be4", "a120902b27c04d88a3e93e43eb418f92", "d8ba096c094142568e5685850227ca01", "f4dcc40365c7488e8a0e3e4d3aa06826", "5901fa87e7b24fefb813c0b4657a1786", "bebf2c2da8cf48eea3a72820045f0937", "b0a77af7b06d4842bde904047902b68c", "a350823ebf0940c9a217e25ac9e2e96d", "14da0a7882d44739a9632db85ed19385"]} id="dTacyen5HGrn" executionInfo={"status": "ok", "timestamp": 1641901617100, "user_tz": -330, "elapsed": 4467, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="14a811e9-9de2-40a2-d7e3-3f3082178de8"
from recohut.trainers.pl_trainer import pl_trainer

pl_trainer(model, ds, max_epochs=5)
```

```python id="d7Iqbq8K9lqc"

```
