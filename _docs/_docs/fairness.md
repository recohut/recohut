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

<!-- #region id="QAVqXpOE4QaM" -->
# Personalized Counterfactual Fairness in Recommendation
<!-- #endregion -->

<!-- #region id="iVJdn7A33_bk" -->
## Setup
<!-- #endregion -->

<!-- #region id="BXJY8c9d4Xi5" -->
### Installations
<!-- #endregion -->

```python id="TGJ5fBMK4Xgj"
!pip install -q wget
```

<!-- #region id="GB_yDppW3_Yt" -->
### Imports
<!-- #endregion -->

```python id="yebvVSIT3_WD"
import sys
import os
import wget
import logging
import os.path as osp
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import pickle
from sklearn.metrics import *
from sklearn.preprocessing import LabelBinarizer
import itertools as it
from time import time
import gc
from collections import defaultdict, namedtuple, OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
```

<!-- #region id="NyxCtlrJ3_Ta" -->
### Params
<!-- #endregion -->

```python id="MXBwnUCD3_RD"
class Args:
    path = '/content'
    train_suffix = '.train.tsv'             # train file suffix
    validation_suffix = '.validation.tsv'   # validation file suffix
    test_suffix = '.test.tsv'               # test file suffix
    all_suffix = '.all.tsv'                 # all data file
    feature_suffix = '.features.txt'         # feature file
    test_pkl_suffix = '.test.pkl'         # prepared test data pickle file suffix
    valid_pkl_suffix = '.validation.pkl'  # prepared validation data pickle file suffix
    USER = 'uid'                   # user column name
    ITEM = 'iid'                   # item column name
    LABEL = 'label'                 # label column name
    RANK_FILE_NAME = 'rank.csv'     # Trained model generated ranking list
    SAMPLE_ID = 'sample_id'         # sample id for each record
    gpu ='0' # Set CUDA_VISIBLE_DEVICES
    verbose = logging.INFO # Logging Level, 0, 10, ..., 50
    log_file = 'log.txt' # Logging file path
    result_file = 'result.npy' # Result file path
    random_seed = 2020 # Random seed of numpy and tensorflow
    train = 1 # To train the model or not
    dataset = 'ml1M'
    sep = '\t' # sep of csv file
    label = 'label' # name of dataset label column
    disc_batch_size = 7000 # discriminator train batch size
    train_num_neg = 1 # Negative sample num for each instance in train set
    vt_num_neg = -1 # Number of negative sample in validation/testing stage
    model_path ='model.pt' # Model save path
    u_vector_size = 64 # user vector size
    i_vector_size = 64 # item vector size
    filter_mode = 'combine' # combine for using one filter per sensitive feature, separate for using one filter per sensitive feature combination
    load = 0 # Whether load model and continue to train
    load_attack = False # Whether load attacker model and continue to train
    epoch = 100 # Number of epochs
    disc_epoch = 500 # Number of epochs for training extra discriminator
    check_epoch = 1 # Check every epochs
    early_stop = 1 # whether to early-stop
    lr = 0.001 # Learning rate
    lr_attack = 0.001 # attacker learning rate
    batch_size = 128 # Batch size during training
    vt_batch_size = 512 # Batch size during testing
    dropout = 0.2 # Dropout probability for each deep layer
    l2 = 1e-4 # Weight of l2_regularize in loss
    l2_attack = 1e-4 # Weight of attacker l2_regularize in loss
    no_filter = False # if or not use filters
    reg_weight = 1 # Trade off for adversarial penalty
    d_steps = 10 # the number of steps of updating discriminator
    optimizer = 'GD' # 'optimizer: GD, Adam, Adagrad
    metric = "RMSE" # metrics: RMSE, MAE, AUC, F1, Accuracy, Precision, Recall
    skip_eval = 0 # number of epochs without evaluation
    num_worker = 2 # number of processes for multi-processing data loading
    fix_one = False # fix one feature for evaluation
    eval_disc = False # train extra discriminator for evaluation
    data_reader = 'RecDataReader' # Choose data_reader
    data_processor = 'RecDataset' # Choose data_processor
    model_name = 'BiasedMF' # Choose model to run
    runner = 'RecRunner' # Choose runner

args = Args()
```

```python id="DSwbYFgC_6F0"
LOWER_METRIC_LIST = ["rmse", 'mae']
```

<!-- #region id="Q40X4lHf4JHw" -->
### Logger
<!-- #endregion -->

```python id="cibwpV5L4JFb"
logging.basicConfig(stream=sys.stdout,
                    level = logging.DEBUG,
                    format='%(asctime)s [%(levelname)s] : %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')

logger = logging.getLogger('T297944 Logger')
```

<!-- #region id="yTA5LxDK4ES0" -->
## Utilities
<!-- #endregion -->

<!-- #region id="fznS-E31A53b" -->
### Dataset
<!-- #endregion -->

```python id="fLUcWP4p4EQd"
def download_movielens():
    download_link = 'https://github.com/sparsh-ai/fairness-recsys/raw/main/data/bronze/ml1m/ml1m_t297944.zip'
    save_path = osp.join(args.path,args.dataset+'.zip')
    save_path_extracted = osp.join(args.path,args.dataset)
    Path(save_path_extracted).mkdir(parents=True, exist_ok=True)
    if not os.listdir(save_path_extracted):
        wget.download(download_link, out=save_path)
        with zipfile.ZipFile(save_path, 'r') as zip_ref:
            zip_ref.extractall(save_path_extracted)
        logger.info('Files saved in {}'.format(save_path_extracted))
    else:
        logger.info('Files already exists in {}, skipping!'.format(save_path_extracted))
```

```python id="MxKtfPC1A7nD"
class DataReader:
    def __init__(self, path, dataset_name, sep='\t', seq_sep=','):
        self.path = osp.join(path, dataset_name)
        self.dataset_name = dataset_name
        self.sep = sep
        self.seq_sep = seq_sep
        self.train_file = osp.join(self.path, dataset_name + args.train_suffix)
        self.validation_file = osp.join(self.path, dataset_name + args.validation_suffix)
        self.test_file = osp.join(self.path, dataset_name + args.test_suffix)
        self.all_file = osp.join(self.path, dataset_name + args.all_suffix)
        self.feature_file = osp.join(self.path, dataset_name + args.feature_suffix)
        self._load_data()
        self.features = self._load_feature() if osp.exists(self.feature_file) else None

    def _load_data(self):
        if osp.exists(self.all_file):
            logger.info("load all csv...")
            self.all_df = pd.read_csv(self.all_file, sep=self.sep)
        else:
            raise FileNotFoundError('all file is not found.')
        if osp.exists(self.train_file):
            logger.info("load train csv...")
            self.train_df = pd.read_csv(self.train_file, sep=self.sep)
            logger.info("size of train: %d" % len(self.train_df))
        else:
            raise FileNotFoundError('train file is not found.')
        if osp.exists(self.validation_file):
            logger.info("load validation csv...")
            self.validation_df = pd.read_csv(self.validation_file, sep=self.sep)
            logger.info("size of validation: %d" % len(self.validation_df))
        else:
            raise FileNotFoundError('validation file is not found.')
        if osp.exists(self.test_file):
            logger.info("load test csv...")
            self.test_df = pd.read_csv(self.test_file, sep=self.sep)
            logger.info("size of test: %d" % len(self.test_df))
        else:
            raise FileNotFoundError('test file is not found.')

    def _load_feature(self):
        """
        load pre-trained/feature embeddings. It is saved as a numpy text file.
        :return:
        """
        return np.loadtxt(self.feature_file, dtype=np.float32)
```

```python id="hCqf2POUCoeQ"
class RecDataReader(DataReader):
    def __init__(self, path, dataset_name, sep='\t', seq_sep=','):
        super().__init__(path, dataset_name, sep, seq_sep)
        self.user_ids_set = set(self.all_df[args.USER].tolist())
        self.item_ids_set = set(self.all_df[args.ITEM].tolist())
        self.num_nodes = len(self.user_ids_set) + len(self.item_ids_set)
        self.train_item2users_dict = self._prepare_item2users_dict(self.train_df)

        self.all_user2items_dict = self._prepare_user2items_dict(self.all_df)
        self.train_user2items_dict = self._prepare_user2items_dict(self.train_df)
        self.valid_user2items_dict = self._prepare_user2items_dict(self.validation_df)
        self.test_user2items_dict = self._prepare_user2items_dict(self.test_df)
        # add feature info for discriminator and filters
        uid_iid_label = [args.USER, args.ITEM, args.LABEL]
        self.feature_columns = [name for name in self.train_df.columns.tolist() if name not in uid_iid_label]
        Feature = namedtuple('Feature', ['num_class', 'label_min', 'label_max', 'name'])
        self.feature_info = \
            OrderedDict({idx + 1: Feature(self.all_df[col].nunique(), self.all_df[col].min(), self.all_df[col].max(),
                                          col) for idx, col in enumerate(self.feature_columns)})
        self.num_features = len(self.feature_columns)

    @staticmethod
    def _prepare_user2items_dict(df):
        df_groups = df.groupby(args.USER)
        user_sample_dict = defaultdict(set)
        for uid, group in df_groups:
            user_sample_dict[uid] = set(group[args.ITEM].tolist())
        return user_sample_dict

    @staticmethod
    def _prepare_item2users_dict(df):
        df_groups = df.groupby(args.ITEM)
        user_sample_dict = defaultdict(set)
        for uid, group in df_groups:
            user_sample_dict[uid] = set(group[args.USER].tolist())
        return user_sample_dict
```

```python id="Q_wJAytIDRoB"
class DiscriminatorDataReader:
    def __init__(self, path, dataset_name, sep='\t', seq_sep=',', test_ratio=0.1):
        self.path = osp.join(path, dataset_name)
        self.sep = sep
        self.seq_sep = seq_sep
        self.all_file = osp.join(self.path, dataset_name + args.all_suffix)
        self.train_attacker_file = osp.join(self.path, dataset_name + '.attacker' + args.train_suffix)
        self.test_attacker_file = osp.join(self.path, dataset_name + '.attacker' + args.test_suffix)
        self.all_df = pd.read_csv(self.all_file, sep='\t')

        # add feature info for discriminator and filters
        uid_iid_label = [args.USER, args.ITEM, args.LABEL]
        self.feature_columns = [name for name in self.all_df.columns.tolist() if name not in uid_iid_label]

        Feature = namedtuple('Feature', ['num_class', 'label_min', 'label_max', 'name'])
        self.feature_info = \
            OrderedDict({idx + 1: Feature(self.all_df[col].nunique(), self.all_df[col].min(), self.all_df[col].max(),
                                          col) for idx, col in enumerate(self.feature_columns)})
        self.f_name_2_idx = {f_name: idx + 1 for idx, f_name in enumerate(self.feature_columns)}
        self.num_features = len(self.feature_columns)
        if osp.exists(self.train_attacker_file) and osp.exists(self.test_attacker_file):
            self.train_df = pd.read_csv(self.train_attacker_file, sep='\t')
            self.test_df = pd.read_csv(self.test_attacker_file, sep='\t')
        else:
            self.train_df, self.test_df = self._init_feature_df(self.all_df, test_ratio)

    def _init_feature_df(self, all_df, test_ratio):
        logger.info('Initializing attacker train/test file...')
        feature_df = pd.DataFrame()
        all_df = all_df.sort_values(by='uid')
        all_group = all_df.groupby('uid')

        uid_list = []
        feature_list_dict = {key: [] for key in self.feature_columns}
        for uid, group in all_group:
            uid_list.append(uid)
            for key in feature_list_dict:
                feature_list_dict[key].append(group[key].tolist()[0])
        feature_df[args.USER] = uid_list
        for f in self.feature_columns:
            feature_df[f] = feature_list_dict[f]

        test_size = int(len(feature_df) * test_ratio)
        sign = True
        counter = 0
        while sign:
            test_set = feature_df.sample(n=test_size).sort_index()
            for f in self.feature_columns:
                num_class = self.feature_info[self.f_name_2_idx[f]].num_class
                val_range = set([i for i in range(num_class)])
                test_range = set(test_set[f].tolist())
                if len(val_range) != len(test_range):
                    sign = True
                    break
                else:
                    sign = False
            print(counter)
            counter += 1

        train_set = feature_df.drop(test_set.index)
        train_set.to_csv(self.train_attacker_file, sep='\t', index=False)
        test_set.to_csv(self.test_attacker_file, sep='\t', index=False)
        return train_set, test_set
```

```python id="XlYwe4eAD8yt"
class RecDataset:
    def __init__(self, data_reader, stage, batch_size=128, num_neg=1):
        self.data_reader = data_reader
        self.num_user = len(data_reader.user_ids_set)
        self.num_item = len(data_reader.item_ids_set)
        self.batch_size = batch_size
        self.stage = stage
        self.num_neg = num_neg
        # prepare test/validation dataset
        valid_pkl_path = osp.join(self.data_reader.path, self.data_reader.dataset_name + args.valid_pkl_suffix)
        test_pkl_path = osp.join(self.data_reader.path, self.data_reader.dataset_name + args.test_pkl_suffix)
        if self.stage == 'valid':
            if osp.exists(valid_pkl_path):
                with open(valid_pkl_path, 'rb') as file:
                    logger.info('Load validation data from pickle file.')
                    self.data = pickle.load(file)
            else:
                self.data = self._get_data()
                with open(valid_pkl_path, 'wb') as file:
                    pickle.dump(self.data, file)
        elif self.stage == 'test':
            if osp.exists(test_pkl_path):
                with open(test_pkl_path, 'rb') as file:
                    logger.info('Load test data from pickle file.')
                    self.data = pickle.load(file)
            else:
                self.data = self._get_data()
                with open(test_pkl_path, 'wb') as file:
                    pickle.dump(self.data, file)
        else:
            self.data = self._get_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def _get_data(self):
        if self.stage == 'train':
            return self._get_train_data()
        else:
            return self._get_vt_data()

    def _get_train_data(self):
        df = self.data_reader.train_df
        df[args.SAMPLE_ID] = df.index
        columns_order = [args.USER, args.ITEM, args.SAMPLE_ID, args.LABEL] + [f_col for f_col in self.data_reader.feature_columns]
        data = df[columns_order].to_numpy()
        return data

    def _get_vt_data(self):
        if self.stage == 'valid':
            df = self.data_reader.validation_df
            logger.info('Prepare validation data...')
        elif self.stage == 'test':
            df = self.data_reader.test_df
            logger.info('Prepare test data...')
        else:
            raise ValueError('Wrong stage in dataset.')
            
        df[args.SAMPLE_ID] = df.index
        columns_order = [args.USER, args.ITEM, args.SAMPLE_ID, args.LABEL] + [f_col for f_col in self.data_reader.feature_columns]
        data = df[columns_order].to_numpy()

        total_batches = int((len(df) + self.batch_size - 1) / self.batch_size)
        batches = []

        for n_batch in tqdm(range(total_batches), leave=False, ncols=100, mininterval=1, desc='Prepare Batches'):
            batch_start = n_batch * self.batch_size
            batch_end = min(len(df), batch_start + self.batch_size)

            real_batch_size = batch_end - batch_start

            batch = data[batch_start:batch_start + real_batch_size, :]

            inputs = np.asarray(batch)[:, 0:3]
            labels = np.asarray(batch)[:, 3]
            features = np.asarray(batch)[:, 4:]
            inputs = np.concatenate((inputs, features), axis=1)

            neg_samples = self._neg_samples_from_all(inputs, self.num_neg)
            neg_labels = np.asarray([0] * neg_samples.shape[0])

            tmp_sample = np.concatenate((inputs, neg_samples), axis=0)
            samples = torch.from_numpy(tmp_sample[:, 0:3])
            labels = torch.from_numpy(np.concatenate((labels, neg_labels), axis=0))
            features = torch.from_numpy(tmp_sample[:, 3:])

            feed_dict = {'X': samples, args.LABEL: labels, 'features': features}
            batches.append(feed_dict)

            gc.collect()

        return batches

    def collate_fn(self, batch):
        if self.stage == 'train':
            feed_dict = self._collate_train(batch)
        else:
            feed_dict = self._collate_vt(batch)
        return feed_dict

    def _collate_train(self, batch):
        inputs = np.asarray(batch)[:, 0:3]
        labels = np.asarray(batch)[:, 3]
        features = np.asarray(batch)[:, 4:]
        neg_samples = self._neg_sampler(inputs)
        neg_samples = np.insert(neg_samples, 0, inputs[:, 0], axis=1)
        neg_samples = np.insert(neg_samples, 2, inputs[:, 2], axis=1)
        neg_labels = np.asarray([0] * neg_samples.shape[0])
        neg_features = np.copy(features)
        assert len(inputs) == len(neg_samples)
        samples = torch.from_numpy(np.concatenate((inputs, neg_samples), axis=0))
        labels = torch.from_numpy(np.concatenate((labels, neg_labels), axis=0))
        features = torch.from_numpy((np.concatenate((features, neg_features), axis=0)))
        feed_dict = {'X': samples, args.LABEL: labels, 'features': features}
        return feed_dict

    @staticmethod
    def _collate_vt(data):
        return data

    def _neg_sampler(self, batch):
        neg_items = np.random.randint(1, self.num_item, size=(len(batch), self.num_neg))
        for i, (user, _, _) in enumerate(batch):
            user_clicked_set = self.data_reader.all_user2items_dict[user]
            for j in range(self.num_neg):
                while neg_items[i][j] in user_clicked_set:
                    neg_items[i][j] = np.random.randint(1, self.num_item)
        return neg_items

    def _neg_samples_from_all(self, batch, num_neg=-1):
        neg_items = None
        for idx, data in enumerate(batch):
            user = data[0]
            sample_id = data[2]
            features = data[3:]
            neg_candidates = list(self.data_reader.item_ids_set - self.data_reader.all_user2items_dict[user])
            if num_neg != -1:
                if num_neg <= len(neg_candidates):
                    neg_candidates = np.random.choice(neg_candidates, num_neg, replace=False)
                else:
                    neg_candidates = np.random.choice(neg_candidates, len(neg_candidates), replace=False)
            user_arr = np.asarray([user] * len(neg_candidates))
            id_arr = np.asarray([sample_id] * len(neg_candidates))
            feature_arr = np.tile(features, (len(neg_candidates), 1))
            neg_candidates = np.expand_dims(np.asarray(neg_candidates), axis=1)
            neg_candidates = np.insert(neg_candidates, 0, user_arr, axis=1)
            neg_candidates = np.insert(neg_candidates, 2, id_arr, axis=1)
            neg_candidates = np.concatenate((neg_candidates, feature_arr), axis=1)

            if neg_items is None:
                neg_items = neg_candidates
            else:
                neg_items = np.concatenate((neg_items, neg_candidates), axis=0)

        return neg_items
```

```python id="73FySA2dEOgn"
class DiscriminatorDataset:
    def __init__(self, data_reader, stage, batch_size=1000):
        self.data_reader = data_reader
        self.stage = stage
        self.batch_size = batch_size
        self.data = self._get_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def _get_data(self):
        if self.stage == 'train':
            return self._get_train_data()
        else:
            return self._get_test_data()

    def _get_train_data(self):
        data = self.data_reader.train_df.to_numpy()
        return data

    def _get_test_data(self):
        data = self.data_reader.test_df.to_numpy()
        return data

    @staticmethod
    def collate_fn(data):
        feed_dict = dict()
        feed_dict['X'] = torch.from_numpy(np.asarray(data)[:, 0])
        feed_dict['features'] = torch.from_numpy(np.asarray(data)[:, 1:])
        return feed_dict
```

<!-- #region id="3xeYrB4zLsZA" -->
### Models
<!-- #endregion -->

```python id="EvGqNW4cL6Z7"
class BaseRecModel(nn.Module):
    @staticmethod
    def init_weights(m):
        """
        initialize nn weights，called in main.py
        :param m: parameter or the nn
        :return:
        """
        if type(m) == torch.nn.Linear:
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)
            if m.bias is not None:
                torch.nn.init.normal_(m.bias, mean=0.0, std=0.01)
        elif type(m) == torch.nn.Embedding:
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)

    def __init__(self, data_processor_dict, user_num, item_num, u_vector_size, i_vector_size,
                 random_seed=2020, dropout=0.2, model_path='../model/Model/Model.pt', filter_mode='combine'):
        """
        :param data_processor_dict:
        :param user_num:
        :param item_num:
        :param u_vector_size:
        :param i_vector_size:
        :param random_seed:
        :param dropout:
        :param model_path:
        :param filter_mode: 'combine'-> for each combination train one filter;
        'separate' -> one filter for one sensitive feature, do combination for complex case.
        """
        super(BaseRecModel, self).__init__()
        self.data_processor_dict = data_processor_dict
        self.user_num = user_num
        self.item_num = item_num
        self.u_vector_size = u_vector_size
        self.i_vector_size = i_vector_size
        self.dropout = dropout
        self.random_seed = random_seed
        self.filter_mode = filter_mode
        torch.manual_seed(self.random_seed)
        torch.cuda.manual_seed(self.random_seed)
        self.model_path = model_path

        self._init_nn()
        self._init_sensitive_filter()
        logger.debug(list(self.parameters()))

        self.total_parameters = self.count_variables()
        logger.info('# of params: %d' % self.total_parameters)

        # optimizer assigned by *_runner.py
        self.optimizer = None

    def _init_nn(self):
        """
        Initialize neural networks
        :return:
        """
        raise NotImplementedError

    def _init_sensitive_filter(self):
        def get_sensitive_filter(embed_dim):
            sequential = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 2),
                nn.LeakyReLU(),
                nn.Linear(embed_dim * 2, embed_dim),
                nn.LeakyReLU(),
                nn.BatchNorm1d(embed_dim)
            )
            return sequential
        num_features = len(self.data_processor_dict['train'].data_reader.feature_columns)
        self.filter_num = num_features if self.filter_mode == 'combine' else 2**num_features
        self.num_features = num_features
        self.filter_dict = nn.ModuleDict(
            {str(i + 1): get_sensitive_filter(self.u_vector_size) for i in range(self.filter_num)})

    def apply_filter(self, vectors, filter_mask):
        if self.filter_mode == 'separate' and np.sum(filter_mask) != 0:
            filter_mask = np.asarray(filter_mask)
            idx = filter_mask.dot(2**np.arange(filter_mask.size))
            sens_filter = self.filter_dict[str(idx)]
            result = sens_filter(vectors)
        elif self.filter_mode == 'combine' and np.sum(filter_mask) != 0:
            result = None
            for idx, val in enumerate(filter_mask):
                if val != 0:
                    sens_filter = self.filter_dict[str(idx + 1)]
                    result = sens_filter(vectors) if result is None else result + sens_filter(vectors)
            result = result / np.sum(filter_mask)   # average the embedding
        else:
            result = vectors
        return result

    def count_variables(self):
        """
        Total number of parameters in the model
        :return:
        """
        total_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_parameters

    def l2(self):
        """
        calc the summation of l2 of all parameters
        :return:
        """
        l2 = 0
        for p in self.parameters():
            l2 += (p ** 2).sum()
        return l2

    def predict(self, feed_dict, filter_mask):
        """
        prediction only without loss calculation
        :param feed_dict: input dictionary
        :param filter_mask: mask for filter selection
        :return: output dictionary，with keys (at least)
                "prediction": predicted values;
                "check": intermediate results to be checked and printed out
        """
        check_list = []
        x = self.x_bn(feed_dict['X'].float())
        x = torch.nn.Dropout(p=feed_dict['dropout'])(x)
        prediction = F.relu(self.prediction(x)).view([-1])
        out_dict = {'prediction': prediction,
                    'check': check_list}
        return out_dict

    def forward(self, feed_dict, filter_mask):
        out_dict = self.predict(feed_dict, filter_mask)
        batch_size = int(feed_dict[args.LABEL].shape[0] / 2)
        pos, neg = out_dict['prediction'][:batch_size], out_dict['prediction'][batch_size:]
        loss = -(pos - neg).sigmoid().log().sum()
        out_dict['loss'] = loss
        return out_dict

    def save_model(self, model_path=None):
        if model_path is None:
            model_path = self.model_path
        dir_path = osp.dirname(model_path)
        if not osp.exists(dir_path):
            os.mkdir(dir_path)
        torch.save(self.state_dict(), model_path)
        logger.info('Save model to ' + model_path)

    def load_model(self, model_path=None):
        if model_path is None:
            model_path = self.model_path
        self.load_state_dict(torch.load(model_path))
        self.eval()
        logger.info('Load model from ' + model_path)

    def freeze_model(self):
        self.eval()
        for params in self.parameters():
            params.requires_grad = False
```

```python id="tryT1aMRM_bh"
class PMF(BaseRecModel):
    def _init_nn(self):
        self.uid_embeddings = torch.nn.Embedding(self.user_num, self.u_vector_size)
        self.iid_embeddings = torch.nn.Embedding(self.item_num, self.u_vector_size)

    def predict(self, feed_dict, filter_mask):
        check_list = []
        u_ids = feed_dict['X'][:, 0] - 1
        i_ids = feed_dict['X'][:, 1] - 1

        pmf_u_vectors = self.uid_embeddings(u_ids)
        pmf_i_vectors = self.iid_embeddings(i_ids)

        pmf_u_vectors = self.apply_filter(pmf_u_vectors, filter_mask)

        prediction = (pmf_u_vectors * pmf_i_vectors).sum(dim=1).view([-1])

        out_dict = {'prediction': prediction,
                    'check': check_list,
                    'u_vectors': pmf_u_vectors}
        return out_dict
```

```python id="JfnXVZLONt6h"
class RecRunner:
    def __init__(self, optimizer='GD', learning_rate=0.01, epoch=100, batch_size=128, eval_batch_size=128 * 128,
                 dropout=0.2, l2=1e-5, metrics='RMSE', check_epoch=10, early_stop=1, num_worker=1, no_filter=False,
                 reg_weight=0.1, d_steps=100, disc_epoch=1000):
        """
        初始化
        :param optimizer: optimizer name
        :param learning_rate: learning rate
        :param epoch: total training epochs
        :param batch_size: batch size for training
        :param eval_batch_size: batch size for evaluation
        :param dropout: dropout rate
        :param l2: l2 weight
        :param metrics: evaluation metrics list
        :param check_epoch: check intermediate results in every n epochs
        :param early_stop: 1 for early stop, 0 for not.
        :param no_filter: if or not use filters
        :param reg_weight: adversarial penalty weight
        :param d_steps: the number of steps to optimize discriminator
        :param disc_epoch: number of epoch for training extra discriminator
        """
        self.optimizer_name = optimizer
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.dropout = dropout
        self.no_dropout = 0.0
        self.l2_weight = l2
        self.reg_weight = reg_weight
        self.d_steps = d_steps
        self.no_filter = no_filter
        self.disc_epoch = disc_epoch

        # convert metrics to list of str
        self.metrics = metrics.lower().split(',')
        self.check_epoch = check_epoch
        self.early_stop = early_stop
        self.time = None

        # record train, validation, test results
        self.train_results, self.valid_results, self.test_results = [], [], []
        self.disc_results = []
        self.num_worker = num_worker

    def _build_optimizer(self, model, lr=None, l2_weight=None):
        optimizer_name = self.optimizer_name.lower()
        if lr is None:
            lr = self.learning_rate
        if l2_weight is None:
            l2_weight = self.l2_weight

        if optimizer_name == 'gd':
            logger.info("Optimizer: GD")
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=l2_weight)
        elif optimizer_name == 'adagrad':
            logger.info("Optimizer: Adagrad")
            optimizer = torch.optim.Adagrad(model.parameters(), lr=lr, weight_decay=l2_weight)
        elif optimizer_name == 'adam':
            logger.info("Optimizer: Adam")
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_weight)
        else:
            logging.error("Unknown Optimizer: " + self.optimizer_name)
            assert self.optimizer_name in ['GD', 'Adagrad', 'Adam']
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=l2_weight)
        return optimizer

    def _check_time(self, start=False):
        if self.time is None or start:
            self.time = [time()] * 2
            return self.time[0]
        tmp_time = self.time[1]
        self.time[1] = time()
        return self.time[1] - tmp_time

    @staticmethod
    def get_filter_mask(filter_num):
        return np.random.choice([0, 1], size=(filter_num,))

    @staticmethod
    def _get_masked_disc(disc_dict, labels, mask):
        if np.sum(mask) == 0:
            return []
        masked_disc_label = [(disc_dict[i + 1], labels[:, i]) for i, val in enumerate(mask) if val != 0]
        return masked_disc_label

    def fit(self, model, batches, fair_disc_dict, epoch=-1):  # fit the results for an input set
        """
        Train the model
        :param model: model instance
        :param batches: train data in batches
        :param fair_disc_dict: fairness discriminator dictionary
        :param epoch: epoch number
        :return: return the output of the last round
        """
        gc.collect()
        torch.cuda.empty_cache()

        if model.optimizer is None:
            model.optimizer = self._build_optimizer(model)
        model.train()

        for idx in fair_disc_dict:
            discriminator = fair_disc_dict[idx]
            if discriminator.optimizer is None:
                discriminator.optimizer = self._build_optimizer(discriminator)
            discriminator.train()

        loss_list = list()
        output_dict = dict()
        eval_dict = None
        for batch in tqdm(batches, leave=False, desc='Epoch %5d' % (epoch + 1),
                          ncols=100, mininterval=1):
            # step1: use filter mask select filters
            # step2: use selected filter filter out the embeddings
            # step3: use the filtered embeddings for recommendation task and get rec loss rec_loss
            # step4: apply the discriminator with the filtered embeddings and get discriminator loss d_loss
            #  (use filter_mask to decide use which discriminator)
            # step5: combine rec_loss and d_loss and do optimization (use filter and rec model optimizer)
            # step6: use discriminator optimizer to optimize discriminator K times
            if self.no_filter:
                mask = [0] * model.num_features
                mask = np.asarray(mask)
            else:
                mask = self.get_filter_mask(model.num_features)

            batch = batch_to_gpu(batch)
            model.optimizer.zero_grad()

            labels = batch['features'][:len(batch['features'])//2, :]
            if not self.no_filter:
                masked_disc_label = \
                    self._get_masked_disc(fair_disc_dict, labels, mask)
            else:
                masked_disc_label = \
                    self._get_masked_disc(fair_disc_dict, labels, mask + 1)

            # calculate recommendation loss + fair discriminator penalty
            result_dict = model(batch, mask)
            rec_loss = result_dict['loss']
            vectors = result_dict['u_vectors']
            vectors = vectors[:len(vectors) // 2, :]

            fair_d_penalty = 0
            if not self.no_filter:
                for fair_disc, label in masked_disc_label:
                    fair_d_penalty += fair_disc(vectors, label)
                fair_d_penalty *= -1
                loss = rec_loss + self.reg_weight * fair_d_penalty
            else:
                loss = rec_loss
            loss.backward()
            model.optimizer.step()

            loss_list.append(result_dict['loss'].detach().cpu().data.numpy())
            output_dict['check'] = result_dict['check']

            # update discriminator
            if not self.no_filter:
                if len(masked_disc_label) != 0:
                    for _ in range(self.d_steps):
                        for discriminator, label in masked_disc_label:
                            discriminator.optimizer.zero_grad()
                            disc_loss = discriminator(vectors.detach(), label)
                            disc_loss.backward(retain_graph=False)
                            discriminator.optimizer.step()

            # collect discriminator evaluation results
            if eval_dict is None:
                eval_dict = self._eval_discriminator(model, labels, vectors.detach(), fair_disc_dict, len(mask))
            else:
                batch_eval_dict = self._eval_discriminator(model, labels, vectors.detach(), fair_disc_dict, len(mask))
                for f_name in eval_dict:
                    new_label = batch_eval_dict[f_name]['label']
                    current_label = eval_dict[f_name]['label']
                    eval_dict[f_name]['label'] = torch.cat((current_label, new_label), dim=0)

                    new_prediction = batch_eval_dict[f_name]['prediction']
                    current_prediction = eval_dict[f_name]['prediction']
                    eval_dict[f_name]['prediction'] = torch.cat((current_prediction, new_prediction), dim=0)

        # generate discriminator evaluation scores
        d_score_dict = {}
        if eval_dict is not None:
            for f_name in eval_dict:
                l = eval_dict[f_name]['label']
                pred = eval_dict[f_name]['prediction']
                n_class = eval_dict[f_name]['num_class']
                d_score_dict[f_name] = self._disc_eval_method(l, pred, n_class)

        output_dict['d_score'] = d_score_dict
        output_dict['loss'] = np.mean(loss_list)
        return output_dict

    def train(self, model, dp_dict, fair_disc_dict, skip_eval=0, fix_one=False):
        """
        Train model
        :param model: model obj
        :param dp_dict: Data processors for train valid and test
        :param skip_eval: number of epochs to skip for evaluations
        :param fair_disc_dict: fairness discriminator dictionary
        :return:
        """
        train_data = DataLoader(dp_dict['train'], batch_size=self.batch_size, num_workers=self.num_worker,
                                shuffle=True, collate_fn=dp_dict['train'].collate_fn)
        validation_data = DataLoader(dp_dict['valid'], batch_size=None, num_workers=self.num_worker,
                                     pin_memory=True, collate_fn=dp_dict['test'].collate_fn)
        test_data = DataLoader(dp_dict['test'], batch_size=None, num_workers=self.num_worker,
                               pin_memory=True, collate_fn=dp_dict['test'].collate_fn)

        self._check_time(start=True)  # start time
        try:
            for epoch in range(self.epoch):
                self._check_time()
                output_dict = \
                    self.fit(model, train_data, fair_disc_dict, epoch=epoch)
                if self.check_epoch > 0 and (epoch == 1 or epoch % self.check_epoch == 0):
                    self.check(model, output_dict)
                training_time = self._check_time()

                if epoch >= skip_eval:
                    valid_result_dict, test_result_dict = None, None
                    if self.no_filter:
                        valid_result = self.evaluate(model, validation_data) if \
                            validation_data is not None else [-1.0] * len(self.metrics)
                        test_result = self.evaluate(model, test_data) \
                            if test_data is not None else [-1.0] * len(self.metrics)
                    else:
                        valid_result, valid_result_dict = \
                            self.eval_multi_combination(model, validation_data, fix_one) \
                            if validation_data is not None else [-1.0] * len(self.metrics)
                        test_result, test_result_dict = self.eval_multi_combination(model, test_data, fix_one) \
                            if test_data is not None else [-1.0] * len(self.metrics)

                    testing_time = self._check_time()

                    # self.train_results.append(train_result)
                    self.valid_results.append(valid_result)
                    self.test_results.append(test_result)
                    self.disc_results.append(output_dict['d_score'])

                    if self.no_filter:
                        logger.info("Epoch %5d [%.1f s]\n validation= %s test= %s [%.1f s] "
                                     % (epoch + 1, training_time,
                                        format_metric(valid_result), format_metric(test_result),
                                        testing_time) + ','.join(self.metrics))
                    else:
                        logger.info("Epoch %5d [%.1f s]\t Average: validation= %s test= %s [%.1f s] "
                                     % (epoch + 1, training_time,
                                        format_metric(valid_result), format_metric(test_result),
                                        testing_time) + ','.join(self.metrics))
                        for key in valid_result_dict:
                            logger.info("validation= %s test= %s "
                                         % (format_metric(valid_result_dict[key]),
                                            format_metric(test_result_dict[key])) + ','.join(self.metrics) +
                                         ' (' + key + ') ')

                    if best_result(self.metrics[0], self.valid_results) == self.valid_results[-1]:
                        model.save_model()
                        for idx in fair_disc_dict:
                            fair_disc_dict[idx].save_model()

                    if self.eva_termination() and self.early_stop == 1:
                        logger.info("Early stop at %d based on validation result." % (epoch + 1))
                        break
                if epoch < skip_eval:
                    logger.info("Epoch %5d [%.1f s]" % (epoch + 1, training_time))
        except KeyboardInterrupt:
            logger.info("Early stop manually")
            save_here = input("Save here? (1/0) (default 0):")
            if str(save_here).lower().startswith('1'):
                model.save_model()
                for idx in fair_disc_dict:
                    fair_disc_dict[idx].save_model()

        # Find the best validation result across iterations
        best_valid_score = best_result(self.metrics[0], self.valid_results)
        best_epoch = self.valid_results.index(best_valid_score)
        # prepare disc result string
        disc_info = self.disc_results[best_epoch]
        disc_info_str = ['{}={:.4f}'.format(key, disc_info[key]) for key in disc_info]
        disc_info_str = ','.join(disc_info_str)
        logger.info("Best Iter(validation)= %5d\t valid= %s test= %s [%.1f s] "
                     % (best_epoch + 1,
                        format_metric(self.valid_results[best_epoch]),
                        format_metric(self.test_results[best_epoch]),
                        self.time[1] - self.time[0]) + ','.join(self.metrics) + ' ' + disc_info_str +
                     ' AUC')
        best_test_score = best_result(self.metrics[0], self.test_results)
        best_epoch = self.test_results.index(best_test_score)
        disc_info = self.disc_results[best_epoch]
        disc_info_str = ['{}={:.4f}'.format(key, disc_info[key]) for key in disc_info]
        disc_info_str = ','.join(disc_info_str)
        logger.info("Best Iter(test)= %5d\t valid= %s test= %s [%.1f s] "
                     % (best_epoch + 1,
                        format_metric(self.valid_results[best_epoch]),
                        format_metric(self.test_results[best_epoch]),
                        self.time[1] - self.time[0]) + ','.join(self.metrics) + ' ' + disc_info_str +
                     ' AUC')
        model.load_model()
        for idx in fair_disc_dict:
            fair_disc_dict[idx].load_model()

    def eval_multi_combination(self, model, data, fix_one=False):
        """
        Evaluate model on validation/test dataset under different filter combinations.
        The output is the averaged result over all the possible combinations.
        :param model: trained model
        :param data: validation or test data (not train data)
        :param fix_one: if true, only evaluate on one feature instead of all the combinations (save running time)
        :return: averaged evaluated result on given dataset
        """
        n_features = model.num_features
        feature_info = model.data_processor_dict['train'].data_reader.feature_info

        if not fix_one:
            mask_list = [list(i) for i in it.product([0, 1], repeat=n_features)]
            mask_list.pop(0)
            # mask_list = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        else:
            feature_range = np.arange(n_features)
            shape = (feature_range.size, feature_range.max() + 1)
            one_hot = np.zeros(shape).astype(int)
            one_hot[feature_range, feature_range] = 1
            mask_list = one_hot.tolist()
            mask_list = [mask_list[1]]
        result_dict = {}
        acc_result = None
        for mask in mask_list:
            mask = np.asarray(mask)
            feature_idx = np.where(mask == 1)[0]
            f_name_list = [feature_info[i + 1].name for i in feature_idx]
            f_name = ' '.join(f_name_list)

            cur_result = self.evaluate(model, data, mask) if data is not None else [-1.0] * len(self.metrics)
            acc_result = np.array(cur_result) if acc_result is None else acc_result + np.asarray(cur_result)

            result_dict[f_name] = cur_result

        if acc_result is not None:
            acc_result /= len(mask_list)

        return list(acc_result), result_dict

    @torch.no_grad()
    def evaluate(self, model, batches, mask=None, metrics=None):
        """
        evaluate recommendation performance
        :param model:
        :param batches: data batches, each batch is a dict.
        :param mask: filter mask
        :param metrics: list of str
        :return: list of float number for each metric
        """
        if metrics is None:
            metrics = self.metrics
        model.eval()

        if mask is None:
            mask = [0] * model.filter_num
            mask = np.asarray(mask)

        result_dict = defaultdict(list)
        for batch in tqdm(batches, leave=False, ncols=100, mininterval=1, desc='Predict'):
            batch = batch_to_gpu(batch)
            out_dict = model.predict(batch, mask)
            prediction = out_dict['prediction']
            labels = batch[args.LABEL].cpu()
            sample_ids = batch['X'][:, 2].cpu()
            assert len(labels) == len(prediction)
            assert len(sample_ids == len(prediction))
            prediction = prediction.cpu().numpy()
            data_dict = {args.LABEL: labels, args.SAMPLE_ID: sample_ids}
            results = self.evaluate_method(prediction, data_dict, metrics=metrics)
            for key in results:
                result_dict[key].extend(results[key])

        evaluations = []
        for metric in metrics:
            evaluations.append(np.average(result_dict[metric]))

        return evaluations

    @staticmethod
    def evaluate_method(p, data, metrics):
        """
        Evaluate model predictions.
        :param p: predicted values, np.array
        :param data: data dictionary which include ground truth labels
        :param metrics: metrics list
        :return: a list of results. The order is consistent to metric list.
        """
        label = data[args.LABEL]
        evaluations = {}
        for metric in metrics:
            if metric == 'rmse':
                evaluations[metric] = [np.sqrt(mean_squared_error(label, p))]
            elif metric == 'mae':
                evaluations[metric] = [mean_absolute_error(label, p)]
            elif metric == 'auc':
                evaluations[metric] = [roc_auc_score(label, p)]
            else:
                k = int(metric.split('@')[-1])
                df = pd.DataFrame()
                df[args.SAMPLE_ID] = data[args.SAMPLE_ID]
                df['p'] = p
                df['l'] = label
                df = df.sort_values(by='p', ascending=False)
                df_group = df.groupby(args.SAMPLE_ID)
                if metric.startswith('ndcg@'):
                    ndcgs = []
                    for uid, group in df_group:
                        ndcgs.append(ndcg_at_k(group['l'].tolist()[:k], k=k, method=1))
                    evaluations[metric] = ndcgs
                elif metric.startswith('hit@'):
                    hits = []
                    for uid, group in df_group:
                        hits.append(int(np.sum(group['l'][:k]) > 0))
                    evaluations[metric] = hits
                elif metric.startswith('precision@'):
                    precisions = []
                    for uid, group in df_group:
                        precisions.append(precision_at_k(group['l'].tolist()[:k], k=k))
                    evaluations[metric] = precisions
                elif metric.startswith('recall@'):
                    recalls = []
                    for uid, group in df_group:
                        recalls.append(1.0 * np.sum(group['l'][:k]) / np.sum(group['l']))
                    evaluations[metric] = recalls
                elif metric.startswith('f1@'):
                    f1 = []
                    for uid, group in df_group:
                        num_overlap = 1.0 * np.sum(group['l'][:k])
                        f1.append(2 * num_overlap / (k + 1.0 * np.sum(group['l'])))
                    evaluations[metric] = f1
        return evaluations

    def eva_termination(self):
        """
        Early stopper
        :return:
        """
        metric = self.metrics[0]
        valid = self.valid_results
        if len(valid) > 20 and metric in LOWER_METRIC_LIST and strictly_increasing(valid[-5:]):
            return True
        elif len(valid) > 20 and metric not in LOWER_METRIC_LIST and strictly_decreasing(valid[-5:]):
            return True
        elif len(valid) - valid.index(best_result(metric, valid)) > 20:
            return True
        return False

    @torch.no_grad()
    def _eval_discriminator(self, model, labels, u_vectors, fair_disc_dict, num_disc):
        feature_info = model.data_processor_dict['train'].data_reader.feature_info
        feature_eval_dict = {}
        for i in range(num_disc):
            discriminator = fair_disc_dict[i + 1]
            label = labels[:, i]
            # metric = 'auc' if feature_info[i + 1].num_class == 2 else 'f1'
            feature_name = feature_info[i + 1].name
            discriminator.eval()
            if feature_info[i + 1].num_class == 2:
                prediction = discriminator.predict(u_vectors)['prediction'].squeeze()
            else:
                prediction = discriminator.predict(u_vectors)['output']
            feature_eval_dict[feature_name] = {'label': label.cpu(), 'prediction': prediction.detach().cpu(),
                                               'num_class': feature_info[i + 1].num_class}
            discriminator.train()
        return feature_eval_dict

    @staticmethod
    def _disc_eval_method(label, prediction, num_class, metric='auc'):
        if metric == 'auc':
            if num_class == 2:
                score = roc_auc_score(label, prediction, average='micro')
                # score = roc_auc_score(label, prediction)
                score = max(score, 1 - score)
                return score
            else:
                lb = LabelBinarizer()
                classes = [i for i in range(num_class)]
                lb.fit(classes)
                label = lb.transform(label)
                # label = lb.fit_transform(label)
                score = roc_auc_score(label, prediction, multi_class='ovo', average='macro')
                score = max(score, 1 - score)
                return score
        else:
            raise ValueError('Unknown evaluation metric in _disc_eval_method().')

    def check(self, model, out_dict):
        """
        Check intermediate results
        :param model: model obj
        :param out_dict: output dictionary
        :return:
        """
        check = out_dict
        logger.info(os.linesep)
        for i, t in enumerate(check['check']):
            d = np.array(t[1].detach().cpu())
            logger.info(os.linesep.join([t[0] + '\t' + str(d.shape), np.array2string(d, threshold=20)]) + os.linesep)

        loss, l2 = check['loss'], model.l2()
        l2 = l2 * self.l2_weight
        l2 = l2.detach()
        logger.info('loss = %.4f, l2 = %.4f' % (loss, l2))
        if not (np.absolute(loss) * 0.005 < l2 < np.absolute(loss) * 0.1):
            logging.warning('l2 inappropriate: loss = %.4f, l2 = %.4f' % (loss, l2))

        # for discriminator
        disc_score_dict = out_dict['d_score']
        for feature in disc_score_dict:
            logger.info('{} AUC = {:.4f}'.format(feature, disc_score_dict[feature]))

    def train_discriminator(self, model, dp_dict, fair_disc_dict, lr_attack=None, l2_attack=None):
        """
        Train discriminator to evaluate the quality of learned embeddings
        :param model: trained model
        :param dp_dict: Data processors for train valid and test
        :param fair_disc_dict: fairness discriminator dictionary
        :return:
        """
        train_data = DataLoader(dp_dict['train'], batch_size=dp_dict['train'].batch_size, num_workers=self.num_worker,
                                shuffle=True, collate_fn=dp_dict['train'].collate_fn)
        test_data = DataLoader(dp_dict['test'], batch_size=dp_dict['test'].batch_size, num_workers=self.num_worker,
                               pin_memory=True, collate_fn=dp_dict['test'].collate_fn)
        self._check_time(start=True)  # 记录初始时间s

        feature_results = defaultdict(list)
        best_results = dict()
        try:
            for epoch in range(self.disc_epoch):
                self._check_time()
                output_dict = \
                    self.fit_disc(model, train_data, fair_disc_dict, epoch=epoch,
                                  lr_attack=lr_attack, l2_attack=l2_attack)

                if self.check_epoch > 0 and (epoch == 1 or epoch % (self.disc_epoch // 4) == 0):
                    self.check_disc(output_dict)
                training_time = self._check_time()

                test_result_dict = \
                    self.evaluation_disc(model, fair_disc_dict, test_data, dp_dict['train'])
                d_score_dict = test_result_dict['d_score']
                # testing_time = self._check_time()
                if epoch % (self.disc_epoch // 4) == 0:
                    logger.info("Epoch %5d [%.1f s]" % (epoch + 1, training_time))
                for f_name in d_score_dict:
                    if epoch % (self.disc_epoch // 4) == 0:
                        logger.info("{} AUC= {:.4f}".format(f_name, d_score_dict[f_name]))
                    feature_results[f_name].append(d_score_dict[f_name])
                    if d_score_dict[f_name] == max(feature_results[f_name]):
                        best_results[f_name] = d_score_dict[f_name]
                        idx = dp_dict['train'].data_reader.f_name_2_idx[f_name]
                        fair_disc_dict[idx].save_model()

        except KeyboardInterrupt:
            logger.info("Early stop manually")
            save_here = input("Save here? (1/0) (default 0):")
            if str(save_here).lower().startswith('1'):
                for idx in fair_disc_dict:
                    fair_disc_dict[idx].save_model()

        for f_name in best_results:
            logger.info("{} best AUC: {:.4f}".format(f_name, best_results[f_name]))

        for idx in fair_disc_dict:
            fair_disc_dict[idx].load_model()

    def fit_disc(self, model, batches, fair_disc_dict, epoch=-1, lr_attack=None, l2_attack=None):
        """
        Train the discriminator
        :param model: model instance
        :param batches: train data in batches
        :param fair_disc_dict: fairness discriminator dictionary
        :param epoch: epoch number
        :param lr_attack: attacker learning rate
        :param l2_attack: l2 regularization weight for attacker
        :return: return the output of the last round
        """
        gc.collect()
        torch.cuda.empty_cache()

        for idx in fair_disc_dict:
            discriminator = fair_disc_dict[idx]
            if discriminator.optimizer is None:
                discriminator.optimizer = self._build_optimizer(discriminator, lr=lr_attack, l2_weight=l2_attack)
            discriminator.train()

        output_dict = dict()
        loss_acc = defaultdict(list)

        eval_dict = None
        for batch in tqdm(batches, leave=False, desc='Epoch %5d' % (epoch + 1),
                          ncols=100, mininterval=1):
            if self.no_filter:
                mask = [0] * model.num_features
                mask = np.asarray(mask)
            else:
                mask = self.get_filter_mask(model.num_features)

            batch = batch_to_gpu(batch)

            labels = batch['features']
            if not self.no_filter:
                masked_disc_label = \
                    self._get_masked_disc(fair_disc_dict, labels, mask)
            else:
                masked_disc_label = \
                    self._get_masked_disc(fair_disc_dict, labels, mask + 1)

            # calculate recommendation loss + fair discriminator penalty
            uids = batch['X'] - 1
            vectors = model.apply_filter(model.uid_embeddings(uids), mask)
            output_dict['check'] = []

            # update discriminator
            if len(masked_disc_label) != 0:
                for idx, (discriminator, label) in enumerate(masked_disc_label):
                    discriminator.optimizer.zero_grad()
                    disc_loss = discriminator(vectors.detach(), label)
                    disc_loss.backward()
                    discriminator.optimizer.step()
                    loss_acc[discriminator.name].append(disc_loss.detach().cpu())

        for key in loss_acc:
            loss_acc[key] = np.mean(loss_acc[key])

        output_dict['loss'] = loss_acc
        return output_dict

    @torch.no_grad()
    def evaluation_disc(self, model, fair_disc_dict, test_data, dp):
        num_features = dp.data_reader.num_features

        def eval_disc(labels, u_vectors, fair_disc_dict, mask):
            feature_info = dp.data_reader.feature_info
            feature_eval_dict = {}
            for i, val in enumerate(mask):
                if val == 0:
                    continue
                discriminator = fair_disc_dict[i + 1]
                label = labels[:, i]
                # metric = 'auc' if feature_info[i + 1].num_class == 2 else 'f1'
                feature_name = feature_info[i + 1].name
                discriminator.eval()
                if feature_info[i + 1].num_class == 2:
                    prediction = discriminator.predict(u_vectors)['prediction'].squeeze()
                else:
                    prediction = discriminator.predict(u_vectors)['output']
                feature_eval_dict[feature_name] = {'label': label.cpu(), 'prediction': prediction.detach().cpu(),
                                                   'num_class': feature_info[i + 1].num_class}
                discriminator.train()
            return feature_eval_dict

        eval_dict = {}
        for batch in test_data:
            # VERSION 1
            # if self.no_filter:
            #     # mask = [0] * model.num_features
            #     feature_range = np.arange(num_features)
            #     shape = (feature_range.size, feature_range.max() + 1)
            #     one_hot = np.zeros(shape).astype(int)
            #     one_hot[feature_range, feature_range] = 1
            #     mask_list = one_hot.tolist()
            #     # if num_features == 3:
            #     #     mask_list = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
            #     # elif num_features == 4:
            #     #     mask_list = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
            # else:
            #     mask_list = [list(i) for i in it.product([0, 1], repeat=num_features)]
            #     mask_list.pop(0)

            # VERSION 2
            mask_list = [list(i) for i in it.product([0, 1], repeat=num_features)]
            mask_list.pop(0)

            batch = batch_to_gpu(batch)

            labels = batch['features']
            uids = batch['X'] - 1

            for mask in mask_list:
                if self.no_filter:
                    vectors = model.uid_embeddings(uids)
                else:
                    vectors = model.apply_filter(model.uid_embeddings(uids), mask)
                batch_eval_dict = eval_disc(labels, vectors.detach(), fair_disc_dict, mask)

                for f_name in batch_eval_dict:
                    if f_name not in eval_dict:
                        eval_dict[f_name] = batch_eval_dict[f_name]
                    else:
                        new_label = batch_eval_dict[f_name]['label']
                        current_label = eval_dict[f_name]['label']
                        eval_dict[f_name]['label'] = torch.cat((current_label, new_label), dim=0)

                        new_prediction = batch_eval_dict[f_name]['prediction']
                        current_prediction = eval_dict[f_name]['prediction']
                        eval_dict[f_name]['prediction'] = torch.cat((current_prediction, new_prediction), dim=0)

        # generate discriminator evaluation scores
        d_score_dict = {}
        if eval_dict is not None:
            for f_name in eval_dict:
                l = eval_dict[f_name]['label']
                pred = eval_dict[f_name]['prediction']
                n_class = eval_dict[f_name]['num_class']
                d_score_dict[f_name] = self._disc_eval_method(l, pred, n_class)

        output_dict = dict()
        output_dict['d_score'] = d_score_dict
        return output_dict

    @staticmethod
    def check_disc(out_dict):
        check = out_dict
        logger.info(os.linesep)
        for i, t in enumerate(check['check']):
            d = np.array(t[1].detach().cpu())
            logger.info(os.linesep.join([t[0] + '\t' + str(d.shape), np.array2string(d, threshold=20)]) + os.linesep)

        loss_dict = check['loss']
        for disc_name, disc_loss in loss_dict.items():
            logger.info('%s loss = %.4f' % (disc_name, disc_loss))

        # for discriminator
        if 'd_score' in out_dict:
            disc_score_dict = out_dict['d_score']
            for feature in disc_score_dict:
                logger.info('{} AUC = {:.4f}'.format(feature, disc_score_dict[feature]))
```

<!-- #region id="-mBpD_r2A37k" -->
### Utils
<!-- #endregion -->

```python id="c5BXgA7TAa8s"
def balance_data(data):
    pos_indexes = np.where(data['Y'] == 1)[0]
    copy_num = int((len(data['Y']) - len(pos_indexes)) / len(pos_indexes))
    if copy_num > 1:
        copy_indexes = np.tile(pos_indexes, copy_num)
        sample_index = np.concatenate([np.arange(0, len(data['Y'])), copy_indexes])
        for k in data:
            data[k] = data[k][sample_index]
    return data


def input_data_is_list(data):
    if type(data) is list or type(data) is tuple:
        print("input_data_is_list")
        new_data = {}
        for k in data[0]:
            new_data[k] = np.concatenate([d[k] for d in data])
        return new_data
    return data


def format_metric(metric):
    # print(metric, type(metric))
    if type(metric) is not tuple and type(metric) is not list:
        metric = [metric]
    format_str = []
    if type(metric) is tuple or type(metric) is list:
        for m in metric:
            # print(type(m))
            if type(m) is float or type(m) is np.float or type(m) is np.float32 or type(m) is np.float64:
                format_str.append('%.4f' % m)
            elif type(m) is int or type(m) is np.int or type(m) is np.int32 or type(m) is np.int64:
                format_str.append('%d' % m)
    return ','.join(format_str)


def shuffle_in_unison_scary(data):
    """
    shuffle entire dataset
    :param data:
    :return:
    """
    rng_state = np.random.get_state()
    for d in data:
        np.random.set_state(rng_state)
        np.random.shuffle(data[d])
    return data


def best_result(metric, results_list):
    if type(metric) is list or type(metric) is tuple:
        metric = metric[0]
    if metric in LOWER_METRIC_LIST:
        return min(results_list)
    return max(results_list)


def strictly_increasing(l):
    return all(x < y for x, y in zip(l, l[1:]))


def strictly_decreasing(l):
    return all(x > y for x, y in zip(l, l[1:]))


def non_increasing(l):
    return all(x >= y for x, y in zip(l, l[1:]))


def non_decreasing(l):
    return all(x <= y for x, y in zip(l, l[1:]))


def monotonic(l):
    return non_increasing(l) or non_decreasing(l)


def numpy_to_torch(d):
    t = torch.from_numpy(d)
    if torch.cuda.device_count() > 0:
        t = t.cuda()
    return t


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def batch_to_gpu(batch):
    if torch.cuda.device_count() > 0:
        for c in batch:
            if type(batch[c]) is torch.Tensor:
                batch[c] = batch[c].cuda()
    return batch
```

<!-- #region id="jKJUMRk6Af0x" -->
### Metrics
<!-- #endregion -->

```python id="yiC5cZjOAhD3"
def mean_reciprocal_rank(rs):
    """Score is reciprocal of the rank of the first relevant item
    First element is 'rank 1'.  Relevance is binary (nonzero is relevant).
    Example from http://en.wikipedia.org/wiki/Mean_reciprocal_rank
    >>> rs = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    >>> mean_reciprocal_rank(rs)
    0.61111111111111105
    >>> rs = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]])
    >>> mean_reciprocal_rank(rs)
    0.5
    >>> rs = [[0, 0, 0, 1], [1, 0, 0], [1, 0, 0]]
    >>> mean_reciprocal_rank(rs)
    0.75
    Args:
        rs: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Mean reciprocal rank
    """
    rs = (np.asarray(r).nonzero()[0] for r in rs)
    return np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs])


def reciprocal_rank(rs):
    """Score is reciprocal of the rank of the first relevant item
    Args:
        rs: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        reciprocal rank
    """
    rs = np.asarray(rs).nonzero()[0]
    return 1. / (rs[0] + 1) if rs.size else 0.


def r_precision(r):
    """Score is precision after all relevant documents have been retrieved
    Relevance is binary (nonzero is relevant).
    >>> r = [0, 0, 1]
    >>> r_precision(r)
    0.33333333333333331
    >>> r = [0, 1, 0]
    >>> r_precision(r)
    0.5
    >>> r = [1, 0, 0]
    >>> r_precision(r)
    1.0
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        R Precision
    """
    r = np.asarray(r) != 0
    z = r.nonzero()[0]
    if not z.size:
        return 0.
    return np.mean(r[:z[-1] + 1])


def precision_at_k(r, k):
    """Score is precision @ k
    Relevance is binary (nonzero is relevant).
    >>> r = [0, 0, 1]
    >>> precision_at_k(r, 1)
    0.0
    >>> precision_at_k(r, 2)
    0.0
    >>> precision_at_k(r, 3)
    0.33333333333333331
    >>> precision_at_k(r, 4)
    Traceback (most recent call last):
        File "<stdin>", line 1, in ?
    ValueError: Relevance score length < k
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Precision @ k
    Raises:
        ValueError: len(r) must be >= k
    """
    assert k >= 1
    r = np.asarray(r)[:k] != 0
    if r.size != k:
        raise ValueError('Relevance score length < k')
    return np.mean(r)


def average_precision(r):
    """Score is average precision (area under PR curve)
    Relevance is binary (nonzero is relevant).
    >>> r = [1, 1, 0, 1, 0, 1, 0, 0, 0, 1]
    >>> delta_r = 1. / sum(r)
    >>> sum([sum(r[:x + 1]) / (x + 1.) * delta_r for x, y in enumerate(r) if y])
    0.7833333333333333
    >>> average_precision(r)
    0.78333333333333333
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Average precision
    """
    r = np.asarray(r) != 0
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.
    return np.mean(out)


def mean_average_precision(rs):
    """Score is mean average precision
    Relevance is binary (nonzero is relevant).
    >>> rs = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1]]
    >>> mean_average_precision(rs)
    0.78333333333333333
    >>> rs = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1], [0]]
    >>> mean_average_precision(rs)
    0.39166666666666666
    Args:
        rs: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Mean average precision
    """
    return np.mean([average_precision(r) for r in rs])


def dcg_at_k(r, k, method=0):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> dcg_at_k(r, 1)
    3.0
    >>> dcg_at_k(r, 1, method=1)
    3.0
    >>> dcg_at_k(r, 2)
    5.0
    >>> dcg_at_k(r, 2, method=1)
    4.2618595071429155
    >>> dcg_at_k(r, 10)
    9.6051177391888114
    >>> dcg_at_k(r, 11)
    9.6051177391888114
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k, method=0):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> ndcg_at_k(r, 1)
    1.0
    >>> r = [2, 1, 2, 0]
    >>> ndcg_at_k(r, 4)
    0.9203032077642922
    >>> ndcg_at_k(r, 4, method=1)
    0.96519546960144276
    >>> ndcg_at_k([0], 1)
    0.0
    >>> ndcg_at_k([1], 2)
    1.0
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Normalized discounted cumulative gain
    """
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max
```

<!-- #region id="YqvAt2nl4EOL" -->
## Jobs
<!-- #endregion -->

```python id="nb4As1933lK_"
# # No filters
# python ./main.py --model_name BiasedMF --optimizer Adam --dataset ml1M --data_processor RecDataset --metric ndcg@5,ndcg@10,hit@5,hit@10 --l2 1e-4 --batch_size 1024 --model_path "../model/biasedMF_ml1m_no_filter_neg_sample=100/biasedMF_ml1m_l2=1e-4_dim=64_no_filter_neg_sample=100.pt" --runner RecRunner --d_step 10 --vt_num_neg 100 --vt_batch_size 1024 --no_filter --eval_dict
# python ./main.py --model_name PMF --optimizer Adam --dataset ml1M --data_processor RecDataset --metric ndcg@5,ndcg@10,hit@5,hit@10 --l2 1e-4 --batch_size 1024 --model_path "../model/PMF_ml1m_no_filter_neg_sample=100/PMF_ml1m_l2=1e-4_dim=64_no_filter_neg_sample=100.pt" --runner RecRunner --d_step 10 --vt_num_neg 100 --vt_batch_size 1024 --no_filter --eval_disc
# python ./main.py --model_name DMF --optimizer Adam --dataset ml1M --data_processor RecDataset --metric ndcg@5,ndcg@10,hit@5,hit@10 --l2 1e-4 --batch_size 1024 --model_path "../model/DMF_ml1m_no_filter_neg_sample=100/DMF_ml1m_l2=1e-4_dim=64_no_filter_neg_sample=100.pt" --runner RecRunner --d_step 10 --vt_num_neg 100 --vt_batch_size 1024 --no_filter --eval_disc
# python ./main.py --model_name MLP --optimizer Adam --dataset ml1M --data_processor RecDataset --metric ndcg@5,ndcg@10,hit@5,hit@10 --l2 1e-4 --batch_size 1024 --model_path "../model/MLP_ml1m_no_filter_neg_sample=100/MLP_ml1m_l2=1e-4_dim=64_no_filter_neg_sample=100.pt" --runner RecRunner --d_step 10 --vt_num_neg 100 --vt_batch_size 1024 --no_filter --eval_disc

# # Sample command for separate method
# python ./main.py --model_name BiasedMF --optimizer Adam --dataset ml1M --data_processor RecDataset --metric ndcg@5,ndcg@10,hit@5,hit@10 --l2 1e-4 --batch_size 1024 --model_path "../model/biasedMF_ml1m_neg_sample=100_reg_weight=20_separate/biasedMF_ml1m_l2=1e-4_dim=64_reg_weight=20_neg_sample=100_separate.pt" --runner RecRunner --d_step 10 --reg_weight 20 --epoch 200 --vt_num_neg 100 --vt_batch_size 1024 --filter_mode separate --fix_one --eval_disc
# python ./main.py --model_name PMF --optimizer Adam --dataset ml1M --data_processor RecDataset --metric ndcg@5,ndcg@10,hit@5,hit@10 --l2 1e-4 --batch_size 1024 --model_path "../model/PMF_ml1m_neg_sample=100_reg_weight=20_separate/PMF_ml1m_l2=1e-4_dim=64_reg_weight=20_neg_sample=100_separate.pt" --runner RecRunner --d_step 10 --reg_weight 20 --epoch 200 --vt_num_neg 100 --vt_batch_size 1024 --filter_mode separate --fix_one --eval_disc
# python ./main.py --model_name DMF --optimizer Adam --dataset ml1M --data_processor RecDataset --metric ndcg@5,ndcg@10,hit@5,hit@10 --l2 1e-4 --batch_size 1024 --model_path "../model/DMF_ml1m_neg_sample=100_reg_weight=20_separate/DMF_ml1m_l2=1e-4_dim=64_reg_weight=20_neg_sample=100_separate.pt" --runner RecRunner --d_step 10 --reg_weight 20 --epoch 200 --vt_num_neg 100 --vt_batch_size 1024 --filter_mode separate --fix_one --eval_disc
# python ./main.py --model_name MLP --optimizer Adam --dataset ml1M --data_processor RecDataset --metric ndcg@5,ndcg@10,hit@5,hit@10 --l2 1e-4 --u_vector_size 32 --batch_size 1024 --model_path "../model/MLP_ml1m_neg_sample=100_reg_weight=20_separate/MLP_ml1m_l2=1e-4_dim=32_reg_weight=20_neg_sample=100_separate.pt" --runner RecRunner --d_step 10 --reg_weight 20 --vt_num_neg 100 --vt_batch_size 1024 --fix_one --filter_mode separate --eval_disc

# # Sample command for combination method
# python ./main.py --model_name BiasedMF --optimizer Adam --dataset ml1M --data_processor RecDataset --metric ndcg@5,ndcg@10,hit@5,hit@10 --l2 1e-4 --batch_size 1024 --model_path "../model/biasedMF_ml1m_reg_weight=20_neg_sample=100_combine/biasedMF_ml1m_l2=1e-4_dim=64_reg_weight=20_neg_sample=100_combine.pt" --runner RecRunner --d_step 10 --reg_weight 20 --vt_num_neg 100 --vt_batch_size 1024 --fix_one --eval_disc
# python ./main.py --model_name PMF --optimizer Adam --dataset ml1M --data_processor RecDataset --metric ndcg@5,ndcg@10,hit@5,hit@10 --l2 1e-4 --batch_size 1024 --model_path "../model/PMF_ml1m_reg_weight=20_neg_sample=100_combine/PMF_ml1m_l2=1e-4_dim=64_reg_weight=20_neg_sample=100_combine.pt" --runner RecRunner --d_step 10 --reg_weight 20 --vt_num_neg 100 --vt_batch_size 1024 --fix_one --eval_disc
# python ./main.py --model_name DMF --optimizer Adam --dataset ml1M --data_processor RecDataset --metric ndcg@5,ndcg@10,hit@5,hit@10 --l2 1e-4 --batch_size 1024 --model_path "../model/DMF_ml1m_reg_weight=20_neg_sample=100_combine/DMF_ml1m_l2=1e-4_dim=64_reg_weight=20_neg_sample=100_combine.pt" --runner RecRunner --d_step 10 --reg_weight 20 --vt_num_neg 100 --vt_batch_size 1024 --fix_one --eval_disc
# python ./main.py --model_name MLP --optimizer Adam --dataset ml1M --data_processor RecDataset --metric ndcg@5,ndcg@10,hit@5,hit@10 --l2 1e-4 --u_vector_size 32 --batch_size 1024 --model_path "../model/MLP_ml1m_l2=1e-4_dim=32_reg_weight=20_neg_sample=100/MLP_ml1m_l2=1e-4_dim=32_reg_weight=20_neg_sample=100.pt" --runner RecRunner --d_step 10 --vt_num_neg 100 --vt_batch_size 1024 --reg_weight 20 --fix_one --eval_disc
```

```python colab={"base_uri": "https://localhost:8080/"} id="MmQfVXVf4EME" executionInfo={"status": "ok", "timestamp": 1636457007116, "user_tz": -330, "elapsed": 33, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="300d7bd7-e1e9-4975-94cc-fb9a3d3f4b32"
download_movielens()
```

```python id="d8Pks02yWPt8"
args.model_name = 'PMF'
```

```python colab={"base_uri": "https://localhost:8080/"} id="aOPPaVnacCDa" executionInfo={"status": "ok", "timestamp": 1636457009002, "user_tz": -330, "elapsed": 9, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="7576be4c-da64-477e-9f0f-37fa01e0d53f"
# choose data_reader
data_reader_name = eval(args.data_reader)

# choose model
model_name = eval(args.model_name)
runner_name = eval(args.runner)

# choose data_processor
data_processor_name = eval(args.data_processor)

# logging
logger.info('DataReader: ' + args.data_reader)
logger.info('Model: ' + args.model_name)
logger.info('Runner: ' + args.runner)
logger.info('DataProcessor: ' + args.data_processor)

# random seed
torch.manual_seed(args.random_seed)
torch.cuda.manual_seed_all(args.random_seed)
np.random.seed(args.random_seed)

# cuda
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
logger.info("# cuda devices: %d" % torch.cuda.device_count())
```

```python colab={"base_uri": "https://localhost:8080/"} id="0WsqdYjkcEQP" executionInfo={"status": "ok", "timestamp": 1636457020721, "user_tz": -330, "elapsed": 5949, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="395e7487-a552-422a-cd83-624b80e450e2"
# create data_reader
data_reader = data_reader_name(path=args.path, dataset_name=args.dataset, sep=args.sep)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 214, "referenced_widgets": ["b5c1118b4bfb4f9fa0695e91163a1e47", "47532d7cbe0a4c5fb1f1b3575d836ea9", "32db06de410a4481a5ca0b75b42f7ec2", "f0c36db39c5746e79a5643c8dcb21abf", "2d8aeab54e2d4b0ab739c96445f91394", "a39ae408211c46bd95e4a333f2eabbff", "79f7f84ff4b640c7bc7c3cc40f161798", "62be46a350e846a5995ccad4fc4a5954", "b720c3aaf1644da789fdad2d70251972", "4a16f523a0ca4d83b0dd9ce12ff32e4e", "e02966e60c78464496a57d61417d7aa1"]} id="KZWHNAQWcThk" outputId="45d8f03e-f982-4cff-e242-bcb7ad499d21"
# create data processor
data_processor_dict = {}
for stage in ['train', 'valid', 'test']:
    if stage == 'train':
        if args.data_processor in ['RecDataset']:
            data_processor_dict[stage] = data_processor_name(
                data_reader, stage, batch_size=args.batch_size, num_neg=args.train_num_neg)
        else:
            raise ValueError('Unknown DataProcessor')
    else:
        if args.data_processor in ['RecDataset']:
            data_processor_dict[stage] = data_processor_name(
                data_reader, stage, batch_size=args.vt_batch_size, num_neg=args.vt_num_neg)
        else:
            raise ValueError('Unknown DataProcessor')
    gc.collect()
```

```python id="4DT-IGUEiEMB"
# create model
if args.model_name in ['BiasedMF', 'PMF']:
    model = model_name(data_processor_dict, user_num=len(data_reader.user_ids_set),
                        item_num=len(data_reader.item_ids_set), u_vector_size=args.u_vector_size,
                        i_vector_size=args.i_vector_size, random_seed=args.random_seed, dropout=args.dropout,
                        model_path=args.model_path, filter_mode=args.filter_mode)
elif args.model_name in ['DMF', 'MLP']:
    model = model_name(data_processor_dict, user_num=len(data_reader.user_ids_set),
                        item_num=len(data_reader.item_ids_set), u_vector_size=args.u_vector_size,
                        i_vector_size=args.i_vector_size, num_layers=args.num_layers,
                        random_seed=args.random_seed, dropout=args.dropout,
                        model_path=args.model_path, filter_mode=args.filter_mode)
else:
    logger.error('Unknown Model: ' + args.model_name)

# init model params
model.apply(model.init_weights)

# use gpu
if torch.cuda.device_count() > 0:
    model = model.cuda()
```

```python id="GCxtyvnmibfE"
# create discriminators
fair_disc_dict = {}
for feat_idx in data_reader.feature_info:
    fair_disc_dict[feat_idx] = \
        Discriminator(args.u_vector_size, data_reader.feature_info[feat_idx],
                        random_seed=args.random_seed, dropout=args.dropout, neg_slope=args.neg_slope,
                        model_dir_path=os.path.dirname(args.model_path))
    fair_disc_dict[feat_idx].apply(fair_disc_dict[feat_idx].init_weights)
    if torch.cuda.device_count() > 0:
        fair_disc_dict[feat_idx] = fair_disc_dict[feat_idx].cuda()

if args.runner in ['BaseRunner']:
    runner = runner_name(
        optimizer=args.optimizer, learning_rate=args.lr,
        epoch=args.epoch, batch_size=args.batch_size, eval_batch_size=args.vt_batch_size,
        dropout=args.dropout, l2=args.l2,
        metrics=args.metric, check_epoch=args.check_epoch, early_stop=args.early_stop)
elif args.runner in ['RecRunner']:
    runner = runner_name(
        optimizer=args.optimizer, learning_rate=args.lr,
        epoch=args.epoch, batch_size=args.batch_size, eval_batch_size=args.vt_batch_size,
        dropout=args.dropout, l2=args.l2,
        metrics=args.metric, check_epoch=args.check_epoch, early_stop=args.early_stop, num_worker=args.num_worker,
        no_filter=args.no_filter, reg_weight=args.reg_weight, d_steps=args.d_steps, disc_epoch=args.disc_epoch)
else:
    logger.error('Unknown Runner: ' + args.runner)

if args.load > 0:
    model.load_model()
    for idx in fair_disc_dict:
        fair_disc_dict[idx].load_model()
if args.train > 0:
    runner.train(model, data_processor_dict, fair_disc_dict, skip_eval=args.skip_eval, fix_one=args.fix_one)
```

```python id="ptjwSmJUih-h"
# reset seed
torch.manual_seed(args.random_seed)
torch.cuda.manual_seed_all(args.random_seed)
np.random.seed(args.random_seed)

if args.eval_disc:
    # Train extra discriminator for evaluation
    # create data reader
    disc_data_reader = DiscriminatorDataReader(path=args.path, dataset_name=args.dataset, sep=args.sep)

    # create data processor
    extra_data_processor_dict = {}
    for stage in ['train', 'test']:
        extra_data_processor_dict[stage] = DiscriminatorDataset(disc_data_reader, stage, args.disc_batch_size)

    # create discriminators
    extra_fair_disc_dict = {}
    for feat_idx in disc_data_reader.feature_info:
        if disc_data_reader.feature_info[feat_idx].num_class == 2:
            extra_fair_disc_dict[feat_idx] = \
                BinaryAttacker(args.u_vector_size, disc_data_reader.feature_info[feat_idx],
                                random_seed=args.random_seed, dropout=args.dropout,
                                neg_slope=args.neg_slope, model_dir_path=os.path.dirname(args.model_path),
                                model_name='eval')
        else:
            extra_fair_disc_dict[feat_idx] = \
                MultiClassAttacker(args.u_vector_size, disc_data_reader.feature_info[feat_idx],
                                    random_seed=args.random_seed, dropout=args.dropout, neg_slope=args.neg_slope,
                                    model_dir_path=os.path.dirname(args.model_path), model_name='eval')
        extra_fair_disc_dict[feat_idx].apply(extra_fair_disc_dict[feat_idx].init_weights)
        if torch.cuda.device_count() > 0:
            extra_fair_disc_dict[feat_idx] = extra_fair_disc_dict[feat_idx].cuda()

    if args.load_attack:
        for idx in extra_fair_disc_dict:
            logger.info('load attacker model...')
            extra_fair_disc_dict[idx].load_model()
    model.load_model()
    model.freeze_model()
    runner.train_discriminator(model, extra_data_processor_dict, extra_fair_disc_dict, args.lr_attack,
                                args.l2_attack)
```

```python id="IjNJ3ZmqikOC"
test_data = DataLoader(data_processor_dict['test'], batch_size=None, num_workers=args.num_worker,
                        pin_memory=True, collate_fn=data_processor_dict['test'].collate_fn)

test_result_dict = dict()
if args.no_filter:
    test_result = runner.evaluate(model, test_data)
else:
    test_result, test_result_dict = runner.eval_multi_combination(model, test_data, args.fix_one)

if args.no_filter:
    logger.info("Test After Training = %s "
                    % (format_metric(test_result)) + ','.join(runner.metrics))
else:
    logger.info("Test After Training:\t Average: %s "
                    % (format_metric(test_result)) + ','.join(runner.metrics))
    for key in test_result_dict:
        logger.info("test= %s "
                        % (format_metric(test_result_dict[key])) + ','.join(runner.metrics) +
                        ' (' + key + ') ')
```
