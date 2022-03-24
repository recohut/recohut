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

```python colab={"base_uri": "https://localhost:8080/"} id="DeilBegM_LnC" executionInfo={"status": "ok", "timestamp": 1623516294419, "user_tz": -330, "elapsed": 4904, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="36f9a71c-1bbe-4995-be86-6c74ae5d2188"
!pip install GPUtil
```

```python id="YrHhkJNbghNP" executionInfo={"status": "ok", "timestamp": 1623516771264, "user_tz": -330, "elapsed": 15, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import numpy as np
import pandas as pd
import sys
from collections import defaultdict
from typing import List, Tuple, Dict, Generator
from tqdm.notebook import tqdm
import seaborn as sns

import functools
import hashlib
import inspect
import json
import logging
from enum import Enum
from typing import List, Dict, Tuple, Iterator

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

import os
import time
from datetime import timedelta
import itertools

import gc
import subprocess
from datetime import datetime
from os import listdir
from os.path import isfile

import GPUtil as GPU
import humanize
import psutil

%matplotlib inline
sns.set_theme(style="whitegrid")
```

```python colab={"base_uri": "https://localhost:8080/"} id="LR03pKu4hTyH" executionInfo={"status": "ok", "timestamp": 1623516303103, "user_tz": -330, "elapsed": 2604, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="562b6642-dc1c-43a9-a751-057c8dfcb732"
!wget https://github.com/sparsh-ai/reco-data/raw/master/BookingChallenge.zip
!unzip BookingChallenge.zip
```

```python id="rjOWZsSq0ufD" executionInfo={"status": "ok", "timestamp": 1623516303105, "user_tz": -330, "elapsed": 37, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_SPLITS = 10
EPOCHS = 50
BATCH_SIZE = 256
EMBEDDING_SIZES = {
    'affiliate_id': (3611, 25),
    'booker_country': (5, 25),
    'checkin_day': (31, 5),
    'checkin_month': (12, 5),
    'checkin_year': (3, 5),
    'city_id': (39901, 128),
    'days_stay': (30, 5),
    'device_class': (3, 5),
    'hotel_country': (195, 25),
    'transition_days': (32, 5)
}
FEATURES_TO_ENCODE = ['city_id', 'device_class', 'affiliate_id',
                      'booker_country', 'hotel_country', 'checkin_year',
                      'days_stay', 'checkin_day', 'checkin_month',
                      'transition_days']
FEATURES_EMBEDDING = FEATURES_TO_ENCODE + ['next_' + column for column in
                                           ['affiliate_id', 'booker_country', 'days_stay', 'checkin_day']]
```

```python id="ieAPVx3P1Hfv" executionInfo={"status": "ok", "timestamp": 1623516303107, "user_tz": -330, "elapsed": 38, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
class ModelType(str, Enum):
    MANY_TO_ONE = 1
    MANY_TO_MANY = 2


class WeightType(str, Enum):
    UNWEIGHTED = 1
    UNIFORM = 2
    CUMSUM_CORRECTED = 3


class RecurrentType(str, Enum):
    GRU = 1
    LSTM = 2


class FeatureProjectionType(str, Enum):
    CONCATENATION = 1
    MULTIPLICATION = 2


class OptimizerType(str, Enum):
    ADAM = 1
    ADAMW = 2


BatchType = Dict[str, torch.Tensor]
```

```python id="_M8vXhJp_wlF" executionInfo={"status": "ok", "timestamp": 1623516303108, "user_tz": -330, "elapsed": 37, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
RESOURCES_PATH = './content'


def get_resources_path(file_path: str) -> str:
    """
    Get resources path from file path.
    """
    path = os.path.join(RESOURCES_PATH, file_path)
    dirs = '/'.join(path.split('/')[:-1])

    if not os.path.exists(dirs):
        os.makedirs(dirs)

    return path


def get_path(*args, dirs=None, format=None, filename=None, **kwargs) -> str:
    """
    Get path from args and kwargs.
    """
    path = []
    for arg in args:
        path.append(str(arg))
    for k, v in kwargs.items():
        if isinstance(v, bool):
            if v:
                path.append(k)
        else:
            path.append('{}_{}'.format(k, v))

    dirs_str = ''
    if dirs is not None:
        if type(dirs) is not list:
            dirs = [dirs]
        dirs_str = '/'.join(dirs) + '/'

    path = get_resources_path(dirs_str + '_'.join(path))
    if filename is not None:
        path += filename
    if format is not None:
        path += "." + format

    return path


def get_model_ckpt_paths(model_hash: str, checkpoint_type='accuracy_at_k') -> List:
    """
    Get model checkpoints paths from `model_hash` by `checkpoint_type`.
    """
    base_path = get_path(f"models/{model_hash}")
    ckpt_paths = [f"{base_path}/{f}" for f in listdir(base_path) if isfile(f"{base_path}/{f}")]
    return sorted(list(filter(lambda s: checkpoint_type in s, ckpt_paths)))


def get_model_arch_path(model_hash) -> str:
    """
    Get model architecture paths from `model_hash`.
    """
    return get_path(dirs="architectures",
                    hash=model_hash,
                    filename=None,
                    format='json')
```

```python id="mgaMnKMs-eD9" executionInfo={"status": "ok", "timestamp": 1623518878795, "user_tz": -330, "elapsed": 476, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
class LabelEncoder:
    """
    LabelEncoder similar to `sklearn.preprocessing.LabelEncoder`
    with the exception it ignores `NaN` values.
    .. todo:: Enhance this encoder with the option to set a `min_frequency`.
    """

    def fit_transform(self, col: pd.Series) -> pd.Series:
        self.rev_classes_ = dict(enumerate(sorted(col.dropna().unique())))
        self.classes_ = {v: k for k, v in self.rev_classes_.items()}
        return col.apply(lambda k: self.classes_.get(k, np.nan))

    def inverse_transform(self, col: pd.Series) -> pd.Series:
        return col.apply(lambda k: self.rev_classes_.get(k, np.nan))


class DatasetEncoder:
    """
    DatasetEncoder looks to encapsulate multiple LabelEncoder objects
    to fully transform a dataset.
    """

    def __init__(self, features_embedding: List[str]):
        self.label_encoders = {c: LabelEncoder() for c in features_embedding}

    def fit_transform(self, df: pd.DataFrame) -> None:
        """
        Transform columns in all columns given by feature_embedding.
         df:
        :return:
        """
        logging.info("Running LabelEncoder on columns")
        for column, encoder in self.label_encoders.items():
            # reserve zero index for OOV elements
            df[column] = encoder.fit_transform(df[column]) + 1
            logging.info(f"{column}: {len(encoder.classes_)}")


def get_embedding_complexity_proxy(dataset_encoder: DatasetEncoder) -> Dict:
    """
    Get embedding complexity proxy
    The idea is to find out how many bits (dimension) we need to naively encode each element in the encoder.
    It's a proxy since we have no idea which is the dimension of the underlying manifold for every feature.
    """
    return {k: (len(v.classes_), np.ceil(np.log2(len(v.classes_))))
            for k, v in dataset_encoder.label_encoders.items()}
```

```python id="d0THktfNx2Zl" executionInfo={"status": "ok", "timestamp": 1623518881208, "user_tz": -330, "elapsed": 999, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
_NEXT_CITY_COLUMNS = ['city_id', 'affiliate_id',
                      'booker_country', 'days_stay',
                      'checkin_day']


def build_dataset(reserved_obs: int = 10000) -> pd.DataFrame:
    """
    Builds dataset by unifying training and test set.
    :return: pd.DataFrame with unified dataset.
    """
    train_set = pd.read_csv('train_set.csv', index_col=0,
                            dtype={'user_id': 'int32', 'city_id': 'int32'},
                            parse_dates=['checkin', 'checkout']).sort_values(by=['utrip_id', 'checkin'])
    test_set = pd.read_csv('test_set.csv',
                           dtype={'user_id': 'int32', 'city_id': 'int32'},
                           parse_dates=['checkin', 'checkout']).sort_values(by=['utrip_id', 'checkin'])

    # create dataset identifiers and homogenize dataframes
    train_set['train'] = 1
    test_set['train'] = 0
    # test_set.drop(columns=['row_num', 'total_rows'], inplace=True)
    test_set['city_id'] = test_set['city_id'].replace({0: np.nan})

    # reserve observations for sanity check
    train_set['reserved'] = np.arange(len(train_set)) <= reserved_obs

    # unify datasets
    dataset = pd.concat([train_set, test_set])

    # create some time features
    dataset['days_stay'] = (dataset['checkout'] - dataset['checkin']).dt.days - 1
    dataset['checkin_day'] = dataset['checkin'].dt.dayofweek
    dataset['checkin_month'] = dataset['checkin'].dt.month
    dataset['checkin_year'] = dataset['checkin'].dt.year

    # create transition time feature
    dataset['prev_checkout'] = dataset.groupby('utrip_id')['checkout'].shift(periods=1)
    dataset['transition_days'] = (dataset['checkout'] - dataset['prev_checkout']).dt.days - 1
    dataset['transition_days'].fillna(0, inplace=True)
    dataset.drop(columns="prev_checkout", inplace=True)
    return dataset


def set_future_features(df: pd.DataFrame) -> None:
    """
    Add features about the next city to the dataframe.
    """
    for column in _NEXT_CITY_COLUMNS:
        df['next_' + column] = df.groupby('utrip_id')[column].shift(periods=-1)


def get_training_set_from_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get training set by ignoring reserved test set observations.
    """
    return df[df.reserved != True]


def get_test_set_from_dataset(df: pd.DataFrame,
                              sequence_length: int = 3) -> pd.DataFrame:
    """
    Get test set from unified dataframe and constrain the minimum
    sequence length to avoid a test/submissions set distribution mismatch.
    """
    test_set = df[df.reserved == True]
    return min_sequence_length_transformer(test_set, sequence_length)


def get_submission_set_from_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get submission set from dataset, filtering `NaN` cities that appeared
    when merging the training and test set.
    .. warning::
        You should create the submission set before filtering `NaN`.
    """
    submission_set = df[(df.train == 0) & (~df.city_id.isna())]
    assert len(submission_set) == 308005
    return submission_set


def min_sequence_length_transformer(df: pd.DataFrame,
                                    sequence_length: int = 3) -> pd.DataFrame:
    """
    Constrains the minimum trip length to `sequence_length`.
    """
    return df.groupby('utrip_id').filter(lambda x: len(x) >= sequence_length)
```

```python id="me4XIJ9Zx2WS" executionInfo={"status": "ok", "timestamp": 1623518881211, "user_tz": -330, "elapsed": 15, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
class BookingDataset(Dataset):

    def __init__(self,
                 df: pd.DataFrame,
                 features: List[str],
                 group_var='utrip_id'):
        sorted_groups = sorted(df.groupby(group_var), key=lambda g: len(g[1]), reverse=True)
        self.trips = [BookingDataset.pre_process(group, features) for _, group in tqdm(sorted_groups)]
        self.utrip_ids = [utrip_id for utrip_id, _ in sorted_groups]
        self.group_lengths = [len(g[1]) for g in sorted_groups]

    def __len__(self):
        return len(self.trips)

    def __getitem__(self, idx):
        return self.trips[idx]

    def get_ids(self):
        return pd.DataFrame({'utrip_id': self.utrip_ids,
                             'group_length': self.group_lengths})

    @staticmethod
    def pre_process(group: pd.DataFrame, features: List[str]):
        g = group[features].to_dict(orient='list')
        return {k: torch.LongTensor(np.array(v)) for k, v in g.items()}


def pad_collate(batch: List[BatchType]):
    """
    Unify observations in a padded batch dictionary.
    """
    batch_dict = defaultdict(list)
    lengths = []
    for d in batch:
        for k, v in d.items():
            batch_dict[k].append(v)
        # add the next city id if we are training
        if 'next_city_id' in d:
            batch_dict['last_city'].append(d['next_city_id'][-1])
        lengths.append(v.size())

    res = {k: pad_sequence(v, batch_first=True, padding_value=0)
           for k, v in batch_dict.items() if k != 'last_city'}

    # add last city id if we are training
    if 'next_city_id' in d:
        res['last_city'] = torch.tensor(batch_dict['last_city'])

    lengths = torch.tensor(lengths, dtype=torch.int64).squeeze()
    return res, lengths


def get_dataset_and_dataloader(df: pd.DataFrame,
                               features: List[str],
                               batch_size: int = 256) -> Tuple[BookingDataset, DataLoader]:
    """
    Get dataset and dataloader.
    """
    dataset = BookingDataset(df, features)
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             collate_fn=pad_collate)
    return dataset, data_loader


def batches_to_device(data_loader: DataLoader) -> np.array:
    """
    Batches to device.
    By pre-loading all batches in GPU for training, we avoid transferring data
    from memory to GPU on every fold. The risk of doing this is biasing the gradients,
    reason why we are then careful with the distribution of batches on each fold,
    also shuffling the batches every time we train a model.
    """
    if DEVICE == 'cpu':
        batches = np.array([({k: v for k, v in d.items()}, seq_len)
                            for (d, seq_len) in data_loader])
    else:
        batches = np.array([({k: v.cuda(non_blocking=True)
                              for k, v in d.items()}, seq_len) for (d, seq_len) in data_loader])

    return batches


def filter_batches_by_length(batches: List[BatchType], min_length: int = 3):
    """
    Filter batches to have a minimum length of `min_length`.
    """
    return list(filter(lambda b: b[1].min().item() > min_length, batches))
```

```python id="4ZCjx-m20OxL" executionInfo={"status": "ok", "timestamp": 1623518881213, "user_tz": -330, "elapsed": 16, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
class BookingNet(nn.Module):
    """
    BookingNet Sequence Aware Recommender System Network
    """

    def __init__(self,
                 features_embedding: List[str],
                 hidden_size: int,
                 output_size: int,
                 embedding_sizes: Dict[str, Tuple[int, int]],
                 n_layers: int = 2,
                 dropout: float = 0.3,
                 rnn_dropout: float = 0.1,
                 tie_embedding_and_projection: bool = True,
                 model_type: ModelType = ModelType.MANY_TO_MANY,
                 recurrent_type: RecurrentType = RecurrentType.GRU,
                 weight_type: WeightType = WeightType.UNWEIGHTED,
                 feature_projection_type: FeatureProjectionType = FeatureProjectionType.CONCATENATION,
                 **kwargs: List):
        """
        Args:
             features_embedding: Features to embed at each time step.
             hidden_size: Hidden size of the recurrent encoder (`LSTM` or `GRU`).
             output_size: Quantity of cities to predict.
             embedding_sizes: Sizes of each feature embedding.
             n_layers: Number of recurrent layers.
             dropout: Dropout used in our input layer.
             rnn_dropout: Dropout used in recurrent layer.
             recurrent_type: Select between `RecurrentType.GRU` or `RecurrentType.LSTM`
             tie_embedding_and_projection: If `true`, parameterize last linear layer with embedding matrix.
             feature_projection_type: Select between `FeatureCombinationType.CONCATENATION`
                or `FeatureCombinationType.MULTIPLICATION`
             model_type: The model can either only predict the last city (`ModelType.MANY_TO_ONE`) or
                predict every city in the sequence (`ModelType.MANY_TO_MANY`)
             weight_type:
                1. `WeightType.UNWEIGHTED`: Unweighted cross entropy.
                2. `WeightType.UNIFORM`: Uniform cross entropy.
                3. `WeightType.CUMSUM_CORRECTED`: Cross entropy corrected to reflect original
                    one to many weighting.
        """
        super().__init__()
        # save model arguments to re-initialize later
        model_params = inspect.getargvalues(inspect.currentframe()).locals
        if 'kwargs' in model_params:
            model_params.update(model_params['kwargs'])
            model_params.pop('kwargs')
        model_params.pop('__class__')
        model_params.pop('self')
        self.model_params = model_params

        self.features_embedding = features_embedding
        self.hidden_size = hidden_size
        self.target_variable = "next_city_id"
        self.embedding_layers = nn.ModuleDict(
            {key: nn.Embedding(num_embeddings=int(qty_embeddings) + 1,  # reserve 0 index for padding/OOV.
                               embedding_dim=int(size_embeddings),
                               max_norm=None,  # Failed experiment, enforcing spherical embeddings degraded performance.
                               norm_type=2,
                               padding_idx=0)
             for key, (qty_embeddings, size_embeddings) in embedding_sizes.items()})

        # encode every variable with the prefix `next_` to the embedding matrix of the suffix.
        self.features_dim = int(np.sum([embedding_sizes[k.replace("next_", "")][1]
                                        for k in self.features_embedding]))
        self.city_embedding_size = embedding_sizes['city_id'][1]

        self.feature_combination_type = feature_projection_type
        self.tie_embedding_and_projection = tie_embedding_and_projection
        self.recurrent_encoder = self.get_recurrent_encoder(recurrent_type, n_layers, rnn_dropout)

        if feature_projection_type == FeatureProjectionType.MULTIPLICATION:
            self.attn_weights = nn.ParameterDict(
                {key: nn.Parameter(torch.rand(1)) for key in self.features_embedding}
            )

        if self.city_embedding_size != self.hidden_size:
            logging.info(
                f"Warning: Using linear layer to reconcile output of size "
                f"{self.hidden_size} with city embedding of size {self.city_embedding_size}.")
            self.linear_to_city = nn.Linear(self.hidden_size,
                                            self.city_embedding_size,
                                            bias=False)

        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(self.city_embedding_size, output_size, bias=False)

        if self.tie_embedding_and_projection:
            # ignore first embedding, since it corresponds to padding/OOV
            self.dense.weight = nn.Parameter(self.embedding_layers['city_id'].weight[1:])

        # self.initialize_parameters()

        # other parameters
        self.loss = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
        self.model_type = model_type
        self.weight_type = weight_type
        self.optimizer = None
        self.cross_entropy_weights = None

    def forward(self, batch: BatchType, seq_length: torch.Tensor):
        seq_length = seq_length.squeeze()

        # build feature map
        feature_input = self.get_feature_input(batch)
        feature_input = self.dropout(feature_input)

        # sequence encoder
        feature_input = nn.utils.rnn.pack_padded_sequence(feature_input,
                                                          seq_length,
                                                          batch_first=True,
                                                          enforce_sorted=False)
        seq_out, _ = self.recurrent_encoder(feature_input)
        seq_out, _ = nn.utils.rnn.pad_packed_sequence(seq_out,
                                                      batch_first=True)

        # reconcile encoder output size with city embedding size
        if self.city_embedding_size != self.hidden_size:
            seq_out = self.linear_to_city(seq_out)

        # create final predictions (no softmax)
        city_encoding = self.dropout(seq_out)
        dense_out = self.dense(city_encoding)
        return dense_out

    def get_feature_input(self, batch: BatchType):
        if self.feature_combination_type == FeatureProjectionType.CONCATENATION:
            return self.feature_concatenation(batch)
        else:
            return self.feature_multiplication(batch)

    def feature_concatenation(self, batch: BatchType):
        """
        Enables feature concatenation for every sequential step.
        """
        feature_list = [self.embedding_layers[k.replace("next_", "")](batch[k]) for k in self.features_embedding]
        return torch.cat(feature_list, axis=2)

    def feature_multiplication(self, batch: BatchType):
        """
        Enables feature multiplication for every sequential step.
        """
        attention_embs = [self.attn_weights[k] * self.embedding_layers[k.replace("next_", "")](batch[k])
                          for k in self.features_embedding if k != 'city_id']
        attention = functools.reduce(lambda a, b: a + b, attention_embs)
        return self.embedding_layers['city_id'](batch['city_id']) * attention

    def get_loss(self,
                 city_scores: torch.Tensor,
                 batch: BatchType,
                 seq_len: torch.Tensor,
                 device=DEVICE) -> torch.Tensor:
        """
        Loss function computation for the network, depending on model type:
        Args:
            1. `ModelType.MANY_TO_ONE`: Train many to one sequential model.
            2. `ModelType.MANY_TO_MANY`: Train many to many sequential model.
        """
        bs, ts = batch['city_id'].shape
        loss = self.loss(city_scores, batch['next_city_id'].view(-1) - 1)
        loss = loss.view(-1, ts)
        if self.model_type == ModelType.MANY_TO_ONE:
            return torch.sum(loss * torch.nn.functional.one_hot(seq_len - 1).to(device)) / torch.sum(seq_len)
        elif self.model_type == ModelType.MANY_TO_MANY:
            if isinstance(self.cross_entropy_weights, int):
                return torch.sum(loss) / torch.sum(seq_len)
            else:
                # TODO: Find a way to control for variance. Batches with less
                #  subsequences should have a lower weight.
                return torch.sum(self.cross_entropy_weights[:ts] * loss) / torch.sum(seq_len)
        else:
            logging.error('Invalid model type in get_loss().')

    def get_recurrent_encoder(self,
                              recurrent_type: RecurrentType,
                              n_layers: int,
                              dropout: float):
        if recurrent_type == RecurrentType.LSTM:
            return nn.LSTM(self.features_dim,
                           self.hidden_size,
                           num_layers=n_layers,
                           dropout=dropout,
                           batch_first=True)
        elif recurrent_type == RecurrentType.GRU:
            return nn.GRU(self.features_dim,
                          self.hidden_size,
                          num_layers=n_layers,
                          dropout=dropout,
                          batch_first=True)
        else:
            logging.error('Invalid recurrent encoder type in get_recurrent_encoder().')

    def set_optimizer(self, optimizer_type: OptimizerType) -> None:
        if optimizer_type == OptimizerType.ADAMW:
            self.optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=0.001,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=0.01,
                amsgrad=False)
        elif optimizer_type == OptimizerType.ADAM:
            self.optimizer = torch.optim.Adam(
                self.parameters(),
                lr=0.001,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=0,
                amsgrad=False)
        else:
            logging.error('Invalid optimizer type in set_optimizer().')

    def set_entropy_weights(self,
                            train_set: pd.DataFrame):
        """
        Set entropy weights for `ModelType.MANY_TO_MANY`. These weights
        depend on the `WeightType` passed in the constructor.
        """
        if self.weight_type is WeightType.UNWEIGHTED:
            self.cross_entropy_weights = 1
        elif self.weight_type in (WeightType.UNIFORM, WeightType.CUMSUM_CORRECTED):
            weights_train = dict(train_set.groupby('utrip_id').size().value_counts().items())
            weights_train = np.array([weights_train.get(k, 0) for k in range(1, 50)])
            numerator = 1 if self.weight_type == WeightType.UNIFORM else weights_train
            reweighting = numerator / np.cumsum(weights_train[::-1])[::-1]

            if np.any(np.isnan(reweighting)):
                logging.warning('Warning: NaN found in weights.')

            reweighting[np.isnan(reweighting)] = 0
            reweighting[np.isinf(reweighting)] = 0
            self.cross_entropy_weights = torch.tensor(reweighting, device=DEVICE)
        else:
            logging.error(f"Unknown weight type {self.weight_type} in set_entropy_weights()")

        logging.info(f'Weights: {self.cross_entropy_weights}')

    def initialize_parameters(self):
        """
        Network parameter initialization. Ended up using the default one.
        """
        # https://pytorch.org/docs/stable/nn.init.html
        for name, param in self.named_parameters():
            if len(param.shape) > 1:
                logging.info(f"Initializing {name}")
                nn.init.xavier_uniform_(param)

    def __str__(self):
        return json.dumps(self.model_params, indent=4, sort_keys=True)

    @property
    def hash(self):
        """
        Unique model hash for checkpoint/metrics identification.
        """
        return hashlib.md5(self.__str__().encode('utf-8')).hexdigest()[:8]


def get_model_predictions(model: BookingNet,
                          data_loader: DataLoader,
                          model_ckpt_path: str) -> Iterator[torch.FloatTensor]:
    """
    Get model predictions model checkpoint and batches data loader.
    """
    model.load_state_dict(
        torch.load(model_ckpt_path,
                   map_location=torch.device(DEVICE))
    )
    model.eval()
    with torch.no_grad():
        for batch, seq_len in data_loader:
            if DEVICE == 'cuda':
                batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}

            city_scores = model(batch, seq_len)
            city_scores = torch.bmm(
                torch.nn.functional.one_hot(seq_len - 1).unsqueeze(dim=1).type(torch.FloatTensor).to(DEVICE),
                city_scores).squeeze()
            city_scores = nn.Softmax(dim=1)(city_scores)
            yield city_scores.cpu()
```

```python id="etbEi2bU1QPl" executionInfo={"status": "ok", "timestamp": 1623518881215, "user_tz": -330, "elapsed": 16, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def round_robin_kfold(batches: List[BatchType],
                      n_splits: int = 10) -> Generator:
    """
    Round robin k-fold cross validation.
    Useful when batches are sorted by sequence length, to keep the training
    set and validation set as balanced as possible in sequence length.
    Train indices are shuffled to try to reduce the bias in the gradient updates.
    """
    np.random.seed(42)
    n = len(batches)
    groups = defaultdict(list)

    group_id = 0
    for i in range(n):
        groups[group_id % n_splits].append(i)
        group_id += 1

    for i in range(n_splits):
        train_index = np.concatenate([group for group_id, group in groups.items()
                                      if group_id != i])
        valid_index = np.array(groups[i])
        np.random.shuffle(train_index)
        yield train_index, valid_index
```

```python id="3RVKfKzW1sLJ" executionInfo={"status": "ok", "timestamp": 1623518881929, "user_tz": -330, "elapsed": 729, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def get_model_metrics(batch: BatchType,
                      seq_len: torch.Tensor,
                      city_scores: torch.Tensor) -> Tuple:
    """
    Get model metrics, e.g. accuracy@1, accuracy@4.
    """
    bs, ts = batch['city_id'].shape
    predicted_cities = (city_scores
                        .argmax(1)
                        .view(-1, ts)
                        .gather(1, (seq_len.unsqueeze(1) - 1).to(DEVICE))
                        .squeeze(1) + 1)
    predicted_cities_top_k = (torch.topk(city_scores, 4, dim=1).indices.view(bs, -1, 4)
                              .gather(1, (torch.cat([seq_len.unsqueeze(1)] * 4, axis=1).view(-1, 1, 4) - 1)
                                      .to(DEVICE))).squeeze(1) + 1
    hits_at_1 = (predicted_cities == batch['last_city']).float().sum()
    hits_at_k = torch.sum(predicted_cities_top_k.eq(batch['last_city'].unsqueeze(1)), dim=1).float().sum()
    return hits_at_1, hits_at_k


def train_step(model: BookingNet,
               batch: List[BatchType]) -> Dict:
    """
    Training step, including loss evaluation and backprop.
    """
    batch, seq_len = batch
    model.optimizer.zero_grad(set_to_none=True)
    city_scores = model(batch, seq_len)
    city_scores = city_scores.view(-1, 39901)
    loss = model.get_loss(city_scores,
                          batch,
                          seq_len,
                          device=DEVICE)
    loss.backward()
    model.optimizer.step()
    return {
        'train_loss': loss.item()
    }


def validation_step(model: BookingNet,
                    batch: BatchType) -> Dict:
    """
    Validation step, including metric computation for batch.
    """
    batch, seq_len = batch
    city_scores = model(batch, seq_len)
    city_scores = city_scores.view(-1, 39901)
    loss = model.get_loss(city_scores,
                          batch,
                          seq_len)
    hits_at_1, hits_at_k = get_model_metrics(batch, seq_len, city_scores)
    obs = len(batch['city_id'])
    return {
        'valid_loss': loss.item(),
        'hits_at_1': hits_at_1.item(),
        'hits_at_k': hits_at_k.item(),
        'obs': obs
    }


def train_for_all_batches(model: BookingNet,
                          train_batches: List[BatchType]) -> Dict:
    """
    Train model on all given batches.
    """
    current_time = time.time()
    train_loss = 0
    model.train()
    for batch in train_batches:
        train_step_result = train_step(model, batch)
        train_loss += train_step_result['train_loss']
    train_loss /= len(train_batches)  # loss per batch
    ellapsed_time = timedelta(seconds=int(time.time() - current_time))
    return {
        'train_loss': train_loss,
        'ellapsed_time': ellapsed_time,
    }


def valid_for_all_batches(model: BookingNet,
                          valid_batches: List[BatchType]) -> Dict:
    """
    Run validation set metrics for all batches.
    """
    current_time = time.time()
    valid_result = {
        'valid_loss': 0,
        'hits_at_1': 0,
        'hits_at_k': 0,
        'obs': 0
    }
    model.eval()
    with torch.no_grad():
        for batch in valid_batches:
            batch_result = validation_step(model, batch)
            for key in valid_result.keys():
                valid_result[key] += batch_result[key]
    ellapsed_time = timedelta(seconds=int(time.time() - current_time))
    return {
        'valid_loss': valid_result['valid_loss'] / len(valid_batches),  # loss per batch
        'accuracy@1': valid_result['hits_at_1'] / valid_result['obs'],
        'accuracy@4': valid_result['hits_at_k'] / valid_result['obs'],
        'ellapsed_time_valid': ellapsed_time
    }


def model_checkpoint_exists(model_hash: str,
                            fold: int) -> bool:
    """
    Returns `true` if the model checkpoint given by the path exists, `false` otherwise.
    """
    ckpt_path = get_path(dirs=["models", model_hash],
                         filename=f"fold_{fold}_best_accuracy_at_k",
                         format="pt")
    # ckpt_path = f"./models/{models.hash}/fold_{fold}_best_accuracy_at_k.pt"
    return os.path.exists(ckpt_path)


def train_model(model: BookingNet,
                train_batches: List[BatchType],
                valid_batches: List[BatchType],
                epochs: int = 50,
                fold: int = 0,
                min_epochs_to_save: int = 20,
                verbose: bool = True) -> pd.DataFrame:
    """
    Train model from batches and save checkpoints of best models by accuracy.
    """
    epoch_report = {}
    best_accuracy_at_k = 0
    for epoch in tqdm(range(epochs)):
        train_report = train_for_all_batches(model, train_batches)
        valid_report = valid_for_all_batches(model, valid_batches)

        if epoch >= min_epochs_to_save and valid_report['accuracy@4'] > best_accuracy_at_k:
            best_accuracy_at_k = valid_report['accuracy@4']
            torch.save(model.state_dict(), get_path(dirs=["models", model.hash],
                                                    filename=f"fold_{fold}_best_accuracy_at_k",
                                                    format="pt"))
            # torch.save(model.state_dict(),  f"./models/{models.hash}/fold_{fold}_best_accuracy_at_k.pt")

        r = dict(train_report)
        r.update(valid_report)
        epoch_report[epoch] = r

        if verbose:
            epoch_str = [f"Epoch: {epoch}",
                         f"train loss: {r['train_loss']:.4f}",
                         f"valid loss: {r['valid_loss']:.4f}",
                         f"accuracy@1: {r['accuracy@1']:.4f}",
                         f"accuracy@4: {r['accuracy@4']:.4f}",
                         f"time: {r['ellapsed_time']}"]
            epoch_str = ', '.join(epoch_str)
            logging.info(epoch_str)

    # save report
    pd.DataFrame(epoch_report).T.to_csv(get_path(dirs=["reports", model.hash],
                                                 hash=model.hash,
                                                 fold=fold,
                                                 format='csv'))
    # pd.DataFrame(epoch_report).T.to_csv(f"./reports/{models.hash}/fold_{fold}.csv")

    # with open(f"architectures/{model.hash}", "w") as fhandle:
    with open(get_model_arch_path(model.hash), "w") as fhandle:
        fhandle.write(str(model))

    return pd.DataFrame(epoch_report).T


def train_model_for_folds(dataset_batches: List[BatchType],
                          train_set: pd.DataFrame,
                          model_configuration: Dict,
                          n_models: int = N_SPLITS,
                          min_epochs_to_save: int = 25,
                          skip_checkpoint=False) -> str:
    """
    Train `n_models` given a model configuration, returning the model hash.
    """
    for fold, (train_index, valid_index) in enumerate(round_robin_kfold(dataset_batches,
                                                                        n_splits=N_SPLITS)):
        if fold >= n_models:
            break

        model = BookingNet(**model_configuration).to(DEVICE)
        model.set_optimizer(optimizer_type=OptimizerType.ADAMW)
        model.set_entropy_weights(train_set)

        model_hash = model.hash

        if not skip_checkpoint and model_checkpoint_exists(model.hash, fold):
            continue

        train_batches = dataset_batches[train_index]
        valid_batches = dataset_batches[valid_index]
        # valid_batches = filter_batches_by_length(valid_batches)

        logging.info(f"Training model {model.hash} for fold {fold}")
        train_model(model,
                    train_batches,
                    valid_batches,
                    epochs=EPOCHS,
                    min_epochs_to_save=min_epochs_to_save,
                    fold=fold)

        # Empty CUDA memory
        del model
        torch.cuda.empty_cache()

    return model_hash
```

```python id="q8Yv3qtS-tvA" executionInfo={"status": "ok", "timestamp": 1623518881931, "user_tz": -330, "elapsed": 6, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def print_gpu_usage(gpu_id: int = 0):
    """
    Display GPU usage.
    """
    gpu_list = GPU.getGPUs()
    gpu = gpu_list[gpu_id]
    process = psutil.Process(os.getpid())
    logging.info(f"Gen RAM Free: {humanize.naturalsize(psutil.virtual_memory().available)}"
                 f" | Proc size: {humanize.naturalsize(process.memory_info().rss)}")
    logging.info("GPU RAM Free: {0:.0f}MB | Used: {1:.0f}MB | Util {2:3.0f}% | Total {3:.0f}MB".format(gpu.memoryFree,
                                                                                                       gpu.memoryUsed,
                                                                                                       gpu.memoryUtil * 100,
                                                                                                       gpu.memoryTotal))


def accuracy_at_k(submission: pd.DataFrame,
                  ground_truth: pd.DataFrame) -> Dict:
    """
    Calculates accuracy@k for k in {1, 4, 10} by group length and overall.
    """
    data_to_eval = submission.join(ground_truth, on='utrip_id')

    for k in [1, 4, 10]:
        data_to_eval[f'hits_at_{k}'] = data_to_eval.apply(
            lambda row: row['city_id'] in row[[f'city_id_{i}' for i in range(1, k + 1)]].values, axis=1)
    return {
        'accuracy@1': data_to_eval['hits_at_1'].mean(),
        'accuracy@4': data_to_eval['hits_at_4'].mean(),
        'accuracy@10': data_to_eval['hits_at_10'].mean(),
        'accuracy@4_by_pos': data_to_eval.groupby('group_length')['hits_at_4'].mean().to_dict()
    }


def get_submission(dataset: BookingDataset,
                   data_loader: DataLoader,
                   model: BookingNet,
                   checkpoint_path_list: List[str],
                   dataset_encoder: DatasetEncoder) -> pd.DataFrame:
    """
    Get submission from dataset.
    """
    assert len(checkpoint_path_list) > 0

    ensemble_batch_probs = None
    for checkpoint_path in tqdm(checkpoint_path_list):
        batch_probs_generator = get_model_predictions(model,
                                                      data_loader,
                                                      checkpoint_path)
        if ensemble_batch_probs is None:
            ensemble_batch_probs = list(batch_probs_generator)
        else:
            for i, batch_probs in enumerate(batch_probs_generator):
                ensemble_batch_probs[i] += batch_probs

    top_cities = torch.cat(
        [torch.topk(batch_submission, 10, dim=1).indices + 1
         for batch_submission in ensemble_batch_probs],
        axis=0
    )
    del ensemble_batch_probs
    cities_prediction = pd.DataFrame(top_cities.numpy(),
                                     columns=[f'city_id_{i}' for i in range(1, 11)])
    del top_cities
    gc.collect()

    for city_id in range(1, 11):
        cities_prediction[f'city_id_{city_id}'] = dataset_encoder.label_encoders['city_id'].inverse_transform(
            cities_prediction[f'city_id_{city_id}'] - 1).astype(int)

    submission = pd.concat([dataset.get_ids(), cities_prediction], axis=1)
    return submission


def get_ground_truth_from_dataset(df: pd.DataFrame,
                                  booking_dataset: BookingDataset,
                                  dataset_encoder: DatasetEncoder) -> pd.DataFrame:
    """
    Get ground truth from dataset. Assumes the df is sorted by checkin ASC.
    """
    ground_truth = df.groupby('utrip_id').tail(1)[['utrip_id', 'next_city_id']].set_index('utrip_id')
    ground_truth['city_id'] = (dataset_encoder
                               .label_encoders['city_id']
                               .inverse_transform(ground_truth['next_city_id'] - 1))
    if not ground_truth['city_id'].isnull().values.any():
        ground_truth['city_id'] = ground_truth['city_id'].astype(int)
    else:
        logging.warning("Warning: next_city_id has nulls")

    ground_truth = ground_truth.loc[booking_dataset.utrip_ids]  # reorder obs like batches
    ground_truth.drop(columns="next_city_id", inplace=True)
    return ground_truth


def get_count_distribution(df: pd.DataFrame,
                           by: str = 'utrip_id') -> pd.DataFrame:
    """
    Get count distribution from dataset.
    """
    df_dist = df.groupby(by)[by].count().value_counts(sort=True)
    df_dist /= df_dist.sum()
    return df_dist


def get_distribution_by_pos(**kwargs) -> pd.DataFrame:
    """
    Get distribution by pos from a list of key: dataframe pairs.
    """
    return functools.reduce(lambda a, b: a.join(b),
                            [get_count_distribution(df).to_frame(name)
                             for name, df in kwargs.items()]).sort_index()


def check_device() -> None:
    """
    Check if we are using GPU acceleration and warn the user.
    """
    if DEVICE != 'cuda':
        logging.warning('You are not using a GPU. If you are using colab, go to Runtime -> Change runtime type')
    else:
        current_gpu = subprocess.check_output(['nvidia-smi', '-L']).strip().decode('ascii')
        logging.info(f"Using {current_gpu}")


def get_trained_models() -> Dict:
    """
    Get dictionary of all models trained
    """
    base_path = get_path("architectures")
    model_paths = [f"{base_path}/{f}" for f in listdir(base_path) if isfile(f"{base_path}/{f}")]

    d = {}

    for path in model_paths:
        with open(path) as f:
            model_hash = path[-13:-5]
            d[model_hash] = json.load(f)
    return d


def get_final_submission(submission_set: pd.DataFrame,
                         model_hash: str,
                         dataset_encoder: DatasetEncoder) -> None:
    """
    Get final submission from model hash.
    """
    # create final submission
    dataset_submission, data_loader_submission = get_dataset_and_dataloader(
        df=submission_set,
        features=FEATURES_EMBEDDING
    )

    # get model parameters from hash
    with open(get_model_arch_path(model_hash)) as fhandle:
        model_parameters = json.load(fhandle)
    ckpt_list = get_model_ckpt_paths(model_hash=model_hash,
                                     checkpoint_type='accuracy_at_k')

    # load model and get predictions
    model = BookingNet(**model_parameters).to(DEVICE)
    predictions = get_submission(dataset_submission,
                                 data_loader_submission,
                                 model,
                                 ckpt_list,
                                 dataset_encoder)

    # build final csv and run sanity checks
    timestamp = datetime.now().strftime("%d_%m_%Y_%Hh_%Mm_%Ss")
    cols = ["utrip_id", "city_id_1", "city_id_2", "city_id_3", "city_id_4"]
    filename = f'submission_{model_hash}_{timestamp}'
    final_submission = predictions[cols]
    final_submission.to_csv(get_path(dirs="submissions",
                                     filename=filename,
                                     format='csv'),
                            index=False)
    submission_sanity_checks(final_submission)


def submission_sanity_checks(submission: pd.DataFrame) -> None:
    """
    Run submission sanity checks to make sure our dataframe is healthy.
    """
    _TOTAL_SUBMISSION_ROWS = 70662
    df = pd.read_csv(get_resources_path('booking_test_set.csv'),
                     dtype={'user_id': 'int32', 'city_id': 'int32'},
                     parse_dates=['checkin', 'checkout'])

    utrip_ids = set(df.utrip_id.unique())
    assert len(set(submission.utrip_id.unique()).intersection(utrip_ids)) == _TOTAL_SUBMISSION_ROWS
    assert submission.shape == (_TOTAL_SUBMISSION_ROWS, 5)
    assert submission.notna().values.all()

    df = pd.read_csv(get_resources_path('booking_train_set.csv'),
                     dtype={'user_id': 'int32', 'city_id': 'int32'},
                     parse_dates=['checkin', 'checkout'])

    # verify city ids
    city_ids = set(df.city_id.unique().astype(int))
    assert len(set(submission.city_id_1.unique()).difference(city_ids)) == 0
    assert len(set(submission.city_id_2.unique()).difference(city_ids)) == 0
    assert len(set(submission.city_id_3.unique()).difference(city_ids)) == 0
    assert len(set(submission.city_id_4.unique()).difference(city_ids)) == 0
    logging.info("Passed all sanity checks!")
```

```python colab={"base_uri": "https://localhost:8080/", "height": 644, "referenced_widgets": ["bd5f013f0d8f4a5dbbed3bee35fb026c", "d1f5a71f917042f69cc0fedfbd9237cd", "ba87625893c8417ea8074e98c10a489a", "edc7772376e647868fab2f0e7f559c54", "671e6bb54604409a9b0ae6607774633a", "87a936870f8843478321bcd96cb98c9c", "3f90575f1e594a34bff0fbf3a1804744", "ae0ed2e9ff9e4948817592b33e3f2f59", "b5860510ce0e4017841643bcf020b067", "7dc72bba45cc4619a1e94618daa02532", "0e88f652d702467a94e4b12161502132", "96cd15f06a6a41058eaa342d0897f308", "96a4849c9bb045ada32905baa47d32c9", "f246ae8f24474a5d83d8345e02aae694", "8a79b86bab794a2182ce37afe7b8eb94", "d10bc5b5c8cf4e4da776b912a4cb4260", "b54dee9d9aaf48f8ba4307c27fdf3157", "f1ce3196d41448d8bf46271e04fba7d1", "2ba9ed5697f449af810ae01c43b7a3b8", "d4ef1ea86a6c4600969a5510631f4772", "3004880b49f74960ba633a01ab1b4c35", "0021ba7201d347e09b6568862c64c7bd", "3bf6b552ce7645c6b69ee0a9a61b10bf", "bcb8702518fa4aedb86aea8f5139dec7", "ff6e8aa7364a4f299bd860efce6bd214", "0a4f36728ab74059aee9d1ebc263fc2d", "90ff547c175e4a8f936da7651cd21194", "a27e914a2314471f82370252b49b58da", "c3ead754cd5a49f69ecbed8cbed22402", "3ab1bce6c705443f9ce64860a2d3ecf3", "2487c7282782404898c10e8f12a2abb6", "d4e5ae82c1a34936ace03349423eac99", "4a88d58b2c41407f93a48de60671182a", "6ad668fe52544cb0a1f493df5aa498d9", "4a1e47d717054381878a93d2cbac9453", "0410eccb4b3c4898a96cfe58e8ff9b0a", "bff8902ef03f41b9a49409ef2bef1bd3", "bb2e50a511d545b68231466f365ea5dc", "256e808ecae84e089b76ea4005d87a04", "4b946e6e47f244c98e506ed6320995b6", "116a22d986b94666b9d63c7c0da270da", "5f0e01e76ad34b48af4c0c9d4050a786", "849644cb6a4341cd9c4c42d05123a59b", "5a4a65a2328e4502911c863043ebbb06", "9e9d836a9e2d480189aeecd6d52fdfd3", "c8d3feef68af423da22e5bf2ed4e1523", "bbc79301f77048ff87dd9e6504d5f4d3", "0d9890702aba4fc59d22b5cc4413d93b", "e9cbf6bbd87d492a8783cac7c2b3a148", "8ca6e58df9434c3b84d7c1d8afa7022d", "2ed6b7955cef43efbfdd594a98b1779f", "81d9f8351afd4f7491e8317296ba9322", "20da5c74e5e44c5fb0eba216e6c1fd8d", "c257170ad69240f9a2ce3b717391c7d1", "5e8f9963b1324f838bcbf09db65357a0", "83d70daba9f3408bbcffc44ca8c7bbbb", "1e98c404de6542ad85ab784d17765373", "992409e46fa94aff8058740c6a64a810", "bedc3a91b8c144e4bdcb0fbef877d2d2", "cf5971b21caa4b6baec201dcf6216eed", "3df24f839ac34db386dcc968883fab71", "c44cceb3cb944207b334877b8872f371", "cd3b371138d94d6996740b5b66117065", "3bf19ecfe32643e1859fa38d0a081710", "b2393264f83d480fb7f2bd09bdcbca68", "a93d06da02974b2a9f84594f6bf325f4", "567200e2fa3f4ffbb5cad40e545916f6", "4852e657ad014e22be2ed6549655bbb8", "a130abc6fd3d433daa5e314db8449f82", "94163d6093424d7d96a96f431d948260", "df5de35a223849a5a531c186e2601131", "baa0e25b78e543a697616d5edc2a7e26", "5ff6b9482d7b47c6af44ec46efce6321", "a7730b68e77d49c69bd8476b7b9984e3", "bd45ba78e06d4376ae0d18b7daf413b5", "af538c4e37924720b89e5952f9983974", "59583a662df04788a830c5ef32d5ea84", "3bf4194da3eb4da394ae076582852510", "f0b9659a8eda4a35811dcf5d8b879252", "156917a8062447dd86680c69a388d042", "b38a8d854b124e18aee92f21d670ac4c", "5fd6e4526d3f4af0b7bbae2ea4735bdd", "dcf30c10b4ba44d896a7251269893021", "f7b676044ba14fc0944cc3b882d654c0", "d6f57d40881542d395376d95a8dc96a7", "874ec2b445a94c8d95d4c7a6225714e7", "2ea88188316b451fbe3074e4c9cd1dcb", "3105c1304bef4c3b9c4f66f431a7d611"]} id="IBSm4NNn4rP0" outputId="a74977fc-aca5-4927-8a84-0bcd8f83d904"
def run_experiments(base_configuration: Dict,
                    experiments: List[Dict],
                    n_models: int,
                    dataset_batches: List[BatchType],
                    train_set: pd.DataFrame,
                    skip_checkpoint=False) -> None:
    """
    Given a base configuration in a dictionary, run experiments
    by overriding parameters of this base configuration with
    a list of overrides in `experiments`.
    """
    for model_overrides in tqdm(experiments):
        logging.info(model_overrides)
        model_configuration = dict(base_configuration, **model_overrides)
        train_model_for_folds(dataset_batches,
                              train_set,
                              model_configuration,
                              n_models=n_models,
                              skip_checkpoint=skip_checkpoint)


def get_base_configuration():
    """
    The base configuration describes our best model. Experiments
    change elements of this configuration to try to find an even
    better one.
    """
    return {
        'features_embedding': FEATURES_EMBEDDING,
        'hidden_size': int(EMBEDDING_SIZES['city_id'][1]),
        'output_size': int(EMBEDDING_SIZES['city_id'][0]),
        'embedding_sizes': EMBEDDING_SIZES,
        'n_layers': 2,
        'dropout': 0.3,
        'rnn_dropout': 0.1,
        'tie_embedding_and_projection': True,
        'model_type': ModelType.MANY_TO_MANY,
        'recurrent_type': RecurrentType.GRU,
        'weight_type': WeightType.UNWEIGHTED,
        'feature_projection_type': FeatureProjectionType.CONCATENATION,
        'num_folds': N_SPLITS,
        'batch_size': BATCH_SIZE
    }


def get_experiments() -> List:
    """
    An experiment is a dict that describes the parameters
    that will be overridden in the base configuration
    during an experiment.
    """
    params = ['model_type', 'weight_type', 'recurrent_type', 'tie_embedding_and_projection']
    return [
        dict(zip(params, p))
        for p in itertools.product(
            *map(list, [ModelType, WeightType, RecurrentType, [True, False]])
        )
    ]


def get_model_performance_data(test_set: pd.DataFrame,
                               dataset_encoder: DatasetEncoder,
                               model_hashes: List[str] = None) -> Tuple[pd.DataFrame, Dict]:
    """
    Get model performance data from all trained models.
    """
    booking_dataset_test, dataset_loader_test = get_dataset_and_dataloader(
        df=test_set,
        features=FEATURES_EMBEDDING
    )
    ground_truth_test = get_ground_truth_from_dataset(
        df=test_set,
        booking_dataset=booking_dataset_test,
        dataset_encoder=dataset_encoder
    )

    trained_models = get_trained_models()

    if model_hashes:
        trained_models = {h: trained_models[h] for h in model_hashes}

    df_rows = []
    accuracy_at_4_by_length = {}
    for model_hash, model_parameters in trained_models.items():
        try:
            ckpt_list = get_model_ckpt_paths(model_hash=model_hash,
                                             checkpoint_type='accuracy_at_k')
        except FileNotFoundError:
            continue

        d = {
            'single': ckpt_list[:1],
            'ensemble': ckpt_list
        }

        for model_type, ckpt_list in d.items():
            if model_type == 'ensemble' and len(ckpt_list) == 1:
                continue

            model = BookingNet(**model_parameters).to(DEVICE)
            predictions = get_submission(booking_dataset_test,
                                         dataset_loader_test,
                                         model,
                                         ckpt_list,
                                         dataset_encoder)
            accuracy = accuracy_at_k(predictions, ground_truth_test)
            model_parameters['num_models'] = len(ckpt_list)
            model_parameters['accuracy@1'] = accuracy['accuracy@1']
            model_parameters['accuracy@4'] = accuracy['accuracy@4']
            model_parameters['accuracy@10'] = accuracy['accuracy@10']
            model_parameters['hash'] = model_hash
            df = pd.DataFrame.from_dict(model_parameters, orient='index').T
            df_rows.append(pd.concat(
                [df, pd.DataFrame.from_dict(accuracy['accuracy@4_by_pos'], orient='index').T]
                , axis=1))
            accuracy_at_4_by_length[(model_hash, model_type)] = accuracy['accuracy@4_by_pos']
    return pd.concat(df_rows), accuracy_at_4_by_length


def filter_results_table(results: pd.DataFrame) -> pd.DataFrame:
    """
    Filter results table to get only attributes that change between models.
    """
    columns = ['model_type', 'recurrent_type', 'tie_embedding_and_projection',
               'weight_type', 'accuracy@1', 'accuracy@4', 'accuracy@10', 'hash']
    selected_columns = [col for col in results.columns.values
                        if results[col].apply(str).nunique() > 1
                        or col in columns]
    filtered_results = (results[selected_columns]
                        .sort_values("accuracy@4", ascending=False))

    decode = {
        'model_type': ModelType,
        'weight_type': WeightType,
        'recurrent_type': RecurrentType
    }

    for key, enum_type in decode.items():
        filtered_results[key] = (filtered_results[key]
                                 .apply(enum_type)
                                 .apply(lambda s: str(s).split('.')[1]))

    df_table = filtered_results[columns].sort_values(
        ["model_type", "recurrent_type", "tie_embedding_and_projection", "accuracy@4"],
        ascending=[True, True, False, False])
    return df_table


# build and encode dataset
dataset = build_dataset(reserved_obs=30000)
de = DatasetEncoder(FEATURES_TO_ENCODE)
de.fit_transform(dataset)
set_future_features(dataset)

submission_set = get_submission_set_from_dataset(dataset)

# keep only observations before the last visit
dataset = dataset[~dataset.next_city_id.isna()]

# split training and test set from dataset
train_set = get_training_set_from_dataset(dataset)
test_set = get_test_set_from_dataset(dataset)

logging.info(f"Training set: {train_set.shape}")
logging.info(f"Test set: {test_set.shape}")
logging.info(f"Dataset: {dataset.shape}")

logging.info(get_distribution_by_pos(dataset=dataset,
                                      train_set=train_set[train_set.train == 1],
                                      test_set=test_set,
                                      submission=submission_set).head(10))

# pre-load all batches to GPU
_, dataset_loader = get_dataset_and_dataloader(
    train_set,
    features=FEATURES_EMBEDDING + ['next_city_id'],
    batch_size=BATCH_SIZE
)
dataset_batches_cuda = batches_to_device(dataset_loader)

print_gpu_usage(0)

# run experiments from base configuration
base_configuration = get_base_configuration()
experiments = get_experiments()
run_experiments(base_configuration=base_configuration,
                experiments=experiments,
                n_models=1,
                dataset_batches=dataset_batches_cuda,
                train_set=train_set)

# get and save results table
results, _ = get_model_performance_data(test_set, de)

filter_results_table(results).to_csv(
    get_path(
        filename='experiments',
        format='csv'),
    index=False
)
```

```python id="YWro3lXzFnCX" executionInfo={"status": "aborted", "timestamp": 1623518887595, "user_tz": -330, "elapsed": 11, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def get_plot_from_accuracy(**kwargs) -> None:
    """
    Accuracy plot by position.
    """
    df_list = []
    for key, accuracy_dict in kwargs.items():
        df = pd.DataFrame.from_dict(accuracy_dict, orient='index', columns=['accuracy']).head(8)
        df['type'] = key
        df_list.append(df)
    g = sns.catplot(
        data=pd.concat(df_list).reset_index(), kind="bar",
        x="index", y="accuracy", hue="type",
        palette="bone", height=6, legend_out=False
    )
    g.set(ylim=(0.4, 0.7))
    g.set_axis_labels("Sequence length", "accuracy@4")
    g.savefig("accuracy_by_position.pdf")


def get_plot_from_distribution_by_pos(df: pd.DataFrame):
    """
    Plot distribution by position from dataframe.
    """
    df_melt = pd.melt(df,
                      value_vars=['train_set', 'submission'],
                      var_name='dataset_type',
                      value_name='sequence_length',
                      ignore_index=False)

    sns.set_style('white')
    sns.set_context('paper', font_scale=2)
    sns.set_palette(['#000000', '#ABABAB'])
    sns.set_style('ticks', {'axes.edgecolor': '0',
                            'xtick.color': '0',
                            'ytick.color': '0'})

    g = sns.catplot(
        data=df_melt.reset_index(), kind="bar",
        x="index", y="sequence_length", hue="dataset_type",
        ci="sd", height=6, legend_out=False,
    )
    g.set_axis_labels("Sequence length", "Proportion")
    new_labels = ['Training set', 'Submission set']
    for t, l in zip(g._legend.texts, new_labels):
        t.set_text(l)

    g._legend.set_title('')
    g.savefig("sequence_length_distribution.pdf")
```

```python id="IUgGokMXCFyv" executionInfo={"status": "aborted", "timestamp": 1623518887598, "user_tz": -330, "elapsed": 13, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
get_plot_from_accuracy(single=acc_dict[(model_hash, 'single')])
```

```python id="sgY1umOHFrlI" executionInfo={"status": "aborted", "timestamp": 1623516710168, "user_tz": -330, "elapsed": 34, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
# build submission from single model
get_final_submission(submission_set, model_hash, de)
```

```python id="H_K4sAgoFrgB" executionInfo={"status": "aborted", "timestamp": 1623516710171, "user_tz": -330, "elapsed": 37, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}

```
