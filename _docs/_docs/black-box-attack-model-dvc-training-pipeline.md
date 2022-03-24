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

```python id="cwjDbsFozbUv" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1631872911033, "user_tz": -330, "elapsed": 37797, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="222e0e89-63f5-4ceb-dc92-778019221070"
!pip install -U -q dvc dvc[gdrive]
# !dvc get https://github.com/sparsh-ai/reco-data ml1m/v0/ratings.dat
```

<!-- #region id="q-KQcUGsZ6cD" -->
## Dataset
<!-- #endregion -->

```python id="d-do78dkFnCl"
RAW_DATASET_ROOT_FOLDER = '/content/data/bronze'
PREP_DATASET_ROOT_FOLDER = '/content/data/silver'
FNL_DATASET_ROOT_FOLDER = '/content/data/gold'
```

```python id="DHp3CLL6FiKd"
import pickle
import shutil
import tempfile
import os
from datetime import date
from pathlib import Path
import gzip
from abc import *

import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
```

```python id="IL4UTwazFGMD"
class AbstractDataset(metaclass=ABCMeta):
    def __init__(self, args):
        self.args = args
        self.min_rating = args.min_rating
        self.min_uc = args.min_uc
        self.min_sc = args.min_sc
        self.split = args.split

        assert self.min_uc >= 2, 'Need at least 2 ratings per user for validation and test'

    @classmethod
    @abstractmethod
    def code(cls):
        pass

    @classmethod
    def raw_code(cls):
        return cls.code()

    @classmethod
    def all_raw_file_names(cls):
        return []

    @classmethod
    @abstractmethod
    def url(cls):
        pass

    @abstractmethod
    def preprocess(self):
        pass

    @abstractmethod
    def load_ratings_df(self):
        pass

    @abstractmethod
    def maybe_download_raw_dataset(self):
        pass

    def load_dataset(self):
        self.preprocess()
        dataset_path = self._get_preprocessed_dataset_path()
        dataset = pickle.load(dataset_path.open('rb'))
        return dataset

    def filter_triplets(self, df):
        print('Filtering triplets')
        if self.min_sc > 0:
            item_sizes = df.groupby('sid').size()
            good_items = item_sizes.index[item_sizes >= self.min_sc]
            df = df[df['sid'].isin(good_items)]

        if self.min_uc > 0:
            user_sizes = df.groupby('uid').size()
            good_users = user_sizes.index[user_sizes >= self.min_uc]
            df = df[df['uid'].isin(good_users)]
        return df

    def densify_index(self, df):
        print('Densifying index')
        umap = {u: i for i, u in enumerate(set(df['uid']), start=1)}
        smap = {s: i for i, s in enumerate(set(df['sid']), start=1)}
        df['uid'] = df['uid'].map(umap)
        df['sid'] = df['sid'].map(smap)
        return df, umap, smap

    def split_df(self, df, user_count):
        if self.args.split == 'leave_one_out':
            print('Splitting')
            user_group = df.groupby('uid')
            user2items = user_group.progress_apply(
                lambda d: list(d.sort_values(by=['timestamp', 'sid'])['sid']))
            train, val, test = {}, {}, {}
            for i in range(user_count):
                user = i + 1
                items = user2items[user]
                train[user], val[user], test[user] = items[:-2], items[-2:-1], items[-1:]
            return train, val, test
        else:
            raise NotImplementedError

    def _get_rawdata_root_path(self):
        return Path(RAW_DATASET_ROOT_FOLDER)

    def _get_rawdata_folder_path(self):
        root = self._get_rawdata_root_path()
        return root.joinpath(self.raw_code())

    def _get_preprocessed_root_path(self):
        root = Path(PREP_DATASET_ROOT_FOLDER)
        return root.joinpath(self.raw_code())

    def _get_preprocessed_folder_path(self):
        preprocessed_root = self._get_preprocessed_root_path()
        folder_name = '{}_min_rating{}-min_uc{}-min_sc{}-split{}' \
            .format(self.code(), self.min_rating, self.min_uc, self.min_sc, self.split)
        return preprocessed_root.joinpath(folder_name)

    def _get_preprocessed_dataset_path(self):
        folder = self._get_preprocessed_folder_path()
        return folder.joinpath('dataset.pkl')
```

```python id="eBEM6mZcGBpX"
class ML1MDataset(AbstractDataset):
    @classmethod
    def code(cls):
        return 'ml-1m'

    @classmethod
    def url(cls):
        return {'path':'ml1m/v0',
                'repo':'https://github.com/sparsh-ai/reco-data'}

    @classmethod
    def all_raw_file_names(cls):
        return ['ratings.dat']

    def maybe_download_raw_dataset(self):
        folder_path = self._get_rawdata_folder_path()
        if not folder_path.is_dir():
            folder_path.mkdir(parents=True)
        if all(folder_path.joinpath(filename).is_file() for filename in self.all_raw_file_names()):
            print('Raw data already exists. Skip downloading')
            return
        
        print("Raw file doesn't exist. Downloading...")
        for filename in self.all_raw_file_names():
            with open(os.path.join(folder_path,filename), "wb") as f:
                with dvc.api.open(
                    path=self.url()['path']+'/'+filename,
                    repo=self.url()['repo'],
                    mode='rb') as scan:
                    f.write(scan.read())
        print()

    def preprocess(self):
        dataset_path = self._get_preprocessed_dataset_path()
        if dataset_path.is_file():
            print('Already preprocessed. Skip preprocessing')
            return
        if not dataset_path.parent.is_dir():
            dataset_path.parent.mkdir(parents=True)
        self.maybe_download_raw_dataset()
        df = self.load_ratings_df()
        df = self.filter_triplets(df)
        df, umap, smap = self.densify_index(df)
        train, val, test = self.split_df(df, len(umap))
        dataset = {'train': train,
                   'val': val,
                   'test': test,
                   'umap': umap,
                   'smap': smap}
        with dataset_path.open('wb') as f:
            pickle.dump(dataset, f)

    def load_ratings_df(self):
        folder_path = self._get_rawdata_folder_path()
        file_path = folder_path.joinpath('ratings.dat')
        df = pd.read_csv(file_path, sep='::', header=None)
        df.columns = ['uid', 'sid', 'rating', 'timestamp']
        return df
```

```python id="Su5bfrmJguLA"
DATASETS = {
    ML1MDataset.code(): ML1MDataset,
    # ML20MDataset.code(): ML20MDataset,
    # SteamDataset.code(): SteamDataset,
    # GamesDataset.code(): GamesDataset,
    # BeautyDataset.code(): BeautyDataset,
    # BeautyDenseDataset.code(): BeautyDenseDataset,
    # YooChooseDataset.code(): YooChooseDataset
}


def dataset_factory(args):
    dataset = DATASETS[args.dataset_code]
    return dataset(args)
```

<!-- #region id="QZRvVywhZoYk" -->
## Negative Sampling
<!-- #endregion -->

```python id="CeyMnI3LZqfq"
from abc import *
from pathlib import Path
import pickle


class AbstractNegativeSampler(metaclass=ABCMeta):
    def __init__(self, train, val, test, user_count, item_count, sample_size, seed, flag, save_folder):
        self.train = train
        self.val = val
        self.test = test
        self.user_count = user_count
        self.item_count = item_count
        self.sample_size = sample_size
        self.seed = seed
        self.flag = flag
        self.save_folder = save_folder

    @classmethod
    @abstractmethod
    def code(cls):
        pass

    @abstractmethod
    def generate_negative_samples(self):
        pass

    def get_negative_samples(self):
        savefile_path = self._get_save_path()
        if savefile_path.is_file():
            print('Negatives samples exist. Loading.')
            seen_samples, negative_samples = pickle.load(savefile_path.open('rb'))
            return seen_samples, negative_samples
        print("Negative samples don't exist. Generating.")
        seen_samples, negative_samples = self.generate_negative_samples()
        with savefile_path.open('wb') as f:
            pickle.dump([seen_samples, negative_samples], f)
        return seen_samples, negative_samples

    def _get_save_path(self):
        folder = Path(self.save_folder)
        filename = '{}-sample_size{}-seed{}-{}.pkl'.format(
            self.code(), self.sample_size, self.seed, self.flag)
        return folder.joinpath(filename)
```

```python id="m8VYfuAuZtXz"
from tqdm import trange
import numpy as np


class RandomNegativeSampler(AbstractNegativeSampler):
    @classmethod
    def code(cls):
        return 'random'

    def generate_negative_samples(self):
        assert self.seed is not None, 'Specify seed for random sampling'
        np.random.seed(self.seed)
        num_samples = 2 * self.user_count * self.sample_size
        all_samples = np.random.choice(self.item_count, num_samples) + 1

        seen_samples = {}
        negative_samples = {}
        print('Sampling negative items randomly...')
        j = 0
        for i in trange(self.user_count):
            user = i + 1
            seen = set(self.train[user])
            seen.update(self.val[user])
            seen.update(self.test[user])
            seen_samples[user] = seen

            samples = []
            while len(samples) < self.sample_size:
                item = all_samples[j % num_samples]
                j += 1
                if item in seen or item in samples:
                    continue
                samples.append(item)
            negative_samples[user] = samples

        return seen_samples, negative_samples
```

```python id="SfOLiCHfZtpV"
from tqdm import trange
from collections import Counter
import numpy as np


class PopularNegativeSampler(AbstractNegativeSampler):
    @classmethod
    def code(cls):
        return 'popular'

    def generate_negative_samples(self):
        assert self.seed is not None, 'Specify seed for random sampling'
        np.random.seed(self.seed)
        popularity = self.items_by_popularity()
        items = list(popularity.keys())
        total = 0
        for i in range(len(items)):
            total += popularity[items[i]]
        for i in range(len(items)):
            popularity[items[i]] /= total
        probs = list(popularity.values())
        num_samples = 2 * self.user_count * self.sample_size
        all_samples = np.random.choice(items, num_samples, p=probs)

        seen_samples = {}
        negative_samples = {}
        print('Sampling negative items by popularity...')
        j = 0
        for i in trange(self.user_count):
            user = i + 1
            seen = set(self.train[user])
            seen.update(self.val[user])
            seen.update(self.test[user])
            seen_samples[user] = seen

            samples = []
            while len(samples) < self.sample_size:
                item = all_samples[j % num_samples]
                j += 1
                if item in seen or item in samples:
                    continue
                samples.append(item)
            negative_samples[user] = samples

        return seen_samples, negative_samples

    def items_by_popularity(self):
        popularity = Counter()
        self.users = sorted(self.train.keys())
        for user in self.users:
            popularity.update(self.train[user])
            popularity.update(self.val[user])
            popularity.update(self.test[user])

        popularity = dict(popularity)
        popularity = {k: v for k, v in sorted(popularity.items(), key=lambda item: item[1], reverse=True)}
        return popularity
```

```python id="tPQmZFW3d-v5"
NEGATIVE_SAMPLERS = {
    PopularNegativeSampler.code(): PopularNegativeSampler,
    RandomNegativeSampler.code(): RandomNegativeSampler,
}


def negative_sampler_factory(code, train, val, test, user_count, item_count, sample_size, seed, flag, save_folder):
    negative_sampler = NEGATIVE_SAMPLERS[code]
    return negative_sampler(train, val, test, user_count, item_count, sample_size, seed, flag, save_folder)
```

<!-- #region id="URvSeRPHZ9uK" -->
## Dataloader
<!-- #endregion -->

```python id="yipNZfdnZ-pd"
from abc import *
import random


class AbstractDataloader(metaclass=ABCMeta):
    def __init__(self, args, dataset):
        self.args = args
        self.rng = random.Random()
        self.save_folder = dataset._get_preprocessed_folder_path()
        dataset = dataset.load_dataset()
        self.train = dataset['train']
        self.val = dataset['val']
        self.test = dataset['test']
        self.umap = dataset['umap']
        self.smap = dataset['smap']
        self.user_count = len(self.umap)
        self.item_count = len(self.smap)

    @classmethod
    @abstractmethod
    def code(cls):
        pass

    @abstractmethod
    def get_pytorch_dataloaders(self):
        pass
```

```python id="cGZMtlFZaD2v"
import torch
import random
import torch.utils.data as data_utils


class RNNDataloader():
    def __init__(self, args, dataset):
        self.args = args
        self.rng = random.Random()
        self.save_folder = dataset._get_preprocessed_folder_path()
        dataset = dataset.load_dataset()
        self.train = dataset['train']
        self.val = dataset['val']
        self.test = dataset['test']
        self.umap = dataset['umap']
        self.smap = dataset['smap']
        self.user_count = len(self.umap)
        self.item_count = len(self.smap)

        args.num_items = len(self.smap)
        self.max_len = args.bert_max_len

        val_negative_sampler = negative_sampler_factory(args.test_negative_sampler_code,
                                                        self.train, self.val, self.test,
                                                        self.user_count, self.item_count,
                                                        args.test_negative_sample_size,
                                                        args.test_negative_sampling_seed,
                                                        'val', self.save_folder)
        test_negative_sampler = negative_sampler_factory(args.test_negative_sampler_code,
                                                         self.train, self.val, self.test,
                                                         self.user_count, self.item_count,
                                                         args.test_negative_sample_size,
                                                         args.test_negative_sampling_seed,
                                                         'test', self.save_folder)

        self.seen_samples, self.val_negative_samples = val_negative_sampler.get_negative_samples()
        self.seen_samples, self.test_negative_samples = test_negative_sampler.get_negative_samples()

    @classmethod
    def code(cls):
        return 'rnn'

    def get_pytorch_dataloaders(self):
        train_loader = self._get_train_loader()
        val_loader = self._get_val_loader()
        test_loader = self._get_test_loader()
        return train_loader, val_loader, test_loader

    def _get_train_loader(self):
        dataset = self._get_train_dataset()
        dataloader = data_utils.DataLoader(dataset, batch_size=self.args.train_batch_size,
                                           shuffle=True, pin_memory=True)
        return dataloader

    def _get_train_dataset(self):
        dataset = RNNTrainDataset(
            self.train, self.max_len)
        return dataset

    def _get_val_loader(self):
        return self._get_eval_loader(mode='val')

    def _get_test_loader(self):
        return self._get_eval_loader(mode='test')

    def _get_eval_loader(self, mode):
        batch_size = self.args.val_batch_size if mode == 'val' else self.args.test_batch_size
        dataset = self._get_eval_dataset(mode)
        dataloader = data_utils.DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
        return dataloader

    def _get_eval_dataset(self, mode):
        if mode == 'val':
            dataset = RNNValidDataset(self.train, self.val, self.max_len, self.val_negative_samples)
        elif mode == 'test':
            dataset = RNNTestDataset(self.train, self.val, self.test, self.max_len, self.test_negative_samples)
        return dataset


class RNNTrainDataset(data_utils.Dataset):
    def __init__(self, u2seq, max_len):
        # self.u2seq = u2seq
        # self.users = sorted(self.u2seq.keys())
        self.max_len = max_len
        self.all_seqs = []
        self.all_labels = []
        for u in sorted(u2seq.keys()):
            seq = u2seq[u]
            for i in range(1, len(seq)):
                self.all_seqs += [seq[:-i]]
                self.all_labels += [seq[-i]]

        assert len(self.all_seqs) == len(self.all_labels)

    def __len__(self):
        return len(self.all_seqs)

    def __getitem__(self, index):
        tokens = self.all_seqs[index][-self.max_len:]
        length = len(tokens)
        tokens = tokens + [0] * (self.max_len - length)
        
        return torch.LongTensor(tokens), torch.LongTensor([length]), torch.LongTensor([self.all_labels[index]])


class RNNValidDataset(data_utils.Dataset):
    def __init__(self, u2seq, u2answer, max_len, negative_samples, valid_users=None):
        self.u2seq = u2seq  # train
        if not valid_users:
            self.users = sorted(self.u2seq.keys())
        else:
            self.users = valid_users
        self.users = sorted(self.u2seq.keys())
        self.u2answer = u2answer
        self.max_len = max_len
        self.negative_samples = negative_samples
        
    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        tokens = self.u2seq[user][-self.max_len:]
        length = len(tokens)
        tokens = tokens + [0] * (self.max_len - length)

        answer = self.u2answer[user]
        negs = self.negative_samples[user]
        candidates = answer + negs
        labels = [1] * len(answer) + [0] * len(negs)
        
        return torch.LongTensor(tokens), torch.LongTensor([length]), torch.LongTensor(candidates), torch.LongTensor(labels)


class RNNTestDataset(data_utils.Dataset):
    def __init__(self, u2seq, u2val, u2answer, max_len, negative_samples, test_users=None):
        self.u2seq = u2seq  # train
        self.u2val = u2val  # val
        if not test_users:
            self.users = sorted(self.u2seq.keys())
        else:
            self.users = test_users
        self.users = sorted(self.u2seq.keys())
        self.u2answer = u2answer  # test
        self.max_len = max_len
        self.negative_samples = negative_samples

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        tokens = (self.u2seq[user] + self.u2val[user])[-self.max_len:]  # append validation item after train seq
        length = len(tokens)
        tokens = tokens + [0] * (self.max_len - length)
        answer = self.u2answer[user]
        negs = self.negative_samples[user]
        candidates = answer + negs
        labels = [1] * len(answer) + [0] * len(negs)

        return torch.LongTensor(tokens), torch.LongTensor([length]), torch.LongTensor(candidates), torch.LongTensor(labels)
```

```python id="bGaHGNi_sho2"
import torch
import random
import torch.utils.data as data_utils


class BERTDataloader():
    def __init__(self, args, dataset):
        self.args = args
        self.rng = random.Random()
        self.save_folder = dataset._get_preprocessed_folder_path()
        dataset = dataset.load_dataset()
        self.train = dataset['train']
        self.val = dataset['val']
        self.test = dataset['test']
        self.umap = dataset['umap']
        self.smap = dataset['smap']
        self.user_count = len(self.umap)
        self.item_count = len(self.smap)

        args.num_items = self.item_count
        self.max_len = args.bert_max_len
        self.mask_prob = args.bert_mask_prob
        self.max_predictions = args.bert_max_predictions
        self.sliding_size = args.sliding_window_size
        self.CLOZE_MASK_TOKEN = self.item_count + 1

        val_negative_sampler = negative_sampler_factory(args.test_negative_sampler_code,
                                                        self.train, self.val, self.test,
                                                        self.user_count, self.item_count,
                                                        args.test_negative_sample_size,
                                                        args.test_negative_sampling_seed,
                                                        'val', self.save_folder)
        test_negative_sampler = negative_sampler_factory(args.test_negative_sampler_code,
                                                         self.train, self.val, self.test,
                                                         self.user_count, self.item_count,
                                                         args.test_negative_sample_size,
                                                         args.test_negative_sampling_seed,
                                                         'test', self.save_folder)

        self.seen_samples, self.val_negative_samples = val_negative_sampler.get_negative_samples()
        self.seen_samples, self.test_negative_samples = test_negative_sampler.get_negative_samples()

    @classmethod
    def code(cls):
        return 'bert'

    def get_pytorch_dataloaders(self):
        train_loader = self._get_train_loader()
        val_loader = self._get_val_loader()
        test_loader = self._get_test_loader()
        return train_loader, val_loader, test_loader

    def _get_train_loader(self):
        dataset = self._get_train_dataset()
        dataloader = data_utils.DataLoader(dataset, batch_size=self.args.train_batch_size,
                                           shuffle=True, pin_memory=True)
        return dataloader

    def _get_train_dataset(self):
        dataset = BERTTrainDataset(
            self.train, self.max_len, self.mask_prob, self.max_predictions, self.sliding_size, self.CLOZE_MASK_TOKEN, self.item_count, self.rng)
        return dataset

    def _get_val_loader(self):
        return self._get_eval_loader(mode='val')

    def _get_test_loader(self):
        return self._get_eval_loader(mode='test')

    def _get_eval_loader(self, mode):
        batch_size = self.args.val_batch_size if mode == 'val' else self.args.test_batch_size
        dataset = self._get_eval_dataset(mode)
        dataloader = data_utils.DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
        return dataloader

    def _get_eval_dataset(self, mode):
        if mode == 'val':
            dataset = BERTValidDataset(self.train, self.val, self.max_len, self.CLOZE_MASK_TOKEN, self.val_negative_samples)
        elif mode == 'test':
            dataset = BERTTestDataset(self.train, self.val, self.test, self.max_len, self.CLOZE_MASK_TOKEN, self.test_negative_samples)
        return dataset


class BERTTrainDataset(data_utils.Dataset):
    def __init__(self, u2seq, max_len, mask_prob, max_predictions, sliding_size, mask_token, num_items, rng):
        # self.u2seq = u2seq
        # self.users = sorted(self.u2seq.keys())
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.max_predictions = max_predictions
        self.sliding_step = int(sliding_size * max_len)
        self.mask_token = mask_token
        self.num_items = num_items
        self.rng = rng
        
        assert self.sliding_step > 0
        self.all_seqs = []
        for u in sorted(u2seq.keys()):
            seq = u2seq[u]
            if len(seq) < self.max_len + self.sliding_step:
                self.all_seqs.append(seq)
            else:
                start_idx = range(len(seq) - max_len, -1, -self.sliding_step)
                self.all_seqs = self.all_seqs + [seq[i:i + max_len] for i in start_idx]

    def __len__(self):
        return len(self.all_seqs)
        # return len(self.users)

    def __getitem__(self, index):
        # user = self.users[index]
        # seq = self._getseq(user)
        seq = self.all_seqs[index]

        tokens = []
        labels = []
        covered_items = set()
        for i in range(len(seq)):
            s = seq[i]
            if (len(covered_items) >= self.max_predictions) or (s in covered_items):
                tokens.append(s)
                labels.append(0)
                continue
            
            temp_mask_prob = self.mask_prob
            if i == (len(seq) - 1):
                temp_mask_prob += 0.1 * (1 - self.mask_prob)

            prob = self.rng.random()
            if prob < temp_mask_prob:
                covered_items.add(s)
                prob /= temp_mask_prob
                if prob < 0.8:
                    tokens.append(self.mask_token)
                elif prob < 0.9:
                    tokens.append(self.rng.randint(1, self.num_items))
                else:
                    tokens.append(s)

                labels.append(s)
            else:
                tokens.append(s)
                labels.append(0)

        tokens = tokens[-self.max_len:]
        labels = labels[-self.max_len:]

        mask_len = self.max_len - len(tokens)

        tokens = [0] * mask_len + tokens
        labels = [0] * mask_len + labels

        return torch.LongTensor(tokens), torch.LongTensor(labels)

    def _getseq(self, user):
        return self.u2seq[user]


class BERTValidDataset(data_utils.Dataset):
    def __init__(self, u2seq, u2answer, max_len, mask_token, negative_samples, valid_users=None):
        self.u2seq = u2seq  # train
        if not valid_users:
            self.users = sorted(self.u2seq.keys())
        else:
            self.users = valid_users
        self.u2answer = u2answer
        self.max_len = max_len
        self.mask_token = mask_token
        self.negative_samples = negative_samples

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self.u2seq[user]
        answer = self.u2answer[user]
        negs = self.negative_samples[user]

        candidates = answer + negs
        labels = [1] * len(answer) + [0] * len(negs)

        seq = seq + [self.mask_token]
        seq = seq[-self.max_len:]
        padding_len = self.max_len - len(seq)
        seq = [0] * padding_len + seq

        return torch.LongTensor(seq), torch.LongTensor(candidates), torch.LongTensor(labels)


class BERTTestDataset(data_utils.Dataset):
    def __init__(self, u2seq, u2val, u2answer, max_len, mask_token, negative_samples, test_users=None):
        self.u2seq = u2seq  # train
        self.u2val = u2val  # val
        if not test_users:
            self.users = sorted(self.u2seq.keys())
        else:
            self.users = test_users
        self.users = sorted(self.u2seq.keys())
        self.u2answer = u2answer  # test
        self.max_len = max_len
        self.mask_token = mask_token
        self.negative_samples = negative_samples

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self.u2seq[user] + self.u2val[user]  # append validation item after train seq
        answer = self.u2answer[user]
        negs = self.negative_samples[user]

        candidates = answer + negs
        labels = [1] * len(answer) + [0] * len(negs)

        seq = seq + [self.mask_token]
        seq = seq[-self.max_len:]
        padding_len = self.max_len - len(seq)
        seq = [0] * padding_len + seq

        return torch.LongTensor(seq), torch.LongTensor(candidates), torch.LongTensor(labels)
```

```python id="7xSS1QfzeHei"
def dataloader_factory(args):
    dataset = dataset_factory(args)
    if args.model_code == 'bert':
        dataloader = BERTDataloader(args, dataset)
    elif args.model_code == 'sas':
        dataloader = SASDataloader(args, dataset)
    else:
        dataloader = RNNDataloader(args, dataset)
    train, val, test = dataloader.get_pytorch_dataloaders()
    return train, val, test
```

<!-- #region id="N2PIoEWWgA8X" -->
## Args
<!-- #endregion -->

```python id="_fbBKtg6iOF7"
import numpy as np
import random
import torch
import argparse
```

```python id="dUiLMT_VjAhO"
def set_template(args):
    args.min_uc = 5
    args.min_sc = 5
    args.split = 'leave_one_out'
    dataset_code = {'1': 'ml-1m', '20': 'ml-20m', 'b': 'beauty', 'bd': 'beauty_dense' , 'g': 'games', 's': 'steam', 'y': 'yoochoose'}
    args.dataset_code = dataset_code[input('Input 1 / 20 for movielens, b for beauty, bd for dense beauty, g for games, s for steam and y for yoochoose: ')]
    if args.dataset_code == 'ml-1m':
        args.sliding_window_size = 0.5
        args.bert_hidden_units = 64
        args.bert_dropout = 0.1
        args.bert_attn_dropout = 0.1
        args.bert_max_len = 200
        args.bert_mask_prob = 0.2
        args.bert_max_predictions = 40
    elif args.dataset_code == 'ml-20m':
        args.sliding_window_size = 0.5
        args.bert_hidden_units = 64
        args.bert_dropout = 0.1
        args.bert_attn_dropout = 0.1
        args.bert_max_len = 200
        args.bert_mask_prob = 0.2
        args.bert_max_predictions = 20
    elif args.dataset_code in ['beauty', 'beauty_dense']:
        args.sliding_window_size = 0.5
        args.bert_hidden_units = 64
        args.bert_dropout = 0.5
        args.bert_attn_dropout = 0.2
        args.bert_max_len = 50
        args.bert_mask_prob = 0.6
        args.bert_max_predictions = 30
    elif args.dataset_code == 'games':
        args.sliding_window_size = 0.5
        args.bert_hidden_units = 64
        args.bert_dropout = 0.5
        args.bert_attn_dropout = 0.5
        args.bert_max_len = 50
        args.bert_mask_prob = 0.5
        args.bert_max_predictions = 25
    elif args.dataset_code == 'steam':
        args.sliding_window_size = 0.5
        args.bert_hidden_units = 64
        args.bert_dropout = 0.2
        args.bert_attn_dropout = 0.2
        args.bert_max_len = 50
        args.bert_mask_prob = 0.4
        args.bert_max_predictions = 20
    elif args.dataset_code == 'yoochoose':
        args.sliding_window_size = 0.5
        args.bert_hidden_units = 256
        args.bert_dropout = 0.2
        args.bert_attn_dropout = 0.2
        args.bert_max_len = 50
        args.bert_mask_prob = 0.4
        args.bert_max_predictions = 20

    batch = 128
    args.train_batch_size = batch
    args.val_batch_size = batch
    args.test_batch_size = batch
    args.train_negative_sampler_code = 'random'
    args.train_negative_sample_size = 0
    args.train_negative_sampling_seed = 0
    args.test_negative_sampler_code = 'random'
    args.test_negative_sample_size = 100
    args.test_negative_sampling_seed = 98765

    model_codes = {'b': 'bert', 's':'sas', 'n':'narm'}
    args.model_code = model_codes[input('Input model code, b for BERT, s for SASRec and n for NARM: ')]

    if torch.cuda.is_available():
        args.device = 'cuda:' + input('Input GPU ID: ')
    else:
        args.device = 'cpu'
    args.optimizer = 'AdamW'
    args.lr = 0.001
    args.weight_decay = 0.01
    args.enable_lr_schedule = True
    args.decay_step = 10000
    args.gamma = 1.
    args.enable_lr_warmup = False
    args.warmup_steps = 100
    args.num_epochs = 1000

    args.metric_ks = [1, 5, 10]
    args.best_metric = 'NDCG@10'
    args.model_init_seed = 98765
    args.bert_num_blocks = 2
    args.bert_num_heads = 2
    args.bert_head_size = None
```

```python colab={"base_uri": "https://localhost:8080/"} id="1FxRPzNBgChc" executionInfo={"status": "ok", "timestamp": 1631873778329, "user_tz": -330, "elapsed": 463, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="3801d542-e661-46f5-cdbb-9ff9c2d1d1c9"
parser = argparse.ArgumentParser()

################
# Dataset
################
parser.add_argument('--dataset_code', type=str, default='ml-1m', choices=DATASETS.keys())
parser.add_argument('--min_rating', type=int, default=0)
parser.add_argument('--min_uc', type=int, default=5)
parser.add_argument('--min_sc', type=int, default=5)
parser.add_argument('--split', type=str, default='leave_one_out')
parser.add_argument('--dataset_split_seed', type=int, default=0)

################
# Dataloader
################
parser.add_argument('--dataloader_random_seed', type=float, default=0)
parser.add_argument('--train_batch_size', type=int, default=64)
parser.add_argument('--val_batch_size', type=int, default=64)
parser.add_argument('--test_batch_size', type=int, default=64)
parser.add_argument('--sliding_window_size', type=float, default=0.5)

################
# NegativeSampler
################
parser.add_argument('--train_negative_sampler_code', type=str, default='random', choices=['popular', 'random'])
parser.add_argument('--train_negative_sample_size', type=int, default=0)
parser.add_argument('--train_negative_sampling_seed', type=int, default=0)
parser.add_argument('--test_negative_sampler_code', type=str, default='random', choices=['popular', 'random'])
parser.add_argument('--test_negative_sample_size', type=int, default=100)
parser.add_argument('--test_negative_sampling_seed', type=int, default=0)

################
# Trainer
################
# device #
parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
parser.add_argument('--num_gpu', type=int, default=1)
# optimizer & lr#
parser.add_argument('--optimizer', type=str, default='AdamW', choices=['AdamW', 'Adam', 'SGD'])
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--adam_epsilon', type=float, default=1e-9)
parser.add_argument('--momentum', type=float, default=None)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--enable_lr_schedule', type=bool, default=True)
parser.add_argument('--decay_step', type=int, default=100)
parser.add_argument('--gamma', type=float, default=1)
parser.add_argument('--enable_lr_warmup', type=bool, default=True)
parser.add_argument('--warmup_steps', type=int, default=100)
# epochs #
parser.add_argument('--num_epochs', type=int, default=100)
# logger #
parser.add_argument('--log_period_as_iter', type=int, default=12800)
# evaluation #
parser.add_argument('--metric_ks', nargs='+', type=int, default=[1, 5, 10, 20])
parser.add_argument('--best_metric', type=str, default='NDCG@10')

################
# Model
################
parser.add_argument('--model_code', type=str, default='bert', choices=['bert', 'sas', 'narm'])
# BERT specs, used for SASRec and NARM as well #
parser.add_argument('--bert_max_len', type=int, default=None)
parser.add_argument('--bert_hidden_units', type=int, default=64)
parser.add_argument('--bert_num_blocks', type=int, default=2)
parser.add_argument('--bert_num_heads', type=int, default=2)
parser.add_argument('--bert_head_size', type=int, default=32)
parser.add_argument('--bert_dropout', type=float, default=0.1)
parser.add_argument('--bert_attn_dropout', type=float, default=0.1)
parser.add_argument('--bert_mask_prob', type=float, default=0.2)

################
# Distillation & Retraining
################
parser.add_argument('--num_generated_seqs', type=int, default=3000)
parser.add_argument('--num_original_seqs', type=int, default=0)
parser.add_argument('--num_poisoned_seqs', type=int, default=100)
parser.add_argument('--num_alter_items', type=int, default=10)

################

args = parser.parse_args(args={})

print('\n'.join(f'{k}={v}' for k, v in vars(args).items()))
```

```python colab={"base_uri": "https://localhost:8080/"} id="pT6EteS8i6M9" executionInfo={"status": "ok", "timestamp": 1631873850392, "user_tz": -330, "elapsed": 8081, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="1ff4b2e9-d474-4b5d-98ae-4a54b5070265"
set_template(args)
print('\n'.join(f'{k}={v}' for k, v in vars(args).items()))
```

```python id="AknBtJvTiNCV"
def fix_random_seed_as(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

```python id="OeAGQBj7hmwb"
fix_random_seed_as(args.model_init_seed)
```

```python id="A9W86oDnQ3cu"
import dvc.api
```

```python colab={"base_uri": "https://localhost:8080/"} id="yhhewFrziUnW" executionInfo={"status": "ok", "timestamp": 1631874009549, "user_tz": -330, "elapsed": 79194, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="b6df7353-c7ab-422a-85c9-8b98327b4ff0"
train_loader, val_loader, test_loader = dataloader_factory(args)
```

<!-- #region id="oe7BpmDxpVls" -->
## Model
<!-- #endregion -->

```python id="q1HbzTinr7F2"
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size=512):
        super().__init__(vocab_size, embed_size, padding_idx=0)


class PositionalEmbedding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.d_model = d_model
        self.pe = nn.Embedding(max_len+1, d_model)

    def forward(self, x):
        pose = (x > 0) * (x > 0).sum(dim=-1).unsqueeze(1).repeat(1, x.size(-1))
        pose += torch.arange(start=-(x.size(1)-1), end=1, step=1, device=x.device)
        pose = pose * (x > 0)

        return self.pe(pose)


class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.activation = GELU()

    def forward(self, x):
        return self.w_2(self.activation(self.w_1(x)))


# layer norm
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.weight * (x - mean) / (std + self.eps) + self.bias


# layer norm and dropout (dropout and then layer norm)
class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super().__init__()
        self.layer_norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        # return x + self.dropout(sublayer(self.norm(x)))  # original implementation
        return self.layer_norm(x + self.dropout(sublayer(x)))  # BERT4Rec implementation


class Attention(nn.Module):
    def forward(self, query, key, value, mask=None, dropout=None, sas=False):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
            / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        if sas:
            direction_mask = torch.ones_like(scores)
            direction_mask = torch.tril(direction_mask)
            scores = scores.masked_fill(direction_mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, head_size=None, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        self.h = h
        self.d_k = d_model // h
        if head_size is not None:
            self.head_size = head_size
        else:
            self.head_size = d_model // h

        self.linear_layers = nn.ModuleList(
            [nn.Linear(d_model, self.h * self.head_size) for _ in range(3)])
        self.attention = Attention()
        self.dropout = nn.Dropout(p=dropout)
        self.output_linear = nn.Linear(self.h * self.head_size, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1) do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.head_size).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]
        
        # 2) apply attention on all the projected vectors in batch.
        x, attn = self.attention(
            query, key, value, mask=mask, dropout=self.dropout)

        # 3) "concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(
            batch_size, -1, self.h * self.head_size)
        return self.output_linear(x)


class TransformerBlock(nn.Module):
    def __init__(self, hidden, attn_heads, head_size, feed_forward_hidden, dropout, attn_dropout=0.1):
        super().__init__()
        self.attention = MultiHeadedAttention(
            h=attn_heads, d_model=hidden, head_size=head_size, dropout=attn_dropout)
        self.feed_forward = PositionwiseFeedForward(
            d_model=hidden, d_ff=feed_forward_hidden)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)

    def forward(self, x, mask):
        x = self.input_sublayer(
            x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
        x = self.output_sublayer(x, self.feed_forward)
        return x


class SASMultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, head_size=None, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        self.h = h
        self.d_k = d_model // h
        if head_size is not None:
            self.head_size = head_size
        else:
            self.head_size = d_model // h

        self.linear_layers = nn.ModuleList(
            [nn.Linear(d_model, self.h * self.head_size) for _ in range(3)])
        self.attention = Attention()
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = LayerNorm(d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1) do all the linear projections in batch from d_model => h x d_k
        query_, key_, value_ = [l(x).view(batch_size, -1, self.h, self.head_size).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]
        
        # 2) apply attention on all the projected vectors in batch.
        x, attn = self.attention(
            query_, key_, value_, mask=mask, dropout=self.dropout, sas=True)

        # 3) "concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(
            batch_size, -1, self.h * self.head_size)
        
        return self.layer_norm(x + query)


class SASPositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.layer_norm = LayerNorm(d_model)

    def forward(self, x):
        x_ = self.dropout(self.activation(self.conv1(x.permute(0, 2, 1))))
        return self.layer_norm(self.dropout(self.conv2(x_)).permute(0, 2, 1) + x)


class SASTransformerBlock(nn.Module):
    def __init__(self, hidden, attn_heads, head_size, feed_forward_hidden, dropout, attn_dropout=0.1):
        super().__init__()
        self.layer_norm = LayerNorm(hidden)
        self.attention = SASMultiHeadedAttention(
            h=attn_heads, d_model=hidden, head_size=head_size, dropout=attn_dropout)
        self.feed_forward = SASPositionwiseFeedForward(
            d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)

    def forward(self, x, mask):
        x = self.attention(self.layer_norm(x), x, x, mask)
        x = self.feed_forward(x)
        return x
```

```python id="sXgy-q-cr_Db"
from torch import nn as nn
import math


class BERT(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embedding = BERTEmbedding(self.args)
        self.model = BERTModel(self.args)
        self.truncated_normal_init()

    def truncated_normal_init(self, mean=0, std=0.02, lower=-0.04, upper=0.04):
        with torch.no_grad():
            l = (1. + math.erf(((lower - mean) / std) / math.sqrt(2.))) / 2.
            u = (1. + math.erf(((upper - mean) / std) / math.sqrt(2.))) / 2.

            for n, p in self.model.named_parameters():
                if not 'layer_norm' in n:
                    p.uniform_(2 * l - 1, 2 * u - 1)
                    p.erfinv_()
                    p.mul_(std * math.sqrt(2.))
                    p.add_(mean)
        
    def forward(self, x):
        x, mask = self.embedding(x)
        scores = self.model(x, self.embedding.token.weight, mask)
        return scores


class BERTEmbedding(nn.Module):
    def __init__(self, args):
        super().__init__()
        vocab_size = args.num_items + 2
        hidden = args.bert_hidden_units
        max_len = args.bert_max_len
        dropout = args.bert_dropout

        self.token = TokenEmbedding(
            vocab_size=vocab_size, embed_size=hidden)
        self.position = PositionalEmbedding(
            max_len=max_len, d_model=hidden)

        self.layer_norm = LayerNorm(features=hidden)
        self.dropout = nn.Dropout(p=dropout)

    def get_mask(self, x):
        if len(x.shape) > 2:
            x = torch.ones(x.shape[:2]).to(x.device)
        return (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

    def forward(self, x):
        mask = self.get_mask(x)
        if len(x.shape) > 2:
            pos = self.position(torch.ones(x.shape[:2]).to(x.device))
            x = torch.matmul(x, self.token.weight) + pos
        else:
            x = self.token(x) + self.position(x)
        return self.dropout(self.layer_norm(x)), mask


class BERTModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        hidden = args.bert_hidden_units
        heads = args.bert_num_heads
        head_size = args.bert_head_size
        dropout = args.bert_dropout
        attn_dropout = args.bert_attn_dropout
        layers = args.bert_num_blocks

        self.transformer_blocks = nn.ModuleList([TransformerBlock(
            hidden, heads, head_size, hidden * 4, dropout, attn_dropout) for _ in range(layers)])
        self.linear = nn.Linear(hidden, hidden)
        self.bias = torch.nn.Parameter(torch.zeros(args.num_items + 2))
        self.bias.requires_grad = True
        self.activation = GELU()

    def forward(self, x, embedding_weight, mask):
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)
        x = self.activation(self.linear(x))
        scores = torch.matmul(x, embedding_weight.permute(1, 0)) + self.bias
        return scores
```

<!-- #region id="uvIV4L_MtNWc" -->
## Run
<!-- #endregion -->

```python id="oF_w0ys1sDbU"
if args.model_code == 'bert':
    model = BERT(args)
# elif args.model_code == 'sas':
#     model = SASRec(args)
# elif args.model_code == 'narm':
#     model = NARM(args)
```

```python id="ectrw0tJs5S7"
export_root = 'experiments/' + args.model_code + '/' + args.dataset_code
```

```python id="Xm13hL0xs7aV"
resume=False
if resume:
    try: 
        model.load_state_dict(torch.load(os.path.join(export_root, 'models', 'best_acc_model.pth'), map_location='cpu').get(STATE_DICT_KEY))
    except FileNotFoundError:
        print('Failed to load old model, continue training new model...')
```

<!-- #region id="OEnP_ZeetK5w" -->
## Trainer
<!-- #endregion -->

```python id="Ic_s_K8Et0F_"
STATE_DICT_KEY = 'model_state_dict'
OPTIMIZER_STATE_DICT_KEY = 'optimizer_state_dict'
```

```python id="LWZ_v8N5tjq8"
import os
import torch
from abc import ABCMeta, abstractmethod


def save_state_dict(state_dict, path, filename):
    torch.save(state_dict, os.path.join(path, filename))


class LoggerService(object):
    def __init__(self, train_loggers=None, val_loggers=None):
        self.train_loggers = train_loggers if train_loggers else []
        self.val_loggers = val_loggers if val_loggers else []

    def complete(self, log_data):
        for logger in self.train_loggers:
            logger.complete(**log_data)
        for logger in self.val_loggers:
            logger.complete(**log_data)

    def log_train(self, log_data):
        for logger in self.train_loggers:
            logger.log(**log_data)

    def log_val(self, log_data):
        for logger in self.val_loggers:
            logger.log(**log_data)


class AbstractBaseLogger(metaclass=ABCMeta):
    @abstractmethod
    def log(self, *args, **kwargs):
        raise NotImplementedError

    def complete(self, *args, **kwargs):
        pass


class RecentModelLogger(AbstractBaseLogger):
    def __init__(self, checkpoint_path, filename='checkpoint-recent.pth'):
        self.checkpoint_path = checkpoint_path
        if not os.path.exists(self.checkpoint_path):
            os.mkdir(self.checkpoint_path)
        self.recent_epoch = None
        self.filename = filename

    def log(self, *args, **kwargs):
        epoch = kwargs['epoch']

        if self.recent_epoch != epoch:
            self.recent_epoch = epoch
            state_dict = kwargs['state_dict']
            state_dict['epoch'] = kwargs['epoch']
            save_state_dict(state_dict, self.checkpoint_path, self.filename)

    def complete(self, *args, **kwargs):
        save_state_dict(kwargs['state_dict'],
                        self.checkpoint_path, self.filename + '.final')


class BestModelLogger(AbstractBaseLogger):
    def __init__(self, checkpoint_path, metric_key='mean_iou', filename='best_acc_model.pth'):
        self.checkpoint_path = checkpoint_path
        if not os.path.exists(self.checkpoint_path):
            os.mkdir(self.checkpoint_path)

        self.best_metric = 0.
        self.metric_key = metric_key
        self.filename = filename

    def log(self, *args, **kwargs):
        current_metric = kwargs[self.metric_key]
        if self.best_metric < current_metric:
            print("Update Best {} Model at {}".format(
                self.metric_key, kwargs['epoch']))
            self.best_metric = current_metric
            save_state_dict(kwargs['state_dict'],
                            self.checkpoint_path, self.filename)


class MetricGraphPrinter(AbstractBaseLogger):
    def __init__(self, writer, key='train_loss', graph_name='Train Loss', group_name='metric'):
        self.key = key
        self.graph_label = graph_name
        self.group_name = group_name
        self.writer = writer

    def log(self, *args, **kwargs):
        if self.key in kwargs:
            self.writer.add_scalar(
                self.group_name + '/' + self.graph_label, kwargs[self.key], kwargs['accum_iter'])
        else:
            self.writer.add_scalar(
                self.group_name + '/' + self.graph_label, 0, kwargs['accum_iter'])

    def complete(self, *args, **kwargs):
        self.writer.close()
```

```python id="OCpOZTWQtcc5"
import json
import os
import pprint as pp
import random
from datetime import date
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch import optim as optim


def ndcg(scores, labels, k):
    scores = scores.cpu()
    labels = labels.cpu()
    rank = (-scores).argsort(dim=1)
    cut = rank[:, :k]
    hits = labels.gather(1, cut)
    position = torch.arange(2, 2+k)
    weights = 1 / torch.log2(position.float())
    dcg = (hits.float() * weights).sum(1)
    idcg = torch.Tensor([weights[:min(int(n), k)].sum()
                         for n in labels.sum(1)])
    ndcg = dcg / idcg
    return ndcg.mean()


def recalls_and_ndcgs_for_ks(scores, labels, ks):
    metrics = {}

    scores = scores
    labels = labels
    answer_count = labels.sum(1)

    labels_float = labels.float()
    rank = (-scores).argsort(dim=1)

    cut = rank
    for k in sorted(ks, reverse=True):
        cut = cut[:, :k]
        hits = labels_float.gather(1, cut)
        metrics['Recall@%d' % k] = \
            (hits.sum(1) / torch.min(torch.Tensor([k]).to(
                labels.device), labels.sum(1).float())).mean().cpu().item()

        position = torch.arange(2, 2+k)
        weights = 1 / torch.log2(position.float())
        dcg = (hits * weights.to(hits.device)).sum(1)
        idcg = torch.Tensor([weights[:min(int(n), k)].sum()
                             for n in answer_count]).to(dcg.device)
        ndcg = (dcg / idcg).mean()
        metrics['NDCG@%d' % k] = ndcg.cpu().item()
    return metrics


def em_and_agreement(scores_rank, labels_rank):
    em = (scores_rank == labels_rank).float().mean()
    temp = np.hstack((scores_rank.numpy(), labels_rank.numpy()))
    temp = np.sort(temp, axis=1)
    agreement = np.mean(np.sum(temp[:, 1:] == temp[:, :-1], axis=1))
    return em, agreement


def kl_agreements_and_intersctions_for_ks(scores, soft_labels, ks, k_kl=100):
    metrics = {}
    scores = scores.cpu()
    soft_labels = soft_labels.cpu()
    scores_rank = (-scores).argsort(dim=1)
    labels_rank = (-soft_labels).argsort(dim=1)

    top_kl_scores = F.log_softmax(scores.gather(1, labels_rank[:, :k_kl]), dim=-1)
    top_kl_labels = F.softmax(soft_labels.gather(1, labels_rank[:, :k_kl]), dim=-1)
    kl = F.kl_div(top_kl_scores, top_kl_labels, reduction='batchmean')
    metrics['KL-Div'] = kl.item()
    for k in sorted(ks, reverse=True):
        em, agreement = em_and_agreement(scores_rank[:, :k], labels_rank[:, :k])
        metrics['EM@%d' % k] = em.item()
        metrics['Agr@%d' % k] = (agreement / k).item()
    return metrics


class AverageMeterSet(object):
    def __init__(self, meters=None):
        self.meters = meters if meters else {}

    def __getitem__(self, key):
        if key not in self.meters:
            meter = AverageMeter()
            meter.update(0)
            return meter
        return self.meters[key]

    def update(self, name, value, n=1):
        if name not in self.meters:
            self.meters[name] = AverageMeter()
        self.meters[name].update(value, n)

    def reset(self):
        for meter in self.meters.values():
            meter.reset()

    def values(self, format_string='{}'):
        return {format_string.format(name): meter.val for name, meter in self.meters.items()}

    def averages(self, format_string='{}'):
        return {format_string.format(name): meter.avg for name, meter in self.meters.items()}

    def sums(self, format_string='{}'):
        return {format_string.format(name): meter.sum for name, meter in self.meters.items()}

    def counts(self, format_string='{}'):
        return {format_string.format(name): meter.count for name, meter in self.meters.items()}


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

    def __format__(self, format):
        return "{self.val:{format}} ({self.avg:{format}})".format(self=self, format=format)
```

```python id="XFZx45xjt8YY" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1631875337607, "user_tz": -330, "elapsed": 19238, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="b6b99e36-75f5-4140-e796-468d6f406500"
!pip install faiss-cpu --no-cache
!apt-get install libomp-dev
```

```python id="IMszuWPHtQ4T"
# from config import STATE_DICT_KEY, OPTIMIZER_STATE_DICT_KEY
# from .utils import *
# from .loggers import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import json
import faiss
import numpy as np
from abc import *
from pathlib import Path


class BERTTrainer(metaclass=ABCMeta):
    def __init__(self, args, model, train_loader, val_loader, test_loader, export_root):
        self.args = args
        self.device = args.device
        self.model = model.to(self.device)
        self.is_parallel = args.num_gpu > 1
        if self.is_parallel:
            self.model = nn.DataParallel(self.model)

        self.num_epochs = args.num_epochs
        self.metric_ks = args.metric_ks
        self.best_metric = args.best_metric
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = self._create_optimizer()
        if args.enable_lr_schedule:
            if args.enable_lr_warmup:
                self.lr_scheduler = self.get_linear_schedule_with_warmup(
                    self.optimizer, args.warmup_steps, len(train_loader) * self.num_epochs)
            else:
                self.lr_scheduler = optim.lr_scheduler.StepLR(
                    self.optimizer, step_size=args.decay_step, gamma=args.gamma)
            
        self.export_root = export_root
        self.writer, self.train_loggers, self.val_loggers = self._create_loggers()
        self.logger_service = LoggerService(
            self.train_loggers, self.val_loggers)
        self.log_period_as_iter = args.log_period_as_iter

        self.ce = nn.CrossEntropyLoss(ignore_index=0)

    def train(self):
        accum_iter = 0
        self.validate(0, accum_iter)
        for epoch in range(self.num_epochs):
            accum_iter = self.train_one_epoch(epoch, accum_iter)
            self.validate(epoch, accum_iter)
        self.logger_service.complete({
            'state_dict': (self._create_state_dict()),
        })
        self.writer.close()

    def train_one_epoch(self, epoch, accum_iter):
        self.model.train()
        average_meter_set = AverageMeterSet()
        tqdm_dataloader = tqdm(self.train_loader)

        for batch_idx, batch in enumerate(tqdm_dataloader):
            batch_size = batch[0].size(0)
            batch = [x.to(self.device) for x in batch]

            self.optimizer.zero_grad()
            loss = self.calculate_loss(batch)
            loss.backward()
            self.clip_gradients(5)
            self.optimizer.step()
            if self.args.enable_lr_schedule:
                self.lr_scheduler.step()

            average_meter_set.update('loss', loss.item())
            tqdm_dataloader.set_description(
                'Epoch {}, loss {:.3f} '.format(epoch+1, average_meter_set['loss'].avg))

            accum_iter += batch_size

            if self._needs_to_log(accum_iter):
                tqdm_dataloader.set_description('Logging to Tensorboard')
                log_data = {
                    'state_dict': (self._create_state_dict()),
                    'epoch': epoch + 1,
                    'accum_iter': accum_iter,
                }
                log_data.update(average_meter_set.averages())
                self.logger_service.log_train(log_data)

        return accum_iter

    def validate(self, epoch, accum_iter):
        self.model.eval()

        average_meter_set = AverageMeterSet()

        with torch.no_grad():
            tqdm_dataloader = tqdm(self.val_loader)
            for batch_idx, batch in enumerate(tqdm_dataloader):
                batch = [x.to(self.device) for x in batch]

                metrics = self.calculate_metrics(batch)
                self._update_meter_set(average_meter_set, metrics)
                self._update_dataloader_metrics(
                    tqdm_dataloader, average_meter_set)

            log_data = {
                'state_dict': (self._create_state_dict()),
                'epoch': epoch+1,
                'accum_iter': accum_iter,
            }
            log_data.update(average_meter_set.averages())
            self.logger_service.log_val(log_data)

    def test(self):
        best_model_dict = torch.load(os.path.join(
            self.export_root, 'models', 'best_acc_model.pth')).get(STATE_DICT_KEY)
        self.model.load_state_dict(best_model_dict)
        self.model.eval()

        average_meter_set = AverageMeterSet()

        all_scores = []
        average_scores = []
        with torch.no_grad():
            tqdm_dataloader = tqdm(self.test_loader)
            for batch_idx, batch in enumerate(tqdm_dataloader):
                batch = [x.to(self.device) for x in batch]
                metrics = self.calculate_metrics(batch)
                
                # seqs, candidates, labels = batch
                # scores = self.model(seqs)
                # scores = scores[:, -1, :]
                # scores_sorted, indices = torch.sort(scores, dim=-1, descending=True)
                # all_scores += scores_sorted[:, :100].cpu().numpy().tolist()
                # average_scores += scores_sorted.cpu().numpy().tolist()
                # scores = scores.gather(1, candidates)
                # metrics = recalls_and_ndcgs_for_ks(scores, labels, self.metric_ks)

                self._update_meter_set(average_meter_set, metrics)
                self._update_dataloader_metrics(
                    tqdm_dataloader, average_meter_set)

            average_metrics = average_meter_set.averages()
            with open(os.path.join(self.export_root, 'logs', 'test_metrics.json'), 'w') as f:
                json.dump(average_metrics, f, indent=4)
        
        return average_metrics

    def calculate_loss(self, batch):
        seqs, labels = batch
        logits = self.model(seqs)

        logits = logits.view(-1, logits.size(-1))
        labels = labels.view(-1)
        loss = self.ce(logits, labels)
        return loss

    def calculate_metrics(self, batch):
        seqs, candidates, labels = batch

        scores = self.model(seqs)
        scores = scores[:, -1, :]
        scores = scores.gather(1, candidates)

        metrics = recalls_and_ndcgs_for_ks(scores, labels, self.metric_ks)
        return metrics

    def clip_gradients(self, limit=5):
        for p in self.model.parameters():
            nn.utils.clip_grad_norm_(p, 5)

    def _update_meter_set(self, meter_set, metrics):
        for k, v in metrics.items():
            meter_set.update(k, v)

    def _update_dataloader_metrics(self, tqdm_dataloader, meter_set):
        description_metrics = ['NDCG@%d' % k for k in self.metric_ks[:3]
                               ] + ['Recall@%d' % k for k in self.metric_ks[:3]]
        description = 'Eval: ' + \
            ', '.join(s + ' {:.3f}' for s in description_metrics)
        description = description.replace('NDCG', 'N').replace('Recall', 'R')
        description = description.format(
            *(meter_set[k].avg for k in description_metrics))
        tqdm_dataloader.set_description(description)

    def _create_optimizer(self):
        args = self.args
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'layer_norm']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay': args.weight_decay,
            },
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        ]
        if args.optimizer.lower() == 'adamw':
            return optim.AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon)
        elif args.optimizer.lower() == 'adam':
            return optim.Adam(optimizer_grouped_parameters, lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer.lower() == 'sgd':
            return optim.SGD(optimizer_grouped_parameters, lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
        else:
            raise ValueError

    def get_linear_schedule_with_warmup(self, optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
        # based on hugging face get_linear_schedule_with_warmup
        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(
                0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
            )

        return LambdaLR(optimizer, lr_lambda, last_epoch)

    def _create_loggers(self):
        root = Path(self.export_root)
        writer = SummaryWriter(root.joinpath('logs'))
        model_checkpoint = root.joinpath('models')

        train_loggers = [
            MetricGraphPrinter(writer, key='epoch',
                               graph_name='Epoch', group_name='Train'),
            MetricGraphPrinter(writer, key='loss',
                               graph_name='Loss', group_name='Train'),
        ]

        val_loggers = []
        for k in self.metric_ks:
            val_loggers.append(
                MetricGraphPrinter(writer, key='NDCG@%d' % k, graph_name='NDCG@%d' % k, group_name='Validation'))
            val_loggers.append(
                MetricGraphPrinter(writer, key='Recall@%d' % k, graph_name='Recall@%d' % k, group_name='Validation'))
        val_loggers.append(RecentModelLogger(model_checkpoint))
        val_loggers.append(BestModelLogger(
            model_checkpoint, metric_key=self.best_metric))
        return writer, train_loggers, val_loggers

    def _create_state_dict(self):
        return {
            STATE_DICT_KEY: self.model.module.state_dict() if self.is_parallel else self.model.state_dict(),
            OPTIMIZER_STATE_DICT_KEY: self.optimizer.state_dict(),
        }

    def _needs_to_log(self, accum_iter):
        return accum_iter % self.log_period_as_iter < self.args.train_batch_size and accum_iter != 0
```

<!-- #region id="ImYTqhiQtMEV" -->
## Run
<!-- #endregion -->

```python id="LMcGCoJbtGrh"
if args.model_code == 'bert':
    trainer = BERTTrainer(args, model, train_loader, val_loader, test_loader, export_root)
if args.model_code == 'sas':
    trainer = SASTrainer(args, model, train_loader, val_loader, test_loader, export_root)
eli f args.model_code == 'narm':
    args.num_epochs = 100
    trainer = RNNTrainer(args, model, train_loader, val_loader, test_loader, export_root)
```

```python colab={"base_uri": "https://localhost:8080/"} id="MIwZM-5GtS5f" outputId="7503f208-7976-409c-8813-91e9b07286d5"
trainer.train()
```

```python id="NzxzWNf4v8sk"
trainer.test()
```
