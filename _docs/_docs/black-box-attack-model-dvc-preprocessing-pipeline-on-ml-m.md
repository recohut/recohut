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

```python colab={"base_uri": "https://localhost:8080/"} id="ehMex-aqyDNF" executionInfo={"status": "ok", "timestamp": 1631362563548, "user_tz": -330, "elapsed": 32393, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="fcda2983-3857-428b-f137-6e5ea182d8c5"
import os
project_name = "recobase"; branch = "US567625"; account = "recohut"
project_path = os.path.join('/content', branch)

if not os.path.exists(project_path):
    !pip install -U -q dvc dvc[gdrive]
    !cp -r /content/drive/MyDrive/git_credentials/. ~
    !mkdir "{project_path}"
    %cd "{project_path}"
    !git init
    !git remote add origin https://github.com/"{account}"/"{project_name}".git
    !git pull origin "{branch}"
    !git checkout -b "{branch}"
else:
    %cd "{project_path}"
```

```python colab={"base_uri": "https://localhost:8080/"} id="V-iFgtQizQfH" executionInfo={"status": "ok", "timestamp": 1631362815619, "user_tz": -330, "elapsed": 673, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="0cd8f129-cf3d-48f7-d497-708ad26a19c6"
!git status
```

```python id="ZfU-gZ86yDNN"
!git add .
!git commit -m 'commit'
!git push origin "{branch}"
```

```python colab={"base_uri": "https://localhost:8080/"} id="danfLsnnyODO" executionInfo={"status": "ok", "timestamp": 1631364384179, "user_tz": -330, "elapsed": 754, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="85186705-3f5d-4fe4-ec52-d6aaca1014b2"
%%writefile ./src/preprocess.py
import pickle
import shutil
import tempfile
import os
from datetime import date
from pathlib import Path
import gzip
import argparse
from abc import *

import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()


class AbstractDataset(metaclass=ABCMeta):
    def __init__(self, args):
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

    @abstractmethod
    def preprocess(self):
        pass

    @abstractmethod
    def load_ratings_df(self):
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
        if self.split == 'leave_one_out':
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
        # folder_name = '{}_min_rating{}-min_uc{}-min_sc{}-split{}' \
        #     .format(self.code(), self.min_rating, self.min_uc, self.min_sc, self.split)
        # return preprocessed_root.joinpath(folder_name)
        return preprocessed_root

    def _get_preprocessed_dataset_path(self):
        folder = self._get_preprocessed_folder_path()
        return folder.joinpath('dataset.pkl')


class ML1MDataset(AbstractDataset):
    @classmethod
    def code(cls):
        return 'ml-1m'

    def preprocess(self):
        dataset_path = self._get_preprocessed_dataset_path()
        if not dataset_path.parent.is_dir():
            dataset_path.parent.mkdir(parents=True)
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
        df = pd.read_csv(file_path, sep='::', header=None, engine='python')
        df.columns = ['uid', 'sid', 'rating', 'timestamp']
        return df


if __name__ == '__main__':
    RAW_DATASET_ROOT_FOLDER = 'data/bronze'
    PREP_DATASET_ROOT_FOLDER = 'data/silver'
    class Args:
        min_rating = 0
        min_uc = 5
        min_sc = 5
        split = 'leave_one_out'
    args = Args()
    dataset = ML1MDataset(args)
    dataset.preprocess()
```

```python colab={"base_uri": "https://localhost:8080/"} id="4GSf9WmayqgG" executionInfo={"status": "ok", "timestamp": 1631364397094, "user_tz": -330, "elapsed": 12922, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="df0aca93-3593-41f8-a2f2-947254e24236"
!python ./src/preprocess.py
```

```python colab={"base_uri": "https://localhost:8080/"} id="kMkIAJCN4i63" executionInfo={"status": "ok", "timestamp": 1631364566037, "user_tz": -330, "elapsed": 15354, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="1ef949d1-f965-4eb1-caf6-4344a4c64713"
!dvc run -n preprocess \
          -d src/preprocess.py -d data/bronze/ml-1m/ratings.dat \
          -o data/silver/ml-1m \
          python src/preprocess.py
```

```python colab={"base_uri": "https://localhost:8080/"} id="pM9lssEo541j" executionInfo={"status": "ok", "timestamp": 1631364607279, "user_tz": -330, "elapsed": 491, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="ad40fdc9-e96a-4dc3-ea73-abf709e2f519"
!git status -u
```

```python colab={"base_uri": "https://localhost:8080/"} id="pmIx9Y1F5_TD" executionInfo={"status": "ok", "timestamp": 1631364668470, "user_tz": -330, "elapsed": 4427, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="7b4fa21c-945e-4cc3-de15-1af95d065867"
!dvc status
```

```python colab={"base_uri": "https://localhost:8080/"} id="tYuj8l5M6URQ" executionInfo={"status": "ok", "timestamp": 1631364714769, "user_tz": -330, "elapsed": 39726, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="db1dd29f-e5b4-4117-ce98-cb20bd92398c"
!dvc commit
!dvc push
```

```python colab={"base_uri": "https://localhost:8080/"} id="prWgCZcT6W2_" executionInfo={"status": "ok", "timestamp": 1631364742662, "user_tz": -330, "elapsed": 1868, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="3bdda2af-b912-4dc7-d0cd-b4fed75034b1"
!git add .
!git commit -m 'commit'
!git push origin "{branch}"
```

```python id="wqnAUiFQ6nO3"

```
