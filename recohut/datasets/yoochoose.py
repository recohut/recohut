# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/datasets/datasets.yoochoose.ipynb (unless otherwise specified).

__all__ = ['YoochooseDataset']

# Cell
import numpy as np
import pandas as pd
import datetime
import os
import os.path as osp

from .bases.session import SessionDataset
from ..utils.common_utils import extract_zip

# Cell
class YoochooseDataset(SessionDataset):
    data_id = '1UEcKC4EfgMVD2n_zBvAyp0vRNyv7ndSF'

    def __init__(self,
                 root,
                 min_session_length: int = 2,
                 min_item_support: int = 5,
                 eval_sec: int = 86400,
                 ):
        super().__init__(root, min_session_length, min_item_support, eval_sec)

    @property
    def raw_file_names(self) -> str:
        return 'rsc15-clicks.dat'

    @property
    def processed_file_names(self) -> str:
        return ['yoochoose_train.txt','yoochoose_valid.txt']

    def download(self):
        from google_drive_downloader import GoogleDriveDownloader as gdd
        from shutil import move, rmtree

        path = osp.join(self.raw_dir, 'rsc15.zip')
        gdd.download_file_from_google_drive(self.data_id, path)
        extract_zip(path, self.raw_dir)
        move(osp.join(self.raw_dir, 'rsc15', 'raw', self.raw_file_names),
             osp.join(self.raw_dir, self.raw_file_names))
        rmtree(osp.join(self.raw_dir, 'rsc15'))
        os.unlink(path)

    def process(self):
        df = self.load_ratings_df()
        if self.min_session_length is not None:
            df = self.remove_short_sessions(df)
        if self.min_item_support is not None:
            df = self.remove_sparse_items(df)
        train, test = self.split_df(df)
        train.to_csv(self.processed_paths[0], sep=',', index=False)
        test.to_csv(self.processed_paths[1], sep=',', index=False)

    def load_ratings_df(self):
        df = pd.read_csv(self.raw_paths[0], header=None, usecols=[0, 1, 2],
                         dtype={0: np.int32, 1: str, 2: np.int64})
        df.columns = ['uid', 'timestamp', 'sid']
        df['timestamp'] = df['timestamp'].apply(lambda x: datetime.datetime.strptime(
            x, '%Y-%m-%dT%H:%M:%S.%fZ').timestamp())
        return df