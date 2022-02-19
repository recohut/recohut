# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/datasets/datasets.aotm.ipynb (unless otherwise specified).

__all__ = ['AOTMDataset']

# Cell
from typing import List, Optional, Callable, Union, Any, Tuple

import os
import os.path as osp
from collections.abc import Sequence
import sys

import numpy as np
import pandas as pd
from datetime import timezone, datetime, timedelta
import time

from .bases.session import SessionDataset
from ..utils.common_utils import download_url, extract_zip, makedirs

# Cell
class AOTMDataset(SessionDataset):
    url = 'https://github.com/RecoHut-Datasets/aotm/raw/v1/aotm.zip'

    def __init__(self, root, process_method, min_session_length=2, min_item_support=2,
                 num_slices=5, days_offset=0, days_shift=95, days_train=90, days_test=5):
        min_date = session_length = None
        super().__init__(root, process_method, min_date, session_length,
                         min_session_length, min_item_support, num_slices, days_offset,
                         days_shift, days_train, days_test)

    @property
    def raw_file_names(self) -> str:
        return 'playlists-aotm.csv'

    @property
    def processed_file_names(self) -> str:
        return 'dataset.pkl'

    def download(self):
        path = download_url(self.url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        from shutil import move, rmtree
        move(osp.join(self.raw_dir, 'aotm', 'raw', 'playlists-aotm.csv'),
             osp.join(self.raw_dir, 'playlists-aotm.csv'))
        rmtree(osp.join(self.raw_dir, 'aotm'))
        os.unlink(path)

    def load(self):
        #load csv
        data = pd.read_csv(osp.join(self.raw_dir,self.raw_file_names), sep='\t')
        data.sort_values(by=['SessionId','Time'], inplace=True)

        #output
        data_start = datetime.fromtimestamp(data.Time.min(), timezone.utc)
        data_end = datetime.fromtimestamp(data.Time.max(), timezone.utc)

        print('Loaded data set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}\n\n'.
            format(len(data), data.SessionId.nunique(), data.ItemId.nunique(), data_start.date().isoformat(), data_end.date().isoformat()))

        self.data = data