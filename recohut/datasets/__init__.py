from .aotm import *
from .avazu import *
from criteo import *
from diginetica import *
from gowalla import *
from kkbox import *
from lastfm import *
from .movielens import *
from .music30 import *

__all__ = [
    'AOTMDataset',
    'AvazuDataset',
    'AvazuDataModule',
    'CriteoDataset',
    'CriteoDataModule',
    'CriteoSampleDataset',
    'DigineticaDataset',
    'DigineticaDatasetv2',
    'GowallaDataset',
    'KKBoxDataset',
    'KKBoxDataModule',
    'LastfmDataset',
    'ML1mDataset',
    'ML1mDataModule',
    'ML1mDataset_v2',
    'ML1mDataModule_v2',
    'ML1mDataset_v3',
    'ML1mDataModule_v3',
    'ML100kDataset',
    'Music30Dataset'
]

classes = __all__