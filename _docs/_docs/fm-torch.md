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

<!-- #region id="Ad4vDWVaEEM0" -->
# Comparing DCN, DeepFM and xDeepFM in PyTorch
<!-- #endregion -->

<!-- #region id="rk7Sa73eGMoq" -->
![](https://github.com/RecoHut-Stanzas/S516304/raw/main/images/process_flow.svg)
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="QAe_2VaZ9uBb" executionInfo={"status": "ok", "timestamp": 1638462370036, "user_tz": -330, "elapsed": 12000, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="8234472c-dae5-4011-d088-d8c5b03cf019"
!mkdir recohut && git clone https://github.com/RecoHut-Stanzas/S516304.git recohut
%cd recohut
```

```python id="R2H6XX0xAA8C"
import logging
from datetime import datetime
import sys
import os

from recohut.code.pytorch.models import DCN, DeepFM, xDeepFM
from recohut.code.pytorch.utils import seed_everything
from recohut.code.datasets import data_generator
from recohut.code.datasets.taobao import FeatureEncoder
from recohut.code.utils import set_logger, print_to_json
```

```python id="qOZ4DGUb-nKy"
sys.path.insert(0,'.')
```

<!-- #region id="GHHRovKV_4dD" -->
## DCN
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="onMyHAuv-PBm" executionInfo={"status": "ok", "timestamp": 1638462832630, "user_tz": -330, "elapsed": 6921, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="8c25dba1-c4fc-4d4a-99f3-4ceeef59a34c"
if __name__ == '__main__':
    feature_cols = [{'name': ["userid","adgroup_id","pid","cate_id","campaign_id","customer","brand","cms_segid",
                              "cms_group_id","final_gender_code","age_level","pvalue_level","shopping_level","occupation"],
                     'active': True, 'dtype': 'str', 'type': 'categorical'}]
    label_col = {'name': 'clk', 'dtype': float}

    params = {'model_id': 'DCN_demo',
              'dataset_id': 'tiny_data_demo',
              'train_data': './data/tiny_data/train_sample.csv',
              'valid_data': './data/tiny_data/valid_sample.csv',
              'test_data': './data/tiny_data/test_sample.csv',
              'model_root': '../checkpoints/',
              'data_root': './data/',
              'feature_cols': feature_cols,
              'label_col': label_col,
              'embedding_regularizer': 0,
              'net_regularizer': 0,
              'dnn_hidden_units': [64, 64],
              'dnn_activations': "relu",
              'crossing_layers': 3,
              'learning_rate': 1e-3,
              'net_dropout': 0,
              'batch_norm': False,
              'optimizer': 'adam',
              'task': 'binary_classification',
              'loss': 'binary_crossentropy',
              'metrics': ['logloss', 'AUC'],
              'min_categr_count': 1,
              'embedding_dim': 10,
              'batch_size': 64,
              'epochs': 3,
              'shuffle': True,
              'seed': 2019,
              'monitor': 'AUC',
              'monitor_mode': 'max',
              'use_hdf5': True,
              'pickle_feature_encoder': True,
              'save_best_only': True,
              'every_x_epochs': 1,
              'patience': 2,
              'workers': 1,
              'verbose': 0,
              'version': 'pytorch',
              'gpu': -1}

    set_logger(params)
    logging.info('Start the demo...')
    logging.info(print_to_json(params))
    seed_everything(seed=params['seed'])

    feature_encoder = FeatureEncoder(feature_cols, 
                                     label_col, 
                                     dataset_id=params['dataset_id'], 
                                     data_root=params["data_root"],
                                     version=params['version'])
    feature_encoder.fit(train_data=params['train_data'], 
                        min_categr_count=params['min_categr_count'])

    train_gen, valid_gen, test_gen = data_generator(feature_encoder,
                                                    train_data=params['train_data'],
                                                    valid_data=params['valid_data'],
                                                    test_data=params['test_data'],
                                                    batch_size=params['batch_size'],
                                                    shuffle=params['shuffle'],
                                                    use_hdf5=params['use_hdf5'])
    model = DCN(feature_encoder.feature_map, **params)
    model.fit_generator(train_gen, validation_data=valid_gen, epochs=params['epochs'],
                        verbose=params['verbose'])
    model.load_weights(model.checkpoint)
    
    logging.info('***** validation/test results *****')
    model.evaluate_generator(valid_gen)
    model.evaluate_generator(test_gen)
```

<!-- #region id="AU-gYC6a-5eL" -->
## DeepFM
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="ygmSV5-pAEnh" executionInfo={"status": "ok", "timestamp": 1638462957414, "user_tz": -330, "elapsed": 1370, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="c1c64c11-e987-429c-9dec-4704669bf3cd"
if __name__ == '__main__':
    feature_cols = [{'name': ["userid","adgroup_id","pid","cate_id","campaign_id","customer","brand","cms_segid",
                              "cms_group_id","final_gender_code","age_level","pvalue_level","shopping_level","occupation"],
                     'active': True, 'dtype': 'str', 'type': 'categorical'}]
    label_col = {'name': 'clk', 'dtype': float}

    params = {'model_id': 'DeepFM_demo',
              'dataset_id': 'tiny_data_demo',
              'train_data': './data/tiny_data/train_sample.csv',
              'valid_data': './data/tiny_data/valid_sample.csv',
              'test_data': './data/tiny_data/test_sample.csv',
              'model_root': '../checkpoints/',
              'data_root': './data/',
              'feature_cols': feature_cols,
              'label_col': label_col,
              'embedding_regularizer': 0,
              'net_regularizer': 0,
              'hidden_units': [64, 64],
              'hidden_activations': "relu",
              'learning_rate': 1e-3,
              'net_dropout': 0,
              'batch_norm': False,
              'optimizer': 'adam',
              'task': 'binary_classification',
              'loss': 'binary_crossentropy',
              'metrics': ['logloss', 'AUC'],
              'min_categr_count': 1,
              'embedding_dim': 10,
              'batch_size': 16,
              'epochs': 3,
              'shuffle': True,
              'seed': 2019,
              'monitor': 'AUC',
              'monitor_mode': 'max',
              'use_hdf5': True,
              'pickle_feature_encoder': True,
              'save_best_only': True,
              'every_x_epochs': 1,
              'patience': 2,
              'workers': 1,
              'verbose': 0,
              'version': 'pytorch',
              'gpu': -1}

    set_logger(params)
    logging.info('Start the demo...')
    logging.info(print_to_json(params))
    seed_everything(seed=params['seed'])

    feature_encoder = FeatureEncoder(feature_cols, 
                                     label_col, 
                                     dataset_id=params['dataset_id'], 
                                     data_root=params["data_root"],
                                     version=params['version'])
    feature_encoder.fit(train_data=params['train_data'], 
                        min_categr_count=params['min_categr_count'])

    train_gen, valid_gen, test_gen = data_generator(feature_encoder,
                                                    train_data=params['train_data'],
                                                    valid_data=params['valid_data'],
                                                    test_data=params['test_data'],
                                                    batch_size=params['batch_size'],
                                                    shuffle=params['shuffle'],
                                                    use_hdf5=params['use_hdf5'])
    model = DeepFM(feature_encoder.feature_map, **params)
    model.fit_generator(train_gen, validation_data=valid_gen, epochs=params['epochs'],
                        verbose=params['verbose'])
    model.load_weights(model.checkpoint)
    
    logging.info('***** validation/test results *****')
    model.evaluate_generator(valid_gen)
    model.evaluate_generator(test_gen)
```

<!-- #region id="i8W3qp5u_7D0" -->
## xDeepFM
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="X5BIJDlhAOEz" executionInfo={"status": "ok", "timestamp": 1638463013027, "user_tz": -330, "elapsed": 1667, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="0f75cd18-66ba-49dd-fea3-8ee41b474bbb"
if __name__ == '__main__':
    feature_cols = [{'name': ["userid","adgroup_id","pid","cate_id","campaign_id","customer","brand","cms_segid",
                              "cms_group_id","final_gender_code","age_level","pvalue_level","shopping_level","occupation"],
                     'active': True, 'dtype': 'str', 'type': 'categorical'}]
    label_col = {'name': 'clk', 'dtype': float}

    params = {'model_id': 'xDeepFM_demo',
              'dataset_id': 'tiny_data_demo',
              'train_data': './data/tiny_data/train_sample.csv',
              'valid_data': './data/tiny_data/valid_sample.csv',
              'test_data': './data/tiny_data/test_sample.csv',
              'model_root': '../checkpoints/',
              'data_root': './data/',
              'feature_cols': feature_cols,
              'label_col': label_col,
              'embedding_regularizer': 0,
              'net_regularizer': 0,
              'dnn_hidden_units': [64, 64],
              'dnn_activations': "relu",
              'learning_rate': 1e-3,
              'net_dropout': 0,
              'cin_layer_units': [16, 16, 16],
              'batch_norm': False,
              'optimizer': 'adam',
              'task': 'binary_classification',
              'loss': 'binary_crossentropy',
              'metrics': ['logloss', 'AUC'],
              'min_categr_count': 1,
              'embedding_dim': 10,
              'batch_size': 16,
              'epochs': 3,
              'shuffle': True,
              'seed': 2019,
              'monitor': 'AUC',
              'monitor_mode': 'max',
              'use_hdf5': True,
              'pickle_feature_encoder': True,
              'save_best_only': True,
              'every_x_epochs': 1,
              'patience': 2,
              'workers': 1,
              'verbose': 0,
              'version': 'pytorch',
              'gpu': -1}

    set_logger(params)
    logging.info('Start the demo...')
    logging.info(print_to_json(params))
    seed_everything(seed=params['seed'])

    feature_encoder = FeatureEncoder(feature_cols, 
                                     label_col, 
                                     dataset_id=params['dataset_id'], 
                                     data_root=params["data_root"],
                                     version=params['version'])
    feature_encoder.fit(train_data=params['train_data'], 
                        min_categr_count=params['min_categr_count'])

    train_gen, valid_gen, test_gen = data_generator(feature_encoder,
                                                    train_data=params['train_data'],
                                                    valid_data=params['valid_data'],
                                                    test_data=params['test_data'],
                                                    batch_size=params['batch_size'],
                                                    shuffle=params['shuffle'],
                                                    use_hdf5=params['use_hdf5'])
    model = xDeepFM(feature_encoder.feature_map, **params)
    model.fit_generator(train_gen, validation_data=valid_gen, epochs=params['epochs'],
                        verbose=params['verbose'])
    model.load_weights(model.checkpoint)
    
    logging.info('***** Train/validation/test results *****')
    model.evaluate_generator(train_gen)
    model.evaluate_generator(valid_gen)
    model.evaluate_generator(test_gen)
```

<!-- #region id="Po97V5ZCDiJw" -->
---
<!-- #endregion -->

```python id="uzQi0VCHDiJx"
!apt-get -qq install tree
```

```python colab={"base_uri": "https://localhost:8080/"} id="mDuT0d4fDiJy" executionInfo={"status": "ok", "timestamp": 1638463894636, "user_tz": -330, "elapsed": 490, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="79dfda6d-66a8-4482-a9c1-b3ffcacd1e0a"
!tree -h --du ../checkpoints
```

```python colab={"base_uri": "https://localhost:8080/"} id="3TPzYYxLDiJz" executionInfo={"status": "ok", "timestamp": 1638463899955, "user_tz": -330, "elapsed": 3605, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="c972d20d-a896-4213-aa5e-51895e5f937d"
!pip install -q watermark
%reload_ext watermark
%watermark -a "Sparsh A." -m -iv -u -t -d
```

<!-- #region id="ADdeoXN3DiJz" -->
---
<!-- #endregion -->

<!-- #region id="8aJvbotPDiJ0" -->
**END**
<!-- #endregion -->
