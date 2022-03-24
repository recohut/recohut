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

```python colab={"base_uri": "https://localhost:8080/"} id="e1kbTu0hTTXh" executionInfo={"status": "ok", "timestamp": 1635758835389, "user_tz": -330, "elapsed": 5760, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="e404b5ad-67aa-4b64-ec96-282faca8cc4d"
!git clone https://github.com/rootlu/MetaHIN.git
```

```python colab={"base_uri": "https://localhost:8080/"} id="cVHali9JULmq" executionInfo={"status": "ok", "timestamp": 1635758835390, "user_tz": -330, "elapsed": 8, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="357b9fe7-e2f6-4049-c00c-e6a805642083"
%cd MetaHIN
```

```python id="HEVPQhC5UND-" executionInfo={"status": "ok", "timestamp": 1635758840047, "user_tz": -330, "elapsed": 4, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
# !python code/main.py
```

```python colab={"base_uri": "https://localhost:8080/"} id="YAehd92jz_Sd" executionInfo={"status": "ok", "timestamp": 1635758854761, "user_tz": -330, "elapsed": 5, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="25f76d1e-b0af-408a-9ce1-e69c52ee6d7c"
%cd code
```

```python colab={"base_uri": "https://localhost:8080/"} id="J7NSG-BD03qx" executionInfo={"status": "ok", "timestamp": 1635758889623, "user_tz": -330, "elapsed": 826, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="3e1ecef4-e74a-425f-e722-bf359aeb0abd"
%%writefile Config.py
# coding: utf-8
# author: lu yf
# create date: 2019-11-20 19:46

config_db = {
    'dataset': 'dbook',
    # 'mp': ['ub'],
    # 'mp': ['ub','ubab'],
    'mp': ['ub','ubab','ubub'],
    'use_cuda': False,
    'file_num': 10,  # each task contains 10 files

    # user
    'num_location': 453,
    'num_fea_item': 2,

    # item
    'num_publisher': 1698,
    'num_fea_user': 1,
    'item_fea_len': 1,

    # model setting
    # 'embedding_dim': 32,
    # 'user_embedding_dim': 32*1,  # 1 features
    # 'item_embedding_dim': 32*1,  # 1 features

    'embedding_dim': 32,
    'user_embedding_dim': 32*1,  # 1 features
    'item_embedding_dim': 32*1,  # 1 features

    'first_fc_hidden_dim': 64,
    'second_fc_hidden_dim': 64,

    'mp_update': 1,
    'local_update': 1,
    'lr': 5e-4,
    'mp_lr': 5e-3,
    'local_lr': 5e-3,
    'batch_size': 32,  # for each batch, the number of tasks
    'num_epoch': 50,
    'neigh_agg': 'mean',
    # 'neigh_agg': 'attention',
    'mp_agg': 'mean',
    # 'mp_agg': 'attention',
}


config_ml = {
    'dataset': 'movielens',
    # 'mp': ['um'],
    # 'mp': ['um','umdm'],
    # 'mp': ['um','umam','umdm'],
    'mp': ['um','umum','umam','umdm'],
    'use_cuda': False,
    'file_num': 12,  # each task contains 12 files for movielens

    # item
    'num_rate': 6,
    'num_genre': 25,
    'num_fea_item': 2,
    'item_fea_len': 26,

    # user
    'num_gender': 2,
    'num_age': 7,
    'num_occupation': 21,
    'num_zipcode': 3402,
    'num_fea_user': 4,

    # model setting
    'embedding_dim': 32,
    'user_embedding_dim': 32*4,  # 4 features
    'item_embedding_dim': 32*2,  # 2 features

    'first_fc_hidden_dim': 64,
    'second_fc_hidden_dim': 64,

    'mp_update': 1,
    'local_update': 1,
    'lr': 5e-4,
    'mp_lr': 5e-3,
    'local_lr': 5e-3,
    'batch_size': 32,  # for each batch, the number of tasks
    'num_epoch': 100,
    'neigh_agg': 'mean',
    # 'neigh_agg': 'max',
    'mp_agg': 'mean',
    # 'mp_agg': 'attention',
}


config_yelp = {
    'dataset': 'yelp',
    # 'mp': ['ubub'],
    'mp': ['ub','ubcb','ubtb','ubub'],
    'use_cuda': False,
    'file_num': 12,  # each task contains 12 files

    # item
    'num_stars': 9,
    'num_postalcode': 6127,
    'num_fea_item': 2,
    'item_fea_len': 2,

    # user
    'num_fans': 412,
    'num_avgrating': 359,
    'num_fea_user': 2,

    # model setting
    'embedding_dim': 32,
    'user_embedding_dim': 32*2,  # 1 features
    'item_embedding_dim': 32*2,  # 1 features

    'first_fc_hidden_dim': 64,
    'second_fc_hidden_dim': 64,

    'mp_update': 1,
    'local_update': 1,
    'lr': 5e-4,
    'mp_lr': 1e-3,
    'local_lr': 1e-3,
    'batch_size': 32,  # for each batch, the number of tasks
    'num_epoch': 50,
    'neigh_agg': 'mean',
    # 'neigh_agg': 'attention',
    'mp_agg': 'mean',
    # 'mp_agg': 'attention',
}


states = ["meta_training","warm_up", "user_cold_testing", "item_cold_testing", "user_and_item_cold_testing"]
```

```python colab={"base_uri": "https://localhost:8080/"} id="6UKX-8MU1Jl0" executionInfo={"status": "ok", "timestamp": 1635759066902, "user_tz": -330, "elapsed": 7722, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="b28c30a8-dc34-4064-9525-baccfa773df6"
# !cd ../data/movielens && gdown --id 1bTpwctr6ZgdosU8FU4sLqlC_Gt1-z77G
```

```python colab={"base_uri": "https://localhost:8080/"} id="XJRl4gr61pKW" outputId="2e4863b2-00a4-4af8-dac6-de883850d48e"
!cd ../data/movielens && tar -xvf movielens.tar.bz2
```

```python id="Qo0a9QTnUShm" colab={"base_uri": "https://localhost:8080/", "height": 523} executionInfo={"status": "error", "timestamp": 1635758743571, "user_tz": -330, "elapsed": 39955, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="dcb57f52-71a8-4c86-fa73-e68ba27fb028"
import gc
import glob
import random
import time
import numpy as np
import torch
from HeteML_new import HML
from DataHelper import DataHelper
from tqdm.notebook import tqdm
# from Config import states
# random.seed(13)
np.random.seed(13)
torch.manual_seed(13)


def training(model, model_save=True, model_file=None, device='cpu'):
    print('training model...')
    if config['use_cuda']:
        model.cuda()
    model.train()

    batch_size = config['batch_size']
    num_epoch = config['num_epoch']

    for _ in range(num_epoch):  # 20
        loss, mae, rmse = [], [], []
        ndcg_at_5 = []
        start = time.time()

        random.shuffle(train_data)
        num_batch = int(len(train_data) / batch_size)  # ~80
        supp_xs_s, supp_ys_s, supp_mps_s, query_xs_s, query_ys_s, query_mps_s = zip(*train_data)  # supp_um_s:(list,list,...,2553)
        for i in range(num_batch):  # each batch contains some tasks (each task contains a support set and a query set)
            support_xs = list(supp_xs_s[batch_size * i:batch_size * (i + 1)])
            support_ys = list(supp_ys_s[batch_size * i:batch_size * (i + 1)])
            support_mps = list(supp_mps_s[batch_size * i:batch_size * (i + 1)])
            query_xs = list(query_xs_s[batch_size * i:batch_size * (i + 1)])
            query_ys = list(query_ys_s[batch_size * i:batch_size * (i + 1)])
            query_mps = list(query_mps_s[batch_size * i:batch_size * (i + 1)])

            _loss, _mae, _rmse, _ndcg_5 = model.global_update(support_xs,support_ys,support_mps,
                                                              query_xs,query_ys,query_mps,device)
            loss.append(_loss)
            mae.append(_mae)
            rmse.append(_rmse)
            ndcg_at_5.append(_ndcg_5)

        print('epoch: {}, loss: {:.6f}, cost time: {:.1f}s, mae: {:.5f}, rmse: {:.5f}, ndcg@5: {:.5f}'.
              format(_, np.mean(loss), time.time() - start,
                     np.mean(mae), np.mean(rmse), np.mean(ndcg_at_5)))
        if _ % 10 == 0 and _ != 0:
            testing(model, device)
            model.train()

    if model_save:
        print('saving model...')
        torch.save(model.state_dict(), model_file)


def testing(model, device='cpu'):
    # testing
    print('evaluating model...')
    if config['use_cuda']:
        model.cuda()
    model.eval()
    for state in states:
        if state == 'meta_training':
            continue
        print(state + '...')
        evaluate(model, state, device)


def evaluate(model, state, device='cpu'):
    test_data = data_helper.load_data(data_set=data_set, state=state,
                                      load_from_file=True)
    supp_xs_s, supp_ys_s, supp_mps_s, query_xs_s, query_ys_s, query_mps_s = zip(*test_data)  # supp_um_s:(list,list,...,2553)
    loss, mae, rmse = [], [], []
    ndcg_at_5 = []

    for i in range(len(test_data)):  # each task
        _mae, _rmse, _ndcg_5 = model.evaluation(supp_xs_s[i], supp_ys_s[i], supp_mps_s[i],
                                                query_xs_s[i], query_ys_s[i], query_mps_s[i],device)
        mae.append(_mae)
        rmse.append(_rmse)
        ndcg_at_5.append(_ndcg_5)
    print('mae: {:.5f}, rmse: {:.5f}, ndcg@5: {:.5f}'.
          format(np.mean(mae), np.mean(rmse),np.mean(ndcg_at_5)))

    # print('fine tuning...')
    # model.train()
    # for i in range(len(test_data)):
    #     model.fine_tune(supp_xs_s[i], supp_ys_s[i], supp_mps_s[i])
    # model.eval()
    # for i in range(len(test_data)):  # each task
    #     _mae, _rmse, _ndcg_5 = model.evaluation(supp_xs_s[i], supp_ys_s[i], supp_mps_s[i],
    #                                             query_xs_s[i], query_ys_s[i], query_mps_s[i],device)
    #     mae.append(_mae)
    #     rmse.append(_rmse)
    #     ndcg_at_5.append(_ndcg_5)
    # print('mae: {:.5f}, rmse: {:.5f}, ndcg@5: {:.5f}'.
    #       format(np.mean(mae), np.mean(rmse), np.mean(ndcg_at_5)))


if __name__ == "__main__":
    # data_set = 'dbook'
    data_set = 'movielens'
    # data_set = 'yelp'

    input_dir = '../data/'
    output_dir = '../data/'
    res_dir = '../res/'+data_set
    load_model = False

    if data_set == 'movielens':
        from Config import config_ml as config
    elif data_set == 'yelp':
        from Config import config_yelp as config
    elif data_set == 'dbook':
        from Config import config_db as config
    cuda_or_cpu = torch.device("cuda" if config['use_cuda'] else "cpu")
    print (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print(config)

    model_filename = "{}/hml.pkl".format(res_dir)
    data_helper = DataHelper(input_dir, output_dir, config)

    # training model.
    model_name = 'mp_update'
    # model_name = 'mp_MAML'
    # model_name = 'mp_update_multi_MAML'
    # model_name = 'mp_update_no_f'
    # model_name = 'no_MAML'
    # model_name = 'no_MAML_with_finetuning'
    hml = HML(config, model_name)

    print('--------------- {} ---------------'.format(model_name))

    if not load_model:
        # Load training dataset
        print('loading train data...')
        train_data = data_helper.load_data(data_set=data_set,state='meta_training',load_from_file=True)
        # print('loading warm data...')
        # warm_data = data_helper.load_data(data_set=data_set, state='warm_up',load_from_file=True)
        training(hml, model_save=True, model_file=model_filename,device=cuda_or_cpu)
    else:
        trained_state_dict = torch.load(model_filename)
        hml.load_state_dict(trained_state_dict)

    # testing
    testing(hml, device=cuda_or_cpu)
    print('--------------- {} ---------------'.format(model_name))
```

```python id="Ci9axlli0PWA"

```
