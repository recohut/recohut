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

```python colab={"base_uri": "https://localhost:8080/"} id="xgFQ3iF3F5x-" executionInfo={"status": "ok", "timestamp": 1635696238491, "user_tz": -330, "elapsed": 1968, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="b5354a23-fd8a-4f18-e853-2acb36de9379"
!git clone https://github.com/hoyeoplee/MeLU.git
```

```python colab={"base_uri": "https://localhost:8080/"} id="LMHcqcJnSA_Y" executionInfo={"status": "ok", "timestamp": 1635699535138, "user_tz": -330, "elapsed": 1022, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="0949a542-346f-4236-908a-f0c2d49c61c8"
%cd MeLU
```

<!-- #region id="Q1XFYBBgF8iv" -->
## Preparing dataset
<!-- #endregion -->

```python id="28aqQZT3SrKq"
import os
from data_generation import generate
```

```python id="dlgHj1TNSnz1"
master_path= "./ml"
```

```python colab={"base_uri": "https://localhost:8080/"} id="GxWV9w_MGAMs" executionInfo={"status": "ok", "timestamp": 1635696737504, "user_tz": -330, "elapsed": 477065, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="40a4950b-bbc6-4e82-dedb-c0adda85194b"
if not os.path.exists("{}/".format(master_path)):
    os.mkdir("{}/".format(master_path))
    generate(master_path)
```

<!-- #region id="QX7T2hB7GDGs" -->
## Training a model

Our model needs support and query sets. The support set is for local update, and the query set is for global update.
<!-- #endregion -->

```python id="qJZYKG-aR6Sh"
import torch
import pickle
from MeLU import MeLU
from options import config
from model_training import training
```

```python colab={"base_uri": "https://localhost:8080/"} id="xtCnvZ14R7JP" executionInfo={"status": "ok", "timestamp": 1635699539016, "user_tz": -330, "elapsed": 14, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="c7fec823-73e8-4588-de1b-fb4d87d1ef70"
config
```

```python id="x3wtFGO2R8RT"
config['use_cuda'] = False
config['num_epoch'] = 1
config['embedding_dim'] = 8
config['first_fc_hidden_dim'] = 16
```

```python colab={"base_uri": "https://localhost:8080/", "height": 442} id="ePCLKoL_GIwn" executionInfo={"status": "error", "timestamp": 1635699763819, "user_tz": -330, "elapsed": 186033, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="f357be3c-e0fa-4a13-9b4b-4dd7acdc2947"
melu = MeLU(config)
model_filename = "{}/models.pkl".format(master_path)
if not os.path.exists(model_filename):
    # Load training dataset.
    training_set_size = int(len(os.listdir("{}/warm_state".format(master_path))) / 4)
    supp_xs_s = []
    supp_ys_s = []
    query_xs_s = []
    query_ys_s = []
    for idx in range(training_set_size):
        supp_xs_s.append(pickle.load(open("{}/warm_state/supp_x_{}.pkl".format(master_path, idx), "rb")))
        supp_ys_s.append(pickle.load(open("{}/warm_state/supp_y_{}.pkl".format(master_path, idx), "rb")))
        query_xs_s.append(pickle.load(open("{}/warm_state/query_x_{}.pkl".format(master_path, idx), "rb")))
        query_ys_s.append(pickle.load(open("{}/warm_state/query_y_{}.pkl".format(master_path, idx), "rb")))
    total_dataset = list(zip(supp_xs_s, supp_ys_s, query_xs_s, query_ys_s))
    del(supp_xs_s, supp_ys_s, query_xs_s, query_ys_s)
    training(melu, total_dataset, batch_size=config['batch_size'], num_epoch=config['num_epoch'], model_save=True, model_filename=model_filename)
else:
    trained_state_dict = torch.load(model_filename)
    melu.load_state_dict(trained_state_dict)
```

```python id="-BgtY3yIRBIp"
!apt-get install tree
```

```python id="dNKhNvdxRt-K" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1635699856717, "user_tz": -330, "elapsed": 539, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="f5e0e912-7eb5-488a-d94e-3bcd615f01f2"
!tree --du -h -C -L 2 .
```

<!-- #region id="vfLfolS6GOwg" -->
## Extracting evidence candidates

We extract evidence candidate list based on the MeLU.
<!-- #endregion -->

```python id="467NiQOLGQJG"
from evidence_candidate import selection

evidence_candidate_list = selection(melu, master_path, config['num_candidate'])
for movie, score in evidence_candidate_list:
    print(movie, score)
```
