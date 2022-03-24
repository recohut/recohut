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

<!-- #region id="UIe1TYuYudPY" -->
# MAMO Framework
<!-- #endregion -->

<!-- #region id="63IP87fSr-yZ" -->
## API setup
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="YkOSKuK1r_3x" executionInfo={"status": "ok", "timestamp": 1635270038457, "user_tz": -330, "elapsed": 1040, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="fd56a6e1-e9c6-4235-9f08-c8966168fd0b"
!git clone https://github.com/swisscom/ai-research-mamo-framework.git
%cd ai-research-mamo-framework
```

```python id="9TXAYj86sITI" executionInfo={"status": "ok", "timestamp": 1635270038460, "user_tz": -330, "elapsed": 10, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
import sys
sys.path.insert(0,'.')
```

<!-- #region id="WUPM0inHr2R3" -->
## API run for movielens
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="MNBHXcP-r6Mh" executionInfo={"status": "ok", "timestamp": 1635270066306, "user_tz": -330, "elapsed": 23156, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="070022c7-f8ef-4e70-8119-782534939a83"
!mkdir -p data
!cd data && gdown --id 15KwO7tk9S4M5raro2ndkYswFLh7MpPkt
!cd data && unzip movielens_data.zip
```

```python id="GBoyIeEgtaMK" executionInfo={"status": "ok", "timestamp": 1635270376708, "user_tz": -330, "elapsed": 644, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
!mkdir -p outputs
```

```python colab={"base_uri": "https://localhost:8080/"} id="k2tOVfk3sKMK" outputId="3658fa0d-7920-4e34-d394-91561d236dd9"
import torch
import numpy as np
import os

from dataloader.ae_data_handler import AEDataHandler
from models.multi_VAE import MultiVAE
from loss.vae_loss import VAELoss
from metric.recall_at_k import RecallAtK
from metric.revenue_at_k import RevenueAtK
from trainer import Trainer

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default="data", help="the path to the directory where the data is stored")
parser.add_argument("--models_dir", default="outputs", help="the path to the directory where to save the models, it must be empty")
args = parser.parse_args(args={})

# get the arguments
dir_path = args.data_dir
save_to_path = args.models_dir

# set up logging
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger("main")
logger.setLevel(logging.INFO)

# set cuda if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_data_path = os.path.join(
    dir_path, "movielens_small_training.npy")
validation_input_data_path = os.path.join(
    dir_path, "movielens_small_validation_input.npy")
validation_output_data_path = os.path.join(
    dir_path, "movielens_small_validation_test.npy")
test_input_data_path = os.path.join(
    dir_path, "movielens_small_test_input.npy")
test_output_data_path = os.path.join(
    dir_path, "movielens_small_test_test.npy")
products_data_path = os.path.join(
    dir_path, "movielens_products_data.npy")

data_handler = AEDataHandler(
    "MovieLensSmall", train_data_path, validation_input_data_path,
    validation_output_data_path, test_input_data_path,
    test_output_data_path)

input_dim = data_handler.get_input_dim()
output_dim = data_handler.get_output_dim()

products_data_np = np.load(products_data_path)
products_data_torch = torch.tensor(
    products_data_np, dtype=torch.float32).to(device)

# create model
model = MultiVAE(params="yaml_files/params_multi_VAE_training.yaml")

correctness_loss = VAELoss()
revenue_loss = VAELoss(weighted_vector=products_data_torch)
losses = [correctness_loss, revenue_loss]

recallAtK = RecallAtK(k=10)
revenueAtK = RevenueAtK(k=10, revenue=products_data_np)
validation_metrics = [recallAtK, revenueAtK]

trainer = Trainer(data_handler, model, losses, validation_metrics, save_to_path)
trainer.train()
print(trainer.pareto_manager._pareto_front)
```
