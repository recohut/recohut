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

<!-- #region id="KlNI72qJZnUv" -->
# BERT4Rec on ML-25m in PyTorch Lightning

Implementing BERT4Rec model on Movielens 25m dataset in PyTorch.
<!-- #endregion -->

<!-- #region id="6OZVcq6rYphR" -->
## Setup
<!-- #endregion -->

<!-- #region id="sZaJrJ1q96Jf" -->
### Installations
<!-- #endregion -->

```sh id="UPJ9X5rjXzwI"
mkdir /content/_temp
cd /content/_temp
wget https://files.grouplens.org/datasets/movielens/ml-25m.zip
unzip ml-25m.zip
cd /content
```

```python id="Zqn_uGP6YuEe"
!pip install -q pytorch_lightning
```

<!-- #region id="iM0d1bGcYhRV" -->
### Imports
<!-- #endregion -->

```python id="7duH6OQ1YzZ4"
import random
import os
import numpy as np
import pandas as pd
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import Linear
from torch.nn import functional as F
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
```

<!-- #region id="4ehDGA7GYse2" -->
### Params
<!-- #endregion -->

```python id="293b8dieYSoR"
class Args:
    PAD = 0
    MASK = 1
    CAP = 0
    SEED = 42
    RAW_DATA_PATH = '/content/_temp/ml-25m'
    VOCAB_SIZE = 10000
    CHANNELS = 128
    DROPOUT = 0.4
    LR = 1e-4
    HISTORY_SIZE = 120
    DEBUG_MODE = True
    DEBUG_LOAD = 1000
    LOG_DIR = '/content/recommender_logs'
    MODEL_DIR = '/content/recommender_models'
    BATCH_SIZE = 32
    EPOCHS = 2000

args = Args()
```

<!-- #region id="Oy459Dd2Yz0d" -->
## Utils
<!-- #endregion -->

```python id="-ila65IcJaBf"
def map_column(df: pd.DataFrame, col_name: str):
    """Maps column values to integers.
    """
    values = sorted(list(df[col_name].unique()))
    mapping = {k: i + 2 for i, k in enumerate(values)}
    inverse_mapping = {v: k for k, v in mapping.items()}
    df[col_name + "_mapped"] = df[col_name].map(mapping)
    return df, mapping, inverse_mapping

def get_context(df: pd.DataFrame, split: str, context_size: int = 120, val_context_size: int = 5, seed: int = 42):
    """Create a training / validation samples.
    """
    random.seed(seed)
    if split == "train":
        end_index = random.randint(10, df.shape[0] - val_context_size)
    elif split in ["val", "test"]:
        end_index = df.shape[0]
    else:
        raise ValueError
    start_index = max(0, end_index - context_size)
    context = df[start_index:end_index]
    return context

def pad_arr(arr: np.ndarray, expected_size: int = 30):
    """Pad top of array when there is not enough history.
    """
    arr = np.pad(arr, [(expected_size - arr.shape[0], 0), (0, 0)], mode="edge")
    return arr

def pad_list(list_integers, history_size: int, pad_val: int = 0, mode="left"):
    """Pad list from left or right
    """
    if len(list_integers) < history_size:
        if mode == "left":
            list_integers = [pad_val] * (history_size - len(list_integers)) + list_integers
        else:
            list_integers = list_integers + [pad_val] * (history_size - len(list_integers))
    return list_integers
```

```python id="FAXCIhFBTQsN"
def masked_accuracy(y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.Tensor):
    _, predicted = torch.max(y_pred, 1)
    y_true = torch.masked_select(y_true, mask)
    predicted = torch.masked_select(predicted, mask)
    acc = (y_true == predicted).double().mean()
    return acc
```

```python id="Do_cLe-4YgRM"
def masked_ce(y_pred, y_true, mask):
    loss = F.cross_entropy(y_pred, y_true, reduction="none")
    loss = loss * mask
    return loss.sum() / (mask.sum() + 1e-8)
```

```python id="hmX_49DMlumu"
def mask_list(l1, p=0.8):
    random.seed(args.SEED)
    l1 = [a if random.random() < p else args.MASK for a in l1]
    return l1
```

```python id="oVIfejNVT_a0"
def mask_last_elements_list(l1, val_context_size: int = 5):
    l1 = l1[:-val_context_size] + mask_list(l1[-val_context_size:], p=0.5)
    return l1
```

<!-- #region id="-usSJNrHY66A" -->
## Dataset
<!-- #endregion -->

```python id="rKFEK03VZNTZ"
class ML25Dataset(torch.utils.data.Dataset):
    def __init__(self, args, split='train'):
        self.args = args
        self.grp_by = None
        self.groups = None
        self.split = split
        self.history_size = args.HISTORY_SIZE
        self.mapping = None
        self.inverse_mapping = None
        self.load_dataset()

    def load_dataset(self):
        filepath = os.path.join(self.args.RAW_DATA_PATH, 'ratings.csv')
        if args.DEBUG_MODE:
            data = pd.read_csv(filepath, nrows=1000)
        else:
            data = pd.read_csv(filepath)
        data.sort_values(by="timestamp", inplace=True)
        data, self.mapping, self.inverse_mapping = map_column(data, col_name="movieId")
        self.grp_by = data.groupby(by="userId")
        self.groups = list(self.grp_by.groups)
        
    def genome_mapping(self, genome):
        """movie id to relevance mapping
        """
        genome.sort_values(by=["movieId", "tagId"], inplace=True)
        movie_genome = genome.groupby("movieId")["relevance"].agg(list).reset_index()
        movie_genome = {a: b for a, b in zip(movie_genome['movieId'], movie_genome['relevance'])}
        return movie_genome

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        group = self.groups[idx]
        df = self.grp_by.get_group(group)
        context = get_context(df, split=self.split, context_size=self.history_size)
        trg_items = context["movieId_mapped"].tolist()
        if self.split == "train":
            src_items = mask_list(trg_items)
        else:
            src_items = mask_last_elements_list(trg_items)
        pad_mode = "left" if random.random() < 0.5 else "right"
        trg_items = pad_list(trg_items, history_size=self.history_size, mode=pad_mode)
        src_items = pad_list(src_items, history_size=self.history_size, mode=pad_mode)
        src_items = torch.tensor(src_items, dtype=torch.long)
        trg_items = torch.tensor(trg_items, dtype=torch.long)
        return src_items, trg_items
```

<!-- #region id="a2mxEHOOZGWr" -->
## Model
<!-- #endregion -->

```python id="7E0d-NsuZPTJ"
class BERT4Rec(pl.LightningModule):
    def __init__(self, args, vocab_size):
        super().__init__()
        self.cap = args.CAP
        self.mask = args.MASK
        self.lr = args.LR
        self.dropout = args.DROPOUT
        self.vocab_size = vocab_size
        self.channels = args.CHANNELS

        self.item_embeddings = torch.nn.Embedding(
            self.vocab_size, embedding_dim=self.channels
        )
        self.input_pos_embedding = torch.nn.Embedding(512, embedding_dim=self.channels)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.channels, nhead=4, dropout=self.dropout
        )
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.linear_out = Linear(self.channels, self.vocab_size)
        self.do = nn.Dropout(p=self.dropout)

    def encode_src(self, src_items):
        src_items = self.item_embeddings(src_items)
        batch_size, in_sequence_len = src_items.size(0), src_items.size(1)
        pos_encoder = (
            torch.arange(0, in_sequence_len, device=src_items.device)
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )
        pos_encoder = self.input_pos_embedding(pos_encoder)
        src_items += pos_encoder
        src = src_items.permute(1, 0, 2)
        src = self.encoder(src)
        return src.permute(1, 0, 2)

    def forward(self, src_items):
        src = self.encode_src(src_items)
        out = self.linear_out(src)
        return out

    def training_step(self, batch, batch_idx):
        src_items, y_true = batch
        y_pred = self(src_items)
        y_pred = y_pred.view(-1, y_pred.size(2))
        y_true = y_true.view(-1)
        src_items = src_items.view(-1)
        mask = src_items == self.mask
        loss = masked_ce(y_pred=y_pred, y_true=y_true, mask=mask)
        accuracy = masked_accuracy(y_pred=y_pred, y_true=y_true, mask=mask)
        self.log("train_loss", loss)
        self.log("train_accuracy", accuracy)
        return loss

    def validation_step(self, batch, batch_idx):
        src_items, y_true = batch
        y_pred = self(src_items)
        y_pred = y_pred.view(-1, y_pred.size(2))
        y_true = y_true.view(-1)
        src_items = src_items.view(-1)
        mask = src_items == self.mask
        loss = masked_ce(y_pred=y_pred, y_true=y_true, mask=mask)
        accuracy = masked_accuracy(y_pred=y_pred, y_true=y_true, mask=mask)
        self.log("valid_loss", loss)
        self.log("valid_accuracy", accuracy)
        return loss

    def test_step(self, batch, batch_idx):
        src_items, y_true = batch
        y_pred = self(src_items)
        y_pred = y_pred.view(-1, y_pred.size(2))
        y_true = y_true.view(-1)
        src_items = src_items.view(-1)
        mask = src_items == self.mask
        loss = masked_ce(y_pred=y_pred, y_true=y_true, mask=mask)
        accuracy = masked_accuracy(y_pred=y_pred, y_true=y_true, mask=mask)
        self.log("test_loss", loss)
        self.log("test_accuracy", accuracy)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=10, factor=0.1
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "valid_loss",
        }
```

<!-- #region id="ju9vFxseZHCK" -->
## Trainer
<!-- #endregion -->

```python id="WJyXo8psZSIz"
args.DEBUG_MODE = True
args.DEBUG_LOAD = 10000
train_data = ML25Dataset(args, split='train')
val_data = ML25Dataset(args, split='val')

train_loader = DataLoader(
    train_data,
    batch_size=args.BATCH_SIZE,
    num_workers=2,
    shuffle=True,
)
val_loader = DataLoader(
    val_data,
    batch_size=args.BATCH_SIZE,
    num_workers=2,
    shuffle=False,
)

model = BERT4Rec(
    args, vocab_size=len(train_data.mapping) + 2)

logger = TensorBoardLogger(
    save_dir=args.LOG_DIR,
)

checkpoint_callback = ModelCheckpoint(
    monitor="valid_loss",
    mode="min",
    dirpath=args.MODEL_DIR,
    filename="recommender",
)

trainer = pl.Trainer(
    max_epochs=args.EPOCHS,
    gpus=1,
    logger=logger,
    callbacks=[checkpoint_callback],
)
trainer.fit(model, train_loader, val_loader)

result_val = trainer.test(test_dataloaders=val_loader)

output_json = {
    "val_loss": result_val[0]["test_loss"],
    "best_model_path": checkpoint_callback.best_model_path,
}

print(output_json)
```

<!-- #region id="V7u3_SyPZHrn" -->
## Inference
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 206} id="v_kL-604D95Y" executionInfo={"status": "ok", "timestamp": 1632575167340, "user_tz": -330, "elapsed": 1978, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="0c61a357-c8b8-4905-a889-8b87e2260697"
movies_path = "/content/ml-25m/movies.csv"
movies = pd.read_csv(movies_path)
movies.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="IwxX14ivC3Fe" executionInfo={"status": "ok", "timestamp": 1632575262464, "user_tz": -330, "elapsed": 1863, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="3644ce8a-bc11-4781-9de4-4e398b0660d2"
args.DEBUG_MODE = True
args.DEBUG_LOAD = 10000
data = ML25Dataset(args, split='train')

random.seed(args.SEED)
random.sample(list(data.grp_by.groups), k=2)
```

```python id="fOAsjdSSEqDY"
model = BERT4Rec(args, vocab_size=len(data.mapping) + 2)
model.eval()

model_path = "/content/recommender_models/recommender.ckpt"
model.load_state_dict(torch.load(model_path)["state_dict"])
movie_to_idx = {a: data.mapping[b] for a, b in zip(movies.title.tolist(), movies.movieId.tolist()) if b in data.mapping}
idx_to_movie = {v: k for k, v in movie_to_idx.items()}
```

```python id="mmc36T7q5OpM"
def predict(list_movies, model, movie_to_idx, idx_to_movie):
    ids = [args.PAD] * (120 - len(list_movies) - 1) + [movie_to_idx[a] for a in list_movies] + [args.MASK]
    src = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        prediction = model(src)
    masked_pred = prediction[0, -1].numpy()
    sorted_predicted_ids = np.argsort(masked_pred).tolist()[::-1]
    sorted_predicted_ids = [a for a in sorted_predicted_ids if a not in ids]
    return [idx_to_movie[a] for a in sorted_predicted_ids[:30] if a in idx_to_movie]
```

```python colab={"base_uri": "https://localhost:8080/"} id="FLzE_aWE6JTQ" executionInfo={"status": "ok", "timestamp": 1632575326058, "user_tz": -330, "elapsed": 24, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="c517192c-5914-40bf-caa9-8cc25c1493e0"
list_movies = ["Harry Potter and the Sorcerer's Stone (a.k.a. Harry Potter and the Philosopher's Stone) (2001)",
               "Harry Potter and the Chamber of Secrets (2002)",
               "Harry Potter and the Prisoner of Azkaban (2004)",
               "Harry Potter and the Goblet of Fire (2005)"]

top_movie = predict(list_movies, model, movie_to_idx, idx_to_movie)
top_movie
```

```python colab={"base_uri": "https://localhost:8080/"} id="CbSlRaGH6JQm" executionInfo={"status": "ok", "timestamp": 1632575352677, "user_tz": -330, "elapsed": 1238, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="3d65c863-d065-4e9e-92eb-e767c53a7938"
list_movies = ["Black Panther (2017)",
               "Avengers, The (2012)",
               "Avengers: Infinity War - Part I (2018)",
               "Logan (2017)",
               "Spider-Man (2002)",
               "Spider-Man 3 (2007)"]

top_movie = predict(list_movies, model, movie_to_idx, idx_to_movie)
top_movie
```

```python colab={"base_uri": "https://localhost:8080/"} id="yHmsGbxr6JNN" executionInfo={"status": "ok", "timestamp": 1632575382501, "user_tz": -330, "elapsed": 35, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="ed4c67de-db0e-48ac-8789-f4e4acb80a49"
list_movies = ["Toy Story 3 (2010)",
               "Finding Nemo (2003)",
               "Ratatouille (2007)",
               "The Lego Movie (2014)",
               "Ghostbusters (a.k.a. Ghost Busters) (1984)"]
top_movie = predict(list_movies, model, movie_to_idx, idx_to_movie)
top_movie
```

<!-- #region id="UljgDDKUZIYw" -->
## Unit Testing
<!-- #endregion -->

```python id="w_t_4MbCJ9Oh"
import unittest
from numpy.testing import assert_array_equal
```

```python id="ezKBo6Q0J-gk"
class TestUtils(unittest.TestCase):
    def testColMapping(self):
        "test the column mapping function"
        df = pd.DataFrame(
            {'uid': [1,2,3,4],
             'sid': [1,3,5,7]}
        )
        df, _, _ = map_column(df, col_name='sid')
        assert_array_equal(df.sid_mapped.values,
                           [2, 3, 4, 5])
        
    def testSplit(self):
        "test the train/test/val split"
        SEED = 42
        df = pd.DataFrame(
            {'uid': list(np.arange(50)),
                'sid': list(np.arange(50))}
        )
        context = get_context(df, split='train', context_size=5, seed=SEED)
        assert_array_equal(context.sid.values,
                           [12, 13, 14, 15, 16])
        
    def testArrayPadding(self):
        "test array padding function"
        pad_output_1 = pad_arr(np.array([[1,2,3],[7,8,9]]), expected_size=5)
        pad_output_2 = pad_arr(np.array([[1,2,3]]), expected_size=3)
        assert_array_equal(pad_output_1,
                           [[1, 2, 3],
                            [1, 2, 3],
                            [1, 2, 3],
                            [1, 2, 3],
                            [7, 8, 9]])
        assert_array_equal(pad_output_2,
                           [[1, 2, 3],
                            [1, 2, 3],
                            [1, 2, 3]])
        
    def testListPadding(self):
        "test list padding function"
        pad_output_1 = pad_list([1,2,3], history_size=5, pad_val=0, mode='left')
        pad_output_2 = pad_list([1,2,3], history_size=6, pad_val=1, mode='right')
        assert_array_equal(pad_output_1,
                           [0, 0, 1, 2, 3])
        assert_array_equal(pad_output_2,
                           [1, 2, 3, 1, 1, 1])
```

```python id="9TfTrrQ19Q1F"
class TestML25Dataset(unittest.TestCase):
    def testRecordsCount(self):
        train_data = ML25Dataset(args, split='train')
        self.assertEqual(len(train_data), 4)
```

```python id="XJbEQG5LiXi2"
class TestModelBERT4Rec(unittest.TestCase):
    def testBERT4Rec(self):
        n_items = 1000
        recommender = BERT4Rec(args, vocab_size=1000)
        src_items = torch.randint(low=0, high=n_items, size=(32, 30))
        src_items[:, 0] = 1
        trg_out = torch.randint(low=0, high=n_items, size=(32, 30))
        out = recommender(src_items)
        loss = recommender.training_step((src_items, trg_out), batch_idx=1)
        self.assertEqual(out.shape, torch.Size([32, 30, 1000]))
        self.assertIsInstance(loss, torch.Tensor)
        self.assertFalse(torch.isnan(loss).any())
        self.assertEqual(loss.size(),torch.Size([]))
```

```python id="lVmtbhoeWE3K"
class TestModelUtils(unittest.TestCase):
    def testMaskedAccuracy(self):
        "test the masked accuracy"
        output1 = masked_accuracy(torch.Tensor([[0,1,1,0]]),
                                torch.Tensor([[0,1,1,1]]),
                                torch.tensor([1,1,1,1], dtype=torch.bool))

        output2 = masked_accuracy(torch.Tensor([[0,1,1,0]]),
                                torch.Tensor([[0,1,1,1]]),
                                torch.tensor([1,0,0,1], dtype=torch.bool))

        self.assertEqual(output1, torch.tensor(0.75, dtype=torch.float64))
        self.assertEqual(output2, torch.tensor(0.5, dtype=torch.float64))

    def testMaskedCrossEntropy(self):
        input = [[1.1049, 1.5729, 1.4864],
        [-1.8321, -0.3137, -0.3257]]
        target = [0,2]

        output1 = masked_ce(torch.tensor(input),
                            torch.tensor(target),
                            torch.tensor([1,0], dtype=torch.bool))

        output2 = masked_ce(torch.tensor(input), 
                            torch.tensor(target),
                            torch.tensor([1,1], dtype=torch.bool))
        
        assert_array_equal(output1.numpy().round(4),
                           np.array(1.4015, dtype=np.float32))
        assert_array_equal(output2.numpy().round(4),
                           np.array(1.1026, dtype=np.float32))
        
    def testMaskList(self):
        args.SEED = 42
        assert_array_equal(mask_list([1,2,3,4,5,6,7,8]),
                           [1,2,3,4,5,6,1,8])
        args.SEED = 40
        assert_array_equal(mask_list([1,2,3,4,5,6,7,8]),
                           [1,1,3,4,1,6,7,8])

    def testMaskListLastElement(self):
        args.SEED = 42
        output1 = mask_last_elements_list([1,2,3,4,5,6,7,8], val_context_size=5)
        output2 = mask_last_elements_list([1,2,3,4,5,6,7,8], val_context_size=3)
        assert_array_equal(output1, [1,2,3,1,5,6,7,1])
        assert_array_equal(output2, [1,2,3,4,5,1,7,8])
```

```python id="VbqpEHhO-94V"
unittest.main(argv=[''], verbosity=2, exit=False)
```
