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

```python id="ehMex-aqyDNF" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1631384536630, "user_tz": -330, "elapsed": 16, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="f6b7612f-88e3-49bb-f5b9-dfe4aeca0fb9"
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

```python id="V-iFgtQizQfH" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1631384541129, "user_tz": -330, "elapsed": 727, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="93cb8c80-8f09-42b8-b1a8-769856832452"
!git status
```

```python id="9lRfcKvOGNcJ"
/content/
```

```python id="ZfU-gZ86yDNN" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1631384639841, "user_tz": -330, "elapsed": 1648, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="d2dffd46-1917-4729-ead1-24cb6c1b2bfe"
!git add .
!git commit -m 'commit'
!git push origin "{branch}"
```

```python id="001B38bSC_KY"
!dvc pull data/bronze/ml-1m/ratings.dat
```

```python id="vT9-NFmqD-MT"
!dvc repro
```

```python id="3E4R89nU5X3N"
!dvc push
```

```python colab={"base_uri": "https://localhost:8080/"} id="7q1pU49x5lat" executionInfo={"status": "ok", "timestamp": 1631384359961, "user_tz": -330, "elapsed": 1016, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="86df2ae1-133e-4dbc-f0dc-be37aa171b50"
%%writefile ./src/data_loading.py
from abc import *
import random
import torch
import pickle
import random
import os
from pathlib import Path
import torch.utils.data as data_utils


class AbstractDataloader(metaclass=ABCMeta):
    def __init__(self, args, dataset):
        self.args = args
        self.rng = random.Random()
        self.train = dataset['train']
        self.val = dataset['val']
        self.test = dataset['test']
        self.smap = dataset['smap']
        self.item_count = len(self.smap)

    @classmethod
    @abstractmethod
    def code(cls):
        pass

    @abstractmethod
    def get_pytorch_dataloaders(self):
        pass


class BERTDataloader():
    def __init__(self, args, dataset, seen_samples, val_negative_samples):
        self.args = args
        self.rng = random.Random()
        self.train = dataset['train']
        self.val = dataset['val']
        self.test = dataset['test']
        self.smap = dataset['smap']
        self.item_count = len(self.smap)

        self.max_len = args.bert_max_len
        self.mask_prob = args.bert_mask_prob
        self.max_predictions = args.bert_max_predictions
        self.sliding_size = args.sliding_window_size
        self.CLOZE_MASK_TOKEN = self.item_count + 1

        self.seen_samples = seen_samples
        self.val_negative_samples = val_negative_samples
        self.test_negative_samples = val_negative_samples

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


def dataloader_factory(args, dataset, seen_samples, val_negative_samples):
    if args.model_code == 'bert':
        dataloader = BERTDataloader(args, dataset, seen_samples, val_negative_samples)
    train, val, test = dataloader.get_pytorch_dataloaders()
    return train, val, test


if __name__ == '__main__':
    PREP_DATASET_ROOT_FOLDER = 'data/silver'
    FEATURES_ROOT_FOLDER = 'data/gold'
    prep_filepath = Path(os.path.join(PREP_DATASET_ROOT_FOLDER, 'ml-1m/dataset.pkl'))
    dataset = pickle.load(prep_filepath.open('rb'))
    ns_val_filepath = Path(os.path.join(FEATURES_ROOT_FOLDER, 'ml-1m/negative_samples/negative_samples_val.pkl'))
    seen_samples, val_negative_samples = pickle.load(ns_val_filepath.open('rb'))
    class Args:
        model_code = 'bert'
        sliding_window_size = 0.5
        bert_hidden_units = 64
        bert_dropout = 0.1
        bert_attn_dropout = 0.1
        bert_max_len = 200
        bert_mask_prob = 0.2
        bert_max_predictions = 40
        batch = 128
        train_batch_size = batch
        val_batch_size = batch
        test_batch_size = batch
        device = 'cpu'
        optimizer = 'AdamW'
        lr = 0.001
        weight_decay = 0.01
        enable_lr_schedule = True
        decay_step = 10000
        gamma = 1.
        enable_lr_warmup = False
        warmup_steps = 100
        num_epochs = 1
        metric_ks = [1, 5, 10]
        best_metric = 'NDCG@10'
        model_init_seed = 98765
        bert_num_blocks = 2
        bert_num_heads = 2
        bert_head_size = None
    args = Args()
    train_loader, val_loader, test_loader = dataloader_factory(args, dataset,
                                 seen_samples,
                                 val_negative_samples)
    dataloader_savepath = Path(os.path.join(FEATURES_ROOT_FOLDER, 'ml-1m/dataloaders'))
    if not dataloader_savepath.is_dir():
        dataloader_savepath.mkdir(parents=True)
    torch.save(train_loader, dataloader_savepath.joinpath('train_loader.pt'))
    torch.save(val_loader, dataloader_savepath.joinpath('val_loader.pt'))
    torch.save(test_loader, dataloader_savepath.joinpath('test_loader.pt'))
```

```python id="4GSf9WmayqgG" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1631384280816, "user_tz": -330, "elapsed": 1840, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="43233b68-96be-433f-e836-43a96cad9bc8"
!python ./src/data_loading.py
```

```python id="8pjcu2yOCmS6"
torch.save(a, './data/gold/ml-1m/dataloaders/train.pt')
```

```python colab={"base_uri": "https://localhost:8080/"} id="4NUKJgYsDG3l" executionInfo={"status": "ok", "timestamp": 1631383752923, "user_tz": -330, "elapsed": 710, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="eb9cf117-5e75-49ab-de78-e271c2d1a67e"
a.batch_size
```

```python colab={"base_uri": "https://localhost:8080/"} id="DoQl1bm9DJiC" executionInfo={"status": "ok", "timestamp": 1631383827884, "user_tz": -330, "elapsed": 1034, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="bebd5b47-6704-4121-a7e2-f59e8397e9d7"
dataiter = iter(a)
dataiter.next()   
```

```python colab={"base_uri": "https://localhost:8080/"} id="lSCRdFzwDcXo" executionInfo={"status": "ok", "timestamp": 1631383873985, "user_tz": -330, "elapsed": 1559, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="0f98cbcb-3b33-4e43-8d7d-825b6838143b"
aa = torch.load('./data/gold/ml-1m/dataloaders/train.pt')

dataiter = iter(aa)
dataiter.next()
```

```python colab={"base_uri": "https://localhost:8080/"} id="kMkIAJCN4i63" executionInfo={"status": "ok", "timestamp": 1631384387104, "user_tz": -330, "elapsed": 17312, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="eaa1757b-8048-4c46-f66b-6bb6bcd0fe07"
!dvc run -n data_loading \
          -d src/data_loading.py -d data/silver/ml-1m/dataset.pkl \
          -d data/gold/ml-1m/negative_samples/negative_samples_val.pkl \
          -o data/gold/ml-1m/dataloaders \
          python src/data_loading.py
```

```python colab={"base_uri": "https://localhost:8080/"} id="pM9lssEo541j" executionInfo={"status": "ok", "timestamp": 1631384494023, "user_tz": -330, "elapsed": 723, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="5594992e-8aed-4ffe-bb4d-161521a35540"
!git status -u
```

```python colab={"base_uri": "https://localhost:8080/"} id="pmIx9Y1F5_TD" executionInfo={"status": "ok", "timestamp": 1631364668470, "user_tz": -330, "elapsed": 4427, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="7b4fa21c-945e-4cc3-de15-1af95d065867"
!dvc status
```

```python id="tYuj8l5M6URQ"
!dvc commit
!dvc push
```

```python colab={"base_uri": "https://localhost:8080/"} id="prWgCZcT6W2_" executionInfo={"status": "ok", "timestamp": 1631384500491, "user_tz": -330, "elapsed": 1053, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="c5ab64a8-c9e6-4794-c90e-78a9d63f4f5a"
!git add .
!git commit -m 'commit'
!git push origin "{branch}"
```

```python id="wqnAUiFQ6nO3"

```
