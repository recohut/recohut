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

<!-- #region id="uDw7iV-2ynqJ" -->
# SASRec on ML-1m in PaddlePaddle
<!-- #endregion -->

<!-- #region id="UjLxWVPkKyhF" -->
## Setup
<!-- #endregion -->

<!-- #region id="vIWlbC_jKzeL" -->
### Installations
<!-- #endregion -->

```python id="S0fgO4r4w7nD"
!pip install -q paddlepaddle
```

<!-- #region id="v89X-JBeK1cl" -->
### Downloads
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="x4f-Rv89woy0" executionInfo={"status": "ok", "timestamp": 1633207647131, "user_tz": -330, "elapsed": 454, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="b9d9191b-e730-467b-ee7b-0ab95b1e6261"
!wget -q --show-progress https://github.com/paddorch/SASRec.paddle/raw/main/data/preprocessed/ml-1m.txt
```

<!-- #region id="ABOGAwyyK26M" -->
### Imports
<!-- #endregion -->

```python id="ZQOjPPXyK4C_"
import os
import sys
import copy
import random
import numpy as np
from multiprocessing import Process, Queue
from collections import defaultdict

import random
from tqdm import tqdm

import paddle
import paddle.nn as nn
from paddle import optimizer
import paddle.nn.functional as F

import argparse
```

<!-- #region id="jBWlW-NsLNYP" -->
### Params
<!-- #endregion -->

```python id="xmkjSjaJLPcN"
set_seed(42)

parser = argparse.ArgumentParser(description='SASRec training')
# data
parser.add_argument('--dataset_path', metavar='DIR',
                    default='ml-1m.txt')
# learning
learn = parser.add_argument_group('Learning options')
learn.add_argument('--lr', type=float, default=0.001, help='initial learning rate [default: 0.01]')
learn.add_argument('--epochs', type=int, default=100, help='number of epochs for train')
learn.add_argument('--batch_size', type=int, default=128, help='batch size for training')
learn.add_argument('--optimizer', default='AdamW',
                   help='Type of optimizer. Adagrad|Adam|AdamW are supported [default: Adagrad]')
# model
model_cfg = parser.add_argument_group('Model options')
model_cfg.add_argument('--hidden_units', type=int, default=50,
                       help='hidden size of LSTM [default: 300]')
model_cfg.add_argument('--maxlen', type=int, default=200,
                       help='hidden size of LSTM [default: 300]')
model_cfg.add_argument('--dropout', type=float, default=0.2, help='the probability for dropout')
model_cfg.add_argument('--l2_emb', type=float, default=0.0, help='penalty term coefficient')
model_cfg.add_argument('--num_blocks', type=int, default=2,
                       help='d_a size [default: 150]')
model_cfg.add_argument('--num_heads', type=int, default=1,
                       help='row size of sentence embedding [default: 30]')
# device
device = parser.add_argument_group('Device options')
device.add_argument('--num_workers', default=8, type=int, help='Number of workers used in data-loading')
device.add_argument('--cuda', action='store_true', default=True, help='enable the gpu')
device.add_argument('--device', type=int, default=None)

# experiment options
experiment = parser.add_argument_group('Experiment options')
experiment.add_argument('--continue_from', default='', help='Continue from checkpoint model')
experiment.add_argument('--checkpoint', dest='checkpoint', default=True, action='store_true',
                        help='Enables checkpoint saving of model')
experiment.add_argument('--checkpoint_per_batch', default=10000, type=int,
                        help='Save checkpoint per batch. 0 means never save [default: 10000]')
experiment.add_argument('--save_folder', default='output/',
                        help='Location to save epoch models, training configurations and results.')
experiment.add_argument('--log_config', default=True, action='store_true', help='Store experiment configuration')
experiment.add_argument('--log_result', default=True, action='store_true', help='Store experiment result')
experiment.add_argument('--log_interval', type=int, default=30,
                        help='how many steps to wait before logging training status')
experiment.add_argument('--val_interval', type=int, default=800,
                        help='how many steps to wait before vaidation')
experiment.add_argument('--val_start_batch', type=int, default=8000,
                        help='how many steps to wait before vaidation')
experiment.add_argument('--save_interval', type=int, default=20,
                        help='how many epochs to wait before saving')
experiment.add_argument('--test', type=bool, default=False, help='test only')
experiment.add_argument('--model_path', type=str, default=False, help='test only')
```

<!-- #region id="n6BMp5aLwsFD" -->
## Data
<!-- #endregion -->

```python id="cDS6GBujw0OU"
# sampler for batch generation
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue=None, SEED=42):
    def sample():
        user = np.random.randint(1, usernum + 1)
        while len(user_train[user]) <= 1:
            user = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[user][-1]
        idx = maxlen - 1

        ts = set(user_train[user])
        for i in reversed(user_train[user][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1: break

        return (user, seq, pos, neg)  # TODO

    if result_queue is None:
        np.random.seed(SEED)
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())
        return zip(*one_batch)
    else:
        np.random.seed(SEED)
        while True:
            one_batch = []
            for i in range(batch_size):
                one_batch.append(sample())

            result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        self.n_workers = n_workers
        if self.n_workers != 0:
            self.result_queue = Queue(maxsize=n_workers * 10)
            self.processors = []
            for i in range(n_workers):
                self.processors.append(
                    Process(target=sample_function, args=(User,
                                                          usernum,
                                                          itemnum,
                                                          batch_size,
                                                          maxlen,
                                                          self.result_queue,
                                                          np.random.randint(2e9)
                                                          )))
                self.processors[-1].daemon = True
                self.processors[-1].start()
        else:
            self.User = User
            self.usernum = usernum
            self.itemnum = itemnum
            self.batch_size = batch_size
            self.maxlen = maxlen

    def next_batch(self):
        if self.n_workers != 0:
            return self.result_queue.get()
        return sample_function(self.User,
                               self.usernum,
                               self.itemnum,
                               self.batch_size,
                               self.maxlen,
                               None,
                               np.random.randint(2e9))

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()
```

<!-- #region id="P1EgnHMlxVh-" -->
## Utils
<!-- #endregion -->

```python id="YVTtmt4ww0kj"
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


# train/val/test data generation
def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    with open(fname, 'r') as f:
        for line in f:
            u, i = line.rstrip().split(' ')
            u = int(u)
            i = int(i)
            usernum = max(u, usernum)
            itemnum = max(i, itemnum)
            User[u].append(i)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    return [user_train, user_valid, user_test, usernum, itemnum]
```

<!-- #region id="9Lt_z2cZxXY4" -->
## Model
<!-- #endregion -->

```python id="JiURfKinw6FR"
class SASRec(paddle.nn.Layer):
    def __init__(self, item_num, args):
        super(SASRec, self).__init__()
        self.item_emb = nn.Embedding(item_num + 1, args.hidden_units)  # [pad] is 0
        self.pos_emb = nn.Embedding(args.maxlen, args.hidden_units)
        self.emb_dropout = paddle.nn.Dropout(p=args.dropout)

        self.subsequent_mask = (paddle.triu(paddle.ones((args.maxlen, args.maxlen))) == 0)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=args.hidden_units,
                                                        nhead=args.num_heads,
                                                        dim_feedforward=args.hidden_units,
                                                        dropout=args.dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=args.num_blocks)

    def position_encoding(self, seqs):
        seqs_embed = self.item_emb(seqs)  # (batch_size, max_len, embed_size)
        positions = np.tile(np.array(range(seqs.shape[1])), [seqs.shape[0], 1])
        position_embed = self.pos_emb(paddle.to_tensor(positions, dtype='int64'))
        return self.emb_dropout(seqs_embed + position_embed)

    def forward(self, log_seqs, pos_seqs, neg_seqs):
        # all input seqs: (batch_size, seq_len)
        seqs_embed = self.position_encoding(log_seqs)  # (batch_size, seq_len, embed_size)
        log_feats = self.encoder(seqs_embed, self.subsequent_mask)  # (batch_size, seq_len, embed_size)

        pos_embed = self.item_emb(pos_seqs)  # (batch_size, seq_len, embed_size)
        neg_embed = self.item_emb(neg_seqs)

        pos_logits = (log_feats * pos_embed).sum(axis=-1)
        neg_logits = (log_feats * neg_embed).sum(axis=-1)

        return pos_logits, neg_logits

    def predict(self, log_seqs, item_indices):  # for inference
        seqs = self.position_encoding(log_seqs)
        log_feats = self.encoder(seqs, self.subsequent_mask)  # (batch_size, seq_len, embed_size)

        final_feat = log_feats[:, -1, :]
        item_embs = self.item_emb(paddle.to_tensor(item_indices, dtype='int64'))

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        return logits
```

<!-- #region id="18NJ5h3mxQfD" -->
## Eval
<!-- #endregion -->

```python id="-h0TdKVbxZVK"
def evaluate(dataset, model, epoch_train, batch_train, args, is_val=True):
    model.eval()
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    before = train
    now = valid if is_val else test

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)

    for u in tqdm(users):
        if len(before[u]) < 1 or len(now[u]) < 1:
            continue
        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        if not is_val:
            seq[idx] = valid[u][0]
            idx -= 1
        for i in reversed(before[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        rated = set(before[u])
        rated.add(0)
        item_idx = [now[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated:
                t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)
        predictions = -model.predict(*[paddle.to_tensor(l) for l in [[seq], item_idx]])
        predictions = predictions[0]  # - for 1st argsort DESC

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1
        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1

    NDCG /= valid_user
    HT /= valid_user

    model.train()
    print('\nEpoch {} Evaluation - NDCG: {:.4f}  HIT@10: {:.4f}'.format(epoch_train,  NDCG, HT))
    if args.log_result and is_val:
        with open(os.path.join(args.save_folder, 'result.csv'), 'a') as r:
            r.write('\n{:d},{:d},{:.4f},{:.4f}'.format(epoch_train, batch_train, NDCG, HT))
    return (HT, NDCG)
```

<!-- #region id="bCjflfkfxbLJ" -->
## Train
<!-- #endregion -->

```python id="Kq7KJBoAxc5L"
class MyBCEWithLogitLoss(paddle.nn.Layer):
    def __init__(self):
        super(MyBCEWithLogitLoss, self).__init__()

    def forward(self, pos_logits, neg_logits, labels):
        return paddle.sum(
            - paddle.log(F.sigmoid(pos_logits) + 1e-24) * labels -
            paddle.log(1 - F.sigmoid(neg_logits) + 1e-24) * labels,
            axis=(0, 1)
        ) / paddle.sum(labels, axis=(0, 1))


def train(sampler, model, args, num_batch, dataset):
    clip = None
    # optimization scheme
    if args.optimizer == 'Adam':
        optim = optimizer.Adam(parameters=model.parameters(), learning_rate=args.lr, grad_clip=clip)
    elif args.optimizer == 'Adagrad':
        optim = optimizer.Adagrad(parameters=model.parameters(), learning_rate=args.lr, grad_clip=clip)
    elif args.optimizer == 'AdamW':
        optim = optimizer.AdamW(parameters=model.parameters(), learning_rate=args.lr, grad_clip=clip)

    # loss
    # criterion = nn.BCEWithLogitsLoss()
    criterion = MyBCEWithLogitLoss()

    # continue training from checkpoint model
    if args.continue_from:
        print("=> loading checkpoint from '{}'".format(args.continue_from))
        assert os.path.isfile(args.continue_from), "=> no checkpoint found at '{}'".format(args.continue_from)
        checkpoint = paddle.load(args.continue_from)
        start_epoch = checkpoint['epoch']
        best_pair = checkpoint.get('best_pair', None)
        model.set_state_dict(checkpoint['state_dict'])
    else:
        start_epoch = 1
        best_pair = None

    model.train()

    tot_batch = 0
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_loss = 0
        for i_batch in range(num_batch):
            tot_batch += 1
            u, seq, pos, neg = sampler.next_batch()  # tuples to ndarray
            u, seq, pos, neg = paddle.to_tensor(u, dtype='int64'), paddle.to_tensor(seq,
                                                                                    dtype='int64'), paddle.to_tensor(
                pos), paddle.to_tensor(neg)
            pos_logits, neg_logits = model(seq, pos, neg)  # ()

            targets = (pos != 0).astype(dtype='int32')
            # targets = targets.reshape((args.batch_size*args.maxlen, -1))
            loss = criterion(pos_logits, neg_logits, targets)
            for param in model.item_emb.parameters():
                loss += args.l2_emb * paddle.norm(param)
            loss.backward()
            epoch_loss += loss.numpy()[0]
            optim.step()
            optim.clear_grad()

            # validation
            if tot_batch >= args.val_start_batch and tot_batch % args.val_interval == 0 and i_batch != 0:
                valid_pair = evaluate(dataset, model, epoch, i_batch, args, is_val=True)
                if best_pair is None or valid_pair > best_pair:
                    best_pair = valid_pair
                    file_path = '%s/SASRec_best.pth.tar' % (args.save_folder)
                    print("=> found better validated model, saving to %s" % file_path)
                    save_checkpoint(model,
                                    {'epoch': epoch,
                                     'optimizer': optim.state_dict(),
                                     'best_pair': best_pair},
                                    file_path)

        print('Epoch {:3} - loss: {:.4f}  lr: {:.5f}'.format(epoch,
                                                             epoch_loss / num_batch,
                                                             optim._learning_rate,
                                                             ))

        if args.checkpoint and epoch % args.save_interval == 0:
            file_path = '%s/SASRec_epoch_%d.pth.tar' % (args.save_folder, epoch)
            print("\r=> saving checkpoint model to %s" % file_path)
            save_checkpoint(model, {'epoch': epoch,
                                    'optimizer': optim.state_dict(),
                                    'best_pair': best_pair},
                            file_path)


def save_checkpoint(model, state, filename):
    state['state_dict'] = model.state_dict()
    paddle.save(state, filename)
```

<!-- #region id="XqlsIzOqxgTk" -->
## Run
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="kceOYcKvxk3O" outputId="bb13e1c5-a0ed-4231-a096-a64ebf93e052"
def main():
    print(paddle.__version__)
    args = parser.parse_args(args={})

    # gpu
    if args.cuda and args.device:
        paddle.set_device(f"gpu:{args.device}")
    print(paddle.get_device())

    dataset = data_partition(args.dataset_path)

    [user_train, _, _, usernum, itemnum] = dataset
    num_batch = len(user_train) // args.batch_size  # tail? + ((len(user_train) % args.batch_size) != 0)
    print("batches / epoch:", num_batch)

    seq_len = 0.0
    for u in user_train:
        seq_len += len(user_train[u])
    print('\nAverage sequence length: %.2f' % (seq_len / len(user_train)))

    # make save folder
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    # configuration
    print("\nConfiguration:")
    for attr, value in sorted(args.__dict__.items()):
        print("\t{}:".format(attr.capitalize().replace('_', ' ')).ljust(25) + "{}".format(value))

    # log result
    if args.log_result:
        with open(os.path.join(args.save_folder, 'result.csv'), 'w') as r:
            r.write('{:s},{:s},{:s},{:s},{:s}'.format('epoch', 'batch', 'loss', 'acc', 'lr'))

    # model
    model = SASRec(itemnum, args)
    print(model)

    if not args.test:  # train
        # dataloader
        sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen,
                              n_workers=args.num_workers)
        train(sampler, model, args, num_batch, dataset)
        sampler.close()
    else:  # test
        print("=> loading weights from '{}'".format(args.model_path))
        assert os.path.isfile(args.model_path), "=> no checkpoint found at '{}'".format(args.model_path)
        checkpoint = paddle.load(args.model_path)
        model.set_state_dict(checkpoint['state_dict'])
        evaluate(dataset, model, checkpoint['epoch'], 0, args, is_val=False)


if __name__ == '__main__':
    main()
```

```python id="9F65suc6xxuO"
%%writefile train.sh
python run.py \
  --dataset_path=data/preprocessed/ml-1m.txt \
  --hidden_units=50
  --dropout=0.2
  --num_blocks=2
  --num_heads=1
  --device=0
```

```python id="07Ej6jTDyAzk"
%%writefile eval.sh
python run.py \
  --dataset_path=data/preprocessed/ml-1m.txt \
  --hidden_units=50 \
  --num_blocks=2 \
  --num_heads=1 \
  --device=0 \
  --test=True\
  --model_path=output/SASRec_epoch_420.pth.tar
```
