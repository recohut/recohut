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

<!-- #region id="CuUYKeCrxb-N" -->
# BST in PyTorch

> BST Model Implementation in PyTorch. Main purpose is to get familier with BST model, so only code is available upto trainer module. Inference and dataset runs will be added in future possibly.
<!-- #endregion -->

<!-- #region id="ayhIfDiPxPFY" -->
### Imports
<!-- #endregion -->

```python id="FhS247l1SymE"
import random
import numpy as np
import time

import torch
from torch import nn
```

<!-- #region id="ngbBIZikxNUx" -->
### Params
<!-- #endregion -->

```python id="N21knoI8SZTG"
%%writefile config_sample.py
config = {'item_embed': {
    'num_embeddings': 500,
    'embedding_dim': 32,
    'sparse': False,
    'padding_idx': -1,
},
    'trans': {
        'input_size': 32,
        'hidden_size': 16,
        'n_layers': 2,
        'n_heads': 4,
        'max_len': 5,
    },
    'context_features': [
            {'num_embeddings': 6, 'embedding_dim': 10, 'sparse': False, 'padding_idx': -1},
            {'num_embeddings': 4, 'embedding_dim': 10, 'sparse': False, 'padding_idx': -1},

        ],

    'cuda': False,
    'max_seq_len': 6,
}
```

<!-- #region id="-dqEg3nXxQY8" -->
## Utils
<!-- #endregion -->

```python id="A7gp8uLUSpDn"
def pad(seq, max_seq_len, pad_with=0):
    seq_len = len(seq)
    return [pad_with]*(max_seq_len - seq_len) + seq


def batch_fn(user_seq, context_features, batch_size, max_seq_len, shuffle=True):
    if shuffle:
        data = list(zip(user_seq, context_features))
        random.shuffle(data)
        user_seq, context_features = zip(*data)
    context_features = np.array(context_features).T
    for start_idx in range(0, len(user_seq) - batch_size + 1, batch_size):
        batch = user_seq[start_idx:start_idx + batch_size]
        context_batch = context_features[..., start_idx:start_idx + batch_size].tolist()
        batch = [seq[-max_seq_len:] for seq in batch]
        user_seq_batch = []
        for seq in batch:
            pseq = pad(seq, max_seq_len)
            user_seq_batch += [pseq]
        yield user_seq_batch, context_batch
```

```python id="VuehNyDYSz2d"
class GradientClipping:
    def __init__(self, clip_value):
        self.epoch_grads = []
        self.total_grads = []
        self.clip = clip_value

    def track_grads(self, x, grad_input, grad_output):
        self.epoch_grads.append(grad_input[0].norm().cpu().data.numpy())

    def register_hook(self, encoder):
        encoder.register_backward_hook(self.track_grads)

    def gradient_mean(self):
        return np.mean(self.epoch_grads)

    def gradient_std(self):
        return np.std(self.epoch_grads)

    def reset_gradients(self):
        self.total_grads.append(self.epoch_grads)
        self.epoch_grads = []

    def update_clip_value(self):
        self.clip = self.gradient_mean() + self.gradient_std()

    def update_clip_value_total(self):
        grads = [y for x in self.total_grads.append(self.epoch_grads) for y in x]
        self.clip = np.mean(grads)
```

<!-- #region id="mJs3EOt0xVLu" -->
## Model
<!-- #endregion -->

```python id="xyH62_g6Szy5"
class FF(nn.Module):
    """
    Feed-forward in a transformer layer.
    """
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lin_1 = nn.Linear(input_size, hidden_size)
        self.lin_2 = nn.Linear(hidden_size, input_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        output = self.lin_2(self.relu(self.lin_1(x)))
        return output


class MultiHeadAttention(nn.Module):
    """
    Multi-head Attention block in a transformer layer.
    """
    def __init__(self, att_dim, n_heads):
        super().__init__()
        # Check for compatible  #Attention Heads
        self.n_heads = n_heads
        # Check compatibility for input size and #attention heads.
        assert att_dim % self.n_heads == 0
        self.att_size = int(att_dim / n_heads)

        # Query, Key, Value
        self._query = nn.Linear(att_dim, att_dim, bias=False)
        self._key = nn.Linear(att_dim, att_dim, bias=False)
        self._value = nn.Linear(att_dim, att_dim, bias=False)

        # Attention Block
        self.dense = nn.Linear(att_dim, att_dim, bias=False)
        self.activation = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, q, k, v, mask=None):
        scale_factor = torch.sqrt(torch.FloatTensor([self.n_heads])).item()
        batch_size = q.size(0)

        # To Multiple Attention Heads
        _query = self._query(q).view(batch_size, -1, self.n_heads, self.att_size).transpose(1, 2)
        _key = self._key(k).view(batch_size, -1, self.n_heads, self.att_size).transpose(1, 2)
        _value = self._value(v).view(batch_size, -1, self.n_heads, self.att_size).transpose(1, 2)

        # Scaled dot-product Attention score
        score = torch.matmul(_query, _key.transpose(-2, -1)) / scale_factor
        # Mask applied.
        if mask is not None:
            mask = mask.unsqueeze(1)
            score = score.masked_fill(mask == 0, -1e9)
        # Softmax on Score
        score = self.activation(score)
        z = torch.matmul(self.dropout(score), _value)

        # To fully-connected layer
        z = z.transpose(1, 2).reshape(batch_size, -1, self.att_size * self.n_heads)
        return self.dense(z)


class EncoderCell(nn.Module):
    """
    Encoder Cell contains MultiHeadAttention > Add & LayerNorm1 >
    Feed Forward > Add & LayerNorm2
    """
    def __init__(self, input_size, hidden_size, n_heads):
        super().__init__()
        # Attention Block
        self.mh_attention = MultiHeadAttention(input_size, n_heads)
        self.lnorm_1 = nn.LayerNorm(input_size)
        # Feed forward block
        self.ff = FF(input_size, hidden_size)
        self.lnorm_2 = nn.LayerNorm(input_size)
        # Dropout
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, mask=None):
        attention_out = self.mh_attention(x, x, x, mask)
        attention_out = self.lnorm_1(self.dropout(attention_out) + x)

        ff_attention = self.ff(attention_out)
        return self.lnorm_2(self.dropout(ff_attention) + attention_out)


class Encoder(nn.Module):
    """
    Encoder Block with n stacked encoder cells.
    """
    def __init__(self, input_size, hidden_size, n_layers, n_heads):
        super().__init__()
        # Stack of encoder-cells n_layers high
        self.stack = nn.ModuleList()
        # Building encoder stack
        for layer in range(n_layers):
            self.stack.append(EncoderCell(input_size, hidden_size, n_heads))
        # Dropout layer
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, mask=None):
        for cell in self.stack:
            x = cell(self.dropout(x), mask)
        return x
```

```python id="Dy2RRdd3TFoY"
class BSTransformer(nn.Module):
    """
    Behaviour Sequence Transformer with dynamic context embeddings
    and sinusoidal pos-encoding.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.item_embed = nn.Embedding(num_embeddings=config['item_embed']['num_embeddings'],
                                       embedding_dim=config['item_embed']['embedding_dim'],
                                       sparse=config['item_embed']['sparse'],
                                       padding_idx=config['item_embed']['padding_idx'])

        self.pos_embedding = self.pos_embedding_sinusoidal(config['max_seq_len'], 
                                                           config['item_embed']['embedding_dim'],
                                                           config['cuda'])
        self.context_embeddings = nn.ModuleList([nn.Embedding(num_embeddings=feat['num_embeddings'],
                                                              embedding_dim=feat['embedding_dim'],
                                                              sparse=feat['sparse'],
                                                              padding_idx=feat['padding_idx'])
                                                 for feat in config['context_features']])

        self.encoder = Encoder(input_size=config['trans']['input_size'],
                               hidden_size=config['trans']['hidden_size'],
                               n_layers=config['trans']['n_layers'],
                               n_heads=config['trans']['n_heads'])

        mlp_input_size = config['trans']['input_size'] + sum(
            [feat['embedding_dim'] for feat in config['context_features']])

        self.mlp = nn.Sequential(nn.Linear(mlp_input_size, 1024),
                                 nn.LeakyReLU(),
                                 nn.Linear(1024, config['item_embed']['num_embeddings'])
                                 )

        for param in self.parameters():
            if param.dim() > 1 and config['init_method'] == 'xavier':
                torch.nn.init.xavier_uniform_(param)
            if param.dim() > 1 and config['init_method'] == 'kaiming':
                torch.nn.init.kaiming_uniform_(param)
        print(f"Parameters initialised using {config['init_method']} initialisation!")

    def forward(self, x, context):
        targets = x[..., -1:].long()
        enc_mask = self.get_mask(x)
        item_embed = self.item_embed(x.long()) * np.sqrt(self.config['item_embed']['embedding_dim'])
        agg_encoding = torch.mean(self.encoder(item_embed + self.pos_embedding[:x.size(1), :], mask=enc_mask), dim=1)
        context_embs = torch.tensor([]).to(x.device)
        for emb, feat in zip(self.context_embeddings, context):
            context_embs = torch.cat([context_embs, emb(feat)], dim=1)
        output = self.mlp(torch.cat([agg_encoding, context_embs], dim=1))
        return output, targets

    def get_mask(self, x):
        seq_len = x.size(1)
        mask = (x != 0).unsqueeze(1).byte()
        triu = (np.triu(np.ones([1, seq_len, seq_len]), k=1) == 0).astype('uint8')
        if self.config['cuda']:
            dtype = torch.cuda.ByteTensor
        else:
            dtype = torch.ByteTensor
        return dtype(triu) & dtype(mask)

    @staticmethod
    def pos_embedding_sinusoidal(max_seq_len, embedding_dim, is_cuda):
        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(max_seq_len, dtype=torch.float).unsqueeze(
            1
        ) * emb.unsqueeze(0)
        emb = torch.stack((torch.sin(emb), torch.cos(emb)), dim=0).view(
            max_seq_len, -1).t().contiguous().view(max_seq_len, -1)
        if embedding_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(max_seq_len, 1)], dim=1)
        if is_cuda:
            return emb.cuda()
        return emb
```

<!-- #region id="iCHbf9v-xWnp" -->
## Trainer
<!-- #endregion -->

```python id="uVepRYk5SzxL"
class Trainer:
    def __init__(self, config, loss_fn, batch_fn, device, grad_clipping=True):
        self.config = config
        self.bst = self.init_bst_encoder()
        self.optimizer = torch.optim.AdamW(self.bst.parameters(), lr=config['lr'])
        self.loss_fn = loss_fn
        self.batch_fn = batch_fn
        self.training_start = None
        self.device = device
        self.train_loss = 0
        self.best_loss = np.inf
        self.batch_num = 0
        self.epoch_num = 0
        self.scheduler = None
        try:
            if grad_clipping:
                self.clipper = GradientClipping(config['clip_value'])
                self.clipper.register_hook(self.bst)
        except KeyError:
            print("Gradient Clipping not available! Pass clip value in config!")

    def epoch(self, user_seq, context_features, batch_size, max_seq_len):
        self.training_start = time.time()
        self.bst.train()
        self.train_loss = 0

        # Iterate through batch.
        for user_seq_batch, context_batch in self.batch_fn(user_seq, context_features, batch_size, max_seq_len):
            pred, target = self.bst(torch.tensor(user_seq_batch).to(self.device),
                                    torch.tensor(context_batch).to(self.device))
            loss = self.loss_fn(pred.view(-1, pred.size(-1)), target.view(-1))

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.bst.parameters(),
                                           self.clipper.clip)
            self.optimizer.step()
            self.train_loss += loss.data
            self.batch_num += 1

            self.scheduler.step()  # set_to_none=True
        self.train_loss = self.train_loss.cpu().data.numpy() / self.batch_num

        # Log
        print(f'Loss after {self.batch_num * batch_size} sequences: '
              f'{self.train_loss}'
              f'\nTraining time: {time.time() - self.training_start}')

        # Save best weights
        if self.train_loss < self.best_loss:
            self.save_state('best', save_grads=False)
            self.best_loss = self.train_loss

    def init_bst_encoder(self):
        # Init Behaviour Seq Transformer model.
        bst = BSTransformer(self.config)
        bst = bst.cuda() if self.config['cuda'] else bst
        return bst

    def save_state(self, path, save_grads=False):
        # Save state to path.
        torch.save(self.bst.state_dict(), path)
        if save_grads:
            np.save(f'{path}_grads', self.clipper.total_grads)

    def set_lr_scheduler(self, milestones, gamma, last_epoch):
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones,
                                                              gamma=gamma, last_epoch=last_epoch)
```
