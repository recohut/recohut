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

<!-- #region id="OwSaXMsA7tEN" -->
# TAGNN++ Session-based Recommendations on Diginetica dataset
<!-- #endregion -->

<!-- #region id="9P7DngMS7cIA" -->
## Setup
<!-- #endregion -->

```python id="oTkBob7DTs2L"
import torch
from torch import nn, optim
from torch.nn import Module, Parameter
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


from collections import Iterable
from tqdm.notebook import tqdm
import datetime
import math
import numpy as np
import pickle
import time
import sys

```

```python id="Uk_7AdvECZxr"
import warnings
warnings.filterwarnings('ignore')
```

<!-- #region id="LE5-b7OQ7aep" -->
## Dataset
<!-- #endregion -->

```python id="U4fNVWWSTgWa" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637777066024, "user_tz": -330, "elapsed": 1349, "user": {"displayName": "sparsh agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "00322518567794762549"}} outputId="9e311caf-6b57-41ba-f815-cba259020321"
!wget -q --show-progress https://github.com/sparsh-ai/stanza/raw/S969796/datasets/cikm16/raw/train.txt
!wget -q --show-progress https://github.com/sparsh-ai/stanza/raw/S969796/datasets/cikm16/raw/test.txt
```

```python id="BH4N35xr90o9"
def split_validation(train_set, valid_portion):
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)
```

```python id="KNFB6Rl293Zc"
def data_masks(all_usr_pois, item_tail):
    us_lens = [len(upois) for upois in all_usr_pois]
    len_max = max(us_lens)
    us_pois = [upois + item_tail * (len_max - le)
               for upois, le in zip(all_usr_pois, us_lens)]
    us_msks = [[1] * le + [0] * (len_max - le) for le in us_lens]
    return us_pois, us_msks, len_max
```

```python id="bLRy6o3196VW"
class Dataset():
    def __init__(self, data, shuffle=False, graph=None):
        inputs = data[0]
        inputs, mask, len_max = data_masks(inputs, [0])
        self.inputs = np.asarray(inputs)
        self.mask = np.asarray(mask)
        self.len_max = len_max
        self.targets = np.asarray(data[1])
        self.length = len(inputs)
        self.shuffle = shuffle
        self.graph = graph

    def generate_batch(self, batch_size):
        if self.shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            self.inputs = self.inputs[shuffled_arg]
            self.mask = self.mask[shuffled_arg]
            self.targets = self.targets[shuffled_arg]
        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = slices[-1][:(self.length - batch_size * (n_batch - 1))]
        return slices

    def get_slice(self, i):
        inputs, mask, targets = self.inputs[i], self.mask[i], self.targets[i]
        items, n_node, A, alias_inputs = [], [], [], []
        for u_input in inputs:
            n_node.append(len(np.unique(u_input)))
        max_n_node = np.max(n_node)
        for u_input in inputs:
            node = np.unique(u_input)
            items.append(node.tolist() + (max_n_node - len(node)) * [0])
            u_A = np.zeros((max_n_node, max_n_node))
            for i in np.arange(len(u_input) - 1):
                if u_input[i + 1] == 0:
                    break
                u = np.where(node == u_input[i])[0][0]
                v = np.where(node == u_input[i + 1])[0][0]
                u_A[u][v] = 1
            u_sum_in = np.sum(u_A, 0)
            u_sum_in[np.where(u_sum_in == 0)] = 1
            u_A_in = np.divide(u_A, u_sum_in)
            u_sum_out = np.sum(u_A, 1)
            u_sum_out[np.where(u_sum_out == 0)] = 1
            u_A_out = np.divide(u_A.transpose(), u_sum_out)
            u_A = np.concatenate([u_A_in, u_A_out]).transpose()
            A.append(u_A)
            alias_inputs.append([np.where(node == i)[0][0] for i in u_input])
        return alias_inputs, A, items, mask, targets
```

<!-- #region id="rFug-Z4-7X4C" -->
## Model
<!-- #endregion -->

```python id="LeJK_7u-9Z3v"
class AGC(optim.Optimizer):
    
    """Generic implementation of the Adaptive Gradient Clipping
    Args:
      params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
      optim (torch.optim.Optimizer): Optimizer with base class optim.Optimizer
      clipping (float, optional): clipping value (default: 1e-3)
      eps (float, optional): eps (default: 1e-3)
      model (torch.nn.Module, optional): The original model
      ignore_agc (str, Iterable, optional): Layers for AGC to ignore
    """

    def __init__(self, params, optim: optim.Optimizer, clipping: float = 1e-2, eps: float = 1e-3, model=None, ignore_agc= ['']):
        if clipping < 0.0:
            raise ValueError("Invalid clipping value: {}".format(clipping))
        if eps < 0.0:
            raise ValueError("Invalid eps value: {}".format(eps))

        self.optim = optim

        defaults = dict(clipping=clipping, eps=eps)
        defaults = {**defaults, **optim.defaults}

        if not isinstance(ignore_agc, Iterable):
            ignore_agc = [ignore_agc]

        if model is not None:
            assert ignore_agc not in [
                None, []], "Specify args ignore_agc to ignore fc-like(or other) layers"
            names = [name for name, module in model.named_modules()]

            for module_name in ignore_agc:
                if module_name not in names:
                    raise ModuleNotFoundError(
                        "Module name {} not found in the model".format(module_name))
            parameters = [{"params": module.parameters()} for name,
                          module in model.named_modules() if name not in ignore_agc]

        super(AGC, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                param_norm = torch.max(unitwise_norm(
                    p.detach()), torch.tensor(group['eps']).to(p.device))
                grad_norm = unitwise_norm(p.grad.detach())
                max_norm = param_norm * group['clipping']

                trigger = grad_norm < max_norm

                clipped_grad = p.grad * \
                    (max_norm / torch.max(grad_norm,
                                          torch.tensor(1e-6).to(grad_norm.device)))
                p.grad.data.copy_(torch.where(trigger, clipped_grad, p.grad))

        return self.optim.step(closure)
```

```python id="nUAi6Q6n9Z6F"
def unitwise_norm(x: torch.Tensor):
    if x.ndim <= 1:
        dim = 0
        keepdim = False
    elif x.ndim in [2, 3]:
        dim = 0
        keepdim = True
    elif x.ndim == 4:
        dim = [1, 2, 3]
        keepdim = True
    else:
        raise ValueError('Wrong dimensions of x')

    return torch.sum(x**2, dim=dim, keepdim=keepdim) ** 0.5
```

```python id="xMhrwkxC9Z1q"
class Attention_GNN(Module):
    def __init__(self, hidden_size, step=1):
        super(Attention_GNN, self).__init__()
        self.step = step
        self.hidden_size = hidden_size
        self.input_size = hidden_size * 2
        self.gate_size = 3 * hidden_size
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))
        self.b_iah = Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = Parameter(torch.Tensor(self.hidden_size))

        self.linear_edge_in = nn.Linear(
            self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(
            self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_f = nn.Linear(
            self.hidden_size, self.hidden_size, bias=True)

    def GNNCell(self, A, hidden):
        input_in = torch.matmul(A[:, :, :A.shape[1]],
                                self.linear_edge_in(hidden)) + self.b_iah

        input_out = torch.matmul(
            A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah

        inputs = torch.cat([input_in, input_out], 2)
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate)
        return hy

    def forward(self, A, hidden):
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden)
        return hidden
```

```python id="OBrHhi2M9Zzu"
class Attention_SessionGraph(Module):
    def __init__(self, opt, n_node):
        super(Attention_SessionGraph, self).__init__()
        self.hidden_size = args.hiddenSize
        self.n_node = n_node
        self.batch_size = args.batchSize
        self.nonhybrid = args.nonhybrid
        self.embedding = nn.Embedding(self.n_node, self.hidden_size)
        self.tagnn = Attention_GNN(self.hidden_size, step=args.step)

        self.layer_norm1 = nn.LayerNorm(self.hidden_size)
        self.attn = nn.MultiheadAttention(
            embed_dim=self.hidden_size, num_heads=2, dropout=0.1)

        self.linear_one = nn.Linear(
            self.hidden_size, self.hidden_size, bias=True)

        self.linear_two = nn.Linear(
            self.hidden_size, self.hidden_size, bias=True)

        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_transform = nn.Linear(
            self.hidden_size * 2, self.hidden_size, bias=True)
        self.linear_t = nn.Linear(
            self.hidden_size, self.hidden_size, bias=False)  # target attention
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=args.lr, weight_decay=args.l2)
        self.agc_optimizer = AGC(self.parameters(), self.optimizer, model=self)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=args.lr_dc_step, gamma=args.lr_dc)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def compute_scores(self, hidden, mask):
        ht = hidden[torch.arange(mask.shape[0]).long(), torch.sum(
            mask, 1) - 1]  # batch_size x latent_size
        # batch_size x 1 x latent_size
        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])
        q2 = self.linear_two(hidden)  # batch_size x seq_length x latent_size
        # batch_size x seq_length x 1
        alpha = self.linear_three(torch.sigmoid(q1 + q2))
        alpha = F.softmax(alpha, 1)  # batch_size x seq_length x 1
        # batch_size x latent_size
        a = torch.sum(alpha * hidden *
                      mask.view(mask.shape[0], -1, 1).float(), 1)

        if not self.nonhybrid:
            a = self.linear_transform(torch.cat([a, ht], 1))
        b = self.embedding.weight[1:]  # n_nodes x latent_size

        # batch_size x seq_length x latent_size
        hidden = hidden * mask.view(mask.shape[0], -1, 1).float()
        qt = self.linear_t(hidden)  # batch_size x seq_length x latent_size
        # batch_size x n_nodes x seq_length
        beta = F.softmax(b @ qt.transpose(1, 2), -1)
        target = beta @ hidden  # batch_size x n_nodes x latent_size
        a = a.view(ht.shape[0], 1, ht.shape[1])  # batch_size x 1 x latent_size
        a = a + target  # batch_size x n_nodes x latent_size
        scores = torch.sum(a * b, -1)  # batch_size x n_nodes
        return scores

    def forward(self, inputs, A):
        hidden = self.embedding(inputs)
        hidden = self.tagnn(A, hidden)
        hidden = hidden.permute(1, 0, 2)

        skip = self.layer_norm1(hidden)
        hidden, attn_w = self.attn(
            hidden, hidden, hidden, attn_mask=get_mask(hidden.shape[0]))
        hidden = hidden+skip
        hidden = hidden.permute(1, 0, 2)

        return hidden
```

<!-- #region id="Chu4GEAt7fn5" -->
## Training
<!-- #endregion -->

```python id="LISE5EPV9Zxg"
def get_mask(seq_len):
    return torch.from_numpy(np.triu(np.ones((seq_len, seq_len)), k=1).astype('bool')).to('cuda')


def to_cuda(input_variable):
    if torch.cuda.is_available():
        return input_variable.cuda()
    else:
        return input_variable


def to_cpu(input_variable):
    if torch.cuda.is_available():
        return input_variable.cpu()
    else:
        return input_variable


def forward(model, i, data):
    alias_inputs, A, items, mask, targets = data.get_slice(i)
    alias_inputs = to_cuda(torch.Tensor(alias_inputs).long())
    items = to_cuda(torch.Tensor(items).long())
    A = to_cuda(torch.Tensor(A).float())
    mask = to_cuda(torch.Tensor(mask).long())
    hidden = model(items, A)

    def get(i): return hidden[i][alias_inputs[i]]
    seq_hidden = torch.stack([get(i)
                             for i in torch.arange(len(alias_inputs)).long()])

    return targets, model.compute_scores(seq_hidden, mask)


def train_test(model, train_data, test_data):
    model.scheduler.step()
    print('Start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size)

    for i, j in tqdm(zip(slices, np.arange(len(slices))), total=len(slices)):
        model.optimizer.zero_grad()
        targets, scores = forward(model, i, train_data)
        targets = to_cuda(torch.Tensor(targets).long())
        loss = model.loss_function(scores, targets - 1)
        loss.backward()
        model.optimizer.step()
        total_loss += loss.item()

        if j % int(len(slices) / 5 + 1) == 0:
            print('[%d/%d] Loss: %.4f' % (j, len(slices), loss.item()))

    print('\tLoss Value:\t%.3f' % total_loss)
    print('Start Prediction: ', datetime.datetime.now())

    model.eval()
    hit, mrr = [], []
    slices = test_data.generate_batch(model.batch_size)

    for i in slices:
        targets, scores = forward(model, i, test_data)
        sub_scores = scores.topk(20)[1]
        sub_scores = to_cpu(sub_scores).detach().numpy()

        for score, target, mask in zip(sub_scores, targets, test_data.mask):
            hit.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr.append(0)
            else:
                mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))

    hit = np.mean(hit) * 100
    mrr = np.mean(mrr) * 100
    return hit, mrr


def get_pos(seq_len):
    return torch.arange(seq_len).unsqueeze(0)
```

```python id="6TAZMGZV9Zvx"
def str2bool(v):
    return v.lower() in ('true')
```

```python id="MO5yEePq9ZtY"
class Args():
    dataset = 'diginetica'
    batchSize = 50
    hiddenSize = 100 # Hidden state dimensions
    epoch = 1
    lr = 0.001
    lr_dc = 0.1 # Set the decay rate for Learning rate
    lr_dc_step = 3 # Steps for learning rate decay
    l2 = 1e-5 # Assign L2 Penalty
    step = 1
    patience = 10 # Used for early stopping criterion
    nonhybrid = True
    validation = True
    valid_portion = 0.1 # Portion of train set to split into val set
    defaults = True # Use default configuration

args = Args()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 171, "referenced_widgets": ["2532292a67b54901ae69cc3a270ec97d", "e9a0bd285197471e955bcd3309f0f9f4", "8e131d3b74d14eefbca397b93d49ccb1", "ef31a250c90b4786b61b98125decd8d7", "35b3c0271ceb47cf9edd6f70ffb34af3", "51a1648191584f8192b5f8065e7bbd75", "3c0b22f7b0b84ae8a2d53059b36bfc16", "98d6ae7b07e34c2c89ba5a5fd6a1ed49", "1403f29751ec4e9b83fbbd8c505f1b4a", "93f9689f0c9d4e41b3f121896e983507", "c6f37224738843ffa487cbda4c1bef90"]} id="uCfErpBd9ZrY" outputId="3ce85cf0-20a8-4e4b-ae3b-4a9c69fa3e35"
model_save_dir = 'saved/'
writer = SummaryWriter(log_dir='with_pos/logs')

train_data = pickle.load(open('train.txt', 'rb'))
test_data = pickle.load(open('test.txt', 'rb'))

if args.validation:
    train_data, valid_data = split_validation(
        train_data, args.valid_portion)
    test_data = valid_data
else:
    print('Testing dataset used validation set')

train_data = Dataset(train_data, shuffle=True)
test_data = Dataset(test_data, shuffle=False)

n_node = 43098

model = to_cuda(Attention_SessionGraph(args, n_node))
start = time.time()
best_result = [0, 0]
best_epoch = [0, 0]
bad_counter = 0

for epoch in range(args.epoch):
    print('-' * 50)
    print('Epoch: ', epoch)
    hit, mrr = train_test(model, train_data, test_data)
    flag = 0

    # Logging
    writer.add_scalar('epoch/recall', hit, epoch)
    writer.add_scalar('epoch/mrr', mrr, epoch)

    flag = 0

    if hit >= best_result[0]:
        best_result[0] = hit
        best_epoch[0] = epoch
        flag = 1
        torch.save(model, model_save_dir + 'epoch_' +
                    str(epoch) + '_recall_' + str(hit) + '_.pt')
    if mrr >= best_result[1]:
        best_result[1] = mrr
        best_epoch[1] = epoch
        flag = 1
        torch.save(model, model_save_dir + 'epoch_' +
                    str(epoch) + '_mrr_' + str(mrr) + '_.pt')

    print('Best Result:')
    print('\tRecall@20:\t%.4f\tMRR@20:\t%.4f\tEpoch:\t%d,\t%d' %
            (best_result[0], best_result[1], best_epoch[0], best_epoch[1]))

    bad_counter += 1 - flag

    if bad_counter >= args.patience:
        break

print('-' * 50)
end = time.time()
print("Running time: %f seconds" % (end - start))
```
