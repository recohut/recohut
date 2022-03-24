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

<!-- #region id="Ys6CTcxp-XgC" -->
# Movielens Deep Neural Net model from scratch in PyTorch
> Training Pytorch MLP model on movielens-100k dataset and visualizing factors by decomposing using PCA

- toc: true
- badges: true
- comments: true
- categories: [Pytorch, Movie, MLP, NCF, Visualization]
- image:
<!-- #endregion -->

```python id="I2f_R0Yo6BUp"
!pip install -U -q git+https://github.com/sparsh-ai/recochef.git
```

```python id="KXT07lHDBzAQ"
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F 
from torch.optim.lr_scheduler import _LRScheduler

from recochef.datasets.movielens import MovieLens
from recochef.preprocessing.encode import label_encode
from recochef.utils.iterators import batch_generator
from recochef.models.embedding import EmbeddingNet

import math
import copy
import pickle
import numpy as np
import pandas as pd
from textwrap import wrap
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
plt.style.use('ggplot')
```

<!-- #region id="NINhOhYAxt5n" -->
## Data loading and preprocessing
<!-- #endregion -->

```python id="3Z4R3bXNjaNP"
data = MovieLens()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="A2Xgw-sXk7Ac" outputId="1e698595-cc1c-4b54-8e23-6ef8b7006cdf"
ratings_df = data.load_interactions()
ratings_df.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 343} id="wIUc-Ba_6xBK" outputId="448958fc-7ab5-4417-88ff-f484462f577f"
movies_df = data.load_items()
movies_df.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="X-IcaBzgmOrN" outputId="53d39c4e-5cce-4f79-d33b-c6ae166d515d"
ratings_df, maps = label_encode(ratings_df, ['USERID','ITEMID'])
ratings_df.head()
```

```python id="36dsiSqWwNRz"
X = ratings_df[['USERID','ITEMID']]
y = ratings_df[['RATING']]
```

<!-- #region id="Jte9qDDl2Icu" -->
## Unit tests
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="Mp-OqA2jyNAS" outputId="661ef9e2-43a8-43b7-c76a-2cac63cd00c3"
for _x_batch, _y_batch in batch_generator(X, y, bs=4):
    print(_x_batch)
    print(_y_batch)
    break
```

```python colab={"base_uri": "https://localhost:8080/"} id="oQlnTmST0cx6" outputId="33ec1ed7-c575-4b7e-86c8-f53345cc66e3"
_x_batch[:, 1]
```

<!-- #region id="-8m9jSYc_dGz" -->
## Concepts
<!-- #endregion -->

<!-- #region id="mVavggU2_WUY" -->
### Embedding Net
<!-- #endregion -->

<!-- #region id="D39WDxT5_f3l" -->
The PyTorch is a framework that allows to build various computational graphs (not only neural networks) and run them on GPU. The conception of tensors, neural networks, and computational graphs is outside the scope of this article but briefly speaking, one could treat the library as a set of tools to create highly computationally efficient and flexible machine learning models. In our case, we want to create a neural network that could help us to infer the similarities between users and predict their ratings based on available data.
<!-- #endregion -->

<!-- #region id="x1J4IYEV_h_v" -->
<!-- #endregion -->

<!-- #region id="9Gve8w7f_l8i" -->
The picture above schematically shows the model we're going to build. At the very beginning, we put our embeddings matrices, or look-ups, which convert integer IDs into arrays of floating-point numbers. Next, we put a bunch of fully-connected layers with dropouts. Finally, we need to return a list of predicted ratings. For this purpose, we use a layer with sigmoid activation function and rescale it to the original range of values (in case of MovieLens dataset, it is usually from 1 to 5).
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="9zDzhH2c0Cv-" outputId="b62a8495-8e81-45d4-d841-77324705257f"
netx = EmbeddingNet(
    n_users=50, n_items=20, 
    n_factors=10, hidden=[500], 
    embedding_dropout=0.05, dropouts=[0.5])
netx
```

<!-- #region id="o4vQFZ6iwiyM" -->
### Cyclical Learning Rate (CLR)
<!-- #endregion -->

<!-- #region id="6RIaav5rwk66" -->
One of the `fastai` library features is the cyclical learning rate scheduler. We can implement something similar inheriting the `_LRScheduler` class from the `torch` library. Following the [original paper's](https://arxiv.org/abs/1506.01186) pseudocode, this [CLR Keras callback implementation](https://github.com/bckenstler/CLR), and making a couple of adjustments to support [cosine annealing](https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.CosineAnnealingLR) with restarts, let's create our own CLR scheduler.

The implementation of this idea is quite simple. The [base PyTorch scheduler class](https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html) has the `get_lr()` method that is invoked each time when we call the `step()` method. The method should return a list of learning rates depending on the current training epoch. In our case, we have the same learning rate for all of the layers, and therefore, we return a list with a single value. 

The next cell defines a `CyclicLR` class that expectes a single callback function. This function should accept the current training epoch and the base value of learning rate, and return a new learning rate value.
<!-- #endregion -->

```python id="eYQh4ZCmmgW9"
class CyclicLR(_LRScheduler):
    
    def __init__(self, optimizer, schedule, last_epoch=-1):
        assert callable(schedule)
        self.schedule = schedule
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.schedule(self.last_epoch, lr) for lr in self.base_lrs]
```

<!-- #region id="1bpK5hOvw7Hg" -->
Our scheduler is very similar to [LambdaLR](https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.LambdaLR) one but expects a bit different callback signature. 

So now we only need to define appropriate scheduling functions. We're createing a couple of functions that accept scheduling parameters and return a _new function_ with the appropriate signature:
<!-- #endregion -->

```python id="I6st2zPctj1T"
def triangular(step_size, max_lr, method='triangular', gamma=0.99):
    
    def scheduler(epoch, base_lr):
        period = 2 * step_size
        cycle = math.floor(1 + epoch/period)
        x = abs(epoch/step_size - 2*cycle + 1)
        delta = (max_lr - base_lr)*max(0, (1 - x))

        if method == 'triangular':
            pass  # we've already done
        elif method == 'triangular2':
            delta /= float(2 ** (cycle - 1))
        elif method == 'exp_range':
            delta *= (gamma**epoch)
        else:
            raise ValueError('unexpected method: %s' % method)
            
        return base_lr + delta
        
    return scheduler
```

```python id="k-CjYin0toWa"
def cosine(t_max, eta_min=0):
    
    def scheduler(epoch, base_lr):
        t = epoch % t_max
        return eta_min + (base_lr - eta_min)*(1 + math.cos(math.pi*t/t_max))/2
    
    return scheduler
```

<!-- #region id="oB_zTLW-wwdM" -->
To understand how the created functions work, and to check the correctness of our implementation, let's create a couple of plots visualizing learning rates changes depending on the number of epoch:
<!-- #endregion -->

```python id="Dl-TWx4OwwdN"
def plot_lr(schedule):
    ts = list(range(1000))
    y = [schedule(t, 0.001) for t in ts]
    plt.plot(ts, y)
```

```python id="wCfhKAoMwwdN" colab={"base_uri": "https://localhost:8080/", "height": 265} outputId="6b9cfa60-49b7-44db-d61e-278d800d18a8"
plot_lr(triangular(250, 0.005))
```

```python id="UPXizw35wwdO" colab={"base_uri": "https://localhost:8080/", "height": 265} outputId="682f10b0-a181-4e34-ea9b-fe1d9fbc58c4"
plot_lr(triangular(250, 0.005, 'triangular2'))
```

```python id="2MMOua0WwwdO" colab={"base_uri": "https://localhost:8080/", "height": 265} outputId="e43170c8-4a6b-403c-8cec-163bd012d75d"
plot_lr(triangular(250, 0.005, 'exp_range', gamma=0.999))
```

```python id="gdwJu6i3wwdP" colab={"base_uri": "https://localhost:8080/", "height": 265} outputId="dcc8fa92-50be-4818-e8b7-f78b758a37d0"
plot_lr(cosine(t_max=500, eta_min=0.0005))
```

<!-- #region id="6zcqiJUEwwdQ" -->
Note that cosine annealing scheduler is a bit different from other schedules as soon as it starts with `base_lr` and gradually decreases it to the minimal value while triangle schedulers increase the original rate.
<!-- #endregion -->

<!-- #region id="agK_98cWwwdQ" -->
## Training Loop

Now we're ready to start the training process. First of all, let's split the original dataset using [train_test_split](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) function from the `scikit-learn` library. (Though you can use anything else instead, like, [get_cv_idxs](https://github.com/fastai/fastai/blob/921777feb46f215ed2b5f5dcfcf3e6edd299ea92/fastai/dataset.py#L6-L22) from `fastai`).
<!-- #endregion -->

```python id="216J9e39wwdR"
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
datasets = {'train': (X_train, y_train), 'val': (X_valid, y_valid)}
dataset_sizes = {'train': len(X_train), 'val': len(X_valid)}
```

```python id="fFgH5tuLwwdR" colab={"base_uri": "https://localhost:8080/"} outputId="098bfcb3-395d-4fae-d998-ba41442be614"
minmax = ratings_df.RATING.astype(float).min(), ratings_df.RATING.astype(float).max()
minmax
```

```python colab={"base_uri": "https://localhost:8080/"} id="WlAmBo9Jys0m" outputId="cbafe2a2-bbb6-4104-8ac4-698cfbd1572b"
n_users = ratings_df.USERID.nunique()
n_movies = ratings_df.ITEMID.nunique()
n_users, n_movies
```

```python id="KHeaux79wwdS"
net = EmbeddingNet(
    n_users=n_users, n_items=n_movies, 
    n_factors=150, hidden=[500, 500, 500], 
    embedding_dropout=0.05, dropouts=[0.5, 0.5, 0.25])
```

<!-- #region id="onmMNylKwwdS" -->
The next cell is preparing and running the training loop with cyclical learning rate, validation and early stopping. We use `Adam` optimizer with cosine-annealing learnign rate. The rate is decreased on each batch during `2` epochs, and then is reset to the original value.

Note that our loop has two phases. One of them is called `train`. During this phase, we update our network's weights and change the learning rate. The another one is called `val` and is used to check the model's performence. When the loss value decreases, we save model parameters to restore them later. If there is no improvements after `10` sequential training epochs, we exit from the loop.
<!-- #endregion -->

```python id="4mRf0N9kwwdT" colab={"base_uri": "https://localhost:8080/"} outputId="7caabf60-f2bc-4a1a-8d9a-e88964ffcf9f"
lr = 1e-3
wd = 1e-5
bs = 50
n_epochs = 100
patience = 10
no_improvements = 0
best_loss = np.inf
best_weights = None
history = []
lr_history = []

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

net.to(device)
criterion = nn.MSELoss(reduction='sum')
optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
iterations_per_epoch = int(math.ceil(dataset_sizes['train'] // bs))
scheduler = CyclicLR(optimizer, cosine(t_max=iterations_per_epoch * 2, eta_min=lr/10))

for epoch in range(n_epochs):
    stats = {'epoch': epoch + 1, 'total': n_epochs}
    
    for phase in ('train', 'val'):
        training = phase == 'train'
        running_loss = 0.0
        n_batches = 0
        
        for batch in batch_generator(*datasets[phase], shuffle=training, bs=bs):
            x_batch, y_batch = [b.to(device) for b in batch]
            optimizer.zero_grad()
        
            # compute gradients only during 'train' phase
            with torch.set_grad_enabled(training):
                outputs = net(x_batch[:, 0], x_batch[:, 1], minmax)
                loss = criterion(outputs, y_batch)
                
                # don't update weights and rates when in 'val' phase
                if training:
                    scheduler.step()
                    loss.backward()
                    optimizer.step()
                    lr_history.extend(scheduler.get_lr())
                    
            running_loss += loss.item()
            
        epoch_loss = running_loss / dataset_sizes[phase]
        stats[phase] = epoch_loss
        
        # early stopping: save weights of the best model so far
        if phase == 'val':
            if epoch_loss < best_loss:
                print('loss improvement on epoch: %d' % (epoch + 1))
                best_loss = epoch_loss
                best_weights = copy.deepcopy(net.state_dict())
                no_improvements = 0
            else:
                no_improvements += 1
                
    history.append(stats)
    print('[{epoch:03d}/{total:03d}] train: {train:.4f} - val: {val:.4f}'.format(**stats))
    if no_improvements >= patience:
        print('early stopping after epoch {epoch:03d}'.format(**stats))
        break
```

<!-- #region id="EGrWJQGnwwdT" -->
## Metrics

To visualize the training process and to check the correctness of the learning rate scheduling, let's create a couple of plots using collected stats:
<!-- #endregion -->

```python id="y3vAtcy9wwdU" colab={"base_uri": "https://localhost:8080/", "height": 282} outputId="e53d9bcf-bf4f-4160-ca66-c27d69801e7b"
ax = pd.DataFrame(history).drop(columns='total').plot(x='epoch')
```

```python id="8SNXU2CKwwdU" colab={"base_uri": "https://localhost:8080/", "height": 265} outputId="911a55af-d486-44c5-c339-5fce87ad7301"
_ = plt.plot(lr_history[:2*iterations_per_epoch])
```

<!-- #region id="Yy8lkzpGwwdV" -->
As expected, the learning rate is updated in accordance with cosine annealing schedule.
<!-- #endregion -->

<!-- #region id="t_4cZUXgwwdV" -->
The training process was terminated after _16 epochs_. Now we're going to restore the best weights saved during training, and apply the model to the validation subset of the data to see the final model's performance:
<!-- #endregion -->

```python id="b9WGkwZ7wwdV" colab={"base_uri": "https://localhost:8080/"} outputId="57894b4d-afae-4fe0-e408-f5c95f031f8a"
net.load_state_dict(best_weights)
```

```python id="lI_DqmEIwwda"
# groud_truth, predictions = [], []

# with torch.no_grad():
#     for batch in batch_generator(*datasets['val'], shuffle=False, bs=bs):
#         x_batch, y_batch = [b.to(device) for b in batch]
#         outputs = net(x_batch[:, 1], x_batch[:, 0], minmax)
#         groud_truth.extend(y_batch.tolist())
#         predictions.extend(outputs.tolist())

# groud_truth = np.asarray(groud_truth).ravel()
# predictions = np.asarray(predictions).ravel()
```

```python id="YdXslUMBwwda"
# final_loss = np.sqrt(np.mean((predictions - groud_truth)**2))
# print(f'Final RMSE: {final_loss:.4f}')
```

```python id="hB9t-ARGwwdb"
with open('best.weights', 'wb') as file:
    pickle.dump(best_weights, file)
```

<!-- #region id="8Qcj-GoNwwdb" -->
## Bonus: Embeddings Visualization

Finally, we can create a couple of visualizations to show how various movies are encoded in embeddings space. Again, we're repeting the approach shown in the original post and apply the [Principal Components Analysis](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) to reduce the dimentionality of embeddings and show some of them with bar plots.
<!-- #endregion -->

<!-- #region id="OqwMsqWHwwdc" -->
Loading previously saved weights:
<!-- #endregion -->

```python id="JKlP4S1zwwdc" colab={"base_uri": "https://localhost:8080/"} outputId="60ca6f00-d754-426b-d9b0-c21a724976a2"
with open('best.weights', 'rb') as file:
    best_weights = pickle.load(file)
net.load_state_dict(best_weights)
```

```python id="omN9TxzFwwdd"
def to_numpy(tensor):
    return tensor.cpu().numpy()
```

<!-- #region id="_8x4DFcbwwdd" -->
Creating the mappings between original users's and movies's IDs, and new contiguous values:
<!-- #endregion -->

```python id="qdHe7Nekwwdd"
user_id_map = maps['USERID_TO_IDX']
movie_id_map = maps['ITEMID_TO_IDX']
embed_to_original = maps['IDX_TO_ITEMID']

popular_movies = ratings_df.groupby('ITEMID').ITEMID.count().sort_values(ascending=False).values[:1000]
```

<!-- #region id="L7cJFbDXwwde" -->
Reducing the dimensionality of movie embeddings vectors:
<!-- #endregion -->

```python id="A0My7Epuwwde" colab={"base_uri": "https://localhost:8080/"} outputId="037e6d7d-9650-4664-a5f4-df54fd4e1fdb"
embed = to_numpy(net.m.weight.data)
pca = PCA(n_components=5)
components = pca.fit(embed[popular_movies].T).components_
components.shape
```

<!-- #region id="MeHZJsE3wwdf" -->
Finally, creating a joined data frame with projected embeddings and movies they represent:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="g4_xKa2p7bIO" outputId="9b589c05-f953-4aa6-98d1-9e8a1196a056"
movies = movies_df[['ITEMID','TITLE']].dropna()
movies.shape
```

```python id="YkGwvIUqwwdf" colab={"base_uri": "https://localhost:8080/", "height": 204} outputId="56b48acf-a8d1-4f68-f575-92b51524c358"
components_df = pd.DataFrame(components.T, columns=[f'fc{i}' for i in range(pca.n_components_)])
components_df.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="chcYgtkw8ij6" outputId="9963df39-c7a0-4988-da15-8e3ae3caec77"
components_df.shape
```

```python colab={"base_uri": "https://localhost:8080/", "height": 173} id="lTu7ZbLL8NaO" outputId="d958df14-896f-4273-c7e5-61deab71120c"
movie_ids = [embed_to_original[idx] for idx in components_df.index]
meta = movies.set_index('ITEMID')
components_df['ITEMID'] = movie_ids
components_df['TITLE'] = meta.loc[movie_ids].TITLE.values
components_df.sample(4)
```

```python id="rW-l-0Izwwdg"
def plot_components(components, component, ascending=False):
    fig, ax = plt.subplots(figsize=(18, 12))
    
    subset = components.sort_values(by=component, ascending=ascending).iloc[:12]
    columns = components_df.columns
    features = columns[columns.str.startswith('fc')].tolist()
    
    fc = subset[features]
    labels = ['\n'.join(wrap(t, width=10)) for t in subset.TITLE]
    
    fc.plot(ax=ax, kind='bar')
    y_ticks = [f'{t:2.2f}' for t in ax.get_yticks()]
    ax.set_xticklabels(labels, rotation=0, fontsize=14)
    ax.set_yticklabels(y_ticks, fontsize=14)
    ax.legend(loc='best', fontsize=14)
    
    plot_title = f"Movies with {['highest', 'lowest'][ascending]} '{component}' component values" 
    ax.set_title(plot_title, fontsize=20)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 718} id="WpJkrTX9-Asp" outputId="ec1087a1-0157-445a-9548-fe29313e3510"
plot_components(components_df, 'fc0', ascending=False)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 718} id="VCnP_uOT-ApN" outputId="53b4ad7e-e7f2-4085-f93d-f64ea59217a4"
plot_components(components_df, 'fc0', ascending=True)
```

```python id="Z8sdZC7-wwdh" colab={"base_uri": "https://localhost:8080/", "height": 690} outputId="e05a8902-a2a5-4505-b24c-ccf1b404564b"
plot_components(components_df, 'fc1', ascending=False)
```

```python id="8xp79OV8wwdh" colab={"base_uri": "https://localhost:8080/", "height": 704} outputId="7ad971e4-59d3-4756-ed06-83fb9805ac7f"
plot_components(components_df, 'fc1', ascending=True)
```
