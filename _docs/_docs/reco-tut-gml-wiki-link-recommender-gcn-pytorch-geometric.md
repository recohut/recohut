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

```python id="mVNL2ZNYUihx"
!pip install -q jina git+https://github.com/jina-ai/jina-commons
!pip install -q torch-scatter -f https://pytorch-geometric.com/whl/torch-1.9.0+cu102.html
!pip install -q torch-sparse -f https://pytorch-geometric.com/whl/torch-1.9.0+cu102.html
!pip install -q git+https://github.com/rusty1s/pytorch_geometric.git
```

```python colab={"base_uri": "https://localhost:8080/"} id="9DMxdGbtUvqm" executionInfo={"status": "ok", "timestamp": 1630014159313, "user_tz": -330, "elapsed": 582, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="b6f7d14e-15fb-4d0e-ea0c-df85b2db700d"
!mkdir /content/x && git clone https://github.com/pmernyei/wiki-cs-dataset /content/x
```

```python id="YEcHbgorcM_W" executionInfo={"status": "ok", "timestamp": 1630014631920, "user_tz": -330, "elapsed": 1033, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import json
from itertools import chain

import torch
from torch_geometric.data import InMemoryDataset, Data, download_url
from torch_geometric.datasets import wikics

import pandas as pd
import json
```

<!-- #region id="Ci8nXUy1dm48" -->
### Data loading
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="_GjDOKSKcM9I" executionInfo={"status": "ok", "timestamp": 1630014650564, "user_tz": -330, "elapsed": 831, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="e3a2d9e1-4002-4a29-8840-16029388ea32"
dataset = wikics.WikiCS('./wiki-cs-dataset_autodownload')
dataset.data
```

<!-- #region id="h2-sj5Lvdk-4" -->
### Data exploration
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="3aeNLymZcM6T" executionInfo={"status": "ok", "timestamp": 1630014665006, "user_tz": -330, "elapsed": 619, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="2dc92f0c-bc18-499e-83f7-6dd71ac8ce8b"
dataset.num_classes
```

```python colab={"base_uri": "https://localhost:8080/"} id="783VLyklcM3a" executionInfo={"status": "ok", "timestamp": 1630014680340, "user_tz": -330, "elapsed": 846, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="dcd0a9d0-8cf2-4d04-b03c-b5ab3f5efc1a"
# the 300 dimension corresponds to glove embebeddings
# for each word in the document averaged over
dataset.data.x.shape
```

```python colab={"base_uri": "https://localhost:8080/"} id="jdq8vYDacjrU" executionInfo={"status": "ok", "timestamp": 1630014792546, "user_tz": -330, "elapsed": 6206, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="6cc98a3f-ee52-4f2d-8fc1-f6a34ca88208"
metadata = json.load(open('/content/x/dataset/metadata.json'))
metadata.keys()
```

```python colab={"base_uri": "https://localhost:8080/"} id="lg2d7roYcjou" executionInfo={"status": "ok", "timestamp": 1630014795942, "user_tz": -330, "elapsed": 752, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="60890f84-a056-4c2d-ac65-87d82d38d47f"
metadata['labels']
```

```python colab={"base_uri": "https://localhost:8080/"} id="QLWTTVLLcjl6" executionInfo={"status": "ok", "timestamp": 1630014802159, "user_tz": -330, "elapsed": 584, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="06756df3-53ca-4c3a-e0f1-e0267870d01e"
len(metadata['nodes'])
```

<!-- #region id="32d64346" -->
For each node we have the following information
<!-- #endregion -->

```python id="2e9fffdc" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1630014845660, "user_tz": -330, "elapsed": 647, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="fabd4c4f-f208-4583-baf9-f59edb9119b1"
metadata['nodes'][40].keys()
```

<!-- #region id="a8f1a3ca" -->
Note that from a node `title` we can construct a valid URL from wikipedia as follows:
<!-- #endregion -->

```python id="374b8743" executionInfo={"status": "ok", "timestamp": 1630014848917, "user_tz": -330, "elapsed": 1085, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def create_url(title):
    return f'https://en.wikipedia.org/wiki/{title}'
```

```python id="04081a77" colab={"base_uri": "https://localhost:8080/", "height": 35} executionInfo={"status": "ok", "timestamp": 1630014849787, "user_tz": -330, "elapsed": 16, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="b1c6d61b-87b1-436f-9be7-df3fdb930543"
pos = 1900
create_url(metadata['nodes'][pos]['title'])
```

<!-- #region id="Wii36Mx3cjgJ" -->
### Defining a GCN
<!-- #endregion -->

```python id="spsdxFDuV61_" executionInfo={"status": "ok", "timestamp": 1630014886965, "user_tz": -330, "elapsed": 1025, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
class GCN(torch.nn.Module):
    def __init__(self, num_node_features=300, num_classes=10, hidden_channels=128):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def encode(self, x, edge_index):
        feature_map = None

        def get_activation(model, model_inputs, output):
            nonlocal feature_map
            feature_map = output.detach()

        handle = self.conv1.register_forward_hook(get_activation)
        self.forward(x, edge_index)
        handle.remove()
        return feature_map

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x
```

```python id="BSy-Zq0_cjeG" executionInfo={"status": "ok", "timestamp": 1630014931457, "user_tz": -330, "elapsed": 620, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
num_classes = len(dataset.data.y.unique())
num_features = dataset.data.x.shape[1]

model = GCN(num_node_features=num_features, 
            num_classes=num_classes,
            hidden_channels=128)
```

```python colab={"base_uri": "https://localhost:8080/"} id="9uqlpi51cja8" executionInfo={"status": "ok", "timestamp": 1630014940107, "user_tz": -330, "elapsed": 693, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="0839ffdd-0d6b-423d-ee54-cf7e407cd59b"
model
```

<!-- #region id="388e0e93" -->
### Training a GCN
<!-- #endregion -->

```python id="55771b24" executionInfo={"status": "ok", "timestamp": 1630015028003, "user_tz": -330, "elapsed": 1888, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
data = dataset.data 
loss_values = []
```

```python id="4150d8f9" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1630015789615, "user_tz": -330, "elapsed": 761622, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="7a86ec18-3090-4a25-d744-2f008eb4b77a"
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

def train(model, data):
      model.train()
      optimizer.zero_grad()  # Clear gradients.
      out = model(data.x, data.edge_index)  # Perform a single forward pass.
      #mask = data.train_mask[:,0]
      #loss = criterion(out[mask], data.y[mask])  # Compute the loss solely based on the training nodes.
      loss = criterion(out, data.y)  # Compute the loss solely based on the training nodes.  
      loss.backward()  # Derive gradients.
      optimizer.step()  # Update parameters based on gradients.
      return loss

def test(model, data):
      model.eval()
      out = model(data.x, data.edge_index)
      pred = out.argmax(dim=1)  # Use the class with highest probability.
      test_correct = pred[data.test_mask] == data.y[data.test_mask]  # Check against ground-truth labels.
      test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.
      return test_acc

n_epochs = 400

for epoch in range(1, n_epochs):
    loss = train(model, data)
    loss_values.append(loss)
    print(f'\rEpoch: {epoch:03d}, Loss: {loss:.4f}', end='')
```

```python id="0949b204" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1630016492388, "user_tz": -330, "elapsed": 1649, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="5146afcb-4788-424e-d334-5370daf1659c"
test_acc = test(model, data)
print(f'Test Accuracy: {test_acc:.4f}')
```

<!-- #region id="6f8af27b" -->
### Storing model to disk
<!-- #endregion -->

```python id="7a3a0caf" executionInfo={"status": "ok", "timestamp": 1630016516911, "user_tz": -330, "elapsed": 729, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
torch.save(model.state_dict(), './saved_model.torch')
```

<!-- #region id="82f51264" -->
### Load model from disk
<!-- #endregion -->

```python id="72972626" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1630016530971, "user_tz": -330, "elapsed": 703, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="cc11da04-2600-4205-b48c-0daaf5b3080d"
model2 = GCN(num_node_features= num_features, 
             num_classes= num_classes,
             hidden_channels=128)

model2.load_state_dict(torch.load('./saved_model.torch'))
```

```python id="a7cf10ec" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1630016537738, "user_tz": -330, "elapsed": 1601, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="8f8e0a87-70ed-46a8-cd8e-21c8a49a09c9"
test(model2, data)
```

<!-- #region id="b3f37d3c" -->
### Visualized learned embeddings
<!-- #endregion -->

```python id="621ae501" executionInfo={"status": "ok", "timestamp": 1630016551994, "user_tz": -330, "elapsed": 1107, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
# Helper function for visualization.
%matplotlib inline
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def visualize(h, color):
    z = TSNE(n_components=2).fit_transform(out.detach().cpu().numpy())

    plt.figure(figsize=(10,10))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    plt.show()
```

```python id="0e14b722" colab={"base_uri": "https://localhost:8080/", "height": 578} executionInfo={"status": "ok", "timestamp": 1630016694224, "user_tz": -330, "elapsed": 140375, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="d120fc78-920c-4f4a-b563-8792765c4813"
out = model(data.x, data.edge_index)
visualize(out, color=data.y)
```
