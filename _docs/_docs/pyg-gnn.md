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

<!-- #region id="Yqgsr8ahMGQV" -->
# GNNs with PyTorch Geometric
> Understanding the Fundamentals of Graph Neural and Graph Convolutional Networks with PyTorch Geometric
<!-- #endregion -->

```python id="33HhJOZ_6wxl"
# torch geometric
try: 
    import torch_geometric
except ModuleNotFoundError:
    # Installing torch geometric packages with specific CUDA+PyTorch version. 
    # See https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html for details 
    import torch
    TORCH = torch.__version__.split('+')[0]
    CUDA = 'cu' + torch.version.cuda.replace('.','')

    !pip install torch-scatter     -f https://pytorch-geometric.com/whl/torch-{TORCH}+{CUDA}.html
    !pip install torch-sparse      -f https://pytorch-geometric.com/whl/torch-{TORCH}+{CUDA}.html
    !pip install torch-cluster     -f https://pytorch-geometric.com/whl/torch-{TORCH}+{CUDA}.html
    !pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-{TORCH}+{CUDA}.html
    !pip install torch-geometric 
    import torch_geometric
import torch_geometric.nn as geom_nn
import torch_geometric.data as geom_data
```

<!-- #region id="BrQyCrq-nFYA" -->
## Graph attention networks (GAT)
<!-- #endregion -->

<!-- #region id="i3P3cuVJqWpF" -->
### Code Practice
<!-- #endregion -->

```python id="fIxCZoQBsXGf"
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
```

<!-- #region id="escsS-iusy5t" -->
#### Structure
<!-- #endregion -->

```python id="CdxIh9-msXGh"
class GATLayer(nn.Module):
    """
    Simple PyTorch Implementation of the Graph Attention layer.
    """
    def __init__(self):
        super(GATLayer, self).__init__()
      
    def forward(self, input, adj):
        print("")
```

<!-- #region id="9pHUmxNzsXGi" -->
Let's start from the forward method
<!-- #endregion -->

<!-- #region id="4V3WzW7vsXGj" -->
#### Linear Transformation

$$
\bar{h'}_i = \textbf{W}\cdot \bar{h}_i
$$
with $\textbf{W}\in\mathbb R^{F'\times F}$ and $\bar{h}_i\in\mathbb R^{F}$.

$$
\bar{h'}_i \in \mathbb{R}^{F'}
$$
<!-- #endregion -->

```python id="vHxrLNcvsXGk" colab={"base_uri": "https://localhost:8080/"} outputId="b2d06c43-421b-40bb-a000-c2ea454d4cf5"
in_features = 5
out_features = 2
nb_nodes = 3

W = nn.Parameter(torch.zeros(size=(in_features, out_features))) #xavier paramiter inizializator
nn.init.xavier_uniform_(W.data, gain=1.414)

input = torch.rand(nb_nodes,in_features) 


# linear transformation
h = torch.mm(input, W)
N = h.size()[0]

print(h.shape)
```

<!-- #region id="BmEJAQYrsXGn" -->
#### Attention Mechanism
<!-- #endregion -->

```python id="mM952UIQsXGp" colab={"base_uri": "https://localhost:8080/"} outputId="3ea2b14c-9395-4471-b595-0cae2e27136a"
a = nn.Parameter(torch.zeros(size=(2*out_features, 1))) #xavier parameter inizializator
nn.init.xavier_uniform_(a.data, gain=1.414)
print(a.shape)

leakyrelu = nn.LeakyReLU(0.2)  # LeakyReLU
```

```python id="kgco_qXqsXGq"
a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * out_features)
```

<!-- #region id="WD0v0CCHtMqr" -->
<!-- #endregion -->

```python id="yfjy06mDsXGr"
e = leakyrelu(torch.matmul(a_input, a).squeeze(2))
```

```python id="xd5Tgf7MsXGs" colab={"base_uri": "https://localhost:8080/"} outputId="091b6e77-71ea-4dfd-d386-94ad172824ba"
print(a_input.shape,a.shape)
print("")
print(torch.matmul(a_input,a).shape)
print("")
print(torch.matmul(a_input,a).squeeze(2).shape)
```

<!-- #region id="2JEveCgDsXGs" -->
#### Masked Attention
<!-- #endregion -->

```python id="90nbf6gZsXGs" colab={"base_uri": "https://localhost:8080/"} outputId="cf6d4722-d255-45d9-cc5a-468983a7195c"
# Masked Attention
adj = torch.randint(2, (3, 3))

zero_vec  = -9e15*torch.ones_like(e)
print(zero_vec.shape)
```

```python id="MC7wZsDCsXGt" colab={"base_uri": "https://localhost:8080/"} outputId="13f26258-7e3f-4962-be17-7eac198bb17f"
attention = torch.where(adj > 0, e, zero_vec)
print(adj,"\n",e,"\n",zero_vec)
attention
```

```python id="TmbtuV2hsXGu"
attention = F.softmax(attention, dim=1)
h_prime   = torch.matmul(attention, h)
```

```python id="_kqS_t-esXGv" colab={"base_uri": "https://localhost:8080/"} outputId="13f0e4c4-cdfe-4101-b471-d703f5592cab"
attention
```

```python id="H7fwMik-sXGw" colab={"base_uri": "https://localhost:8080/"} outputId="4c84027d-f8a0-4f50-c80a-456011b7ce3c"
h_prime
```

<!-- #region id="xp49RxTQsXGx" -->
h_prime vs h
<!-- #endregion -->

```python id="dwu9p5tQsXGy" outputId="8d60105e-7f92-44a0-c105-a0f47492382c"
33print(h_prime,"\n",h)
```

<!-- #region id="MELcnTAlwvfl" -->
### Loading the dataset
<!-- #endregion -->

```python id="T_l8_z4GsXG1" colab={"base_uri": "https://localhost:8080/"} outputId="7f0a0ce4-9ae3-4bf8-afc6-45378fec1747"
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T

import matplotlib.pyplot as plt

name_data = 'Cora'
dataset = Planetoid(root= '/content/' + name_data, name = name_data)
dataset.transform = T.NormalizeFeatures()

print(f"Number of Classes in {name_data}:", dataset.num_classes)
print(f"Number of Node Features in {name_data}:", dataset.num_node_features)
```

<!-- #region id="Vd67RsopsXGy" -->
### Assembling the components
<!-- #endregion -->

```python id="LGQg18xYsXGz"
class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GATLayer, self).__init__()
        self.dropout       = dropout        # drop prob = 0.6
        self.in_features   = in_features    # 
        self.out_features  = out_features   # 
        self.alpha         = alpha          # LeakyReLU with negative input slope, alpha = 0.2
        self.concat        = concat         # conacat = True for all layers except the output layer.

        
        # Xavier Initialization of Weights
        # Alternatively use weights_init to apply weights of choice 
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        # LeakyReLU
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        # Linear Transformation
        h = torch.mm(input, self.W) # matrix multiplication
        N = h.size()[0]
        print(N)

        # Attention Mechanism
        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e       = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        # Masked Attention
        zero_vec  = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime   = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime
```

```python id="HjTPUBsfw_e4"
class GAT(torch.nn.Module):
    def __init__(self):
        super(GAT, self).__init__()
        self.hid = 8
        self.in_head = 8
        self.out_head = 1
        
        
        self.conv1 = GATConv(dataset.num_features, self.hid, heads=self.in_head, dropout=0.6)
        self.conv2 = GATConv(self.hid*self.in_head, dataset.num_classes, concat=False,
                             heads=self.out_head, dropout=0.6)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
                
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        
        return F.log_softmax(x, dim=1)
```

<!-- #region id="lx7QPj9SsXG0" -->
### Use it
<!-- #endregion -->

```python id="LD79VrMosXG3" colab={"base_uri": "https://localhost:8080/"} outputId="d7467f8c-3a1a-4efa-ff77-7144c78bc7ec"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = "cpu"

model = GAT().to(device)
data = dataset[0].to(device)


optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

model.train()
for epoch in range(1000):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    
    if epoch%200 == 0:
        print(loss)
    
    loss.backward()
    optimizer.step()
```

```python id="bW0OMhKqsXG4" colab={"base_uri": "https://localhost:8080/"} outputId="a567917b-c1e1-43b1-d80f-215870be3bac"
model.eval()
_, pred = model(data).max(dim=1)
correct = float(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
acc = correct / data.test_mask.sum().item()
print('Accuracy: {:.4f}'.format(acc))
```

<!-- #region id="e6J6bpLXLqlD" -->
## Graph representation

Before starting the discussion of specific neural network operations on graphs, we should consider how to represent a graph. Mathematically, a graph $\mathcal{G}$ is defined as a tuple of a set of nodes/vertices $V$, and a set of edges/links $E$: $\mathcal{G}=(V,E)$. Each edge is a pair of two vertices, and represents a connection between them. For instance, let's look at the following graph:
<!-- #endregion -->

<!-- #region id="PhodeYd0LSes" -->
![](data:image/png;base64,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)
<!-- #endregion -->

<!-- #region id="KZHGr561Jd07" -->
The vertices are $V=\{1,2,3,4\}$, and edges $E=\{(1,2), (2,3), (2,4), (3,4)\}$. Note that for simplicity, we assume the graph to be undirected and hence don't add mirrored pairs like $(2,1)$. In application, vertices and edge can often have specific attributes, and edges can even be directed. The question is how we could represent this diversity in an efficient way for matrix operations. Usually, for the edges, we decide between two variants: an adjacency matrix, or a list of paired vertex indices. 

The **adjacency matrix** $A$ is a square matrix whose elements indicate whether pairs of vertices are adjacent, i.e. connected, or not. In the simplest case, $A_{ij}$ is 1 if there is a connection from node $i$ to $j$, and otherwise 0. If we have edge attributes or different categories of edges in a graph, this information can be added to the matrix as well. For an undirected graph, keep in mind that $A$ is a symmetric matrix ($A_{ij}=A_{ji}$). For the example graph above, we have the following adjacency matrix:

$$
A = \begin{bmatrix}
    0 & 1 & 0 & 0\\
    1 & 0 & 1 & 1\\
    0 & 1 & 0 & 1\\
    0 & 1 & 1 & 0
\end{bmatrix}
$$

While expressing a graph as a list of edges is more efficient in terms of memory and (possibly) computation, using an adjacency matrix is more intuitive and simpler to implement. In our implementations below, we will rely on the adjacency matrix to keep the code simple. However, common libraries use edge lists, which we will discuss later more.
Alternatively, we could also use the list of edges to define a sparse adjacency matrix with which we can work as if it was a dense matrix, but allows more memory-efficient operations. PyTorch supports this with the sub-package `torch.sparse` ([documentation](https://pytorch.org/docs/stable/sparse.html)) which is however still in a beta-stage (API might change in future).
<!-- #endregion -->

<!-- #region id="jZQwHaxZM9n1" -->
## Graph Convolutions

Graph Convolutional Networks have been introduced by [Kipf et al.](https://openreview.net/pdf?id=SJU4ayYgl) in 2016 at the University of Amsterdam. He also wrote a great [blog post](https://tkipf.github.io/graph-convolutional-networks/) about this topic, which is recommended if you want to read about GCNs from a different perspective. GCNs are similar to convolutions in images in the sense that the "filter" parameters are typically shared over all locations in the graph. At the same time, GCNs rely on message passing methods, which means that vertices exchange information with the neighbors, and send "messages" to each other. Before looking at the math, we can try to visually understand how GCNs work. The first step is that each node creates a feature vector that represents the message it wants to send to all its neighbors. In the second step, the messages are sent to the neighbors, so that a node receives one message per adjacent node. Below we have visualized the two steps for our example graph. 
<!-- #endregion -->

<!-- #region id="RoAqgLUINDQ5" -->
![](data:image/png;base64,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)
<!-- #endregion -->

<!-- #region id="v3LT905AJd1J" -->
If we want to formulate that in more mathematical terms, we need to first decide how to combine all the messages a node receives. As the number of messages vary across nodes, we need an operation that works for any number. Hence, the usual way to go is to sum or take the mean. Given the previous features of nodes $H^{(l)}$, the GCN layer is defined as follows:

$$H^{(l+1)} = \sigma\left(\hat{D}^{-1/2}\hat{A}\hat{D}^{-1/2}H^{(l)}W^{(l)}\right)$$

$W^{(l)}$ is the weight parameters with which we transform the input features into messages ($H^{(l)}W^{(l)}$). To the adjacency matrix $A$ we add the identity matrix so that each node sends its own message also to itself: $\hat{A}=A+I$. Finally, to take the average instead of summing, we calculate the matrix $\hat{D}$ which is a diagonal matrix with $D_{ii}$ denoting the number of neighbors node $i$ has. $\sigma$ represents an arbitrary activation function, and not necessarily the sigmoid (usually a ReLU-based activation function is used in GNNs). 

When implementing the GCN layer in PyTorch, we can take advantage of the flexible operations on tensors. Instead of defining a matrix $\hat{D}$, we can simply divide the summed messages by the number of neighbors afterward. Additionally, we replace the weight matrix with a linear layer, which additionally allows us to add a bias. Written as a PyTorch module, the GCN layer is defined as follows:
<!-- #endregion -->

```python id="QGoxbcxb-FTO"
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
```

```python id="pXhfaEuGKNP2"
class GCNLayer(nn.Module):
    
    def __init__(self, c_in, c_out):
        super().__init__()
        self.projection = nn.Linear(c_in, c_out)

    def forward(self, node_feats, adj_matrix):
        """
        Inputs:
            node_feats - Tensor with node features of shape [batch_size, num_nodes, c_in]
            adj_matrix - Batch of adjacency matrices of the graph. If there is an edge from i to j, adj_matrix[b,i,j]=1 else 0.
                         Supports directed edges by non-symmetric matrices. Assumes to already have added the identity connections. 
                         Shape: [batch_size, num_nodes, num_nodes]
        """
        # Num neighbours = number of incoming edges
        num_neighbours = adj_matrix.sum(dim=-1, keepdims=True)
        node_feats = self.projection(node_feats)
        node_feats = torch.bmm(adj_matrix, node_feats)
        node_feats = node_feats / num_neighbours
        return node_feats
```

<!-- #region id="bhJEKQIjJd1R" -->
To further understand the GCN layer, we can apply it to our example graph above. First, let's specify some node features and the adjacency matrix with added self-connections:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="SRny94_XOgou" outputId="11bb492e-b59f-4728-c24d-b3972cd87b06"
node_feats = torch.arange(8, dtype=torch.float32).view(1, 4, 2)
adj_matrix = torch.Tensor([[[1, 1, 0, 0],
                            [1, 1, 1, 1],
                            [0, 1, 1, 1],
                            [0, 1, 1, 1]]])

print("Node features:\n", node_feats)
print("\nAdjacency matrix:\n", adj_matrix)
```

<!-- #region id="0p7_bfgPJd1X" -->
Next, let's apply a GCN layer to it. For simplicity, we initialize the linear weight matrix as an identity matrix so that the input features are equal to the messages. This makes it easier for us to verify the message passing operation.
<!-- #endregion -->

```python id="rKvNBJ6zJd1Z" colab={"base_uri": "https://localhost:8080/"} outputId="ecfc0711-5090-42df-9095-a54eef34299d"
layer = GCNLayer(c_in=2, c_out=2)
layer.projection.weight.data = torch.Tensor([[1., 0.], [0., 1.]])
layer.projection.bias.data = torch.Tensor([0., 0.])

with torch.no_grad():
    out_feats = layer(node_feats, adj_matrix)

print("Adjacency matrix", adj_matrix)
print("Input features", node_feats)
print("Output features", out_feats)
```

<!-- #region id="4-iXNxPTPScd" -->
Next, let's apply a GCN layer to it. For simplicity, we initialize the linear weight matrix as an identity matrix so that the input features are equal to the messages. This makes it easier for us to verify the message passing operation.
<!-- #endregion -->

```python id="EUGQ3PjhPSce" outputId="f9f400b9-9adb-4c00-bcf5-e6d40ced69fc"
layer = GCNLayer(c_in=2, c_out=2)
layer.projection.weight.data = torch.Tensor([[1., 0.], [0., 1.]])
layer.projection.bias.data = torch.Tensor([0., 0.])

with torch.no_grad():
    out_feats = layer(node_feats, adj_matrix)

print("Adjacency matrix", adj_matrix)
print("Input features", node_feats)
print("Output features", out_feats)
```

<!-- #region id="L_JuW8BgPULc" -->
As we can see, the first node's output values are the average of itself and the second node. Similarly, we can verify all other nodes. However, in a GNN, we would also want to allow feature exchange between nodes beyond its neighbors. This can be achieved by applying multiple GCN layers, which gives us the final layout of a GNN. The GNN can be build up by a sequence of GCN layers and non-linearities such as ReLU. For a visualization, see below (figure credit - [Thomas Kipf, 2016](https://tkipf.github.io/graph-convolutional-networks/)).
<!-- #endregion -->

<!-- #region id="MDWG9Lo5PV8j" -->
![](data:image/png;base64,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)
<!-- #endregion -->

<!-- #region id="i_nHrF0YJd1f" -->
However, one issue we can see from looking at the example above is that the output features for nodes 3 and 4 are the same because they have the same adjacent nodes (including itself). Therefore, GCN layers can make the network forget node-specific information if we just take a mean over all messages. Multiple possible improvements have been proposed. While the simplest option might be using residual connections, the more common approach is to either weigh the self-connections higher or define a separate weight matrix for the self-connections. Alternatively, we can re-visit a familiar concept: attention. 
<!-- #endregion -->

<!-- #region id="G2RHXjTISmo8" -->
## Graph Attention 

If you remember from the last tutorial, attention describes a weighted average of multiple elements with the weights dynamically computed based on an input query and elements' keys (if you haven't read Tutorial 6 yet, it is recommended to at least go through the very first section called [What is Attention?](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html#What-is-Attention?)). This concept can be similarly applied to graphs, one of such is the Graph Attention Network (called GAT, proposed by [Velickovic et al., 2017](https://arxiv.org/abs/1710.10903)). Similarly to the GCN, the graph attention layer creates a message for each node using a linear layer/weight matrix. For the attention part, it uses the message from the node itself as a query, and the messages to average as both keys and values (note that this also includes the message to itself). The score function $f_{attn}$ is implemented as a one-layer MLP which maps the query and key to a single value. The MLP looks as follows (figure credit - [Velickovic et al.](https://arxiv.org/abs/1710.10903)):
<!-- #endregion -->

<!-- #region id="kvCeqy8OSpHS" -->
![](data:image/png;base64,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)
<!-- #endregion -->

<!-- #region id="neiG2SqzJd1h" -->
$h_i$ and $h_j$ are the original features from node $i$ and $j$ respectively, and represent the messages of the layer with $\mathbf{W}$ as weight matrix. $\mathbf{a}$ is the weight matrix of the MLP, which has the shape $[1,2\times d_{\text{message}}]$, and $\alpha_{ij}$ the final attention weight from node $i$ to $j$. The calculation can be described as follows:

$$\alpha_{ij} = \frac{\exp\left(\text{LeakyReLU}\left(\mathbf{a}\left[\mathbf{W}h_i||\mathbf{W}h_j\right]\right)\right)}{\sum_{k\in\mathcal{N}_i} \exp\left(\text{LeakyReLU}\left(\mathbf{a}\left[\mathbf{W}h_i||\mathbf{W}h_k\right]\right)\right)}$$

The operator $||$ represents the concatenation, and $\mathcal{N}_i$ the indices of the neighbors of node $i$. Note that in contrast to usual practice, we apply a non-linearity (here LeakyReLU) before the softmax over elements. Although it seems like a minor change at first, it is crucial for the attention to depend on the original input. Specifically, let's remove the non-linearity for a second, and try to simplify the expression:

$$
\begin{split}
    \alpha_{ij} & = \frac{\exp\left(\mathbf{a}\left[\mathbf{W}h_i||\mathbf{W}h_j\right]\right)}{\sum_{k\in\mathcal{N}_i} \exp\left(\mathbf{a}\left[\mathbf{W}h_i||\mathbf{W}h_k\right]\right)}\\[5pt]
    & = \frac{\exp\left(\mathbf{a}_{:,:d/2}\mathbf{W}h_i+\mathbf{a}_{:,d/2:}\mathbf{W}h_j\right)}{\sum_{k\in\mathcal{N}_i} \exp\left(\mathbf{a}_{:,:d/2}\mathbf{W}h_i+\mathbf{a}_{:,d/2:}\mathbf{W}h_k\right)}\\[5pt]
    & = \frac{\exp\left(\mathbf{a}_{:,:d/2}\mathbf{W}h_i\right)\cdot\exp\left(\mathbf{a}_{:,d/2:}\mathbf{W}h_j\right)}{\sum_{k\in\mathcal{N}_i} \exp\left(\mathbf{a}_{:,:d/2}\mathbf{W}h_i\right)\cdot\exp\left(\mathbf{a}_{:,d/2:}\mathbf{W}h_k\right)}\\[5pt]
    & = \frac{\exp\left(\mathbf{a}_{:,d/2:}\mathbf{W}h_j\right)}{\sum_{k\in\mathcal{N}_i} \exp\left(\mathbf{a}_{:,d/2:}\mathbf{W}h_k\right)}\\
\end{split}
$$

We can see that without the non-linearity, the attention term with $h_i$ actually cancels itself out, resulting in the attention being independent of the node itself. Hence, we would have the same issue as the GCN of creating the same output features for nodes with the same neighbors. This is why the LeakyReLU is crucial and adds some dependency on $h_i$ to the attention. 

Once we obtain all attention factors, we can calculate the output features for each node by performing the weighted average:

$$h_i'=\sigma\left(\sum_{j\in\mathcal{N}_i}\alpha_{ij}\mathbf{W}h_j\right)$$

$\sigma$ is yet another non-linearity, as in the GCN layer. Visually, we can represent the full message passing in an attention layer as follows (figure credit - [Velickovic et al.](https://arxiv.org/abs/1710.10903)):
<!-- #endregion -->

<!-- #region id="lKnT7Grl8Fsh" -->
<!-- #endregion -->

<!-- #region id="2jwDUpz9StK8" -->
To increase the expressiveness of the graph attention network, [Velickovic et al.](https://arxiv.org/abs/1710.10903) proposed to extend it to multiple heads similar to the Multi-Head Attention block in Transformers. This results in $N$ attention layers being applied in parallel. In the image above, it is visualized as three different colors of arrows (green, blue, and purple) that are afterward concatenated. The average is only applied for the very final prediction layer in a network. 

After having discussed the graph attention layer in detail, we can implement it below:
<!-- #endregion -->

```python id="o_169MUj-KOr"
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
```

```python id="8GKQh008O0HT"
class GATLayer(nn.Module):
    
    def __init__(self, c_in, c_out, num_heads=1, concat_heads=True, alpha=0.2):
        """
        Inputs:
            c_in - Dimensionality of input features
            c_out - Dimensionality of output features
            num_heads - Number of heads, i.e. attention mechanisms to apply in parallel. The 
                        output features are equally split up over the heads if concat_heads=True.
            concat_heads - If True, the output of the different heads is concatenated instead of averaged.
            alpha - Negative slope of the LeakyReLU activation.
        """
        super().__init__()
        self.num_heads = num_heads
        self.concat_heads = concat_heads
        if self.concat_heads:
            assert c_out % num_heads == 0, "Number of output features must be a multiple of the count of heads."
            c_out = c_out // num_heads
        
        # Sub-modules and parameters needed in the layer
        self.projection = nn.Linear(c_in, c_out * num_heads)
        self.a = nn.Parameter(torch.Tensor(num_heads, 2 * c_out)) # One per head
        self.leakyrelu = nn.LeakyReLU(alpha)
        
        # Initialization from the original implementation
        nn.init.xavier_uniform_(self.projection.weight.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
    def forward(self, node_feats, adj_matrix, print_attn_probs=False):
        """
        Inputs:
            node_feats - Input features of the node. Shape: [batch_size, c_in]
            adj_matrix - Adjacency matrix including self-connections. Shape: [batch_size, num_nodes, num_nodes]
            print_attn_probs - If True, the attention weights are printed during the forward pass (for debugging purposes)
        """
        batch_size, num_nodes = node_feats.size(0), node_feats.size(1)
        
        # Apply linear layer and sort nodes by head
        node_feats = self.projection(node_feats)
        node_feats = node_feats.view(batch_size, num_nodes, self.num_heads, -1)
        
        # We need to calculate the attention logits for every edge in the adjacency matrix 
        # Doing this on all possible combinations of nodes is very expensive
        # => Create a tensor of [W*h_i||W*h_j] with i and j being the indices of all edges
        edges = adj_matrix.nonzero(as_tuple=False) # Returns indices where the adjacency matrix is not 0 => edges
        node_feats_flat = node_feats.view(batch_size * num_nodes, self.num_heads, -1)
        edge_indices_row = edges[:,0] * num_nodes + edges[:,1]
        edge_indices_col = edges[:,0] * num_nodes + edges[:,2]
        a_input = torch.cat([
            torch.index_select(input=node_feats_flat, index=edge_indices_row, dim=0),
            torch.index_select(input=node_feats_flat, index=edge_indices_col, dim=0)
        ], dim=-1) # Index select returns a tensor with node_feats_flat being indexed at the desired positions along dim=0
        
        # Calculate attention MLP output (independent for each head)
        attn_logits = torch.einsum('bhc,hc->bh', a_input, self.a) 
        attn_logits = self.leakyrelu(attn_logits)
        
        # Map list of attention values back into a matrix
        attn_matrix = attn_logits.new_zeros(adj_matrix.shape+(self.num_heads,)).fill_(-9e15)
        attn_matrix[adj_matrix[...,None].repeat(1,1,1,self.num_heads) == 1] = attn_logits.reshape(-1)
        
        # Weighted average of attention
        attn_probs = F.softmax(attn_matrix, dim=2)
        if print_attn_probs:
            print("Attention probs\n", attn_probs.permute(0, 3, 1, 2))
        node_feats = torch.einsum('bijh,bjhc->bihc', attn_probs, node_feats)
        
        # If heads should be concatenated, we can do this by reshaping. Otherwise, take mean
        if self.concat_heads:
            node_feats = node_feats.reshape(batch_size, num_nodes, -1)
        else:
            node_feats = node_feats.mean(dim=2)
        
        return node_feats 
```

<!-- #region id="16j4vshTJd1l" -->
Again, we can apply the graph attention layer on our example graph above to understand the dynamics better. As before, the input layer is initialized as an identity matrix, but we set $\mathbf{a}$ to be a vector of arbitrary numbers to obtain different attention values. We use two heads to show the parallel, independent attention mechanisms working in the layer.
<!-- #endregion -->

```python id="BiP4hn7HJd1m" colab={"base_uri": "https://localhost:8080/"} outputId="93369719-35e3-4bf4-a6b0-544a3d4e0366"
layer = GATLayer(2, 2, num_heads=2)
layer.projection.weight.data = torch.Tensor([[1., 0.], [0., 1.]])
layer.projection.bias.data = torch.Tensor([0., 0.])
layer.a.data = torch.Tensor([[-0.2, 0.3], [0.1, -0.1]])

with torch.no_grad():
    out_feats = layer(node_feats, adj_matrix, print_attn_probs=True)

print("Adjacency matrix", adj_matrix)
print("Input features", node_feats)
print("Output features", out_feats)
```

<!-- #region id="EX__hkkOJd1o" -->
We recommend that you try to calculate the attention matrix at least for one head and one node for yourself. The entries are 0 where there does not exist an edge between $i$ and $j$. For the others, we see a diverse set of attention probabilities. Moreover, the output features of node 3 and 4 are now different although they have the same neighbors.
<!-- #endregion -->

<!-- #region id="kNvD63-qmKBK" -->
## Convolution Fundamentals
<!-- #endregion -->

<!-- #region id="-s6Oz4zujU1l" -->
**Why convolution in ML?**

- Weight sharing
- Detection of translational invariant and local features
<!-- #endregion -->

<!-- #region id="7YAtQEgSjU1t" -->
![](https://github.com/AntonioLonga/PytorchGeometricTutorial/blob/main/Tutorial4/fig/Convolution_of_box_signal_with_itself2.gif?raw=1)

[Source](https://en.wikipedia.org/wiki/File:Convolution_of_box_signal_with_itself2.gif)
<!-- #endregion -->

<!-- #region id="31nP7bAUmoOO" -->
### Imports
<!-- #endregion -->

```python id="5EZLWtKTmncC"
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})
```

<!-- #region id="1-SFGEbOjU1u" -->
### Definition
<!-- #endregion -->

<!-- #region id="6tc5FeJAjU1v" -->
\begin{align*}
c[n] = (v * w)[n] = \sum_{m=0}^{N-1} v[m] \cdot w[n-m]
\end{align*}
<!-- #endregion -->

```python id="P-w3eN4njU1x"
def conv(v, w):
    c = np.zeros(v.shape)
    for n in range(len(v)):
        c[n] = 0
        for m in range(len(v)):
            c[n] += v[m] * w[n - m]  
    return c
```

```python id="yWuC2SVijU1z" colab={"base_uri": "https://localhost:8080/", "height": 265} outputId="8f30bb14-a141-49ca-9f2f-f88cf7612c63"
N = 20
v = np.zeros(N)
v[8:12] = 1
w = np.zeros(N)
w[1:5] = 1
c = conv(v, w)

fig = plt.figure()
ax = fig.gca()
ax.plot(v, '.-')
ax.plot(w, '.-')
ax.plot(c, '.-')
ax.legend(['v', 'w', 'c'])
ax.grid(True)
```

<!-- #region id="_ub3xokljU11" -->
### Fourier transform
<!-- #endregion -->

<!-- #region id="MMc7h-07jU12" -->
Transformation $\mathcal F: \mathbb{R}^N \to \mathbb{R}^N$ with

\begin{align*}
\mathcal F^{-1}(\mathcal F (v)) &= v\\
\mathcal F(v * w) &= \mathcal F(v) \cdot \mathcal F(w).
\end{align*}
<!-- #endregion -->

<!-- #region id="eiFAIAZnjU12" -->
This implies
\begin{align*}
v * w &= \mathcal F^{-1}(\mathcal F (v * w))\\
&= \mathcal F^{-1}(\mathcal F(v) \cdot \mathcal F(w))
\end{align*}
<!-- #endregion -->

```python id="wZBnJFtXjU13" colab={"base_uri": "https://localhost:8080/"} outputId="dcb0378a-5082-4a35-a01e-a8fb2d9875a8"
v, w = np.random.rand(N), np.random.rand(N)
conv(v, w)
```

```python id="S6y-jQyljU14" colab={"base_uri": "https://localhost:8080/"} outputId="70ac4870-753e-4561-d529-9b25f9ebf32b"
from scipy.fft import fft, ifft # Fast Fourier Transform / Inverse FFT
np.abs(ifft(fft(v) * fft(w)))
```

<!-- #region id="twDIgZFFjU16" -->
#### Definition of the Fourier transform
<!-- #endregion -->

<!-- #region id="sW7sZJ08jU17" -->
The Fourier transform can be computed as

\begin{align*}
\mathcal F(v) = U\cdot v, \;\;\mathcal F^{-1}(v) = \frac{1}{N}\ U^H \cdot v
\end{align*}

where the $N\times N$ matrix $U$ is defined as
\begin{align*}
\\
U = 
\begin{bmatrix}
u_0(0) & u_1(0) & \dots & u_{N-1}(0)\\
u_0(1) & u_1(1) & \dots & u_{N-1}(1)\\
\vdots & \vdots& & \vdots\\
u_0(N-1) & u_1(N-1) & \dots & u_{N-1}(N-1)\\
\end{bmatrix} 
\end{align*}

and $u_0, \dots, u_{N-1}$ are functions defined as

\begin{align*}
u_n(x)&:= \cos\left(2 \pi \frac{n}{N} x\right) - i \sin\left(2 \pi \frac{n}{N} x\right).
\end{align*}
<!-- #endregion -->

```python id="UF6lObDzjU18"
def matrix_U(N):
    u = lambda n, N: np.cos(2 * np.pi / N * n * np.arange(N)) - 1j * np.sin(2 * np.pi / N * n * np.arange(N))
    U = np.empty((N, 0))
    for n in range(N):
        U = np.c_[U, u(n, N)]
    return U


def fourier_transform(v):
    N = v.shape[0]
    U = matrix_U(N)
    return U @ v


def inverse_fourier_transform(v):
    N = v.shape[0]
    U = matrix_U(N)
    return (U.conj().transpose() @ v) / N
```

```python id="NGjk8BnrjU19" colab={"base_uri": "https://localhost:8080/"} outputId="f5bdbd3e-7afb-4271-a931-d38ad7f2dd4f"
fft(v) - fourier_transform(v)
```

```python id="3bYkgQCVjU1-" colab={"base_uri": "https://localhost:8080/"} outputId="30cc2d55-63fe-4050-cc8d-791865a5000c"
ifft(v) - inverse_fourier_transform(v)
```

<!-- #region id="5h5jdqCujU1_" -->
#### Connection with the Laplacian
<!-- #endregion -->

<!-- #region id="ABnRkGpVjU2A" -->
The functions $u_n$ (the columns of the Fourier transform matrix) are eigenvectors of the Laplacian:

\begin{align*}
u_n(x)&:= \cos\left(2 \pi \frac{n}{N} x\right) - i \sin\left(2 \pi \frac{n}{N} x\right)\\
\Delta u_n(x)&:= \left(-4 \pi\frac{n^2}{N^2}\right) u_n(x)
\end{align*}
<!-- #endregion -->

<!-- #region id="uETONhRcjU2B" -->
#### Summary
<!-- #endregion -->

<!-- #region id="IeeGZUUEjU2D" -->
\begin{align*}
v * w 
= U^H ((U  w) \odot (U  v))
\end{align*}

or if $g_w=\mbox{diag}(U w)$ is  filter
\begin{align*}
v * w 
= U^H g_w U  w
\end{align*}
<!-- #endregion -->

```python id="INwk9GfVjU2D" colab={"base_uri": "https://localhost:8080/"} outputId="f1e846a3-9110-46e0-9167-af428c1e5278"
U = matrix_U(N)
np.abs((U.conj().transpose() / N) @ ((U @ v) * (U @ w)))
```

```python id="uYiFMzD0jU2E" colab={"base_uri": "https://localhost:8080/"} outputId="eed7a679-fd6b-4d82-99d0-753edc7e9acb"
conv(v, w)
```

<!-- #region id="81CxozvcjU2F" -->
### Convolution on graphs
<!-- #endregion -->

<!-- #region id="XTLi4iv4jU2F" -->
**Plan**:
- Define the graph Laplacian
- Compute the spectrum
- Define a Fourier transform
- Define convolution on a graph
<!-- #endregion -->

<!-- #region id="llJtDpsqjU2G" -->
**Note:** From now on $G = (V, E)$ is an undirected, unweighted, simple graph.
<!-- #endregion -->

<!-- #region id="-xGl79_djU2J" -->
#### Graph Laplacian
<!-- #endregion -->

<!-- #region id="krHNV6BDjU2K" -->
Adjacency matrix
\begin{align*}
A_{ij} = \left\{
    \begin{array}{ll}
    1 &\text{ if } e_{ij}\in E\\
    0 &\text{ if } e_{ij}\notin E
    \end{array}
    \right.
\end{align*}

Degree matrix
\begin{align*}
D_{ij} = \left\{
    \begin{array}{ll}
    \mbox{deg}(v_i) &\text{ if } i=j\\
    0 &\text{ if } i\neq j
    \end{array}
    \right.
\end{align*}

Laplacian
\begin{align*}
L &= D - A.
\end{align*}

Normalized Laplacian
\begin{align*}
L &= I - D^{-1/2} A D^{-1/2}.
\end{align*}
<!-- #endregion -->

<!-- #region id="2cd_QoNejU2K" -->
#### Graph spectrum, Fourier transform, and convolution
<!-- #endregion -->

<!-- #region id="HuDedISkjU2L" -->
1. Spectral decomposition of the Laplacian:
\begin{align*}
L = U \Lambda U^T\\
\end{align*}


2. Fourier transform: if $v$ is a vector of features on the graph, then
\begin{align*}
\mathcal F (v) = U \cdot v, \;\;\mathcal F^{-1} (v) = U^T \cdot v\\
\end{align*}


3. Convolution with a filter $U \cdot w$
\begin{align*}
v * w = U ((U^T  w) \odot (U^T  v) )
\end{align*}


Or $g_w = \mbox{diag}(U^T w)$ is a filter, then
\begin{align*}
v * w = U g_w U^T  v
\end{align*}

<!-- #endregion -->

<!-- #region id="o-4jTYNqjU2M" -->
## Spectral-convolutional layers in PyTorch Geometric
<!-- #endregion -->

<!-- #region id="ymGdvbfpjU2M" -->
**Problem:** Computing the spectrum is a global and very expensive property.

**Goal:** Implementation as message passing.
<!-- #endregion -->

<!-- #region id="BiwOrRmpjU2N" -->
### ChebConv
<!-- #endregion -->

<!-- #region id="prNjVo4mjU2N" -->
- Original [paper](https://arxiv.org/pdf/1606.09375.pdf)
- PyTorch [doc](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.ChebConv)
<!-- #endregion -->

<!-- #region id="sbn9saMpjU2O" -->
#### Goal: 
Compute $U g_w U^T x$ with $g_w = g_w(\Lambda)$ a filter.
<!-- #endregion -->

<!-- #region id="81fAsO0_jU2O" -->
#### Chebyshev approximation

Chebyshev polynomials $T_k$:
\begin{align*}
T_{k}(x) = 2 x T_{k-1}(x) - T_{k-2}(x), \;\; T_0(x) = 1, T_1(x) = x
\end{align*}

#### Chebyshev approximation of the filter
Aproximation of the filter:
\begin{align*}
g_w(\Lambda) = \sum_{k=0}^K \theta_k T_k(\tilde \Lambda),\;\;\;\;\tilde \Lambda = \frac{2}{\lambda_\max} \Lambda - I \cdot \lambda_\max
\end{align*}


#### Property
If $L = U \Lambda U^T$ then $T_k(L) = U T_k(\Lambda) U^T$.

<!-- #endregion -->

<!-- #region id="T3iEp4oZjU2O" -->
#### Fast approximated convolution 
\begin{align*}
v * w &= U g_w U^T x
= U \left(\sum_{k=0}^K \theta_k T_k(\tilde \Lambda) \right)U^T x
=\sum_{k=0}^K  \theta_k U  T_k(\tilde \Lambda) U^T x\\ 
&=\sum_{k=0}^K  \theta_k T_k(\tilde L) x 
\end{align*}

\begin{align*}
\tilde L = \frac{2}{\lambda_\max} L - I
\end{align*}
<!-- #endregion -->

<!-- #region id="XT68q77zjU2P" -->
#### Properties:
- Depends on $L$ and $\lambda_\max$, not on $U, \Sigma$
- Uses only $K$-powers $\Rightarrow$ only the $K$-th neighborhood of each node, localized filter
<!-- #endregion -->

<!-- #region id="z2hIk9MIjU2Q" -->
**As message passing:**

![](https://github.com/AntonioLonga/PytorchGeometricTutorial/blob/main/Tutorial4/fig/cheb_init.png?raw=1)


![](https://github.com/AntonioLonga/PytorchGeometricTutorial/blob/main/Tutorial4/fig/cheb_norm.png?raw=1)


![](https://github.com/AntonioLonga/PytorchGeometricTutorial/blob/main/Tutorial4/fig/cheb_forward.png?raw=1)


![](https://github.com/AntonioLonga/PytorchGeometricTutorial/blob/main/Tutorial4/fig/cheb_message.png?raw=1)
<!-- #endregion -->

<!-- #region id="Xuz5GKV0jU2R" -->
### GCNConv
<!-- #endregion -->

<!-- #region id="bjhjOvQMjU2S" -->
- Original [paper](https://arxiv.org/pdf/1609.02907.pdf)
- PyTorch [doc](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GCNConv)
<!-- #endregion -->

<!-- #region id="ec6WKPXRjU2S" -->
Start from `ChebConv` and assume 
1. $K=1$ (linear approximation) so
\begin{align*}
v * w 
&=\sum_{k=0}^1  \theta_k T_k(\tilde L) x
= \theta_0 x + \theta_1 \tilde L x\\
\end{align*}

2. $\lambda_\max =2$ so
\begin{align*}
v * w 
&= \theta_0 x + \theta_1 (L - I) x\\
&= \theta_0 x - \theta_1 D^{-1/2} A D^{1/2} x\\
\end{align*}


3. $\theta_0=-\theta_1= \theta$ so 
\begin{align*}
v * w = \left(I + D^{-1/2} A D^{1/2}\right) x \theta
\end{align*}

4. Renormalization of $\theta$ by using 
\begin{align*}
\tilde A&:= I + A\\
\tilde D_{ii}&:= \sum_j \tilde A_{ij}
\end{align*}
so 
\begin{align*}
v * w = \left(D^{-1/2} A D^{1/2}\right) x \theta
\end{align*}

If $x$ is a $F$-dimensional feature vector, and we want an $F'$-dimensional feature vector as output:
use $W'\in \mathbb{R}^{F\times F'}$
\begin{align*}
v * w = \left(D^{-1/2} A D^{1/2}\right) x \Theta
\end{align*}

<!-- #endregion -->

<!-- #region id="FxMRHNrKjU2T" -->
Nodewise:

![image.png](https://github.com/AntonioLonga/PytorchGeometricTutorial/blob/main/Tutorial4/fig/gcn_nodewise.png?raw=1)
<!-- #endregion -->

<!-- #region id="OiSa_k7Mr7ov" -->
## Aggregation Functions in GNNs
<!-- #endregion -->

<!-- #region id="CJGFIV83tIfx" -->
### Context
<!-- #endregion -->

<!-- #region id="EIkXNFKmsXhr" -->
We explore how to perform neighborhood aggregation in GNNs, describing the GIN model and other recent techniques for selecting the right aggregation (PNA) or learn it (LAF).

We will override the aggregation method of the GIN convolution module of Pytorch Geometric implementing the following methods:

- Principal Neighborhood Aggregation (PNA)
- Learning Aggregation Functions (LAF)
<!-- #endregion -->

<!-- #region id="_Z30Q0IeuVQp" -->
### WL Isomorphism Test
<!-- #endregion -->

<!-- #region id="2DrPGN1pvhOH" -->
**Step 1**
<!-- #endregion -->

<!-- #region id="USskVxG4veZS" -->
<!-- #endregion -->

<!-- #region id="KHf0gVkTvmsk" -->
**Step 2**
<!-- #endregion -->

<!-- #region id="qbXMBYaDvoQr" -->
<!-- #endregion -->

<!-- #region id="L5iTSBldxhJ-" -->
<!-- #endregion -->

<!-- #region id="lDpSF5orswqp" -->
### Imports
<!-- #endregion -->

```python id="PwrdyvfbseoG" colab={"base_uri": "https://localhost:8080/"} outputId="1e41f55f-fde7-4062-fbad-97f7af1cca67"
import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import GINConv
from torch.nn import Linear
from torch_geometric.nn import MessagePassing, SAGEConv, GINConv, global_add_pool
import torch_scatter
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch.nn import Parameter, Module, Sigmoid
import os.path as osp

torch.manual_seed(42)
```

<!-- #region id="lplheNqFseoS" -->
### Message Passing Class
<!-- #endregion -->

<!-- #region id="Cvl2EURpseoW" -->
We are interested in the <span style='color:Blue'>aggregate</span> method, or, if you are using a sparse adjacency matrix, in the <span style='color:Blue'>message_and_aggregate</span> method. Convolutional classes in PyG extend MessagePassing, we construct our custom convoutional class extending GINConv.
<!-- #endregion -->

<!-- #region id="D220S_OrseoY" -->
Scatter operation in <span style='color:Blue'>aggregate</span>:
<!-- #endregion -->

<!-- #region id="80xKlB8oseoZ" -->
![](https://raw.githubusercontent.com/rusty1s/pytorch_scatter/master/docs/source/_figures/add.svg?sanitize=true)
<!-- #endregion -->

<!-- #region id="YtHRWKDdtn7K" -->
### LAF Aggregation Module
<!-- #endregion -->

<!-- #region id="rvmNHu9Etn7L" -->
![](https://github.com/AntonioLonga/PytorchGeometricTutorial/blob/main/Tutorial5/laf.png?raw=true)
<!-- #endregion -->

<!-- #region id="C3R487ld0vHM" -->
**LAF Layer**
<!-- #endregion -->

```python id="mpsMsIF00wcV"
class AbstractLAFLayer(Module):
    def __init__(self, **kwargs):
        super(AbstractLAFLayer, self).__init__()
        assert 'units' in kwargs or 'weights' in kwargs
        if 'device' in kwargs.keys():
            self.device = kwargs['device']
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ngpus = torch.cuda.device_count()
        
        if 'kernel_initializer' in kwargs.keys():
            assert kwargs['kernel_initializer'] in [
                'random_normal',
                'glorot_normal',
                'he_normal',
                'random_uniform',
                'glorot_uniform',
                'he_uniform']
            self.kernel_initializer = kwargs['kernel_initializer']
        else:
            self.kernel_initializer = 'random_normal'

        if 'weights' in kwargs.keys():
            self.weights = Parameter(kwargs['weights'].to(self.device), \
                                     requires_grad=True)
            self.units = self.weights.shape[1]
        else:
            self.units = kwargs['units']
            params = torch.empty(12, self.units, device=self.device)
            if self.kernel_initializer == 'random_normal':
                torch.nn.init.normal_(params)
            elif self.kernel_initializer == 'glorot_normal':
                torch.nn.init.xavier_normal_(params)
            elif self.kernel_initializer == 'he_normal':
                torch.nn.init.kaiming_normal_(params)
            elif self.kernel_initializer == 'random_uniform':
                torch.nn.init.uniform_(params)
            elif self.kernel_initializer == 'glorot_uniform':
                torch.nn.init.xavier_uniform_(params)
            elif self.kernel_initializer == 'he_uniform':
                torch.nn.init.kaiming_uniform_(params)
            self.weights = Parameter(params, \
                                     requires_grad=True)
        e = torch.tensor([1,-1,1,-1], dtype=torch.float32, device=self.device)
        self.e = Parameter(e, requires_grad=False)
        num_idx = torch.tensor([1,1,0,0], dtype=torch.float32, device=self.device).\
                                view(1,1,-1,1)
        self.num_idx = Parameter(num_idx, requires_grad=False)
        den_idx = torch.tensor([0,0,1,1], dtype=torch.float32, device=self.device).\
                                view(1,1,-1,1)
        self.den_idx = Parameter(den_idx, requires_grad=False)
        

class LAFLayer(AbstractLAFLayer):
    def __init__(self, eps=1e-7, **kwargs):
        super(LAFLayer, self).__init__(**kwargs)
        self.eps = eps
    
    def forward(self, data, index, dim=0, **kwargs):
        eps = self.eps
        sup = 1.0 - eps 
        e = self.e

        x = torch.clamp(data, eps, sup)
        x = torch.unsqueeze(x, -1)
        e = e.view(1,1,-1)        

        exps = (1. - e)/2. + x*e 
        exps = torch.unsqueeze(exps, -1)
        exps = torch.pow(exps, torch.relu(self.weights[0:4]))

        scatter = torch_scatter.scatter_add(exps, index.view(-1), dim=dim)
        scatter = torch.clamp(scatter, eps)

        sqrt = torch.pow(scatter, torch.relu(self.weights[4:8]))
        alpha_beta = self.weights[8:12].view(1,1,4,-1)
        terms = sqrt * alpha_beta

        num = torch.sum(terms * self.num_idx, dim=2)
        den = torch.sum(terms * self.den_idx, dim=2)
        
        multiplier = 2.0*torch.clamp(torch.sign(den), min=0.0) - 1.0

        den = torch.where((den < eps) & (den > -eps), multiplier*eps, den)

        res = num / den
        return res
```

```python id="XgskJll0tn7M"
class GINLAFConv(GINConv):
    def __init__(self, nn, units=1, node_dim=32, **kwargs):
        super(GINLAFConv, self).__init__(nn, **kwargs)
        self.laf = LAFLayer(units=units, kernel_initializer='random_uniform')
        self.mlp = torch.nn.Linear(node_dim*units, node_dim)
        self.dim = node_dim
        self.units = units
    
    def aggregate(self, inputs, index):
        x = torch.sigmoid(inputs)
        x = self.laf(x, index)
        x = x.view((-1, self.dim * self.units))
        x = self.mlp(x)
        return x
```

<!-- #region id="q0Qwqbrvtn7N" -->
### PNA Aggregation
<!-- #endregion -->

<!-- #region id="i29StuLltn7O" -->
![](https://github.com/AntonioLonga/PytorchGeometricTutorial/blob/main/Tutorial5/pna.png?raw=true)
<!-- #endregion -->

```python id="ysP9SOV5tn7P"
class GINPNAConv(GINConv):
    def __init__(self, nn, node_dim=32, **kwargs):
        super(GINPNAConv, self).__init__(nn, **kwargs)
        self.mlp = torch.nn.Linear(node_dim*12, node_dim)
        self.delta = 2.5749
    
    def aggregate(self, inputs, index):
        sums = torch_scatter.scatter_add(inputs, index, dim=0)
        maxs = torch_scatter.scatter_max(inputs, index, dim=0)[0]
        means = torch_scatter.scatter_mean(inputs, index, dim=0)
        var = torch.relu(torch_scatter.scatter_mean(inputs ** 2, index, dim=0) - means ** 2)
        
        aggrs = [sums, maxs, means, var]
        c_idx = index.bincount().float().view(-1, 1)
        l_idx = torch.log(c_idx + 1.)
        
        amplification_scaler = [c_idx / self.delta * a for a in aggrs]
        attenuation_scaler = [self.delta / c_idx * a for a in aggrs]
        combinations = torch.cat(aggrs+ amplification_scaler+ attenuation_scaler, dim=1)
        x = self.mlp(combinations)
    
        return x
```

<!-- #region id="dRiavOjHtn7Q" -->
### Test the new classes
<!-- #endregion -->

```python id="8L6L2VEltn7R" colab={"base_uri": "https://localhost:8080/"} outputId="dcbe9ec6-2185-4885-c678-8451b4a8b4d8"
path = osp.join('./', 'data', 'TU')
dataset = TUDataset(path, name='MUTAG').shuffle()
test_dataset = dataset[:len(dataset) // 10]
train_dataset = dataset[len(dataset) // 10:]
test_loader = DataLoader(test_dataset, batch_size=128)
train_loader = DataLoader(train_dataset, batch_size=128)
```

<!-- #region id="KfmUuo0F2H5o" -->
### LAF Model
<!-- #endregion -->

```python id="9UJS6CRltn7S"
class LAFNet(torch.nn.Module):
    def __init__(self):
        super(LAFNet, self).__init__()

        num_features = dataset.num_features
        dim = 32
        units = 3
        
        nn1 = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
        self.conv1 = GINLAFConv(nn1, units=units, node_dim=num_features)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv2 = GINLAFConv(nn2, units=units, node_dim=dim)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv3 = GINLAFConv(nn3, units=units, node_dim=dim)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        nn4 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv4 = GINLAFConv(nn4, units=units, node_dim=dim)
        self.bn4 = torch.nn.BatchNorm1d(dim)

        nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv5 = GINLAFConv(nn5, units=units, node_dim=dim)
        self.bn5 = torch.nn.BatchNorm1d(dim)

        self.fc1 = Linear(dim, dim)
        self.fc2 = Linear(dim, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = F.relu(self.conv5(x, edge_index))
        x = self.bn5(x)
        x = global_add_pool(x, batch)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)

```

<!-- #region id="O4QzD1Z72FKD" -->
### PNA Model
<!-- #endregion -->

```python id="Zw63xP4ctn7T"
class PNANet(torch.nn.Module):
    def __init__(self):
        super(PNANet, self).__init__()

        num_features = dataset.num_features
        dim = 32

        nn1 = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
        self.conv1 = GINPNAConv(nn1, node_dim=num_features)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv2 = GINPNAConv(nn2, node_dim=dim)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv3 = GINPNAConv(nn3, node_dim=dim)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        nn4 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv4 = GINPNAConv(nn4, node_dim=dim)
        self.bn4 = torch.nn.BatchNorm1d(dim)

        nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv5 = GINPNAConv(nn5, node_dim=dim)
        self.bn5 = torch.nn.BatchNorm1d(dim)

        self.fc1 = Linear(dim, dim)
        self.fc2 = Linear(dim, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = F.relu(self.conv5(x, edge_index))
        x = self.bn5(x)
        x = global_add_pool(x, batch)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)
```

<!-- #region id="K1XJf1R42CmD" -->
### GIN Model
<!-- #endregion -->

```python id="PsbosIB4tn7V"
class GINNet(torch.nn.Module):
    def __init__(self):
        super(GINNet, self).__init__()

        num_features = dataset.num_features
        dim = 32

        nn1 = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv3 = GINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        nn4 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv4 = GINConv(nn4)
        self.bn4 = torch.nn.BatchNorm1d(dim)

        nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv5 = GINConv(nn5)
        self.bn5 = torch.nn.BatchNorm1d(dim)

        self.fc1 = Linear(dim, dim)
        self.fc2 = Linear(dim, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = F.relu(self.conv5(x, edge_index))
        x = self.bn5(x)
        x = global_add_pool(x, batch)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)
```

<!-- #region id="dP8Ca1Of2ZMW" -->
### Training
<!-- #endregion -->

```python id="E_uoVQX-2YFW"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = "PNA"
if net == "LAF":
    model = LAFNet().to(device)
elif net == "PNA":
    model = PNANet().to(device)
elif net == "GIN":
    GINNet().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train(epoch):
    model.train()

    if epoch == 51:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.5 * param_group['lr']

    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data.x, data.edge_index, data.batch)
        loss = F.nll_loss(output, data.y)
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()
    return loss_all / len(train_dataset)


def test(loader):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        output = model(data.x, data.edge_index, data.batch)
        pred = output.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)
```

```python id="jd8O2SB_tn7W" colab={"base_uri": "https://localhost:8080/"} outputId="99ff799f-56b3-4816-a43c-4ab81100ebfb"
for epoch in range(1, 101):
    train_loss = train(epoch)
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    print('Epoch: {:03d}, Train Loss: {:.7f}, '
          'Train Acc: {:.7f}, Test Acc: {:.7f}'.format(epoch, train_loss,
                                                       train_acc, test_acc))
```

```python id="1MUjl0Sf6iym"
import torch
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from torch_geometric.utils import train_test_split_edges
```

<!-- #region id="bO7Cksrg6iyt" -->
## Graph AutoEncoders - GAE & VGAE    
<!-- #endregion -->

<!-- #region id="B5Ffn8dJ7Ukv" -->
[paper](https://arxiv.org/pdf/1611.07308.pdf)  
[code](https://github.com/rusty1s/pytorch_geometric/blob/master/examples/autoencoder.py)
<!-- #endregion -->

<!-- #region id="Geskh6iL8iIm" -->
### Context
<!-- #endregion -->

<!-- #region id="KWz6PRHT8jrG" -->
<!-- #endregion -->

<!-- #region id="M15cBj408yg-" -->
<!-- #endregion -->

<!-- #region id="bHIpDgbW8_Cv" -->
#### Loss function
<!-- #endregion -->

<!-- #region id="d3hCUama9BOM" -->
<!-- #endregion -->

<!-- #region id="oSlE-8HH7cBM" -->
### Imports
<!-- #endregion -->

```python id="ROnCeb8m7dHz"
from torch_geometric.nn import GAE
from torch_geometric.nn import VGAE
from torch.utils.tensorboard import SummaryWriter
import torch
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from torch_geometric.utils import train_test_split_edges
```

<!-- #region id="kqtK6kIl6iyx" -->
### Load the CiteSeer data
<!-- #endregion -->

```python id="xJcHe3-U6iyy" colab={"base_uri": "https://localhost:8080/"} outputId="687f2324-402d-4c38-cd0f-b56bd83aa644"
dataset = Planetoid("\..", "CiteSeer", transform=T.NormalizeFeatures())
dataset.data
```

```python id="wFajQVdR6iyz" colab={"base_uri": "https://localhost:8080/"} outputId="c5690042-afb3-41a3-986f-7254fbc0446c"
data = dataset[0]
data.train_mask = data.val_mask = data.test_mask = None
data
```

```python id="ilpI-cCL6iy0" colab={"base_uri": "https://localhost:8080/"} outputId="43d89ce4-49f8-4c9a-89ea-eb349a605db6"
data = train_test_split_edges(data)
data
```

<!-- #region id="DTHs2luw6iy4" -->
### Define the Encoder
<!-- #endregion -->

```python id="e-9gsNuE6iy6"
class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True) # cached only for transductive learning
        self.conv2 = GCNConv(2 * out_channels, out_channels, cached=True) # cached only for transductive learning

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)
```

<!-- #region id="A5H_VoaH6iy9" -->
### Define the Autoencoder
<!-- #endregion -->

```python id="ZYYew3im6izA"
# parameters
out_channels = 2
num_features = dataset.num_features
epochs = 100

# model
model = GAE(GCNEncoder(num_features, out_channels))

# move to GPU (if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
x = data.x.to(device)
train_pos_edge_index = data.train_pos_edge_index.to(device)

# inizialize the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
```

```python id="hqrC2jls6izC"
def train():
    model.train()
    optimizer.zero_grad()
    z = model.encode(x, train_pos_edge_index)
    loss = model.recon_loss(z, train_pos_edge_index)
    #if args.variational:
    #   loss = loss + (1 / data.num_nodes) * model.kl_loss()
    loss.backward()
    optimizer.step()
    return float(loss)


def test(pos_edge_index, neg_edge_index):
    model.eval()
    with torch.no_grad():
        z = model.encode(x, train_pos_edge_index)
    return model.test(z, pos_edge_index, neg_edge_index)
```

```python id="SFPciyaw6izD" colab={"base_uri": "https://localhost:8080/"} outputId="decb264b-f120-40b1-8c7b-fb3251d3810d"
for epoch in range(1, epochs + 1):
    loss = train()

    auc, ap = test(data.test_pos_edge_index, data.test_neg_edge_index)
    print('Epoch: {:03d}, AUC: {:.4f}, AP: {:.4f}'.format(epoch, auc, ap))
```

```python id="v-m09mHT6izD" colab={"base_uri": "https://localhost:8080/"} outputId="9e52c43f-4606-40e0-c2e9-d370dcc2f79e"
Z = model.encode(x, train_pos_edge_index)
Z
```

<!-- #region id="2KqlZtCd6izF" -->
### Result analysis with Tensorboard
<!-- #endregion -->

```python id="UJctXayE6izG"
# parameters
out_channels = 2
num_features = dataset.num_features
epochs = 100

# model
model = GAE(GCNEncoder(num_features, out_channels))

# move to GPU (if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
x = data.x.to(device)
train_pos_edge_index = data.train_pos_edge_index.to(device)

# inizialize the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
```

```python id="FVqv-NxA6izH"
writer = SummaryWriter('runs/GAE1_experiment_'+'2d_100_epochs')
```

```python id="MSrnWNIq6izH" colab={"base_uri": "https://localhost:8080/"} outputId="7753c658-7736-4814-f0d0-64e2b1a70cdd"
for epoch in range(1, epochs + 1):
    loss = train()
    auc, ap = test(data.test_pos_edge_index, data.test_neg_edge_index)
    print('Epoch: {:03d}, AUC: {:.4f}, AP: {:.4f}'.format(epoch, auc, ap))
    
    
    writer.add_scalar('auc train',auc,epoch) # new line
    writer.add_scalar('ap train',ap,epoch)   # new line
```

<!-- #region id="vrjYGm2L6izI" -->
### Graph Variational AutoEncoder (GVAE)
<!-- #endregion -->

```python id="ATQTuF5Z6izJ"
dataset = Planetoid("\..", "CiteSeer", transform=T.NormalizeFeatures())
data = dataset[0]
data.train_mask = data.val_mask = data.test_mask = data.y = None
data = train_test_split_edges(data)


class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VariationalGCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True) # cached only for transductive learning
        self.conv_mu = GCNConv(2 * out_channels, out_channels, cached=True)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels, cached=True)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)
```

```python id="ORqNPw6v6izK"
out_channels = 2
num_features = dataset.num_features
epochs = 300


model = VGAE(VariationalGCNEncoder(num_features, out_channels))  # new line

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
x = data.x.to(device)
train_pos_edge_index = data.train_pos_edge_index.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
```

```python id="QXaCVkwY6izL"
def train():
    model.train()
    optimizer.zero_grad()
    z = model.encode(x, train_pos_edge_index)
    loss = model.recon_loss(z, train_pos_edge_index)
    
    loss = loss + (1 / data.num_nodes) * model.kl_loss()  # new line
    loss.backward()
    optimizer.step()
    return float(loss)


def test(pos_edge_index, neg_edge_index):
    model.eval()
    with torch.no_grad():
        z = model.encode(x, train_pos_edge_index)
    return model.test(z, pos_edge_index, neg_edge_index)
```

```python id="vS3PKukE6izN" colab={"base_uri": "https://localhost:8080/"} outputId="c2c95e67-72d1-4964-ab80-1b1fbb928f3b"
writer = SummaryWriter('runs/VGAE_experiment_'+'2d_100_epochs')

for epoch in range(1, epochs + 1):
    loss = train()
    auc, ap = test(data.test_pos_edge_index, data.test_neg_edge_index)
    print('Epoch: {:03d}, AUC: {:.4f}, AP: {:.4f}'.format(epoch, auc, ap))
    
    
    writer.add_scalar('auc train',auc,epoch) # new line
    writer.add_scalar('ap train',ap,epoch)   # new line
```

<!-- #region id="n83l4fRt9luz" -->
---
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="trEfQuxo9lu5" outputId="86f4bbba-495f-4d90-b6f1-e5303bfd5d13"
!pip install -q watermark
%reload_ext watermark
%watermark -a "Sparsh A." -m -iv -u -t -d
```

<!-- #region id="xe6ycIM89lu6" -->
---
<!-- #endregion -->

<!-- #region id="Cy6OjZxd9lu6" -->
**END**
<!-- #endregion -->
