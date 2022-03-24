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

<!-- #region id="4SuL06w0jfZR" -->
# Transformer from scratch in PyTorch
<!-- #endregion -->

<!-- #region id="PKbsUSTuiPQU" -->
Coding the scaled dot-product attention is pretty straightforward — just a few matrix multiplications, plus a softmax function. For added simplicity, we omit the optional Mask operation.
<!-- #endregion -->

```python id="fdvcDkbKiMA6"
from torch import Tensor
import torch.nn.functional as f


def scaled_dot_product_attention(query: Tensor, key: Tensor, value: Tensor) -> Tensor:
    temp = query.bmm(key.transpose(1, 2))
    scale = query.size(-1) ** 0.5
    softmax = f.softmax(temp / scale, dim=-1)
    return softmax.bmm(value)
```

<!-- #region id="fiLDxyPJiQty" -->
Note that MatMul operations are translated to torch.bmm in PyTorch. That’s because Q, K, and V (query, key, and value arrays) are batches of matrices, each with shape (batch_size, sequence_length, num_features). Batch matrix multiplication is only performed over the last two dimensions.

From the diagram above, we see that multi-head attention is composed of several identical attention heads. Each attention head contains 3 linear layers, followed by scaled dot-product attention. Let’s encapsulate this in an AttentionHead layer:
<!-- #endregion -->

```python id="5t097JfqiQrc"
import torch
from torch import nn


class AttentionHead(nn.Module):
    def __init__(self, dim_in: int, dim_k: int, dim_v: int):
        super().__init__()
        self.q = nn.Linear(dim_in, dim_k)
        self.k = nn.Linear(dim_in, dim_k)
        self.v = nn.Linear(dim_in, dim_v)

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        return scaled_dot_product_attention(self.q(query), self.k(key), self.v(value))
```

<!-- #region id="e4V_iqTSiQno" -->
Now, it’s very easy to build the multi-head attention layer. Just combine num_heads different attention heads and a Linear layer for the output.
<!-- #endregion -->

```python id="2L_SNeYFiQlz"
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, dim_in: int, dim_k: int, dim_v: int):
        super().__init__()
        self.heads = nn.ModuleList(
            [AttentionHead(dim_in, dim_k, dim_v) for _ in range(num_heads)]
        )
        self.linear = nn.Linear(num_heads * dim_v, dim_in)

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        return self.linear(
            torch.cat([h(query, key, value) for h in self.heads], dim=-1)
        )
```

<!-- #region id="KBaOoNjMiQjm" -->
We need one more component before building the complete transformer: positional encoding. Notice that MultiHeadAttention has no trainable components that operate over the sequence dimension (axis 1). Everything operates over the feature dimension (axis 2), and so it is independent of sequence length. We have to provide positional information to the model, so that it knows about the relative position of data points in the input sequences.
<!-- #endregion -->

```python id="vH2OGPkniQhH"
def position_encoding(
    seq_len: int, dim_model: int, device: torch.device = torch.device("cpu"),
) -> Tensor:
    pos = torch.arange(seq_len, dtype=torch.float, device=device).reshape(1, -1, 1)
    dim = torch.arange(dim_model, dtype=torch.float, device=device).reshape(1, 1, -1)
    phase = pos / 1e4 ** (dim // dim_model)

    return torch.where(dim.long() % 2 == 0, torch.sin(phase), torch.cos(phase))
```

<!-- #region id="JnmZrPpFid3w" -->
We can code the encoder/decoder modules independently of one another, and then combine them at the end. But first we need a few more pieces of information, which aren’t included in the figure above. For example, how should we choose to build the feed forward networks?
<!-- #endregion -->

```python id="RXG7YhX8ihQl"
def feed_forward(dim_input: int = 512, dim_feedforward: int = 2048) -> nn.Module:
    return nn.Sequential(
        nn.Linear(dim_input, dim_feedforward),
        nn.ReLU(),
        nn.Linear(dim_feedforward, dim_input),
    )
```

<!-- #region id="q-NWu_5uiuSn" -->
The output of each sub-layer is LayerNorm(x + Sublayer(x)), where Sublayer(x) is the function implemented by the sub-layer itself. … We apply dropout to the output of each sub-layer, before it is added to the sub-layer input and normalized.
<!-- #endregion -->

```python id="B-IOnfeyihNA"
class Residual(nn.Module):
    def __init__(self, sublayer: nn.Module, dimension: int, dropout: float = 0.1):
        super().__init__()
        self.sublayer = sublayer
        self.norm = nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, *tensors: Tensor) -> Tensor:
        # Assume that the "value" tensor is given last, so we can compute the
        # residual.  This matches the signature of 'MultiHeadAttention'.
        return self.norm(tensors[-1] + self.dropout(self.sublayer(*tensors)))
```

<!-- #region id="W5D8RZrqihK2" -->
Time to dive in and create the encoder. Using the utility methods we just built, this is pretty easy.
<!-- #endregion -->

```python id="tbCpBmFcihIk"
class TransformerEncoderLayer(nn.Module):
    def __init__(
        self, 
        dim_model: int = 512, 
        num_heads: int = 6, 
        dim_feedforward: int = 2048, 
        dropout: float = 0.1, 
    ):
        super().__init__()
        dim_k = dim_v = dim_model // num_heads
        self.attention = Residual(
            MultiHeadAttention(num_heads, dim_model, dim_k, dim_v),
            dimension=dim_model,
            dropout=dropout,
        )
        self.feed_forward = Residual(
            feed_forward(dim_model, dim_feedforward),
            dimension=dim_model,
            dropout=dropout,
        )

    def forward(self, src: Tensor) -> Tensor:
        src = self.attention(src, src, src)
        return self.feed_forward(src)


class TransformerEncoder(nn.Module):
    def __init__(
        self, 
        num_layers: int = 6,
        dim_model: int = 512, 
        num_heads: int = 8, 
        dim_feedforward: int = 2048, 
        dropout: float = 0.1, 
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(dim_model, num_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, src: Tensor) -> Tensor:
        seq_len, dimension = src.size(1), src.size(2)
        src += position_encoding(seq_len, dimension)
        for layer in self.layers:
            src = layer(src)

        return src
```

<!-- #region id="GZrsGydnihGL" -->
The decoder module is extremely similar. Just a few small differences:
- The decoder accepts two arguments (target and memory), rather than one.
- There are two multi-head attention modules per layer, instead of one.
- The second multi-head attention accepts memory for two of its inputs.
<!-- #endregion -->

```python id="dxezkFJzihEL"
class TransformerDecoderLayer(nn.Module):
    def __init__(
        self, 
        dim_model: int = 512, 
        num_heads: int = 6, 
        dim_feedforward: int = 2048, 
        dropout: float = 0.1, 
    ):
        super().__init__()
        dim_k = dim_v = dim_model // num_heads
        self.attention_1 = Residual(
            MultiHeadAttention(num_heads, dim_model, dim_k, dim_v),
            dimension=dim_model,
            dropout=dropout,
        )
        self.attention_2 = Residual(
            MultiHeadAttention(num_heads, dim_model, dim_k, dim_v),
            dimension=dim_model,
            dropout=dropout,
        )
        self.feed_forward = Residual(
            feed_forward(dim_model, dim_feedforward),
            dimension=dim_model,
            dropout=dropout,
        )

    def forward(self, tgt: Tensor, memory: Tensor) -> Tensor:
        tgt = self.attention_1(tgt, tgt, tgt)
        tgt = self.attention_2(memory, memory, tgt)
        return self.feed_forward(tgt)


class TransformerDecoder(nn.Module):
    def __init__(
        self, 
        num_layers: int = 6,
        dim_model: int = 512, 
        num_heads: int = 8, 
        dim_feedforward: int = 2048, 
        dropout: float = 0.1, 
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(dim_model, num_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.linear = nn.Linear(dim_model, dim_model)

    def forward(self, tgt: Tensor, memory: Tensor) -> Tensor:
        seq_len, dimension = tgt.size(1), tgt.size(2)
        tgt += position_encoding(seq_len, dimension)
        for layer in self.layers:
            tgt = layer(tgt, memory)

        return torch.softmax(self.linear(tgt), dim=-1)
```

<!-- #region id="qISz9Njni6Y_" -->
Lastly, we need to wrap everything up into a single Transformer class. This requires minimal work, because it’s nothing new — just throw an encoder and decoder together, and pass data through them in the correct order.
<!-- #endregion -->

```python id="mtdMH0rbi66X"
class Transformer(nn.Module):
    def __init__(
        self, 
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_model: int = 512, 
        num_heads: int = 6, 
        dim_feedforward: int = 2048, 
        dropout: float = 0.1, 
        activation: nn.Module = nn.ReLU(),
    ):
        super().__init__()
        self.encoder = TransformerEncoder(
            num_layers=num_encoder_layers,
            dim_model=dim_model,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.decoder = TransformerDecoder(
            num_layers=num_decoder_layers,
            dim_model=dim_model,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

    def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
        return self.decoder(tgt, self.encoder(src))
```

<!-- #region id="F_VtLkoci8zB" -->
And we’re done! Let’s create a simple test, as a sanity check for our implementation. We can construct random tensors for src and tgt, check that our model executes without errors, and confirm that the output tensor has the correct shape.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="np60EZkci-JP" executionInfo={"status": "ok", "timestamp": 1634026193801, "user_tz": -330, "elapsed": 2353, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="e53e7f16-7e37-4d8f-cdcd-96654401456c"
src = torch.rand(64, 16, 512)
tgt = torch.rand(64, 16, 512)
out = Transformer()(src, tgt)
print(out.shape)
# torch.Size([64, 16, 512])
```
