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

<!-- #region id="NBj3rQU4ydEV" -->
# BST in MXNet

> Implementing BST transformer model in MXNet library framework. After implementation, running on a sample dataset.
<!-- #endregion -->

<!-- #region id="PwZFDzR2-_w6" -->
## Setup
<!-- #endregion -->

<!-- #region id="0eZJphSm_BC5" -->
### Installations
<!-- #endregion -->

```python id="Ir5ol9MsUPuc"
!pip install -q mxnet
!pip install -q gluonnlp
```

<!-- #region id="mxpuBk6h_ClV" -->
### Imports
<!-- #endregion -->

```python id="mj9PWmgJT_ce"
import numpy as np
import random

import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
from mxnet import autograd as ag
from mxnet.gluon.nn import HybridBlock, HybridSequential, LeakyReLU
from mxnet.gluon.block import HybridBlock
from mxnet.ndarray import L2Normalization
from gluonnlp.model import AttentionCell
```

<!-- #region id="WXtMWjYL_bfv" -->
### Params
<!-- #endregion -->

```python id="TcLEXP0E_c--"
np.random.seed(100)
ctx = mx.cpu()
mx.random.seed(100)
random.seed(100)
```

```python id="MiY7zOFk_ej-"
_BATCH = 1
_SEQ_LEN = 32
_OTHER_LEN = 32
_EMB_DIM = 32
_NUM_HEADS = 8
_DROP = 0.2
_UNITS = 32
```

<!-- #region id="sZG6z7KB_Ubu" -->
## Model
<!-- #endregion -->

```python id="01vZAyZoT_yI"
def _masked_softmax(F, att_score, mask, dtype):
    """Ignore the masked elements when calculating the softmax
    Parameters
    ----------
    F : symbol or ndarray
    att_score : Symborl or NDArray
        Shape (batch_size, query_length, memory_length)
    mask : Symbol or NDArray or None
        Shape (batch_size, query_length, memory_length)
    Returns
    -------
    att_weights : Symborl or NDArray
        Shape (batch_size, query_length, memory_length)
    """
    if mask is not None:
        # Fill in the masked scores with a very small value
        neg = -1e18
        if np.dtype(dtype) == np.float16:
            neg = -1e4
        else:
            try:
                # if AMP (automatic mixed precision) is enabled, -1e18 will cause NaN.
                from mxnet.contrib import amp
                if amp.amp._amp_initialized:
                    neg = -1e4
            except ImportError:
                pass
        att_score = F.where(mask, att_score, neg * F.ones_like(att_score))
        att_weights = F.softmax(att_score, axis=-1) * mask
    else:
        att_weights = F.softmax(att_score, axis=-1)
    return att_weights

def _get_attention_cell(attention_cell, units=None,
                        scaled=True, num_heads=None,
                        use_bias=False, dropout=0.0, activation='relu'):
    """
    Parameters
    ----------
    attention_cell : AttentionCell or str
    units : int or None
    Returns
    -------
    attention_cell : AttentionCell
    """
    if isinstance(attention_cell, str):
        if attention_cell == 'scaled_luong':
            return DotProductAttentionCell(units=units, scaled=True, normalized=False,
                                           use_bias=use_bias, dropout=dropout, luong_style=True)
        elif attention_cell == 'scaled_dot':
            return DotProductAttentionCell(units=units, scaled=True, normalized=False,
                                           use_bias=use_bias, dropout=dropout, luong_style=False)
        elif attention_cell == 'dot':
            return DotProductAttentionCell(units=units, scaled=False, normalized=False,
                                           use_bias=use_bias, dropout=dropout, luong_style=False)
        elif attention_cell == 'cosine':
            return DotProductAttentionCell(units=units, scaled=False, use_bias=use_bias,
                                           dropout=dropout, normalized=True)
        # elif attention_cell == 'mlp':
        #     return MLPAttentionCell(units=units, normalized=False)
        # elif attention_cell == 'normed_mlp':
        #     return MLPAttentionCell(units=units, normalized=True)
        elif attention_cell == 'multi_head':
            base_cell = DotProductAttentionCell(scaled=scaled, dropout=dropout, activation=activation)
            return MultiHeadAttentionCell(base_cell=base_cell, query_units=units, use_bias=use_bias,
                                          key_units=units, value_units=units, num_heads=num_heads
                                          )
        else:
            raise NotImplementedError
    else:
        assert isinstance(attention_cell, AttentionCell),\
            'attention_cell must be either string or AttentionCell. Received attention_cell={}'\
                .format(attention_cell)
        return attention_cell

class DotProductAttentionCell(AttentionCell):
    r"""Dot product attention between the query and the key.
    Depending on parameters, defined as::
        units is None:
            score = <h_q, h_k>
        units is not None and luong_style is False:
            score = <W_q h_q, W_k h_k>
        units is not None and luong_style is True:
            score = <W h_q, h_k>
    Parameters
    ----------
    units: int or None, default None
        Project the query and key to vectors with `units` dimension
        before applying the attention. If set to None,
        the query vector and the key vector are directly used to compute the attention and
        should have the same dimension::
            If the units is None,
                score = <h_q, h_k>
            Else if the units is not None and luong_style is False:
                score = <W_q h_q, W_k h_k>
            Else if the units is not None and luong_style is True:
                score = <W h_q, h_k>
    luong_style: bool, default False
        If turned on, the score will be::
            score = <W h_q, h_k>
        `units` must be the same as the dimension of the key vector
    scaled: bool, default True
        Whether to divide the attention weights by the sqrt of the query dimension.
        This is first proposed in "[NIPS2017] Attention is all you need."::
            score = <h_q, h_k> / sqrt(dim_q)
    normalized: bool, default False
        If turned on, the cosine distance is used, i.e::
            score = <h_q / ||h_q||, h_k / ||h_k||>
    use_bias : bool, default True
        Whether to use bias in the projection layers.
    dropout : float, default 0.0
        Attention dropout
    weight_initializer : str or `Initializer` or None, default None
        Initializer of the weights
    bias_initializer : str or `Initializer`, default 'zeros'
        Initializer of the bias
    prefix : str or None, default None
        See document of `Block`.
    params : str or None, default None
        See document of `Block`.
    """
    def __init__(self, units=None, luong_style=False, scaled=True, normalized=False, use_bias=True,
                 activation=None,
                 dropout=0.0, weight_initializer=None, bias_initializer='zeros',
                 prefix=None, params=None):
        super(DotProductAttentionCell, self).__init__(prefix=prefix, params=params)
        self._units = units
        self._scaled = scaled
        self._normalized = normalized
        self._use_bias = use_bias
        self._luong_style = luong_style
        self._dropout = dropout
        self._activation = activation

        if self._luong_style:
            assert units is not None, 'Luong style attention is not available without explicitly ' \
                                      'setting the units'
        with self.name_scope():
            self._dropout_layer = nn.Dropout(dropout)

        if self._activation is not None:
            with self.name_scope():
                self.act = gluon.nn.LeakyReLU(alpha=0.1)

        if units is not None:
            with self.name_scope():
                self._proj_query = nn.Dense(units=self._units, use_bias=self._use_bias,
                                            flatten=False, weight_initializer=weight_initializer,
                                            bias_initializer=bias_initializer,
                                            prefix='query_')

                if not self._luong_style:
                    self._proj_key = nn.Dense(units=self._units, use_bias=self._use_bias,
                                              flatten=False, weight_initializer=weight_initializer,
                                              bias_initializer=bias_initializer, prefix='key_')
        if self._normalized:
            with self.name_scope():
                self._l2_norm = L2Normalization(axis=-1)

    def _compute_weight(self, F, query, key, mask=None):
        if self._units is not None:
            query = self._proj_query(query)

            # leakyrelu activation per alibaba rec article is used in self-attention and ffn
            if self._activation is not None:
                query = self.act(query)

            if not self._luong_style:
                key = self._proj_key(key)

                # leakyrelu activation per alibaba rec article is used in self-attention and ffn
                if self._activation is not None:
                    key = self.act(key)

            elif F == mx.nd:
                assert query.shape[-1] == key.shape[-1], 'Luong style attention requires key to ' \
                                                         'have the same dim as the projected ' \
                                                         'query. Received key {}, query {}.'.format(
                                                             key.shape, query.shape)
        if self._normalized:
            query = self._l2_norm(query)
            key = self._l2_norm(key)
        if self._scaled:
            query = F.contrib.div_sqrt_dim(query)

        att_score = F.batch_dot(query, key, transpose_b=True)

        att_weights = self._dropout_layer(_masked_softmax(F, att_score, mask, self._dtype))
        return att_weights


class MultiHeadAttentionCell(AttentionCell):
    r"""Multi-head Attention Cell.
    In the MultiHeadAttentionCell, the input query/key/value will be linearly projected
    for `num_heads` times with different projection matrices. Each projected key, value, query
    will be used to calculate the attention weights and values. The output of each head will be
    concatenated to form the final output.
    The idea is first proposed in "[Arxiv2014] Neural Turing Machines" and
    is later adopted in "[NIPS2017] Attention is All You Need" to solve the
    Neural Machine Translation problem.
    Parameters
    ----------
    base_cell : AttentionCell
    query_units : int
        Total number of projected units for query. Must be divided exactly by num_heads.
    key_units : int
        Total number of projected units for key. Must be divided exactly by num_heads.
    value_units : int
        Total number of projected units for value. Must be divided exactly by num_heads.
    num_heads : int
        Number of parallel attention heads
    use_bias : bool, default True
        Whether to use bias when projecting the query/key/values
    weight_initializer : str or `Initializer` or None, default None
        Initializer of the weights.
    bias_initializer : str or `Initializer`, default 'zeros'
        Initializer of the bias.
    prefix : str or None, default None
        See document of `Block`.
    params : str or None, default None
        See document of `Block`.
    """
    def __init__(self, base_cell, query_units, key_units, value_units, num_heads, use_bias=True,
                 weight_initializer=None, bias_initializer='zeros', prefix=None, params=None):
        super(MultiHeadAttentionCell, self).__init__(prefix=prefix, params=params)
        self._base_cell = base_cell
        self._num_heads = num_heads
        self._use_bias = use_bias
        units = {'query': query_units, 'key': key_units, 'value': value_units}
        for name, unit in units.items():
            if unit % self._num_heads != 0:
                raise ValueError(
                    'In MultiHeadAttetion, the {name}_units should be divided exactly'
                    ' by the number of heads. Received {name}_units={unit}, num_heads={n}'.format(
                        name=name, unit=unit, n=num_heads))
            setattr(self, '_{}_units'.format(name), unit)
            with self.name_scope():
                setattr(
                    self, 'proj_{}'.format(name),
                    nn.Dense(units=unit, use_bias=self._use_bias, flatten=False,
                             weight_initializer=weight_initializer,
                             bias_initializer=bias_initializer, prefix='{}_'.format(name)))

    def __call__(self, query, key, value=None, mask=None):
        """Compute the attention.
        Parameters
        ----------
        query : Symbol or NDArray
            Query vector. Shape (batch_size, query_length, query_dim)
        key : Symbol or NDArray
            Key of the memory. Shape (batch_size, memory_length, key_dim)
        value : Symbol or NDArray or None, default None
            Value of the memory. If set to None, the value will be set as the key.
            Shape (batch_size, memory_length, value_dim)
        mask : Symbol or NDArray or None, default None
            Mask of the memory slots. Shape (batch_size, query_length, memory_length)
            Only contains 0 or 1 where 0 means that the memory slot will not be used.
            If set to None. No mask will be used.
        Returns
        -------
        context_vec : Symbol or NDArray
            Shape (batch_size, query_length, context_vec_dim)
        att_weights : Symbol or NDArray
            Attention weights of multiple heads.
            Shape (batch_size, num_heads, query_length, memory_length)
        """
        return super(MultiHeadAttentionCell, self).__call__(query, key, value, mask)

    def _project(self, F, name, x):
        # Shape (batch_size, query_length, query_units)
        x = getattr(self, 'proj_{}'.format(name))(x)
        # Shape (batch_size * num_heads, query_length, ele_units)
        x = F.transpose(x.reshape(shape=(0, 0, self._num_heads, -1)),
                        axes=(0, 2, 1, 3))\
             .reshape(shape=(-1, 0, 0), reverse=True)
        return x

    def _compute_weight(self, F, query, key, mask=None):
        query = self._project(F, 'query', query)
        key = self._project(F, 'key', key)
        if mask is not None:
            mask = F.broadcast_axis(F.expand_dims(mask, axis=1),
                                    axis=1, size=self._num_heads)\
                    .reshape(shape=(-1, 0, 0), reverse=True)
        att_weights = self._base_cell._compute_weight(F, query, key, mask)
        return att_weights.reshape(shape=(-1, self._num_heads, 0, 0), reverse=True)

    def _read_by_weight(self, F, att_weights, value):
        att_weights = att_weights.reshape(shape=(-1, 0, 0), reverse=True)
        value = self._project(F, 'value', value)
        context_vec = self._base_cell._read_by_weight(F, att_weights, value)
        context_vec = F.transpose(context_vec.reshape(shape=(-1, self._num_heads, 0, 0),
                                                      reverse=True),
                                  axes=(0, 2, 1, 3)).reshape(shape=(0, 0, -1))
        return context_vec


def _get_layer_norm(use_bert, units, layer_norm_eps=None):
    # from gluonnlp.model.bert import BERTLayerNorm
    layer_norm = nn.LayerNorm
    if layer_norm_eps:
        return layer_norm(in_channels=units, epsilon=layer_norm_eps)
    else:
        return layer_norm(in_channels=units)


class BasePositionwiseFFN(HybridBlock):
    """Base Structure of the Positionwise Feed-Forward Neural Network.
    Parameters
    ----------
    units : int
        Number of units for the output
    hidden_size : int
        Number of units in the hidden layer of position-wise feed-forward networks
    dropout : float
    use_residual : bool
    weight_initializer : str or Initializer
        Initializer for the input weights matrix, used for the linear
        transformation of the inputs.
    bias_initializer : str or Initializer
        Initializer for the bias vector.
    activation : str, default 'relu'
        Activation function
    use_bert_layer_norm : bool, default False.
        Whether to use the BERT-stype layer norm implemented in Tensorflow, where
        epsilon is added inside the square root. Set to True for pre-trained BERT model.
    ffn1_dropout : bool, default False
        If True, apply dropout both after the first and second Positionwise
        Feed-Forward Neural Network layers. If False, only apply dropout after
        the second.
    prefix : str, default None
        Prefix for name of `Block`s
        (and name of weight if params is `None`).
    params : Parameter or None
        Container for weight sharing between cells.
        Created if `None`.
    layer_norm_eps : float, default None
        Epsilon for layer_norm
    Inputs:
        - **inputs** : input sequence of shape (batch_size, length, C_in).
    Outputs:
        - **outputs** : output encoding of shape (batch_size, length, C_out).
    """

    def __init__(self, units=512, hidden_size=2048, dropout=0.0, use_residual=True,
                 weight_initializer=None, bias_initializer='zeros', activation='leakyrelu',
                 use_bert_layer_norm=False, ffn1_dropout=False, prefix=None, params=None,
                 layer_norm_eps=None):
        super(BasePositionwiseFFN, self).__init__(prefix=prefix, params=params)
        self._hidden_size = hidden_size
        self._units = units
        self._use_residual = use_residual
        self._dropout = dropout
        self._ffn1_dropout = ffn1_dropout
        with self.name_scope():
            self.ffn_1 = nn.Dense(units=hidden_size, flatten=False,
                                  weight_initializer=weight_initializer,
                                  bias_initializer=bias_initializer,
                                  prefix='ffn_1_')
            self.activation = self._get_activation(activation) if activation else None
            self.ffn_2 = nn.Dense(units=units, flatten=False,
                                  weight_initializer=weight_initializer,
                                  bias_initializer=bias_initializer,
                                  prefix='ffn_2_')
            if dropout:
                self.dropout_layer = nn.Dropout(rate=dropout)
            self.layer_norm = _get_layer_norm(use_bert_layer_norm, units,
                                              layer_norm_eps=layer_norm_eps)

    def _get_activation(self, act):
        """Get activation block based on the name. """
        if isinstance(act, str):

            # per alibaba rec article leakyRELU is used in self-attention and ffn
            if act.lower() == 'leakyrelu':
                return gluon.nn.LeakyReLU(alpha=0.1)
            else:
                return gluon.nn.Activation(act)
        assert isinstance(act, gluon.Block)
        return act

    def hybrid_forward(self, F, inputs):  # pylint: disable=arguments-differ
        # pylint: disable=unused-argument
        """Position-wise encoding of the inputs.
        Parameters
        ----------
        inputs : Symbol or NDArray
            Input sequence. Shape (batch_size, length, C_in)
        Returns
        -------
        outputs : Symbol or NDArray
            Shape (batch_size, length, C_out)
        """
        outputs = self.ffn_1(inputs)
        if self.activation:
            outputs = self.activation(outputs)
        if self._dropout and self._ffn1_dropout:
            outputs = self.dropout_layer(outputs)
        outputs = self.ffn_2(outputs)
        if self.activation:
            outputs = self.activation(outputs)
        if self._dropout:
            outputs = self.dropout_layer(outputs)
        if self._use_residual:
            outputs = outputs + inputs
        outputs = self.layer_norm(outputs)
        return outputs


class PositionwiseFFN(BasePositionwiseFFN):
    """Structure of the Positionwise Feed-Forward Neural Network for
    Transformer.
    Computes the positionwise encoding of the inputs.
    Parameters
    ----------
    units : int
        Number of units for the output
    hidden_size : int
        Number of units in the hidden layer of position-wise feed-forward networks
    dropout : float
        Dropout probability for the output
    use_residual : bool
        Add residual connection between the input and the output
    ffn1_dropout : bool, default False
        If True, apply dropout both after the first and second Positionwise
        Feed-Forward Neural Network layers. If False, only apply dropout after
        the second.
    weight_initializer : str or Initializer
        Initializer for the input weights matrix, used for the linear
        transformation of the inputs.
    bias_initializer : str or Initializer
        Initializer for the bias vector.
    prefix : str, default None
        Prefix for name of `Block`s (and name of weight if params is `None`).
    params : Parameter or None
        Container for weight sharing between cells. Created if `None`.
    activation : str, default 'relu'
        Activation methods in PositionwiseFFN
    layer_norm_eps : float, default None
        Epsilon for layer_norm
    Inputs:
        - **inputs** : input sequence of shape (batch_size, length, C_in).
    Outputs:
        - **outputs** : output encoding of shape (batch_size, length, C_out).
    """

    def __init__(self, units=512, hidden_size=2048, dropout=0.0, use_residual=True,
                 ffn1_dropout=False, weight_initializer=None, bias_initializer='zeros', prefix=None,
                 params=None, activation='relu', layer_norm_eps=None):
        super(PositionwiseFFN, self).__init__(
            units=units,
            hidden_size=hidden_size,
            dropout=dropout,
            use_residual=use_residual,
            weight_initializer=weight_initializer,
            bias_initializer=bias_initializer,
            prefix=prefix,
            params=params,
            # extra configurations for transformer
            activation=activation,
            use_bert_layer_norm=False,
            layer_norm_eps=layer_norm_eps,
            ffn1_dropout=ffn1_dropout)
```

```python id="D5-OEpEtUuP7"
def _position_encoding_init(max_length, dim):
    """Init the sinusoid position encoding table """
    position_enc = np.arange(max_length).reshape((-1, 1)) \
                   / (np.power(10000, (2. / dim) * np.arange(dim).reshape((1, -1))))
    position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])
    position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])
    return position_enc

def _position_encoding_init_BST(max_length, dim):
    """For the BST recommender, the positional embedding takes the time of item being clicked as
    input feature and calculates the position value of item vi as p(vt) - p(vi) where
    p(vt) is recommending time and p(vi) is time the user clicked on item vi
    """
    # Assume position_enc is the p(vt) - p(vi) fed as input
    position_enc = np.arange(max_length).reshape((-1, 1)) \
                   / (np.power(10000, (2. / dim) * np.arange(dim).reshape((1, -1))))
    return position_enc

class Rec(HybridBlock):
    """Alibaba transformer based recommender"""
    def __init__(self, **kwargs):
        super(Rec, self).__init__(**kwargs)
        with self.name_scope():
            self.otherfeatures = nn.Embedding(input_dim=_OTHER_LEN,
                                               output_dim=_EMB_DIM)
            self.features = nn.Embedding(input_dim=_SEQ_LEN,
                                           output_dim=_EMB_DIM)
            # Transformer layers
            # Multi-head attention with base cell scaled dot-product attention
            # Use b=1 self-attention blocks per article recommendation
            self.cell = _get_attention_cell('multi_head',
                                            units=_UNITS,
                                            scaled=True,
                                            dropout=_DROP,
                                            num_heads=_NUM_HEADS,
                                            use_bias=False)
            self.proj = nn.Dense(units=_UNITS,
                                 use_bias=False,
                                 bias_initializer='zeros',
                                 weight_initializer=None,
                                 flatten=False
                                 )
            self.drop_out_layer = nn.Dropout(rate=_DROP)
            self.ffn = PositionwiseFFN(hidden_size=_UNITS,
                                       use_residual=True,
                                       dropout=_DROP,
                                       units=_UNITS,
                                       weight_initializer=None,
                                       bias_initializer='zeros',
                                       activation='leakyrelu'
                                       )
            self.layer_norm = nn.LayerNorm(in_channels=_UNITS)
            # Final MLP layers; BST dimensions in the article were 1024, 512, 256
            self.output = HybridSequential()
            self.output.add(nn.Dense(8))
            self.output.add(LeakyReLU(alpha=0.1))
            self.output.add(nn.Dense(4))
            self.output.add(LeakyReLU(alpha=0.1))
            self.output.add(nn.Dense(2))
            self.output.add(LeakyReLU(alpha=0.1))
            self.output.add(nn.Dense(1))

    def _arange_like(self, F, inputs, axis):
        """Helper function to generate indices of a range"""
        if F == mx.ndarray:
            seq_len = inputs.shape[axis]
            arange = F.arange(seq_len, dtype=inputs.dtype, ctx=inputs.context)
        else:
            input_axis = inputs.slice(begin=(0, 0, 0), end=(1, None, 1)).reshape((-1))
            zeros = F.zeros_like(input_axis)
            arange = F.arange(start=0, repeat=1, step=1,
                              infer_range=True, dtype=inputs.dtype)
            arange = F.elemwise_add(arange, zeros)
            # print(arange)
        return arange

    def _get_positional(self, weight_type, max_length, units):
        if weight_type == 'sinusoidal':
            encoding = _position_encoding_init(max_length, units)
        elif weight_type == 'BST':
            # BST position fed as input
            encoding = _position_encoding_init_BST(max_length, units)
        else:
            raise ValueError('Not known')
        return mx.nd.array(encoding)

    def hybrid_forward(self, F, x, x_other, mask=None):
        # The manually engineered features
        x1 = self.otherfeatures(x_other)

        # The transformer encoder
        steps = self._arange_like(F, x, axis=1)
        x = self.features(x)
        position_weight = self._get_positional('BST', _SEQ_LEN, _UNITS)
        # add positional embedding
        positional_embedding = F.Embedding(steps, position_weight, _SEQ_LEN, _UNITS)
        x = F.broadcast_add(x, F.expand_dims(positional_embedding, axis=0))
        # attention cell with dropout
        out_x, attn_w = self.cell(x, x, x, mask)
        out_x = self.proj(out_x)
        out_x = self.drop_out_layer(out_x)
        # add and norm
        out_x = x + out_x
        out_x = self.layer_norm(out_x)
        # ffn
        out_x = self.ffn(out_x)

        # concat engineered features with transformer representations
        out_x = mx.ndarray.concat(out_x, x1)
        # leakyrelu final layers
        out_x = self.output(out_x)
        return out_x
```

<!-- #region id="fRjnFotyUvda" -->
## Train
<!-- #endregion -->

```python id="r64OhBu1U47O"
def generate_sample():
    """Generate toy X and y. One target item.
    """
    X = mx.random.uniform(shape=(100, 64), ctx=ctx)
    y = mx.random.uniform(shape=(100, 1), ctx=ctx)
    y = y > 0.5
    # Data loader
    d = gluon.data.ArrayDataset(X, y, )
    return gluon.data.DataLoader(d, _BATCH, last_batch='keep')


train_metric = mx.metric.Accuracy()


def train():
    train_data = generate_sample()
    optimizer = mx.optimizer.Adam()
    # Binary classification problem; predict if user clicks target item
    loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()
    net = Rec()
    net.initialize()
    trainer = gluon.Trainer(net.collect_params(),
                            optimizer)
    # train_metric = mx.metric.Accuracy()
    epochs = 1

    for epoch in range(epochs):
        train_metric.reset()
        for x, y in train_data:
            with ag.record():
                # assume x contains sequential inputs and manually engineered features
                output = net(x[:, :32], x[:, 32:])
                l = loss(output, y).sum()
        l.backward()
        trainer.step(_BATCH)
        train_metric.update(y, output)
```

```python id="zS1Zwl1dU7un"
train()
```

```python colab={"base_uri": "https://localhost:8080/"} id="9vsk8mq_YXre" executionInfo={"status": "ok", "timestamp": 1633117398581, "user_tz": -330, "elapsed": 790, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="ab493efd-9d59-4561-cb65-10490c01ca8c"
train_metric.get()
```

<!-- #region id="XBKrp1i9VCd-" -->
## Test
<!-- #endregion -->

```python id="XWZ_aIuwVFKG"
_SEQ_LEN = 32
_BATCH = 1
ctx = mx.cpu()

def _tst_module(net, x):
    net.initialize()
    net.collect_params()
    net(x,x)
    mx.nd.waitall()

def test():
    x = mx.random.uniform(shape=(_BATCH, _SEQ_LEN), ctx=ctx)
    net = Rec()
    _tst_module(net, x)
```

```python id="oCqjWGuGVIEh"
test()
```
