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

<!-- #region id="8VP4UHlM8CG5" -->
# BST in Deepctr
<!-- #endregion -->

<!-- #region id="yeTnOiBIJU48" -->
## Installation
<!-- #endregion -->

```python id="mcwCEbdn5Mej"
!pip install -q deepctr
```

<!-- #region id="g9Bagen2JWiB" -->
## API run
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="jkwBtowr6c6I" executionInfo={"status": "ok", "timestamp": 1633948670717, "user_tz": -330, "elapsed": 9017, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="c20a2eaa-2200-434b-979c-5593d8747d4c"
import numpy as np

from deepctr.models import BST
from deepctr.feature_column import SparseFeat, VarLenSparseFeat, DenseFeat, get_feature_names


def get_xy_fd():
    feature_columns = [SparseFeat('user', 3, embedding_dim=10), SparseFeat(
        'gender', 2, embedding_dim=4), SparseFeat('item_id', 3 + 1, embedding_dim=8),
                       SparseFeat('cate_id', 2 + 1, embedding_dim=4), DenseFeat('pay_score', 1)]
    feature_columns += [
        VarLenSparseFeat(SparseFeat('hist_item_id', vocabulary_size=3 + 1, embedding_dim=8, embedding_name='item_id'),
                         maxlen=4, length_name="seq_length"),
        VarLenSparseFeat(SparseFeat('hist_cate_id', 2 + 1, embedding_dim=4, embedding_name='cate_id'), maxlen=4,
                         length_name="seq_length")]
    # Notice: History behavior sequence feature name must start with "hist_".
    behavior_feature_list = ["item_id", "cate_id"]
    uid = np.array([0, 1, 2])
    ugender = np.array([0, 1, 0])
    iid = np.array([1, 2, 3])  # 0 is mask value
    cate_id = np.array([1, 2, 2])  # 0 is mask value
    pay_score = np.array([0.1, 0.2, 0.3])

    hist_iid = np.array([[1, 2, 3, 0], [3, 2, 1, 0], [1, 2, 0, 0]])
    hist_cate_id = np.array([[1, 2, 2, 0], [2, 2, 1, 0], [1, 2, 0, 0]])
    seq_length = np.array([3, 3, 2])  # the actual length of the behavior sequence

    feature_dict = {'user': uid, 'gender': ugender, 'item_id': iid, 'cate_id': cate_id,
                    'hist_item_id': hist_iid, 'hist_cate_id': hist_cate_id,
                    'pay_score': pay_score, 'seq_length': seq_length}
    x = {name: feature_dict[name] for name in get_feature_names(feature_columns)}
    y = np.array([1, 0, 1])
    return x, y, feature_columns, behavior_feature_list


if __name__ == "__main__":
    x, y, feature_columns, behavior_feature_list = get_xy_fd()
    model = BST(feature_columns, behavior_feature_list,att_head_num=4)
    model.compile('adam', 'binary_crossentropy',
                  metrics=['binary_crossentropy'])
    history = model.fit(x, y, verbose=1, epochs=10, validation_split=0.5)
```

<!-- #region id="LxbTlPCwJNn0" -->
## Model definition
<!-- #endregion -->

```python id="_qDCeT8h6c1S"
def BST(dnn_feature_columns, history_feature_list, transformer_num=1, att_head_num=8,
        use_bn=False, dnn_hidden_units=(256, 128, 64), dnn_activation='relu', l2_reg_dnn=0,
        l2_reg_embedding=1e-6, dnn_dropout=0.0, seed=1024, task='binary'):
    """Instantiates the BST architecture.
     :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
     :param history_feature_list: list, to indicate sequence sparse field.
     :param transformer_num: int, the number of transformer layer.
     :param att_head_num: int, the number of heads in multi-head self attention.
     :param use_bn: bool. Whether use BatchNormalization before activation or not in deep net
     :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN
     :param dnn_activation: Activation function to use in DNN
     :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
     :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
     :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
     :param seed: integer ,to use as random seed.
     :param task: str, ``"binary"`` for  binary logloss or ``"regression"`` for regression loss
     :return: A Keras model instance.
     """

    features = build_input_features(dnn_feature_columns)
    inputs_list = list(features.values())

    user_behavior_length = features["seq_length"]

    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, SparseFeat), dnn_feature_columns)) if dnn_feature_columns else []
    dense_feature_columns = list(
        filter(lambda x: isinstance(x, DenseFeat), dnn_feature_columns)) if dnn_feature_columns else []
    varlen_sparse_feature_columns = list(
        filter(lambda x: isinstance(x, VarLenSparseFeat), dnn_feature_columns)) if dnn_feature_columns else []

    history_feature_columns = []
    sparse_varlen_feature_columns = []
    history_fc_names = list(map(lambda x: "hist_" + x, history_feature_list))

    for fc in varlen_sparse_feature_columns:
        feature_name = fc.name
        if feature_name in history_fc_names:
            history_feature_columns.append(fc)
        else:
            sparse_varlen_feature_columns.append(fc)

    embedding_dict = create_embedding_matrix(dnn_feature_columns, l2_reg_embedding, seed, prefix="",
                                             seq_mask_zero=True)

    query_emb_list = embedding_lookup(embedding_dict, features, sparse_feature_columns,
                                      return_feat_list=history_feature_list, to_list=True)
    hist_emb_list = embedding_lookup(embedding_dict, features, history_feature_columns,
                                     return_feat_list=history_fc_names, to_list=True)
    dnn_input_emb_list = embedding_lookup(embedding_dict, features, sparse_feature_columns,
                                          mask_feat_list=history_feature_list, to_list=True)
    dense_value_list = get_dense_input(features, dense_feature_columns)
    sequence_embed_dict = varlen_embedding_lookup(embedding_dict, features, sparse_varlen_feature_columns)
    sequence_embed_list = get_varlen_pooling_list(sequence_embed_dict, features, sparse_varlen_feature_columns,
                                                  to_list=True)

    dnn_input_emb_list += sequence_embed_list
    query_emb = concat_func(query_emb_list)
    deep_input_emb = concat_func(dnn_input_emb_list)
    hist_emb = concat_func(hist_emb_list)

    transformer_output = hist_emb
    for _ in range(transformer_num):
        att_embedding_size = transformer_output.get_shape().as_list()[-1] // att_head_num
        transformer_layer = Transformer(att_embedding_size=att_embedding_size, head_num=att_head_num,
                                        dropout_rate=dnn_dropout, use_positional_encoding=True, use_res=True,
                                        use_feed_forward=True, use_layer_norm=True, blinding=False, seed=seed,
                                        supports_masking=False, output_type=None)
        transformer_output = transformer_layer([transformer_output, transformer_output,
                                                user_behavior_length, user_behavior_length])

    attn_output = AttentionSequencePoolingLayer(att_hidden_units=(64, 16), weight_normalization=True,
                                                supports_masking=False)([query_emb, transformer_output,
                                                                         user_behavior_length])
    deep_input_emb = concat_func([deep_input_emb, attn_output], axis=-1)
    deep_input_emb = Flatten()(deep_input_emb)

    dnn_input = combined_dnn_input([deep_input_emb], dense_value_list)
    output = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, use_bn, seed=seed)(dnn_input)
    final_logit = Dense(1, use_bias=False, kernel_initializer=tf.keras.initializers.glorot_normal(seed))(output)
    output = PredictionLayer(task)(final_logit)

    model = tf.keras.models.Model(inputs=inputs_list, outputs=output)

    return model
```
