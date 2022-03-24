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

<!-- #region id="QjaoZMXuzYyl" -->
# Contextualized Knowledge Graph Embedding
<!-- #endregion -->

```python id="gEhj7V1vk0D5"
!apt-get -qq install tree
```

```python colab={"base_uri": "https://localhost:8080/"} id="PXiCrlpojkDs" executionInfo={"status": "ok", "timestamp": 1633959331424, "user_tz": -330, "elapsed": 436, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="bbb7ea29-aa2b-49ce-c94a-964ba2ab31be"
%%writefile wget_dataset.sh
#!/bin/bash
mkdir data
cd data
##downloads the 4 widely used KBC dataset
wget -q --show-progress --no-check-certificate https://everest.hds.utc.fr/lib/exe/fetch.php?media=en:fb15k.tgz -O fb15k.tgz
wget -q --show-progress --no-check-certificate https://everest.hds.utc.fr/lib/exe/fetch.php?media=en:wordnet-mlj12.tar.gz -O wordnet-mlj12.tar.gz
wget -q --show-progress --no-check-certificat https://download.microsoft.com/download/8/7/0/8700516A-AB3D-4850-B4BB-805C515AECE1/FB15K-237.2.zip -O FB15K-237.2.zip
wget -q --show-progress --no-check-certificat https://raw.githubusercontent.com/TimDettmers/ConvE/master/WN18RR.tar.gz -O WN18RR.tar.gz 

##downloads the path query dataset
wget -q --show-progress --no-check-certificate https://worksheets.codalab.org/rest/bundles/0xdb6b691c2907435b974850e8eb9a5fc2/contents/blob/ -O freebase_paths.tar.gz
wget -q --show-progress --no-check-certificate https://worksheets.codalab.org/rest/bundles/0xf91669f6c6d74987808aeb79bf716bd0/contents/blob/ -O wordnet_paths.tar.gz

## organize the train/valid/test files by renaming
#fb15k
tar -xvf fb15k.tgz 
mv FB15k fb15k
mv ./fb15k/freebase_mtr100_mte100-train.txt ./fb15k/train.txt
mv ./fb15k/freebase_mtr100_mte100-test.txt ./fb15k/test.txt
mv ./fb15k/freebase_mtr100_mte100-valid.txt ./fb15k/valid.txt

#wn18
tar -zxvf wordnet-mlj12.tar.gz && mv wordnet-mlj12 wn18
mv wn18/wordnet-mlj12-train.txt wn18/train.txt
mv wn18/wordnet-mlj12-test.txt wn18/test.txt
mv wn18/wordnet-mlj12-valid.txt wn18/valid.txt


#fb15k237
unzip FB15K-237.2.zip && mv Release fb15k237

#wn18rr
mkdir wn18rr && tar -zxvf WN18RR.tar.gz -C wn18rr

#pathqueryWN
mkdir pathqueryWN && tar -zxvf wordnet_paths.tar.gz -C pathqueryWN

#pathqueryFB
mkdir pathqueryFB && tar -zxvf freebase_paths.tar.gz -C pathqueryFB

##rm tmp zip files
rm ./*.gz
rm ./*.tgz
rm ./*.zip
```

```python colab={"base_uri": "https://localhost:8080/"} id="Ui_FPRbyjtH5" executionInfo={"status": "ok", "timestamp": 1633959384850, "user_tz": -330, "elapsed": 52994, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="8ba6da11-7f91-4e76-a5e3-703f71077db6"
!sh wget_dataset.sh
```

```python colab={"base_uri": "https://localhost:8080/"} id="tgXup_B7kD-f" executionInfo={"status": "ok", "timestamp": 1633959615986, "user_tz": -330, "elapsed": 459, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="56b2b559-6420-42a5-96f3-e58a20fff1a4"
!tree /content/data
```

```python id="iBTzRb35k7o8"
# Attention! Python 2.7.14  and python3 gives different vocabulary order. We use Python 2.7.14 to preprocess files.

# input files: train.txt valid.txt test.txt  
# (these are default filenames, change files name with the following arguments:  --train $trainname --valid $validname --test $testname)
# output files: vocab.txt train.coke.txt valid.coke.txt test.coke.txt
python ./bin/kbc_data_preprocess.py --task fb15k --dir ./data/fb15k
python ./bin/kbc_data_preprocess.py --task wn18 --dir ./data/wn18
python ./bin/kbc_data_preprocess.py --task fb15k237 --dir ./data/fb15k237
python ./bin/kbc_data_preprocess.py --task wn18rr --dir ./data/wn18rr

# input files: train dev test
# (these are default filenames, change files name with the following arguments: --train $trainname --valid $validname --test $testname)
# output files: vocab.txt train.coke.txt valid.coke.txt test.coke.txt sen_candli.txt trivial_sen.txt
python ./bin/pathquery_data_preprocess.py --task pathqueryFB --dir ./data/pathqueryFB 
python ./bin/pathquery_data_preprocess.py --task pathqueryWN --dir ./data/pathqueryWN
```

```python id="ned-nTDelA5W"
"""
data preprocess for KBC datasets
"""


def get_unique_entities_relations(train_file, dev_file, test_file):
    entity_lst = dict()
    relation_lst = dict()
    all_files = [train_file, dev_file, test_file]
    for input_file in all_files:
        print("dealing %s" % train_file)
        with open(input_file, "r") as f:
            for line in f.readlines():
                tokens = line.strip().split("\t")
                assert len(tokens) == 3
                entity_lst[tokens[0]] = len(entity_lst)
                entity_lst[tokens[2]] = len(entity_lst)
                relation_lst[tokens[1]] = len(relation_lst)
    print(">> Number of unique entities: %s" % len(entity_lst))
    print(">> Number of unique relations: %s" % len(relation_lst))
    return entity_lst, relation_lst


def write_vocab(output_file, entity_lst, relation_lst):
    fout = open(output_file, "w")
    fout.write("[PAD]" + "\n")
    for i in range(95):
        fout.write("[unused{}]\n".format(i))
    fout.write("[UNK]" + "\n")
    fout.write("[CLS]" + "\n")
    fout.write("[SEP]" + "\n")
    fout.write("[MASK]" + "\n")
    for e in entity_lst.keys():
        fout.write(e + "\n")
    for r in relation_lst.keys():
        fout.write(r + "\n")
    vocab_size = 100 + len(entity_lst) + len(relation_lst)
    print(">> vocab_size: %s" % vocab_size)
    fout.close()


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    fin = open(vocab_file)
    for num, line in enumerate(fin):
        items = line.strip().split("\t")
        if len(items) > 2:
            break
        token = items[0]
        index = items[1] if len(items) == 2 else num
        token = token.strip()
        vocab[token] = int(index)
    return vocab


def write_true_triples(train_file, dev_file, test_file, vocab, output_file):
    true_triples = []
    all_files = [train_file, dev_file, test_file]
    for input_file in all_files:
        with open(input_file, "r") as f:
            for line in f.readlines():
                h, r, t = line.strip('\r \n').split('\t')
                assert (h in vocab) and (r in vocab) and (t in vocab)
                hpos = vocab[h]
                rpos = vocab[r]
                tpos = vocab[t]
                true_triples.append((hpos, rpos, tpos))

    print(">> Number of true triples: %d" % len(true_triples))
    fout = open(output_file, "w")
    for hpos, rpos, tpos in true_triples:
        fout.write(str(hpos) + "\t" + str(rpos) + "\t" + str(tpos) + "\n")
    fout.close()


def generate_mask_type(input_file, output_file):
    with open(output_file, "w") as fw:
        with open(input_file, "r") as fr:
            for line in fr.readlines():
                fw.write(line.strip('\r \n') + "\tMASK_HEAD\n")
                fw.write(line.strip('\r \n') + "\tMASK_TAIL\n")


def kbc_data_preprocess(train_file, dev_file, test_file, vocab_path,
                        true_triple_path, new_train_file, new_dev_file,
                        new_test_file):
    entity_lst, relation_lst = get_unique_entities_relations(
        train_file, dev_file, test_file)
    write_vocab(vocab_path, entity_lst, relation_lst)
    vocab = load_vocab(vocab_path)
    write_true_triples(train_file, dev_file, test_file, vocab,
                       true_triple_path)

    generate_mask_type(train_file, new_train_file)
    generate_mask_type(dev_file, new_dev_file)
    generate_mask_type(test_file, new_test_file)
```

```python id="n3GTzXRQrNv4"
TASK_NAME = 'fb15k'
TASK_DATA_PATH = '/content/data/fb15k'
```

```python id="Xehn5e9qqw_Y"
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        default=TASK_NAME,
        help="task name: fb15k, fb15k237, wn18rr, wn18, pathqueryFB, pathqueryWN"
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=TASK_DATA_PATH,
        help="task data directory")
    parser.add_argument(
        "--train",
        type=str,
        required=False,
        default="train.txt",
        help="train file name, default train.txt")
    parser.add_argument(
        "--valid",
        type=str,
        required=False,
        default="valid.txt",
        help="valid file name, default valid.txt")
    parser.add_argument(
        "--test",
        type=str,
        required=False,
        default="test.txt",
        help="test file name, default test.txt")
    args = parser.parse_args(args={})
    return args
```

```python colab={"base_uri": "https://localhost:8080/"} id="Lzi7TzUOq2h2" executionInfo={"status": "ok", "timestamp": 1633961222193, "user_tz": -330, "elapsed": 5617, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="4fd8556e-efee-4bb1-9ce1-996ba8572238"
args = get_args()
task = args.task.lower()
assert task in ["fb15k", "wn18", "fb15k237", "wn18rr"]

raw_train_file = os.path.join(args.dir, args.train)
raw_dev_file = os.path.join(args.dir, args.valid)
raw_test_file = os.path.join(args.dir, args.test)

vocab_file = os.path.join(args.dir, "vocab.txt")
true_triple_file = os.path.join(args.dir, "all.txt")
new_train_file = os.path.join(args.dir, "train.coke.txt")
new_test_file = os.path.join(args.dir, "test.coke.txt")
new_dev_file = os.path.join(args.dir, "valid.coke.txt")

kbc_data_preprocess(raw_train_file, raw_dev_file, raw_test_file,
                    vocab_file, true_triple_file, new_train_file,
                    new_dev_file, new_test_file)
```

```python id="Q81zKOWlswv3"
class Args:
    TASK='fb15k'
    NUM_VOCAB=16396  #NUM_VOCAB and NUM_RELATIONS must be consistent with vocab.txt file 
    NUM_RELATIONS=1345

    # training hyper-paramters
    BATCH_SIZE=512
    LEARNING_RATE=5e-4
    EPOCH=400
    SOFT_LABEL=0.8
    SKIP_STEPS=1000
    MAX_SEQ_LEN=3
    HIDDEN_DROPOUT_PROB=0.1
    ATTENTION_PROBS_DROPOUT_PROB=0.1

    # file paths for training and evaluation 
    DATA="./data"
    OUTPUT="./output_fb15k"
    TRAIN_FILE="./data/fb15k/train.coke.txt"
    VALID_FILE="./data/fb15k/valid.coke.txt"
    TEST_FILE="./data/fb15k/test.coke.txt"
    VOCAB_PATH="./data/fb15k/vocab.txt"
    TRUE_TRIPLE_PATH="./data/fb15k/all.txt"
    CHECKPOINTS="./output_fb15k/models"
    INIT_CHECKPOINTS=CHECKPOINTS
    LOG_FILE="./output_fb15k/train.log"
    LOG_EVAL_FILE="./output_fb15k/test.log"

    # transformer net config the follwoing are default configs for all tasks
    HIDDEN_SIZE=256
    NUM_HIDDEN_LAYERS=12
    NUM_ATTENTION_HEADS=4
    MAX_POSITION_EMBEDDINS=40

    hidden_size=256
    num_hidden_layers=6
    num_attention_heads=4
    vocab_size=-1
    num_relations=None
    max_position_embeddings=10
    hidden_act="gelu"
    hidden_dropout_prob=0.1
    attention_probs_dropout_prob=0.1
    initializer_range=0.02
    intermediate_size=512
    init_checkpoint=None
    init_pretraining_params=None
    checkpoints="checkpoints"
    weight_sharing=True
    epoch=100
    learning_rate=5e-5
    lr_scheduler="linear_warmup_decay"
    soft_label=0.9
    weight_decay=0.01
    warmup_proportion=0.1
    use_ema=True
    ema_decay=0.9999
    use_fp16=False
    loss_scaling=1.0
    skip_steps=1000
    verbose=False
    dataset=""
    train_file=None
    sen_candli_file=None
    sen_trivial_file=None
    predict_file=None
    vocab_path=None
    true_triple_path=None
    max_seq_len=3
    batch_size=12
    in_tokens=False
    do_train=False
    do_predict=False
    use_cuda=False
    use_fast_executor=False
    num_iteration_per_drop_scope=1

args =Args()
```

```python id="V96vxiictavi"
!pip install -q paddlepaddle
```

```python id="SKiV0ivEuVAG"
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import multiprocessing
import os
import time
import logging
import json
import random
import six

import numpy as np
import paddle
import paddle.fluid as fluid

from functools import partial, reduce
import numpy as np

import paddle.fluid as fluid
import paddle.fluid.layers as layers
from paddle.fluid.layer_helper import LayerHelper

import os
import six
import ast
import copy
import logging

import numpy as np
import paddle.fluid as fluid
```

```python id="sBVRcq3eu2Y5"
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)
```

```python id="Vbm2jUuXuj6x"
def str2bool(v):
    # because argparse does not support to parse "true, False" as python
    # boolean directly
    return v.lower() in ("true", "t", "1")


class ArgumentGroup(object):
    def __init__(self, parser, title, des):
        self._group = parser.add_argument_group(title=title, description=des)

    def add_arg(self, name, type, default, help, **kwargs):
        type = str2bool if type == bool else type
        self._group.add_argument(
            "--" + name,
            default=default,
            type=type,
            help=help + ' Default: %(default)s.',
            **kwargs)


def print_arguments(args):
    logger.info('-----------  Configuration Arguments -----------')
    for arg, value in sorted(six.iteritems(vars(args))):
        logger.info('%s: %s' % (arg, value))
    logger.info('------------------------------------------------')
```

```python id="GsXRDbDsumZf"
def cast_fp16_to_fp32(i, o, prog):
    prog.global_block().append_op(
        type="cast",
        inputs={"X": i},
        outputs={"Out": o},
        attrs={
            "in_dtype": fluid.core.VarDesc.VarType.FP16,
            "out_dtype": fluid.core.VarDesc.VarType.FP32
        })


def cast_fp32_to_fp16(i, o, prog):
    prog.global_block().append_op(
        type="cast",
        inputs={"X": i},
        outputs={"Out": o},
        attrs={
            "in_dtype": fluid.core.VarDesc.VarType.FP32,
            "out_dtype": fluid.core.VarDesc.VarType.FP16
        })


def copy_to_master_param(p, block):
    v = block.vars.get(p.name, None)
    if v is None:
        raise ValueError("no param name %s found!" % p.name)
    new_p = fluid.framework.Parameter(
        block=block,
        shape=v.shape,
        dtype=fluid.core.VarDesc.VarType.FP32,
        type=v.type,
        lod_level=v.lod_level,
        stop_gradient=p.stop_gradient,
        trainable=p.trainable,
        optimize_attr=p.optimize_attr,
        regularizer=p.regularizer,
        gradient_clip_attr=p.gradient_clip_attr,
        error_clip=p.error_clip,
        name=v.name + ".master")
    return new_p


def create_master_params_grads(params_grads, main_prog, startup_prog,
                               loss_scaling):
    master_params_grads = []
    tmp_role = main_prog._current_role
    OpRole = fluid.core.op_proto_and_checker_maker.OpRole
    main_prog._current_role = OpRole.Backward
    for p, g in params_grads:
        # create master parameters
        master_param = copy_to_master_param(p, main_prog.global_block())
        startup_master_param = startup_prog.global_block()._clone_variable(
            master_param)
        startup_p = startup_prog.global_block().var(p.name)
        cast_fp16_to_fp32(startup_p, startup_master_param, startup_prog)
        # cast fp16 gradients to fp32 before apply gradients
        if g.name.find("layer_norm") > -1:
            if loss_scaling > 1:
                scaled_g = g / float(loss_scaling)
            else:
                scaled_g = g
            master_params_grads.append([p, scaled_g])
            continue
        master_grad = fluid.layers.cast(g, "float32")
        if loss_scaling > 1:
            master_grad = master_grad / float(loss_scaling)
        master_params_grads.append([master_param, master_grad])
    main_prog._current_role = tmp_role
    return master_params_grads


def master_param_to_train_param(master_params_grads, params_grads, main_prog):
    for idx, m_p_g in enumerate(master_params_grads):
        train_p, _ = params_grads[idx]
        if train_p.name.find("layer_norm") > -1:
            continue
        with main_prog._optimized_guard([m_p_g[0], m_p_g[1]]):
            cast_fp32_to_fp16(m_p_g[0], train_p, main_prog)
```

```python id="0TY649WNug58"
def layer_norm(x,
               begin_norm_axis=1,
               epsilon=1e-12,
               param_attr=None,
               bias_attr=None):
    """
    Replace build-in layer_norm op with this function
    """
    helper = LayerHelper('layer_norm', **locals())
    mean = layers.reduce_mean(x, dim=begin_norm_axis, keep_dim=True)
    shift_x = layers.elementwise_sub(x=x, y=mean, axis=0)
    variance = layers.reduce_mean(
        layers.square(shift_x), dim=begin_norm_axis, keep_dim=True)
    r_stdev = layers.rsqrt(variance + epsilon)
    norm_x = layers.elementwise_mul(x=shift_x, y=r_stdev, axis=0)

    param_shape = [reduce(lambda x, y: x * y, norm_x.shape[begin_norm_axis:])]
    param_dtype = norm_x.dtype
    scale = helper.create_parameter(
        attr=param_attr,
        shape=param_shape,
        dtype=param_dtype,
        default_initializer=fluid.initializer.Constant(1.))
    bias = helper.create_parameter(
        attr=bias_attr,
        shape=param_shape,
        dtype=param_dtype,
        is_bias=True,
        default_initializer=fluid.initializer.Constant(0.))

    out = layers.elementwise_mul(x=norm_x, y=scale, axis=-1)
    out = layers.elementwise_add(x=out, y=bias, axis=-1)

    return out


def multi_head_attention(queries,
                         keys,
                         values,
                         attn_bias,
                         d_key,
                         d_value,
                         d_model,
                         n_head=1,
                         dropout_rate=0.,
                         cache=None,
                         param_initializer=None,
                         name='multi_head_att'):
    """
    Multi-Head Attention. Note that attn_bias is added to the logit before
    computing softmax activiation to mask certain selected positions so that
    they will not considered in attention weights.
    """
    keys = queries if keys is None else keys
    values = keys if values is None else values

    if not (len(queries.shape) == len(keys.shape) == len(values.shape) == 3):
        raise ValueError(
            "Inputs: quries, keys and values should all be 3-D tensors.")

    def __compute_qkv(queries, keys, values, n_head, d_key, d_value):
        """
        Add linear projection to queries, keys, and values.
        """
        q = layers.fc(input=queries,
                      size=d_key * n_head,
                      num_flatten_dims=2,
                      param_attr=fluid.ParamAttr(
                          name=name + '_query_fc.w_0',
                          initializer=param_initializer),
                      bias_attr=name + '_query_fc.b_0')
        k = layers.fc(input=keys,
                      size=d_key * n_head,
                      num_flatten_dims=2,
                      param_attr=fluid.ParamAttr(
                          name=name + '_key_fc.w_0',
                          initializer=param_initializer),
                      bias_attr=name + '_key_fc.b_0')
        v = layers.fc(input=values,
                      size=d_value * n_head,
                      num_flatten_dims=2,
                      param_attr=fluid.ParamAttr(
                          name=name + '_value_fc.w_0',
                          initializer=param_initializer),
                      bias_attr=name + '_value_fc.b_0')
        return q, k, v

    def __split_heads(x, n_head):
        """
        Reshape the last dimension of inpunt tensor x so that it becomes two
        dimensions and then transpose. Specifically, input a tensor with shape
        [bs, max_sequence_length, n_head * hidden_dim] then output a tensor
        with shape [bs, n_head, max_sequence_length, hidden_dim].
        """
        hidden_size = x.shape[-1]
        # The value 0 in shape attr means copying the corresponding dimension
        # size of the input as the output dimension size.
        reshaped = layers.reshape(
            x=x, shape=[0, 0, n_head, hidden_size // n_head], inplace=True)

        # permuate the dimensions into:
        # [batch_size, n_head, max_sequence_len, hidden_size_per_head]
        return layers.transpose(x=reshaped, perm=[0, 2, 1, 3])

    def __combine_heads(x):
        """
        Transpose and then reshape the last two dimensions of inpunt tensor x
        so that it becomes one dimension, which is reverse to __split_heads.
        """
        if len(x.shape) == 3: return x
        if len(x.shape) != 4:
            raise ValueError("Input(x) should be a 4-D Tensor.")

        trans_x = layers.transpose(x, perm=[0, 2, 1, 3])
        # The value 0 in shape attr means copying the corresponding dimension
        # size of the input as the output dimension size.
        return layers.reshape(
            x=trans_x,
            shape=[0, 0, trans_x.shape[2] * trans_x.shape[3]],
            inplace=True)

    def scaled_dot_product_attention(q, k, v, attn_bias, d_key, dropout_rate):
        """
        Scaled Dot-Product Attention
        """
        scaled_q = layers.scale(x=q, scale=d_key**-0.5)
        product = layers.matmul(x=scaled_q, y=k, transpose_y=True)
        if attn_bias:
            product += attn_bias
        weights = layers.softmax(product)
        if dropout_rate:
            weights = layers.dropout(
                weights,
                dropout_prob=dropout_rate,
                dropout_implementation="upscale_in_train",
                is_test=False)
        out = layers.matmul(weights, v)
        return out

    q, k, v = __compute_qkv(queries, keys, values, n_head, d_key, d_value)

    if cache is not None:  # use cache and concat time steps
        # Since the inplace reshape in __split_heads changes the shape of k and
        # v, which is the cache input for next time step, reshape the cache
        # input from the previous time step first.
        k = cache["k"] = layers.concat(
            [layers.reshape(
                cache["k"], shape=[0, 0, d_model]), k], axis=1)
        v = cache["v"] = layers.concat(
            [layers.reshape(
                cache["v"], shape=[0, 0, d_model]), v], axis=1)

    q = __split_heads(q, n_head)
    k = __split_heads(k, n_head)
    v = __split_heads(v, n_head)

    ctx_multiheads = scaled_dot_product_attention(q, k, v, attn_bias, d_key,
                                                  dropout_rate)

    out = __combine_heads(ctx_multiheads)

    # Project back to the model size.
    proj_out = layers.fc(input=out,
                         size=d_model,
                         num_flatten_dims=2,
                         param_attr=fluid.ParamAttr(
                             name=name + '_output_fc.w_0',
                             initializer=param_initializer),
                         bias_attr=name + '_output_fc.b_0')
    return proj_out


def positionwise_feed_forward(x,
                              d_inner_hid,
                              d_hid,
                              dropout_rate,
                              hidden_act,
                              param_initializer=None,
                              name='ffn'):
    """
    Position-wise Feed-Forward Networks.
    This module consists of two linear transformations with a ReLU activation
    in between, which is applied to each position separately and identically.
    """
    hidden = layers.fc(input=x,
                       size=d_inner_hid,
                       num_flatten_dims=2,
                       act=hidden_act,
                       param_attr=fluid.ParamAttr(
                           name=name + '_fc_0.w_0',
                           initializer=param_initializer),
                       bias_attr=name + '_fc_0.b_0')
    if dropout_rate:
        hidden = layers.dropout(
            hidden,
            dropout_prob=dropout_rate,
            dropout_implementation="upscale_in_train",
            is_test=False)
    out = layers.fc(input=hidden,
                    size=d_hid,
                    num_flatten_dims=2,
                    param_attr=fluid.ParamAttr(
                        name=name + '_fc_1.w_0',
                        initializer=param_initializer),
                    bias_attr=name + '_fc_1.b_0')
    return out


def pre_post_process_layer(prev_out,
                           out,
                           process_cmd,
                           dropout_rate=0.,
                           name=''):
    """
    Add residual connection, layer normalization and droput to the out tensor
    optionally according to the value of process_cmd.
    This will be used before or after multi-head attention and position-wise
    feed-forward networks.
    """
    for cmd in process_cmd:
        if cmd == "a":  # add residual connection
            out = out + prev_out if prev_out else out
        elif cmd == "n":  # add layer normalization
            out_dtype = out.dtype
            if out_dtype == fluid.core.VarDesc.VarType.FP16:
                out = layers.cast(x=out, dtype="float32")
            out = layer_norm(
                out,
                begin_norm_axis=len(out.shape) - 1,
                param_attr=fluid.ParamAttr(
                    name=name + '_layer_norm_scale',
                    initializer=fluid.initializer.Constant(1.)),
                bias_attr=fluid.ParamAttr(
                    name=name + '_layer_norm_bias',
                    initializer=fluid.initializer.Constant(0.)))
            if out_dtype == fluid.core.VarDesc.VarType.FP16:
                out = layers.cast(x=out, dtype="float16")
        elif cmd == "d":  # add dropout
            if dropout_rate:
                out = layers.dropout(
                    out,
                    dropout_prob=dropout_rate,
                    dropout_implementation="upscale_in_train",
                    is_test=False)
    return out


pre_process_layer = partial(pre_post_process_layer, None)
post_process_layer = pre_post_process_layer


def encoder_layer(enc_input,
                  attn_bias,
                  n_head,
                  d_key,
                  d_value,
                  d_model,
                  d_inner_hid,
                  prepostprocess_dropout,
                  attention_dropout,
                  relu_dropout,
                  hidden_act,
                  preprocess_cmd="n",
                  postprocess_cmd="da",
                  param_initializer=None,
                  name=''):
    """The encoder layers that can be stacked to form a deep encoder.
    This module consits of a multi-head (self) attention followed by
    position-wise feed-forward networks and both the two components companied
    with the post_process_layer to add residual connection, layer normalization
    and droput.
    """
    attn_output = multi_head_attention(
        pre_process_layer(
            enc_input,
            preprocess_cmd,
            prepostprocess_dropout,
            name=name + '_pre_att'),
        None,
        None,
        attn_bias,
        d_key,
        d_value,
        d_model,
        n_head,
        attention_dropout,
        param_initializer=param_initializer,
        name=name + '_multi_head_att')
    attn_output = post_process_layer(
        enc_input,
        attn_output,
        postprocess_cmd,
        prepostprocess_dropout,
        name=name + '_post_att')
    ffd_output = positionwise_feed_forward(
        pre_process_layer(
            attn_output,
            preprocess_cmd,
            prepostprocess_dropout,
            name=name + '_pre_ffn'),
        d_inner_hid,
        d_model,
        relu_dropout,
        hidden_act,
        param_initializer=param_initializer,
        name=name + '_ffn')
    return post_process_layer(
        attn_output,
        ffd_output,
        postprocess_cmd,
        prepostprocess_dropout,
        name=name + '_post_ffn')


def encoder(enc_input,
            attn_bias,
            n_layer,
            n_head,
            d_key,
            d_value,
            d_model,
            d_inner_hid,
            prepostprocess_dropout,
            attention_dropout,
            relu_dropout,
            hidden_act,
            preprocess_cmd="n",
            postprocess_cmd="da",
            param_initializer=None,
            name=''):
    """
    The encoder is composed of a stack of identical layers returned by calling
    encoder_layer.
    """
    for i in range(n_layer):
        enc_output = encoder_layer(
            enc_input,
            attn_bias,
            n_head,
            d_key,
            d_value,
            d_model,
            d_inner_hid,
            prepostprocess_dropout,
            attention_dropout,
            relu_dropout,
            hidden_act,
            preprocess_cmd,
            postprocess_cmd,
            param_initializer=param_initializer,
            name=name + '_layer_' + str(i))
        enc_input = enc_output
    enc_output = pre_process_layer(
        enc_output,
        preprocess_cmd,
        prepostprocess_dropout,
        name="post_encoder")

    return enc_output
```

```python id="8f-ejWXzt5Rz"
def mask(input_tokens, input_mask_type, max_len, mask_id):
    """
    Add mask for batch_tokens, return out, mask_label, mask_pos;
    Note: mask_pos responding the batch_tokens after padded;
    """
    output_tokens = []
    mask_label = []
    mask_pos = []
    for sent_index, sent in enumerate(input_tokens):
        mask_type = input_mask_type[sent_index]
        if mask_type == "MASK_HEAD":
            token_index = 0
            mask_label.append(sent[token_index])
            mask_pos.append(sent_index * max_len + token_index)
            sent_out = sent[:]
            sent_out[token_index] = mask_id
            output_tokens.append(sent_out)
        elif mask_type == "MASK_TAIL":
            token_index = len(sent) - 1
            mask_label.append(sent[token_index])
            mask_pos.append(sent_index * max_len + token_index)
            sent_out = sent[:]
            sent_out[token_index] = mask_id
            output_tokens.append(sent_out)
        else:
            raise ValueError(
                "Unknown mask type, which should be in ['MASK_HEAD', 'MASK_TAIL']."
            )
    mask_label = np.array(mask_label).astype("int64").reshape([-1, 1])
    mask_pos = np.array(mask_pos).astype("int64").reshape([-1, 1])
    return output_tokens, mask_label, mask_pos


def pad_batch_data(insts,
                   max_len,
                   pad_idx=0,
                   return_pos=False,
                   return_input_mask=False):
    """
    Pad the instances to the max sequence length in batch, and generate the
    corresponding position data and input mask.
    """
    return_list = []

    # Any token included in dict can be used to pad, since the paddings' loss
    # will be masked out by weights and make no effect on parameter gradients.

    inst_data = np.array([
        list(inst) + list([pad_idx] * (max_len - len(inst))) for inst in insts
    ])
    return_list += [inst_data.astype("int64").reshape([-1, max_len, 1])]

    # position data
    if return_pos:
        inst_pos = np.array([
            list(range(0, len(inst))) + [pad_idx] * (max_len - len(inst))
            for inst in insts
        ])

        return_list += [inst_pos.astype("int64").reshape([-1, max_len, 1])]

    if return_input_mask:
        # This is used to avoid attention on paddings.
        input_mask_data = np.array([[1] * len(inst) + [0] *
                                    (max_len - len(inst)) for inst in insts])
        input_mask_data = np.expand_dims(input_mask_data, axis=-1)
        return_list += [input_mask_data.astype("float32")]

    return return_list if len(return_list) > 1 else return_list[0]


def prepare_batch_data(insts, max_len, pad_id=None, mask_id=None):
    """ masking, padding, turn list data into numpy arrays, for batch examples
    """
    batch_src_ids = [inst[0] for inst in insts]
    batch_mask_type = [inst[1] for inst in insts]

    # First step: do mask without padding
    if mask_id >= 0:
        out, mask_label, mask_pos = mask(
            input_tokens=batch_src_ids,
            input_mask_type=batch_mask_type,
            max_len=max_len,
            mask_id=mask_id)
    else:
        out = batch_src_ids

    # Second step: padding and turn into numpy arrays
    src_id, pos_id, input_mask = pad_batch_data(
        out,
        max_len=max_len,
        pad_idx=pad_id,
        return_pos=True,
        return_input_mask=True)

    if mask_id >= 0:
        return_list = [src_id, pos_id, input_mask, mask_label, mask_pos]
    else:
        return_list = [src_id, pos_id, input_mask]

    return return_list if len(return_list) > 1 else return_list[0]
```

```python id="s0TpSytNt6JQ"
RawExample = collections.namedtuple("RawExample", ["token_ids", "mask_type"])


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


#def printable_text(text):
#    """Returns text encoded in a way suitable for print or `tf.logging`."""
#
#    # These functions want `str` for both Python2 and Python3, but in one case
#    # it's a Unicode string and in the other it's a byte string.
#    if six.PY3:
#        if isinstance(text, str):
#            return text
#        elif isinstance(text, bytes):
#            return text.decode("utf-8", "ignore")
#        else:
#            raise ValueError("Unsupported string type: %s" % (type(text)))
#    elif six.PY2:
#        if isinstance(text, str):
#            return text
#        elif isinstance(text, unicode):
#            return text.encode("utf-8")
#        else:
#            raise ValueError("Unsupported string type: %s" % (type(text)))
#    else:
#        raise ValueError("Not running on Python2 or Python 3?")


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    fin = open(vocab_file)
    for num, line in enumerate(fin):
        items = line.strip().split("\t")
        if len(items) > 2:
            break
        token = items[0]
        index = items[1] if len(items) == 2 else num
        token = token.strip()
        vocab[token] = int(index)
    return vocab


#def convert_by_vocab(vocab, items):
#    """Converts a sequence of [tokens|ids] using the vocab."""
#    output = []
#    for item in items:
#        output.append(vocab[item])
#    return output


def convert_tokens_to_ids(vocab, tokens):
    """Converts a sequence of tokens into ids using the vocab."""
    output = []
    for item in tokens:
        output.append(vocab[item])
    return output


class KBCDataReader(object):
    """ DataReader
    """

    def __init__(self,
                 vocab_path,
                 data_path,
                 max_seq_len=3,
                 batch_size=4096,
                 is_training=True,
                 shuffle=True,
                 dev_count=1,
                 epoch=10,
                 vocab_size=-1):
        self.vocab = load_vocab(vocab_path)
        if vocab_size > 0:
            assert len(self.vocab) == vocab_size, \
                "Assert Error! Input vocab_size(%d) is not consistant with voab_file(%d)" % \
                (vocab_size, len(self.vocab))
        self.pad_id = self.vocab["[PAD]"]
        self.mask_id = self.vocab["[MASK]"]

        self.max_seq_len = max_seq_len
        self.batch_size = batch_size

        self.is_training = is_training
        self.shuffle = shuffle
        self.dev_count = dev_count
        self.epoch = epoch
        if not is_training:
            self.shuffle = False
            self.dev_count = 1
            self.epoch = 1

        self.examples = self.read_example(data_path)
        self.total_instance = len(self.examples)

        self.current_epoch = -1
        self.current_instance_index = -1

    def get_progress(self):
        """return current progress of traning data
        """
        return self.current_instance_index, self.current_epoch

    def line2tokens(self, line):
        tokens = line.split("\t")
        return tokens

    def read_example(self, input_file):
        """Reads the input file into a list of examples."""
        examples = []
        with open(input_file, "r") as f:
            for line in f.readlines():
                line = convert_to_unicode(line.strip())
                tokens = self.line2tokens(line)
                assert len(tokens) <= (self.max_seq_len + 1), \
                    "Expecting at most [max_seq_len + 1]=%d tokens each line, current tokens %d" \
                    % (self.max_seq_len + 1, len(tokens))
                token_ids = convert_tokens_to_ids(self.vocab, tokens[:-1])
                if len(token_ids) <= 0:
                    continue
                examples.append(
                    RawExample(
                        token_ids=token_ids, mask_type=tokens[-1]))
                # if len(examples) <= 10:
                #     logger.info("*** Example ***")
                #     logger.info("tokens: %s" % " ".join([printable_text(x) for x in tokens]))
                #     logger.info("token_ids: %s" % " ".join([str(x) for x in token_ids]))
        return examples

    def data_generator(self):
        """ wrap the batch data generator
        """
        range_list = [i for i in range(self.total_instance)]

        def wrapper():
            """ wrapper batch data
            """

            def reader():
                for epoch_index in range(self.epoch):
                    self.current_epoch = epoch_index
                    if self.shuffle is True:
                        np.random.shuffle(range_list)
                    for idx, sample in enumerate(range_list):
                        self.current_instance_index = idx
                        yield self.examples[sample]

            def batch_reader(reader, batch_size):
                """reader generator for batches of examples
                :param reader: reader generator for one example
                :param batch_size: int batch size
                :return: a list of examples for batch data
                """
                batch = []
                for example in reader():
                    token_ids = example.token_ids
                    mask_type = example.mask_type
                    example_out = [token_ids] + [mask_type]
                    to_append = len(batch) < batch_size
                    if to_append is False:
                        yield batch
                        batch = [example_out]
                    else:
                        batch.append(example_out)
                if len(batch) > 0:
                    yield batch

            all_device_batches = []
            for batch_data in batch_reader(reader, self.batch_size):
                batch_data = prepare_batch_data(
                    batch_data,
                    max_len=self.max_seq_len,
                    pad_id=self.pad_id,
                    mask_id=self.mask_id)
                if len(all_device_batches) < self.dev_count:
                    all_device_batches.append(batch_data)

                if len(all_device_batches) == self.dev_count:
                    for batch in all_device_batches:
                        yield batch
                    all_device_batches = []

        return wrapper


class PathqueryDataReader(KBCDataReader):
    def __init__(self,
                 vocab_path,
                 data_path,
                 max_seq_len=3,
                 batch_size=4096,
                 is_training=True,
                 shuffle=True,
                 dev_count=1,
                 epoch=10,
                 vocab_size=-1):

        KBCDataReader.__init__(self, vocab_path, data_path, max_seq_len,
                               batch_size, is_training, shuffle, dev_count,
                               epoch, vocab_size)

    def line2tokens(self, line):
        tokens = []
        s, path, o, mask_type = line.split("\t")
        path_tokens = path.split(",")
        tokens.append(s)
        tokens.extend(path_tokens)
        tokens.append(o)
        tokens.append(mask_type)
        return tokens
```

```python id="Wqa-q13Yt6G1"
def cast_fp32_to_fp16(exe, main_program):
    logger.info("Cast parameters to float16 data format.")
    for param in main_program.global_block().all_parameters():
        if not param.name.endswith(".master"):
            param_t = fluid.global_scope().find_var(param.name).get_tensor()
            data = np.array(param_t)
            if param.name.find("layer_norm") == -1:
                param_t.set(np.float16(data).view(np.uint16), exe.place)
            master_param_var = fluid.global_scope().find_var(param.name +
                                                             ".master")
            if master_param_var is not None:
                master_param_var.get_tensor().set(data, exe.place)


def init_checkpoint(exe,
                    init_checkpoint_path,
                    main_program,
                    use_fp16=False,
                    print_var_verbose=False):
    assert os.path.exists(
        init_checkpoint_path), "[%s] cann't be found." % init_checkpoint_path

    def existed_persitables(var):
        if not fluid.io.is_persistable(var):
            return False
        return os.path.exists(os.path.join(init_checkpoint_path, var.name))

    fluid.io.load_vars(
        exe,
        init_checkpoint_path,
        main_program=main_program,
        predicate=existed_persitables)
    logger.info("Load model from {}".format(init_checkpoint_path))

    if use_fp16:
        cast_fp32_to_fp16(exe, main_program)

    # Used for debug on parameters
    if print_var_verbose is True:

        def params(var):
            if not isinstance(var, fluid.framework.Parameter):
                return False
            return True

        existed_vars = list(filter(params, main_program.list_vars()))
        existed_vars = sorted(existed_vars, key=lambda x: x.name)
        for var in existed_vars:
            logger.info("var name:{} shape:{}".format(var.name, var.shape))


def init_pretraining_params(exe,
                            pretraining_params_path,
                            main_program,
                            use_fp16=False):
    assert os.path.exists(pretraining_params_path
                          ), "[%s] cann't be found." % pretraining_params_path

    def existed_params(var):
        if not isinstance(var, fluid.framework.Parameter):
            return False
        return os.path.exists(os.path.join(pretraining_params_path, var.name))

    fluid.io.load_vars(
        exe,
        pretraining_params_path,
        main_program=main_program,
        predicate=existed_params)
    logger.info("Load pretraining parameters from {}.".format(
        pretraining_params_path))

    if use_fp16:
        cast_fp32_to_fp16(exe, main_program)
```

```python id="ZiE9AuVKt6Co"
class CoKEModel(object):
    def __init__(self,
                 src_ids,
                 position_ids,
                 input_mask,
                 config,
                 soft_label=0.9,
                 weight_sharing=True,
                 use_fp16=False):

        self._emb_size = config['hidden_size']
        self._n_layer = config['num_hidden_layers']
        self._n_head = config['num_attention_heads']
        self._voc_size = config['vocab_size']
        self._n_relation = config['num_relations']
        self._max_position_seq_len = config['max_position_embeddings']
        self._hidden_act = config['hidden_act']
        self._prepostprocess_dropout = config['hidden_dropout_prob']
        self._attention_dropout = config['attention_probs_dropout_prob']
        self._intermediate_size = config['intermediate_size']
        self._soft_label = soft_label
        self._weight_sharing = weight_sharing

        self._word_emb_name = "word_embedding"
        self._pos_emb_name = "pos_embedding"
        self._dtype = "float16" if use_fp16 else "float32"

        # Initialize all weigths by truncated normal initializer, and all biases
        # will be initialized by constant zero by default.
        self._param_initializer = fluid.initializer.TruncatedNormal(
            scale=config['initializer_range'])

        self._build_model(src_ids, position_ids, input_mask)

    def _build_model(self, src_ids, position_ids, input_mask):
        # padding id in vocabulary must be set to 0
        emb_out = fluid.layers.embedding(
            input=src_ids,
            size=[self._voc_size, self._emb_size],
            dtype=self._dtype,
            param_attr=fluid.ParamAttr(
                name=self._word_emb_name, initializer=self._param_initializer),
            is_sparse=False)
        position_emb_out = fluid.layers.embedding(
            input=position_ids,
            size=[self._max_position_seq_len, self._emb_size],
            dtype=self._dtype,
            param_attr=fluid.ParamAttr(
                name=self._pos_emb_name, initializer=self._param_initializer))

        emb_out = emb_out + position_emb_out

        emb_out = pre_process_layer(
            emb_out, 'nd', self._prepostprocess_dropout, name='pre_encoder')

        if self._dtype == "float16":
            input_mask = fluid.layers.cast(x=input_mask, dtype=self._dtype)

        self_attn_mask = fluid.layers.matmul(
            x=input_mask, y=input_mask, transpose_y=True)
        self_attn_mask = fluid.layers.scale(
            x=self_attn_mask, scale=10000.0, bias=-1.0, bias_after_scale=False)
        n_head_self_attn_mask = fluid.layers.stack(
            x=[self_attn_mask] * self._n_head, axis=1)
        n_head_self_attn_mask.stop_gradient = True

        self._enc_out = encoder(
            enc_input=emb_out,
            attn_bias=n_head_self_attn_mask,
            n_layer=self._n_layer,
            n_head=self._n_head,
            d_key=self._emb_size // self._n_head,
            d_value=self._emb_size // self._n_head,
            d_model=self._emb_size,
            d_inner_hid=self._intermediate_size,
            prepostprocess_dropout=self._prepostprocess_dropout,
            attention_dropout=self._attention_dropout,
            relu_dropout=0,
            hidden_act=self._hidden_act,
            preprocess_cmd="",
            postprocess_cmd="dan",
            param_initializer=self._param_initializer,
            name='encoder')

    #def get_sequence_output(self):
    #    return self._enc_out

    def get_pretraining_output(self, mask_label, mask_pos):
        """Get the loss & fc_out for training"""
        mask_pos = fluid.layers.cast(x=mask_pos, dtype='int32')

        reshaped_emb_out = fluid.layers.reshape(
            x=self._enc_out, shape=[-1, self._emb_size])
        # extract masked tokens' feature
        mask_feat = fluid.layers.gather(input=reshaped_emb_out, index=mask_pos)

        # transform: fc
        mask_trans_feat = fluid.layers.fc(
            input=mask_feat,
            size=self._emb_size,
            act=self._hidden_act,
            param_attr=fluid.ParamAttr(
                name='mask_lm_trans_fc.w_0',
                initializer=self._param_initializer),
            bias_attr=fluid.ParamAttr(name='mask_lm_trans_fc.b_0'))
        # transform: layer norm
        mask_trans_feat = pre_process_layer(
            mask_trans_feat, 'n', name='mask_lm_trans')

        mask_lm_out_bias_attr = fluid.ParamAttr(
            name="mask_lm_out_fc.b_0",
            initializer=fluid.initializer.Constant(value=0.0))
        if self._weight_sharing:
            fc_out = fluid.layers.matmul(
                x=mask_trans_feat,
                y=fluid.default_main_program().global_block().var(
                    self._word_emb_name),
                transpose_y=True)
            fc_out += fluid.layers.create_parameter(
                shape=[self._voc_size],
                dtype=self._dtype,
                attr=mask_lm_out_bias_attr,
                is_bias=True)
        else:
            fc_out = fluid.layers.fc(input=mask_trans_feat,
                                     size=self._voc_size,
                                     param_attr=fluid.ParamAttr(
                                         name="mask_lm_out_fc.w_0",
                                         initializer=self._param_initializer),
                                     bias_attr=mask_lm_out_bias_attr)
        #generate soft labels for loss cross entropy loss
        one_hot_labels = fluid.layers.one_hot(
            input=mask_label, depth=self._voc_size)
        entity_indicator = fluid.layers.fill_constant_batch_size_like(
            input=mask_label,
            shape=[-1, (self._voc_size - self._n_relation)],
            dtype='int64',
            value=0)
        relation_indicator = fluid.layers.fill_constant_batch_size_like(
            input=mask_label,
            shape=[-1, self._n_relation],
            dtype='int64',
            value=1)
        is_relation = fluid.layers.concat(
            input=[entity_indicator, relation_indicator], axis=-1)
        soft_labels = one_hot_labels * self._soft_label \
                      + (1.0 - one_hot_labels - is_relation) \
                      * ((1.0 - self._soft_label) / (self._voc_size - 1 - self._n_relation))
        soft_labels.stop_gradient = True

        mask_lm_loss = fluid.layers.softmax_with_cross_entropy(
            logits=fc_out, label=soft_labels, soft_label=True)
        mean_mask_lm_loss = fluid.layers.mean(mask_lm_loss)

        return mean_mask_lm_loss, fc_out
```

```python id="SqHYnH5YvTTG"
def kbc_batch_evaluation(eval_i, all_examples, batch_results, tt):
    r_hts_idx = collections.defaultdict(list)
    scores_head = collections.defaultdict(list)
    scores_tail = collections.defaultdict(list)
    batch_r_hts_cnt = 0
    b_size = len(batch_results)
    for j in range(b_size):
        result = batch_results[j]
        i = eval_i + j
        example = all_examples[i]
        assert len(example.token_ids
                   ) == 3, "For kbc task each example consists of 3 tokens"
        h, r, t = example.token_ids

        _mask_type = example.mask_type
        if i % 2 == 0:
            r_hts_idx[r].append((h, t))
            batch_r_hts_cnt += 1
        if _mask_type == "MASK_HEAD":
            scores_head[(r, t)] = result
        elif _mask_type == "MASK_TAIL":
            scores_tail[(r, h)] = result
        else:
            raise ValueError("Unknown mask type in prediction example:%d" % i)

    rank = {}
    f_rank = {}
    for r, hts in r_hts_idx.items():
        r_rank = {'head': [], 'tail': []}
        r_f_rank = {'head': [], 'tail': []}
        for h, t in hts:
            scores_t = scores_tail[(r, h)][:]
            sortidx_t = np.argsort(scores_t)[::-1]
            r_rank['tail'].append(np.where(sortidx_t == t)[0][0] + 1)

            rm_idx = tt[r]['ts'][h]
            rm_idx = [i for i in rm_idx if i != t]
            for i in rm_idx:
                scores_t[i] = -np.Inf
            sortidx_t = np.argsort(scores_t)[::-1]
            r_f_rank['tail'].append(np.where(sortidx_t == t)[0][0] + 1)

            scores_h = scores_head[(r, t)][:]
            sortidx_h = np.argsort(scores_h)[::-1]
            r_rank['head'].append(np.where(sortidx_h == h)[0][0] + 1)

            rm_idx = tt[r]['hs'][t]
            rm_idx = [i for i in rm_idx if i != h]
            for i in rm_idx:
                scores_h[i] = -np.Inf
            sortidx_h = np.argsort(scores_h)[::-1]
            r_f_rank['head'].append(np.where(sortidx_h == h)[0][0] + 1)
        rank[r] = r_rank
        f_rank[r] = r_f_rank

    h_pos = [p for k in rank.keys() for p in rank[k]['head']]
    t_pos = [p for k in rank.keys() for p in rank[k]['tail']]
    f_h_pos = [p for k in f_rank.keys() for p in f_rank[k]['head']]
    f_t_pos = [p for k in f_rank.keys() for p in f_rank[k]['tail']]

    ranks = np.asarray(h_pos + t_pos)
    f_ranks = np.asarray(f_h_pos + f_t_pos)
    return ranks, f_ranks


def pathquery_batch_evaluation(eval_i, all_examples, batch_results,
                               sen_negli_dict, trivial_sen_set):
    """ evaluate the metrics for batch datas for pathquery datasets """
    mqs = []
    ranks = []
    for j, result in enumerate(batch_results):
        i = eval_i + j
        example = all_examples[i]
        token_ids, mask_type = example
        assert mask_type in ["MASK_TAIL", "MASK_HEAD"
                             ], " Unknown mask type in pathquery evaluation"
        label = token_ids[-1] if mask_type == "MASK_TAIL" else token_ids[0]

        sen = " ".join([str(x) for x in token_ids])
        if sen in trivial_sen_set:
            mq = rank = -1
        else:
            # candidate vocab set
            cand_set = sen_negli_dict[sen]
            assert label in set(
                cand_set), "predict label must be in the candidate set"

            cand_idx = np.sort(np.array(cand_set))
            cand_ret = result[
                cand_idx]  #logits for candidate words(neg + gold words)
            cand_ranks = np.argsort(cand_ret)[::-1]
            pred_y = cand_idx[cand_ranks]

            rank = (np.argwhere(pred_y == label).ravel().tolist())[0] + 1
            mq = (len(cand_set) - rank) / (len(cand_set) - 1.0)
        mqs.append(mq)
        ranks.append(rank)
    return mqs, ranks


def compute_kbc_metrics(rank_li, frank_li, output_evaluation_result_file):
    """ combine the kbc rank results from batches into the final metrics """
    rank_rets = np.array(rank_li).ravel()
    frank_rets = np.array(frank_li).ravel()
    mrr = np.mean(1.0 / rank_rets)
    fmrr = np.mean(1.0 / frank_rets)

    hits1 = np.mean(rank_rets <= 1.0)
    hits3 = np.mean(rank_rets <= 3.0)
    hits10 = np.mean(rank_rets <= 10.0)
    # filtered metrics
    fhits1 = np.mean(frank_rets <= 1.0)
    fhits3 = np.mean(frank_rets <= 3.0)
    fhits10 = np.mean(frank_rets <= 10.0)

    eval_result = {
        'mrr': mrr,
        'hits1': hits1,
        'hits3': hits3,
        'hits10': hits10,
        'fmrr': fmrr,
        'fhits1': fhits1,
        'fhits3': fhits3,
        'fhits10': fhits10
    }
    with open(output_evaluation_result_file, "w") as fw:
        fw.write(json.dumps(eval_result, indent=4) + "\n")
    return eval_result


def compute_pathquery_metrics(mq_li, rank_li, output_evaluation_result_file):
    """ combine the pathquery mq, rank results from batches into the final metrics """
    rank_rets = np.array(rank_li).ravel()
    _idx = np.where(rank_rets != -1)

    non_trivial_eval_rets = rank_rets[_idx]
    non_trivial_mq = np.array(mq_li).ravel()[_idx]
    non_trivial_cnt = non_trivial_eval_rets.size

    mq = np.mean(non_trivial_mq)
    mr = np.mean(non_trivial_eval_rets)
    mrr = np.mean(1.0 / non_trivial_eval_rets)
    fhits10 = np.mean(non_trivial_eval_rets <= 10.0)

    eval_result = {
        'fcnt': non_trivial_cnt,
        'mq': mq,
        'mr': mr,
        'fhits10': fhits10
    }

    with open(output_evaluation_result_file, "w") as fw:
        fw.write(json.dumps(eval_result, indent=4) + "\n")
    return eval_result
```

```python id="fD5NX0mdvTQK"
def linear_warmup_decay(learning_rate, warmup_steps, num_train_steps):
    """ Applies linear warmup of learning rate from 0 and decay to 0."""
    with fluid.default_main_program()._lr_schedule_guard():
        lr = fluid.layers.tensor.create_global_var(
            shape=[1],
            value=0.0,
            dtype='float32',
            persistable=True,
            name="scheduled_learning_rate")

        global_step = fluid.layers.learning_rate_scheduler._decay_step_counter(
        )

        with fluid.layers.control_flow.Switch() as switch:
            with switch.case(global_step < num_train_steps * 0.1):
                warmup_lr = learning_rate * (global_step /
                                             (num_train_steps * 0.1))
                fluid.layers.tensor.assign(warmup_lr, lr)
            with switch.default():
                decayed_lr = fluid.layers.learning_rate_scheduler.polynomial_decay(
                    learning_rate=learning_rate,
                    decay_steps=num_train_steps,
                    end_learning_rate=0.0,
                    power=1.0,
                    cycle=False)
                fluid.layers.tensor.assign(decayed_lr, lr)

        return lr


def optimization(loss,
                 warmup_steps,
                 num_train_steps,
                 learning_rate,
                 train_program,
                 startup_prog,
                 weight_decay,
                 scheduler='linear_warmup_decay',
                 use_fp16=False,
                 loss_scaling=1.0):
    if warmup_steps > 0:
        if scheduler == 'noam_decay':
            scheduled_lr = fluid.layers.learning_rate_scheduler\
             .noam_decay(1/(warmup_steps *(learning_rate ** 2)),
                         warmup_steps)
        elif scheduler == 'linear_warmup_decay':
            scheduled_lr = linear_warmup_decay(learning_rate, warmup_steps,
                                               num_train_steps)
        else:
            raise ValueError("Unkown learning rate scheduler, should be "
                             "'noam_decay' or 'linear_warmup_decay'")
        optimizer = fluid.optimizer.Adam(
            learning_rate=scheduled_lr, epsilon=1e-6)
    else:
        optimizer = fluid.optimizer.Adam(
            learning_rate=learning_rate, epsilon=1e-6)
        scheduled_lr = learning_rate

    clip_norm_thres = 1.0
    # When using mixed precision training, scale the gradient clip threshold
    # by loss_scaling
    if use_fp16 and loss_scaling > 1.0:
        clip_norm_thres *= loss_scaling
    fluid.clip.set_gradient_clip(
        clip=fluid.clip.GradientClipByGlobalNorm(clip_norm=clip_norm_thres))

    def exclude_from_weight_decay(name):
        if name.find("layer_norm") > -1:
            return True
        bias_suffix = ["_bias", "_b", ".b_0"]
        for suffix in bias_suffix:
            if name.endswith(suffix):
                return True
        return False

    param_list = dict()

    if use_fp16:
        param_grads = optimizer.backward(loss)
        master_param_grads = create_master_params_grads(
            param_grads, train_program, startup_prog, loss_scaling)

        for param, _ in master_param_grads:
            param_list[param.name] = param * 1.0
            param_list[param.name].stop_gradient = True

        optimizer.apply_gradients(master_param_grads)

        if weight_decay > 0:
            for param, grad in master_param_grads:
                # if exclude_from_weight_decay(param.name.rstrip(".master")):
                #     continue
                with param.block.program._optimized_guard(
                    [param, grad]), fluid.framework.name_scope("weight_decay"):
                    updated_param = param - param_list[
                        param.name] * weight_decay * scheduled_lr
                    fluid.layers.assign(output=param, input=updated_param)

        master_param_to_train_param(master_param_grads, param_grads,
                                    train_program)

    else:
        for param in train_program.global_block().all_parameters():
            param_list[param.name] = param * 1.0
            param_list[param.name].stop_gradient = True

        _, param_grads = optimizer.minimize(loss)

        if weight_decay > 0:
            for param, grad in param_grads:
                # if exclude_from_weight_decay(param.name):
                #     continue
                with param.block.program._optimized_guard(
                    [param, grad]), fluid.framework.name_scope("weight_decay"):
                    updated_param = param - param_list[
                        param.name] * weight_decay * scheduled_lr
                    fluid.layers.assign(output=param, input=updated_param)

    return scheduled_lr
```

```python id="ncwP_RQHtr6f"
# # yapf: disable
# parser = argparse.ArgumentParser()
# model_g = ArgumentGroup(parser, "model", "model configuration and paths.")
# model_g.add_arg("hidden_size",              int, 256,            "CoKE model config: hidden size, default 256")
# model_g.add_arg("num_hidden_layers",        int, 6,              "CoKE model config: num_hidden_layers, default 6")
# model_g.add_arg("num_attention_heads",      int, 4,              "CoKE model config: num_attention_heads, default 4")
# model_g.add_arg("vocab_size",               int, -1,           "CoKE model config: vocab_size")
# model_g.add_arg("num_relations",         int, None,           "CoKE model config: vocab_size")
# model_g.add_arg("max_position_embeddings",  int, 10,             "CoKE model config: max_position_embeddings")
# model_g.add_arg("hidden_act",               str, "gelu",         "CoKE model config: hidden_ac, default gelu")
# model_g.add_arg("hidden_dropout_prob",      float, 0.1,          "CoKE model config: attention_probs_dropout_prob, default 0.1")
# model_g.add_arg("attention_probs_dropout_prob", float, 0.1,      "CoKE model config: attention_probs_dropout_prob, default 0.1")
# model_g.add_arg("initializer_range",        int, 0.02,           "CoKE model config: initializer_range")
# model_g.add_arg("intermediate_size",        int, 512,            "CoKE model config: intermediate_size, default 512")

# model_g.add_arg("init_checkpoint",          str,  None,          "Init checkpoint to resume training from, or for prediction only")
# model_g.add_arg("init_pretraining_params",  str,  None,          "Init pre-training params which preforms fine-tuning from. If the "
#                  "arg 'init_checkpoint' has been set, this argument wouldn't be valid.")
# model_g.add_arg("checkpoints",              str,  "checkpoints", "Path to save checkpoints.")
# model_g.add_arg("weight_sharing",           bool, True,          "If set, share weights between word embedding and masked lm.")

# train_g = ArgumentGroup(parser, "training", "training options.")
# train_g.add_arg("epoch",             int,    100,                "Number of epoches for training.")
# train_g.add_arg("learning_rate",     float,  5e-5,               "Learning rate used to train with warmup.")
# train_g.add_arg("lr_scheduler",     str, "linear_warmup_decay",  "scheduler of learning rate.",
#                 choices=['linear_warmup_decay', 'noam_decay'])
# train_g.add_arg("soft_label",               float, 0.9,          "Value of soft labels for loss computation")
# train_g.add_arg("weight_decay",      float,  0.01,               "Weight decay rate for L2 regularizer.")
# train_g.add_arg("warmup_proportion", float,  0.1,                "Proportion of training steps to perform linear learning rate warmup for.")
# train_g.add_arg("use_ema",           bool,   True,               "Whether to use ema.")
# train_g.add_arg("ema_decay",         float,  0.9999,             "Decay rate for expoential moving average.")
# train_g.add_arg("use_fp16",          bool,   False,              "Whether to use fp16 mixed precision training.")
# train_g.add_arg("loss_scaling",      float,  1.0,                "Loss scaling factor for mixed precision training, only valid when use_fp16 is enabled.")

# log_g = ArgumentGroup(parser, "logging", "logging related.")
# log_g.add_arg("skip_steps",          int,    1000,               "The steps interval to print loss.")
# log_g.add_arg("verbose",             bool,   False,              "Whether to output verbose log.")

# data_g = ArgumentGroup(parser, "data", "Data paths, vocab paths and data processing options")
# data_g.add_arg("dataset",                   str,   "",    "dataset name")
# data_g.add_arg("train_file",                str,   None,  "Data for training.")
# data_g.add_arg("sen_candli_file",           str,   None,  "sentence_candicate_list file for path query evaluation. Only used for path query datasets")
# data_g.add_arg("sen_trivial_file",           str,   None,  "trivial sentence file for pathquery evaluation. Only used for path query datasets")
# data_g.add_arg("predict_file",              str,   None,  "Data for predictions.")
# data_g.add_arg("vocab_path",                str,   None,  "Path to vocabulary.")
# data_g.add_arg("true_triple_path",          str,   None,  "Path to all true triples. Only used for KBC evaluation.")
# data_g.add_arg("max_seq_len",               int,   3,     "Number of tokens of the longest sequence.")
# data_g.add_arg("batch_size",                int,   12,    "Total examples' number in batch for training. see also --in_tokens.")
# data_g.add_arg("in_tokens",                 bool,  False,
#                "If set, the batch size will be the maximum number of tokens in one batch. "
#                "Otherwise, it will be the maximum number of examples in one batch.")

# run_type_g = ArgumentGroup(parser, "run_type", "running type options.")
# run_type_g.add_arg("do_train",                     bool,   False,  "Whether to perform training.")
# run_type_g.add_arg("do_predict",                   bool,   False,  "Whether to perform prediction.")
# run_type_g.add_arg("use_cuda",                     bool,   True,   "If set, use GPU for training, default is True.")
# run_type_g.add_arg("use_fast_executor",            bool,   False,  "If set, use fast parallel executor (in experiment).")
# run_type_g.add_arg("num_iteration_per_drop_scope", int,    1,      "Ihe iteration intervals to clean up temporary variables.")

# args = parser.parse_args(args={})
```

```python colab={"base_uri": "https://localhost:8080/"} id="MSWD-fs9vqDR" outputId="a8a8908a-cfbd-454a-c164-983db362925d"
def create_model(pyreader_name, coke_config):
    pyreader = fluid.layers.py_reader\
            (
        capacity=50,
        shapes=[[-1, args.max_seq_len, 1],
                [-1, args.max_seq_len, 1],
                [-1, args.max_seq_len, 1], [-1, 1], [-1, 1]],
        dtypes=[
            'int64', 'int64', 'float32', 'int64', 'int64'],
        lod_levels=[0, 0, 0, 0, 0],
        name=pyreader_name,
        use_double_buffer=True)
    (src_ids, pos_ids, input_mask, mask_labels, mask_positions) = fluid.layers.read_file(pyreader)

    coke = CoKEModel(
        src_ids=src_ids,
        position_ids=pos_ids,
        input_mask=input_mask,
        config=coke_config,
        soft_label=args.soft_label,
        weight_sharing=args.weight_sharing,
        use_fp16=args.use_fp16)

    loss, fc_out = coke.get_pretraining_output(mask_label=mask_labels, mask_pos=mask_positions)
    if args.use_fp16 and args.loss_scaling > 1.0:
        loss = loss * args.loss_scaling

    batch_ones = fluid.layers.fill_constant_batch_size_like(
        input=mask_labels, dtype='int64', shape=[1], value=1)
    num_seqs = fluid.layers.reduce_sum(input=batch_ones)

    return pyreader, loss, fc_out, num_seqs


def pathquery_predict(test_exe, test_program, test_pyreader, fetch_list, all_examples,
                      sen_negli_dict, trivial_sen_set, eval_result_file):
    eval_i = 0
    step = 0
    batch_mqs = []
    batch_ranks = []
    test_pyreader.start()
    while True:
        try:
            np_fc_out = test_exe.run(fetch_list=fetch_list, program=test_program)[0]
            mqs, ranks = pathquery_batch_evaluation(eval_i, all_examples, np_fc_out,
                                                    sen_negli_dict, trivial_sen_set)
            batch_mqs.extend(mqs)
            batch_ranks.extend(ranks)
            step += 1
            if step % 10 == 0:
                logger.info("Processing pathquery_predict step:%d example: %d" % (step, eval_i))
            _batch_len = np_fc_out.shape[0]
            eval_i += _batch_len
        except fluid.core.EOFException:
            test_pyreader.reset()
            break

    eval_result = compute_pathquery_metrics(batch_mqs, batch_ranks, eval_result_file)
    return eval_result


def kbc_predict(test_exe, test_program, test_pyreader, fetch_list, all_examples, true_triplets_dict, eval_result_file):
    eval_i = 0
    step = 0
    batch_eval_rets = []
    f_batch_eval_rets = []
    test_pyreader.start()
    while True:
        try:
            batch_results = []
            np_fc_out = test_exe.run(fetch_list=fetch_list, program=test_program)[0]
            _batch_len = np_fc_out.shape[0]
            for idx in range(np_fc_out.shape[0]):
                logits = [float(x) for x in np_fc_out[idx].flat]
                batch_results.append(logits)
            rank, frank = kbc_batch_evaluation(eval_i, all_examples, batch_results, true_triplets_dict)
            batch_eval_rets.extend(rank)
            f_batch_eval_rets.extend(frank)
            if step % 10 == 0:
                logger.info("Processing kbc_predict step: %d exmaples:%d" % (step, eval_i))
            step += 1
            eval_i += _batch_len
        except fluid.core.EOFException:
            test_pyreader.reset()
            break
    eval_result = compute_kbc_metrics(batch_eval_rets, f_batch_eval_rets, eval_result_file)
    return eval_result


def predict(test_exe, test_program, test_pyreader, fetch_list, all_examples, args):
    dataset = args.dataset
    if not os.path.exists(args.checkpoints):
        os.makedirs(args.checkpoints)
    eval_result_file = os.path.join(args.checkpoints, "eval_result.json")
    logger.info(">> Evaluation result file: %s" % eval_result_file)

    if dataset.lower() in ["pathquerywn", "pathqueryfb"]:
        sen_candli_dict, trivial_sen_set = _load_pathquery_eval_dict(args.sen_candli_file,
                                                                   args.sen_trivial_file)
        logger.debug(">> Load sen_candli_dict size: %d" % len(sen_candli_dict))
        logger.debug(">> Trivial sen set size: %d" % len(trivial_sen_set))
        logger.debug(">> Finish load sen_candli set at:{}".format(time.ctime()))
        eval_performance = pathquery_predict(test_exe, test_program, test_pyreader, fetch_list,
                                              all_examples, sen_candli_dict, trivial_sen_set,
                                              eval_result_file)

        outs = "%s\t%.3f\t%.3f" % (args.dataset, eval_performance['mq'], eval_performance['fhits10'])
        logger.info("\n---------- Evaluation Performance --------------\n%s\n%s" %
                    ("\t".join(["TASK", "MQ", "Hits@10"]), outs))
    else:
        true_triplets_dict = _load_kbc_eval_dict(args.true_triple_path)
        logger.info(">> Finish loading true triplets dict %s" % time.ctime())
        eval_performance = kbc_predict(test_exe, test_program, test_pyreader, fetch_list,
                                        all_examples, true_triplets_dict, eval_result_file)
        outs = "%s\t%.3f\t%.3f\t%.3f\t%.3f" % (args.dataset,
                                               eval_performance['fmrr'],
                                               eval_performance['fhits1'],
                                               eval_performance['fhits3'],
                                               eval_performance['fhits10'])
        logger.info("\n----------- Evaluation Performance --------------\n%s\n%s" %
                    ("\t".join(["TASK", "MRR", "Hits@1", "Hits@3", "Hits@10"]), outs))
    return eval_performance


def _load_kbc_eval_dict(true_triple_file):
    def load_true_triples(true_triple_file):
        true_triples = []
        with open(true_triple_file, "r") as fr:
            for line in fr.readlines():
                tokens = line.strip("\r \n").split("\t")
                assert len(tokens) == 3
                true_triples.append(
                    (int(tokens[0]), int(tokens[1]), int(tokens[2])))
        logger.debug("Finish loading %d true triples" % len(true_triples))
        return true_triples
    true_triples = load_true_triples(true_triple_file)
    true_triples_dict = collections.defaultdict(lambda: {'hs': collections.defaultdict(list),
                                          'ts': collections.defaultdict(list)})
    for h, r, t in true_triples:
        true_triples_dict[r]['ts'][h].append(t)
        true_triples_dict[r]['hs'][t].append(h)
    return true_triples_dict


def _load_pathquery_eval_dict(sen_candli_file, trivial_sen_file, add_gold_o = True):
    sen_candli_dict = dict()
    for line in open(sen_candli_file):
        line = line.strip()
        segs = line.split("\t")
        assert len(segs) == 2, " Illegal format for sen_candli_dict, expects 2 columns data"
        sen = segs[0]
        candset = set(segs[1].split(" "))
        if add_gold_o is True:
            gold_o = sen.split(" ")[-1]
            candset.add(gold_o)
        _li = list(candset)
        int_li = [int(x) for x in _li]
        sen_candli_dict[sen] = int_li
    trivial_senset = {x.strip() for x in open(trivial_sen_file)}

    return sen_candli_dict, trivial_senset


def init_coke_net_config(args, print_config = True):
    config = dict()
    config["hidden_size"] = args.hidden_size
    config["num_hidden_layers"] = args.num_hidden_layers
    config["num_attention_heads"] = args.num_attention_heads
    config["vocab_size"] = args.vocab_size
    config["num_relations"] = args.num_relations
    config["max_position_embeddings"] = args.max_position_embeddings
    config["hidden_act"] = args.hidden_act
    config["hidden_dropout_prob"] = args.hidden_dropout_prob
    config["attention_probs_dropout_prob"] = args.attention_probs_dropout_prob
    config["initializer_range"] = args.initializer_range
    config["intermediate_size"] = args.intermediate_size

    if print_config is True:
        logger.info('----------- CoKE Network Configuration -------------')
        for arg, value in config.items():
            logger.info('%s: %s' % (arg, value))
        logger.info('------------------------------------------------')
    return config


def main(args):
    if not (args.do_train or args.do_predict):
        raise ValueError("For args `do_train` and `do_predict`, at "
                         "least one of them must be True.")
    if args.use_cuda:
        place = fluid.CUDAPlace(0)
        dev_count = fluid.core.get_cuda_device_count()
    else:
        place = fluid.CPUPlace()
        dev_count = int(os.environ.get('CPU_NUM', multiprocessing.cpu_count()))
    exe = fluid.Executor(place)

    startup_prog = fluid.Program()

    # Init programs
    coke_config = init_coke_net_config(args, print_config=True)
    if args.do_train:
        train_data_reader = get_data_reader(args, args.train_file, is_training=True,
                                            epoch=args.epoch, shuffle=True, dev_count=dev_count,
                                            vocab_size=args.vocab_size)

        num_train_examples = train_data_reader.total_instance
        if args.in_tokens:
            max_train_steps = args.epoch * num_train_examples // (
                    args.batch_size // args.max_seq_len) // dev_count
        else:
            max_train_steps = args.epoch * num_train_examples // (
                args.batch_size) // dev_count
        warmup_steps = int(max_train_steps * args.warmup_proportion)
        logger.info("Device count: %d" % dev_count)
        logger.info("Num train examples: %d" % num_train_examples)
        logger.info("Max train steps: %d" % max_train_steps)
        logger.info("Num warmup steps: %d" % warmup_steps)

        train_program = fluid.Program()

        # Create model and set optimization for train
        with fluid.program_guard(train_program, startup_prog):
            with fluid.unique_name.guard():
                train_pyreader, loss, _, num_seqs = create_model(
                    pyreader_name='train_reader',
                    coke_config=coke_config)

                scheduled_lr = optimization(
                    loss=loss,
                    warmup_steps=warmup_steps,
                    num_train_steps=max_train_steps,
                    learning_rate=args.learning_rate,
                    train_program=train_program,
                    startup_prog=startup_prog,
                    weight_decay=args.weight_decay,
                    scheduler=args.lr_scheduler,
                    use_fp16=args.use_fp16,
                    loss_scaling=args.loss_scaling)

                if args.use_ema:
                    ema = fluid.optimizer.ExponentialMovingAverage(args.ema_decay)
                    ema.update()

                fluid.memory_optimize(train_program, skip_opt_set=[loss.name, num_seqs.name])

        if args.verbose:
            if args.in_tokens:
                lower_mem, upper_mem, unit = fluid.contrib.memory_usage(
                    program=train_program,
                    batch_size=args.batch_size // args.max_seq_len)
            else:
                lower_mem, upper_mem, unit = fluid.contrib.memory_usage(
                    program=train_program, batch_size=args.batch_size)
            logger.info("Theoretical memory usage in training:  %.3f - %.3f %s" %
                        (lower_mem, upper_mem, unit))

    if args.do_predict:
        # Create model for prediction
        test_prog = fluid.Program()
        with fluid.program_guard(test_prog, startup_prog):
            with fluid.unique_name.guard():
                test_pyreader, _, fc_out, num_seqs = create_model(
                    pyreader_name='test_reader',
                    coke_config=coke_config)

                if args.use_ema and 'ema' not in dir():
                    ema = fluid.optimizer.ExponentialMovingAverage(args.ema_decay)

                fluid.memory_optimize(test_prog, skip_opt_set=[fc_out.name, num_seqs.name])

        test_prog = test_prog.clone(for_test=True)

    exe.run(startup_prog)

    # Init checkpoints
    if args.do_train:
        init_train_checkpoint(args, exe, startup_prog)
    elif args.do_predict:
        init_predict_checkpoint(args, exe, startup_prog)

    # Run training
    if args.do_train:
        exec_strategy = fluid.ExecutionStrategy()
        exec_strategy.use_experimental_executor = args.use_fast_executor
        exec_strategy.num_threads = dev_count
        exec_strategy.num_iteration_per_drop_scope = args.num_iteration_per_drop_scope

        train_exe = fluid.ParallelExecutor(
            use_cuda=args.use_cuda,
            loss_name=loss.name,
            exec_strategy=exec_strategy,
            main_program=train_program)

        train_pyreader.decorate_tensor_provider(train_data_reader.data_generator())

        train_pyreader.start()
        steps = 0
        total_cost, total_num_seqs = [], []
        time_begin = time.time()
        while steps < max_train_steps:
            try:
                steps += 1
                if steps % args.skip_steps == 0:
                    if warmup_steps <= 0:
                        fetch_list = [loss.name, num_seqs.name]
                    else:
                        fetch_list = [
                            loss.name, scheduled_lr.name, num_seqs.name
                        ]
                else:
                    fetch_list = []

                outputs = train_exe.run(fetch_list=fetch_list)

                if steps % args.skip_steps == 0:
                    if warmup_steps <= 0:
                        np_loss, np_num_seqs = outputs
                    else:
                        np_loss, np_lr, np_num_seqs = outputs
                    total_cost.extend(np_loss * np_num_seqs)
                    total_num_seqs.extend(np_num_seqs)

                    if args.verbose:
                        verbose = "train pyreader queue size: %d, " % train_pyreader.queue.size(
                        )
                        verbose += "learning rate: %f" % (
                            np_lr[0]
                            if warmup_steps > 0 else args.learning_rate)
                        logger.info(verbose)

                    time_end = time.time()
                    used_time = time_end - time_begin
                    current_example, epoch = train_data_reader.get_progress()

                    logger.info("epoch: %d, progress: %d/%d, step: %d, loss: %f, "
                                "speed: %f steps/s" %
                                (epoch, current_example, num_train_examples, steps,
                                 np.sum(total_cost) / np.sum(total_num_seqs),
                                 args.skip_steps / used_time))
                    total_cost, total_num_seqs = [], []
                    time_begin = time.time()

                if steps == max_train_steps:
                    save_path = os.path.join(args.checkpoints, "step_" + str(steps))
                    fluid.io.save_persistables(exe, save_path, train_program)
            except fluid.core.EOFException:
                logger.warning(">> EOFException")
                save_path = os.path.join(args.checkpoints, "step_" + str(steps) + "_final")
                fluid.io.save_persistables(exe, save_path, train_program)
                train_pyreader.reset()
                break
        logger.info(">>Finish training at %s " % time.ctime())

    # Run prediction
    if args.do_predict:
        assert dev_count == 1, "During prediction, dev_count expects 1, current is %d" % dev_count
        test_data_reader = get_data_reader(args, args.predict_file, is_training=False,
                                           epoch=1, shuffle=False, dev_count=dev_count,
                                           vocab_size=args.vocab_size)
        test_pyreader.decorate_tensor_provider(test_data_reader.data_generator())

        if args.use_ema:
            with ema.apply(exe):
                eval_performance = predict(exe, test_prog, test_pyreader,
                                           [fc_out.name], test_data_reader.examples, args)
        else:
            eval_performance = predict(exe, test_prog, test_pyreader,
                                       [fc_out.name], test_data_reader.examples, args)

        logger.info(">>Finish predicting at %s " % time.ctime())


def init_predict_checkpoint(args, exe, startup_prog):
    if args.dataset in ["pathQueryWN", "pathQueryFB"]:
        assert args.sen_candli_file is not None and args.sen_trivial_file is not None, "during test, pathQuery sen_candli_file and path_trivial_file must be set "
    if not args.init_checkpoint:
        raise ValueError("args 'init_checkpoint' should be set if"
                         "only doing prediction!")
    init_checkpoint(
        exe,
        args.init_checkpoint,
        main_program=startup_prog,
        use_fp16=args.use_fp16)


def init_train_checkpoint(args, exe, startup_prog):
    if args.init_checkpoint and args.init_pretraining_params:
        logger.info(
            "WARNING: args 'init_checkpoint' and 'init_pretraining_params' "
            "both are set! Only arg 'init_checkpoint' is made valid.")
    if args.init_checkpoint:
        init_checkpoint(
            exe,
            args.init_checkpoint,
            main_program=startup_prog,
            use_fp16=args.use_fp16,
            print_var_verbose=False)
    elif args.init_pretraining_params:
        init_pretraining_params(
            exe,
            args.init_pretraining_params,
            main_program=startup_prog,
            use_fp16=args.use_fp16)


def get_data_reader(args, data_file, epoch, is_training, shuffle, dev_count, vocab_size):
    if args.dataset.lower() in ["pathqueryfb", "pathquerywn"]:
        Reader = PathqueryDataReader
    else:
        Reader = KBCDataReader
    data_reader = Reader(
        vocab_path=args.vocab_path,
        data_path=data_file,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        is_training=is_training,
        shuffle=shuffle,
        dev_count=dev_count,
        epoch=epoch,
        vocab_size=vocab_size)
    return data_reader


if __name__ == '__main__':
    args.do_train=True
    print_arguments(args)
    main(args)
```
