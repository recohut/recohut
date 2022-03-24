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

<!-- #region id="LOAjWAu1zj56" -->
# MB-GMN on BeiBei
<!-- #endregion -->

<!-- #region id="bpPKhl0fzj2G" -->
## Executive summary
| | |
| --- | --- |
| Problem | Modern recommender systems often embed users and items into low-dimensional latent representations, based on their observed interactions. In practical recommendation scenarios, users often exhibit various intents which drive them to interact with items with multiple behavior types (e.g., click, tag-as-favorite, purchase). However, the diversity of user behaviors is ignored in most of the existing approaches, which makes them difficult to capture heterogeneous relational structures across different types of interactive behaviors. |
| Prblm Stmt. | We define a three-way tensor $X \in \mathbb{R}^{𝐼×𝐽×𝐾}$ to reflect the multi-typed interaction (e.g., click, tag-as-favorite, purchase) between user ($𝑢_𝑖 \in U$) and item ($𝑣_𝑗 \in V$). Here, 𝐾 denotes the number of behavior types. Specifically, the individual element $𝑥_{i,j}^k \in X$ is set as 1 if user $𝑢_𝑖$ interacts with item $𝑣_𝑗$ under the 𝑘-th behavior type, and $𝑘_{𝑖,𝑗}$ = 0 otherwise. In the multi-behavior recommendation scenario, one type of user-item interaction will be treated as target behavior (e.g., purchase). Other behaviors are referred to as context behaviors (e.g., click, tag-as-favorite, add-to-cart) for providing insightful knowledge in assisting the target behavior prediction. Based on the aforementioned definitions, we formally state our studied problem as: • Input: the observed multi-behavior interaction tensor $X \in \mathbb{R}^{𝐼×𝐽×𝐾}$ between users U and items V across 𝐾 behavior types. • Output: the predictive function which estimates the likelihood of user $𝑢_𝑖$ adopts the item $𝑣_𝑗$ with the target behavior type. |
| Solution | Multi-Behavior recommendation framework with Graph Meta Network (MB-GMN) incorporate the multi-behavior pattern modeling into a meta-learning paradigm. Our developed MB-GMN empowers the user-item interaction learning with the capability of uncovering type-dependent behavior representations, which automatically distills the behavior heterogeneity and interaction diversity for recommendations. MB-GMN is composed of two key components: i) multi-behavior pattern encoding, a meta-knowledge learner that captures the personalized multi-behavior characteristics; ii) cross-type behavior dependency modeling, a transfer learning paradigm which learns a well-customized prediction network by transferring knowledge across different behavior types. |
| Dataset | BeiBei. |
| Preprocessing | Leave-one-out evaluation is leveraged for training and test set partition. We generate the test data set by including the last interactive item and consider the rest of data for training. For efficient and fair model evaluation, we pair each positive item instance with 99 randomly sampled non-interactive items for each user. We regard users’ purchases as the target behaviors. |
| Metrics | NDCG, HR |
| Models | MB-GMN |
| Cluster | Python 3.6+, Tensorflow 1.x |
| Tags | `MetaLearning`, `GNN` |
| Credits | akaxlh |
<!-- #endregion -->

<!-- #region id="B3NwTUYRzjyz" -->
## Process flow

![](https://github.com/RecoHut-Stanzas/S346877/raw/main/images/process_flow.svg)
<!-- #endregion -->

<!-- #region id="UiUls4wo5IBp" -->
## Model architecture

![](https://github.com/RecoHut-Stanzas/S346877/raw/main/images/mbgmn_model.png)
<!-- #endregion -->

<!-- #region id="i35fbAIe5P2d" -->
*The model architecture of MB-GMN. (a) Multi-behavior pattern modeling with meta-knowledge learner for behavior heterogeneity; (b) Meta graph neural network which preserves the behavior semantics with high-order connectivity; (c) Metaknowledge transfer networks that customize the parameter of prediction layer to capture cross-type behavior dependency.*
<!-- #endregion -->

<!-- #region id="uZ5MRgBc0v49" -->
## Setup
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="A3mT2WvpnDfP" executionInfo={"status": "ok", "timestamp": 1639298230167, "user_tz": -330, "elapsed": 541, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="2819ceb9-ae56-40fe-e732-f7879df70c5e"
%tensorflow_version 1.x
```

```python id="_sPDAiS8nNN_"
import os
import numpy as np
import tensorflow as tf
from tensorflow.core.protobuf import config_pb2
import pickle
import argparse
import random
import gc
import datetime
from tensorflow.contrib.layers import xavier_initializer
from scipy.sparse import csr_matrix
import scipy.sparse as sp
from tqdm.notebook import tqdm
```

```python id="kIIytIU8s3Jq"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
```

```python id="vSKHNr0Atb8S"
os.makedirs('./History', exist_ok=True)
```

```python id="92B7jEwLtA0j"
def parse_args():
	parser = argparse.ArgumentParser(description='Model Params')
	parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
	parser.add_argument('--batch', default=256, type=int, help='batch size')
	parser.add_argument('--reg', default=1e-2, type=float, help='weight decay regularizer')
	parser.add_argument('--epoch', default=10, type=int, help='number of epochs')
	parser.add_argument('--decay', default=0.96, type=float, help='weight decay rate')
	parser.add_argument('--save_path', default='tem', help='file name to save model and training record')
	parser.add_argument('--latdim', default=32, type=int, help='embedding size')
	parser.add_argument('--rank', default=4, type=int, help='embedding size')
	parser.add_argument('--memosize', default=2, type=int, help='memory size')
	parser.add_argument('--sampNum', default=40, type=int, help='batch size for sampling')
	parser.add_argument('--att_head', default=2, type=int, help='number of attention heads')
	parser.add_argument('--gnn_layer', default=2, type=int, help='number of gnn layers')
	parser.add_argument('--trnNum', default=10000, type=int, help='number of training instances per epoch')
	parser.add_argument('--load_model', default=None, help='model name to load')
	parser.add_argument('--shoot', default=10, type=int, help='K of top k')
	parser.add_argument('--data', default='beibei', type=str, help='name of dataset')
	parser.add_argument('--target', default='buy', type=str, help='target behavior to predict on')
	parser.add_argument('--deep_layer', default=0, type=int, help='number of deep layers to make the final prediction')
	parser.add_argument('--mult', default=100, type=float, help='multiplier for the result')
	parser.add_argument('--keepRate', default=0.7, type=float, help='rate for dropout')
	parser.add_argument('--slot', default=5, type=float, help='length of time slots')
	parser.add_argument('--graphSampleN', default=15000, type=int, help='use 25000 for training and 200000 for testing, empirically')
	parser.add_argument('--divSize', default=10000, type=int, help='div size for smallTestEpoch')
	parser.add_argument('--tstEpoch', default=3, type=int, help='number of epoch to test while training')
	parser.add_argument('--subUsrSize', default=10, type=int, help='number of item for each sub-user')
	parser.add_argument('--subUsrDcy', default=0.9, type=float, help='decay factor for sub-users over time')
	return parser.parse_args([])
 
args = parse_args()

args.user = 21716
args.item = 7977
args.decay_step = args.trnNum//args.batch
```

<!-- #region id="wVCw2Kc31DdY" -->
## Utils
<!-- #endregion -->

```python id="Z80jSTrUtr8g"
logmsg = ''
timemark = dict()
saveDefault = False

def log(msg, save=None, oneline=False):
	global logmsg
	global saveDefault
	time = datetime.datetime.now()
	tem = '%s: %s' % (time, msg)
	if save != None:
		if save:
			logmsg += tem + '\n'
	elif saveDefault:
		logmsg += tem + '\n'
	if oneline:
		print(tem, end='\r')
	else:
		print(tem)

def marktime(marker):
	global timemark
	timemark[marker] = datetime.datetime.now()

def SpentTime(marker):
	global timemark
	if marker not in timemark:
		msg = 'LOGGER ERROR, marker', marker, ' not found'
		tem = '%s: %s' % (time, msg)
		print(tem)
		return False
	return datetime.datetime.now() - timemark[marker]

def SpentTooLong(marker, day=0, hour=0, minute=0, second=0):
	global timemark
	if marker not in timemark:
		msg = 'LOGGER ERROR, marker', marker, ' not found'
		tem = '%s: %s' % (time, msg)
		print(tem)
		return False
	return datetime.datetime.now() - timemark[marker] >= datetime.timedelta(days=day, hours=hour, minutes=minute, seconds=second)
```

```python id="0BvkMCQytcGV"
def RandomShuffle(infile, outfile, deleteSchema=False):
	with open(infile, 'r', encoding='utf-8') as fs:
		arr = fs.readlines()
	if not arr[-1].endswith('\n'):
		arr[-1] += '\n'
	if deleteSchema:
		arr = arr[1:]
	random.shuffle(arr)
	with open(outfile, 'w', encoding='utf-8') as fs:
		for line in arr:
			fs.write(line)
	del arr

def WriteToBuff(buff, line, out):
	BUFF_SIZE = 1000000
	buff.append(line)
	if len(buff) == BUFF_SIZE:
		WriteToDisk(buff, out)

def WriteToDisk(buff, out):
	with open(out, 'a', encoding='utf-8') as fs:
		for line in buff:
			fs.write(line)
	buff.clear()


def SubDataSet(infile, outfile1, outfile2, rate):
	out1 = list()
	out2 = list()
	with open(infile, 'r', encoding='utf-8') as fs:
		for line in fs:
			if random.random() < rate:
				WriteToBuff(out1, line, outfile1)
			else:
				WriteToBuff(out2, line, outfile2)
	WriteToDisk(out1, outfile1)
	WriteToDisk(out2, outfile2)

def CombineFiles(files, out):
	buff = list()
	for file in files:
		with open(file, 'r') as fs:
			for line in fs:
				WriteToBuff(buff, line, out)
	WriteToDisk(buff, out)
```

<!-- #region id="X5d4EeIU1ARg" -->
## NN Layers
<!-- #endregion -->

```python id="C5z3c78ctcD0"
paramId = 0
biasDefault = False
params = {}
regParams = {}
ita = 0.2
leaky = 0.1

def getParamId():
	global paramId
	paramId += 1
	return paramId

def setIta(ITA):
	ita = ITA

def setBiasDefault(val):
	global biasDefault
	biasDefault = val

def getParam(name):
	return params[name]

def addReg(name, param):
	global regParams
	if name not in regParams:
		regParams[name] = param
	else:
		print('ERROR: Parameter already exists')

def addParam(name, param):
	global params
	if name not in params:
		params[name] = param

def defineRandomNameParam(shape, dtype=tf.float32, reg=False, initializer='xavier', trainable=True):
	name = 'defaultParamName%d'%getParamId()
	return defineParam(name, shape, dtype, reg, initializer, trainable)

def defineParam(name, shape, dtype=tf.float32, reg=False, initializer='xavier', trainable=True):
	global params
	global regParams
	assert name not in params, 'name %s already exists' % name
	if initializer == 'xavier':
		ret = tf.get_variable(name=name, dtype=dtype, shape=shape,
			initializer=xavier_initializer(dtype=tf.float32),
			trainable=trainable)
	elif initializer == 'trunc_normal':
		ret = tf.get_variable(name=name, initializer=tf.random.truncated_normal(shape=[int(shape[0]), shape[1]], mean=0.0, stddev=0.03, dtype=dtype))
	elif initializer == 'zeros':
		ret = tf.get_variable(name=name, dtype=dtype,
			initializer=tf.zeros(shape=shape, dtype=tf.float32),
			trainable=trainable)
	elif initializer == 'ones':
		ret = tf.get_variable(name=name, dtype=dtype, initializer=tf.ones(shape=shape, dtype=tf.float32), trainable=trainable)
	elif not isinstance(initializer, str):
		ret = tf.get_variable(name=name, dtype=dtype,
			initializer=initializer, trainable=trainable)
	else:
		print('ERROR: Unrecognized initializer')
		exit()
	params[name] = ret
	if reg:
		regParams[name] = ret
	return ret

def getOrDefineParam(name, shape, dtype=tf.float32, reg=False, initializer='xavier', trainable=True, reuse=False):
	global params
	global regParams
	if name in params:
		assert reuse, 'Reusing Param %s Not Specified' % name
		if reg and name not in regParams:
			regParams[name] = params[name]
		return params[name]
	return defineParam(name, shape, dtype, reg, initializer, trainable)

def BN(inp, name=None):
	global ita
	dim = inp.get_shape()[1]
	name = 'defaultParamName%d'%getParamId()
	scale = tf.Variable(tf.ones([dim]))
	shift = tf.Variable(tf.zeros([dim]))
	fcMean, fcVar = tf.nn.moments(inp, axes=[0])
	ema = tf.train.ExponentialMovingAverage(decay=0.5)
	emaApplyOp = ema.apply([fcMean, fcVar])
	with tf.control_dependencies([emaApplyOp]):
		mean = tf.identity(fcMean)
		var = tf.identity(fcVar)
	ret = tf.nn.batch_normalization(inp, mean, var, shift,
		scale, 1e-8)
	return ret

def FC(inp, outDim, name=None, useBias=False, activation=None, reg=False, useBN=False, dropout=None, initializer='xavier', reuse=False, biasReg=False, biasInitializer='zeros'):
	global params
	global regParams
	global leaky
	inDim = inp.get_shape()[1]
	temName = name if name!=None else 'defaultParamName%d'%getParamId()
	W = getOrDefineParam(temName, [inDim, outDim], reg=reg, initializer=initializer, reuse=reuse)
	if dropout != None:
		ret = tf.nn.dropout(inp, rate=dropout) @ W
	else:
		ret = inp @ W
	if useBias:
		ret = Bias(ret, name=name, reuse=reuse, reg=biasReg, initializer=biasInitializer)
	if useBN:
		ret = BN(ret)
	if activation != None:
		ret = Activate(ret, activation)
	return ret

def Bias(data, name=None, reg=False, reuse=False, initializer='zeros'):
	inDim = data.get_shape()[-1]
	temName = name if name!=None else 'defaultParamName%d'%getParamId()
	temBiasName = temName + 'Bias'
	bias = getOrDefineParam(temBiasName, inDim, reg=False, initializer=initializer, reuse=reuse)
	if reg:
		regParams[temBiasName] = bias
	return data + bias

def ActivateHelp(data, method):
	if method == 'relu':
		ret = tf.nn.relu(data)
	elif method == 'sigmoid':
		ret = tf.nn.sigmoid(data)
	elif method == 'tanh':
		ret = tf.nn.tanh(data)
	elif method == 'softmax':
		ret = tf.nn.softmax(data, axis=-1)
	elif method == 'leakyRelu':
		ret = tf.maximum(leaky*data, data)
	elif method == 'twoWayLeakyRelu6':
		temMask = tf.to_float(tf.greater(data, 6.0))
		ret = temMask * (6 + leaky * (data - 6)) + (1 - temMask) * tf.maximum(leaky * data, data)
	elif method == '-1relu':
		ret = tf.maximum(-1.0, data)
	elif method == 'relu6':
		ret = tf.maximum(0.0, tf.minimum(6.0, data))
	elif method == 'relu3':
		ret = tf.maximum(0.0, tf.minimum(3.0, data))
	else:
		raise Exception('Error Activation Function')
	return ret

def Activate(data, method, useBN=False):
	global leaky
	if useBN:
		ret = BN(data)
	else:
		ret = data
	ret = ActivateHelp(ret, method)
	return ret

def Regularize(names=None, method='L2'):
	ret = 0
	if method == 'L1':
		if names != None:
			for name in names:
				ret += tf.reduce_sum(tf.abs(getParam(name)))
		else:
			for name in regParams:
				ret += tf.reduce_sum(tf.abs(regParams[name]))
	elif method == 'L2':
		if names != None:
			for name in names:
				ret += tf.reduce_sum(tf.square(getParam(name)))
		else:
			for name in regParams:
				ret += tf.reduce_sum(tf.square(regParams[name]))
	return ret

def Dropout(data, rate):
	if rate == None:
		return data
	else:
		return tf.nn.dropout(data, rate=rate)

def selfAttention(localReps, number, inpDim, numHeads):
	Q = defineRandomNameParam([inpDim, inpDim], reg=True)
	K = defineRandomNameParam([inpDim, inpDim], reg=True)
	V = defineRandomNameParam([inpDim, inpDim], reg=True)
	rspReps = tf.reshape(tf.stack(localReps, axis=1), [-1, inpDim])
	q = tf.reshape(rspReps @ Q, [-1, number, 1, numHeads, inpDim//numHeads])
	k = tf.reshape(rspReps @ K, [-1, 1, number, numHeads, inpDim//numHeads])
	v = tf.reshape(rspReps @ V, [-1, 1, number, numHeads, inpDim//numHeads])
	att = tf.nn.softmax(tf.reduce_sum(q * k, axis=-1, keepdims=True) / tf.sqrt(inpDim/numHeads), axis=2)
	attval = tf.reshape(tf.reduce_sum(att * v, axis=2), [-1, number, inpDim])
	rets = [None] * number
	paramId = 'dfltP%d' % getParamId()
	for i in range(number):
		tem1 = tf.reshape(tf.slice(attval, [0, i, 0], [-1, 1, -1]), [-1, inpDim])
		# tem2 = FC(tem1, inpDim, useBias=True, name=paramId+'_1', reg=True, activation='relu', reuse=True) + localReps[i]
		rets[i] = tem1 + localReps[i]
	return rets

def lightSelfAttention(localReps, number, inpDim, numHeads):
	Q = defineRandomNameParam([inpDim, inpDim], reg=True)
	rspReps = tf.reshape(tf.stack(localReps, axis=1), [-1, inpDim])
	tem = rspReps @ Q
	q = tf.reshape(tem, [-1, number, 1, numHeads, inpDim//numHeads])
	k = tf.reshape(tem, [-1, 1, number, numHeads, inpDim//numHeads])
	v = tf.reshape(rspReps, [-1, 1, number, numHeads, inpDim//numHeads])
	# att = tf.nn.softmax(tf.reduce_sum(q * k, axis=-1, keepdims=True) * tf.sqrt(inpDim/numHeads), axis=2)
	att = tf.nn.softmax(tf.reduce_sum(q * k, axis=-1, keepdims=True) / tf.sqrt(inpDim/numHeads), axis=2)
	attval = tf.reshape(tf.reduce_sum(att * v, axis=2), [-1, number, inpDim])
	rets = [None] * number
	paramId = 'dfltP%d' % getParamId()
	for i in range(number):
		tem1 = tf.reshape(tf.slice(attval, [0, i, 0], [-1, 1, -1]), [-1, inpDim])
		# tem2 = FC(tem1, inpDim, useBias=True, name=paramId+'_1', reg=True, activation='relu', reuse=True) + localReps[i]
		rets[i] = tem1 + localReps[i]
	return rets#, tf.squeeze(att)
```

<!-- #region id="ZBlNyUpR08Qr" -->
## Dataset
<!-- #endregion -->

```python id="LiC15-XSnFCC"
!git clone --branch v1 https://github.com/RecoHut-Datasets/beibei.git
```

```python id="WJf_ck4htcBW"
def transpose(mat):
	coomat = sp.coo_matrix(mat)
	return csr_matrix(coomat.transpose())

def negSamp(temLabel, sampSize, nodeNum):
	negset = [None] * sampSize
	cur = 0
	while cur < sampSize:
		rdmItm = np.random.choice(nodeNum)
		if temLabel[rdmItm] == 0:
			negset[cur] = rdmItm
			cur += 1
	return negset

def transToLsts(mat, mask=False, norm=False):
	shape = [mat.shape[0], mat.shape[1]]
	coomat = sp.coo_matrix(mat)
	indices = np.array(list(map(list, zip(coomat.row, coomat.col))), dtype=np.int32)
	data = coomat.data.astype(np.float32)

	if norm:
		rowD = np.squeeze(np.array(1 / (np.sqrt(np.sum(mat, axis=1) + 1e-8) + 1e-8)))
		colD = np.squeeze(np.array(1 / (np.sqrt(np.sum(mat, axis=0) + 1e-8) + 1e-8)))
		for i in range(len(data)):
			row = indices[i, 0]
			col = indices[i, 1]
			data[i] = data[i] * rowD[row] * colD[col]

	# half mask
	if mask:
		spMask = (np.random.uniform(size=data.shape) > 0.5) * 1.0
		data = data * spMask

	if indices.shape[0] == 0:
		indices = np.array([[0, 0]], dtype=np.int32)
		data = np.array([0.0], np.float32)
	return indices, data, shape

class DataHandler:
    def __init__(self):
        predir = './beibei/'
        behs = ['pv', 'cart', 'buy']
        self.predir = predir
        self.behs = behs
        self.trnfile = predir + 'trn_'
        self.tstfile = predir + 'tst_'

    def LoadData(self):
        trnMats = list()
        for i in range(len(self.behs)):
            beh = self.behs[i]
            path = self.trnfile + beh
            with open(path, 'rb') as fs:
                mat = (pickle.load(fs) != 0).astype(np.float32)
            trnMats.append(mat)
        # test set
        path = self.tstfile + 'int'
        with open(path, 'rb') as fs:
            tstInt = np.array(pickle.load(fs))
        tstStat = (tstInt != None)
        tstUsrs = np.reshape(np.argwhere(tstStat != False), [-1])
        self.trnMats = trnMats
        self.tstInt = tstInt
        self.tstUsrs = tstUsrs
        args.user, args.item = self.trnMats[0].shape
        args.behNum = len(self.behs)
        self.prepareGlobalData()

    def prepareGlobalData(self):
        adj = 0
        for i in range(args.behNum):
            adj = adj + self.trnMats[i]
        adj = (adj != 0).astype(np.float32)
        self.labelP = np.squeeze(np.array(np.sum(adj, axis=0)))
        tpadj = transpose(adj)
        adjNorm = np.reshape(np.array(np.sum(adj, axis=1)), [-1])
        tpadjNorm = np.reshape(np.array(np.sum(tpadj, axis=1)), [-1])
        for i in range(adj.shape[0]):
            for j in range(adj.indptr[i], adj.indptr[i+1]):
                adj.data[j] /= adjNorm[i]
        for i in range(tpadj.shape[0]):
            for j in range(tpadj.indptr[i], tpadj.indptr[i+1]):
                tpadj.data[j] /= tpadjNorm[i]
        self.adj = adj
        self.tpadj = tpadj

    def sampleLargeGraph(self, pckUsrs, pckItms=None, sampDepth=2, sampNum=args.graphSampleN, preSamp=False):
        adj = self.adj
        tpadj = self.tpadj
        def makeMask(nodes, size):
            mask = np.ones(size)
            if not nodes is None:
                mask[nodes] = 0.0
            return mask

        def updateBdgt(adj, nodes):
            if nodes is None:
                return 0
            tembat = 1000
            ret = 0
            for i in range(int(np.ceil(len(nodes) / tembat))):
                st = tembat * i
                ed = min((i+1) * tembat, len(nodes))
                temNodes = nodes[st: ed]
                ret += np.sum(adj[temNodes], axis=0)
            return ret

        def sample(budget, mask, sampNum):
            score = (mask * np.reshape(np.array(budget), [-1])) ** 2
            norm = np.sum(score)
            if norm == 0:
                return np.random.choice(len(score), 1), sampNum - 1
            score = list(score / norm)
            arrScore = np.array(score)
            posNum = np.sum(np.array(score)!=0)
            if posNum < sampNum:
                pckNodes1 = np.squeeze(np.argwhere(arrScore!=0))
                # pckNodes2 = np.random.choice(np.squeeze(np.argwhere(arrScore==0.0)), min(len(score) - posNum, sampNum - posNum), replace=False)
                # pckNodes = np.concatenate([pckNodes1, pckNodes2], axis=0)
                pckNodes = pckNodes1
            else:
                pckNodes = np.random.choice(len(score), sampNum, p=score, replace=False)
            return pckNodes, max(sampNum - posNum, 0)

        def constructData(usrs, itms):
            adjs = self.trnMats
            pckAdjs = []
            pckTpAdjs = []
            for i in range(len(adjs)):
                pckU = adjs[i][usrs]
                tpPckI = transpose(pckU)[itms]
                pckTpAdjs.append(tpPckI)
                pckAdjs.append(transpose(tpPckI))
            return pckAdjs, pckTpAdjs, usrs, itms

        usrMask = makeMask(pckUsrs, adj.shape[0])
        itmMask = makeMask(pckItms, adj.shape[1])
        itmBdgt = updateBdgt(adj, pckUsrs)
        if pckItms is None:
            pckItms, _ = sample(itmBdgt, itmMask, len(pckUsrs))
            itmMask = itmMask * makeMask(pckItms, adj.shape[1])
        usrBdgt = updateBdgt(tpadj, pckItms)
        uSampRes = 0
        iSampRes = 0
        for i in range(sampDepth + 1):
            uSamp = uSampRes + (sampNum if i < sampDepth else 0)
            iSamp = iSampRes + (sampNum if i < sampDepth else 0)
            newUsrs, uSampRes = sample(usrBdgt, usrMask, uSamp)
            usrMask = usrMask * makeMask(newUsrs, adj.shape[0])
            newItms, iSampRes = sample(itmBdgt, itmMask, iSamp)
            itmMask = itmMask * makeMask(newItms, adj.shape[1])
            if i == sampDepth or i == sampDepth and uSampRes == 0 and iSampRes == 0:
                break
            usrBdgt += updateBdgt(tpadj, newItms)
            itmBdgt += updateBdgt(adj, newUsrs)
        usrs = np.reshape(np.argwhere(usrMask==0), [-1])
        itms = np.reshape(np.argwhere(itmMask==0), [-1])
        return constructData(usrs, itms)
```

<!-- #region id="cxWiPevg2o4x" -->
## Model
<!-- #endregion -->

```python id="TNDAiXW0tb-l"
class Recommender:
	def __init__(self, sess, handler):
		self.sess = sess
		self.handler = handler

		print('USER', args.user, 'ITEM', args.item)
		self.metrics = dict()
		mets = ['Loss', 'preLoss', 'HR', 'NDCG']
		for met in mets:
			self.metrics['Train' + met] = list()
			self.metrics['Test' + met] = list()

	def makePrint(self, name, ep, reses, save):
		ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
		for metric in reses:
			val = reses[metric]
			ret += '%s = %.4f, ' % (metric, val)
			tem = name + metric
			if save and tem in self.metrics:
				self.metrics[tem].append(val)
		ret = ret[:-2] + '  '
		return ret

	def run(self):
		self.prepareModel()
		log('Model Prepared')
		if args.load_model != None:
			self.loadModel()
			stloc = len(self.metrics['TrainLoss']) * args.tstEpoch - (args.tstEpoch - 1)
		else:
			stloc = 0
			init = tf.global_variables_initializer()
			self.sess.run(init)
			log('Variables Initiated')
		for ep in tqdm(range(stloc, args.epoch)):
			test = (ep % args.tstEpoch == 0)
			reses = self.trainEpoch()
			log(self.makePrint('Train', ep, reses, test))
			if test:
				reses = self.testEpoch()
				log(self.makePrint('Test', ep, reses, test))
			if ep % args.tstEpoch == 0:
				self.saveHistory()
			print()
		reses = self.testEpoch()
		log(self.makePrint('Test', args.epoch, reses, True))
		self.saveHistory()

	def messagePropagate(self, lats, adj, lats2):
		return Activate(tf.sparse.sparse_dense_matmul(adj, lats), self.actFunc)

	def metaForSpecialize(self, uEmbed, iEmbed, behEmbed, adjs, tpAdjs):
		latdim = args.latdim // 2
		rank = args.rank
		assert len(adjs) == len(tpAdjs)
		uNeighbor = iNeighbor = 0
		for i in range(len(adjs)):
			uNeighbor += tf.sparse.sparse_dense_matmul(adjs[i], iEmbed)
			iNeighbor += tf.sparse.sparse_dense_matmul(tpAdjs[i], uEmbed)
		ubehEmbed = tf.expand_dims(behEmbed, axis=0) * tf.ones_like(uEmbed)
		ibehEmbed = tf.expand_dims(behEmbed, axis=0) * tf.ones_like(iEmbed)
		uMetaLat = FC(tf.concat([ubehEmbed, uEmbed, uNeighbor], axis=-1), latdim, useBias=True, activation=self.actFunc, reg=True, name='specMeta_FC1', reuse=True)
		iMetaLat = FC(tf.concat([ibehEmbed, iEmbed, iNeighbor], axis=-1), latdim, useBias=True, activation=self.actFunc, reg=True, name='specMeta_FC1', reuse=True)
		uW1 = tf.reshape(FC(uMetaLat, rank * latdim, useBias=True, reg=True, biasInitializer='xavier', biasReg=True, name='specMeta_FC2', reuse=True), [-1, latdim, rank])
		uW2 = tf.reshape(FC(uMetaLat, rank * latdim, useBias=True, reg=True, biasInitializer='xavier', biasReg=True, name='specMeta_FC3', reuse=True), [-1, rank, latdim])
		iW1 = tf.reshape(FC(iMetaLat, rank * latdim, useBias=True, reg=True, biasInitializer='xavier', biasReg=True, name='specMeta_FC4', reuse=True), [-1, latdim, rank])
		iW2 = tf.reshape(FC(iMetaLat, rank * latdim, useBias=True, reg=True, biasInitializer='xavier', biasReg=True, name='specMeta_FC5', reuse=True), [-1, rank, latdim])

		params = {'uW1': uW1, 'uW2': uW2, 'iW1': iW1, 'iW2': iW2}
		return params

	def specialize(self, uEmbed, iEmbed, params):
		retUEmbed = tf.reduce_sum(tf.expand_dims(uEmbed, axis=-1) * params['uW1'], axis=1)
		retUEmbed = tf.reduce_sum(tf.expand_dims(retUEmbed, axis=-1) * params['uW2'], axis=1)
		retUEmbed = tf.concat([retUEmbed, uEmbed], axis=-1)
		retIEmbed = tf.reduce_sum(tf.expand_dims(iEmbed, axis=-1) * params['iW1'], axis=1)
		retIEmbed = tf.reduce_sum(tf.expand_dims(retIEmbed, axis=-1) * params['iW2'], axis=1)
		retIEmbed = tf.concat([retIEmbed, iEmbed], axis=-1)
		return retUEmbed, retIEmbed

	def defineModel(self):
		uEmbed0 = defineParam('uEmbed0', [args.user, args.latdim//2], reg=True)
		iEmbed0 = defineParam('iEmbed0', [args.item, args.latdim//2], reg=True)
		behEmbeds = defineParam('behEmbeds', [args.behNum + 1, args.latdim//2])
		self.ulat = [0] * (args.behNum + 1)
		self.ilat = [0] * (args.behNum + 1)
		for beh in range(args.behNum):
			params = self.metaForSpecialize(uEmbed0, iEmbed0, behEmbeds[beh], [self.adjs[beh]], [self.tpAdjs[beh]])
			behUEmbed0, behIEmbed0 = self.specialize(uEmbed0, iEmbed0, params)
			# behUEmbed0 = uEmbed0
			# behIEmbed0 = iEmbed0
			ulats = [behUEmbed0]
			ilats = [behIEmbed0]
			for i in range(args.gnn_layer):
				ulat = self.messagePropagate(ilats[-1], self.adjs[beh], ulats[-1])
				ilat = self.messagePropagate(ulats[-1], self.tpAdjs[beh], ilats[-1])
				ulats.append(ulat + ulats[-1])
				ilats.append(ilat + ilats[-1])
			self.ulat[beh] = tf.add_n(ulats)
			self.ilat[beh] = tf.add_n(ilats)

		params = self.metaForSpecialize(uEmbed0, iEmbed0, behEmbeds[-1], self.adjs, self.tpAdjs)
		behUEmbed0, behIEmbed0 = self.specialize(uEmbed0, iEmbed0, params)
		ulats = [behUEmbed0]
		ilats = [behIEmbed0]
		for i in range(args.gnn_layer):
			ubehLats = []
			ibehLats = []
			for beh in range(args.behNum):
				ulat = self.messagePropagate(ilats[-1], self.adjs[beh], ulats[-1])
				ilat = self.messagePropagate(ulats[-1], self.tpAdjs[beh], ilats[-1])
				ubehLats.append(ulat)
				ibehLats.append(ilat)
			ulat = tf.add_n(lightSelfAttention(ubehLats, args.behNum, args.latdim, args.att_head))
			ilat = tf.add_n(lightSelfAttention(ibehLats, args.behNum, args.latdim, args.att_head))
			ulats.append(ulat)
			ilats.append(ilat)
		self.ulat[-1] = tf.add_n(ulats)
		self.ilat[-1] = tf.add_n(ilats)

	def metaForPredict(self, src_ulat, src_ilat, tgt_ulat, tgt_ilat):
		latdim = args.latdim
		src_ui = FC(tf.concat([src_ulat * src_ilat, src_ulat, src_ilat], axis=-1), latdim, reg=True, useBias=True, activation=self.actFunc, name='predMeta_FC1', reuse=True)
		tgt_ui = FC(tf.concat([tgt_ulat * tgt_ilat, tgt_ulat, tgt_ilat], axis=-1), latdim, reg=True, useBias=True, activation=self.actFunc, name='predMeta_FC1', reuse=True)
		metalat = FC(tf.concat([src_ui * tgt_ui, src_ui, tgt_ui], axis=-1), latdim * 3, reg=True, useBias=True, activation=self.actFunc, name='predMeta_FC2', reuse=True)
		w1 = tf.reshape(FC(metalat, latdim * 3 * latdim, reg=True, useBias=True, name='predMeta_FC3', reuse=True, biasReg=True, biasInitializer='xavier'), [-1, latdim * 3, latdim])
		b1 = tf.reshape(FC(metalat, latdim, reg=True, useBias=True, name='predMeta_FC4', reuse=True), [-1, 1, latdim])
		w2 = tf.reshape(FC(metalat, latdim, reg=True, useBias=True, name='predMeta_FC5', reuse=True, biasReg=True,biasInitializer='xavier'), [-1, latdim, 1])

		params = {
			'w1': w1,
			'b1': b1,
			'w2': w2
		}
		return params

	def _predict(self, ulat, ilat, params):
		predEmbed = tf.expand_dims(tf.concat([ulat * ilat, ulat, ilat], axis=-1), axis=1)
		predEmbed = Activate(predEmbed @ params['w1'] + params['b1'], self.actFunc)
		preds = tf.squeeze(predEmbed @ params['w2'])
		return preds

	def predict(self, src, tgt):
		uids = self.uids[tgt]
		iids = self.iids[tgt]

		src_ulat = tf.nn.embedding_lookup(self.ulat[src], uids)
		src_ilat = tf.nn.embedding_lookup(self.ilat[src], iids)
		tgt_ulat = tf.nn.embedding_lookup(self.ulat[tgt], uids)
		tgt_ilat = tf.nn.embedding_lookup(self.ilat[tgt], iids)

		predParams = self.metaForPredict(src_ulat, src_ilat, tgt_ulat, tgt_ilat)
		return self._predict(src_ulat, src_ilat, predParams) * args.mult

	def prepareModel(self):
		self.actFunc = 'leakyRelu'
		self.adjs = []
		self.tpAdjs = []
		self.uids, self.iids = [], []
		for i in range(args.behNum):
			adj = self.handler.trnMats[i]
			idx, data, shape = transToLsts(adj, norm=True)
			self.adjs.append(tf.sparse.SparseTensor(idx, data, shape))
			idx, data, shape = transToLsts(transpose(adj), norm=True)
			self.tpAdjs.append(tf.sparse.SparseTensor(idx, data, shape))
			self.uids.append(tf.placeholder(name='uids'+str(i), dtype=tf.int32, shape=[None]))
			self.iids.append(tf.placeholder(name='iids'+str(i), dtype=tf.int32, shape=[None]))
		
		self.defineModel()
		self.preLoss = 0
		for src in range(args.behNum + 1):
			for tgt in range(args.behNum):
				preds = self.predict(src, tgt)
				sampNum = tf.shape(self.uids[tgt])[0] // 2
				posPred = tf.slice(preds, [0], [sampNum])
				negPred = tf.slice(preds, [sampNum], [-1])
				self.preLoss += tf.reduce_mean(tf.maximum(0.0, 1.0 - (posPred - negPred)))
				if src == args.behNum and tgt == args.behNum - 1:
					self.targetPreds = preds
		self.regLoss = args.reg * Regularize()
		self.loss = self.preLoss + self.regLoss

		globalStep = tf.Variable(0, trainable=False)
		learningRate = tf.train.exponential_decay(args.lr, globalStep, args.decay_step, args.decay, staircase=True)
		self.optimizer = tf.train.AdamOptimizer(learningRate).minimize(self.loss, global_step=globalStep)

	def sampleTrainBatch(self, batIds, labelMat):
		temLabel = labelMat[batIds].toarray()
		batch = len(batIds)
		temlen = batch * 2 * args.sampNum
		uLocs = [None] * temlen
		iLocs = [None] * temlen
		cur = 0
		for i in range(batch):
			posset = np.reshape(np.argwhere(temLabel[i]!=0), [-1])
			sampNum = min(args.sampNum, len(posset))
			if sampNum == 0:
				poslocs = [np.random.choice(args.item)]
				neglocs = [poslocs[0]]
			else:
				poslocs = np.random.choice(posset, sampNum)
				neglocs = negSamp(temLabel[i], sampNum, args.item)
			for j in range(sampNum):
				posloc = poslocs[j]
				negloc = neglocs[j]
				uLocs[cur] = uLocs[cur+temlen//2] = batIds[i]
				iLocs[cur] = posloc
				iLocs[cur+temlen//2] = negloc
				cur += 1
		uLocs = uLocs[:cur] + uLocs[temlen//2: temlen//2 + cur]
		iLocs = iLocs[:cur] + iLocs[temlen//2: temlen//2 + cur]
		return uLocs, iLocs

	def trainEpoch(self):
		num = args.user
		sfIds = np.random.permutation(num)[:args.trnNum]
		epochLoss, epochPreLoss = [0] * 2
		num = len(sfIds)
		steps = int(np.ceil(num / args.batch))

		for i in range(steps):
			st = i * args.batch
			ed = min((i+1) * args.batch, num)
			batIds = sfIds[st: ed]

			target = [self.optimizer, self.preLoss, self.regLoss, self.loss]
			feed_dict = {}
			for beh in range(args.behNum):
				uLocs, iLocs = self.sampleTrainBatch(batIds, self.handler.trnMats[beh])
				feed_dict[self.uids[beh]] = uLocs
				feed_dict[self.iids[beh]] = iLocs

			res = self.sess.run(target, feed_dict=feed_dict, options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True))

			preLoss, regLoss, loss = res[1:]

			epochLoss += loss
			epochPreLoss += preLoss
			log('Step %d/%d: loss = %.2f, regLoss = %.2f         ' % (i, steps, loss, regLoss), save=False, oneline=True)
		ret = dict()
		ret['Loss'] = epochLoss / steps
		ret['preLoss'] = epochPreLoss / steps
		return ret

	def sampleTestBatch(self, batIds, labelMat):
		batch = len(batIds)
		temTst = self.handler.tstInt[batIds]
		temLabel = labelMat[batIds].toarray()
		temlen = batch * 100
		uLocs = [None] * temlen
		iLocs = [None] * temlen
		tstLocs = [None] * batch
		cur = 0
		for i in range(batch):
			posloc = temTst[i]
			negset = np.reshape(np.argwhere(temLabel[i]==0), [-1])
			rdnNegSet = np.random.permutation(negset)[:99]
			locset = np.concatenate((rdnNegSet, np.array([posloc])))
			tstLocs[i] = locset
			for j in range(100):
				uLocs[cur] = batIds[i]
				iLocs[cur] = locset[j]
				cur += 1
		return uLocs, iLocs, temTst, tstLocs

	def testEpoch(self):
		epochHit, epochNdcg = [0] * 2
		ids = self.handler.tstUsrs
		num = len(ids)
		tstBat = args.batch
		steps = int(np.ceil(num / tstBat))
		for i in range(steps):
			st = i * tstBat
			ed = min((i+1) * tstBat, num)
			batIds = ids[st: ed]
			feed_dict = {}
			uLocs, iLocs, temTst, tstLocs = self.sampleTestBatch(batIds, self.handler.trnMats[-1])
			feed_dict[self.uids[-1]] = uLocs
			feed_dict[self.iids[-1]] = iLocs
			preds = self.sess.run(self.targetPreds, feed_dict=feed_dict, options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True))
			hit, ndcg = self.calcRes(np.reshape(preds, [ed-st, 100]), temTst, tstLocs)
			epochHit += hit
			epochNdcg += ndcg
			log('Steps %d/%d: hit = %d, ndcg = %d          ' % (i, steps, hit, ndcg), save=False, oneline=True)
		ret = dict()
		ret['HR'] = epochHit / num
		ret['NDCG'] = epochNdcg / num
		return ret

	def calcRes(self, preds, temTst, tstLocs):
		hit = 0
		ndcg = 0
		for j in range(preds.shape[0]):
			predvals = list(zip(preds[j], tstLocs[j]))
			predvals.sort(key=lambda x: x[0], reverse=True)
			shoot = list(map(lambda x: x[1], predvals[:args.shoot]))
			if temTst[j] in shoot:
				hit += 1
				ndcg += np.reciprocal(np.log2(shoot.index(temTst[j])+2))
		return hit, ndcg
	
	def saveHistory(self):
		if args.epoch == 0:
			return
		with open('History/' + args.save_path + '.his', 'wb') as fs:
			pickle.dump(self.metrics, fs)

		saver = tf.train.Saver()
		saver.save(self.sess, 'Models/' + args.save_path)
		log('Model Saved: %s' % args.save_path)

	def loadModel(self):
		saver = tf.train.Saver()
		saver.restore(sess, 'Models/' + args.load_model)
		with open('History/' + args.load_model + '.his', 'rb') as fs:
			self.metrics = pickle.load(fs)
		log('Model Loaded')	
```

<!-- #region id="VV1cEgWJ2q9L" -->
## Training & Evaluation
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 729, "referenced_widgets": ["24c87c2912974b6d8becb5b9190aa829", "4314716903e64982b95392e7e37156bd", "6f654bcada5d41f78b08c66d2e692fde", "9dd6ba2d32ff4c0cb76b467e51e8060c", "7fd736fa68424784a086f63ba9a43657", "0bbb945cc6d54d289cbf67827ae1b723", "551b14c6f28644eea7c40845274e672c", "02be78ca98ab4b39adaa14cfd532ce00", "0343c7f1998946c68782b6fa7993b824", "439b1cc5861d446b84eba9b023d4d410", "1ceba6676a0f4736961546da9a54dac6"]} id="j9AC1tZouQWn" executionInfo={"status": "ok", "timestamp": 1639298905239, "user_tz": -330, "elapsed": 669639, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="051d89cd-13b8-45fe-ac02-ba61e5b5ccf8"
if __name__ == '__main__':
	saveDefault = True
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True

	log('Start')
	handler = DataHandler()
	handler.LoadData()
	log('Load Data')

	with tf.Session(config=config) as sess:
		recom = Recommender(sess, handler)
		recom.run()
```

<!-- #region id="cRuCzMKk5wAP" -->
## References

1. [https://github.com/RecoHut-Stanzas/S346877](https://github.com/RecoHut-Stanzas/S346877)
2. [https://arxiv.org/abs/2110.03969v1](https://arxiv.org/abs/2110.03969v1)
3. [https://github.com/akaxlh/MB-GMN](https://github.com/akaxlh/MB-GMN)
<!-- #endregion -->

<!-- #region id="Z9_DARRQ1eUt" -->
---
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="xNBE4KFf1jN9" executionInfo={"status": "ok", "timestamp": 1639299049247, "user_tz": -330, "elapsed": 6161, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="38813f0a-27d4-4fc0-fd65-822e0dc1678e"
!apt-get -qq install tree
!rm -r sample_data
```

```python colab={"base_uri": "https://localhost:8080/"} id="_4QWNrMv1eUx" executionInfo={"status": "ok", "timestamp": 1639299049978, "user_tz": -330, "elapsed": 745, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="a4b90a2c-705d-43ff-abe2-10ff4c94ca1a"
!tree -h --du .
```

```python colab={"base_uri": "https://localhost:8080/"} id="GKmE1saI1eUx" executionInfo={"status": "ok", "timestamp": 1639299060337, "user_tz": -330, "elapsed": 3910, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="c27afee4-57c1-40ef-f54c-d86fa7d30b6a"
!pip install -q watermark
%reload_ext watermark
%watermark -a "Sparsh A." -m -iv -u -t -d
```

<!-- #region id="qvid872_1eUy" -->
---
<!-- #endregion -->

<!-- #region id="O0inFkFz1eUz" -->
**END**
<!-- #endregion -->
